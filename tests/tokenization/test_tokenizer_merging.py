# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for vocabulary merging and embedding remapping (`transformers.tokenizer_merging`)."""

import json
import os
import tempfile
import unittest

from transformers import (
    AddedToken,
    AutoTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers.testing_utils import require_torch
from transformers.tokenizer_merging import (
    VOCAB_MERGING_FILE_NAME,
    EmbeddingShapeError,
    SpecialTokenConflictError,
    TokenizerMergePlan,
    VocabMergingError,
    build_merged_tokenizer,
    load_vocab_merging_mapping,
    merge_model_embeddings,
    merge_tokenizers,
    plan_vocab_merging,
    remap_embeddings,
    remap_weight_matrix,
    save_vocab_merging_mapping,
)


# Two WordPiece vocabularies with a large overlap but completely different id assignments, each side owning unique
# tokens and both sharing a custom special token. Sizes stay in the few-dozen range so every test runs in fast CI.
WORLDPIECE_FIRST = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "a": 5,
    "b": 6,
    "ab": 7,
    "cat": 8,
    "dog": 9,
    "##s": 10,
    "##er": 11,
    "run": 12,
    "walk": 13,
    "entity": 14,
}
WORLDPIECE_SECOND = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "##s": 5,
    "##er": 6,
    "a": 7,
    "b": 8,
    "ab": 9,
    "cat": 10,
    "run": 11,
    "ner": 12,
    "tag": 13,
    "span": 14,
    "token": 15,
    "label": 16,
}

BPE_FIRST = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4}
BPE_FIRST_MERGES = [("a", "b"), ("ab", "c")]
BPE_SECOND = {
    "z": 0,
    "a": 1,
    "b": 2,
    "ab": 3,
    "c": 4,
    "bc": 5,
    "abc": 6,
    "new": 7,
}
BPE_SECOND_MERGES = [("a", "b"), ("b", "c"), ("ab", "c")]


def make_wordpiece_pair(**kwargs):
    first = BertTokenizer(
        vocab=dict(WORLDPIECE_FIRST), do_lower_case=False, additional_special_tokens=["[ENT]"], **kwargs
    )
    second = BertTokenizer(
        vocab=dict(WORLDPIECE_SECOND), do_lower_case=False, additional_special_tokens=["[ENT]"], **kwargs
    )
    return first, second


def make_bpe_pair(**kwargs):
    first = GPT2Tokenizer(vocab=dict(BPE_FIRST), merges=list(BPE_FIRST_MERGES), **kwargs)
    second = GPT2Tokenizer(vocab=dict(BPE_SECOND), merges=list(BPE_SECOND_MERGES), **kwargs)
    return first, second


class TokenizerMergingPlanTest(unittest.TestCase):
    def test_union_vocab_is_bijection_without_holes(self):
        first, second = make_wordpiece_pair()
        plan = plan_vocab_merging(first, second)

        # One id per token, contiguous 0..N-1, no duplicate token strings.
        self.assertEqual(sorted(record.new_id for record in plan.records), list(range(plan.vocab_size)))
        self.assertEqual(len({record.token for record in plan.records}), plan.vocab_size)
        self.assertEqual(plan.merged_vocab, {record.token: record.new_id for record in plan.records})

        # Every id of both source tokenizers is mapped, and the two maps jointly cover the whole merged space.
        self.assertEqual(set(plan.first_id_to_new), set(first.get_vocab().values()))
        self.assertEqual(set(plan.second_id_to_new), set(second.get_vocab().values()))
        self.assertEqual(
            set(plan.first_id_to_new.values()) | set(plan.second_id_to_new.values()),
            set(range(plan.vocab_size)),
        )
        # The two old maps must be injective: no two old ids may share a merged id.
        self.assertEqual(len(set(plan.first_id_to_new.values())), len(plan.first_id_to_new))
        self.assertEqual(len(set(plan.second_id_to_new.values())), len(plan.second_id_to_new))

    def test_shared_tokens_keep_base_ids_and_unique_tokens_append(self):
        first, second = make_wordpiece_pair()
        plan = plan_vocab_merging(first, second, strategy="first")
        self.assertEqual(plan.base_index, 0)

        for token, old_id in first.get_vocab().items():
            self.assertEqual(plan.first_id_to_new[old_id], plan.merged_vocab[token])
        # With the first tokenizer as base, its ids are preserved verbatim.
        for token, old_id in first.get_vocab().items():
            self.assertEqual(plan.merged_vocab[token], old_id)

        # Second-only tokens are appended after the base vocabulary, in the second tokenizer's id order.
        second_only = sorted(
            (token for token in second.get_vocab() if token not in first.get_vocab()),
            key=lambda token: second.get_vocab()[token],
        )
        base_size = len(first.get_vocab())
        self.assertEqual([plan.merged_vocab[t] - base_size for t in second_only], list(range(len(second_only))))

    def test_strategies_select_the_expected_base(self):
        first, second = make_wordpiece_pair()
        self.assertEqual(plan_vocab_merging(first, second, strategy="first").base_index, 0)
        self.assertEqual(plan_vocab_merging(first, second, strategy="second").base_index, 1)
        # The second vocabulary is larger; ties deterministically keep the first tokenizer.
        self.assertEqual(plan_vocab_merging(first, second, strategy="larger").base_index, 1)
        self.assertEqual(plan_vocab_merging(first, second, strategy="smaller").base_index, 0)
        with self.assertRaises(ValueError):
            plan_vocab_merging(first, second, strategy="nonsense")

    def test_planning_is_deterministic(self):
        first, second = make_wordpiece_pair()
        plan_a = plan_vocab_merging(first, second, strategy="larger")
        plan_b = plan_vocab_merging(first, second, strategy="larger")
        self.assertEqual(plan_a.to_dict(), plan_b.to_dict())

        bpe_first, bpe_second = make_bpe_pair()
        bpe_a = plan_vocab_merging(bpe_first, bpe_second)
        bpe_b = plan_vocab_merging(bpe_first, bpe_second)
        self.assertEqual(bpe_a.to_dict(), bpe_b.to_dict())

    def test_special_token_conflict_is_rejected_with_the_conflicting_token(self):
        first = BertTokenizer(
            vocab={"[PAD]": 0, "[UNK]": 1, "a": 2},
            do_lower_case=False,
            additional_special_tokens=[AddedToken("[X]", special=True, normalized=False)],
        )
        second = BertTokenizer(vocab={"[PAD]": 0, "[UNK]": 1, "a": 2}, do_lower_case=False)
        second.add_tokens([AddedToken("[X]", special=False, normalized=True)])

        with self.assertRaises(SpecialTokenConflictError) as context:
            plan_vocab_merging(first, second)
        self.assertIn("[X]", str(context.exception))

    def test_different_backend_types_are_rejected(self):
        wordpiece, _ = make_wordpiece_pair()
        bpe, _ = make_bpe_pair()
        with self.assertRaises(VocabMergingError) as context:
            plan_vocab_merging(wordpiece, bpe)
        self.assertIn("WordPiece", str(context.exception))
        self.assertIn("BPE", str(context.exception))

    def test_gappy_source_vocabulary_is_rejected(self):
        # Id 3 is missing: the Rust backend returns None for it, which would break the contiguous-id guarantee.
        broken = BertTokenizer(vocab={"[PAD]": 0, "[UNK]": 1, "a": 2, "c": 4}, do_lower_case=False)
        healthy, _ = make_wordpiece_pair()
        with self.assertRaises(VocabMergingError) as context:
            plan_vocab_merging(broken, healthy)
        self.assertIn("hole", str(context.exception))

    def test_bpe_broken_merge_chain_produces_unreachable_list(self):
        # t1 consumes "pq" (plain token for it) to build "pqc"; t2 turns "pq" into a merge output. In the union the
        # merge producing "pq" is ordered last, so t1's ("pq", "c") merge is dropped and "pqc" becomes unreachable.
        first = GPT2Tokenizer(
            vocab={"a": 0, "b": 1, "c": 2, "ab": 3, "pq": 4, "pqc": 5},
            merges=[("a", "b"), ("pq", "c")],
        )
        second = GPT2Tokenizer(vocab={"p": 0, "q": 1, "pq": 2}, merges=[("p", "q")])
        plan = plan_vocab_merging(first, second)

        self.assertEqual(plan.unreachable_tokens, ["pqc"])
        self.assertNotIn(("pq", "c"), plan.merges)
        self.assertIn(("a", "b"), plan.merges)
        self.assertIn(("p", "q"), plan.merges)
        # Unreachable tokens still keep an id, so embedding rows stay addressable.
        self.assertIn("pqc", plan.merged_vocab)

    def test_wordpiece_continuation_only_tokens_are_unreachable(self):
        first = BertTokenizer(vocab={"[PAD]": 0, "[UNK]": 1, "##x": 2, "##y": 3}, do_lower_case=False)
        second = BertTokenizer(vocab={"[PAD]": 0, "[UNK]": 1, "##x": 2, "##z": 3}, do_lower_case=False)
        plan = plan_vocab_merging(first, second)
        self.assertEqual(sorted(plan.unreachable_tokens), ["##x", "##y", "##z"])
        for token in plan.unreachable_tokens:
            self.assertFalse(next(r for r in plan.records if r.token == token).reachable)

    def test_mapping_roundtrip_and_repeated_saves_are_identical(self):
        first, second = make_wordpiece_pair()
        plan = plan_vocab_merging(first, second)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_vocab_merging_mapping(plan, tmpdir)
            self.assertEqual(os.path.basename(path), VOCAB_MERGING_FILE_NAME)
            reloaded = load_vocab_merging_mapping(tmpdir)
            self.assertEqual(reloaded.first_id_to_new, plan.first_id_to_new)
            self.assertEqual(reloaded.second_id_to_new, plan.second_id_to_new)
            self.assertEqual(reloaded.merged_vocab, plan.merged_vocab)
            self.assertEqual(reloaded.merges, plan.merges)

            # Saving the same plan twice must produce byte-identical content (sorted keys).
            with open(path, encoding="utf-8") as handle:
                first_dump = handle.read()
            save_vocab_merging_mapping(plan, tmpdir)
            with open(path, encoding="utf-8") as handle:
                second_dump = handle.read()
            self.assertEqual(first_dump, second_dump)
            # The file is plain JSON and carries both old-id maps.
            payload = json.loads(first_dump)
            self.assertEqual(payload["merged_vocab_size"], plan.vocab_size)


class TokenizerMergingTokenizerTest(unittest.TestCase):
    def test_constructed_vocabulary_matches_plan(self):
        first, second = make_wordpiece_pair()
        merged, plan = build_merged_tokenizer(first, second)
        self.assertEqual(merged.get_vocab(), plan.merged_vocab)
        self.assertEqual(merged.backend_tokenizer.get_vocab_size(), plan.vocab_size)
        # Convenience wrapper returns the same merged tokenizer.
        self.assertEqual(merge_tokenizers(first, second).get_vocab(), merged.get_vocab())

    def test_shared_text_keeps_segmentation_and_decode_roundtrip(self):
        first, second = make_wordpiece_pair()
        merged, _ = build_merged_tokenizer(first, second, strategy="first")
        for text in ["ab", "a", "b", "cats", "dog", "runner", "cat dog ab"]:
            self.assertEqual(merged.tokenize(text), first.tokenize(text), msg=text)
            ids = merged.encode(text)
            self.assertEqual(merged.decode(ids), first.decode(first.encode(text)), msg=text)

    def test_bpe_merges_are_unioned_and_segmentation_matches(self):
        first, second = make_bpe_pair()
        merged, plan = build_merged_tokenizer(first, second)
        self.assertEqual(merged.backend_tokenizer.model.token_to_id("new"), plan.merged_vocab["new"])
        for text in ["ab", "abc", "a b", "new"]:
            self.assertEqual(merged.tokenize(text), second.tokenize(text), msg=text)
        # The shared merge survives and the second-side-only merges are present.
        self.assertIn(("b", "c"), plan.merges)

    def test_save_load_twice_keeps_ids_and_encodings(self):
        first, second = make_wordpiece_pair()
        merged, plan = build_merged_tokenizer(first, second)
        probe = merged.encode("cat dog entity ner label")
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            merged.save_pretrained(first_dir)
            save_vocab_merging_mapping(plan, first_dir)
            loaded_once = AutoTokenizer.from_pretrained(first_dir)
            loaded_once.save_pretrained(second_dir)
            loaded_twice = AutoTokenizer.from_pretrained(second_dir)

            self.assertEqual(loaded_twice.get_vocab(), plan.merged_vocab)
            self.assertEqual(loaded_twice.encode("cat dog entity ner label"), probe)
            self.assertEqual(loaded_twice.tokenize("cats"), merged.tokenize("cats"))
            # The custom special token survives the two save/load cycles at its merged id.
            self.assertEqual(
                loaded_twice.convert_tokens_to_ids("[ENT]"),
                plan.merged_vocab["[ENT]"],
            )

    def test_backend_is_preserved(self):
        rust_first, rust_second = make_wordpiece_pair()
        merged_rust, _ = build_merged_tokenizer(rust_first, rust_second)
        self.assertEqual(merged_rust.backend, "tokenizers")

        python_first, python_second = make_wordpiece_pair(backend="python")
        merged_python, plan = build_merged_tokenizer(python_first, python_second)
        self.assertEqual(merged_python.backend, "python")
        # Ids change by design; the segmentation (token strings) must not.
        self.assertEqual(merged_python.tokenize("ab"), python_first.tokenize("ab"))
        # And every emitted id follows the persisted old -> new map.
        for old_id in python_first.encode("ab"):
            self.assertIn(plan.first_id_to_new[old_id], merged_python.encode("ab"))

    def test_bpe_merged_tokenizer_breaks_no_existing_segmentation(self):
        first, second = make_bpe_pair()
        merged, _ = build_merged_tokenizer(first, second)
        with tempfile.TemporaryDirectory() as tmpdir:
            merged.save_pretrained(tmpdir)
            reloaded = AutoTokenizer.from_pretrained(tmpdir)
            for text in ["ab", "abc", "a b c"]:
                self.assertEqual(reloaded.tokenize(text), merged.tokenize(text), msg=text)


@require_torch
class TokenizerMergingWeightTest(unittest.TestCase):
    def _tiny_gpt2(self, vocab_size):
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=vocab_size - 1,
            eos_token_id=vocab_size - 1,
        )
        return GPT2LMHeadModel(config)

    def test_shared_rows_are_bit_identical_and_new_rows_are_controlled(self):
        import torch

        first, second = make_bpe_pair()
        plan = plan_vocab_merging(first, second)
        model = self._tiny_gpt2(len(first.get_vocab()))
        original = model.get_input_embeddings().weight.detach().clone()

        remap_embeddings(model, plan, source="first", new_row_init="zeros")
        weight = model.get_input_embeddings().weight
        self.assertEqual(tuple(weight.shape), (plan.vocab_size, 16))

        for old_id, new_id in plan.first_id_to_new.items():
            self.assertTrue(torch.equal(weight[new_id], original[old_id]))
        # Rows this side does not own use the requested initializer.
        second_only = [record.new_id for record in plan.records if record.source == "second"]
        self.assertTrue(all(torch.count_nonzero(weight[row]) == 0 for row in second_only))
        # The config follows the new vocabulary size and the model can keep training on it.
        self.assertEqual(model.config.vocab_size, plan.vocab_size)
        self.assertTrue(weight.requires_grad)

    def test_source_matrix_is_not_mutated_in_place(self):
        import torch

        plan = plan_vocab_merging(*make_bpe_pair())
        source = torch.randn(len(plan.first_id_to_new), 12)
        snapshot = source.clone()
        remap_weight_matrix(source, plan.first_id_to_new, plan.vocab_size, new_row_init="zeros")
        self.assertTrue(torch.equal(source, snapshot))

    def test_mean_and_callable_initializers(self):
        import torch

        plan = plan_vocab_merging(*make_bpe_pair())
        source = torch.randn(len(plan.first_id_to_new), 12)
        mean_remapped = remap_weight_matrix(source, plan.first_id_to_new, plan.vocab_size, new_row_init="mean")
        free_rows = sorted(set(range(plan.vocab_size)) - set(plan.first_id_to_new.values()))
        for row in free_rows:
            self.assertTrue(torch.allclose(mean_remapped[row], source.mean(dim=0)))

        custom = remap_weight_matrix(
            source,
            plan.first_id_to_new,
            plan.vocab_size,
            new_row_init=lambda matrix: torch.full((len(free_rows), matrix.shape[1]), 7.0),
        )
        self.assertTrue(torch.all(custom[free_rows] == 7.0))

    def test_row_count_mismatch_raises_before_copying(self):
        import torch

        plan = plan_vocab_merging(*make_bpe_pair())
        bad = torch.nn.Embedding(3, 16)
        with self.assertRaises(EmbeddingShapeError):
            remap_weight_matrix(bad.weight, plan.first_id_to_new, plan.vocab_size)

        model = self._tiny_gpt2(3)
        with self.assertRaises(EmbeddingShapeError):
            remap_embeddings(model, plan, source="first")

    def test_tied_embeddings_stay_tied_and_model_runs_inference(self):
        import torch

        first, second = make_bpe_pair()
        merged, plan = build_merged_tokenizer(first, second)
        model = self._tiny_gpt2(len(first.get_vocab()))
        self.assertIs(model.get_output_embeddings().weight, model.get_input_embeddings().weight)

        remap_embeddings(model, plan, source="first", new_row_init="mean")
        self.assertIs(model.get_output_embeddings().weight, model.get_input_embeddings().weight)
        self.assertEqual(model.config.vocab_size, plan.vocab_size)

        inputs = merged("abc new", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        self.assertEqual(outputs.logits.shape[-1], plan.vocab_size)

    def test_untied_input_and_output_matrices_are_both_remapped(self):
        import torch

        first, second = make_wordpiece_pair()
        plan = plan_vocab_merging(first, second)
        config = BertConfig(
            vocab_size=len(first.get_vocab()),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            tie_word_embeddings=False,
        )
        model = BertForMaskedLM(config)
        input_weight = model.get_input_embeddings().weight.detach().clone()
        output_weight = model.get_output_embeddings().weight.detach().clone()
        self.assertIsNot(model.get_output_embeddings().weight, model.get_input_embeddings().weight)

        remap_embeddings(model, plan, source="first", new_row_init="zeros")
        new_input = model.get_input_embeddings().weight
        new_output = model.get_output_embeddings().weight
        for old_id, new_id in plan.first_id_to_new.items():
            self.assertTrue(torch.equal(new_input[new_id], input_weight[old_id]))
            self.assertTrue(torch.equal(new_output[new_id], output_weight[old_id]))

    def test_second_side_checkpoint_uses_its_own_map(self):
        import torch

        first, second = make_bpe_pair()
        plan = plan_vocab_merging(first, second)
        model = self._tiny_gpt2(len(second.get_vocab()))
        original = model.get_input_embeddings().weight.detach().clone()

        remap_embeddings(model, plan, source="second", new_row_init="zeros")
        weight = model.get_input_embeddings().weight
        for old_id, new_id in plan.second_id_to_new.items():
            self.assertTrue(torch.equal(weight[new_id], original[old_id]))

    def test_end_to_end_helper_remaps_both_models_and_returns_tokenizer(self):
        import torch

        first, second = make_bpe_pair()
        first_model = self._tiny_gpt2(len(first.get_vocab()))
        second_model = self._tiny_gpt2(len(second.get_vocab()))
        first_rows = first_model.get_input_embeddings().weight.detach().clone()
        second_rows = second_model.get_input_embeddings().weight.detach().clone()

        remapped_first, remapped_second, merged_tokenizer, plan = merge_model_embeddings(
            first_model, second_model, first, second, new_row_init="zeros"
        )
        self.assertEqual(remapped_first.config.vocab_size, plan.vocab_size)
        self.assertEqual(remapped_second.config.vocab_size, plan.vocab_size)
        self.assertEqual(merged_tokenizer.get_vocab(), plan.merged_vocab)

        first_weight = remapped_first.get_input_embeddings().weight
        second_weight = remapped_second.get_input_embeddings().weight
        for old_id, new_id in plan.first_id_to_new.items():
            self.assertTrue(torch.equal(first_weight[new_id], first_rows[old_id]))
        for old_id, new_id in plan.second_id_to_new.items():
            self.assertTrue(torch.equal(second_weight[new_id], second_rows[old_id]))

    def test_plan_serialization_is_loadable_for_old_checkpoints(self):
        first, second = make_bpe_pair()
        plan = plan_vocab_merging(first, second)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_vocab_merging_mapping(plan, tmpdir)
            reloaded = load_vocab_merging_mapping(tmpdir)
            self.assertIsInstance(reloaded, TokenizerMergePlan)
            # An old checkpoint row can always be interpreted through the persisted map.
            for old_id in range(len(first.get_vocab())):
                self.assertIn(old_id, reloaded.first_id_to_new)


if __name__ == "__main__":
    unittest.main()
