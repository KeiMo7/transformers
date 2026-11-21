# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""
Tests for tokenizer encoding-config export, comparison and verification.

Everything here runs on the tiny fixtures in `tests/fixtures` (or on a small vocabulary written to a temporary
directory), so no network or real checkpoint is required.
"""

import json
import os
import tempfile
import unittest

from transformers import (
    BertTokenizer,
    EncodingConfig,
    EncodingConfigMismatchError,
    EncodingConfigMissingError,
    fingerprint_vocabulary,
)
from transformers.testing_utils import require_sentencepiece, require_tokenizers


TINY_VOCAB = (
    ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    + [f"tok{i}" for i in range(60)]
    + [
        "hello",
        "world",
        "the",
        "quick",
        "brown",
    ]
)


def write_vocab(directory, vocab=None, filename="vocab.txt"):
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(vocab or TINY_VOCAB) + "\n")
    return path


class EncodingConfigMixin:
    """Shared helpers. Subclasses provide `make_tokenizer(directory)`."""

    tokenizer_kwargs = {}

    def make_tokenizer(self, directory):
        write_vocab(directory)
        return BertTokenizer(os.path.join(directory, "vocab.txt"), **self.tokenizer_kwargs)

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tokenizer = self.make_tokenizer(self.tmpdir)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def samples(self):
        return ["Hello world", "the quick brown fox", "tok3 tok4", ""]

    def assert_same_encoding(self, left, right):
        for text in self.samples():
            self.assertEqual(left.encode(text), right.encode(text), f"encoding differs for {text!r}")
            self.assertEqual(left.tokenize(text), right.tokenize(text), f"tokens differ for {text!r}")
            self.assertEqual(dict(left(text)), dict(right(text)), f"call output differs for {text!r}")


@require_tokenizers
class EncodingConfigTest(EncodingConfigMixin, unittest.TestCase):
    def test_export_contains_every_encoding_component(self):
        config = self.tokenizer.export_encoding_config().to_dict()
        self.assertEqual(config["format"], "transformers-tokenizer-encoding-config")
        self.assertEqual(config["format_version"], 1)

        encoding = config["encoding"]
        # Vocabulary fingerprint, special tokens with explicit ids, truncation/padding and normalization settings.
        self.assertIn("fingerprint", config["vocabulary"])
        for name in ("unk_token", "sep_token", "pad_token", "cls_token", "mask_token"):
            self.assertIn(name, encoding["special_tokens"])
            self.assertIsInstance(encoding["special_tokens"][name]["id"], int)
        self.assertIn("truncation", encoding)
        self.assertIn("padding", encoding)
        self.assertIn("normalizer", encoding["pipeline"])
        self.assertIn("pre_tokenizer", encoding["pipeline"])
        self.assertIn("post_processor", encoding["pipeline"])

        # The fingerprint is content-based: it does not mention the model name or a version string.
        fingerprint = config["fingerprints"]["config"]
        self.assertEqual(fingerprint["algorithm"], "sha256-config-v1")
        self.assertNotIn(self.tokenizer.name_or_path or "unused", fingerprint["digest"])
        self.assertEqual(len(fingerprint["digest"]), 64)

    def test_fingerprint_is_content_based_not_name_based(self):
        other_dir = tempfile.mkdtemp()
        try:
            write_vocab(other_dir)
            other = BertTokenizer(os.path.join(other_dir, "vocab.txt"), name_or_path="totally-different-name")
            self.assertEqual(
                self.tokenizer.export_encoding_config().fingerprint, other.export_encoding_config().fingerprint
            )
        finally:
            import shutil

            shutil.rmtree(other_dir, ignore_errors=True)

    def test_identical_configs_compare_equal(self):
        left = self.tokenizer.export_encoding_config()
        right = self.tokenizer.export_encoding_config()
        self.assertEqual(left.fingerprint, right.fingerprint)
        self.assertEqual(left.diff(right), [])
        left.assert_matches(right)

    def test_config_difference_is_localised(self):
        other_dir = tempfile.mkdtemp()
        try:
            write_vocab(other_dir)
            other = BertTokenizer(os.path.join(other_dir, "vocab.txt"), do_lower_case=False)
        finally:
            import shutil

            shutil.rmtree(other_dir, ignore_errors=True)

        left = self.tokenizer.export_encoding_config()
        right = other.export_encoding_config()
        differences = left.diff(right)
        self.assertTrue(differences, "expected at least one difference")
        paths = [difference.path for difference in differences]
        # The concrete setting is named, not just "the configs differ".
        self.assertIn("encoding.pipeline.normalizer.lowercase", paths)
        lowercase = next(d for d in differences if d.path == "encoding.pipeline.normalizer.lowercase")
        self.assertEqual((lowercase.left, lowercase.right), (True, False))
        self.assertIn("lowercase", lowercase.describe())

        with self.assertRaises(EncodingConfigMismatchError) as context:
            left.assert_matches(right)
        self.assertEqual(len(context.exception.differences), len(differences))
        self.assertIn("encoding.pipeline.normalizer.lowercase", str(context.exception))

    def test_replaced_vocabulary_same_filename_is_detected(self):
        other_dir = tempfile.mkdtemp()
        try:
            # Same file name (`vocab.txt`), same length, one swapped entry: the file is replaced, the name is not.
            replaced = list(TINY_VOCAB)
            replaced[replaced.index("hello")], replaced[replaced.index("world")] = "HELLO", "world"
            write_vocab(other_dir, vocab=replaced)
            other = BertTokenizer(os.path.join(other_dir, "vocab.txt"))
        finally:
            import shutil

            shutil.rmtree(other_dir, ignore_errors=True)

        left = self.tokenizer.export_encoding_config()
        right = other.export_encoding_config()
        self.assertNotEqual(left.vocabulary_fingerprint["digest"], right.vocabulary_fingerprint["digest"])

        differences = left.diff(right)
        vocabulary = [d for d in differences if d.path == "fingerprints.vocabulary.digest"]
        self.assertEqual(len(vocabulary), 1)
        # The difference is localised to the token-id range that changed.
        self.assertIn("changed token-id ranges", vocabulary[0].note)
        self.assertIn("ids 0..", vocabulary[0].note)

        with self.assertRaises(EncodingConfigMismatchError) as context:
            left.assert_matches(right)
        self.assertTrue(any(d.path == "fingerprints.vocabulary.digest" for d in context.exception.differences))

    def test_save_and_load_with_the_model(self):
        self.tokenizer.save_pretrained(self.tmpdir, save_encoding_config=True)
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, "encoding_config.json")))

        saved = EncodingConfig.load(self.tmpdir)
        self.assertEqual(saved.fingerprint, self.tokenizer.export_encoding_config().fingerprint)

        reloaded = BertTokenizer.from_pretrained(self.tmpdir)
        self.assertIsNotNone(reloaded.encoding_config)
        self.assertEqual(reloaded.encoding_config.fingerprint, saved.fingerprint)
        self.assert_same_encoding(self.tokenizer, reloaded)

    def test_verification_does_not_change_existing_outputs(self):
        self.tokenizer.save_pretrained(self.tmpdir, save_encoding_config=True)
        plain = BertTokenizer.from_pretrained(self.tmpdir)
        verified = BertTokenizer.from_pretrained(self.tmpdir, verify_encoding_config=True)
        self.assert_same_encoding(self.tokenizer, plain)
        self.assert_same_encoding(self.tokenizer, verified)

    def test_missing_config_raises_in_strict_mode(self):
        directory = tempfile.mkdtemp()
        try:
            self.tokenizer.save_pretrained(directory)  # no encoding config
            self.assertFalse(os.path.exists(os.path.join(directory, "encoding_config.json")))
            with self.assertRaises(EncodingConfigMissingError):
                BertTokenizer.from_pretrained(directory, verify_encoding_config=True)
            # Without verification the tokenizer loads exactly as before.
            tokenizer = BertTokenizer.from_pretrained(directory)
            self.assert_same_encoding(self.tokenizer, tokenizer)
        finally:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)

    def test_mismatched_config_raises_with_differences(self):
        directory = tempfile.mkdtemp()
        try:
            self.tokenizer.save_pretrained(directory, save_encoding_config=True)
            # Replace the saved vocabulary while keeping the file name.
            replaced = list(TINY_VOCAB)
            replaced[replaced.index("quick")] = "slow"
            write_vocab(directory, vocab=replaced)
            # Remove the cached tokenizer.json so the replaced vocab.txt is actually used.
            os.remove(os.path.join(directory, "tokenizer.json"))

            with self.assertRaises(EncodingConfigMismatchError) as context:
                BertTokenizer.from_pretrained(directory, verify_encoding_config=True)
            self.assertTrue(any("vocabulary" in d.path for d in context.exception.differences))
        finally:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)

    def test_verification_failure_does_not_block_when_not_strict(self):
        # An expected config that cannot be compared must warn, not prevent the tokenizer from loading.
        directory = tempfile.mkdtemp()
        try:
            self.tokenizer.save_pretrained(directory, save_encoding_config=True)
            # A structurally valid config for a different encoding (case folding off) does not match.
            mismatched = BertTokenizer(os.path.join(self.tmpdir, "vocab.txt"), do_lower_case=False)
            with self.assertLogs("transformers.tokenization_utils_base", level="WARNING"):
                tokenizer = BertTokenizer.from_pretrained(
                    directory, encoding_config=mismatched, verify_encoding_config=False
                )
            self.assert_same_encoding(self.tokenizer, tokenizer)

            # A malformed config file is ignored with a warning and does not break loading.
            with open(os.path.join(directory, "encoding_config.json"), "w", encoding="utf-8") as handle:
                json.dump({"format": "transformers-tokenizer-encoding-config", "format_version": 1}, handle)
            with self.assertLogs("transformers.tokenization_utils_base", level="WARNING"):
                tokenizer = BertTokenizer.from_pretrained(directory)
            self.assert_same_encoding(self.tokenizer, tokenizer)
        finally:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)

    def test_saved_config_is_independent_of_external_files(self):
        directory = tempfile.mkdtemp()
        try:
            config = self.tokenizer.export_encoding_config(include_vocabulary=True)
            path = config.save(os.path.join(directory, "encoding_config.json"))
            self.assertNotIn("vocab_file", open(path, encoding="utf-8").read())

            # Delete every vocabulary file, then rebuild from the config alone.
            for name in os.listdir(self.tmpdir):
                os.remove(os.path.join(self.tmpdir, name))
            rebuilt = EncodingConfig.load(path).build_tokenizer()
        finally:
            import shutil

            shutil.rmtree(directory, ignore_errors=True)

        self.assert_same_encoding(self.tokenizer, rebuilt)
        for text in self.samples():
            self.assertEqual(self.tokenizer.encode(text), rebuilt.encode(text))
        self.assertEqual(
            rebuilt.export_encoding_config().fingerprint, self.tokenizer.export_encoding_config().fingerprint
        )

    def test_rebuilt_tokenizer_matches_token_for_token(self):
        config = self.tokenizer.export_encoding_config(include_vocabulary=True)
        rebuilt = config.build_tokenizer()
        for text in self.samples() + ["Hello World!", "  leading and trailing  "]:
            self.assertEqual(self.tokenizer.encode(text), rebuilt.encode(text), f"ids differ for {text!r}")
            self.assertEqual(self.tokenizer.tokenize(text), rebuilt.tokenize(text), f"tokens differ for {text!r}")

    def test_compact_config_cannot_rebuild_but_says_so(self):
        config = self.tokenizer.export_encoding_config()
        with self.assertRaises(ValueError) as context:
            config.build_tokenizer()
        self.assertIn("include_vocabulary=True", str(context.exception))

    def test_config_roundtrips_through_json(self):
        config = self.tokenizer.export_encoding_config()
        restored = EncodingConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        self.assertEqual(restored.fingerprint, config.fingerprint)
        self.assertEqual(restored.diff(config), [])

    def test_unknown_format_is_rejected(self):
        with self.assertRaises(ValueError):
            EncodingConfig.from_dict({"format": "something-else", "format_version": 1})

    def test_tampered_config_is_rejected(self):
        data = self.tokenizer.export_encoding_config().to_dict()
        data["encoding"]["pipeline"]["normalizer"]["lowercase"] = False
        with self.assertRaises(ValueError) as context:
            EncodingConfig.from_dict(data)
        self.assertIn("does not match its contents", str(context.exception))

    def test_export_does_not_mutate_the_tokenizer(self):
        before = {text: self.tokenizer.encode(text) for text in self.samples()}
        self.tokenizer.export_encoding_config(include_vocabulary=True)
        after = {text: self.tokenizer.encode(text) for text in self.samples()}
        self.assertEqual(before, after)

    def test_unigram_scores_roundtrip(self):
        # Unigram stores per-token scores; `tokenizers` serializes them as doubles that lose one ULP on reload.
        # The fingerprint must be a fixed point of serialize -> rebuild -> serialize.
        unigram = [("▁a", -1.234567890123), ("▁b", -2.345678901234), ("ab", -3.456789012345)]

        def make():
            from tokenizers import Tokenizer
            from tokenizers.models import Unigram

            return Tokenizer(Unigram(unigram))

        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        tokenizer = TokenizersBackend(tokenizer_object=make())
        config = tokenizer.export_encoding_config(include_vocabulary=True)
        rebuilt = config.build_tokenizer()
        self.assertEqual(rebuilt.export_encoding_config().fingerprint, config.fingerprint)


@require_tokenizers
class VocabularyFingerprintTest(unittest.TestCase):
    def test_digest_is_order_independent(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        shuffled = {"c": 2, "a": 0, "b": 1}
        self.assertEqual(fingerprint_vocabulary(vocab)["digest"], fingerprint_vocabulary(shuffled)["digest"])

    def test_digest_changes_when_an_id_changes(self):
        self.assertNotEqual(
            fingerprint_vocabulary({"a": 0, "b": 1})["digest"],
            fingerprint_vocabulary({"a": 1, "b": 0})["digest"],
        )

    def test_chunks_localise_the_change(self):
        vocab = {f"t{i}": i for i in range(12)}
        changed = dict(vocab, t7=7000)
        left = fingerprint_vocabulary(vocab, chunk_size=4)
        right = fingerprint_vocabulary(changed, chunk_size=4)
        differing = [c["start"] for c, d in zip(left["chunks"], right["chunks"]) if c["digest"] != d["digest"]]
        self.assertEqual(differing, [4])


@require_tokenizers
class EncodingConfigThreadSafetyTest(EncodingConfigMixin, unittest.TestCase):
    """
    Exporting the same tokenizer from several threads must produce one deterministic fingerprint.

    `export_encoding_config` only reads the tokenizer; the vocabulary is walked with plain local buffers, so there is
    no shared mutable state and no lock. Concurrency can only change how long it takes, never the result.
    """

    def test_concurrent_exports_are_deterministic(self):
        from concurrent.futures import ThreadPoolExecutor

        expected = self.tokenizer.export_encoding_config().fingerprint
        with ThreadPoolExecutor(max_workers=8) as pool:
            fingerprints = list(pool.map(lambda _: self.tokenizer.export_encoding_config().fingerprint, range(32)))
        self.assertEqual(set(fingerprints), {expected})


@require_sentencepiece
class SentencePieceEncodingConfigTest(unittest.TestCase):
    """The same API on a SentencePiece backend, which has no `tokenizers` pipeline serialization."""

    @classmethod
    def setUpClass(cls):
        from transformers import BertGenerationTokenizer
        from transformers.testing_utils import get_tests_dir

        cls.tokenizer = BertGenerationTokenizer(get_tests_dir("fixtures/test_sentencepiece.model"))

    def test_export_and_rebuild(self):
        config = self.tokenizer.export_encoding_config(include_vocabulary=True)
        payload_format = config.to_dict()["build"]["payload"]["format"]
        self.assertEqual(payload_format, "sentencepiece-proto")

        rebuilt = config.build_tokenizer()
        self.assertEqual(type(rebuilt).__name__, type(self.tokenizer).__name__)
        for text in ["Hello world", "  spaces  and pieces"]:
            self.assertEqual(self.tokenizer.encode(text), rebuilt.encode(text))
        self.assertEqual(rebuilt.export_encoding_config().fingerprint, config.fingerprint)

    def test_explicit_special_token_ids(self):
        config = self.tokenizer.export_encoding_config().to_dict()
        self.assertEqual(config["encoding"]["special_tokens"]["unk_token"]["id"], self.tokenizer.unk_token_id)
        self.assertEqual(config["encoding"]["special_tokens"]["bos_token"]["id"], self.tokenizer.bos_token_id)
