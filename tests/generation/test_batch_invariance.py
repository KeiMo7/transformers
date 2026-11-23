# Copyright 2026 The HuggingFace Team Inc.
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
"""Tests for batch-composition invariance of generation.

The fixtures come from the repository's own model testers (no Hub download): a tiny randomly-initialized
`LlamaForCausalLM` plus mixed-length prompts derived from `ids_tensor`, exactly like the rest of the generation
test suite builds models.
"""

import unittest

from transformers import LlamaConfig, LlamaForCausalLM, set_seed
from transformers.testing_utils import require_torch, torch_device


if torch_device is not None:
    import torch

    from transformers.generation.batch_invariance import (
        STATUS_ERROR,
        STATUS_PASS,
        STATUS_UNCOVERED,
        BatchInvarianceChecker,
        batch_fingerprint,
        counter_multinomial,
        counter_uniform,
        coverage_reasons,
        detect_padding_side,
        flip_padding,
        row_key,
        validate_batch_invariance,
    )


def _tiny_config(vocab_size=99, num_hidden_layers=2):
    """A tiny decoder-only config, matching the shape the llama test fixture uses."""
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _tiny_model(vocab_size=99, seed=0):
    torch.manual_seed(seed)
    model = LlamaForCausalLM(_tiny_config(vocab_size=vocab_size)).to(torch_device).eval()
    torch.set_grad_enabled(False)
    return model


def _mixed_length_prompts(vocab_size=99, batch_size=8, max_len=9, seed=1234):
    """Prompts of decreasing length (max_len ... max_len - batch_size + 1), token ids drawn from the vocabulary."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.randint(3, vocab_size, (batch_size, max_len), generator=generator)
    return [base[i, : max_len - i] for i in range(batch_size)]


def _pad(prompts, side="left", pad_token_id=0):
    width = max(p.shape[0] for p in prompts)
    rows, masks = [], []
    for prompt in prompts:
        pad = width - prompt.shape[0]
        if pad == 0:
            rows.append(prompt.clone())
            masks.append(torch.ones(prompt.shape[0], dtype=torch.long))
        elif side == "left":
            rows.append(torch.cat([torch.full((pad,), pad_token_id), prompt]))
            masks.append(
                torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(prompt.shape[0], dtype=torch.long)])
            )
        else:
            rows.append(torch.cat([prompt, torch.full((pad,), pad_token_id)]))
            masks.append(
                torch.cat([torch.ones(prompt.shape[0], dtype=torch.long), torch.zeros(pad, dtype=torch.long)])
            )
    return torch.stack(rows).to(torch_device), torch.stack(masks).to(torch_device)


def _strip_generated(tokens, prompt_len, num_generated):
    """Keep the generated part of a row, for a row whose prompt is `prompt_len` tokens long."""
    start = tokens.shape[0] - num_generated
    return tokens[start:]


GEN_KWARGS = {"max_new_tokens": 8, "do_sample": False, "pad_token_id": 0, "eos_token_id": 2}


@require_torch
class BatchInvarianceLayoutTest(unittest.TestCase):
    """Pure layout helpers: padding classification and re-rendering."""

    def test_detect_padding_side(self):
        prompts = _mixed_length_prompts(batch_size=4)
        for side in ("left", "right"):
            ids, mask = _pad(prompts, side=side)
            self.assertEqual(detect_padding_side(ids, mask), side)
        # full-width rows do not vote
        ids, mask = _pad(prompts, side="left")
        full = torch.ones_like(mask)
        self.assertEqual(detect_padding_side(ids, full), "none")
        # mixed
        left_ids, left_mask = _pad(prompts, side="left")
        right_ids, right_mask = _pad(prompts, side="right")
        mixed_ids, mixed_mask = left_ids.clone(), left_mask.clone()
        # row 0 has no padding (full width), so swap a padded row instead
        mixed_ids[3], mixed_mask[3] = right_ids[3], right_mask[3]
        self.assertEqual(detect_padding_side(mixed_ids, mixed_mask), "mixed")
        # no mask / single row
        self.assertEqual(detect_padding_side(left_ids, None), "none")
        self.assertEqual(detect_padding_side(left_ids[:1], left_mask[:1]), "none")

    def test_flip_padding_preserves_prompts(self):
        prompts = _mixed_length_prompts(batch_size=5)
        ids, mask = _pad(prompts, side="left")
        flipped_ids, flipped_mask = flip_padding(ids, mask, to="right", pad_token_id=0)
        self.assertEqual(detect_padding_side(flipped_ids, flipped_mask), "right")
        for row, prompt in enumerate(prompts):
            self.assertTrue(torch.equal(flipped_ids[row][flipped_mask[row].bool()], prompt))
            self.assertTrue(torch.equal(ids[row][mask[row].bool()], prompt))

    def test_batch_fingerprint_is_stable_and_content_free(self):
        a = torch.tensor([5, 6, 7, 8, 9])
        self.assertEqual(batch_fingerprint(a), batch_fingerprint(a.clone()))
        self.assertNotEqual(batch_fingerprint(a), batch_fingerprint(torch.tensor([5, 6, 7, 8, 10])))
        # The fingerprint contains no token id from the payload.
        for token in a.tolist():
            self.assertNotIn(f" {token} ", batch_fingerprint(a))


@require_torch
class BatchInvariantSamplingTest(unittest.TestCase):
    """The per-row counter RNG that makes sampling independent of batch composition."""

    def test_counter_uniform_is_per_row_and_per_step(self):
        keys = torch.tensor([11, 22, 33], dtype=torch.int64)
        step1 = counter_uniform(keys, step=0)
        step2 = counter_uniform(keys, step=1)
        self.assertTrue(torch.equal(counter_uniform(keys, step=0), step1))
        self.assertFalse(torch.equal(step1, step2))
        self.assertTrue(((step1 >= 0) & (step1 < 1)).all())
        # a row's draw does not depend on the other rows in the tensor
        self.assertTrue(torch.equal(counter_uniform(keys[1:2], step=0), step1[1:2]))

    def test_row_key_ignores_padding(self):
        prompt = torch.tensor([7, 8, 9])
        left = torch.tensor([0, 0, 0, 7, 8, 9])
        mask = torch.tensor([0, 0, 0, 1, 1, 1])
        # `per_row_keys` strips padding, so the same prompt gets the same key in every layout.
        from transformers.generation.batch_invariance import per_row_keys

        self.assertEqual(per_row_keys(left[None], mask[None])[0].item(), row_key(prompt))
        self.assertEqual(per_row_keys(prompt[None], None)[0].item(), row_key(prompt))

    def test_counter_multinomial_matches_distribution(self):
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        counts = torch.zeros(4)
        trials = 4000
        for step in range(trials):
            token = counter_multinomial(probs, torch.tensor([12345]), step).item()
            counts[token] += 1
        empirical = counts / trials
        self.assertTrue(torch.allclose(empirical, probs[0], atol=0.03), empirical)


@require_torch
class BatchInvarianceGenerationTest(unittest.TestCase):
    """End-to-end invariance on the repository's tiny llama fixture."""

    @classmethod
    def setUpClass(cls):
        cls.model = _tiny_model()
        cls.prompts = _mixed_length_prompts()
        cls.left_ids, cls.left_mask = _pad(cls.prompts, side="left")
        cls.num_generated = GEN_KWARGS["max_new_tokens"]

    def _single_row_outputs(self, **kwargs):
        outputs = []
        for prompt in self.prompts:
            set_seed(0)
            out = self.model.generate(prompt[None].to(torch_device), **{**GEN_KWARGS, **kwargs})
            outputs.append(_strip_generated(out[0], prompt.shape[0], self.num_generated))
        return outputs

    def test_single_vs_batched_greedy(self):
        """A row generated alone is token-for-token identical to the same row inside any batch composition."""
        singles = self._single_row_outputs()
        for batch_size in (1, 2, 3, 4, 8):
            ids, mask = self.left_ids[:batch_size], self.left_mask[:batch_size]
            out = self.model.generate(ids, attention_mask=mask, **GEN_KWARGS)
            for row in range(batch_size):
                self.assertTrue(
                    torch.equal(out[row][-self.num_generated :], singles[row]),
                    f"batch_size={batch_size} row={row}: {out[row][-self.num_generated :].tolist()} "
                    f"!= {singles[row].tolist()}",
                )

    def test_batch_size_sweep_is_stable(self):
        """Batch size 1..8 gives the same tokens for the same row (left-padded, greedy)."""
        reference = None
        for batch_size in range(1, 9):
            ids, mask = self.left_ids[:batch_size], self.left_mask[:batch_size]
            out = self.model.generate(ids, attention_mask=mask, **GEN_KWARGS)
            generated = out[0][-self.num_generated :]
            if reference is None:
                reference = generated
            self.assertTrue(torch.equal(generated, reference), f"row 0 changed at batch_size={batch_size}")

    def test_beam_search_invariance(self):
        """Beam search keeps beam width and ordering when peers end early."""
        kwargs = {
            "max_new_tokens": 8,
            "num_beams": 3,
            "num_return_sequences": 2,
            "do_sample": False,
            "pad_token_id": 0,
            "eos_token_id": 2,
        }
        singles = []
        for prompt in self.prompts:
            set_seed(0)
            out = self.model.generate(
                prompt[None].to(torch_device), output_scores=True, return_dict_in_generate=True, **kwargs
            )
            singles.append((out.sequences[0][-8:], out.sequences_scores[0]))
        out = self.model.generate(
            self.left_ids, attention_mask=self.left_mask, output_scores=True, return_dict_in_generate=True, **kwargs
        )
        differences = []
        for row in range(len(self.prompts)):
            for sequence in range(2):
                flat = row * 2 + sequence
                got = out.sequences[flat][-8:]
                expected = singles[row][0] if sequence == 0 else None
                if sequence == 0:
                    if not torch.equal(got, expected):
                        differences.append((row, "tokens", got.tolist(), expected.tolist()))
                # ordering is preserved by comparing score ordering within a row
            score_pair = out.sequences_scores[row * 2 : row * 2 + 2]
            self.assertGreaterEqual(score_pair[0].item(), score_pair[1].item(), f"row {row} beam order changed")
        self.assertEqual(differences, [], f"beam differences: {differences}")

    def test_seeded_sampling_single_vs_batched(self):
        """With `batch_invariant_sampling`, a fixed seed gives a row the same random stream alone and batched."""
        kwargs = {
            "max_new_tokens": 8,
            "do_sample": True,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 0.95,
            "batch_invariant_sampling": True,
            "pad_token_id": 0,
            "eos_token_id": 2,
        }
        singles = []
        for prompt in self.prompts:
            set_seed(7)
            singles.append(self.model.generate(prompt[None].to(torch_device), **kwargs)[0][-8:])
        set_seed(7)
        out = self.model.generate(self.left_ids, attention_mask=self.left_mask, **kwargs)
        for row in range(len(self.prompts)):
            self.assertTrue(
                torch.equal(out[row][-8:], singles[row]),
                f"row {row}: {out[row][-8:].tolist()} != {singles[row].tolist()}",
            )

    def test_sampling_without_flag_is_not_covered(self):
        """Historical sampling is batch-coupled, and the checker says so instead of passing."""
        kwargs = {"max_new_tokens": 8, "do_sample": True, "pad_token_id": 0, "eos_token_id": 2}
        _, report = validate_batch_invariance(
            self.model, self.left_ids, self.left_mask, compositions=("single",), **kwargs
        )
        self.assertEqual(report.status, STATUS_UNCOVERED)
        self.assertTrue(any("RNG" in reason for reason in report.uncovered_reasons))


@require_torch
class BatchInvarianceRejectionTest(unittest.TestCase):
    """The explicit rejection list: unsupported combinations are `uncovered`, never `pass`."""

    @classmethod
    def setUpClass(cls):
        cls.model = _tiny_model()
        cls.prompts = _mixed_length_prompts(batch_size=4)

    def test_right_padding_is_rejected(self):
        ids, mask = _pad(self.prompts, side="right")
        _, report = validate_batch_invariance(self.model, ids, mask, **GEN_KWARGS)
        self.assertEqual(report.status, STATUS_UNCOVERED)
        self.assertTrue(any("right padding" in reason for reason in report.uncovered_reasons))

    def test_mixed_padding_is_rejected(self):
        left_ids, left_mask = _pad(self.prompts, side="left")
        right_ids, right_mask = _pad(self.prompts, side="right")
        mixed_ids, mixed_mask = left_ids.clone(), left_mask.clone()
        # row 0 is full width and carries no padding, so use a padded row for the mixed layout
        mixed_ids[2], mixed_mask[2] = right_ids[2], right_mask[2]
        _, report = validate_batch_invariance(self.model, mixed_ids, mixed_mask, **GEN_KWARGS)
        self.assertEqual(report.status, STATUS_UNCOVERED)
        self.assertEqual(report.padding_side, "mixed")

    def test_custom_logits_processor_is_rejected(self):
        ids, mask = _pad(self.prompts, side="left")

        from transformers import LogitsProcessorList

        def callback(input_ids, scores):
            return scores

        _, report = validate_batch_invariance(
            self.model, ids, mask, logits_processor=LogitsProcessorList([callback]), **GEN_KWARGS
        )
        self.assertEqual(report.status, STATUS_UNCOVERED)
        self.assertTrue(any("logits_processor" in reason for reason in report.uncovered_reasons))

    def test_encoder_decoder_is_rejected(self):
        from transformers import GenerationConfig

        config = GenerationConfig(do_sample=False, num_beams=1)
        reasons = coverage_reasons(config, "left", is_encoder_decoder=True)
        self.assertTrue(any("encoder-decoder" in reason for reason in reasons))

    def test_max_time_is_rejected(self):
        from transformers import GenerationConfig

        config = GenerationConfig(do_sample=False, num_beams=1, max_time=1.0)
        self.assertTrue(any("max_time" in reason for reason in coverage_reasons(config, "left")))

    def test_coverage_empty_for_supported_greedy(self):
        from transformers import GenerationConfig

        config = GenerationConfig(do_sample=False, num_beams=1)
        self.assertEqual(coverage_reasons(config, "left"), [])
        self.assertEqual(coverage_reasons(config, "none"), [])


@require_torch
class BatchInvarianceReportTest(unittest.TestCase):
    """Report shape, localization, degradation and content-free guarantees."""

    @classmethod
    def setUpClass(cls):
        cls.model = _tiny_model()
        cls.prompts = _mixed_length_prompts(batch_size=4)
        cls.left_ids, cls.left_mask = _pad(cls.prompts, side="left")

    def test_pass_report_has_no_differences_and_stop_info(self):
        out, report = validate_batch_invariance(
            self.model, self.left_ids, self.left_mask, compositions=("single", "permute"), **GEN_KWARGS
        )
        self.assertEqual(report.status, STATUS_PASS, report.summary())
        self.assertEqual(report.differences, [])
        self.assertEqual(len(report.stop_info), len(self.prompts))
        for stop in report.stop_info:
            self.assertIn(stop.reason, {"eos", "max_length", "unknown"})
            self.assertGreaterEqual(stop.position, 0)

    def test_truncation_position_is_observable(self):
        """A row that hits the length cap reports the exact truncation position."""
        kwargs = {**GEN_KWARGS, "max_new_tokens": 5, "eos_token_id": None}
        _, report = validate_batch_invariance(
            self.model, self.left_ids, self.left_mask, compositions=("none",), **kwargs
        )
        self.assertTrue(all(stop.reason == "max_length" for stop in report.stop_info))
        self.assertTrue(all(stop.position == 5 for stop in report.stop_info))

    def test_eos_position_is_observable(self):
        """A row that hits EOS reports `eos` and the position it stopped at."""
        config = {"max_new_tokens": 6, "do_sample": False, "pad_token_id": 0, "eos_token_id": 2}
        out = self.model.generate(self.left_ids, attention_mask=self.left_mask, **config)
        _, report = validate_batch_invariance(
            self.model, self.left_ids, self.left_mask, compositions=("none",), **config
        )
        eos_rows = [s for s in report.stop_info if s.reason == "eos"]
        if eos_rows:
            for stop in eos_rows:
                row = out[stop.row][-6:]
                self.assertEqual(int(row[stop.position - 1].item()), 2)
        else:
            # No row hit EOS with these random weights: the classification is still well-formed.
            self.assertTrue(all(s.reason == "max_length" for s in report.stop_info))

    def test_checker_failure_degrades_to_error(self):
        """A checker that raises records `error` and leaves the generated output untouched."""
        ids, mask = self.left_ids, self.left_mask
        clean = self.model.generate(ids, attention_mask=mask, **GEN_KWARGS)
        original = BatchInvarianceChecker.run

        def boom(*args, **kwargs):
            raise RuntimeError("synthetic checker failure")

        BatchInvarianceChecker.run = boom
        try:
            out = self.model.generate(ids, attention_mask=mask, validate_batch_invariance=True, **GEN_KWARGS)
            report = self.model._last_batch_invariance_report
        finally:
            BatchInvarianceChecker.run = original
        self.assertTrue(torch.equal(out, clean))
        self.assertEqual(report.status, STATUS_ERROR)
        self.assertIn("synthetic checker failure", report.error)
        self.assertIn("generation unaffected", report.summary())

    def test_flag_off_changes_nothing(self):
        """Default config runs no check and leaves the report attribute unset."""
        report_before = self.model._last_batch_invariance_report
        try:
            self.model._last_batch_invariance_report = None
            out_default = self.model.generate(self.left_ids, attention_mask=self.left_mask, **GEN_KWARGS)
            self.assertIsNone(self.model._last_batch_invariance_report)
            out_checked = self.model.generate(
                self.left_ids, attention_mask=self.left_mask, validate_batch_invariance=True, **GEN_KWARGS
            )
            self.assertIsNotNone(self.model._last_batch_invariance_report)
            self.assertTrue(torch.equal(out_default, out_checked))
        finally:
            self.model._last_batch_invariance_report = report_before

    def test_report_is_content_free(self):
        """Reports carry fingerprints and digests, never raw prompt token ids."""
        out, report = validate_batch_invariance(
            self.model, self.left_ids, self.left_mask, compositions=("single",), **GEN_KWARGS
        )
        payload = report.to_json()
        prompt_tokens = sorted(set(self.left_ids[self.left_mask.bool()].tolist()))
        leaked = [token for token in prompt_tokens if f"[{token}," in payload or f", {token}]" in payload]
        self.assertEqual(leaked, [], f"prompt token ids leaked into the report: {leaked}")
        for difference in report.differences:
            self.assertTrue(difference.fingerprint)
            self.assertIn(":", difference.fingerprint)

    def test_right_padding_difference_is_localized_to_position(self):
        """Forced diagnostics on a right-padded batch attribute the divergence to position ids at step 0."""
        ids, mask = _pad(self.prompts, side="right")
        out = self.model.generate(ids, attention_mask=mask, **GEN_KWARGS)
        from transformers import GenerationConfig

        config = GenerationConfig(pad_token_id=0, eos_token_id=2, max_new_tokens=8, do_sample=False)
        checker = BatchInvarianceChecker(
            config,
            ids,
            out,
            attention_mask=mask,
            generation_mode="greedy_search",
        )
        report = checker.run_single_row_oracle(
            lambda prompt: (
                self.model.generate(prompt, pad_token_id=0, eos_token_id=2, max_new_tokens=8, do_sample=False),
                None,
            ),
            force=True,
        )
        # Rejected, never passing, but with localized diagnostics.
        self.assertEqual(report.status, STATUS_UNCOVERED)
        self.assertTrue(report.uncovered_reasons)
        self.assertTrue(report.differences, "expected localized differences on a right-padded batch")
        for difference in report.differences:
            self.assertEqual(difference.layer, "position")
            self.assertEqual(difference.step, 0)

    def test_uncovered_report_is_not_pass(self):
        _, report = validate_batch_invariance(
            self.model, self.left_ids, self.left_mask, compositions=("none",), **GEN_KWARGS
        )
        # "none" performs no comparison on a supported config -> pass, but with no differences.
        self.assertEqual(report.status, STATUS_PASS)
        right_ids, right_mask = _pad(self.prompts, side="right")
        _, report = validate_batch_invariance(self.model, right_ids, right_mask, **GEN_KWARGS)
        self.assertFalse(report.is_pass)
        self.assertEqual(report.status, STATUS_UNCOVERED)

    def test_report_round_trips_through_json(self):
        import json

        _, report = validate_batch_invariance(
            self.model, self.left_ids, self.left_mask, compositions=("permute",), **GEN_KWARGS
        )
        payload = json.loads(report.to_json())
        self.assertEqual(payload["status"], report.status)
        self.assertIn("summary", payload)
        self.assertEqual(payload["batch_size"], len(self.prompts))


@require_torch
class BatchInvarianceStreamingTest(unittest.TestCase):
    """Streamed tokens must match the returned tokens exactly."""

    def test_streamed_tokens_match_returned(self):
        model = _tiny_model()
        prompts = _mixed_length_prompts(batch_size=4)
        ids, mask = _pad(prompts, side="left")

        class Collector:
            def __init__(self):
                self.parts = []

            def put(self, value):
                self.parts.append(value.clone())

            def end(self):
                pass

        collector = Collector()
        model.generate(ids, attention_mask=mask, streamer=collector, **GEN_KWARGS)
        returned_out = model.generate(ids, attention_mask=mask, **GEN_KWARGS)
        first = collector.parts[0]
        rest = [part[:, None] for part in collector.parts[1:]]
        streamed = torch.cat([first] + rest, dim=1)
        self.assertTrue(torch.equal(streamed[:, ids.shape[1] :], returned_out[:, ids.shape[1] :]))

    def test_streamer_with_validation_is_uncovered(self):
        """Comparing under a streamer would stream twice, so the check declines rather than passing."""
        model = _tiny_model()
        prompts = _mixed_length_prompts(batch_size=4)
        ids, mask = _pad(prompts, side="left")

        class Collector:
            def put(self, value):
                pass

            def end(self):
                pass

        model.generate(ids, attention_mask=mask, streamer=Collector(), validate_batch_invariance=True, **GEN_KWARGS)
        report = model._last_batch_invariance_report
        self.assertEqual(report.status, STATUS_UNCOVERED)
        self.assertTrue(any("streamer" in reason for reason in report.uncovered_reasons))


if __name__ == "__main__":
    unittest.main()
