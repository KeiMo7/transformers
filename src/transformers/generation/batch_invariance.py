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
"""Batch-composition invariance for generation: checks, explicit rejections, and a per-row RNG.

Generation is sequential, and two things couple a row to the rest of its batch:

1. **Floating point reduction order.** A row's logits depend very slightly on the shape of the tensor they are
   computed in. On CPU float32 the same row alone and inside a batch differ by ~1e-7 in the last bits. That is
   harmless by itself, but it can flip an argmax when the top two tokens are near-tied.
2. **Stateful decoding decisions.** Padding side, position ids, whether batch peers are still running, and which
   random numbers a row draws all depend on the batch it happens to sit in.

This module makes that coupling observable instead of surprising, and fixes the part that is fixable:

* [`BatchInvarianceChecker`] compares a generated batch against re-compositions of the same prompts and localizes
  each difference to a layer (`mask`, `position`, `eos`, `cache`, `structural`).
* [`validate_batch_invariance`] is the explicit offline/audit entry point. It returns the untouched output plus a
  [`BatchInvarianceReport`].
* `GenerationConfig.validate_batch_invariance=True` runs the same comparison inside `generate`. The returned
  structure does not change; the report is logged and stored on `model._last_batch_invariance_report`.
* `GenerationConfig.batch_invariant_sampling=True` switches sampling to a counter-based per-row RNG, so a row's
  random stream no longer depends on its batch peers. When the flag is off, the sampler behaves exactly as before.

Combinations that genuinely cannot be guaranteed are **listed and rejected** with report status `uncovered`, never
silently passed: mixed padding, right padding for decoder-only models, encoder-decoder models, custom samplers or
stopping criteria, `max_time`, multi-token prediction, and sampling without `batch_invariant_sampling`.

A failure inside the checker never affects generation: the report becomes `error` and the output is returned as-is.
Everything is torch-only, CPU-friendly, and adds no dependency. Reports carry batch fingerprints and token digests
only -- never prompt content.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


logger = logging.getLogger(__name__)


# Difference layers.
DIFF_STRUCTURAL = "structural"  # tokens differ and the checker cannot attribute it to a specific layer
DIFF_MASK = "mask"  # padding / attention-mask handling
DIFF_POSITION = "position"  # position ids (typically right or mixed padding)
DIFF_EOS = "eos"  # end-of-sequence handling / early stop
DIFF_CACHE = "cache"  # batch-shaped kv-cache
DIFF_NUMERIC = "numeric"  # effective tokens are identical; only float noise remains
DIFF_TOKENS_TOLERATED = "tokens_tolerated"  # tokens differ, but the divergence is a near-tie

# Report statuses.
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_UNCOVERED = "uncovered"
STATUS_ERROR = "error"

# Layers that make a report fail.
_FAILING_LAYERS = frozenset({DIFF_STRUCTURAL, DIFF_MASK, DIFF_POSITION, DIFF_EOS, DIFF_CACHE})

# Generation modes the checker can reason about. Anything else is `uncovered`, never `pass`.
SUPPORTED_MODES = frozenset({"greedy_search", "sample", "beam_search"})

LEFT = "left"
RIGHT = "right"
MIXED = "mixed"

# Why a row stopped generating.
STOP_EOS = "eos"
STOP_MAX_LENGTH = "max_length"
STOP_UNKNOWN = "unknown"


def _to_builtin(value: Any) -> Any:
    """Recursively convert tensors into JSON-serializable Python objects."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def batch_fingerprint(sequence: torch.Tensor) -> str:
    """Stable, content-free fingerprint of a prompt or a generated row.

    Reports carry this instead of the tokens themselves, so logs and alerts never contain prompt content while
    two calls on the same ids still agree across processes and machines.
    """
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.as_tensor(sequence)
    payload = sequence.detach().to(torch.int64).cpu().contiguous()
    digest = hashlib.blake2b(payload.numpy().tobytes(), digest_size=8).hexdigest()
    return f"{digest}:{tuple(payload.shape)}"


def token_digest(tokens: torch.Tensor | list[int]) -> str:
    """Short digest of a token window, used in reports so the tokens themselves stay out of logs."""
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens)
    tokens = tokens.detach().to(torch.int64).cpu().contiguous()
    return hashlib.blake2b(tokens.numpy().tobytes(), digest_size=6).hexdigest()


def detect_padding_side(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> str:
    """Classify a 2D batch as `left`, `right`, `mixed`, or `none` padding.

    A row that fills the whole width carries no padding and therefore says nothing about the side; only padded
    rows vote. A batch where every row is full width has no padding.
    """
    if attention_mask is None or input_ids.ndim != 2 or input_ids.shape[0] < 2:
        return "none"
    if attention_mask.shape != input_ids.shape:
        return "none"
    if bool((attention_mask == 1).all()):
        return "none"
    sides = set()
    for row in attention_mask.tolist():
        if 1 not in row:
            continue
        first_one = row.index(1)
        last_one = len(row) - 1 - row[::-1].index(1)
        if first_one > 0:
            sides.add(LEFT)
        elif last_one < len(row) - 1:
            sides.add(RIGHT)
        # else: full-width row, no padding -> no vote
    if not sides:
        return "none"
    if len(sides) == 1:
        return sides.pop()
    return MIXED


def flip_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    to: str,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-render a padded 2D batch with its padding moved to `to` (`"left"` or `"right"`).

    Rows keep their token order and their real length; only the padding columns move, so the result represents
    exactly the same prompts. The checker does not use this for decoder-only models (right padding is rejected);
    it is exposed for diagnostics and for encoder-decoder audits.
    """
    if to not in (LEFT, RIGHT):
        raise ValueError(f"`to` must be 'left' or 'right', got {to!r}")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("`input_ids` and `attention_mask` must be 2D tensors of the same shape")

    batch, width = input_ids.shape
    new_ids = input_ids.new_full((batch, width), pad_token_id)
    new_mask = attention_mask.new_zeros((batch, width))
    for row in range(batch):
        real = input_ids[row][attention_mask[row].bool()]
        length = int(real.shape[0])
        if to == LEFT:
            new_ids[row, width - length :] = real
            new_mask[row, width - length :] = 1
        else:
            new_ids[row, :length] = real
            new_mask[row, :length] = 1
    return new_ids, new_mask


def add_padding_column(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    side: str,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one fully-masked padding column on `side`, keeping every prompt unchanged."""
    if side not in (LEFT, RIGHT):
        raise ValueError(f"`side` must be 'left' or 'right', got {side!r}")
    pad_ids = input_ids.new_full((input_ids.shape[0], 1), pad_token_id)
    pad_mask = attention_mask.new_zeros((attention_mask.shape[0], 1))
    if side == LEFT:
        return torch.cat([pad_ids, input_ids], dim=1), torch.cat([pad_mask, attention_mask], dim=1)
    return torch.cat([input_ids, pad_ids], dim=1), torch.cat([attention_mask, pad_mask], dim=1)


# -- counter-based, per-row RNG ------------------------------------------------------------------------------
#
# `torch.multinomial` draws from the global generator in flat batch order, so a row's random stream depends on how
# many rows precede it and how many columns the batch has. The helpers below derive a uniform from
# `(seed, row key, step)` with a hash instead, which is independent of batch composition. They are used only when
# `GenerationConfig.batch_invariant_sampling=True`.

_M64 = (1 << 64) - 1
_M53 = (1 << 53) - 1
_SIGN = 1 << 63


def _to_int64(value: int) -> int:
    """Wrap an unsigned 64-bit value into the signed range accepted by `torch.tensor`/`torch.full`."""
    value &= _M64
    return value - (1 << 64) if value >= _SIGN else value


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    """Vectorized `splitmix64` finalizer on int64 tensors (the multiplications wrap, which is intended)."""
    z = x + _to_int64(0x9E3779B97F4A7C15)
    z = (z ^ ((z >> 30) & 0x3FFFFFFFFFFFFFFF)) * _to_int64(0xBF58476D1CE4E5B9)
    z = (z ^ ((z >> 27) & 0x1FFFFFFFFFFFFFFF)) * _to_int64(0x94D049BB133111EB)
    return z ^ ((z >> 31) & 0x1FFFFFFFFFFFFFFF)


def row_key(sequence: torch.Tensor, seed: int = 0) -> int:
    """A per-row 63-bit key from a row's real tokens and the run seed."""
    if not isinstance(sequence, torch.Tensor):
        sequence = torch.as_tensor(sequence)
    payload = sequence.detach().to(torch.int64).cpu().contiguous()
    digest = hashlib.blake2b(payload.numpy().tobytes(), digest_size=8).digest()
    return (int.from_bytes(digest, "big") ^ ((int(seed) * 0x9E3779B97F4A7C15) & _M64)) & 0x7FFFFFFFFFFFFFFF


def per_row_keys(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    seed: int = 0,
) -> torch.Tensor:
    """One key per row of `input_ids`, derived from the row's *unmasked* tokens.

    Padding is stripped first, so the same prompt gets the same key no matter where it sits in a batch, which is
    what makes a row's random stream reproducible across batch compositions.
    """
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    keys = []
    for row in range(input_ids.shape[0]):
        real = input_ids[row]
        if attention_mask is not None and attention_mask.shape == input_ids.shape:
            real = real[attention_mask[row].bool()]
        keys.append(row_key(real, seed))
    return torch.tensor(keys, dtype=torch.int64, device=input_ids.device)


def counter_uniform(keys: torch.Tensor, step: int, seed: int = 0) -> torch.Tensor:
    """Uniform floats in `[0, 1)` for each row key at a given generation step."""
    keys = keys.to(torch.int64)
    x = torch.full_like(keys, _to_int64((int(seed) * 0x9E3779B97F4A7C15) & _M64))
    x = x ^ keys ^ _to_int64((int(step) * 0xD1342543DE82EF95) & _M64)
    z = _splitmix64(x)
    return (z >> 11 & _M53).to(torch.float64) / float(1 << 53)


def counter_uniform_vocab(keys: torch.Tensor, vocab_size: int, step: int, seed: int = 0) -> torch.Tensor:
    """Uniform floats in `[0, 1)` of shape `(batch, vocab_size)` for a given generation step.

    Each `(row, token)` pair gets its own draw, derived from the row key, the step and the token index, so the
    whole matrix is reproducible regardless of batch shape or composition.
    """
    keys = keys.to(torch.int64).reshape(-1, 1)
    vocab = torch.arange(vocab_size, dtype=torch.int64, device=keys.device).reshape(1, -1)
    x = torch.full_like(keys, _to_int64((int(seed) * 0x9E3779B97F4A7C15) & _M64))
    x = x ^ keys ^ _to_int64((int(step) * 0xD1342543DE82EF95) & _M64)
    # Mix the token index in with the golden-ratio constant so neighbouring vocab entries are decorrelated.
    x = x ^ (vocab * _to_int64(0x9E3779B97F4A7C15))
    z = _splitmix64(x)
    return (z >> 11 & _M53).to(torch.float64) / float(1 << 53)


def counter_multinomial(probs: torch.Tensor, keys: torch.Tensor, step: int, seed: int = 0) -> torch.Tensor:
    """Sample one token per row from `probs`, using a per-row counter RNG.

    Gumbel-max sampling, which is equivalent in distribution to `torch.multinomial(probs, 1)` but with the random
    draw derived per `(row, token)` instead of from the flat batch stream.
    """
    vocab_size = probs.shape[-1]
    u = counter_uniform_vocab(keys.to(probs.device), vocab_size, step, seed)
    u = u.clamp_(min=1e-30, max=1.0 - 1e-16)
    gumbel = -torch.log(-torch.log(u))
    scores = probs.to(torch.float64).log() + gumbel
    return scores.argmax(dim=-1, keepdim=True)


# -- report types --------------------------------------------------------------------------------------------


@dataclass
class RowDifference:
    """One localized inconsistency for one row.

    Attributes:
        row (`int`): Row index in the validated batch.
        sequence (`int`): Returned-sequence index within that row (when `num_return_sequences > 1`).
        fingerprint (`str`): Fingerprint of the row's prompt. Reports never carry the prompt itself.
        layer (`str`): Best-supported responsible layer (`mask`, `position`, `eos`, `cache`, `structural`).
        step (`int`): Generation step (0-based over generated tokens) of the first difference.
        observed_digest (`str`): Digest of the comparison run's token window around `step`.
        expected_digest (`str`): Digest of the reference run's token window around `step`.
        detail (`str`): Prompt-free explanation, including the tie margin when one is available.
    """

    row: int
    fingerprint: str
    layer: str
    step: int
    sequence: int = 0
    observed_digest: str = ""
    expected_digest: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "sequence": self.sequence,
            "fingerprint": self.fingerprint,
            "layer": self.layer,
            "step": self.step,
            "observed_digest": self.observed_digest,
            "expected_digest": self.expected_digest,
            "detail": self.detail,
        }

    def to_repro(self) -> dict[str, Any]:
        """Prompt-free reproducer description: fingerprint, position, layer."""
        return {
            "fingerprint": self.fingerprint,
            "sequence": self.sequence,
            "layer": self.layer,
            "step": self.step,
        }


@dataclass
class RowStopInfo:
    """Why a row stopped generating, and where.

    Attributes:
        row (`int`): Row index.
        sequence (`int`): Returned-sequence index within the row.
        reason (`str`): `eos`, `max_length`, or `unknown`.
        position (`int`): Number of generated tokens effectively kept (truncation position for `max_length`).
    """

    row: int
    sequence: int
    reason: str
    position: int

    def to_dict(self) -> dict[str, Any]:
        return {"row": self.row, "sequence": self.sequence, "reason": self.reason, "position": self.position}


@dataclass
class BatchInvarianceReport:
    """Result of one validation run.

    A report is `pass` only when every comparable row matched token-for-token (near-tie flips are reported as
    `tokens_tolerated` and kept out of the pass decision unless `tolerance=0`). Configurations that cannot be
    guaranteed are `uncovered` with a reason, never `pass`. A failure inside the checker makes it `error` and
    leaves generation untouched.

    Attributes:
        status (`str`): `pass`, `fail`, `uncovered`, or `error`.
        mode (`str`): Generation mode the report applies to.
        compositions (`list[str]`): Comparisons that ran.
        batch_size (`int`): Number of rows validated.
        padding_side (`str`): Padding side of the user's batch (`left`, `right`, `mixed`, `none`).
        differences (`list[RowDifference]`): Localized differences; empty on `pass`.
        stop_info (`list[RowStopInfo]`): Per-row truncation position and reason.
        uncovered_reasons (`list[str]`): Why rows could not be judged (the explicit rejection list).
        error (`str | None`): Set when the checker failed and generation was left untouched.
        metadata (`dict`): Prompt-free facts (mode flags, dtype, tolerance, ...).
    """

    status: str = STATUS_PASS
    mode: str = "unknown"
    compositions: list[str] = field(default_factory=list)
    batch_size: int = 0
    padding_side: str = "none"
    differences: list[RowDifference] = field(default_factory=list)
    stop_info: list[RowStopInfo] = field(default_factory=list)
    uncovered_reasons: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_pass(self) -> bool:
        return self.status == STATUS_PASS

    def summary(self) -> str:
        """One-line, content-free summary suitable for logs and alerts."""
        if self.status == STATUS_ERROR:
            return f"batch-invariance check could not run ({self.error}); generation unaffected"
        if self.status == STATUS_UNCOVERED:
            return "batch-invariance not covered: " + "; ".join(self.uncovered_reasons)
        if self.status == STATUS_PASS:
            return (
                f"batch-invariance ok (mode={self.mode}, batch={self.batch_size}, padding={self.padding_side}, "
                f"compositions={self.compositions})"
            )
        layers = sorted({d.layer for d in self.differences})
        rows = sorted({d.row for d in self.differences})
        return (
            f"batch-invariance failed (mode={self.mode}, batch={self.batch_size}, padding={self.padding_side}): "
            f"{len(self.differences)} difference(s), layers={layers}, rows={rows}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mode": self.mode,
            "compositions": list(self.compositions),
            "batch_size": self.batch_size,
            "padding_side": self.padding_side,
            "differences": [d.to_dict() for d in self.differences],
            "reproducers": [d.to_repro() for d in self.differences],
            "stop_info": [s.to_dict() for s in self.stop_info],
            "uncovered_reasons": list(self.uncovered_reasons),
            "error": self.error,
            "metadata": _to_builtin(self.metadata),
            "summary": self.summary(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def raise_on_failure(self):
        """Raise a `RuntimeError` carrying `summary()` when the report is a failure."""
        if self.status == STATUS_FAIL:
            raise RuntimeError(self.summary())


# -- row comparison ------------------------------------------------------------------------------------------


def eos_ids_of(generation_config) -> set[int]:
    eos = getattr(generation_config, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(e) for e in eos}


def first_eos_position(row: torch.Tensor, eos_ids: set[int]) -> int | None:
    if not eos_ids:
        return None
    for index, token in enumerate(row.tolist()):
        if token in eos_ids:
            return index
    return None


def stop_info_for_row(
    reference_row: torch.Tensor,
    eos_ids: set[int],
    generated_length: int,
    row: int = 0,
    sequence: int = 0,
) -> RowStopInfo:
    """Classify why a row stopped: hit EOS, or ran to the maximum generated length."""
    eos_position = first_eos_position(reference_row, eos_ids)
    if eos_position is not None:
        return RowStopInfo(row=row, sequence=sequence, reason=STOP_EOS, position=eos_position + 1)
    if reference_row.shape[0] >= generated_length:
        return RowStopInfo(row=row, sequence=sequence, reason=STOP_MAX_LENGTH, position=int(generated_length))
    return RowStopInfo(row=row, sequence=sequence, reason=STOP_UNKNOWN, position=int(reference_row.shape[0]))


def compare_rows(
    reference: torch.Tensor,
    comparison: torch.Tensor,
    tolerance: float,
    eos_ids: set[int] | None = None,
    comparison_scores: torch.Tensor | None = None,
    cache_is_batch_shaped: bool = False,
) -> tuple[str, int, str, str, str]:
    """Compare two generated rows and localize the first difference.

    Returns `(layer, step, observed_digest, expected_digest, detail)`. The comparison truncates at the first EOS
    so that padding after EOS (a batch artifact) is not reported as a model difference.

    Localization is evidence-driven: EOS first (a row that stopped earlier explains everything after it), then a
    near-tie check using the comparison distribution's top-2 margin, then cache shape, then `structural`.
    Padding/position attribution is left to the caller, which knows the batch layout.
    """
    eos_ids = eos_ids or set()
    ref_eos = first_eos_position(reference, eos_ids)
    cmp_eos = first_eos_position(comparison, eos_ids)
    ref_body = reference if ref_eos is None else reference[: ref_eos + 1]
    cmp_body = comparison if cmp_eos is None else comparison[: cmp_eos + 1]

    length = min(ref_body.shape[0], cmp_body.shape[0])
    if not torch.equal(ref_body[:length], cmp_body[:length]):
        step = int((ref_body[:length] != cmp_body[:length]).nonzero()[0])
        window = slice(max(0, step - 2), step + 3)
        observed_digest = token_digest(cmp_body[window])
        expected_digest = token_digest(ref_body[window])
        if ref_eos is not None and cmp_eos is not None and ref_eos != cmp_eos and min(ref_eos, cmp_eos) <= step:
            return (
                DIFF_EOS,
                step,
                observed_digest,
                expected_digest,
                f"rows stop at different EOS positions ({ref_eos} vs {cmp_eos}); first divergence at step {step}",
            )
        if comparison_scores is not None and step < comparison_scores.shape[0]:
            step_scores = comparison_scores[step].float()
            if step_scores.numel() >= 2:
                top2 = step_scores.topk(2).values
                margin = float((top2[0] - top2[1]).abs())
                if margin <= tolerance:
                    return (
                        DIFF_TOKENS_TOLERATED,
                        step,
                        observed_digest,
                        expected_digest,
                        f"tokens differ at step {step} on a near-tie (top-2 margin {margin:.3e} <= {tolerance:.1e})",
                    )
        return DIFF_STRUCTURAL, step, observed_digest, expected_digest, f"tokens differ at step {step}"
    if ref_body.shape[0] != cmp_body.shape[0]:
        return (
            DIFF_EOS,
            length,
            token_digest(comparison[length : length + 1]),
            token_digest(reference[length : length + 1]),
            f"identical prefix but different effective lengths ({ref_body.shape[0]} vs {cmp_body.shape[0]})",
        )
    if cache_is_batch_shaped and not torch.equal(reference, comparison):
        return DIFF_CACHE, 0, "", "", "rows agree but the batch-shaped cache holds different values"
    return DIFF_NUMERIC, -1, "", "", ""


def refine_layer(layer: str, step: int, padding_side: str, cache_implementation: str | None) -> str:
    """Turn a generic `structural` verdict into a padding/position/cache verdict when the layout explains it."""
    if layer != DIFF_STRUCTURAL:
        return layer
    if padding_side == RIGHT and step <= 1:
        return DIFF_POSITION
    if padding_side in (RIGHT, MIXED):
        return DIFF_MASK
    if cache_implementation in {"static", "offloaded_static", "quantized"}:
        return DIFF_CACHE
    return DIFF_STRUCTURAL


# -- coverage rules ------------------------------------------------------------------------------------------


def coverage_reasons(
    generation_config,
    padding_side: str,
    mode: str | None = None,
    logits_processor=None,
    logits_warper=None,
    stopping_criteria=None,
    is_encoder_decoder: bool = False,
) -> list[str]:
    """Return every reason why this configuration cannot be certified (empty when it can).

    These are the explicit rejections: a configuration with a reason here gets report status `uncovered`
    instead of a misleading `pass`.
    """
    reasons = []
    mode = _mode_name(mode if mode is not None else generation_config.get_generation_mode())
    if is_encoder_decoder:
        reasons.append("encoder-decoder generation is not covered by the batch-invariance checker")
    if mode not in SUPPORTED_MODES:
        reasons.append(f"generation mode `{mode}` is not covered by the checker")
    if logits_processor is not None:
        reasons.append("a custom `logits_processor` was passed; its batch behavior cannot be reproduced")
    if logits_warper is not None:
        reasons.append("a custom `logits_warper` was passed; its batch behavior cannot be reproduced")
    if stopping_criteria is not None:
        reasons.append("a custom `stopping_criteria` was passed; its batch behavior cannot be reproduced")
    if bool(getattr(generation_config, "use_mtp", False)):
        reasons.append("multi-token prediction uses a custom decoding loop")
    if getattr(generation_config, "max_time", None) is not None:
        reasons.append("`max_time` depends on wall-clock time and is not reproducible")
    if padding_side == MIXED:
        reasons.append(
            "mixed left/right padding in one batch cannot be recomposed deterministically; use a uniform "
            "padding side to validate"
        )
    elif padding_side == RIGHT and not is_encoder_decoder:
        reasons.append(
            "right padding changes decoder-only position ids and can change the first token; use "
            "`padding_side='left'` (or pre-pad left) before batching"
        )
    do_sample = bool(getattr(generation_config, "do_sample", False))
    if do_sample and not bool(getattr(generation_config, "batch_invariant_sampling", False)):
        reasons.append(
            "sampling draws from the batch-wide RNG stream, so single and batched runs cannot agree; set "
            "`batch_invariant_sampling=True` for a per-row random stream"
        )
    if do_sample and int(getattr(generation_config, "num_beams", 1) or 1) > 1:
        reasons.append("beam-search sampling consumes the batch-wide RNG stream and is not covered")
    return reasons


def _mode_name(mode) -> str:
    """Normalize a `GenerationMode` (an `str` enum) into its plain string value."""
    return str(getattr(mode, "value", mode))


class BatchInvarianceChecker:
    """Compare a generated batch against re-compositions of the same prompts.

    The checker reuses the user's already-generated output as the reference, so a runtime check costs one extra
    batched forward over the same batch. It never splits the batch into single rows unless the caller explicitly
    asks for the `single` composition, which is a diagnostic oracle used by the test suite.

    Args:
        generation_config (`GenerationConfig`): The config actually used by `generate`.
        input_ids (`torch.Tensor`): The (possibly padded) input ids, 2D.
        reference_sequences (`torch.Tensor`): Tokens returned by the reference run, including the prompt.
        attention_mask (`torch.Tensor`, *optional*): The attention mask.
        reference_scores (`torch.Tensor`, *optional*): Per-step scores of the reference run, if collected.
        tolerance (`float`): Top-2 logit margin under which a token flip counts as a near-tie.
        generation_mode (`str`, *optional*): Overrides the mode inferred from the config.
        prompt_width (`int`, *optional*): Width of the prompt columns in `reference_sequences`. Defaults to
            `input_ids.shape[1]`.
        is_encoder_decoder (`bool`): Whether the model is encoder-decoder (only used for coverage).
    """

    def __init__(
        self,
        generation_config,
        input_ids: torch.Tensor,
        reference_sequences: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        reference_scores: torch.Tensor | None = None,
        tolerance: float = 1e-4,
        generation_mode: str | None = None,
        prompt_width: int | None = None,
        is_encoder_decoder: bool = False,
        custom_components: tuple[str, ...] = (),
    ):
        if input_ids.ndim != 2:
            raise ValueError("`input_ids` must be a 2D tensor")
        self.generation_config = generation_config
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.reference_sequences = reference_sequences
        self.reference_scores = reference_scores
        self.tolerance = tolerance
        self.is_encoder_decoder = is_encoder_decoder
        # Names of user-supplied pieces the checker's reference config does not otherwise record, e.g. a custom
        # `logits_processor` passed straight to `generate` / `validate_batch_invariance`.
        self.custom_components = tuple(custom_components)
        self.padding_side = detect_padding_side(input_ids, attention_mask)
        self.mode = _mode_name(
            generation_mode if generation_mode is not None else generation_config.get_generation_mode()
        )
        self.prompt_width = int(prompt_width if prompt_width is not None else input_ids.shape[1])
        self.num_generated = int(reference_sequences.shape[1] - self.prompt_width)
        self.num_return_sequences = int(getattr(generation_config, "num_return_sequences", 1) or 1)
        self.num_beams = int(getattr(generation_config, "num_beams", 1) or 1)

    # -- public entry points --------------------------------------------------------------------------------

    def coverage(self, logits_processor=None, logits_warper=None, stopping_criteria=None) -> list[str]:
        reasons = coverage_reasons(
            self.generation_config,
            self.padding_side,
            mode=self.mode,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        for name, label in (
            ("logits_processor", "logits_processor"),
            ("logits_warper", "logits_warper"),
            ("stopping_criteria", "stopping_criteria"),
        ):
            if name in self.custom_components:
                reason = f"a custom `{label}` was passed; its batch behavior cannot be reproduced"
                if reason not in reasons:
                    reasons.append(reason)
        return reasons

    def run(self, rerun_fn: Callable, composition: str = "permute", force: bool = False) -> BatchInvarianceReport:
        """Compare against one re-composition produced by `rerun_fn`.

        `rerun_fn(input_ids, attention_mask)` must return `(sequences, scores)` shaped like the reference, using
        the same model parameters and generation config. `scores` may be `None`.

        With `force=True`, a configuration that would normally be rejected still runs, but the report stays
        `uncovered` and carries the localized `differences` as diagnostics -- it never reports `pass`.
        """
        report = self._base_report([composition])
        reasons = self.coverage()
        if reasons:
            if not force:
                report.status = STATUS_UNCOVERED
                report.uncovered_reasons.extend(reasons)
                return report
            report.uncovered_reasons.extend(reasons)

        ids, mask, row_map = self._build_comparison(composition)
        comparison_sequences, comparison_scores = rerun_fn(ids, mask)
        if comparison_sequences.ndim != 2 or comparison_sequences.shape[0] != self.reference_sequences.shape[0]:
            report.status = STATUS_ERROR
            report.error = (
                "the comparison run returned an incompatible shape "
                f"{tuple(comparison_sequences.shape)} for {tuple(self.reference_sequences.shape)}"
            )
            return report
        expected_len = self.num_generated + ids.shape[1]
        if comparison_sequences.shape[1] < expected_len:
            report.status = STATUS_ERROR
            report.error = (
                "the comparison run returned fewer tokens than generated "
                f"({comparison_sequences.shape[1]} < {expected_len})"
            )
            return report
        report.differences = self._diff(comparison_sequences, comparison_scores, row_map, report)
        if reasons:
            # Diagnostics only: a rejected configuration can never be certified as passing.
            report.status = STATUS_UNCOVERED
        else:
            report.status = STATUS_FAIL if any(d.layer in _FAILING_LAYERS for d in report.differences) else STATUS_PASS
        return report

    def run_single_row_oracle(self, rerun_single_fn: Callable, force: bool = False) -> BatchInvarianceReport:
        """Compare every row against the same row generated alone.

        The strongest oracle and the slowest: it runs `batch_size` extra generations. It exists for tests and
        offline audits, not for the hot path. `force=True` yields localized diagnostics for a rejected
        configuration while keeping the report `uncovered`.
        """
        report = self._base_report(["single"])
        reasons = self.coverage()
        if reasons:
            if not force:
                report.status = STATUS_UNCOVERED
                report.uncovered_reasons.extend(reasons)
                return report
            report.uncovered_reasons.extend(reasons)

        differences: list[RowDifference] = []
        for row in range(self.input_ids.shape[0]):
            if self.attention_mask is not None and self.attention_mask.shape == self.input_ids.shape:
                real = self.input_ids[row][self.attention_mask[row].bool()]
            else:
                real = self.input_ids[row]
            single_sequences, single_scores = rerun_single_fn(real[None])
            if single_sequences.shape[0] != self.num_return_sequences:
                report.status = STATUS_ERROR
                report.error = (
                    f"the single-row oracle returned {single_sequences.shape[0]} rows, "
                    f"expected {self.num_return_sequences}"
                )
                return report
            # The single-row run has no padding, so its prompt width is the row's real length. The batched
            # reference may have stopped earlier (EOS) than the oracle or vice versa; `compare_rows` truncates at
            # the first EOS, so slicing each side with its own prompt width is correct.
            single_generated = single_sequences[:, real.shape[0] :]
            for sequence in range(self.num_return_sequences):
                diff = self._row_difference(
                    row,
                    sequence,
                    self._reference_row(row, sequence),
                    single_generated[sequence],
                    comparison_scores=self._scores_row(single_scores, sequence),
                    report=report,
                )
                if diff is not None:
                    differences.append(diff)
        report.differences = differences
        if reasons:
            # Diagnostics only: a rejected configuration can never be certified as passing.
            report.status = STATUS_UNCOVERED
        else:
            report.status = STATUS_FAIL if any(d.layer in _FAILING_LAYERS for d in differences) else STATUS_PASS
        return report

    # -- internals ------------------------------------------------------------------------------------------

    def _reference_row(self, row: int, sequence: int) -> torch.Tensor:
        flat = row * self.num_return_sequences + sequence
        start = self.reference_sequences.shape[1] - self.num_generated
        return self.reference_sequences[flat, start:]

    def _scores_row(self, scores, index: int) -> torch.Tensor | None:
        # Per-step scores have shape (batch * num_beams, vocab); only num_beams == 1 has a stable per-row index.
        if scores is None or len(scores) == 0 or self.num_beams != 1:
            return None
        if index >= scores[0].shape[0]:
            return None
        return torch.stack([step[index] for step in scores])

    def _base_report(self, compositions: list[str]) -> BatchInvarianceReport:
        report = BatchInvarianceReport(
            mode=self.mode,
            compositions=list(compositions),
            batch_size=int(self.input_ids.shape[0]),
            padding_side=self.padding_side,
            metadata={
                "do_sample": bool(getattr(self.generation_config, "do_sample", False)),
                "num_beams": self.num_beams,
                "num_return_sequences": self.num_return_sequences,
                "cache_implementation": getattr(self.generation_config, "cache_implementation", None),
                "batch_invariant_sampling": bool(getattr(self.generation_config, "batch_invariant_sampling", False)),
                "tolerance": self.tolerance,
                "num_generated": self.num_generated,
            },
        )
        eos_ids = eos_ids_of(self.generation_config)
        for row in range(self.input_ids.shape[0]):
            for sequence in range(self.num_return_sequences):
                report.stop_info.append(
                    stop_info_for_row(
                        self._reference_row(row, sequence), eos_ids, self.num_generated, row=row, sequence=sequence
                    )
                )
        return report

    def _build_comparison(self, composition: str) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Return `(input_ids, attention_mask, row_map)` for the comparison run.

        `row_map[k]` is the reference row index that comparison row `k` must be compared against.
        """
        batch = self.input_ids.shape[0]
        identity = torch.arange(batch)
        if composition == "permute" and batch > 1:
            order = torch.arange(batch - 1, -1, -1)
            ids = self.input_ids[order]
            mask = self.attention_mask[order] if self.attention_mask is not None else None
            return ids, mask, order
        if composition == "extend_padding":
            side = self.padding_side if self.padding_side in (LEFT, RIGHT) else LEFT
            ids, mask = add_padding_column(
                self.input_ids,
                self.attention_mask if self.attention_mask is not None else torch.ones_like(self.input_ids),
                side=side,
                pad_token_id=self._pad_token_id(),
            )
            return ids, mask, identity
        if composition == "flip_padding" and self.attention_mask is not None:
            target = LEFT if self.padding_side == RIGHT else RIGHT
            ids, mask = flip_padding(self.input_ids, self.attention_mask, to=target, pad_token_id=self._pad_token_id())
            return ids, mask, identity
        return self.input_ids, self.attention_mask, identity

    def _pad_token_id(self) -> int:
        pad = getattr(self.generation_config, "_pad_token_tensor", None)
        if pad is not None:
            return int(pad.flatten()[0])
        if getattr(self.generation_config, "pad_token_id", None) is not None:
            return int(self.generation_config.pad_token_id)
        return 0

    def _prompt_fingerprint(self, row: int) -> str:
        if self.attention_mask is not None and self.attention_mask.shape == self.input_ids.shape:
            return batch_fingerprint(self.input_ids[row][self.attention_mask[row].bool()])
        return batch_fingerprint(self.input_ids[row])

    def _diff(
        self,
        comparison_sequences: torch.Tensor,
        comparison_scores,
        row_map: torch.Tensor,
        report: BatchInvarianceReport,
    ) -> list[RowDifference]:
        differences = []
        # The comparison prompt may be wider (e.g. `extend_padding`), so slice by its own width.
        start = comparison_sequences.shape[1] - self.num_generated
        cache_is_batch_shaped = getattr(self.generation_config, "cache_implementation", None) in {
            "static",
            "offloaded_static",
            "quantized",
        }
        for position in range(self.input_ids.shape[0]):
            reference_row = int(row_map[position])
            for sequence in range(self.num_return_sequences):
                flat = position * self.num_return_sequences + sequence
                if flat >= comparison_sequences.shape[0]:
                    continue
                diff = self._row_difference(
                    reference_row,
                    sequence,
                    self._reference_row(reference_row, sequence),
                    comparison_sequences[flat, start:],
                    comparison_scores=self._scores_row(comparison_scores, flat),
                    report=report,
                    cache_is_batch_shaped=cache_is_batch_shaped,
                )
                if diff is not None:
                    differences.append(diff)
        return differences

    def _row_difference(
        self,
        row: int,
        sequence: int,
        reference_row: torch.Tensor,
        comparison_row: torch.Tensor,
        report: BatchInvarianceReport,
        comparison_scores: torch.Tensor | None = None,
        cache_is_batch_shaped: bool = False,
    ) -> RowDifference | None:
        layer, step, observed_digest, expected_digest, detail = compare_rows(
            reference_row,
            comparison_row,
            tolerance=self.tolerance,
            eos_ids=eos_ids_of(self.generation_config),
            comparison_scores=comparison_scores,
            cache_is_batch_shaped=cache_is_batch_shaped,
        )
        if layer == DIFF_NUMERIC:
            return None
        if layer == DIFF_TOKENS_TOLERATED:
            report.metadata["tolerated_differences"] = report.metadata.get("tolerated_differences", 0) + 1
        layer = refine_layer(
            layer, step, self.padding_side, getattr(self.generation_config, "cache_implementation", None)
        )
        return RowDifference(
            row=row,
            sequence=sequence,
            fingerprint=self._prompt_fingerprint(row),
            layer=layer,
            step=step,
            observed_digest=observed_digest,
            expected_digest=expected_digest,
            detail=detail,
        )


# -- explicit entry point ------------------------------------------------------------------------------------


def split_output(output) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return `(sequences, scores)` from any `generate` output type."""
    if isinstance(output, dict):
        return output["sequences"], output.get("scores")
    if hasattr(output, "sequences"):
        return output.sequences, getattr(output, "scores", None)
    return output, None


def clone_generation_config(base_config, **overrides):
    """Deep-copy a generation config (or build a default) and apply overrides."""
    from .configuration_utils import GenerationConfig

    if base_config is None:
        config = GenerationConfig()
    else:
        config = type(base_config).from_dict(base_config.to_dict())
    config.update(**overrides)
    return config


def validate_batch_invariance(
    model,
    inputs: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    compositions: tuple[str, ...] = ("single",),
    tolerance: float = 1e-4,
    raise_on_failure: bool = False,
    **generate_kwargs,
) -> tuple[torch.Tensor, BatchInvarianceReport]:
    """Generate a batch and check that each row is independent of its batch composition.

    This is the explicit offline/audit entry point: it never changes `generate`'s behavior, and it returns the
    normal output alongside the report. A checker failure yields `status="error"` and the untouched output rather
    than raising, unless `raise_on_failure=True` and the generation itself was inconsistent.

    Args:
        model (`PreTrainedModel`): A model with a `generate` method.
        inputs (`torch.Tensor`): A 2D `(batch, sequence)` tensor of input ids.
        attention_mask (`torch.Tensor`, *optional*): The attention mask for `inputs`.
        compositions (`tuple[str, ...]`): Which comparisons to run. `"single"` regenerates every row alone (the
            strongest oracle, and the slowest). `"permute"` re-runs the same batch with rows reversed (one extra
            forward). `"extend_padding"` adds one fully-masked padding column (one extra forward). `"none"` only
            reports the layout.
        tolerance (`float`): Top-2 logit margin under which a token flip is tolerated as a near-tie.
        raise_on_failure (`bool`): Whether to raise when the report is a failure.
        **generate_kwargs: Forwarded verbatim to `generate` for the reference call.

    Returns:
        `tuple[torch.Tensor, BatchInvarianceReport]`: the reference output (exactly what `generate` returned) and
        the report.
    """
    if inputs.ndim != 2:
        raise ValueError("`inputs` must be a 2D tensor of input ids")
    generate_kwargs = dict(generate_kwargs)
    if attention_mask is None:
        attention_mask = generate_kwargs.pop("attention_mask", None)
    else:
        generate_kwargs.pop("attention_mask", None)

    # The reference call is exactly the user's call; nothing about it is altered.
    reference = model.generate(inputs, attention_mask=attention_mask, **generate_kwargs)
    reference_sequences, _reference_scores = split_output(reference)

    comparison_config = clone_generation_config(
        generate_kwargs.get("generation_config") or getattr(model, "generation_config", None),
        validate_batch_invariance=False,
    )
    config_keys = set(comparison_config.to_dict())
    comparison_config.update(
        **{
            key: value
            for key, value in generate_kwargs.items()
            if key in config_keys and key not in {"output_scores", "return_dict_in_generate"}
        }
    )
    rerun_kwargs = {
        key: value
        for key, value in generate_kwargs.items()
        if key not in {"streamer", "output_scores", "return_dict_in_generate", "generation_config"}
    }

    def _rerun(ids: torch.Tensor, mask: torch.Tensor | None):
        out = model.generate(
            ids,
            attention_mask=mask,
            generation_config=comparison_config,
            output_scores=compositions != ("none",),
            return_dict_in_generate=True,
            **rerun_kwargs,
        )
        return split_output(out)

    def _rerun_single(prompt: torch.Tensor):
        return _rerun(prompt, None)

    checker = BatchInvarianceChecker(
        comparison_config,
        inputs,
        reference_sequences,
        attention_mask=attention_mask,
        reference_scores=None,
        tolerance=tolerance,
        is_encoder_decoder=bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False)),
        custom_components=tuple(
            name
            for name in ("logits_processor", "logits_warper", "stopping_criteria")
            if generate_kwargs.get(name) is not None
        ),
    )
    report = checker._base_report(list(compositions))
    try:
        merged: list[RowDifference] = []
        for composition in compositions:
            if composition == "none":
                continue
            if composition == "single":
                partial = checker.run_single_row_oracle(_rerun_single)
            elif composition in {"permute", "extend_padding", "flip_padding"}:
                partial = checker.run(_rerun, composition=composition)
            else:
                raise ValueError(f"unknown composition {composition!r}")
            if partial.status == STATUS_UNCOVERED:
                report.status = STATUS_UNCOVERED
                report.uncovered_reasons = partial.uncovered_reasons
                return reference, report
            if partial.status == STATUS_ERROR:
                report.status = STATUS_ERROR
                report.error = partial.error
                return reference, report
            merged.extend(partial.differences)
        report.differences = merged
        report.status = STATUS_FAIL if any(d.layer in _FAILING_LAYERS for d in merged) else STATUS_PASS
    except Exception as exc:  # noqa: BLE001 - the checker must never break generation
        report.status = STATUS_ERROR
        report.error = f"{type(exc).__name__}: {exc}"
        logger.warning("batch-invariance check failed to run; generation output is unaffected (%s)", report.error)

    if raise_on_failure:
        report.raise_on_failure()
    return reference, report
