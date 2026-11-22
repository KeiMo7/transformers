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

"""Utilities for merging two tokenizers (vocabularies) into one and remapping model weights onto the merged ids.

The merger never concatenates id spaces: the merged vocabulary is the *union* of the two token strings, every token
gets a single contiguous id, and every embedding row is copied to the id of the token it belongs to. Tokens shared by
both inputs keep one id (the id assigned by the base tokenizer), tokens only present in one side are appended, and the
old-id -> new-id maps are persisted so that older checkpoints stay interpretable.

The public entry points are:

- [`~tokenizer_merging.merge_tokenizers`] / [`~tokenizer_merging.build_merged_tokenizer`] — build the merged
  tokenizer(s) and a [`~tokenizer_merging.TokenizerMergePlan`] describing the id mapping;
- [`~tokenizer_merging.remap_embeddings`] / [`~tokenizer_merging.merge_model_embeddings`] — move embedding (and, when
  present, output/lm-head) rows onto the merged ids;
- [`~tokenizer_merging.save_vocab_merging_mapping`] / [`~tokenizer_merging.load_vocab_merging_mapping`] — persist the
  old-id -> new-id mapping next to the merged tokenizer.

Only the `tokenizers` library (a required dependency of this package) is needed for the tokenizer part; torch is only
imported when model weights are remapped.
"""

import copy
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from .tokenization_utils_base import AddedToken, PreTrainedTokenizerBase
from .utils import is_torch_available, logging


if TYPE_CHECKING:
    import torch

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)

VOCAB_MERGING_FILE_NAME = "vocab_merging_map.json"


class VocabMergingError(ValueError):
    """Raised when two tokenizers cannot be merged (incompatible backend model, conflicting special tokens, ...)."""


class SpecialTokenConflictError(VocabMergingError):
    """Raised when a token string is a special token on both sides with incompatible special-token attributes."""


class EmbeddingShapeError(VocabMergingError):
    """Raised when an embedding matrix does not have one row per id of the tokenizer it is remapped from."""


@dataclass
class TokenRecord:
    """One entry of the merged vocabulary.

    Args:
        token (`str`):
            The token string as seen by the tokenizer backend (the key of `get_vocab()`).
        new_id (`int`):
            The unique, contiguous id assigned to this token in the merged vocabulary.
        source (`str`):
            Where the token comes from: `"both"`, `"first"` or `"second"`.
        first_id (`int | None`):
            The id of this token in the first tokenizer, or `None` if it was absent.
        second_id (`int | None`):
            The id of this token in the second tokenizer, or `None` if it was absent.
        added_token (`AddedToken | None`):
            The added/special token metadata attached to this token, or `None` for plain model vocabulary entries.
        reachable (`bool`):
            Whether the token can ever be produced by the merged backend algorithm. Unreachable entries are kept (and
            still get an id and an embedding row) but listed in `TokenizerMergePlan.unreachable_tokens`.
    """

    token: str
    new_id: int
    source: str
    first_id: int | None
    second_id: int | None
    added_token: AddedToken | None
    reachable: bool = True


@dataclass
class TokenizerMergePlan:
    """The deterministic result of planning the union of two vocabularies.

    A plan is independent of any model: it only describes token strings, old/new ids and the data needed to rebuild a
    tokenizer backend. The same pair of inputs and the same strategy always produces an identical plan.

    Args:
        strategy (`str`):
            The base selection strategy that was used (see [`~tokenizer_merging.merge_tokenizers`]).
        base_index (`int`):
            Which input is the base (`0` for the first tokenizer, `1` for the second). Shared tokens keep the base's
            ids.
        backend_type (`str`):
            The common tokenizers backend model type, e.g. `"WordPiece"` or `"BPE"`.
        records (`list[TokenRecord]`):
            Every merged token, ordered by `new_id`. `new_id` values are exactly `range(len(records))`: no duplicates,
            no holes.
        first_id_to_new (`dict[int, int]`):
            Mapping from every id of the first tokenizer to its merged id.
        second_id_to_new (`dict[int, int]`):
            Mapping from every id of the second tokenizer to its merged id.
        merges (`list[tuple[str, str]]`):
            The ordered union of BPE merge rules (empty for non-BPE backends).
        unreachable_tokens (`list[str]`):
            Tokens that can never be produced by the merged backend algorithm, sorted by merged id.
        backend_kwargs (`dict[str, Any]`):
            Keyword arguments needed to reconstruct the tokenizers backend model (e.g. `unk_token`,
            `continuing_subword_prefix`).
    """

    strategy: str
    base_index: int
    backend_type: str
    records: list[TokenRecord]
    first_id_to_new: dict[int, int]
    second_id_to_new: dict[int, int]
    merges: list[tuple[str, str]] = field(default_factory=list)
    unreachable_tokens: list[str] = field(default_factory=list)
    backend_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def vocab_size(self) -> int:
        """Number of token ids in the merged vocabulary."""
        return len(self.records)

    @property
    def merged_vocab(self) -> dict[str, int]:
        """The merged vocabulary as a `{token: id}` dictionary."""
        return {record.token: record.new_id for record in self.records}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a plain JSON-friendly dictionary."""
        return {
            "version": 1,
            "strategy": self.strategy,
            "base_index": self.base_index,
            "backend_type": self.backend_type,
            "merged_vocab_size": self.vocab_size,
            "first_id_to_new": self.first_id_to_new,
            "second_id_to_new": self.second_id_to_new,
            "merges": [list(pair) for pair in self.merges],
            "unreachable_tokens": list(self.unreachable_tokens),
            "tokens": [
                {
                    "token": record.token,
                    "new_id": record.new_id,
                    "source": record.source,
                    "first_id": record.first_id,
                    "second_id": record.second_id,
                    "is_added": record.added_token is not None,
                    "reachable": record.reachable,
                }
                for record in self.records
            ],
            "backend_kwargs": _jsonable(self.backend_kwargs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenizerMergePlan":
        """Rebuild a plan previously written by [`~tokenizer_merging.TokenizerMergePlan.to_dict`]."""
        records = [
            TokenRecord(
                token=item["token"],
                new_id=item["new_id"],
                source=item["source"],
                first_id=item["first_id"],
                second_id=item["second_id"],
                added_token=None,
                reachable=item.get("reachable", True),
            )
            for item in data["tokens"]
        ]
        return cls(
            strategy=data["strategy"],
            base_index=data["base_index"],
            backend_type=data["backend_type"],
            records=records,
            first_id_to_new={int(k): int(v) for k, v in data["first_id_to_new"].items()},
            second_id_to_new={int(k): int(v) for k, v in data["second_id_to_new"].items()},
            merges=[tuple(pair) for pair in data.get("merges", [])],
            unreachable_tokens=list(data.get("unreachable_tokens", [])),
            backend_kwargs=dict(data.get("backend_kwargs", {})),
        )


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of backend metadata into JSON-friendly primitives."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    return str(obj)


# ---------------------------------------------------------------------------------------------------
# Tokenizer introspection
# ---------------------------------------------------------------------------------------------------


def _backend_type(tokenizer: PreTrainedTokenizerBase) -> str:
    return type(tokenizer.backend_tokenizer.model).__name__


def _vocab_by_id(tokenizer: PreTrainedTokenizerBase) -> list[tuple[str, int]]:
    """Return the whole vocabulary as `(token, id)` pairs enumerated in ascending id order.

    `get_vocab()` is backed by a dictionary built on the Rust side whose iteration order is not stable across
    processes, so every ordering decision in this module goes through this helper to keep merges reproducible.

    Raises:
        VocabMergingError: If the backend id space contains a hole (an id without a token), which would break the
            contiguous-id guarantee of the merged vocabulary.
    """
    backend = tokenizer.backend_tokenizer
    pairs: list[tuple[str, int]] = []
    for index in range(backend.get_vocab_size()):
        token = backend.id_to_token(index)
        if token is None:
            raise VocabMergingError(
                f"Tokenizer of type {type(tokenizer).__name__} has a hole at id {index}: every id must name a token. "
                "Rebuild the tokenizer with a contiguous vocabulary before merging."
            )
        pairs.append((token, index))
    return pairs


def _added_vocab(tokenizer: PreTrainedTokenizerBase) -> dict[str, AddedToken]:
    """The added-token part of the vocabulary (special and non-special added tokens)."""
    return {str(token): token for token in tokenizer.added_tokens_decoder.values()}


def _bpe_merges(tokenizer: PreTrainedTokenizerBase) -> list[tuple[str, str]]:
    """Extract the ordered BPE merge list from a saved backend, working across tokenizers versions."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tokenizer.json")
        tokenizer.backend_tokenizer.save(path)
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    raw_merges = data["model"].get("merges", [])
    return [tuple(pair) if isinstance(pair, list) else tuple(pair.split(" ", 1)) for pair in raw_merges]


def _backend_kwargs(backend_type: str, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    model = tokenizer.backend_tokenizer.model
    if backend_type == "WordPiece":
        return {
            "unk_token": model.unk_token,
            "continuing_subword_prefix": model.continuing_subword_prefix,
            "max_input_chars_per_word": model.max_input_chars_per_word,
        }
    if backend_type == "BPE":
        return {
            "unk_token": model.unk_token,
            "continuing_subword_prefix": model.continuing_subword_prefix,
            "end_of_word_suffix": model.end_of_word_suffix,
            "fuse_unk": model.fuse_unk,
            "byte_fallback": model.byte_fallback,
        }
    if backend_type == "Unigram":
        return {"unk_id": getattr(model, "unk_id", None)}
    return {}


# ---------------------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------------------


def _added_tokens_compatible(first: AddedToken, second: AddedToken) -> bool:
    """Two added-token declarations for the same string are compatible when their semantics match exactly."""
    return (
        first.special == second.special
        and first.normalized == second.normalized
        and first.single_word == second.single_word
        and first.lstrip == second.lstrip
        and first.rstrip == second.rstrip
    )


def _check_special_conflicts(
    first_added: dict[str, AddedToken], second_added: dict[str, AddedToken]
) -> dict[str, AddedToken]:
    """Validate the added-token parts of both vocabularies and return their union keyed by token string.

    A shared special token with different attributes (e.g. `normalized=True` on one side and `False` on the other, or a
    token marked special on one side only) is a semantic conflict rather than something the merger may silently pick a
    side for.
    """
    conflicts: list[str] = []
    merged: dict[str, AddedToken] = {}
    for token, first_decl in first_added.items():
        second_decl = second_added.get(token)
        if second_decl is None:
            merged[token] = first_decl
            continue
        if not _added_tokens_compatible(first_decl, second_decl):
            conflicts.append(token)
            continue
        merged[token] = first_decl
    for token, decl in second_added.items():
        if token not in merged and token not in conflicts:
            merged[token] = decl
    if conflicts:
        details = "; ".join(
            f"{token!r}: {repr(first_added[token])} vs {repr(second_added[token])}" for token in sorted(conflicts)
        )
        raise SpecialTokenConflictError(
            f"Found {len(conflicts)} conflicting special/added token declaration(s) for the same token string(s). "
            "Unify the AddedToken attributes (special, normalized, single_word, lstrip, rstrip) on both tokenizers "
            f"before merging. Conflicts: {details}"
        )
    return merged


def _select_base(
    first: PreTrainedTokenizerBase,
    second: PreTrainedTokenizerBase,
    strategy: str,
) -> int:
    first_size = len(first.get_vocab())
    second_size = len(second.get_vocab())
    if strategy == "first":
        return 0
    if strategy == "second":
        return 1
    if strategy == "larger":
        # Tie-breaking on equal sizes makes the choice deterministic: keep the first tokenizer.
        return 0 if first_size >= second_size else 1
    if strategy == "smaller":
        return 1 if second_size < first_size else 0
    raise ValueError(f"Unknown base strategy {strategy!r}. Expected one of 'first', 'second', 'larger', 'smaller'.")


def _plan_bpe_reachability(
    vocab: dict[str, int], merges: list[tuple[str, str]]
) -> tuple[set[str], list[tuple[str, str]]]:
    """Return the set of BPE tokens reachable from the initial symbols and the merges that remain usable.

    A BPE merge `(left, right)` consumes the two operand tokens and emits their concatenation `left + right`, so the
    initial symbols are the vocabulary tokens that never equal any merge's concatenated output. A token is reachable
    when it is initial or produced by a merge whose operands are themselves reachable. Merges whose output or operands
    are missing from the union vocabulary, or whose operand chain is broken, are dropped; their outputs then stay
    unreachable (they keep an id and an embedding row but can never be decoded from text).
    """
    merge_outputs = {left + right for left, right in merges}
    # Seed reachability with the initial symbols only: tokens that no merge can produce. A merge output whose
    # producing merge is absent (or dropped below) therefore stays out of this set forever.
    reachable = {token for token in vocab if token not in merge_outputs}
    kept_merges: list[tuple[str, str]] = []
    for left, right in merges:
        output = left + right
        if left not in vocab or right not in vocab or output not in vocab:
            continue
        if left not in reachable or right not in reachable:
            continue
        reachable.add(output)
        kept_merges.append((left, right))
    return reachable, kept_merges


def _plan_wordpiece_reachability(
    vocab: dict[str, int], continuing_prefix: str | None, added_tokens: set[str]
) -> set[str]:
    """Return the WordPiece tokens reachable at some position of a word.

    Added/special tokens bypass the WordPiece algorithm entirely, so they do not count as word-starting subwords. A
    model vocabulary made only of continuing subwords (e.g. `##x`) can never emit anything: no plain token may start a
    word, hence no continuing piece may follow. When at least one non-continuing model entry exists it may start a
    word and every continuing piece becomes reachable behind it.
    """
    prefix = continuing_prefix if continuing_prefix is not None else "##"
    model_tokens = {token for token in vocab if token not in added_tokens}
    starts = {token for token in model_tokens if not token.startswith(prefix)}
    if not starts:
        # Added tokens stay reachable through the added-token matching path; model subwords do not.
        return set(added_tokens)
    return set(vocab)


def plan_vocab_merging(
    first: PreTrainedTokenizerBase,
    second: PreTrainedTokenizerBase,
    strategy: str = "larger",
) -> TokenizerMergePlan:
    """Plan the union of two tokenizer vocabularies without constructing anything.

    This is the deterministic core of the merger: given the same two tokenizers and the same `strategy`, the returned
    [`~tokenizer_merging.TokenizerMergePlan`] is byte-for-byte identical across runs.

    Args:
        first (`PreTrainedTokenizerBase`):
            The first tokenizer.
        second (`PreTrainedTokenizerBase`):
            The second tokenizer. It must use the same backend model type as `first` (e.g. two WordPiece or two BPE
            tokenizers).
        strategy (`str`, *optional*, defaults to `"larger"`):
            Which tokenizer provides the ids of shared tokens — `"first"`, `"second"`, `"larger"` (the larger
            vocabulary; ties keep the first) or `"smaller"`.

    Returns:
        `TokenizerMergePlan`: The merge plan with contiguous merged ids and the old-id -> new-id maps.

    Raises:
        VocabMergingError: If the backend model types differ or are not supported.
        SpecialTokenConflictError: If a shared added/special token has incompatible declarations on both sides.
    """
    first_backend = _backend_type(first)
    second_backend = _backend_type(second)
    if first_backend != second_backend:
        raise VocabMergingError(
            f"Cannot merge tokenizers with different backend model types: {first_backend!r} vs {second_backend!r}."
        )
    backend_type = first_backend
    if backend_type not in ("WordPiece", "BPE", "Unigram", "WordLevel"):
        raise VocabMergingError(
            f"Vocabulary merging is implemented for WordPiece, BPE, Unigram and WordLevel backends, got {backend_type!r}."
        )

    base_index = _select_base(first, second, strategy)
    base = first if base_index == 0 else second
    other = second if base_index == 0 else first

    first_added = _added_vocab(first)
    second_added = _added_vocab(other)
    merged_added = _check_special_conflicts(first_added, second_added)

    # 1. Base vocabulary keeps its ids and its exact id ordering.
    ordered_tokens: list[str] = [token for token, _ in _vocab_by_id(base)]
    token_to_new: dict[str, int] = {token: index for index, token in enumerate(ordered_tokens)}

    # 2. Tokens only known to the other side are appended in the other side's own id order.
    for token, _ in _vocab_by_id(other):
        if token not in token_to_new:
            token_to_new[token] = len(ordered_tokens)
            ordered_tokens.append(token)

    first_full_vocab = first.get_vocab()
    second_full_vocab = second.get_vocab()

    records: list[TokenRecord] = []
    for token in ordered_tokens:
        new_id = token_to_new[token]
        in_first = token in first_full_vocab
        in_second = token in second_full_vocab
        source = "both" if in_first and in_second else ("first" if in_first else "second")
        records.append(
            TokenRecord(
                token=token,
                new_id=new_id,
                source=source,
                first_id=first_full_vocab.get(token),
                second_id=second_full_vocab.get(token),
                added_token=merged_added.get(token),
            )
        )

    # 3. Backend-specific data: BPE merges (ordered union, base first) and reachability analysis.
    merged_vocab = token_to_new
    merges: list[tuple[str, str]] = []
    reachable: set[str] = set(merged_vocab)
    if backend_type == "BPE":
        base_merges = _bpe_merges(base)
        other_merges = _bpe_merges(other)
        seen: set[tuple[str, str]] = set()
        for pair in [*base_merges, *other_merges]:
            if pair not in seen:
                seen.add(pair)
                merges.append(pair)
        reachable, merges = _plan_bpe_reachability(merged_vocab, merges)
    elif backend_type == "WordPiece":
        prefix = base.backend_tokenizer.model.continuing_subword_prefix
        reachable = _plan_wordpiece_reachability(merged_vocab, prefix, set(merged_added))
    # Unigram and WordLevel: every entry can be emitted directly, nothing is structurally unreachable.

    unreachable: list[str] = []
    for record in records:
        if record.token not in reachable:
            record.reachable = False
            unreachable.append(record.token)

    first_id_to_new = {old_id: token_to_new[token] for token, old_id in first_full_vocab.items()}
    second_id_to_new = {old_id: token_to_new[token] for token, old_id in second_full_vocab.items()}

    plan = TokenizerMergePlan(
        strategy=strategy,
        base_index=base_index,
        backend_type=backend_type,
        records=records,
        first_id_to_new=first_id_to_new,
        second_id_to_new=second_id_to_new,
        merges=merges,
        unreachable_tokens=unreachable,
        backend_kwargs=_backend_kwargs(backend_type, base),
    )
    _validate_plan(plan, first_full_vocab, second_full_vocab)
    return plan


def _validate_plan(plan: TokenizerMergePlan, first_vocab: dict[str, int], second_vocab: dict[str, int]) -> None:
    """Invariants the rest of the library relies on: bijection, no holes, both old spaces fully covered."""
    new_ids = [record.new_id for record in plan.records]
    if sorted(new_ids) != list(range(plan.vocab_size)):
        raise VocabMergingError("Internal error: merged ids are not a contiguous 0..N-1 permutation.")
    if len({record.token for record in plan.records}) != plan.vocab_size:
        raise VocabMergingError("Internal error: duplicated token strings in the merged vocabulary.")
    if set(plan.first_id_to_new) != set(first_vocab.values()):
        raise VocabMergingError("Internal error: first tokenizer id space is not fully mapped.")
    if set(plan.second_id_to_new) != set(second_vocab.values()):
        raise VocabMergingError("Internal error: second tokenizer id space is not fully mapped.")
    if set(plan.first_id_to_new.values()) | set(plan.second_id_to_new.values()) != set(range(plan.vocab_size)):
        raise VocabMergingError("Internal error: some merged ids are unreachable from the old id maps.")


# ---------------------------------------------------------------------------------------------------
# Tokenizer construction
# ---------------------------------------------------------------------------------------------------


def _build_backend_model(plan: TokenizerMergePlan):
    """Construct a fresh tokenizers backend model from the plan's union vocabulary."""
    if plan.backend_type == "WordPiece":
        from tokenizers.models import WordPiece

        kwargs = {key: value for key, value in plan.backend_kwargs.items() if value is not None or key == "unk_token"}
        return WordPiece(vocab=dict(plan.merged_vocab), **kwargs)
    if plan.backend_type == "BPE":
        from tokenizers.models import BPE

        kwargs = {key: value for key, value in plan.backend_kwargs.items() if key != "unk_token" or value is not None}
        return BPE(vocab=dict(plan.merged_vocab), merges=[tuple(pair) for pair in plan.merges], **kwargs)
    if plan.backend_type == "Unigram":
        from tokenizers.models import Unigram

        unk_id = plan.backend_kwargs.get("unk_id")
        pieces = [(token, 0.0) for token in plan.merged_vocab]
        return Unigram(vocab=pieces, unk_id=unk_id)
    if plan.backend_type == "WordLevel":
        from tokenizers.models import WordLevel

        return WordLevel(vocab=dict(plan.merged_vocab), unk_token=plan.backend_kwargs.get("unk_token"))
    raise VocabMergingError(f"Unsupported backend type {plan.backend_type!r}.")


def _build_fast_tokenizer(plan: TokenizerMergePlan, base: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Build a tokenizers-backed tokenizer from the plan, reusing the base tokenizer's pipeline components."""
    from tokenizers import Tokenizer

    backend = Tokenizer(_build_backend_model(plan))
    source_backend = base.backend_tokenizer
    # Preserve every pipeline component except the model itself: normalizer / pre-tokenization / post-processing /
    # decoding behavior must stay identical to the base tokenizer's.
    backend.normalizer = copy.deepcopy(source_backend.normalizer)
    backend.pre_tokenizer = copy.deepcopy(source_backend.pre_tokenizer)
    backend.post_processor = copy.deepcopy(source_backend.post_processor)
    backend.decoder = copy.deepcopy(source_backend.decoder)

    merged = base.__class__(tokenizer_object=backend, **_reconstruction_kwargs(base))
    added = [record.added_token for record in plan.records if record.added_token is not None]
    if added:
        merged.add_tokens(copy.deepcopy(added))
    return merged


def _reconstruction_kwargs(tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    """Keyword arguments needed to rebuild a tokenizer of the same class and backend as `tokenizer`."""
    kwargs: dict[str, Any] = {}
    for key in ("do_lower_case", "strip_accents", "add_prefix_space", "errors", "model_max_length"):
        if hasattr(tokenizer, key):
            kwargs[key] = getattr(tokenizer, key)
    # Preserve the requested backend ("tokenizers" or "python") instead of silently switching a python-backed
    # tokenizer to the Rust implementation.
    if getattr(tokenizer, "backend", None) is not None:
        kwargs["backend"] = tokenizer.backend
    return kwargs


def build_merged_tokenizer(
    first: PreTrainedTokenizerBase,
    second: PreTrainedTokenizerBase,
    strategy: str = "larger",
    plan: TokenizerMergePlan | None = None,
) -> tuple[PreTrainedTokenizerBase, TokenizerMergePlan]:
    """Construct the merged tokenizer from two tokenizers.

    The merged tokenizer is an instance of the same class as the base tokenizer (see the `strategy` argument): its
    normalizer, pre-tokenizer, post-processor and decoder are preserved, while the backend model is rebuilt from the
    union vocabulary. Tokens shared by both input tokenizers keep the base tokenizer's ids.

    Args:
        first (`PreTrainedTokenizerBase`):
            The first tokenizer.
        second (`PreTrainedTokenizerBase`):
            The second tokenizer.
        strategy (`str`, *optional*, defaults to `"larger"`):
            Base selection strategy, see [`~tokenizer_merging.plan_vocab_merging`].
        plan (`TokenizerMergePlan`, *optional*):
            A plan previously computed with [`~tokenizer_merging.plan_vocab_merging`]. If omitted, a fresh one is
            planned from the two tokenizers.

    Returns:
        `tuple[PreTrainedTokenizerBase, TokenizerMergePlan]`: The merged tokenizer and the plan that was used.
    """
    if plan is None:
        plan = plan_vocab_merging(first, second, strategy)
    base = first if plan.base_index == 0 else second
    merged = _build_fast_tokenizer(plan, base)
    if merged.get_vocab() != plan.merged_vocab:
        raise VocabMergingError(
            "Internal error: the constructed tokenizer's id assignment does not match the merge plan."
        )
    if plan.unreachable_tokens:
        logger.warning(
            "%d token(s) in the merged vocabulary can never be produced by the tokenization algorithm and are kept "
            "only for id/embedding compatibility: %s",
            len(plan.unreachable_tokens),
            plan.unreachable_tokens,
        )
    return merged, plan


def merge_tokenizers(
    first: PreTrainedTokenizerBase,
    second: PreTrainedTokenizerBase,
    strategy: str = "larger",
) -> PreTrainedTokenizerBase:
    """Merge two tokenizers into one union vocabulary.

    Thin convenience wrapper around [`~tokenizer_merging.build_merged_tokenizer`] for callers that do not need the
    [`~tokenizer_merging.TokenizerMergePlan`].

    Args:
        first (`PreTrainedTokenizerBase`):
            The first tokenizer.
        second (`PreTrainedTokenizerBase`):
            The second tokenizer.
        strategy (`str`, *optional*, defaults to `"larger"`):
            Base selection strategy, see [`~tokenizer_merging.plan_vocab_merging`].

    Returns:
        `PreTrainedTokenizerBase`: The merged tokenizer with a single contiguous union vocabulary.
    """
    merged, _ = build_merged_tokenizer(first, second, strategy=strategy)
    return merged


# ---------------------------------------------------------------------------------------------------
# Weight remapping
# ---------------------------------------------------------------------------------------------------


def _check_matrix_rows(weight: "torch.Tensor", expected_rows: int, label: str) -> None:
    if weight.ndim != 2:
        raise EmbeddingShapeError(
            f"{label} must be a 2D matrix of shape (vocab_size, hidden_size), got shape {tuple(weight.shape)}."
        )
    actual_rows = weight.shape[0]
    if actual_rows != expected_rows:
        raise EmbeddingShapeError(
            f"{label} has {actual_rows} rows but the source tokenizer has {expected_rows} ids. Remapping would copy "
            "rows onto the wrong tokens; refusing to proceed. Check that the checkpoint matches the tokenizer."
        )


def remap_weight_matrix(
    weight: "torch.Tensor",
    id_to_new: dict[int, int],
    merged_vocab_size: int,
    new_row_init: str | Callable[..., "torch.Tensor"] = "mean",
    generator=None,
) -> "torch.Tensor":
    """Remap one embedding/lm-head matrix from old ids onto the merged ids.

    Existing rows are copied with `index_copy`-equivalent indexing, so shared tokens keep bit-identical rows. Rows for
    tokens absent from this side are freshly initialized; existing rows are never modified in place (the result is a
    new tensor).

    Args:
        weight (`torch.Tensor`):
            A 2D matrix with one row per id of the source tokenizer.
        id_to_new (`dict[int, int]`):
            Old-id -> merged-id map covering every row of `weight`.
        merged_vocab_size (`int`):
            Number of rows of the returned matrix.
        new_row_init (`str | Callable`, *optional*, defaults to `"mean"`):
            How to initialize rows this side does not own: `"mean"` (mean of the existing rows), `"zeros"`,
            `"normal"` (Gaussian matching the existing rows' mean/std) or a callable receiving the source matrix.
        generator:
            Optional torch generator used by the `"normal"` initializer for reproducibility.

    Returns:
        `torch.Tensor`: A new matrix of shape `(merged_vocab_size, hidden_size)`.

    Raises:
        EmbeddingShapeError: If `weight` does not have exactly one row per source id.
    """
    expected_rows = len(id_to_new)
    _check_matrix_rows(weight, expected_rows, "Weight matrix")
    if merged_vocab_size < expected_rows:
        raise EmbeddingShapeError(
            f"Merged vocabulary size ({merged_vocab_size}) is smaller than the source id space ({expected_rows})."
        )
    hidden_size = weight.shape[1]
    new_weight = torch.empty((merged_vocab_size, hidden_size), dtype=weight.dtype, device=weight.device)
    old_ids = torch.tensor(list(id_to_new.keys()), dtype=torch.long, device=weight.device)
    new_ids = torch.tensor(list(id_to_new.values()), dtype=torch.long, device=weight.device)
    # Existing rows are gathered onto their merged ids; every destination is written exactly once because the plan's
    # ids form a bijection. The advanced-indexing assignment itself never modifies a source row in place.
    new_weight[new_ids] = weight[old_ids]
    occupied = torch.zeros(merged_vocab_size, dtype=torch.bool, device=weight.device)
    occupied[new_ids] = True
    free_ids = torch.nonzero(~occupied, as_tuple=False).squeeze(1)
    if free_ids.numel() > 0:
        new_weight[free_ids] = _initializer_rows(weight, free_ids.numel(), new_row_init, generator)
    return new_weight


def _initializer_rows(
    source_rows: "torch.Tensor", num_new_rows: int, init: str | Callable[..., "torch.Tensor"], generator
) -> "torch.Tensor":
    """Build the block of `num_new_rows` fresh rows for tokens this side does not own."""
    hidden_size = source_rows.shape[1]
    if callable(init):
        produced = cast("torch.Tensor", init(source_rows))
        if produced.shape != (num_new_rows, hidden_size):
            raise ValueError(
                "Callable new-row initializer must return a tensor of shape "
                f"(num_new_rows={num_new_rows}, hidden_size={hidden_size}), got {tuple(produced.shape)}."
            )
        return produced.to(dtype=source_rows.dtype, device=source_rows.device)
    if init == "zeros":
        return torch.zeros((num_new_rows, hidden_size), dtype=source_rows.dtype, device=source_rows.device)
    if init == "mean":
        # Match the convention of PreTrainedModel._get_resized_embeddings: every new row starts from the mean of the
        # existing rows, so existing rows stay bit-identical and new rows carry a deterministic, neutral vector.
        mean = source_rows.mean(dim=0).detach().clone()
        return mean.unsqueeze(0).expand(num_new_rows, -1).contiguous()
    if init == "normal":
        std = source_rows.std().item() if source_rows.shape[0] > 1 else 1.0
        return torch.empty((num_new_rows, hidden_size), dtype=source_rows.dtype, device=source_rows.device).normal_(
            mean=0.0, std=std, generator=generator
        )
    raise ValueError(
        f"Unknown new-row initializer {init!r}. Expected 'mean', 'zeros', 'normal' or a callable taking the source "
        "weight matrix and returning a tensor of shape (num_new_rows, hidden_size)."
    )


def remap_embeddings(
    model,
    plan: TokenizerMergePlan,
    source: str = "first",
    new_row_init: str | Callable[..., "torch.Tensor"] = "mean",
    generator=None,
):
    """Remap a model's input (and output, when present) embeddings onto the merged vocabulary ids.

    The model is updated in place and its `config.vocab_size` is bumped to the merged size, so it can keep training on
    the merged tokenizer as well as run inference. Tied input/output embeddings stay tied: the tied module is remapped
    once and re-tied through the model's own `tie_weights`.

    Args:
        model (`PreTrainedModel`):
            The model whose embeddings are remapped.
        plan (`TokenizerMergePlan`):
            The plan describing old/new ids.
        source (`str`, *optional*, defaults to `"first"`):
            Which side of the merge this checkpoint comes from: `"first"` or `"second"`.
        new_row_init (`str | Callable`, *optional*, defaults to `"mean"`):
            Initialization of rows owned by the other side, see
            [`~tokenizer_merging.remap_weight_matrix`].
        generator:
            Optional torch generator for the `"normal"` initializer.

    Returns:
        `PreTrainedModel`: The same model, updated in place.

    Raises:
        EmbeddingShapeError: If an embedding matrix's row count does not match the source tokenizer's id space.
    """
    if not is_torch_available():
        raise ImportError("Remapping model embeddings requires torch, which is not installed.")
    if source not in ("first", "second"):
        raise ValueError(f"source must be 'first' or 'second', got {source!r}.")
    id_to_new = plan.first_id_to_new if source == "first" else plan.second_id_to_new
    source_size = len(id_to_new)

    input_embeddings = model.get_input_embeddings()
    _check_matrix_rows(input_embeddings.weight, source_size, "Input embeddings")
    output_before = model.get_output_embeddings()
    is_tied = output_before is not None and output_before.weight is input_embeddings.weight

    remapped_input = remap_weight_matrix(input_embeddings.weight, id_to_new, plan.vocab_size, new_row_init, generator)
    input_embeddings.weight = torch.nn.Parameter(remapped_input, requires_grad=input_embeddings.weight.requires_grad)

    if output_before is not None and not is_tied:
        _check_matrix_rows(output_before.weight, source_size, "Output embeddings")
        remapped_output = remap_weight_matrix(
            output_before.weight, id_to_new, plan.vocab_size, new_row_init, generator
        )
        output_before.weight = torch.nn.Parameter(remapped_output, requires_grad=output_before.weight.requires_grad)

    text_config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    text_config.vocab_size = plan.vocab_size
    if hasattr(model, "vocab_size"):
        model.vocab_size = plan.vocab_size
    # Re-run tying so a tied lm head points at the freshly remapped input parameter.
    model.tie_weights()
    return model


def merge_model_embeddings(
    first_model,
    second_model,
    first_tokenizer: PreTrainedTokenizerBase,
    second_tokenizer: PreTrainedTokenizerBase,
    strategy: str = "larger",
    new_row_init: str | Callable[..., "torch.Tensor"] = "mean",
    plan: TokenizerMergePlan | None = None,
    generator=None,
) -> tuple[Any, Any, PreTrainedTokenizerBase, TokenizerMergePlan]:
    """End-to-end helper: plan the merge, build the merged tokenizer and remap both models.

    Each model is remapped against its own old-id map, so both end up sharing the merged id space. The two models are
    returned independently (their hidden sizes need not match); no cross-model weight averaging is performed.

    Returns:
        `tuple[Any, Any, PreTrainedTokenizerBase, TokenizerMergePlan]`: The remapped first model, the remapped second
        model, the merged tokenizer and the plan.
    """
    if plan is None:
        plan = plan_vocab_merging(first_tokenizer, second_tokenizer, strategy)
    merged_tokenizer, plan = build_merged_tokenizer(first_tokenizer, second_tokenizer, plan=plan)
    remap_embeddings(first_model, plan, source="first", new_row_init=new_row_init, generator=generator)
    remap_embeddings(second_model, plan, source="second", new_row_init=new_row_init, generator=generator)
    return first_model, second_model, merged_tokenizer, plan


# ---------------------------------------------------------------------------------------------------
# Mapping persistence
# ---------------------------------------------------------------------------------------------------


def save_vocab_merging_mapping(plan: TokenizerMergePlan, save_directory: str) -> str:
    """Persist the old-id -> new-id mapping next to a merged tokenizer.

    The file (`vocab_merging_map.json`) makes old checkpoints interpretable: given an old id, the mapping tells which
    merged id its embedding row now lives at. Saving the same plan twice writes identical content.

    Args:
        plan (`TokenizerMergePlan`):
            The plan to persist.
        save_directory (`str`):
            Directory in which to write the mapping file, typically the same directory passed to
            `tokenizer.save_pretrained`.

    Returns:
        `str`: The path of the written file.
    """
    os.makedirs(save_directory, exist_ok=True)
    path = os.path.join(save_directory, VOCAB_MERGING_FILE_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(plan.to_dict(), handle, indent=2, sort_keys=True, ensure_ascii=False)
    return path


def load_vocab_merging_mapping(directory_or_path: str) -> TokenizerMergePlan:
    """Load a [`~tokenizer_merging.TokenizerMergePlan`] written by [`~tokenizer_merging.save_vocab_merging_mapping`].

    Args:
        directory_or_path (`str`):
            Either the directory containing `vocab_merging_map.json` or the path to the file itself.
    """
    path = (
        os.path.join(directory_or_path, VOCAB_MERGING_FILE_NAME)
        if os.path.isdir(directory_or_path)
        else directory_or_path
    )
    with open(path, encoding="utf-8") as handle:
        return TokenizerMergePlan.from_dict(json.load(handle))
