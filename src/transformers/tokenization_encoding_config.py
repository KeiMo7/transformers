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
Content-addressed encoding configuration for tokenizers.

This module exports everything that participates in turning text into ids, so that a training pipeline and an
inference service can prove they encode text the same way. The exported configuration contains:

- a fingerprint of the vocabulary content (token -> id), not of a model name or version string;
- the ids of every named special token, recorded explicitly;
- the normalization / pre-tokenization / post-processing / decoding pipeline (fast backends) or the scalar encoding
  attributes (slow and SentencePiece backends);
- the truncation and padding strategy that is persisted on the tokenizer;
- optionally, a compressed copy of the backend serialization so the tokenizer can be rebuilt without the original
  files.

All of this is opt-in: nothing here runs during a normal `from_pretrained`/`save_pretrained` call, and the encoding
algorithms themselves are untouched.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .utils import logging


logger = logging.get_logger(__name__)

ENCODING_CONFIG_FILE = "encoding_config.json"
ENCODING_CONFIG_FORMAT = "transformers-tokenizer-encoding-config"
ENCODING_CONFIG_FORMAT_VERSION = 1

VOCABULARY_FINGERPRINT_ALGORITHM = "sha256-vocab-v1"
CONFIG_FINGERPRINT_ALGORITHM = "sha256-config-v1"
MODEL_FINGERPRINT_ALGORITHM = "sha256-model-v1"

# How many token ids are grouped into a single vocabulary "chunk". Chunk digests let `diff` report which id ranges
# changed (which is what an accidental vocabulary swap looks like) instead of only saying the whole file differs.
VOCABULARY_CHUNK_SIZE = 4096

# Values smaller than these are inlined in the exported config; larger ones are replaced by a content digest so that a
# large vocabulary or merge table does not make the config itself huge.
_MAX_INLINE_STRING = 512
_MAX_INLINE_SEQUENCE = 128

# Attributes that never affect encoding and must not leak into a content fingerprint (model names, local paths,
# loading flags...).
_NON_ENCODING_ATTRIBUTES = {
    "name_or_path",
    "backend",
    "is_local",
    "local_files_only",
    "files_loaded",
    "init_inputs",
    "init_kwargs",
    "tokenizer_file",
    "vocab_file",
    "merges_file",
    "spm_file",
    "gguf_file",
    "chat_template",
    "chat_control_tokens",
    "response_template",
    "processor_class",
    "_processor_class",
    "deprecation_warnings",
    "added_tokens_decoder",
    "added_tokens_encoder",
    "encoding_config",
    # Derived caches (a lazily computed vocabulary count and the trie used for splitting), not settings.
    "total_vocab_size",
    "tokens_trie",
}

_STANDARD_SPECIAL_TOKENS = (
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
)


class EncodingConfigMissingError(ValueError):
    """
    Raised when encoding-config verification is requested but no configuration can be found.

    This is deliberately an error rather than a silent fallback: falling back to defaults is exactly the failure mode
    that makes a training and an inference pipeline disagree without anyone noticing.
    """


class EncodingConfigMismatchError(ValueError):
    """
    Raised when a tokenizer's encoding configuration does not match the expected one.

    Attributes:
        differences (`list[EncodingConfigDifference]`): The individual differences, most actionable first.
    """

    def __init__(self, message: str, differences: list[EncodingConfigDifference] | None = None):
        super().__init__(message)
        self.differences = differences or []


@dataclass(frozen=True)
class EncodingConfigDifference:
    """A single difference between two encoding configurations."""

    path: str
    left: Any
    right: Any
    note: str | None = None
    _labels: tuple[str, str] = field(default=("current", "expected"), repr=False, compare=False)

    def describe(self) -> str:
        left = _format_value(self.left)
        right = _format_value(self.right)
        if self.left is _MISSING:
            base = f"{self.path}: only in {self._labels[1]} ({right})"
        elif self.right is _MISSING:
            base = f"{self.path}: only in {self._labels[0]} ({left})"
        else:
            base = f"{self.path}: {left} != {right}"
        return f"{base} ({self.note})" if self.note else base


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "<missing>"


_MISSING = _Missing()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _canonical_float(value: float) -> float:
    """
    Round a backend float to the precision the backend actually stores.

    `tokenizers` keeps Unigram scores as `f32` but serializes them as JSON doubles, so a serialize/parse/rebuild cycle
    can differ by one ULP. Rounding through `f32` makes the digest a fixed point of that cycle while keeping every
    value that could affect tokenization. Non-finite values are kept as-is so they still compare meaningfully.
    """
    if value != value or value in (float("inf"), float("-inf")):  # NaN / infinities
        return value
    return struct.unpack("f", struct.pack("f", value))[0]


def _digest_value(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _format_value(value: Any, max_length: int = 160) -> str:
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, str):
        rendered = repr(value)
    else:
        try:
            rendered = repr(value)
        except Exception:  # pragma: no cover - defensive
            rendered = f"<unrepresentable {type(value).__name__}>"
    if len(rendered) > max_length:
        rendered = rendered[: max_length - 3] + "..."
    return rendered


def _summarize(value: Any, max_string: int = _MAX_INLINE_STRING, max_sequence: int = _MAX_INLINE_SEQUENCE) -> Any:
    """
    Make an arbitrary pipeline description JSON-serializable and comparable without embedding huge blobs.

    Large strings and long sequences (SentencePiece charsmaps, BPE merge tables, vocabularies) are replaced by a
    content digest plus a size, so they still participate in comparisons but do not bloat the config.
    """
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, str):
        if len(value) > max_string:
            return {"__digest__": _sha256_bytes(value.encode("utf-8")), "size": len(value), "type": "string"}
        return value
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
        return {"__digest__": _sha256_bytes(data), "size": len(data), "type": "bytes"}
    if isinstance(value, (list, tuple)):
        if len(value) > max_sequence:
            return {
                "__digest__": _digest_value([_summarize(item, max_string, max_sequence) for item in value]),
                "size": len(value),
                "type": "sequence",
            }
        return [_summarize(item, max_string, max_sequence) for item in value]
    if isinstance(value, Mapping):
        if len(value) > max_sequence:
            return {
                "__digest__": _digest_value([_summarize(k, max_string, max_sequence) for k, _ in value.items()]),
                "size": len(value),
                "type": "mapping",
            }
        return {str(key): _summarize(item, max_string, max_sequence) for key, item in value.items()}
    return _summarize(repr(value), max_string, max_sequence)


def fingerprint_vocabulary(vocab: Mapping[str, int], chunk_size: int = VOCABULARY_CHUNK_SIZE) -> dict[str, Any]:
    """
    Compute a content fingerprint of a token -> id mapping.

    The digest depends only on the `(token, id)` pairs, never on a model name, a version string or the order in which
    the backend happens to return the vocabulary. Ids are grouped into chunks so a replacement can be localised to an
    id range.

    The vocabulary is hashed in one pass, walking ids in order. At a 100k vocabulary this takes ~100 ms, which is only
    paid when a config is exported or a tokenizer is verified, never on the normal load path.
    """
    header = VOCABULARY_FINGERPRINT_ALGORITHM.encode("utf-8") + b"\0"
    # Walk ids in order (`ordered[id] = token`) instead of sorting `(id, token)` pairs: at a 100k vocabulary this is
    # the difference between ~10 ms and ~15-20 ms per export. Ids need not be contiguous; holes are simply skipped.
    max_id = max((int(index) for index in vocab.values()), default=-1)
    ordered: list[str | None] = [None] * (max_id + 1)
    for token, index in vocab.items():
        ordered[int(index)] = token

    chunk_descriptions = []
    overall = hashlib.sha256(header)
    size = 0
    for start in range(0, max_id + 1, chunk_size):
        stop = min(start + chunk_size, max_id + 1)
        digest = hashlib.sha256(header)
        buffer = bytearray()
        for index in range(start, stop):
            token = ordered[index]
            if token is None:
                continue
            token_bytes = token.encode("utf-8")
            buffer += index.to_bytes(8, "big", signed=False)
            buffer += len(token_bytes).to_bytes(4, "big", signed=False)
            buffer += token_bytes
            size += 1
        digest.update(buffer)
        chunk_digest = digest.hexdigest()
        chunk_descriptions.append({"start": start, "stop": stop, "digest": chunk_digest})
        overall.update(start.to_bytes(8, "big", signed=False))
        overall.update(stop.to_bytes(8, "big", signed=False))
        overall.update(bytes.fromhex(chunk_digest))

    return {
        "algorithm": VOCABULARY_FINGERPRINT_ALGORITHM,
        "size": size,
        "chunk_size": chunk_size,
        "chunks": chunk_descriptions,
        "digest": overall.hexdigest(),
    }


def _changed_vocabulary_ranges(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Describe which token-id ranges differ between two vocabulary fingerprints."""
    if left.get("algorithm") != right.get("algorithm") or left.get("chunk_size") != right.get("chunk_size"):
        return ["fingerprint algorithm or chunk size differs; ranges cannot be compared"]

    def by_range(fingerprint):
        return {chunk["start"]: chunk for chunk in fingerprint.get("chunks", [])}

    left_chunks, right_chunks = by_range(left), by_range(right)
    ranges = []
    for start in sorted(set(left_chunks) | set(right_chunks)):
        lchunk, rchunk = left_chunks.get(start), right_chunks.get(start)
        if lchunk is None or rchunk is None or lchunk["digest"] != rchunk["digest"]:
            stop = (lchunk or rchunk)["stop"]
            ranges.append(f"ids {start}..{stop - 1}")
    if not ranges and left.get("size") != right.get("size"):
        ranges.append(f"vocabulary size {left.get('size')} -> {right.get('size')}")
    return ranges


def _compact_vocabulary_summary(fingerprint: dict[str, Any]) -> dict[str, Any]:
    """The part of a vocabulary fingerprint that participates in the config fingerprint."""
    return {
        "algorithm": fingerprint["algorithm"],
        "size": fingerprint["size"],
        "chunk_size": fingerprint["chunk_size"],
        "digest": fingerprint["digest"],
    }


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _is_scalar_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and all(_is_json_scalar(item) for item in value.values())


def _collect_encoding_attributes(tokenizer) -> dict[str, Any]:
    """
    Collect scalar attributes that tokenizer classes use to configure encoding.

    Fast tokenizers get their pipeline from the backend serialization; slow and SentencePiece tokenizers do not expose
    such a serialization, so their scalar attributes are the most reliable content description available. Keys that
    identify a model, a file or a loading mode are excluded so the result is content-based.
    """
    attributes: dict[str, Any] = {}
    candidates: dict[str, Any] = {}
    candidates.update(getattr(tokenizer, "init_kwargs", {}) or {})
    # Instance attributes set after `__init__` (e.g. `add_prefix_space`) are not in `init_kwargs`.
    candidates.update(getattr(tokenizer, "__dict__", {}))

    special_token_keys = set(getattr(tokenizer, "SPECIAL_TOKENS_ATTRIBUTES", ())) | set(_STANDARD_SPECIAL_TOKENS)
    for key in sorted(candidates):
        if key.startswith("_") or key in _NON_ENCODING_ATTRIBUTES or key in special_token_keys:
            continue
        value = candidates[key]
        if _is_json_scalar(value):
            attributes[key] = value
        elif _is_scalar_mapping(value):
            attributes[key] = dict(sorted(value.items()))

    return attributes


def _collect_special_tokens(
    tokenizer, vocab: Mapping[str, int] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Record every named special token with its explicit id, plus extra and added tokens."""
    if vocab is None:
        vocab = tokenizer.get_vocab()
    named: dict[str, Any] = {}
    special_tokens_map = getattr(tokenizer, "_special_tokens_map", {}) or {}
    for name in getattr(tokenizer, "SPECIAL_TOKENS_ATTRIBUTES", ()):  # model-specific names are appended per instance
        value = special_tokens_map.get(name)
        if value is None:
            continue
        token = str(value)
        token_id = tokenizer.convert_tokens_to_ids(token)
        named[name] = {
            "token": token,
            "id": token_id,
            "in_vocabulary": vocab.get(token) == token_id,
        }

    extra = []
    for token in getattr(tokenizer, "_extra_special_tokens", []) or []:
        token = str(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        extra.append({"token": token, "id": token_id, "in_vocabulary": vocab.get(token) == token_id})

    added = []
    try:
        decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    except NotImplementedError:
        # Some wrappers (e.g. `MistralCommonBackend`) deliberately do not expose added tokens.
        decoder = {}
    for token_id in sorted(decoder, key=int):
        token = decoder[token_id]
        added.append(
            {
                "id": int(token_id),
                "content": str(token),
                "single_word": bool(getattr(token, "single_word", False)),
                "lstrip": bool(getattr(token, "lstrip", False)),
                "rstrip": bool(getattr(token, "rstrip", False)),
                "normalized": bool(getattr(token, "normalized", False)),
                "special": bool(getattr(token, "special", False)),
            }
        )
    return named, extra, added


def _backend_truncation(tokenizer) -> dict[str, Any]:
    description: dict[str, Any] = {
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "side": getattr(tokenizer, "truncation_side", None),
    }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    params = getattr(backend, "truncation", None) if backend is not None else None
    description["active"] = params is not None
    if params is not None:
        description["params"] = _summarize(dict(params))
    return description


def _backend_padding(tokenizer) -> dict[str, Any]:
    description: dict[str, Any] = {
        "side": getattr(tokenizer, "padding_side", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "pad_token_type_id": getattr(tokenizer, "pad_token_type_id", None),
    }
    backend = getattr(tokenizer, "backend_tokenizer", None)
    params = getattr(backend, "padding", None) if backend is not None else None
    description["active"] = params is not None
    if params is not None:
        description["params"] = _summarize(dict(params))
    return description


def _encoding_flags(tokenizer) -> dict[str, Any]:
    return {
        name: (
            list(getattr(tokenizer, "model_input_names", []))
            if name == "model_input_names"
            else bool(getattr(tokenizer, name, False))
        )
        for name in _encoding_flags_keys()
    }


def _component_state(component: Any) -> Any:
    """Serialize a `tokenizers` pipeline component (normalizer, pre-tokenizer...) exactly as the backend stores it."""
    if component is None:
        return None
    getstate = getattr(component, "__getstate__", None)
    if getstate is None:
        return repr(component)
    try:
        state = getstate()
    except Exception:  # pragma: no cover - defensive
        return repr(component)
    if isinstance(state, (bytes, bytearray)):
        state = state.decode("utf-8")
    if isinstance(state, str):
        try:
            state = json.loads(state)
        except json.JSONDecodeError:  # pragma: no cover
            return state
    return state


def _model_vocabulary_size(vocab: Any) -> int:
    return len(vocab)


def _model_vocabulary_digest(vocab: Any) -> str:
    """
    Digest of the model's own vocabulary, in whatever shape the backend stores it.

    WordPiece and BPE store `token -> id`; Unigram stores `(token, score)` pairs, already ordered by id. Scores matter
    for Unigram tokenization, so they are hashed too. A Unigram list is hashed in its backend order (which is the id
    order) instead of being re-sorted: sorting a 250k-entry list dominated export time and changed nothing.
    """
    if isinstance(vocab, Mapping):
        normalized = sorted(vocab.items(), key=lambda item: (item[1], item[0]))
        return _digest_value(normalized)
    entries = [[str(token), _canonical_float(float(score))] for token, score in vocab]
    return _digest_value(entries)


def _normalize_serialized_vocabulary(vocab: Any) -> Any:
    """Round Unigram scores in a serialized vocabulary through `f32` (see `_canonical_float`)."""
    if isinstance(vocab, list) and vocab and isinstance(vocab[0], (list, tuple)) and len(vocab[0]) == 2:
        scores = np.asarray([entry[1] for entry in vocab], dtype=np.float32).astype(np.float64).tolist()
        return [[entry[0], score] for entry, score in zip(vocab, scores)]
    return vocab


def _serialized_backend_state(tokenizer) -> tuple[bytes, dict[str, Any]] | None:
    """
    Read the backend serialization once per export and return `(serialized_bytes, parsed)`.

    Fast backends expose everything both as pipeline components and as one JSON document. Parsing that document once
    per export instead of once per section meaningfully shortens a Unigram export. The parsed copy is only used to read
    the pipeline and the model description; the payload is always built from the original bytes so a rebuilt tokenizer
    is byte-for-byte the one that was serialized.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return None
    serialized = backend.to_str().encode("utf-8")
    try:
        parsed = json.loads(serialized)
    except json.JSONDecodeError:  # pragma: no cover - the backend always emits valid JSON
        return serialized, {}
    if not isinstance(parsed, dict):
        return serialized, {}
    # Normalize in place so the same object serves both the fingerprint and the pipeline description.
    model = parsed.get("model")
    if isinstance(model, dict) and "vocab" in model:
        model["vocab"] = _normalize_serialized_vocabulary(model["vocab"])
    return serialized, parsed


def _fast_backend_description(tokenizer, parsed: dict[str, Any] | None = None) -> dict[str, Any]:
    backend = tokenizer.backend_tokenizer
    if parsed is not None and isinstance(parsed.get("model"), dict):
        model = dict(parsed["model"])
    else:
        model = _component_state(backend.model)
    if isinstance(model, dict):
        model = dict(model)
        # The vocabulary is fingerprinted separately (by content, id by id); the merge table only needs a digest.
        # Drop both before `_summarize`, which would otherwise walk a 100k-entry mapping just to replace it.
        if "vocab" in model:
            vocab = model.pop("vocab")
            model["vocab"] = {"__digest__": _model_vocabulary_digest(vocab), "size": _model_vocabulary_size(vocab)}
        if "merges" in model:
            merges = model.pop("merges")
            model["merges"] = {"__digest__": _digest_value(list(merges)), "size": len(merges)}

    if parsed is not None:
        pipeline_components = {
            key: parsed.get(key) for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder")
        }
    else:
        pipeline_components = {
            key: _component_state(getattr(backend, key))
            for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder")
        }

    return {
        "backend": "tokenizers",
        "normalizer": _summarize(pipeline_components["normalizer"]),
        "pre_tokenizer": _summarize(pipeline_components["pre_tokenizer"]),
        "post_processor": _summarize(pipeline_components["post_processor"]),
        "decoder": _summarize(pipeline_components["decoder"]),
        "model": _summarize(model),
    }


def _sentencepiece_description(tokenizer) -> dict[str, Any]:
    description: dict[str, Any] = {"backend": "sentencepiece"}
    sp_model = getattr(tokenizer, "sp_model", None)
    if sp_model is not None:
        try:
            proto = sp_model.serialized_model_proto()
        except Exception:  # pragma: no cover - some sentencepiece builds do not expose the proto
            proto = None
        if proto is not None:
            description["model_proto"] = {
                "algorithm": MODEL_FINGERPRINT_ALGORITHM,
                "digest": _sha256_bytes(proto),
                "size": len(proto),
            }
    return description


def _generic_backend_description(tokenizer) -> dict[str, Any]:
    return {
        "backend": getattr(tokenizer, "backend", None) or "custom",
        "attributes": _summarize(_collect_encoding_attributes(tokenizer)),
    }


def _safe_base_vocab_size(tokenizer) -> int | None:
    """`vocab_size` without the added tokens, or `None` for custom tokenizers that do not expose it."""
    try:
        return tokenizer.vocab_size
    except (NotImplementedError, AttributeError):
        return None


def _collect_source_files(tokenizer) -> list[dict[str, Any]]:
    """Fingerprint the files that were actually read, so a replaced file with the same name is visible."""
    sources: dict[str, dict[str, Any]] = {}
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    for role in ("vocab_file", "merges_file", "spm_file", "tokenizer_file", "gguf_file"):
        path = init_kwargs.get(role)
        if not isinstance(path, (str, os.PathLike)):
            continue
        path = str(path)
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        try:
            digest = hashlib.sha256()
            size = 0
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
        except OSError:  # pragma: no cover - a file can disappear between resolution and export
            continue
        sources[name] = {"name": name, "role": role, "size": size, "sha256": digest.hexdigest()}

    if not sources:
        # SentencePiece tokenizers copy the model path onto `tokenizer.vocab_file` but not always into init_kwargs.
        path = getattr(tokenizer, "vocab_file", None)
        if isinstance(path, (str, os.PathLike)) and os.path.isfile(path):
            name = os.path.basename(str(path))
            with open(path, "rb") as handle:
                data = handle.read()
            sources[name] = {"name": name, "role": "vocab_file", "size": len(data), "sha256": _sha256_bytes(data)}

    return [sources[name] for name in sorted(sources)]


def _model_serialization_description(
    backend: str, tokenizer, serialized: bytes | None = None, parsed: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """
    Fingerprint the backend's own serialization.

    For fast backends the serialized JSON is normalized before hashing (Unigram scores are rounded through `f32`, maps
    are key-sorted) because `tokenizers` itself loses one ULP when it re-serializes an `f32` score as a double. Hashing
    the raw bytes would make a tokenizer rebuilt from its own payload look different from the original.
    """
    if backend == "tokenizers":
        if serialized is None:
            state = _serialized_backend_state(tokenizer)
            serialized, parsed = state if state is not None else (tokenizer.backend_tokenizer.to_str().encode(), None)
        if parsed:
            payload = _canonical_json(parsed)
        else:  # pragma: no cover - defensive
            payload = serialized
        return {
            "algorithm": MODEL_FINGERPRINT_ALGORITHM,
            "size": len(payload),
            "digest": _sha256_bytes(payload),
        }
    if backend == "sentencepiece":
        sp_model = getattr(tokenizer, "sp_model", None)
        if sp_model is None:
            return None
        try:
            proto = sp_model.serialized_model_proto()
        except Exception:  # pragma: no cover
            return None
        return {"algorithm": MODEL_FINGERPRINT_ALGORITHM, "size": len(proto), "digest": _sha256_bytes(proto)}
    return None


def _build_payload(tokenizer, serialized: bytes | None = None) -> tuple[str | None, dict[str, Any] | None]:
    """Return a `(format, payload)` pair that can rebuild the encoding component without the original files."""
    backend = getattr(tokenizer, "backend", None)
    if backend == "tokenizers":
        raw = serialized if serialized is not None else tokenizer.backend_tokenizer.to_str().encode("utf-8")
        payload_format = "tokenizers-json"
    elif backend == "sentencepiece":
        sp_model = getattr(tokenizer, "sp_model", None)
        if sp_model is None:
            return None, None
        try:
            raw = sp_model.serialized_model_proto()
        except Exception:  # pragma: no cover
            return None, None
        payload_format = "sentencepiece-proto"
    else:
        return None, None

    compressed = zlib.compress(raw, level=9)
    return payload_format, {
        "format": payload_format,
        "compression": "zlib+base64",
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def _build_vocabulary_payload(tokenizer, serialized: bytes | None = None) -> tuple[str | None, dict[str, Any] | None]:
    """
    Build a payload that can rebuild the encoding component without the original files.

    Fast backends serialize themselves. SentencePiece backends serialize their model. For other (custom) backends
    there is no serialization API, so the vocabulary and added-token table are recorded as JSON; that is enough for
    `from_pretrained`-style reconstruction as long as the class knows how to load from a vocabulary mapping.
    """
    backend = getattr(tokenizer, "backend", None)
    if backend in ("tokenizers", "sentencepiece"):
        return _build_payload(tokenizer, serialized=serialized)

    vocab = tokenizer.get_vocab()
    added = []
    try:
        decoder = getattr(tokenizer, "added_tokens_decoder", {}) or {}
    except NotImplementedError:
        decoder = {}
    for token_id, token in sorted(decoder.items(), key=lambda i: int(i[0])):
        added.append(
            {
                "id": int(token_id),
                "content": str(token),
                "single_word": bool(getattr(token, "single_word", False)),
                "lstrip": bool(getattr(token, "lstrip", False)),
                "rstrip": bool(getattr(token, "rstrip", False)),
                "normalized": bool(getattr(token, "normalized", False)),
                "special": bool(getattr(token, "special", False)),
            }
        )
    raw = _canonical_json({"vocab": vocab, "added_tokens": added})
    compressed = zlib.compress(raw, level=9)
    return "vocabulary-json", {
        "format": "vocabulary-json",
        "compression": "zlib+base64",
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def _decode_payload(payload: Mapping[str, Any]) -> bytes:
    if payload.get("compression") != "zlib+base64":
        raise ValueError(f"Unsupported encoding config payload compression: {payload.get('compression')!r}")
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    digest = _sha256_bytes(raw)
    if digest != payload.get("sha256"):
        raise ValueError(
            "Encoding config payload is corrupted: its content digest does not match the recorded sha256. "
            f"Expected {payload.get('sha256')}, got {digest}."
        )
    return raw


_NON_ENCODING_ARGUMENTS = {
    "name_or_path",
    "backend",
    "is_local",
    "local_files_only",
    "files_loaded",
    "tokenizer_object",
    "_json_truncation",
    "_json_padding",
    "_spm_precompiled_charsmap",
    "model_specific_special_tokens",
    "extra_special_tokens",
    "added_tokens_decoder",
}


def _rebuild_init_kwargs(tokenizer) -> dict[str, Any]:
    """The JSON-safe subset of `init_kwargs` needed to reconstruct the tokenizer class (never file paths)."""
    special_token_keys = set(getattr(tokenizer, "SPECIAL_TOKENS_ATTRIBUTES", ())) | set(_STANDARD_SPECIAL_TOKENS)
    result: dict[str, Any] = {}
    for key, value in (getattr(tokenizer, "init_kwargs", {}) or {}).items():
        if key.startswith("_") or key in _NON_ENCODING_ARGUMENTS or key in special_token_keys:
            continue
        # Paths are location-dependent, not content: they are fingerprinted separately in `sources` and must never
        # participate in a content fingerprint.
        if isinstance(value, (str, os.PathLike)) and os.path.exists(value):
            continue
        if _is_json_scalar(value) or _is_scalar_mapping(value) or isinstance(value, (list, tuple)):
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                continue
            result[key] = value
    return result


def _behavior_view(encoding: Mapping[str, Any], backend: str | None) -> dict[str, Any]:
    """
    The subset of an exported encoding description that actually determines token ids.

    Class names, module paths, the on-disk source files and the rebuild recipe describe how a tokenizer was obtained,
    not what it does: a tokenizer rebuilt from a saved config is the same encoder even though its class name differs.
    They live outside `encoding`, so this function only has to deal with duplicated settings inside it:

    - `flags` are the canonical names for settings that also appear in `attributes`; the duplicate is dropped so a
      change is reported once.
    - For fast (`tokenizers`) backends the serialized pipeline is authoritative and the Python attributes are a mirror
      of it, so `attributes` is dropped entirely. For slow and SentencePiece backends there is no serialized pipeline
      and the attributes are the encoding settings, so they stay.
    """
    view = copy.deepcopy(dict(encoding))
    attributes = view.get("attributes")
    if isinstance(attributes, Mapping):
        flag_names = set(_encoding_flags_keys())
        view["attributes"] = {key: value for key, value in attributes.items() if key not in flag_names}
        if backend == "tokenizers":
            # Keeping both would report the same change twice (e.g. `lowercase` in the BertNormalizer and
            # `do_lower_case`).
            view.pop("attributes")
    return view


def _encoding_flags_keys() -> tuple[str, ...]:
    return (
        "model_input_names",
        "split_special_tokens",
        "clean_up_tokenization_spaces",
        "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output",
        "add_bos_token",
        "add_eos_token",
    )


def export_encoding_config(tokenizer, *, include_vocabulary: bool = False) -> EncodingConfig:
    """
    Export a complete, content-addressed description of how `tokenizer` turns text into ids.

    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer to describe.
        include_vocabulary (`bool`, *optional*, defaults to `False`):
            If `True`, embed a compressed copy of the backend serialization. The resulting config can rebuild the
            encoding component with [`EncodingConfig.build_tokenizer`] even if the original vocabulary files are gone.
            The compact config (the default) is a few kilobytes and is enough to compare and verify configurations.
    """
    backend = getattr(tokenizer, "backend", None)
    vocab = tokenizer.get_vocab()
    vocab_fingerprint = fingerprint_vocabulary(vocab)

    serialized, parsed = None, None
    if backend == "tokenizers":
        state = _serialized_backend_state(tokenizer)
        if state is not None:
            serialized, parsed = state
        pipeline = _fast_backend_description(tokenizer, parsed)
    elif backend == "sentencepiece":
        pipeline = _sentencepiece_description(tokenizer)
    else:
        pipeline = _generic_backend_description(tokenizer)

    named_special_tokens, extra_special_tokens, added_tokens = _collect_special_tokens(tokenizer, vocab)
    encoding = {
        "pipeline": pipeline,
        "attributes": _summarize(_collect_encoding_attributes(tokenizer)),
        "special_tokens": named_special_tokens,
        "extra_special_tokens": extra_special_tokens,
        "added_tokens": added_tokens,
        "flags": _encoding_flags(tokenizer),
        "truncation": _backend_truncation(tokenizer),
        "padding": _backend_padding(tokenizer),
    }

    vocabulary_summary = {
        "size": len(vocab),
        "base_size": _safe_base_vocab_size(tokenizer),
        "fingerprint": vocab_fingerprint,
    }

    build: dict[str, Any] = {
        "class": type(tokenizer).__name__,
        "module": type(tokenizer).__module__,
        "backend": backend,
        "init_kwargs": _rebuild_init_kwargs(tokenizer),
        "supported": False,
        "payload": None,
    }
    if include_vocabulary:
        payload_format, payload = _build_vocabulary_payload(tokenizer, serialized=serialized)
        if payload is not None:
            build["supported"] = True
            build["payload"] = payload
    else:
        build["supported"] = (
            _model_serialization_description(backend, tokenizer, serialized=serialized, parsed=parsed) is not None
        )

    sources = _collect_source_files(tokenizer)
    model_fingerprint = _model_serialization_description(backend, tokenizer, serialized=serialized, parsed=parsed)

    config_digest = _digest_value(
        {
            "format_version": ENCODING_CONFIG_FORMAT_VERSION,
            "encoding": _behavior_view(encoding, backend),
            "vocabulary": _compact_vocabulary_summary(vocab_fingerprint),
            "model_serialization": model_fingerprint,
        }
    )

    data = {
        "format": ENCODING_CONFIG_FORMAT,
        "format_version": ENCODING_CONFIG_FORMAT_VERSION,
        "tokenizer_class": type(tokenizer).__name__,
        "backend": backend,
        "encoding": encoding,
        "vocabulary": vocabulary_summary,
        "sources": sources,
        "build": build,
        "fingerprints": {
            "config": {"algorithm": CONFIG_FINGERPRINT_ALGORITHM, "digest": config_digest},
            "vocabulary": vocab_fingerprint,
            "model_serialization": model_fingerprint,
        },
    }
    return EncodingConfig(data, _trusted=True)


def _resolve_tokenizer_class(module: str, name: str):
    import importlib

    if not module.startswith("transformers."):
        raise ValueError(
            f"Refusing to rebuild a tokenizer from an untrusted module {module!r}. "
            "Only tokenizer classes shipped with `transformers` can be rebuilt from a config."
        )
    try:
        imported = importlib.import_module(module)
        cls = getattr(imported, name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Could not resolve tokenizer class {module}.{name} to rebuild the tokenizer.") from error

    from .tokenization_utils_base import PreTrainedTokenizerBase

    if not (isinstance(cls, type) and issubclass(cls, PreTrainedTokenizerBase)):
        raise ValueError(f"{module}.{name} is not a tokenizer class; cannot rebuild from an encoding config.")
    return cls


class EncodingConfig:
    """
    A complete, content-addressed description of a tokenizer's encoding behavior.

    Create one with [`~tokenization_utils_base.PreTrainedTokenizerBase.export_encoding_config`], load one with
    [`EncodingConfig.load`], and compare two with [`EncodingConfig.diff`] or verify a tokenizer against one with
    [`~tokenization_utils_base.PreTrainedTokenizerBase.verify_encoding_config`].
    """

    def __init__(self, data: Mapping[str, Any], _trusted: bool = False):
        data = copy.deepcopy(dict(data))
        if not _trusted:
            self._validate(data)
        self._data = data

    @staticmethod
    def _validate(data: Mapping[str, Any]) -> None:
        if data.get("format") != ENCODING_CONFIG_FORMAT:
            raise ValueError(
                f"Not a transformers tokenizer encoding config: expected format {ENCODING_CONFIG_FORMAT!r}, "
                f"got {data.get('format')!r}."
            )
        version = data.get("format_version")
        if version != ENCODING_CONFIG_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported encoding config format_version {version!r}; this version of transformers understands "
                f"format_version {ENCODING_CONFIG_FORMAT_VERSION}. Re-export the config with a matching transformers "
                "release."
            )
        for required in ("encoding", "vocabulary", "fingerprints"):
            if required not in data:
                raise ValueError(f"Encoding config is missing the required section {required!r}.")
        fingerprint = data["fingerprints"].get("config", {}).get("digest")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("Encoding config is missing a valid `fingerprints.config.digest`.")
        # Recompute the digest from the recorded behavior: a hand-edited or stale config must not compare equal to a
        # tokenizer it no longer describes. Configs written by another format version are rejected above, so the
        # handled sections are the ones this version knows about.
        try:
            expected = _digest_value(
                {
                    "format_version": ENCODING_CONFIG_FORMAT_VERSION,
                    "encoding": _behavior_view(data["encoding"], data.get("backend")),
                    "vocabulary": _compact_vocabulary_summary(data["fingerprints"]["vocabulary"]),
                    "model_serialization": data["fingerprints"].get("model_serialization"),
                }
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Encoding config is malformed: {error}") from error
        if expected != fingerprint:
            raise ValueError(
                "Encoding config is inconsistent: `fingerprints.config.digest` does not match its contents. "
                "The file was edited after export or corrupted; re-export it with `tokenizer.export_encoding_config()`."
            )

    # -- construction -----------------------------------------------------
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EncodingConfig:
        return cls(data)

    @classmethod
    def load(cls, source: str | os.PathLike | Mapping[str, Any] | EncodingConfig) -> EncodingConfig:
        """Load a config from a path, a directory containing `encoding_config.json`, a dict, or another config."""
        if isinstance(source, EncodingConfig):
            return source
        if isinstance(source, Mapping):
            return cls.from_dict(source)
        path = Path(source)
        if path.is_dir():
            path = path / ENCODING_CONFIG_FILE
        if not path.exists():
            raise EncodingConfigMissingError(f"No encoding config found at {str(path)!r}.")
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f"{str(path)!r} is not valid JSON: {error}") from error
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def save(self, path: str | os.PathLike) -> str:
        """Write the config to `path` (or to `path/encoding_config.json` when `path` is a directory)."""
        path = Path(path)
        if path.is_dir():
            path = path / ENCODING_CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self._data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        return str(path)

    # -- accessors --------------------------------------------------------
    @property
    def fingerprint(self) -> str:
        """The content digest of the encoding configuration."""
        return self._data["fingerprints"]["config"]["digest"]

    @property
    def vocabulary_fingerprint(self) -> dict[str, Any]:
        return self._data["fingerprints"]["vocabulary"]

    @property
    def tokenizer_class(self) -> str | None:
        return self._data.get("tokenizer_class")

    @property
    def backend(self) -> str | None:
        return self._data.get("backend")

    @property
    def vocabulary_size(self) -> int | None:
        return self._data.get("vocabulary", {}).get("size")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EncodingConfig):
            return NotImplemented
        return self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        return hash(self.fingerprint)

    def __repr__(self) -> str:
        return (
            f"EncodingConfig(class={self.tokenizer_class!r}, backend={self.backend!r}, "
            f"vocabulary_size={self.vocabulary_size}, fingerprint={self.fingerprint[:12]}...)"
        )

    # -- comparison -------------------------------------------------------
    def diff(self, other: EncodingConfig | Mapping[str, Any]) -> list[EncodingConfigDifference]:
        """
        Return the individual differences between this config and `other`, most actionable first.

        Only settings that determine how text is encoded are compared. Class names, module paths, the rebuild recipe
        and whether the vocabulary payload is embedded are deliberately excluded: two configs exported from the same
        encoding agree on `diff` whatever their provenance.

        Differences in concrete settings are listed before fingerprint digests, and a vocabulary digest difference
        is annotated with the token-id ranges that changed.
        """
        other = EncodingConfig.load(other)
        left = self._comparison_view()
        right = other._comparison_view()

        differences: list[EncodingConfigDifference] = []
        _diff_values(left, right, "", differences)

        # Annotate the vocabulary digest difference with localised ranges when possible.
        left_vocab = left.get("fingerprints", {}).get("vocabulary", {})
        right_vocab = right.get("fingerprints", {}).get("vocabulary", {})
        if left_vocab.get("digest") != right_vocab.get("digest"):
            ranges = _changed_vocabulary_ranges(left_vocab, right_vocab)
            note = f"changed token-id ranges: {', '.join(ranges)}" if ranges else "vocabulary content differs"
            differences = [
                EncodingConfigDifference(d.path, d.left, d.right, note) if d.path.endswith("vocabulary.digest") else d
                for d in differences
            ]

        def sort_key(difference: EncodingConfigDifference):
            # Concrete settings first, then vocabulary/model digests, then the derived config digest.
            if difference.path.startswith("encoding."):
                return (0, difference.path)
            if difference.path.startswith("vocabulary.") or difference.path.startswith("fingerprints.vocabulary."):
                return (1, difference.path)
            if difference.path.endswith("config.digest"):
                return (4, difference.path)
            return (2, difference.path)

        differences.sort(key=sort_key)
        return differences

    def _comparison_view(self) -> dict[str, Any]:
        """The behavior-defining sections used by `diff`, in a stable order."""
        return {
            "encoding": _behavior_view(self._data["encoding"], self._data.get("backend")),
            "vocabulary": {
                "size": self._data.get("vocabulary", {}).get("size"),
                "base_size": self._data.get("vocabulary", {}).get("base_size"),
            },
            "fingerprints": {
                # Chunks are kept so a vocabulary change can be localised to id ranges.
                "vocabulary": self._data["fingerprints"]["vocabulary"],
                "model_serialization": self._data["fingerprints"].get("model_serialization"),
            },
        }

    def format_mismatch(
        self, other: EncodingConfig, name: str | None = None, labels: tuple[str, str] = ("current", "expected")
    ) -> tuple[str, list[EncodingConfigDifference]]:
        """Build the message and annotated difference list for a mismatch (used by the raising helpers)."""
        differences = [EncodingConfigDifference(d.path, d.left, d.right, d.note, labels) for d in self.diff(other)]
        name = name or self.tokenizer_class or "Tokenizer"
        header = f"{name}: tokenizer encoding configuration mismatch ({len(differences)} difference(s)):"
        lines = [header]
        lines.extend(f"  - {difference.describe()}" for difference in differences)
        lines.append("Export both configurations with `tokenizer.export_encoding_config()` to inspect them in full.")
        return "\n".join(lines), differences

    def assert_matches(self, other: EncodingConfig, name: str | None = None) -> None:
        """Raise [`EncodingConfigMismatchError`] when this config differs from `other`."""
        if self.fingerprint == other.fingerprint and not self.diff(other):
            return
        message, differences = self.format_mismatch(other, name=name)
        raise EncodingConfigMismatchError(message, differences)

    # -- rebuilding -------------------------------------------------------
    def build_tokenizer(self, **overrides):
        """
        Rebuild a tokenizer from this config.

        Requires the config to have been exported with `include_vocabulary=True`; otherwise there is no vocabulary
        content to rebuild from and a `ValueError` is raised. Rebuilding does not read the original vocabulary files.

        Returns:
            A tokenizer whose encoding of any input matches the original token for token.
        """
        build = self._data.get("build") or {}
        payload = build.get("payload")
        if not payload:
            raise ValueError(
                "This encoding config does not embed a vocabulary payload, so it cannot rebuild a tokenizer. "
                "Export it again with `tokenizer.export_encoding_config(include_vocabulary=True)`."
            )
        raw = _decode_payload(payload)
        payload_format = payload.get("format") or build.get("format")
        special_kwargs = self._special_token_kwargs()
        flags = self._data["encoding"].get("flags", {})
        extra_kwargs = {
            "model_input_names": flags.get("model_input_names"),
            "split_special_tokens": flags.get("split_special_tokens", False),
            "clean_up_tokenization_spaces": flags.get("clean_up_tokenization_spaces", False),
            "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output": flags.get(
                "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output", False
            ),
            "padding_side": self._data["encoding"]["padding"].get("side") or "right",
            "truncation_side": self._data["encoding"]["truncation"].get("side") or "right",
        }
        if self._data["encoding"]["truncation"].get("model_max_length") is not None:
            extra_kwargs["model_max_length"] = self._data["encoding"]["truncation"]["model_max_length"]
        extra_kwargs.update(special_kwargs)
        extra_kwargs.update(overrides)

        if payload_format == "tokenizers-json":
            from tokenizers import Tokenizer as TokenizerFast

            backend = TokenizerFast.from_str(raw.decode("utf-8"))
            # Prefer the original tokenizer class (with the payload as a `vocab`-equivalent) so a rebuilt tokenizer
            # behaves like the original even if the class sets opt-in attributes in `__init__`.
            cls, class_kwargs = self._resolve_build_class()
            if cls is not None:
                try:
                    rebuilt = cls(**{**class_kwargs, **extra_kwargs, "tokenizer_object": backend})
                except Exception as error:  # noqa: BLE001 - fall back to the generic backend
                    logger.debug(
                        f"{cls.__name__} could not be rebuilt from the payload ({error}); using TokenizersBackend."
                    )
                else:
                    # A class `__init__` may read settings (e.g. `extra_ids`) from its arguments and then rebuild the
                    # backend around them, dropping parts of the payload. The payload is the authoritative record of
                    # the encoding, so put it back; the added tokens are already part of it.
                    rebuilt._tokenizer = TokenizerFast.from_str(raw.decode("utf-8"))
                    self._apply_saved_truncation_padding(rebuilt)
                    return rebuilt

            from .tokenization_utils_tokenizers import TokenizersBackend

            rebuilt = TokenizersBackend(tokenizer_object=backend, **extra_kwargs)
            self._apply_saved_truncation_padding(rebuilt)
            return rebuilt

        if payload_format == "sentencepiece-proto":
            import tempfile

            cls = _resolve_tokenizer_class(build["module"], build["class"])
            kwargs = dict(build.get("init_kwargs") or {})
            kwargs.update(extra_kwargs)
            added_tokens = self._data["encoding"].get("added_tokens") or []
            if added_tokens:
                from .tokenization_utils_base import AddedToken

                kwargs["added_tokens_decoder"] = {
                    int(token["id"]): AddedToken(
                        **{
                            key: token[key]
                            for key in ("content", "single_word", "lstrip", "rstrip", "normalized", "special")
                        }
                    )
                    for token in added_tokens
                }
            handle = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
            try:
                handle.write(raw)
                handle.close()
                return cls(vocab_file=handle.name, **kwargs)
            finally:
                try:
                    os.unlink(handle.name)
                except OSError:  # pragma: no cover
                    pass

        if payload_format == "vocabulary-json":
            cls = _resolve_tokenizer_class(build["module"], build["class"])
            payload_data = json.loads(raw.decode("utf-8"))
            kwargs = dict(build.get("init_kwargs") or {})
            kwargs.update(extra_kwargs)
            from .tokenization_utils_base import AddedToken

            kwargs["added_tokens_decoder"] = {
                int(token["id"]): AddedToken(
                    **{
                        key: token[key]
                        for key in ("content", "single_word", "lstrip", "rstrip", "normalized", "special")
                    }
                )
                for token in payload_data.get("added_tokens") or []
            }
            # Custom tokenizers take the vocabulary under one of these names. Pick it from the signature instead of
            # trying each and swallowing `TypeError`, which could hide a real constructor failure.
            import inspect

            parameters = inspect.signature(cls.__init__).parameters
            for key in ("vocab", "vocabulary", "vocab_dict"):
                if key in parameters:
                    return cls(**{key: payload_data["vocab"], **kwargs})
            raise ValueError(
                f"{cls.__name__} cannot be rebuilt from a vocabulary payload: its `__init__` does not accept the "
                "vocabulary as a mapping under `vocab`, `vocabulary` or `vocab_dict`."
            )

        raise ValueError(f"Unsupported encoding config payload format {payload_format!r}.")

    def _resolve_build_class(self):
        """Resolve the recorded tokenizer class, or `None` when it cannot be rebuilt from a payload."""
        build = self._data.get("build") or {}
        if not build.get("class") or not build.get("module"):
            return None, {}
        try:
            cls = _resolve_tokenizer_class(build["module"], build["class"])
        except ValueError:
            return None, {}
        return cls, dict(build.get("init_kwargs") or {})

    def _apply_saved_truncation_padding(self, tokenizer) -> None:
        """
        Re-apply a persistent truncation/padding strategy recorded in the config.

        The strategy is persisted on the tokenizer itself (`encoder.enable_truncation` / `enable_padding` inside
        `tokenizer.json`). The rebuilt tokenizer must reproduce it so that serialize -> rebuild -> serialize is a
        fixed point; otherwise a rebuilt tokenizer's own export would disagree with this config.
        """
        truncation = self._data["encoding"].get("truncation") or {}
        padding = self._data["encoding"].get("padding") or {}
        if truncation.get("active") and isinstance(truncation.get("params"), Mapping):
            try:
                tokenizer.backend_tokenizer.enable_truncation(**dict(truncation["params"]))
            except Exception as error:  # pragma: no cover - defensive, strategy came from the same backend
                logger.warning(f"Could not restore the recorded truncation strategy: {error}")
        if padding.get("active") and isinstance(padding.get("params"), Mapping):
            try:
                tokenizer.backend_tokenizer.enable_padding(**dict(padding["params"]))
            except Exception as error:  # pragma: no cover
                logger.warning(f"Could not restore the recorded padding strategy: {error}")

    def _special_token_kwargs(self) -> dict[str, Any]:
        encoding = self._data["encoding"]
        named = encoding.get("special_tokens") or {}
        kwargs: dict[str, Any] = {}
        model_specific = {}
        for name, info in named.items():
            if name in _STANDARD_SPECIAL_TOKENS:
                kwargs[name] = info["token"]
            else:
                model_specific[name] = info["token"]
        if model_specific:
            kwargs["model_specific_special_tokens"] = model_specific
        extra = encoding.get("extra_special_tokens") or []
        if extra:
            kwargs["extra_special_tokens"] = [item["token"] for item in extra]
        return kwargs


def _diff_values(
    left: Any,
    right: Any,
    path: str,
    out: list[EncodingConfigDifference],
) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left) | set(right), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            # Payload bytes are represented by their sha256 and are not diffed as blobs.
            if path.endswith("build.payload") and key == "data":
                continue
            if key not in left:
                out.append(EncodingConfigDifference(child_path, _MISSING, right[key]))
            elif key not in right:
                out.append(EncodingConfigDifference(child_path, left[key], _MISSING))
            else:
                _diff_values(left[key], right[key], child_path, out)
        return

    if isinstance(left, list) and isinstance(right, list):
        if path.endswith("fingerprints.vocabulary.chunks"):
            # Chunk lists are summarised by the vocabulary digest annotation instead of one entry per chunk.
            return
        if len(left) != len(right):
            out.append(
                EncodingConfigDifference(
                    path,
                    f"list of {len(left)} item(s)",
                    f"list of {len(right)} item(s)",
                )
            )
            return
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _diff_values(left_item, right_item, f"{path}[{index}]", out)
        return

    if type(left) is not type(right) or left != right:
        out.append(EncodingConfigDifference(path, left, right))


__all__ = [
    "ENCODING_CONFIG_FILE",
    "ENCODING_CONFIG_FORMAT",
    "ENCODING_CONFIG_FORMAT_VERSION",
    "EncodingConfig",
    "EncodingConfigDifference",
    "EncodingConfigMismatchError",
    "EncodingConfigMissingError",
    "export_encoding_config",
    "fingerprint_vocabulary",
]
