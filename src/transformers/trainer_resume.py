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
"""Position-based (non-replaying) resumption of the training data stream.

The legacy resume path fast-forwards the dataloader with `skip_first_batches`, which costs one
extra pass over the already-seen data and cannot reproduce the exact batch order in every setup.
This module persists the position of the data stream instead:

* map-style datasets get a per-epoch plan that is a pure function of `(seed, epoch)`, plus the
  number of micro-batches the process has already consumed from that plan;
* streaming datasets get the source's own `state_dict()` captured at item boundaries, so the state
  that reproduces a batch can be looked up by position.

Nothing replays, so the cost of resuming does not depend on how far training got.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import deque
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset


RESUME_STATE_NAME = "resume_state.pt"
RESUME_GRADS_NAME = "resume_grads.pt"
CHECKPOINT_MANIFEST_NAME = "checkpoint_manifest.json"
RESUME_FORMAT_VERSION = 1
# Item boundaries a streaming source keeps so the trainer can look up the state of a batch that
# the loader has already pulled past. This process reads at most a few batches ahead, so the
# window is orders of magnitude larger than needed.
STREAMING_HISTORY = 8192


# `BatchSamplerShard` is what `accelerator.prepare` uses to hand each rank its batches, so reusing
# it here keeps the per-rank batch assignment identical to the legacy path. With `batch_size=1` and
# one id per batch it shards *batch ids* instead of sample ids; `BatchSamplerShard` compares
# `len(batch)` to `batch_size`, so the ids it duplicates at an uneven epoch tail are the same
# batches the legacy path duplicates.
class _IdBatchSampler(BatchSampler):
    def __init__(self, num_batches: int, drop_last: bool):
        self.num_batches = num_batches
        self.batch_size = 1
        self.drop_last = drop_last

    def __iter__(self):
        for batch_id in range(self.num_batches):
            yield [batch_id]

    def __len__(self):
        return self.num_batches


class ResumableBatchSampler(BatchSampler):
    """Batch sampler whose position in the epoch is explicit and survives a restart.

    The epoch's sample order is `randperm(len(dataset), seed + epoch)`, which is cheap to rebuild
    and independent of training progress, and the rank split is delegated to accelerate's
    `BatchSamplerShard` so each process sees exactly the batches the legacy path gives it.

    Unlike a `RandomSampler`, the position is an attribute (`batches_consumed`) rather than hidden
    generator state, so a checkpoint stores it and a resume jumps straight to it.
    """

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        drop_last: bool,
        seed: int,
        num_processes: int = 1,
        process_index: int = 0,
        even_batches: bool = True,
        shuffle: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
        if num_processes < 1 or not 0 <= process_index < num_processes:
            raise ValueError(
                f"Invalid data-parallel layout: process_index={process_index}, num_processes={num_processes}."
            )
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.num_processes = num_processes
        self.process_index = process_index
        self.even_batches = even_batches
        self.shuffle = shuffle

        self.epoch = 0
        self.batches_consumed = 0
        self._order: torch.Tensor | None = None
        self._batch_ids: list[int] | None = None

    # -- plan -----------------------------------------------------------------------------
    def _global_order(self, epoch: int) -> torch.Tensor:
        if not self.shuffle:
            return torch.arange(self.dataset_len)
        generator = torch.Generator()
        # Modulo keeps the seed inside the signed 64-bit range every backend accepts.
        generator.manual_seed((self.seed + epoch) % (2**63 - 1))
        return torch.randperm(self.dataset_len, generator=generator)

    def _rank_batch_ids(self, epoch: int) -> list[int]:
        num_global_batches = self.dataset_len // self.batch_size
        if not self.drop_last and self.dataset_len % self.batch_size:
            num_global_batches += 1
        if num_global_batches == 0:
            return []
        if self.num_processes == 1:
            return list(range(num_global_batches))

        from accelerate.data_loader import BatchSamplerShard

        shard = BatchSamplerShard(
            _IdBatchSampler(num_global_batches, self.drop_last),
            num_processes=self.num_processes,
            process_index=self.process_index,
            split_batches=False,
            even_batches=self.even_batches,
        )
        return [batch[0] for batch in shard]

    def _materialize(self) -> None:
        if self._order is None:
            self._order = self._global_order(self.epoch)
        if self._batch_ids is None:
            self._batch_ids = self._rank_batch_ids(self.epoch)

    # -- sampler API ----------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Select the epoch, keeping the position when the epoch does not change.

        `DataLoaderShard.__iter__` calls `set_epoch` with the epoch it already holds, so resetting
        unconditionally would erase a cursor that was just restored.
        """
        if epoch == self.epoch:
            return
        self.epoch = epoch
        self.batches_consumed = 0
        self._order = None
        self._batch_ids = None

    def __len__(self) -> int:
        self._materialize()
        return len(self._batch_ids)

    def __iter__(self):
        self._materialize()
        start = self.batches_consumed
        end = len(self._batch_ids)
        if start > end:
            raise RuntimeError(
                f"The training data cursor is at batch {start} of epoch {self.epoch}, but that epoch only has "
                f"{end} batches for process {self.process_index} (dataset length {self.dataset_len}, batch size "
                f"{self.batch_size}, {self.num_processes} processes). The checkpoint belongs to a different dataset "
                "or batch configuration; it cannot be resumed from."
            )
        for position in range(start, end):
            batch_id = self._batch_ids[position]
            begin = batch_id * self.batch_size
            yield self._order[begin : begin + self.batch_size].tolist()
            self.batches_consumed = position + 1

    # -- persistence ----------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        self._materialize()
        return {
            "format": RESUME_FORMAT_VERSION,
            "kind": "map",
            "epoch": self.epoch,
            "batches_consumed": self.batches_consumed,
            "batches_per_epoch": len(self._batch_ids),
            "dataset_len": self.dataset_len,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "num_processes": self.num_processes,
            "process_index": self.process_index,
            "even_batches": self.even_batches,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        permanent = (
            "kind",
            "dataset_len",
            "batch_size",
            "drop_last",
            "seed",
            "shuffle",
            "num_processes",
            "process_index",
        )
        mismatches = [
            f"{key}: checkpoint={state.get(key)!r} current={value!r}"
            for key, value in expected.items()
            if key in permanent and state.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                "The training data checkpoint does not match this run, so the sample order cannot be restored:\n  - "
                + "\n  - ".join(mismatches)
            )
        cursor = state.get("batches_consumed")
        planned = state.get("batches_per_epoch")
        if not isinstance(cursor, int) or not isinstance(planned, int) or cursor < 0:
            raise ValueError(
                f"Invalid training data cursor in checkpoint: batches_consumed={cursor!r}, "
                f"batches_per_epoch={planned!r}."
            )
        self.epoch = int(state["epoch"])
        self._order = None
        self._batch_ids = None
        self._materialize()
        if len(self._batch_ids) != planned:
            raise ValueError(
                f"The checkpoint recorded {planned} batches for epoch {self.epoch} but this run plans "
                f"{len(self._batch_ids)} (dataset length {self.dataset_len}, batch size {self.batch_size}, "
                f"process {self.process_index}/{self.num_processes}). The dataset or batch configuration changed."
            )
        if cursor > len(self._batch_ids):
            raise ValueError(
                f"The training data cursor is beyond the end of epoch {self.epoch}: "
                f"batches_consumed={cursor}, batches_per_epoch={len(self._batch_ids)}. The checkpoint was produced "
                "by a different dataset length or batch size, or its saved step and epoch do not belong together."
            )
        self.batches_consumed = cursor

    def set_batches_consumed(self, batches_consumed: int) -> None:
        """Move the cursor to an absolute position, as recorded by the training loop."""
        self._materialize()
        if not 0 <= batches_consumed <= len(self._batch_ids):
            raise ValueError(
                f"Cannot position the training data at batch {batches_consumed} of epoch {self.epoch}: "
                f"this epoch has {len(self._batch_ids)} batches for process {self.process_index}."
            )
        self.batches_consumed = batches_consumed


class SeededDataset:
    """Make per-sample randomness a pure function of `(seed, epoch, process, index)`.

    Augmentation that draws from the global RNGs normally depends on how many samples a process
    has already handled, so a resumed run that starts mid-epoch cannot reproduce it. Seeding around
    each `__getitem__` makes a sample's randomness depend only on its identity, and the caller's RNG
    state is restored so model-side draws are unaffected.
    """

    def __init__(self, dataset, seed: int, process_index: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.process_index = process_index
        self.epoch = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        seed = _mix(self.seed, self.epoch, self.process_index, index)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        try:
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


def _mix(*values: int) -> int:
    """Deterministic, process-independent mixing of a few integers into a 63-bit seed."""
    mask = (1 << 64) - 1
    state = 0x9E3779B97F4A7C15
    for value in values:
        state = (state ^ (int(value) & mask)) & mask
        state = (state * 0xBF58476D1CE4E5B9) & mask
        state ^= state >> 27
    return state % (2**63 - 1)


class TrackingIterableDataset(IterableDataset):
    """Wrap a streaming source and remember its state at item boundaries.

    A source's `state_dict()` describes the position *before* the next item, so the state that
    reproduces item `m` is the one captured when the source sits at boundary `m`. Keeping the last
    `history` such states lets the trainer look up "the state at boundary `m`" after the loader has
    already pulled past it, which is what makes resuming exact without replaying.
    """

    def __init__(self, source, history: int = STREAMING_HISTORY):
        if not hasattr(source, "state_dict") or not hasattr(source, "load_state_dict"):
            raise ValueError(
                "`resumable_dataloader=True` needs a streaming dataset implementing `state_dict`/`load_state_dict` "
                f"(for example `datasets.IterableDataset`), but {type(source).__name__} does not. Use a map-style "
                "dataset or disable `resumable_dataloader`."
            )
        self.source = source
        self.position = 0
        self.epoch = 0
        self._history: deque[tuple[int, dict]] = deque(maxlen=history)
        self._iterator = None
        self._restored = False
        self._record(0)

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        self._iterator = iter(self.source)
        if self._restored:
            self._restored = False
        return self

    def __next__(self):
        if self._iterator is None:
            self.__iter__()
        item = next(self._iterator)
        self.position += 1
        self._record(self.position)
        return item

    def _record(self, position: int) -> None:
        state = normalize_state(self.source.state_dict())
        if self._history and self._history[-1][0] == position:
            self._history[-1] = (position, state)
        else:
            self._history.append((position, state))

    def set_epoch(self, epoch: int) -> None:
        if epoch == self.epoch:
            return
        self.epoch = epoch
        self.position = 0
        self._history.clear()
        self._restored = False
        self._record(0)
        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(epoch)

    def state_at(self, position: int) -> dict:
        for candidate, state in reversed(self._history):
            if candidate == position:
                return state
        oldest = self._history[0][0] if self._history else None
        raise RuntimeError(
            f"No streaming-source snapshot is available for item boundary {position}; the loader currently holds "
            f"boundaries {oldest}..{self.position}. The checkpoint is further behind the iterator than "
            f"{len(self._history)} items of history, which should not happen without data-loader prefetching. "
            "Disable data-loader workers for streaming sources with `resumable_dataloader=True`."
        )

    def restore(self, position: int, state: dict, epoch: int | None = None) -> None:
        """Point the source at `position`; the next pull yields the item that follows it."""
        self.source.load_state_dict(state)
        if epoch is not None:
            self.epoch = epoch
        self.position = position
        self._history.clear()
        self._history.append((position, state))
        self._iterator = None
        self._restored = True


class ResumableStreamingLoader:
    """Loader wrapper that names stream positions by the number of trained batches.

    Each local sample is one item pulled from the source, so the boundary after `n` batches is the
    item boundary `n * items_per_batch`. The trainer supplies `n` from the micro-batches it has
    actually trained, because the loader itself can be a prefetched batch ahead.

    Only single-process streaming is supported: when one stream is sharded or dispatched across
    ranks, a single position does not describe what each rank has seen, so the combination is
    refused at construction instead of being checkpointed wrongly.
    """

    def __init__(self, dataloader, tracker: TrackingIterableDataset, num_processes: int = 1):
        if num_processes != 1:
            raise ValueError(
                "`resumable_dataloader=True` cannot resume a streaming dataset with multiple data-parallel "
                f"processes ({num_processes} are configured): a single source position cannot describe what each "
                "process has consumed. Disable the option to replay the stream on resume instead."
            )
        self.dataloader = dataloader
        self.tracker = tracker
        self.items_per_batch: int | None = None

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            size = _batch_size(batch)
            if size <= 0:
                raise RuntimeError("The streaming loader produced an empty batch, so no position can be recorded.")
            if self.items_per_batch is None:
                self.items_per_batch = size
            elif size != self.items_per_batch:
                raise RuntimeError(
                    f"The streaming loader produced batches of {self.items_per_batch} and then {size} samples. "
                    "Exact resumption needs every batch to consume the same number of source items, so "
                    "`resumable_dataloader=True` cannot guarantee it here. Disable the option to replay on resume."
                )
            yield batch

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(epoch)

    def boundary(self, batches: int) -> int:
        if self.items_per_batch is None:
            return 0
        return batches * self.items_per_batch

    def state_dict(self, batches: int) -> dict[str, Any]:
        """Record the position after `batches` micro-batches of the current epoch.

        `batches` comes from the trainer, which knows how many micro-batches it actually trained;
        the loader may be a prefetched batch ahead of that.
        """
        if self.items_per_batch is None:
            if batches:
                raise RuntimeError("Cannot record a streaming position before the first batch is drawn.")
            if self.tracker.position == 0:
                return {
                    "format": RESUME_FORMAT_VERSION,
                    "kind": "streaming",
                    "epoch": None,
                    "batches_consumed": 0,
                    "items_per_batch": None,
                    "source_state": self.tracker.state_at(0),
                }
        return {
            "format": RESUME_FORMAT_VERSION,
            "kind": "streaming",
            "epoch": self.tracker.epoch,
            "batches_consumed": batches,
            "items_per_batch": self.items_per_batch,
            "source_state": self.tracker.state_at(self.boundary(batches)),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("kind") != "streaming":
            raise ValueError(
                f"The checkpoint holds a `{state.get('kind')}` data cursor but this run streams its dataset. The "
                "dataset type changed between runs; exact resumption is impossible."
            )
        batches = state["batches_consumed"]
        items_per_batch = state.get("items_per_batch")
        if batches:
            if not isinstance(items_per_batch, int) or items_per_batch <= 0:
                raise ValueError(
                    f"The checkpoint records an invalid streaming position: items_per_batch={items_per_batch!r}."
                )
            self.items_per_batch = items_per_batch
        self.tracker.restore(batches * (items_per_batch or 0), state["source_state"], epoch=state.get("epoch"))


def _batch_size(batch) -> int:
    """Number of samples in a collated batch, without assuming a particular key layout."""
    if hasattr(batch, "shape"):
        return int(batch.shape[0])
    if isinstance(batch, dict):
        for value in batch.values():
            size = _batch_size(value)
            if size is not None:
                return size
        raise RuntimeError("Could not determine the number of samples in a streaming batch.")
    if isinstance(batch, (list, tuple)) and batch:
        return _batch_size(batch[0])
    if isinstance(batch, (int, float)):
        return 1
    raise RuntimeError(f"Could not determine the number of samples in a streaming batch of type {type(batch)}.")


def batch_sampler_of(dataloader: DataLoader):
    """Return the resumable sampler this module installed, if the loader owns one."""
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    inner = getattr(batch_sampler, "batch_sampler", None)
    for candidate in (inner, batch_sampler):
        if isinstance(candidate, ResumableBatchSampler):
            return candidate
    return None


def dataset_signature(dataset) -> dict[str, Any]:
    """Describe a dataset well enough to refuse a resume against a different one."""
    while isinstance(dataset, (SeededDataset, TrackingIterableDataset)):
        dataset = dataset.dataset if isinstance(dataset, SeededDataset) else dataset.source
    signature: dict[str, Any] = {"type": f"{type(dataset).__module__}.{type(dataset).__qualname__}"}
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint is not None:
        signature["fingerprint"] = str(fingerprint)
    try:
        signature["length"] = len(dataset)
    except TypeError:
        pass
    columns = getattr(dataset, "column_names", None)
    if columns is not None:
        signature["columns"] = list(columns)
    return signature


def compare_dataset_signatures(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    keys = sorted(set(expected) | set(actual))
    return [
        f"{key}: checkpoint={expected.get(key)!r} current={actual.get(key)!r}"
        for key in keys
        if expected.get(key) != actual.get(key)
    ]


def normalize_state(state: Any) -> Any:
    """Reduce a dataset state to plain JSON types so it can be stored with `weights_only=True`."""
    return json.loads(json.dumps(state))


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _fsync(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_manifest(checkpoint_dir: str, step: int, extra: dict[str, Any] | None = None) -> None:
    """Write the completion marker that makes a checkpoint directory visible as whole.

    The manifest is written last, after every other file, and lists them with their sizes. A
    directory without it is treated as an interrupted write and refused on resume.
    """
    entries = {}
    for name in os.listdir(checkpoint_dir):
        # Never describe the manifest with itself: it is replaced after this snapshot.
        if name == CHECKPOINT_MANIFEST_NAME:
            continue
        path = os.path.join(checkpoint_dir, name)
        if os.path.isfile(path):
            entry = {"size": os.path.getsize(path)}
            # The data cursor is what a resume rewinds to, so guard it against silent corruption too.
            if name in (RESUME_STATE_NAME, RESUME_GRADS_NAME):
                entry["sha256"] = _sha256(path)
            entries[name] = entry
    manifest = {"format": RESUME_FORMAT_VERSION, "step": step, "files": entries}
    if extra:
        manifest.update(extra)
    target = os.path.join(checkpoint_dir, CHECKPOINT_MANIFEST_NAME)
    fd, temporary = tempfile.mkstemp(dir=checkpoint_dir, prefix="tmp-manifest-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    _fsync(checkpoint_dir)


def read_manifest(checkpoint_dir: str) -> dict[str, Any] | None:
    path = os.path.join(checkpoint_dir, CHECKPOINT_MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def verify_manifest(checkpoint_dir: str, manifest: dict[str, Any]) -> list[str]:
    problems = []
    for name, entry in manifest.get("files", {}).items():
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            problems.append(f"missing file: {name}")
            continue
        size = os.path.getsize(path)
        if entry.get("size") is not None and size != entry["size"]:
            problems.append(f"truncated file: {name} ({size} bytes, expected {entry['size']})")
        elif "sha256" in entry and _sha256(path) != entry["sha256"]:
            problems.append(f"corrupted file: {name} (checksum mismatch)")
    return problems


def write_torch_atomic(path: str, payload: Any) -> None:
    """Write a torch file through a temporary sibling so readers never see a partial file."""
    directory = os.path.dirname(os.path.abspath(path))
    # `torch.save` rejects hidden file names, so the temporary is not dot-prefixed.
    fd, temporary = tempfile.mkstemp(dir=directory, prefix="tmp-resume-")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        _fsync(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


@contextmanager
def atomic_checkpoint_dir(checkpoint_dir: str, enabled: bool, is_main: bool = True, barrier=None):
    """Yield a directory to write a checkpoint into, publishing it only once it is complete.

    With `enabled`, files are staged in a sibling directory and renamed into place, so a crash
    leaves either the previous checkpoint or the new one, never a half-written directory. An
    existing directory at the same path (a just-in-time save of a step the periodic strategy
    already saved) is moved aside and restored if the rename fails.
    """
    if not enabled:
        yield checkpoint_dir
        return

    barrier = barrier or (lambda: None)
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    parent = os.path.dirname(checkpoint_dir)
    name = os.path.basename(checkpoint_dir)
    os.makedirs(parent, exist_ok=True)
    staging = os.path.join(parent, f".{name}.staging")
    previous = os.path.join(parent, f".{name}.previous")
    if is_main:
        shutil.rmtree(staging, ignore_errors=True)
        # A previous run that died between the two renames left the old checkpoint aside; put it
        # back before writing, so that failure mode also cannot lose the last good checkpoint.
        if not os.path.exists(checkpoint_dir) and os.path.isdir(previous):
            os.replace(previous, checkpoint_dir)
        shutil.rmtree(previous, ignore_errors=True)
    barrier()
    os.makedirs(staging, exist_ok=True)
    moved_aside = False
    try:
        yield staging
        _fsync(staging)
        barrier()
        if is_main:
            if os.path.exists(checkpoint_dir):
                os.replace(checkpoint_dir, previous)
                moved_aside = True
            try:
                os.replace(staging, checkpoint_dir)
            except Exception:
                if moved_aside:
                    os.replace(previous, checkpoint_dir)
                    moved_aside = False
                raise
        barrier()
    finally:
        if is_main:
            shutil.rmtree(staging, ignore_errors=True)
            if moved_aside:
                shutil.rmtree(previous, ignore_errors=True)


def capture_gradients(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().to("cpu", copy=True)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def restore_gradients(model, gradients: dict[str, torch.Tensor]) -> None:
    missing = []
    for name, parameter in model.named_parameters():
        if name in gradients:
            parameter.grad = gradients[name].to(parameter.device, copy=True)
        elif parameter.requires_grad:
            missing.append(name)
    if missing:
        raise ValueError(
            "The checkpoint holds partially accumulated gradients for a different model: no gradient was stored "
            f"for {len(missing)} trainable parameter(s), starting with {missing[:3]}. Resuming inside a "
            "gradient-accumulation window needs the exact same model."
        )


def read_resume_state_at(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def validate_resume_state(state: dict[str, Any], checkpoint_dir: str) -> None:
    if state.get("format") != RESUME_FORMAT_VERSION:
        raise ValueError(
            f"The training data state in {checkpoint_dir} was written with format {state.get('format')!r}, this "
            f"version of Transformers writes and reads format {RESUME_FORMAT_VERSION}."
        )
    required = ("epoch", "batches_consumed", "dataset_signature", "kind")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(
            f"The training data state in {checkpoint_dir} is incomplete: {missing} missing. Refusing to resume rather "
            "than continue with a partially restored data stream."
        )
    if state["kind"] not in ("map", "streaming"):
        raise ValueError(f"The checkpoint at {checkpoint_dir} records an unknown data cursor kind {state['kind']!r}.")
    if state["epoch"] is not None and (not isinstance(state["epoch"], int) or state["epoch"] < 0):
        raise ValueError(f"Invalid epoch in the checkpoint at {checkpoint_dir}: epoch={state['epoch']!r}.")
    if not isinstance(state["batches_consumed"], int) or state["batches_consumed"] < 0:
        raise ValueError(
            f"Invalid cursor in the checkpoint at {checkpoint_dir}: batches_consumed={state['batches_consumed']!r}."
        )
