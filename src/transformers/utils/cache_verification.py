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
"""Read-only self-verification of the local Hub cache.

The Hub cache layout (managed by `huggingface_hub`) is, per cached repo:

```
<cache_dir>/models--<org>--<name>/
├── refs/<ref>                     # plain text file holding the commit a ref points to
├── snapshots/<commit_hash>/...    # tree of symlinks reproducing the repo structure
├── blobs/<etag>                   # actual file contents; LFS blobs are named after their sha256
└── .no_exist/<commit_hash>/...    # markers for files known not to exist at a commit
```

Everything in this module is **strictly read-only** with respect to the cache: hashing only
opens files, reports only describe what is on disk, and even `cleanup_cached_repo` only removes
whole revisions through `huggingface_hub`'s own cache manager - it never rewrites a blob, a
symlink or a ref. Verification never triggers a download, so it works fully offline.

The content digest is always computed from the actual bytes on disk; the blob name (the Hub
etag, which is the sha256 of the content for LFS files) is used as an *expected* digest, never
trusted on its own.
"""

import hashlib
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE, REPO_TYPES

from .logging import get_logger, tqdm


logger = get_logger(__name__)

# Name of the environment variable that turns verification on for every cache lookup
# (`cached_file`/`cached_files` and every `from_pretrained`-style entry point on top of them),
# without changing any function signature.
CACHE_VERIFICATION_ENV_VAR = "TRANSFORMERS_CACHE_VERIFY"

# 1 MiB streaming reads keep the hashing memory footprint flat regardless of file size.
_HASH_CHUNK_SIZE = 1024 * 1024

# Blobs that are not Git-LFS objects are stored under their Git object hash (a 40-character
# sha1, possibly with a `.lock` suffix during writes). LFS blobs are named after the sha256
# of their content (64 hex characters). Anything else is a regular (non-LFS) file whose blob
# name is not a content digest and therefore cannot be self-verified.
_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


# --------------------------------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------------------------------
class CacheVerificationError(OSError):
    """Base class for cache verification failures.

    All subclasses are also `OSError`s, so existing code that catches `OSError` around
    `from_pretrained` keeps working unchanged when verification is enabled.
    """

    #: machine-readable short code, e.g. `"cache_missing"`
    reason: str = "cache_error"


class CachedRepoNotFoundError(CacheVerificationError):
    """Raised when no revision of the requested repo exists in the local cache."""

    reason = "cache_missing"


class CacheEntryMissingError(CacheVerificationError):
    """Raised when the requested revision/file is not present in the local cache."""

    reason = "cache_missing"


class CacheCorruptedError(CacheVerificationError):
    """Raised when a cached file exists but its content does not match its expected digest."""

    reason = "cache_corrupted"


class CacheIncompleteError(CacheVerificationError):
    """Raised when the cached revision is missing files it is supposed to contain."""

    reason = "cache_incomplete"


class NetworkRequiredOfflineError(CacheVerificationError):
    """Raised in offline mode when the cache is present and intact but does not hold everything requested."""

    reason = "network_required_offline"


class InsufficientDiskSpaceError(CacheVerificationError):
    """Raised when a download would not fit in the free space of the cache directory."""

    reason = "insufficient_disk_space"


class CacheDirectoryNotWritableError(CacheVerificationError):
    """Raised when the cache directory (or one of its parents) cannot be written to."""

    reason = "cache_dir_not_writable"


# --------------------------------------------------------------------------------------------------
# Report data structures
# --------------------------------------------------------------------------------------------------
@dataclass
class CachedFileManifest:
    """Manifest entry for a single file of a cached revision."""

    #: path of the file relative to the repo root, e.g. `"model.safetensors.index.json"`
    file_name: str
    #: sha256 computed from the actual bytes on disk
    sha256: str
    #: file size in bytes (size of the content, after following symlinks)
    size: int
    #: etag recorded by the Hub, i.e. the blob name; `None` when it is not a content digest
    expected_digest: str | None
    #: `True` when `expected_digest` is a content digest that was verified against the bytes
    digest_verified: bool
    #: absolute path of the snapshot entry (a symlink into `blobs/`)
    file_path: str


@dataclass
class CachedRevisionManifest:
    """Manifest of one cached revision (one `snapshots/<commit_hash>` directory)."""

    model_id: str
    repo_type: str
    #: immutable commit hash of the revision
    revision: str
    #: refs (branch/tag names) pointing at this revision, e.g. `["main"]`
    refs: list[str]
    files: list[CachedFileManifest] = field(default_factory=list)
    #: sum of the (deduplicated) blob sizes referenced by this revision, in bytes
    total_size: int = 0
    #: files listed by the snapshot whose blob is missing or dangling
    missing_files: list[str] = field(default_factory=list)
    #: files whose on-disk content does not hash to their expected digest
    corrupted_files: list[str] = field(default_factory=list)
    #: leftover `*.incomplete` blobs from interrupted downloads
    incomplete_files: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return not self.missing_files and not self.corrupted_files and not self.incomplete_files

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CachedRepoManifest:
    """Manifest of every revision cached locally for one repo."""

    model_id: str
    repo_type: str
    cache_path: str
    revisions: list[CachedRevisionManifest]
    #: physical size of the blobs referenced by this repo, in bytes
    total_size: int

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------------------------------
def _expected_digest_from_blob_name(blob_name: str) -> str | None:
    """Return the content digest a blob name encodes, or `None` for non-LFS regular files."""
    if _SHA256_PATTERN.fullmatch(blob_name):
        return blob_name
    if _SHA1_PATTERN.fullmatch(blob_name):
        # Non-LFS Git object: the name is a sha1 of the Git object, not of the raw content,
        # so it cannot be recomputed from the file alone.
        return None
    return None


def hash_file(path: str | os.PathLike, algorithm: str = "sha256", chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Compute the content digest of `path`, streaming it in fixed-size chunks.

    Symlinks are followed, so the hashed bytes are always the actual blob contents. Memory
    usage is one chunk (1 MiB by default), independently of the file size. The function only
    opens the file for reading and never modifies anything.
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _find_cached_repo(model_id: str, cache_dir: str | os.PathLike, repo_type: str = "model"):
    """Return the `CachedRepoInfo` for `model_id`, or `None` if the repo is not cached.

    `scan_cache_dir` drops a whole repo from its report when a single snapshot symlink is
    dangling (it records a `CorruptedCacheException` instead). Since a deleted blob is exactly
    the "incomplete cache" situation this module must report - not a "repo not cached" one -
    a manual, layout-based scan is used as a fallback whenever the repo folder exists on disk
    but the regular scan skipped it.
    """
    cache_dir = str(cache_dir)
    cache_info = scan_cache_dir(cache_dir)
    for repo in cache_info.repos:
        if repo.repo_id == model_id and repo.repo_type == (repo_type or "model"):
            return repo

    from huggingface_hub.file_download import repo_folder_name

    repo_path = Path(cache_dir) / repo_folder_name(repo_id=model_id, repo_type=repo_type or "model")
    if repo_path.is_dir():
        return _walk_repo_manually(model_id, repo_type or "model", repo_path)
    return None


def _walk_repo_manually(model_id: str, repo_type: str, repo_path: Path):
    """Scan a repo straight from its `refs/`/`snapshots/`/`blobs/` layout.

    This fallback tolerates dangling symlinks: they are kept in the revision's file list with a
    `None` blob path so the manifest builder can classify them as missing, instead of dropping
    the whole repository.
    """
    from types import SimpleNamespace

    refs_dir = repo_path / "refs"
    snapshots_dir = repo_path / "snapshots"

    # ref name -> commit hash
    ref_by_commit: dict[str, list[str]] = {}
    if refs_dir.is_dir():
        for ref_file in refs_dir.rglob("*"):
            if ref_file.is_file():
                ref_by_commit.setdefault(ref_file.read_text().strip(), []).append(
                    ref_file.relative_to(refs_dir).as_posix()
                )

    revisions = []
    total_size = 0
    if snapshots_dir.is_dir():
        for snapshot_path in sorted(p for p in snapshots_dir.iterdir() if p.is_dir()):
            files = []
            for entry in sorted(snapshot_path.rglob("*")):
                if entry.is_symlink() or entry.is_file():
                    blob_path = Path(os.path.realpath(entry)) if entry.is_file() else None
                    size = blob_path.stat().st_size if blob_path is not None and os.path.isfile(blob_path) else 0
                    total_size += size
                    files.append(
                        SimpleNamespace(
                            file_name=entry.relative_to(snapshot_path).as_posix(),
                            file_path=entry,
                            blob_path=blob_path,
                            size_on_disk=size,
                        )
                    )
            commit = snapshot_path.name
            revisions.append(
                SimpleNamespace(
                    commit_hash=commit,
                    snapshot_path=snapshot_path,
                    refs=tuple(sorted(ref_by_commit.get(commit, []))),
                    files=tuple(files),
                )
            )

    return SimpleNamespace(
        repo_id=model_id,
        repo_type=repo_type,
        repo_path=repo_path,
        size_on_disk=total_size,
        revisions=tuple(revisions),
        refs={ref: rev for rev in revisions for ref in rev.refs},
    )


def _select_revision(repo, revision: str | None):
    """Pick the cached revision matching `revision`.

    `revision` may be a commit hash or a ref name (`"main"`, a tag, ...). `None` means the
    revision the `main` ref points to, falling back to the only cached revision.
    """
    revisions = list(repo.revisions)
    if revision is not None:
        for rev in revisions:
            if rev.commit_hash == revision or revision in rev.refs:
                return rev
        return None
    refs = repo.refs
    if "main" in refs:
        return refs["main"]
    if len(revisions) == 1:
        return revisions[0]
    return None


def _build_revision_manifest(
    repo,
    revision,
    *,
    verify_digests: bool,
    progress: bool,
    expected_files: set[str] | None = None,
) -> CachedRevisionManifest:
    """Scan one cached revision and hash its files. Read-only: nothing is created or modified."""
    manifest = CachedRevisionManifest(
        model_id=repo.repo_id,
        repo_type=repo.repo_type,
        revision=revision.commit_hash,
        refs=sorted(revision.refs),
    )

    # Sorting the files makes the report (and therefore any test diffing it) deterministic.
    cached_files = sorted(revision.files, key=lambda f: f.file_name)
    iterator = cached_files
    if progress:
        iterator = tqdm(cached_files, desc=f"Verifying {repo.repo_id}@{revision.commit_hash[:10]}")

    seen_files: set[str] = set()
    for cached_file in iterator:
        seen_files.add(cached_file.file_name)
        file_path = Path(cached_file.file_path)
        blob_path = getattr(cached_file, "blob_path", None)
        blob_path = None if blob_path is None else Path(blob_path)

        # A snapshot entry whose symlink target does not exist (or is not a regular file).
        if not file_path.is_file():
            manifest.missing_files.append(cached_file.file_name)
            continue

        actual_digest = hash_file(file_path)
        expected_digest = _expected_digest_from_blob_name(blob_path.name) if blob_path is not None else None
        digest_verified = False
        if verify_digests and expected_digest is not None:
            if actual_digest != expected_digest:
                manifest.corrupted_files.append(cached_file.file_name)
            else:
                digest_verified = True

        manifest.files.append(
            CachedFileManifest(
                file_name=cached_file.file_name,
                sha256=actual_digest,
                size=cached_file.size_on_disk,
                expected_digest=expected_digest,
                digest_verified=digest_verified,
                file_path=str(file_path),
            )
        )

    # Files the caller expects at this revision but that have no snapshot entry at all.
    if expected_files is not None:
        manifest.missing_files.extend(sorted(expected_files - seen_files))

    # Interrupted downloads leave `*.incomplete` blobs that are not part of any snapshot.
    # They live directly in the repo's `blobs/` folder, so a glob is enough and stays read-only.
    incomplete_dir = Path(repo.repo_path) / "blobs"
    if incomplete_dir.is_dir():
        manifest.incomplete_files = sorted(str(p) for p in incomplete_dir.glob("*.incomplete"))

    # Deduplicated size: each blob counts once even if several entries hard-link to it.
    counted_blobs: set[str] = set()
    total_size = 0
    for cached_file in cached_files:
        if getattr(cached_file, "blob_path", None) is None or not Path(cached_file.file_path).is_file():
            continue
        blob = str(Path(cached_file.blob_path).resolve())
        if blob not in counted_blobs:
            counted_blobs.add(blob)
            total_size += cached_file.size_on_disk
    manifest.total_size = total_size
    return manifest


# --------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------
def get_cached_model_manifest(
    model_id: str,
    revision: str | None = None,
    *,
    cache_dir: str | os.PathLike | None = None,
    repo_type: str = "model",
    verify_digests: bool = True,
    progress: bool = False,
) -> CachedRepoManifest:
    """Report what is cached locally for a model, with per-file content digests.

    This never contacts the Hub and never downloads anything. The sha256 of every file is
    computed from the bytes on disk; when the blob is an LFS object its name provides an
    expected digest, which is checked against the recomputed value when `verify_digests=True`.

    Args:
        model_id (`str`):
            Repo id of the model on the Hub, e.g. `"google-bert/bert-base-uncased"`.
        revision (`str`, *optional*):
            Commit hash or ref name. `None` reports every cached revision of the repo.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Cache directory. Defaults to the `huggingface_hub` default (`HF_HUB_CACHE`).
        repo_type (`str`, defaults to `"model"`):
            Type of the cached repository.
        verify_digests (`bool`, defaults to `True`):
            Whether to compare each LFS blob's recomputed sha256 with its expected digest.
            When `False`, digests are still computed and reported but not compared.
        progress (`bool`, defaults to `False`):
            Show a `tqdm` progress bar while hashing - useful for multi-tens-of-GB checkpoints.

    Returns:
        [`CachedRepoManifest`]: A serializable manifest of the cached revisions.

    Raises:
        [`CachedRepoNotFoundError`]: If the repo has no entry at all in the local cache.
        [`CacheEntryMissingError`]: If `revision` is requested but not present in the cache.

    Note:
        This function and `verify_cached_model` never create, delete or modify any cache
        entry. The only place in this module that writes is the optional pre-download
        writability probe (`check_cache_dir_writable`), which creates and immediately removes
        a single temporary probe file without touching blobs, snapshots or refs.
    """
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {list(REPO_TYPES)}")
    cache_dir = HF_HUB_CACHE if cache_dir is None else str(cache_dir)

    repo = _find_cached_repo(model_id, cache_dir, repo_type)
    if repo is None:
        raise CachedRepoNotFoundError(
            f"No local cache entry for {repo_type}/{model_id} in {cache_dir}. "
            "Download the model first or check the cache directory."
        )

    if revision is not None:
        selected = _select_revision(repo, revision)
        if selected is None:
            raise CacheEntryMissingError(
                f"Revision {revision!r} of {model_id} is not cached in {cache_dir}. "
                f"Cached revisions: {sorted(r.commit_hash for r in repo.revisions)}."
            )
        revisions = [selected]
    else:
        revisions = sorted(repo.revisions, key=lambda r: r.commit_hash)

    manifests = [
        _build_revision_manifest(repo, rev, verify_digests=verify_digests, progress=progress) for rev in revisions
    ]
    return CachedRepoManifest(
        model_id=model_id,
        repo_type=repo_type,
        cache_path=str(repo.repo_path),
        revisions=manifests,
        total_size=repo.size_on_disk,
    )


def verify_cached_model(
    model_id: str,
    revision: str | None = None,
    *,
    expected_files: list[str] | set[str] | None = None,
    cache_dir: str | os.PathLike | None = None,
    repo_type: str = "model",
    progress: bool = False,
) -> CachedRevisionManifest:
    """Verify a cached revision without downloading or modifying anything.

    Every LFS blob's sha256 is recomputed from its actual bytes and compared with the digest
    encoded in the blob name; dangling snapshot symlinks and `*.incomplete` leftovers are also
    reported. The function is read-only - it is safe to run concurrently with loads and with a
    read-only cache mount.

    Args:
        model_id (`str`): Repo id of the model on the Hub.
        revision (`str`, *optional*):
            Commit hash or ref name. `None` uses the `main` ref (or the single cached revision).
        expected_files (`list` or `set` of `str`, *optional*):
            File paths (relative to the repo root) the revision must contain. Any missing one
            is reported as an incomplete cache. `None` only checks the files already snapshotted.
        cache_dir (`str` or `os.PathLike`, *optional*): Cache directory.
        repo_type (`str`, defaults to `"model"`): Type of the cached repository.
        progress (`bool`, defaults to `False`): Show a progress bar while hashing.

    Returns:
        [`CachedRevisionManifest`]: The manifest, whose `is_complete` property is `True` iff
        the cache is complete and every verifiable digest matches.

    Raises:
        [`CachedRepoNotFoundError`]: If the repo is not cached at all.
        [`CacheEntryMissingError`]: If the requested revision is not cached.
        [`CacheIncompleteError`]: If expected/snapshot files are missing.
        [`CacheCorruptedError`]: If at least one content digest does not match.
    """
    repo_manifest = get_cached_model_manifest(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        repo_type=repo_type,
        verify_digests=True,
        progress=progress,
    )
    if len(repo_manifest.revisions) != 1:
        # `revision=None` on a repo with several detached revisions: ask the caller to pin one.
        raise CacheEntryMissingError(
            f"{model_id} has {len(repo_manifest.revisions)} cached revisions and no 'main' ref; "
            "pass an explicit revision to verify one of them."
        )
    manifest = repo_manifest.revisions[0]

    if expected_files is not None:
        wanted = {f.lstrip("/") for f in expected_files}
        present = {f.file_name for f in manifest.files}
        missing = sorted(wanted - present - set(manifest.missing_files))
        manifest.missing_files.extend(missing)

    problems: list[str] = []
    if manifest.corrupted_files:
        problems.append(
            "digest mismatch for: "
            + ", ".join(manifest.corrupted_files)
            + " (content on disk does not match the expected sha256; re-download required)"
        )
    if manifest.missing_files:
        problems.append("missing files: " + ", ".join(manifest.missing_files))
    if manifest.incomplete_files:
        problems.append("interrupted downloads: " + ", ".join(manifest.incomplete_files))
    if problems:
        if manifest.corrupted_files:
            raise CacheCorruptedError(f"Cached {model_id}@{manifest.revision} is corrupted. " + " | ".join(problems))
        raise CacheIncompleteError(f"Cached {model_id}@{manifest.revision} is incomplete. " + " | ".join(problems))
    return manifest


def diagnose_offline_failure(
    model_id: str,
    filenames: list[str] | set[str],
    revision: str | None = None,
    *,
    cache_dir: str | os.PathLike | None = None,
    repo_type: str = "model",
) -> CacheVerificationError:
    """Classify an offline load failure into one of three distinguishable causes.

    1. [`CachedRepoNotFoundError`] / [`CacheEntryMissingError`] (`reason="cache_missing"`):
       the repo, revision or requested files are simply not in the cache.
    2. [`CacheCorruptedError`] / [`CacheIncompleteError`] (`reason="cache_corrupted"` /
       `"cache_incomplete"`): the cache has the entry but its bytes are corrupted or incomplete.
    3. [`NetworkRequiredOfflineError`] (`reason="network_required_offline"`): the cache exists
       and is intact, but does not contain everything requested, so serving the request would
       need the Hub while network access is disabled.

    The returned exception is meant to be raised by the caller; this function itself does not
    raise and does not perform any write.
    """
    cache_dir = HF_HUB_CACHE if cache_dir is None else str(cache_dir)
    repo = _find_cached_repo(model_id, cache_dir, repo_type)
    if repo is None:
        return CachedRepoNotFoundError(
            f"Offline mode: {repo_type}/{model_id} is not present in the local cache ({cache_dir}). "
            "Pre-download it on a connected machine, then retry."
        )

    selected = _select_revision(repo, revision)
    if selected is None:
        return CacheEntryMissingError(
            f"Offline mode: revision {revision!r} of {model_id} is not cached. "
            f"Cached revisions: {sorted(r.commit_hash for r in repo.revisions)}."
        )

    manifest = _build_revision_manifest(repo, selected, verify_digests=True, progress=False)
    wanted = {f.lstrip("/") for f in filenames}
    present = {f.file_name for f in manifest.files}
    missing = sorted(wanted - present)

    if manifest.corrupted_files:
        return CacheCorruptedError(
            f"Offline mode: cached {model_id}@{selected.commit_hash} is corrupted "
            f"(digest mismatch: {', '.join(manifest.corrupted_files)}). Re-download on a connected machine."
        )
    if manifest.missing_files or missing:
        all_missing = sorted(set(manifest.missing_files) | set(missing))
        return NetworkRequiredOfflineError(
            f"Offline mode: the intact cache for {model_id}@{selected.commit_hash} does not contain "
            f"{', '.join(all_missing)}. The files exist on the Hub but network access is disabled; "
            "pre-download them (e.g. with snapshot_download) on a connected machine."
        )
    # Everything requested is cached and valid, yet the load failed: genuinely needs the Hub
    # (e.g. a metadata call) while offline.
    return NetworkRequiredOfflineError(
        f"Offline mode: serving {model_id}@{selected.commit_hash} requires Hub access, which is disabled, "
        "although the cached files themselves are complete and valid."
    )


def check_disk_space(
    required_bytes: int,
    cache_dir: str | os.PathLike | None = None,
) -> None:
    """Reject a download up-front when the cache filesystem cannot fit `required_bytes`.

    Raises [`InsufficientDiskSpaceError`] with the numbers, or [`CacheDirectoryNotWritableError`]
    when the target directory is not writable. Read-only: nothing is allocated or written.
    """
    cache_dir = HF_HUB_CACHE if cache_dir is None else str(cache_dir)
    target = Path(cache_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise CacheDirectoryNotWritableError(
            f"Cannot create cache directory {target}: {e}. Check its permissions and ownership."
        ) from e
    if not os.access(target, os.W_OK):
        raise CacheDirectoryNotWritableError(
            f"Cache directory {target} is not writable by the current user "
            f"(uid={os.getuid() if hasattr(os, 'getuid') else '?'})."
        )

    usage = shutil.disk_usage(target)
    if usage.free < required_bytes:
        raise InsufficientDiskSpaceError(
            f"Not enough disk space in {target}: needs {required_bytes} bytes "
            f"({_format_bytes(required_bytes)}) but only {usage.free} bytes "
            f"({_format_bytes(usage.free)}) are free."
        )


def check_cache_dir_writable(cache_dir: str | os.PathLike | None = None) -> None:
    """Verify the cache directory can be written to, naming the offending path on failure.

    A real create/write probe is used instead of `os.access`: on Linux the latter only checks
    permission bits, while file creation can still be denied by read-only mounts, ACLs or
    immutable attributes. The probe file is removed immediately, so the cache content is left
    byte-identical.
    """
    cache_dir = HF_HUB_CACHE if cache_dir is None else str(cache_dir)
    target = Path(cache_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise CacheDirectoryNotWritableError(f"Cache directory {target} cannot be created: {e}.") from e

    probe = target / f".transformers-write-probe-{os.getpid()}"
    try:
        with open(probe, "wb") as f:
            f.write(b"\0")
    except PermissionError as e:
        raise CacheDirectoryNotWritableError(
            f"Cache directory {target} is not writable by the current user: {e}."
        ) from e
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass

    if not os.access(target, os.W_OK):
        raise CacheDirectoryNotWritableError(f"Cache directory {target} is not writable.")


def cleanup_cached_repo(
    model_id: str,
    revision: str | None = None,
    *,
    cache_dir: str | os.PathLike | None = None,
    repo_type: str = "model",
    dry_run: bool = False,
) -> int:
    """Delete cached revisions of a model and report how many bytes would be / were freed.

    With `revision=None`, every cached revision of the repo is removed. Deletion is delegated
    to `huggingface_hub`'s cache manager (`DeleteCacheStrategy`), which only removes blobs,
    snapshots and refs belonging to the selected revisions - the cache layout and lookup
    priority of everything else are untouched.

    Returns:
        `int`: The number of bytes freed (or that would be freed with `dry_run=True`).
    """
    cache_dir = HF_HUB_CACHE if cache_dir is None else str(cache_dir)
    repo = _find_cached_repo(model_id, cache_dir, repo_type)
    if repo is None:
        raise CachedRepoNotFoundError(f"No local cache entry for {repo_type}/{model_id} in {cache_dir}.")

    if revision is not None:
        selected = _select_revision(repo, revision)
        if selected is None:
            raise CacheEntryMissingError(f"Revision {revision!r} of {model_id} is not cached; nothing to clean.")
        commit_hashes = [selected.commit_hash]
    else:
        commit_hashes = [rev.commit_hash for rev in repo.revisions]

    cache_info = scan_cache_dir(str(cache_dir))
    strategy = cache_info.delete_revisions(*commit_hashes)
    freed = strategy.expected_freed_size
    if not dry_run:
        strategy.execute()
        logger.info(
            "Removed %d revision(s) of %s/%s from %s (%s freed).",
            len(commit_hashes),
            repo_type,
            model_id,
            cache_dir,
            _format_bytes(freed),
        )
    return freed


def _format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{num_bytes} B"


__all__ = [
    "CACHE_VERIFICATION_ENV_VAR",
    "CacheCorruptedError",
    "CacheDirectoryNotWritableError",
    "CachedFileManifest",
    "CachedRepoManifest",
    "CachedRepoNotFoundError",
    "CacheEntryMissingError",
    "CacheIncompleteError",
    "CacheVerificationError",
    "InsufficientDiskSpaceError",
    "NetworkRequiredOfflineError",
    "cleanup_cached_repo",
    "diagnose_offline_failure",
    "get_cached_model_manifest",
    "hash_file",
    "verify_cached_model",
    "check_disk_space",
    "check_cache_dir_writable",
]
