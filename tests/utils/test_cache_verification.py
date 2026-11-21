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
"""Tests for read-only Hub cache self-verification (`transformers.utils.cache_verification`).

All fixtures are hand-built on disk with the exact `huggingface_hub` cache layout
(blobs / snapshots symlinks / refs); no test here touches the network.
"""

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from huggingface_hub.errors import LocalEntryNotFoundError

from transformers.utils import cached_file
from transformers.utils.cache_verification import (
    CACHE_VERIFICATION_ENV_VAR,
    CacheCorruptedError,
    CacheDirectoryNotWritableError,
    CachedRepoNotFoundError,
    CacheEntryMissingError,
    CacheIncompleteError,
    InsufficientDiskSpaceError,
    NetworkRequiredOfflineError,
    check_disk_space,
    cleanup_cached_repo,
    diagnose_offline_failure,
    get_cached_model_manifest,
    hash_file,
    verify_cached_model,
)


MODEL_ID = "hf-internal-testing/tiny-verifiable-model"
REPO_DIR_NAME = "models--hf-internal-testing--tiny-verifiable-model"
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40

FILES = {
    "config.json": b'{"hidden_size": 32, "model_type": "bert"}',
    "model.safetensors": b"weights-payload-0" * 100,
    "tokenizer.json": b'{"type": "BertTokenizer"}',
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _force_remove(tmp: tempfile.TemporaryDirectory) -> None:
    """Remove a temp dir even when tests left read-only files or dangling symlinks."""

    def _force_writable(func, path, exc_info):
        os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
        func(path)

    rmtree_kwargs = {"onexc" if sys.version_info >= (3, 12) else "onerror": _force_writable}
    shutil.rmtree(tmp.name, **rmtree_kwargs)


def build_cached_repo(
    cache_dir: str,
    *,
    files: dict[str, bytes] | None = None,
    commit: str = COMMIT_A,
    refs: list[str] | None = None,
    corrupt: set[str] | None = None,
    missing: set[str] | None = None,
    extra_incomplete: bool = False,
    reset: bool = False,
) -> str:
    """Create a valid-looking Hub cache entry on disk and return its repo directory."""
    files = dict(FILES if files is None else files)
    corrupt = corrupt or set()
    missing = missing or set()
    refs = ["main"] if refs is None else refs

    repo_dir = Path(cache_dir) / REPO_DIR_NAME
    if reset and repo_dir.exists():
        shutil.rmtree(repo_dir)
    blobs_dir = repo_dir / "blobs"
    snap_dir = repo_dir / "snapshots" / commit
    refs_dir = repo_dir / "refs"
    for d in (blobs_dir, snap_dir, refs_dir):
        d.mkdir(parents=True, exist_ok=True)

    for name, content in files.items():
        if name in missing:
            continue
        blob_name = _sha256(content)
        blob_path = blobs_dir / blob_name
        # Write the real bytes; corruption tests overwrite through the blob afterwards.
        if not blob_path.exists():
            blob_path.write_bytes(content)
        target = snap_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(os.path.relpath(blob_path, target.parent))

    for name in corrupt:
        blob_target = (snap_dir / name).resolve()
        blob_target.write_bytes(b"corrupted-bytes-not-matching-the-digest")

    if extra_incomplete:
        (blobs_dir / f"{_sha256(b'partial')}.incomplete").write_bytes(b"partial")

    for ref in refs:
        ref_path = refs_dir / ref
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_path.write_text(commit)

    return str(repo_dir)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name
        build_cached_repo(self.cache_dir)

    def tearDown(self):
        _force_remove(self._tmp)

    def test_manifest_reports_identity_revision_files_digests_and_size(self):
        manifest = get_cached_model_manifest(MODEL_ID, cache_dir=self.cache_dir)

        self.assertEqual(manifest.model_id, MODEL_ID)
        self.assertEqual(manifest.repo_type, "model")
        self.assertEqual(len(manifest.revisions), 1)
        revision = manifest.revisions[0]
        self.assertEqual(revision.revision, COMMIT_A)
        self.assertEqual(revision.refs, ["main"])
        self.assertEqual(sorted(f.file_name for f in revision.files), sorted(FILES.keys()))
        # The reported digest really comes from the bytes, not from metadata.
        for entry in revision.files:
            self.assertEqual(entry.sha256, hash_file(entry.file_path))
            self.assertEqual(entry.sha256, _sha256(FILES[entry.file_name]))
            self.assertEqual(entry.expected_digest, entry.sha256)
            self.assertTrue(entry.digest_verified)
            self.assertEqual(entry.size, len(FILES[entry.file_name]))
        expected_total = sum(len(v) for v in FILES.values())
        self.assertEqual(revision.total_size, expected_total)
        self.assertEqual(manifest.total_size, expected_total)
        self.assertTrue(revision.is_complete)

    def test_manifest_is_json_serializable(self):
        manifest = get_cached_model_manifest(MODEL_ID, cache_dir=self.cache_dir)
        payload = json.dumps(manifest.to_dict())
        decoded = json.loads(payload)
        self.assertEqual(decoded["revisions"][0]["revision"], COMMIT_A)
        self.assertEqual(len(decoded["revisions"][0]["files"]), len(FILES))

    def test_repo_not_cached_raises(self):
        with self.assertRaises(CachedRepoNotFoundError) as ctx:
            get_cached_model_manifest("org/never-downloaded", cache_dir=self.cache_dir)
        self.assertEqual(ctx.exception.reason, "cache_missing")

    def test_revision_not_cached_raises(self):
        with self.assertRaises(CacheEntryMissingError) as ctx:
            get_cached_model_manifest(MODEL_ID, revision=COMMIT_B, cache_dir=self.cache_dir)
        self.assertEqual(ctx.exception.reason, "cache_missing")


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name

    def tearDown(self):
        _force_remove(self._tmp)

    def test_complete_cache_verifies(self):
        build_cached_repo(self.cache_dir)
        manifest = verify_cached_model(MODEL_ID, cache_dir=self.cache_dir)
        self.assertTrue(manifest.is_complete)
        self.assertEqual(manifest.missing_files, [])
        self.assertEqual(manifest.corrupted_files, [])

    def test_missing_file_is_reported_as_incomplete(self):
        build_cached_repo(self.cache_dir, missing={"tokenizer.json"}, reset=True)
        with self.assertRaises(CacheIncompleteError) as ctx:
            verify_cached_model(
                MODEL_ID,
                cache_dir=self.cache_dir,
                expected_files=list(FILES.keys()),
            )
        self.assertEqual(ctx.exception.reason, "cache_incomplete")
        self.assertIn("tokenizer.json", str(ctx.exception))

    def test_digest_mismatch_is_reported_as_corrupted(self):
        build_cached_repo(self.cache_dir, corrupt={"model.safetensors"}, reset=True)
        with self.assertRaises(CacheCorruptedError) as ctx:
            verify_cached_model(MODEL_ID, cache_dir=self.cache_dir)
        self.assertEqual(ctx.exception.reason, "cache_corrupted")
        self.assertIn("model.safetensors", str(ctx.exception))

    def test_incomplete_download_leftover_is_reported(self):
        build_cached_repo(self.cache_dir, extra_incomplete=True, reset=True)
        with self.assertRaises(CacheIncompleteError) as ctx:
            verify_cached_model(MODEL_ID, cache_dir=self.cache_dir)
        self.assertIn(".incomplete", str(ctx.exception))

    def test_deleted_blob_dangling_symlink_is_reported_as_incomplete_not_missing_repo(self):
        # `scan_cache_dir` drops a repo entirely when one symlink is dangling; the manual
        # fallback must still locate the repo and name the missing file.
        repo_dir = Path(build_cached_repo(self.cache_dir))
        deleted_blob = next((repo_dir / "blobs").iterdir())
        deleted_blob.unlink()

        manifest = get_cached_model_manifest(MODEL_ID, cache_dir=self.cache_dir)
        revision = manifest.revisions[0]
        self.assertEqual(len(revision.files), len(FILES) - 1)
        self.assertTrue(revision.missing_files)

        with self.assertRaises(CacheIncompleteError):
            verify_cached_model(MODEL_ID, cache_dir=self.cache_dir)

        error = diagnose_offline_failure(MODEL_ID, list(FILES), cache_dir=self.cache_dir)
        # Not CachedRepoNotFoundError: the repo is there, only a blob is gone.
        self.assertNotIsInstance(error, CachedRepoNotFoundError)
        self.assertEqual(error.reason, "network_required_offline")

    def test_verification_is_strictly_read_only(self):
        repo_dir = build_cached_repo(self.cache_dir)
        before = {}
        for root, dirs, files in os.walk(repo_dir):
            for name in files + dirs:
                p = os.path.join(root, name)
                before[p] = (os.lstat(p).st_mode, os.lstat(p).st_mtime_ns, os.lstat(p).st_size)

        manifest = verify_cached_model(MODEL_ID, cache_dir=self.cache_dir)
        # Also exercise the reporting path, which must be read-only as well.
        get_cached_model_manifest(MODEL_ID, cache_dir=self.cache_dir)
        self.assertTrue(manifest.is_complete)

        after = {}
        for root, dirs, files in os.walk(repo_dir):
            for name in files + dirs:
                p = os.path.join(root, name)
                after[p] = (os.lstat(p).st_mode, os.lstat(p).st_mtime_ns, os.lstat(p).st_size)
        self.assertEqual(before, after)
        # And no new entry appeared anywhere in the cache.
        self.assertEqual(set(before), set(after))

    def test_progress_callback_is_invoked_per_file(self):
        build_cached_repo(self.cache_dir)
        seen: list[str] = []

        class FakeTqdm:
            def __init__(self, iterable, **kwargs):
                self.iterable = iterable

            def __iter__(self):
                for item in self.iterable:
                    seen.append(item.file_name)
                    yield item

        with mock.patch("transformers.utils.cache_verification.tqdm", FakeTqdm):
            verify_cached_model(MODEL_ID, cache_dir=self.cache_dir, progress=True)
        self.assertEqual(sorted(seen), sorted(FILES.keys()))


class OfflineDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name

    def tearDown(self):
        _force_remove(self._tmp)

    def test_three_offline_causes_are_distinguishable(self):
        # 1. Cache missing altogether.
        error = diagnose_offline_failure(MODEL_ID, ["config.json"], cache_dir=self.cache_dir)
        self.assertIsInstance(error, CachedRepoNotFoundError)
        self.assertEqual(error.reason, "cache_missing")

        # 2a. Cache present but corrupted.
        build_cached_repo(self.cache_dir, corrupt={"model.safetensors"}, reset=True)
        error = diagnose_offline_failure(MODEL_ID, list(FILES), cache_dir=self.cache_dir)
        self.assertIsInstance(error, CacheCorruptedError)
        self.assertEqual(error.reason, "cache_corrupted")
        shutil.rmtree(Path(self.cache_dir) / REPO_DIR_NAME)

        # 2b. Cache present but incomplete (requested file absent).
        build_cached_repo(self.cache_dir, missing={"tokenizer.json"}, reset=True)
        error = diagnose_offline_failure(MODEL_ID, list(FILES), cache_dir=self.cache_dir)
        self.assertIsInstance(error, NetworkRequiredOfflineError)
        self.assertEqual(error.reason, "network_required_offline")
        self.assertIn("tokenizer.json", str(error))

        # 3. Intact cache, request satisfiable -> still classified as network-required,
        #    i.e. the failure is not a data problem.
        shutil.rmtree(Path(self.cache_dir) / REPO_DIR_NAME)
        build_cached_repo(self.cache_dir)
        error = diagnose_offline_failure(MODEL_ID, ["config.json"], cache_dir=self.cache_dir)
        self.assertIsInstance(error, NetworkRequiredOfflineError)
        self.assertEqual(error.reason, "network_required_offline")

    def test_offline_missing_revision_is_cache_missing(self):
        build_cached_repo(self.cache_dir)
        error = diagnose_offline_failure(MODEL_ID, ["config.json"], revision=COMMIT_B, cache_dir=self.cache_dir)
        self.assertEqual(error.reason, "cache_missing")


class MultipleRevisionsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name

    def tearDown(self):
        _force_remove(self._tmp)

    def test_two_revisions_coexist_and_are_distinguished(self):
        files_b = dict(FILES)
        files_b["config.json"] = b'{"hidden_size": 64, "model_type": "bert"}'
        build_cached_repo(self.cache_dir, commit=COMMIT_A, refs=["main"])
        build_cached_repo(self.cache_dir, files=files_b, commit=COMMIT_B, refs=[])

        manifest = get_cached_model_manifest(MODEL_ID, cache_dir=self.cache_dir)
        revisions = {r.revision: r for r in manifest.revisions}
        self.assertEqual(set(revisions), {COMMIT_A, COMMIT_B})

        # `main` resolves to commit A; a direct commit hash selects B.
        main_manifest = verify_cached_model(MODEL_ID, revision="main", cache_dir=self.cache_dir)
        self.assertEqual(main_manifest.revision, COMMIT_A)
        b_manifest = verify_cached_model(MODEL_ID, revision=COMMIT_B, cache_dir=self.cache_dir)
        self.assertEqual(b_manifest.revision, COMMIT_B)
        self.assertNotEqual(
            next(f.sha256 for f in main_manifest.files if f.file_name == "config.json"),
            next(f.sha256 for f in b_manifest.files if f.file_name == "config.json"),
        )


class CachedFilesIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name
        build_cached_repo(self.cache_dir)

    def tearDown(self):
        _force_remove(self._tmp)

    def _offline_kwargs(self):
        return {"cache_dir": self.cache_dir, "local_files_only": True, "revision": COMMIT_A}

    def test_switch_off_keeps_legacy_behavior(self):
        # No verify_cache, no env var: a corrupted blob is returned like any regular file.
        build_cached_repo(self.cache_dir, corrupt={"model.safetensors"}, reset=True)
        path = cached_file(MODEL_ID, "model.safetensors", verify_cache=False, **self._offline_kwargs())
        self.assertTrue(os.path.isfile(path))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"corrupted-bytes-not-matching-the-digest")

    def test_env_var_enables_verification_without_code_change(self):
        build_cached_repo(self.cache_dir, corrupt={"model.safetensors"}, reset=True)
        with mock.patch.dict(os.environ, {CACHE_VERIFICATION_ENV_VAR: "1"}):
            with self.assertRaises(CacheCorruptedError):
                cached_file(MODEL_ID, "model.safetensors", **self._offline_kwargs())

    def test_explicit_verify_detects_corruption_on_cache_hit(self):
        build_cached_repo(self.cache_dir, corrupt={"config.json"}, reset=True)
        with self.assertRaises(CacheCorruptedError) as ctx:
            cached_file(
                MODEL_ID,
                "config.json",
                verify_cache=True,
                **self._offline_kwargs(),
            )
        self.assertIn("config.json", str(ctx.exception))

    def test_verify_accepts_intact_cache_and_returns_same_path(self):
        path_legacy = cached_file(MODEL_ID, "config.json", **self._offline_kwargs())
        path_verified = cached_file(MODEL_ID, "config.json", verify_cache=True, **self._offline_kwargs())
        self.assertEqual(path_legacy, path_verified)

    def test_offline_missing_cache_raises_classified_error(self):
        with self.assertRaises(CachedRepoNotFoundError) as ctx:
            cached_file(
                "org/absent-model",
                "config.json",
                verify_cache=True,
                cache_dir=self.cache_dir,
                local_files_only=True,
                revision=COMMIT_A,
            )
        self.assertEqual(ctx.exception.reason, "cache_missing")

    def test_offline_missing_file_raises_network_required(self):
        build_cached_repo(self.cache_dir, missing={"tokenizer.json"}, reset=True)
        with self.assertRaises(NetworkRequiredOfflineError) as ctx:
            cached_file(
                MODEL_ID,
                "tokenizer.json",
                verify_cache=True,
                cache_dir=self.cache_dir,
                local_files_only=True,
                revision=COMMIT_A,
            )
        self.assertEqual(ctx.exception.reason, "network_required_offline")

    def test_verification_hook_runs_inside_finalize_without_download_attempt(self):
        # The pinned-commit fast path returns before any download call; verification must
        # still run there. Patch the downloader to prove it is never reached offline.
        with mock.patch("transformers.utils.hub.hf_hub_download", side_effect=AssertionError("must not download")):
            path = cached_file(
                MODEL_ID,
                "config.json",
                verify_cache=True,
                cache_dir=self.cache_dir,
                local_files_only=True,
                revision=COMMIT_A,
            )
        self.assertTrue(os.path.isfile(path))

    def test_local_entry_not_found_path_uses_diagnosis(self):
        # Force the exception branch even on a populated cache: hub download raises, the
        # recovery lookup fails for an un-cached file, and the classifier takes over.
        with mock.patch(
            "transformers.utils.hub.hf_hub_download",
            side_effect=LocalEntryNotFoundError("offline"),
        ):
            with self.assertRaises(NetworkRequiredOfflineError) as ctx:
                cached_file(
                    MODEL_ID,
                    "not-yet-cached.bin",
                    verify_cache=True,
                    cache_dir=self.cache_dir,
                    local_files_only=True,
                    revision=COMMIT_A,
                )
        self.assertEqual(ctx.exception.reason, "network_required_offline")


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name

    def tearDown(self):
        _force_remove(self._tmp)

    def test_insufficient_disk_space_is_rejected_before_download(self):
        huge = shutil.disk_usage(self.cache_dir).free + 1024**3
        with self.assertRaises(InsufficientDiskSpaceError) as ctx:
            check_disk_space(huge, self.cache_dir)
        self.assertEqual(ctx.exception.reason, "insufficient_disk_space")
        # Rejected up-front: nothing was created in the cache.
        self.assertEqual(os.listdir(self.cache_dir), [])

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root bypasses permission bits")
    def test_unwritable_cache_directory_is_named(self):
        os.chmod(self.cache_dir, stat.S_IRUSR | stat.S_IXUSR)
        with self.assertRaises(CacheDirectoryNotWritableError) as ctx:
            check_disk_space(1, self.cache_dir)
        self.assertEqual(ctx.exception.reason, "cache_dir_not_writable")
        self.assertIn(self.cache_dir, str(ctx.exception))

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root bypasses permission bits")
    def test_preflight_failure_does_not_change_cache(self):
        build_cached_repo(self.cache_dir)
        repo_dir = Path(self.cache_dir) / REPO_DIR_NAME
        snapshot_before = sorted(str(p.relative_to(repo_dir)) for p in repo_dir.rglob("*"))
        os.chmod(self.cache_dir, stat.S_IRUSR | stat.S_IXUSR)
        # Requesting a file that is not cached forces the download path, where the writability
        # preflight runs before any Hub call is attempted.
        with self.assertRaises(CacheDirectoryNotWritableError):
            cached_file(
                MODEL_ID,
                "not-yet-cached.bin",
                verify_cache=True,
                cache_dir=self.cache_dir,
                local_files_only=True,
                revision=COMMIT_A,
            )
        os.chmod(self.cache_dir, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
        snapshot_after = sorted(str(p.relative_to(repo_dir)) for p in repo_dir.rglob("*"))
        self.assertEqual(snapshot_before, snapshot_after)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = self._tmp.name

    def tearDown(self):
        _force_remove(self._tmp)

    def test_dry_run_frees_nothing(self):
        build_cached_repo(self.cache_dir)
        repo_dir = Path(self.cache_dir) / REPO_DIR_NAME
        freed = cleanup_cached_repo(MODEL_ID, cache_dir=self.cache_dir, dry_run=True)
        self.assertGreater(freed, 0)
        self.assertTrue(repo_dir.exists())

    def test_cleanup_removes_revision_and_keeps_cache_layout(self):
        files_b = dict(FILES)
        files_b["config.json"] = b'{"hidden_size": 64}'
        build_cached_repo(self.cache_dir, commit=COMMIT_A, refs=["main"])
        build_cached_repo(self.cache_dir, files=files_b, commit=COMMIT_B, refs=[])

        cleanup_cached_repo(MODEL_ID, revision=COMMIT_B, cache_dir=self.cache_dir)
        manifest = get_cached_model_manifest(MODEL_ID, cache_dir=self.cache_dir)
        self.assertEqual([r.revision for r in manifest.revisions], [COMMIT_A])
        # Layout conventions are untouched.
        for subfolder in ["blobs", "refs", "snapshots"]:
            self.assertTrue((Path(self.cache_dir) / REPO_DIR_NAME / subfolder).is_dir())

    def test_cleanup_unknown_repo_raises(self):
        with self.assertRaises(CachedRepoNotFoundError):
            cleanup_cached_repo("org/absent", cache_dir=self.cache_dir)


if __name__ == "__main__":
    unittest.main()
