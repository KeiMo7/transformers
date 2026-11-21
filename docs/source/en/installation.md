<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Installation

Transformers works with [PyTorch](https://pytorch.org/get-started/locally/). It has been tested on Python 3.11+ and PyTorch 2.6+.

## Virtual environment

[uv](https://docs.astral.sh/uv/) is an extremely fast Rust-based Python package and project manager and requires a [virtual environment](https://docs.astral.sh/uv/pip/environments/) by default to manage different projects and avoids compatibility issues between dependencies.

It can be used as a drop-in replacement for [pip](https://pip.pypa.io/en/stable/), but if you prefer to use pip, remove `uv` from the commands below.

> [!TIP]
> Refer to the uv [installation](https://docs.astral.sh/uv/guides/install-python/) docs to install uv.

Create a virtual environment to install Transformers in.

```bash
uv venv .env
source .env/bin/activate
```

## Python

Install Transformers with the following command.

[uv](https://docs.astral.sh/uv/) is a fast Rust-based Python package and project manager.


<hfoptions id="installation">
<hfoption id="CUDA">

To install Transformers with PyTorch for NVIDIA GPU (CUDA), install the appropriate CUDA drivers for [PyTorch](https://pytorch.org/get-started/locally).

Run the command below to check if your system detects an NVIDIA GPU.

```bash
nvidia-smi
```

```bash
uv pip install "transformers[torch]"
```

</hfoption>
<hfoption id="NVIDIA Spark (ARM64)">

To install Transformers with PyTorch on NVIDIA Spark devices (such as an RTX Spark laptop) running ARM64, install PyTorch from the NVIDIA PyPI index. These devices require NVIDIA's ARM64 builds of PyTorch, which are not available on the default PyPI index or the standard PyTorch wheel index.

Run the command below to check if your system detects an NVIDIA GPU.

```bash
nvidia-smi
```

Install PyTorch from the NVIDIA PyPI index, then install Transformers.

```bash
uv pip install torch --index-url https://pypi.nvidia.com
uv pip install transformers
```

</hfoption>
<hfoption id="CPU">

To install a CPU-only version of Transformers, run the following command.

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers
```

</hfoption>
<hfoption id="Intel GPU (XPU)">

To install Transformers with PyTorch for Intel GPU (XPU), install the appropriate [Intel GPU (XPU) drivers for PyTorch](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-13.html) and add the Intel GPU (XPU) PyTorch index URL.

After installing the drivers, run the command below to [check if your system detects an Intel GPU](https://dgpu-docs.intel.com/driver/verification.html).

```bash
xpu-smi
```

```bash
uv pip install "transformers[torch]" --extra-index-url https://download.pytorch.org/whl/xpu
```

</hfoption>
</hfoptions>

Test whether the install was successful with the following command. It should return a label and score for the provided text.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Source install

Installing from source installs the *latest* version rather than the *stable* version of the library. It ensures you have the most up-to-date changes in Transformers and it's useful for experimenting with the latest features or fixing a bug that hasn't been officially released in the stable version yet.

The downside is that the latest version may not always be stable. If you encounter any problems, please open a [GitHub Issue](https://github.com/huggingface/transformers/issues) so we can fix it as soon as possible.

Install from source with the following command.

```bash
uv pip install git+https://github.com/huggingface/transformers
```

Check if the install was successful with the command below. It should return a label and score for the provided text.

```bash
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('hugging face is the best'))"
[{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

### Editable install

An [editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) is useful if you're developing locally with Transformers. It links your local copy of Transformers to the Transformers [repository](https://github.com/huggingface/transformers) instead of copying the files. The files are added to Python's import path.

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
uv pip install -e .
```

> [!WARNING]
> You must keep the local Transformers folder to keep using it.

Update your local version of Transformers with the latest changes in the main repository with the following command.

```bash
cd ~/transformers/
git pull
```

## conda

[conda](https://docs.conda.io/projects/conda/en/stable/#) is a language-agnostic package manager. Install Transformers from the [conda-forge](https://anaconda.org/conda-forge/transformers) channel in your newly created virtual environment.

```bash
conda install conda-forge::transformers
```

## Set up

After installation, you can configure the Transformers cache location or set up the library for offline usage.

### Cache directory

When you load a pretrained model with [`~PreTrainedModel.from_pretrained`], the model is downloaded from the Hub and locally cached.

If you pass a commit hash, Transformers uses the local cache for that commit's files and does not re-check the Hub for each one (including files that are known to be missing). This is the default behavior when no revision is specified.

If you pass a branch or tag, Transformers contacts the Hub once at the start of the load to pick the commit, then uses the cache the same way for that commit's files.

The default directory given by the shell environment variable `HF_HUB_CACHE` is `~/.cache/huggingface/hub`. On Windows, the default directory is `C:\Users\username\.cache\huggingface\hub`.

Cache a model in a different directory by changing the path in the following shell environment variables (listed by priority).

1. [HF_HUB_CACHE](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache) (default)
2. [HF_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#hfhome)
3. [XDG_CACHE_HOME](https://hf.co/docs/huggingface_hub/package_reference/environment_variables#xdgcachehome) + `/huggingface` (only if `HF_HOME` is not set)

### Offline mode

To use Transformers in an offline or firewalled environment requires the downloaded and cached files ahead of time. Download a model repository from the Hub with the [`~huggingface_hub.snapshot_download`] method.

> [!TIP]
> Refer to the [Download files from the Hub](https://hf.co/docs/huggingface_hub/guides/download) guide for more options for downloading files from the Hub. You can download files from specific revisions, download from the CLI, and even filter which files to download from a repository.

```py
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")
```

Set the environment variable `HF_HUB_OFFLINE=1` to prevent HTTP calls to the Hub when loading a model.

Another option for only loading cached files is to set `local_files_only=True` in [`~PreTrainedModel.from_pretrained`].

```py
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("./path/to/local/directory", local_files_only=True)
```

Offline mode (or `local_files_only=True`) can still turn a branch or tag into a commit if an earlier online load saved that mapping in the cache. If the mapping was never saved, Transformers keeps the branch or tag you asked for and continues with the regular offline load. You get the same cache hits or missing-file errors as a normal offline load.

### Verify the local cache

A cache entry can be incomplete (an interrupted download, a deleted blob) or corrupted (bytes on disk that no longer match what the Hub recorded). Transformers ships optional, **read-only** cache self-verification so a model cache can prove its own completeness and integrity. It is off by default; nothing changes until you opt in with either the `verify_cache=True` argument or the `TRANSFORMERS_CACHE_VERIFY=1` environment variable (an explicit argument always wins).

Verification recomputes the sha256 of every cached file from the actual bytes on disk and, for Git-LFS files, compares it with the digest encoded in the blob name. Metadata alone is never trusted. Hashing streams the file in 1 MiB chunks, so memory stays flat even for multi-tens-of-GB checkpoints; pass `verify_progress=True` for a progress bar, and interrupt at any time with `Ctrl+C` (no cache entry is modified).

Inspect what is cached, without any network access:

```py
from transformers.utils import get_cached_model_manifest

manifest = get_cached_model_manifest("google-bert/bert-base-uncased")
for revision in manifest.revisions:
    print(revision.revision, revision.refs, revision.total_size, revision.is_complete)
    for f in revision.files:
        print(f"  {f.file_name}  sha256={f.sha256}  verified={f.digest_verified}")
```

The manifest reports the model identifier, the immutable commit hash and refs (so different revisions of the same model coexist and the one actually used is always explicit), the file list with per-file digests and sizes, and the total size. It is JSON-serializable through `manifest.to_dict()`.

Verify a revision (read-only; never downloads):

```py
from transformers.utils import verify_cached_model

# Raises CacheIncompleteError / CacheCorruptedError, or returns the manifest when intact.
manifest = verify_cached_model(
    "google-bert/bert-base-uncased",
    expected_files=["config.json", "model.safetensors"],
    progress=True,
)
```

When verification is enabled on a load, an offline failure is classified into one of three distinguishable causes instead of a single "couldn't find the files" message:

| `exception.reason` | Meaning |
| --- | --- |
| `cache_missing` | The repo, revision or file is not in the local cache. |
| `cache_corrupted` / `cache_incomplete` | The entry exists but its bytes are corrupted or files are missing. |
| `network_required_offline` | The cache is present and intact but does not hold everything requested; the Hub would be needed and network access is disabled. |

```py
from transformers import cached_file
from transformers.utils import CacheVerificationError

try:
    cached_file("google-bert/bert-base-uncased", "model.safetensors", verify_cache=True, local_files_only=True)
except CacheVerificationError as e:
    print(e.reason)  # 'cache_missing' | 'cache_corrupted' | 'cache_incomplete' | 'network_required_offline'
```

Before a download, verification also pre-checks the cache directory: `InsufficientDiskSpaceError` names the required versus free bytes up front, and `CacheDirectoryNotWritableError` names the directory that cannot be written to. Partially downloaded files are kept on a network interruption (the `*.incomplete` blobs are reported by verification), and resuming re-validates the bytes already on disk.

Clean up a model's cache by revision; `dry_run=True` only reports the bytes that would be freed:

```py
from transformers.utils import cleanup_cached_repo

freed = cleanup_cached_repo("google-bert/bert-base-uncased", dry_run=True)
cleanup_cached_repo("google-bert/bert-base-uncased", revision="<commit-hash>")
```

Known limitations:

- Non-LFS (regular Git) files are stored under their Git object sha1, which is not a digest of the raw file content, so their integrity cannot be self-proved from the cache alone; they are still listed and hashed, with `digest_verified=False`. Git-LFS files (model weights, tokenizers, most large artifacts) are fully verified.
- The disk-space pre-check relies on Hub file metadata; in offline mode the size is unknown so only the writability check runs.
- Verification is per-file and deterministic; running it concurrently with loads or from several processes is safe because it only opens files for reading. The optional writability probe creates and immediately deletes one tiny temporary file, only when a download path is actually entered (never on a pure cache hit).
- Caches produced by older `huggingface_hub` versions use the same `blobs`/`snapshots`/`refs` layout, so reporting and read-only verification work on them as-is; only the new opt-in error classification and pre-download checks are added by this version, and older dictionaries/models/adapters keep loading exactly as before when the switch is off.
