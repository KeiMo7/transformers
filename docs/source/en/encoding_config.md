<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Encoding configuration

A model's inputs depend on more than the vocabulary file. Case folding, accent stripping, how Chinese characters and
whitespace are handled, which special tokens are inserted and at which ids, how sequences are truncated and padded, and
the version of the backend that serialized the vocabulary all change the ids that reach the model. Two checkpoints can
therefore produce different tokens for the same sentence while their `tokenizer_config.json` files look identical.

`export_encoding_config` makes those inputs explicit. It produces a small JSON document describing everything that
determines how text is encoded, addressed by a content fingerprint instead of a model name or version string. The same
document can be compared between a training pipeline and an inference service, saved next to the model, and used to
rebuild the encoding component.

This page covers:

- [what the exported config contains](#what-the-config-contains)
- [exporting, comparing and verifying](#export-compare-and-verify)
- [saving with the model](#save-with-the-model)
- [rebuilding a tokenizer from a config](#rebuild-a-tokenizer-from-a-config)
- [what changes and what does not](#compatibility)
- [known limitations](#known-limitations)

## What the config contains

[`PreTrainedTokenizerBase.export_encoding_config`] records:

| Section | What it captures |
|---|---|
| `vocabulary.fingerprint` | A SHA-256 digest of the `(token, id)` pairs, computed in chunks so a change can be traced to an id range. Content-based: renaming a checkpoint or bumping a version does not change it. |
| `encoding.special_tokens` | Every named special token with its **explicit id** and whether that id really comes from the base vocabulary. Nothing is inferred from defaults. |
| `encoding.added_tokens` / `encoding.extra_special_tokens` | Added tokens, including their `lstrip` / `rstrip` / `normalized` / `special` flags, since those change matching. |
| `encoding.pipeline` | The normalization, pre-tokenization, post-processing and decoding pipeline. For fast tokenizers this is the serialized backend pipeline; for SentencePiece tokenizers it is the model proto digest; for custom backends it is the encoding attributes. |
| `encoding.attributes` | Scalar encoding settings (case folding, accent stripping, Chinese handling, `add_prefix_space`, ...). |
| `encoding.truncation` / `encoding.padding` | The truncation/padding strategy persisted on the tokenizer, plus `model_max_length`. |
| `encoding.flags` | `model_input_names`, `split_special_tokens`, `clean_up_tokenization_spaces`, `add_bos_token`, `add_eos_token`. |
| `sources` | Digest of each vocabulary file that was actually read. A replaced file with the same name is visible here even if nothing else differs. |
| `fingerprints.config` | The digest of all of the above, excluding provenance (class name, module path, rebuild recipe). |

The fingerprint deliberately hashes **content**. `name_or_path`, the tokenizer class and version strings are recorded
for humans but never enter `fingerprints.config`, so the same tokenizer loaded under a different name or from a
different directory has the same fingerprint.

## Export, compare and verify

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
config = tokenizer.export_encoding_config()

print(config.fingerprint)
# '2db205d22c1585acacd453f8892fb94349c15b9f885f6829099959f4a3c80bdf'
print(config.vocabulary_fingerprint["size"])  # 30522
```

The default export is compact (tens of kilobytes even for a 100k vocabulary). Pass `include_vocabulary=True` to also
embed a compressed copy of the vocabulary so the config can [rebuild the tokenizer](#rebuild-a-tokenizer-from-a-config)
later. The embedded payload never changes the fingerprint.

### Compare two configurations

```python
training = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
serving = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", do_lower_case=False)

for difference in training.compare_encoding_configs(
    training.export_encoding_config(), serving.export_encoding_config()
):
    print(difference.describe())
# encoding.pipeline.normalizer.lowercase: True != False
# fingerprints.model_serialization.digest: '91...' != '81...'
# fingerprints.config.digest: '2d...' != '41...'
```

Each difference names the exact path that changed, the value on each side, and the tokenizer-side command to inspect
it. Concrete settings are listed before fingerprint digests. When the vocabulary itself differs, the vocabulary digest
is annotated with the id ranges that changed (for example `ids 0..4095`), which is what an accidentally swapped
`vocab.txt` looks like.

`EncodingConfig.diff` returns the same information programmatically.

### Verify at load time

```python
# Strict: raises EncodingConfigMissingError if there is no config, EncodingConfigMismatchError on a mismatch.
tokenizer = AutoTokenizer.from_pretrained("my-model", verify_encoding_config=True)

# Non-strict: warn and continue. Useful while rolling the config out.
tokenizer = AutoTokenizer.from_pretrained("my-model", verify_encoding_config=False)
```

`verify_encoding_config` never changes what the tokenizer encodes; it only compares and reports. A malformed or
unreadable config during loading is logged and ignored, so a bad config cannot make a working tokenizer unusable. In
strict mode a config that cannot be read counts as missing, and `EncodingConfigMissingError` is raised.

You can also verify a tokenizer you already hold:

```python
tokenizer.verify_encoding_config(expected_config, strict=True)
```

`EncodingConfigMismatchError.differences` holds the same [`EncodingConfigDifference`] objects that `diff` returns.

## Save with the model

```python
tokenizer.save_pretrained("my-model", save_encoding_config=True)
```

This writes `encoding_config.json` next to the existing files. Nothing else is written or modified, and without the
flag `save_pretrained` behaves exactly as before. Loading the directory then attaches the config to
`tokenizer.encoding_config` without verifying it, so `verify_encoding_config=True` (or an explicit `expected=`) can be
used later. To make the config self-contained:

```python
tokenizer.save_pretrained(
    "my-model", save_encoding_config=True, encoding_config_include_vocabulary=True
)
```

[`ProcessorMixin.save_pretrained`] forwards both flags to its tokenizer.

### Compatibility boundary

The config is a *new* artifact. It is only produced when `save_encoding_config=True` is passed; checkpoints, vocabularies
and adapters produced by earlier versions do not contain one. For those:

- Loading is unchanged. Without `encoding_config.json`, `from_pretrained` behaves exactly as before.
- `verify_encoding_config=True` raises `EncodingConfigMissingError` rather than falling back to defaults — a missing
  config is not evidence that the encodings match.
- To adopt the new mechanism for an old artifact, load it once with a known-good release, export a config, and save it
  next to the model. From then on the fingerprint pins the encoding.

Configs are versioned with `format_version`. A config written by a newer format version is rejected with a clear error
instead of being compared against something it does not describe.

## Rebuild a tokenizer from a config

A config exported with `include_vocabulary=True` carries everything needed to reconstruct the encoding component, so it
does not depend on the original vocabulary files still being present:

```python
config = tokenizer.export_encoding_config(include_vocabulary=True)
config.save("artifact/encoding_config.json")

rebuilt = config.build_tokenizer()
assert rebuilt.encode("Sphinx of black quartz") == tokenizer.encode("Sphinx of black quartz")
```

Rebuilding is supported for:

- fast (`tokenizers`) backends, from the serialized pipeline;
- SentencePiece backends, from the serialized model proto;
- custom backends, from a recorded vocabulary mapping (if the class accepts the vocabulary as a `vocab`, `vocabulary`
  or `vocab_dict` argument).

The rebuilt tokenizer has the same [`EncodingConfig.fingerprint`] as the original. A compact config (the default)
raises a `ValueError` explaining that `include_vocabulary=True` is required.

`build_tokenizer` only reconstructs tokenizer classes shipped with `transformers`; it refuses to import an arbitrary
module named by a config.

## Compatibility

- **Encoding is untouched.** The export reads the tokenizer and computes digests; it does not modify the tokenizer,
  the pipelines or any vocabulary.
- **Off by default.** `export_encoding_config`, `verify_encoding_config`, `save_encoding_config` and
  `encoding_config_include_vocabulary` are all opt-in. With them off, `from_pretrained` and `save_pretrained` produce
  the same behavior and the same files as before, and existing encoding outputs are unchanged.
- **No fallback values.** A missing config raises in strict mode; defaults are never substituted.
- **Fingerprint scope.** The fingerprint covers behavior, not provenance: class names, module paths, source file paths
  and the rebuild recipe are recorded in the JSON but excluded from `fingerprints.config`. For fast backends the
  serialized pipeline is authoritative; Python attributes that merely mirror it (for example `do_lower_case` alongside
  the normalizer's `lowercase`) are stored for humans but do not enter the fingerprint, so a setting change is reported
  once, at the pipeline path that actually drives encoding. Two configs with the same fingerprint encode identically;
  two configs exported from the same encoder agree on `diff` even if one was rebuilt from a payload.

## Known limitations

- **Custom backends without a vocabulary hook.** A tokenizer whose class does not accept its vocabulary as a mapping
  can still export and verify a config, but cannot be rebuilt from it. `build.supported` is `false` and
  `build_tokenizer` raises.
- **SentencePiece preprocessing.** For SentencePiece backends the fingerprint includes the model proto digest and the
  encoding attributes. Python-side preprocessing that lives outside the tokenizer (for example custom text
  normalization in a training script) is not captured.
- **Chat templates and processors.** The config describes encoding, not conversation formatting. Chat templates and
  multimodal processors are out of scope.
- **Embedded payload size.** `include_vocabulary=True` roughly reproduces the compressed size of `tokenizer.json`
  (a few hundred kilobytes for a 100k vocabulary after compression). The compact config stays in the tens of kilobytes.

## Concurrency and determinism

Export and verification are read-only and take no locks. The vocabulary is walked with local buffers, so exporting the
same tokenizer from several threads yields the same fingerprint; concurrency can change the duration but not the
result. There is no randomness anywhere in the export, comparison or rebuild path, and dictionary iteration order does
not affect a digest because vocabulary entries are ordered by id.

## Cost

Measured on a 100k-entry vocabulary (`bert-base-uncased`-style `vocab.txt`, no `tokenizer.json`) on CPU:

| Operation | Time | Extra peak memory |
|---|---|---|
| `from_pretrained` (baseline) | ~12–17 ms | — |
| `export_encoding_config()` | ~105–125 ms | ~39 MB |
| Saving the config | < 5 ms extra | — |
| `from_pretrained(verify_encoding_config=True)` | ~180–210 ms (vs ~48–63 ms without verification) | ~63 MB (vs ~30 MB) |

Verification is off by default, so the baseline load path is unaffected. Most of the export time is the content hash
over the 100k `(token, id)` pairs; the rest is the backend serialization digest and the pipeline description. No extra
copy of the vocabulary is kept: the export builds one temporary ordered list of ids for hashing and releases it.

The compact config for that vocabulary is ~14 KB. The configuration is loaded into memory once per verified load and is
not retained beyond the tokenizer's `encoding_config` attribute.

A 250k-entry Unigram vocabulary (`google/mt5-small`, 250k pieces with per-piece scores) exports in ~0.7 s, because both
the piece list and its scores are hashed. This is a one-off cost at export/verification time, not on the training or
serving hot path.
