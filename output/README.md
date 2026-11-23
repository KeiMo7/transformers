# Positional resume: evidence

`TrainingArguments(resumable_dataloader=True)` stores the position of the training data stream in
each checkpoint instead of replaying the data on resume. This directory holds the measurements the
implementation was accepted against.

Files:

- `verify_resume.py` – the experiment script (`PYTHONPATH=src python output/verify_resume.py`).
- `resume_report.json` – its machine-readable output (losses, weights, sample manifests, timings,
  refusal messages).
- `pytest_positional.txt` – the 17 new tests.
- `pytest_trainer_subset.txt` – the existing trainer tests that cover checkpoints, callbacks,
  training arguments and data loading.
- `pytest_test_trainer.txt` – the full `tests/trainer/test_trainer.py`.
- `pytest_test_trainer_baseline.txt` – the same file on the unmodified checkout, for comparison.

## 1. Loss and weights align step by step

Short training with the repository fixtures (`TrackedModel`, a two-parameter regression model; map
dataset with 96 rows, batch 4, 12 update steps). Checkpoint at step 7, resume into a fresh process
state, compare every logged step:

| step | uninterrupted loss | resumed loss | abs diff |
|-----:|-------------------:|-------------:|---------:|
| 1 | 5.677754878997803 | 5.677754878997803 | 0.0 |
| 2 | 13.58321762084961 | 13.58321762084961 | 0.0 |
| 3 | 10.834183692932129 | 10.834183692932129 | 0.0 |
| 4 | 5.300166606903076 | 5.300166606903076 | 0.0 |
| 5 | 3.456646203994751 | 3.456646203994751 | 0.0 |
| 6 | 6.830784320831299 | 6.830784320831299 | 0.0 |
| 7 | 9.115760803222656 | 9.115760803222656 | 0.0 |
| 8 | 14.313712120056152 | 14.313712120056152 | 0.0 |
| 9 | 4.435636043548584 | 4.435636043548584 | 0.0 |
| 10 | 2.83951997756958 | 2.83951997756958 | 0.0 |
| 11 | 7.2862043380737305 | 7.2862043380737305 | 0.0 |
| 12 | 2.139941692352295 | 2.139941692352295 | 0.0 |

`weights_max_abs_diff = 0.0`. The test suite checks exactly the same equality at **every** checkpoint (step 1 through 7),
inside a gradient-accumulation window, and for a streaming dataset, always comparing the full
parameter vector and the whole logged loss list.

## 2. Sample order: nothing repeated, nothing dropped

Same run, rows recorded by the model in the order they were consumed. The resumed process drew 20
of the 48 rows of the epoch.

- rows before the checkpoint: `[15, 92, 62, 14, 64, 13, 48, 84, 47, 57, 45, 37, 34, 68, 17, 7, 1, 67, 54, 10, 38, 78, 58, 76, 29, 21, 87, 90]`
- rows drawn after resuming: `[46, 44, 35, 75, 80, 11, 24, 43, 18, 74, 72, 91, 63, 81, 4, 85, 26, 89, 19, 32]`
- rows the uninterrupted run drew after that point: the same list, element for element
  (`identical: true`)
- duplicates within the epoch: `0`
- rows present on both sides of the interrupt: `[]`

## 3. Resume cost does not depend on progress

40-step run, resume from checkpoint 2 and from checkpoint 30, three repetitions of the resume call
each:

| checkpoint | rows re-read to reach the cursor | resume wall time (s) |
|-----------:|---------------------------------:|----------------------|
| 2  | 0 | 0.402, 0.372, 0.363 |
| 30 | 0 | 0.230, 0.257, 0.262 |

The late resume is not slower; the difference is noise, and it is never slower in proportion to progress. Fewer steps remain in the 40-step budget at the later point, which is the only systematic difference.
`extra_rows_drawn_for_resume` is 0 at both points: the dataloader is positioned, not fast-forwarded.

## 4. Missing components are refused, with the reason

`resume_report.json → refusals`:

- `optimizer.pt` removed →
  `ValueError: ... incomplete or corrupted, refusing to resume: - missing file: optimizer.pt`
- `resume_state.pt` removed →
  `ValueError: ... incomplete or corrupted, refusing to resume: - missing file: resume_state.pt`
- `checkpoint_manifest.json` removed →
  `ValueError: ... has no checkpoint_manifest.json, so there is no proof it was written completely.
  resumable_dataloader=True refuses to resume from a checkpoint it cannot verify ...`

Additional refusals exercised by the tests: dataset signature mismatch (reported field by field),
a cursor beyond the end of the epoch, a cursor whose step counter disagrees with the saved global
step, non-positional sampling strategies, overridden dataloaders, and the incompatible argument
combinations (`ignore_data_skip`, `save_only_model`).

## 5. Existing tests

- `pytest_trainer_subset.txt`: `180 passed, 13 skipped`.
- `pytest_test_trainer.txt`: `2 failed, 36 passed, 11 skipped`;
  `pytest_test_trainer_baseline.txt` on the unmodified checkout shows the identical two failures
  (`test_include_num_input_tokens_seen`, `test_torch_compile_loss_func_compatibility`), so they are
  environmental and not caused by this change.

## Scope and limits (stated, not hidden)

- With accelerate's default seedable sampler the order the run already used is itself a function of
  `(seed, epoch)`, so enabling the option reproduces it exactly: measured identical per-rank orders
  on 5 single-process seeds and on both ranks of a 2-GPU run, and covered by
  `test_sample_order_is_unchanged_by_enabling_the_option`. The option is off by default; with it off
  the code path, checkpoint layout and sample order are unchanged (the only difference when on is
  the two extra files, `resume_state.pt` and `checkpoint_manifest.json`). Train and resume must both
  enable it.
- Supported orders are `"random"` and `"sequential"`. Length-grouped, batch-rebalanced and custom
  samplers are refused rather than resumed approximately.
- Streaming datasets need `state_dict`/`load_state_dict` (e.g. `datasets.IterableDataset`) and a
  single data-parallel process; multi-rank streaming is refused because one source position cannot
  describe what each rank consumed. A streaming checkpoint may land mid gradient-accumulation
  window: the source is rewound to the batch boundary that was actually trained, and the partially
  accumulated gradients are restored from the checkpoint like in the map-style case.
- `dataloader_num_workers > 0` is supported for map-style datasets: the sampler is deterministic and
  the worker seed is reset every epoch, so the order is reproduced. Augmentation randomness is
  seeded per sample, so it does not depend on worker assignment.
