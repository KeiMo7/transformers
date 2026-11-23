"""Evidence for positional resume: loss/weight alignment, sample manifests, timings, refusals."""

import json
import os
import shutil
import tempfile
import time

import torch
from torch import nn

from transformers import Trainer, TrainingArguments


SEED = 1234
ROWS = 96
BATCH = 4


class TrackedDataset(torch.utils.data.Dataset):
    """Map-style dataset; `row_id` identifies exactly which sample a batch contained."""

    def __init__(self, rows=ROWS):
        generator = torch.Generator().manual_seed(SEED)
        self.x = torch.randn(rows, generator=generator)
        self.n = rows

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input_x": self.x[i], "row_id": i, "labels": 2 * self.x[i] + 3}


class TrackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.2))
        self.order = []

    def forward(self, input_x, labels=None, row_id=None, **kwargs):
        y = self.a * input_x + self.b
        if row_id is not None:
            self.order.extend(int(r) for r in row_id)
        if labels is None:
            return {"logits": y}
        return {"loss": nn.functional.mse_loss(y, labels), "logits": y}


def build(output_dir, resume=None, **overrides):
    kwargs = {
        "output_dir": output_dir,
        "max_steps": 12,
        "per_device_train_batch_size": BATCH,
        "save_strategy": "steps",
        "save_steps": 1,
        "logging_steps": 1,
        "learning_rate": 0.05,
        "report_to": [],
        "disable_tqdm": True,
        "resumable_dataloader": True,
        "seed": SEED,
    }
    kwargs.update(overrides)
    trainer = Trainer(TrackedModel(), TrainingArguments(**kwargs), train_dataset=TrackedDataset())
    trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=resume)
    return trainer


def losses(trainer):
    return [(entry["step"], entry["loss"]) for entry in trainer.state.log_history if "loss" in entry]


def params(trainer):
    return torch.cat([p.detach().flatten() for p in trainer.model.parameters()]).tolist()


def main():
    report = {}
    full_dir = tempfile.mkdtemp()
    full = build(full_dir)
    full_losses = losses(full)
    full_params = params(full)
    full_order = list(full.model.order)
    report["full_run"] = {"step_losses": full_losses, "params": full_params, "rows": len(full_order)}

    # 1) loss and weight alignment at a mid-epoch interruption
    resume_step = 7
    resumed = build(tempfile.mkdtemp(), os.path.join(full_dir, f"checkpoint-{resume_step}"))
    resumed_losses = dict(losses(resumed))
    report["resume_step"] = resume_step
    report["step_losses"] = [
        {
            "step": step,
            "uninterrupted": loss,
            "resumed": resumed_losses.get(step),
            "abs_diff": abs(loss - resumed_losses.get(step, float("nan"))),
        }
        for step, loss in full_losses
    ]
    report["weights_max_abs_diff"] = max(abs(a - b) for a, b in zip(full_params, params(resumed)))

    # 2) sample manifest: rows drawn before the interruption vs after resuming
    after = list(resumed.model.order)
    before = full_order[: len(full_order) - len(after)]
    report["sample_manifest"] = {
        "before_interruption": before,
        "after_resume": after,
        "expected_after_resume": full_order[len(full_order) - len(after):],
        "identical": after == full_order[len(full_order) - len(after):],
        "duplicates_within_epoch": len(full_order) - len(set(full_order)),
        "overlap_before_and_after": sorted(set(before) & set(after)),
        "rows_seen_total": len(full_order),
    }

    # 3) resume cost at an early and a late checkpoint
    long_dir = tempfile.mkdtemp()
    long_run = build(long_dir, max_steps=40)
    long_params = params(long_run)
    timings = {}
    for step in (2, 30):
        probe = build(tempfile.mkdtemp(), os.path.join(long_dir, f"checkpoint-{step}"), max_steps=40)
        probe.model.order.clear()
        timings[step] = {
            "extra_rows_drawn_for_resume": len(probe.model.order),
            "matches_uninterrupted": max(abs(a - b) for a, b in zip(long_params, params(probe))),
        }
    # time only the resume call, over a few repeats
    for step in (2, 30):
        samples = []
        for _ in range(3):
            probe = Trainer(
                TrackedModel(),
                TrainingArguments(
                    output_dir=tempfile.mkdtemp(),
                    max_steps=40,
                    per_device_train_batch_size=BATCH,
                    save_strategy="steps",
                    save_steps=1,
                    logging_steps=1,
                    learning_rate=0.05,
                    report_to=[],
                    disable_tqdm=True,
                    resumable_dataloader=True,
                    seed=SEED,
                ),
                train_dataset=TrackedDataset(),
            )
            probe.model_accepts_loss_kwargs = False
            start = time.perf_counter()
            probe.train(resume_from_checkpoint=os.path.join(long_dir, f"checkpoint-{step}"))
            samples.append(time.perf_counter() - start)
        timings[step]["resume_seconds"] = samples
    report["resume_cost"] = {
        "checkpoint_step": list(timings),
        "details": timings,
    }

    # 4) refusal when a component is missing
    refusals = {}
    checkpoint = os.path.join(full_dir, f"checkpoint-{resume_step}")
    for missing in ("optimizer.pt", "resume_state.pt", "checkpoint_manifest.json"):
        broken = os.path.join(tempfile.mkdtemp(), "checkpoint-broken")
        shutil.copytree(checkpoint, broken)
        os.remove(os.path.join(broken, missing))
        probe = Trainer(
            TrackedModel(),
            TrainingArguments(
                output_dir=tempfile.mkdtemp(),
                max_steps=12,
                per_device_train_batch_size=BATCH,
                save_strategy="steps",
                save_steps=1,
                learning_rate=0.05,
                report_to=[],
                disable_tqdm=True,
                resumable_dataloader=True,
                seed=SEED,
            ),
            train_dataset=TrackedDataset(),
        )
        probe.model_accepts_loss_kwargs = False
        try:
            probe.train(resume_from_checkpoint=broken)
            refusals[missing] = "NOT REFUSED"
        except Exception as error:  # noqa: BLE001 - the message is the evidence
            refusals[missing] = f"{type(error).__name__}: {error}"
    report["refusals"] = refusals

    with open(os.path.join(os.path.dirname(__file__), "resume_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
