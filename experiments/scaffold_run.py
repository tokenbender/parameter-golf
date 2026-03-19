#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
DEFAULTS_FILE = ROOT / "wandb.defaults.env"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "run"


def load_defaults() -> dict[str, str]:
    defaults: dict[str, str] = {}
    if not DEFAULTS_FILE.exists():
        return defaults
    for raw_line in DEFAULTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        defaults[key.strip()] = value.strip()
    return defaults


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_id")
    parser.add_argument("--label", default="")
    parser.add_argument("--track", default="track_10min_16mb")
    parser.add_argument("--machine", default="lium-8x-rtx-pro-6000")
    parser.add_argument("--seed", default="1337")
    args = parser.parse_args()

    defaults = load_defaults()
    date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    parts = [date_prefix, slugify(args.experiment_id)]
    if args.label:
        parts.append(slugify(args.label))
    run_slug = "__".join(parts)

    run_dir = RUNS_DIR / run_slug
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir()

    metadata = {
        "experiment_id": args.experiment_id,
        "run_slug": run_slug,
        "track": args.track,
        "machine": args.machine,
        "seed": str(args.seed),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_run_dir": str(run_dir.relative_to(ROOT.parent)),
        "remote_active_dir": f"/workspace/parameter-golf-runs/active/{run_slug}",
        "remote_completed_dir": f"/workspace/parameter-golf-runs/completed/{run_slug}",
        "remote_failed_dir": f"/workspace/parameter-golf-runs/failed/{run_slug}",
        "wandb_entity": defaults.get("WANDB_ENTITY", ""),
        "wandb_project": defaults.get("WANDB_PROJECT", ""),
        "wandb_job_type": defaults.get("WANDB_JOB_TYPE", "training"),
        "wandb_group": args.experiment_id,
        "notes": "",
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    command = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f'export RUN_ID="{run_slug}"',
            f'export WANDB_ENTITY="${{WANDB_ENTITY:-{metadata["wandb_entity"]}}}"',
            f'export WANDB_PROJECT="${{WANDB_PROJECT:-{metadata["wandb_project"]}}}"',
            f'export WANDB_NAME="${{WANDB_NAME:-{run_slug}}}"',
            f'export WANDB_GROUP="${{WANDB_GROUP:-{args.experiment_id}}}"',
            "export NCCL_IB_DISABLE=1",
            "export DATA_PATH=./data/datasets/fineweb10B_sp1024",
            "export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model",
            "export VOCAB_SIZE=1024",
            "export MAX_WALLCLOCK_SECONDS=600",
            "export TRAIN_LOG_EVERY=200",
            "export VAL_LOSS_EVERY=200",
            "",
            "torchrun --standalone --nproc_per_node=8 train_gpt.py",
            "",
        ]
    )
    (run_dir / "command.sh").write_text(command, encoding="utf-8")

    notes = "\n".join(
        [
            f"# {run_slug}",
            "",
            "## Hypothesis",
            "",
            "",
            "## Changes from parent lane",
            "",
            "",
            "## Run checklist",
            "",
            "- Remote active dir created",
            "- Exact command copied into command.sh",
            "- Train log copied back into this folder",
            "- Tracker CSV updated with latest metrics",
            "",
            "## Notes",
            "",
            "",
        ]
    )
    (run_dir / "notes.md").write_text(notes, encoding="utf-8")

    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
