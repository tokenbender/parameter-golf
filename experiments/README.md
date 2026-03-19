# Experiment Tracking

Use `experiments/experiment_tracker.csv` as the source of truth for planned lanes, active work, and best known results.

Directory layout:

- `experiments/experiment_tracker.csv` - lane-level tracker with priorities, hypotheses, and latest results
- `experiments/runs/<date>__<experiment_id>__<label>/` - local run bundle scaffold for one concrete run
- `/workspace/parameter-golf-runs/active/<run_slug>/` - remote run in progress
- `/workspace/parameter-golf-runs/completed/<run_slug>/` - remote archived success bundle
- `/workspace/parameter-golf-runs/failed/<run_slug>/` - remote archived failed bundle

Each run bundle should contain:

- `metadata.json` - machine, track, seed, W&B defaults, and canonical run slug
- `command.sh` - exact launch command used for the run
- `notes.md` - hypothesis, changes, and follow-up notes
- `train.log` - raw training log copied after the run
- `artifacts/` - optional extra outputs if they are not kept at the top level of the run folder

Recommended workflow:

1. Pick a row in `experiments/experiment_tracker.csv` and keep the `experiment_id` stable.
2. Scaffold a concrete run bundle:
   `python3 experiments/scaffold_run.py exp_eval_stride_sweep --label stride64_seed1337`
3. Copy the generated `command.sh` into the remote active run folder and launch from there.
4. When the run finishes, move the remote folder from `active/` to `completed/` or `failed/`.
5. Update the matching CSV row with the latest run name and best metrics.

Weights & Biases defaults live in `experiments/wandb.defaults.env`.
The intended convention is one W&B project for all runs and one W&B group per `experiment_id`.
