# Submission Readiness Checklist

## Core Environment

- [x] 10+ zones supported (`task_hard_region` uses 12 zones)
- [x] Zone state includes severity, requirements, delivered, road access
- [x] Central depot tracks trucks, medical teams, food, rescue units
- [x] Tick-based degradation and transit processing
- [x] Scripted + dynamic events (road block/reopen, priority spike, depletion)
- [x] Reward and task score clamped to `0.001..0.999`
- [x] Massive terminal penalty for abandoned active zones

## API Contract

- [x] `POST /reset` -> returns `observation` and `done`
- [x] `POST /step` -> accepts `action.command`, returns `observation`, `reward`, `done`
- [x] Command parser supports:
  - [x] `get_status`
  - [x] `deploy <resource_type> <amount> to <zone_id>`
  - [x] `check_routes`
  - [x] `wait`

## Task Configurations

- [x] `task_easy_local`
- [x] `task_medium_city`
- [x] `task_hard_region`

## LLM + Training Evidence

- [x] Inference client included (`client/inference.py`)
- [x] Online training script connected to environment (`scripts/train_trl_disaster.py`)
- [x] Colab notebook included (`notebooks/disaster_training_colab.ipynb`)
- [x] Real training run artifacts generated:
  - [x] `artifacts/training_run/loss_curve.png`
  - [x] `artifacts/training_run/reward_curve.png`
  - [x] `artifacts/training_run/training_metrics.csv`
- [x] Evidence report generated (`artifacts/training_run/EVIDENCE_REPORT.md`)

## UX / Demo

- [x] Browser UI at `/`
- [x] Simplified controls and smart auto-step actions
- [x] Color legend, warnings, action history, and snapshot export

