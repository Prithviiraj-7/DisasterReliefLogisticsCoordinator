---
title: Disaster Relief Coordinator
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Disaster Relief Logistics Coordinator

OpenEnv-style disaster response simulation with a FastAPI backend and command-string action interface for LLM agents.

## Implemented Tasks

- `task_easy_local`: fewer zones, static conditions
- `task_medium_city`: more zones, scripted disruptions
- `task_hard_region`: 10+ zones, scripted + dynamic disruptions

## Command Interface

Supported action commands for `POST /step`:

- `get_status`
- `deploy <resource_type> <amount> to <zone_id>`
- `check_routes`
- `wait`

Supported `resource_type` values:

- `trucks`
- `medical_teams`
- `food`
- `rescue_units`

## API

### Reset

- Endpoint: `POST /reset`
- Request:

```json
{"task_id":"task_hard_region"}
```

- Response shape:

```json
{"observation": {...}, "done": false}
```

### Step

- Endpoint: `POST /step`
- Request:

```json
{"action":{"command":"get_status"}}
```

- Response shape:

```json
{"observation": {..., "task_score": 0.731}, "reward": 0.612, "done": false}
```

## Local Run (without Docker)

1. Install Python 3.10+
2. Install deps:

```bash
pip install -r requirements.txt
```

3. Start server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

4. Health check:

```bash
curl http://127.0.0.1:7860/health
```

## Docker Run (Hugging Face compatible port)

Build:

```bash
docker build -t disaster-relief-logistics .
```

Run:

```bash
docker run --rm -p 7860:7860 disaster-relief-logistics
```

The image listens on `0.0.0.0` and uses the `PORT` environment variable (default `7860`), matching [Hugging Face Spaces Docker](https://huggingface.co/docs/hub/spaces-sdks-docker) requirements.

## Hugging Face Space

1. Create a **new Space** at [New Space](https://huggingface.co/new-space): choose a name, **SDK** = **Docker**, then **Create Space**.
2. **Push this repository** to the Space’s Git (each Space is its own `https://huggingface.co/spaces/<user>/<space>` repo). From a clone of this project:
   - `git remote add hf https://huggingface.co/spaces/<user>/<space>`
   - `git push hf <branch>:main` (or push your default branch; match what the Space uses).
3. The **build** runs from the **repository root** `Dockerfile` and `requirements.txt`. The container listens on **`PORT`** (default `7860`), which matches Hugging Face Spaces.
4. When the build is green, open the Space URL: **Command Center** at `/`, **`GET /health`** returns `{"ok": true}`. In **Settings**, keep **app port** aligned with the process port if you change `PORT` (default `7860`).

**Secrets:** The simulation UI does not need API keys. If you add scripts that call OpenAI, store keys under **Settings → Repository secrets** (e.g. `OPENAI_API_KEY`) and use them in your process or CI only as you design.

## Quick API smoke test

PowerShell examples:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:7860/reset" -ContentType "application/json" -Body '{"task_id":"task_hard_region"}'
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:7860/step" -ContentType "application/json" -Body '{"action":{"command":"get_status"}}'
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:7860/step" -ContentType "application/json" -Body '{"action":{"command":"deploy food 2 to zone_1"}}'
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:7860/step" -ContentType "application/json" -Body '{"action":{"command":"wait"}}'
```

## LLM Inference Client

`client/inference.py` runs a loop that:

1. calls `/reset`
2. asks an OpenAI model for one command string
3. sends command to `/step`
4. repeats until done

Environment variables:

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, default in script)
- `OPENENV_BASE_URL` (optional, default `http://127.0.0.1:7860`)
- `DISASTER_TASK` (`easy`, `medium`, `hard`, or explicit task id)

Run:

```bash
python client/inference.py
```

## One-command local smoke test (PowerShell)

This script installs deps, starts server, checks `/health`, runs `/reset` and `/step`, then shuts server down.

```powershell
.\run_local.ps1
```

Optional flags:

```powershell
.\run_local.ps1 -TaskId task_medium_city
.\run_local.ps1 -Port 7860 -SkipInstall
```

## Training + Evidence (TRL)

To satisfy judging requirements, this repo includes a training loop that connects directly to the environment API logic each rollout step:

- Script: `scripts/train_trl_disaster.py`
- Colab notebook: `notebooks/disaster_training_colab.ipynb`

### Local training run

```powershell
py -m pip install -r requirements.txt
py scripts/train_trl_disaster.py --updates 16 --collect-episodes 4 --eval-episodes 3 --max-episode-steps 12 --sft-steps 12 --output-dir artifacts/training_run
```

### Output evidence artifacts

After completion, these files are generated:

- `artifacts/training_run/loss_curve.png`
- `artifacts/training_run/reward_curve.png`
- `artifacts/training_run/training_metrics.csv`

Generate a judge-ready summary report:

```powershell
py scripts/make_evidence_report.py --run-dir artifacts/training_run
```

This creates:

- `artifacts/training_run/EVIDENCE_REPORT.md`

Both plots include axis labels:

- x-axis: `training step`
- y-axis: `loss` or `reward`

### Why this counts as real environment training

- The training loop resets and steps the environment on every rollout via the FastAPI app (`/reset`, `/step`), using `fastapi.testclient`.
- Rewards are collected from live environment transitions, not from a static dataset.

## Submission Bundle

Create a zip with source + notebook + artifacts:

```powershell
py scripts/create_submission_bundle.py --output artifacts/submission_bundle.zip
```

## Setup On Another Computer

### 1) Clone repository

```bash
git clone https://github.com/Prithviiraj-7/DisasterReliefLogisticsCoordinator.git
cd DisasterReliefLogisticsCoordinator
```

### 2) Create and activate a virtual environment

Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run the app locally

```bash
py -m uvicorn server.app:app --host 127.0.0.1 --port 7860
```

Open in browser:

```text
http://127.0.0.1:7860
```

### 5) Optional smoke test

Windows:

```powershell
.\run_local.ps1 -SkipInstall
```

### 6) Optional training + evidence generation

```powershell
py scripts/train_trl_disaster.py --updates 3 --collect-episodes 2 --eval-episodes 2 --max-episode-steps 5 --sft-steps 3 --output-dir artifacts/training_run
py scripts/make_evidence_report.py --run-dir artifacts/training_run
```

### Common issues

- If `python` is not found on Windows, use `py` instead.
- If PowerShell blocks script execution, run:
  - `Set-ExecutionPolicy -Scope Process Bypass`
- If port `7860` is busy, run on another port (example: `7861`).
