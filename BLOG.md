# Building a Disaster-Relief Logistics “Gym” for LLM Agents

*April 2026*

Disaster response is a coordination problem: you have depots, zones, limited trucks, food, and medical teams—and every tick, conditions change. **Disaster Relief Logistics Coordinator** is a small, [OpenEnv](https://github.com/meta-pytorch/OpenEnv)–style **simulation** that turns that into a testbed for **language-model agents** and optional **TRL** fine-tuning.

## Why this project exists

General-purpose models can *talk* about logistics, but to improve them you need a **tight loop**: same observation format, same action grammar, and measurable **rewards** and **task scores** across difficulty levels. This repo is that loop: a **FastAPI** server, a **tick-based engine** (zones, depot, scripted and dynamic events), a **string command** interface, and a **web command center** to watch runs without writing glue code by hand.

## The environment in one pass

- **Reset** a scenario with `POST /reset` and a `task_id` (e.g. `task_hard_region` for 10+ zones and heavier disruption).
- **Act** with `POST /step` and a single `command` string, for example:
  - `get_status` — read the state of zones and the depot.
  - `deploy food 2 to zone_1` — move **food**, **trucks**, **medical\_teams**, or **rescue\_units** in integer amounts to a `zone_id`.
  - `check_routes` — reason about routes under constraints the engine applies.
  - `wait` — advance time without a deployment.
- The server returns an **observation** plus **reward** in a stable band and a **task\_score** so you can rank policies or prompts.

That contract is on purpose: **any** LLM that can emit a line of text is an agent, and you can swap models without retooling the sim.

## Training and evidence (optional)

`scripts/train_trl_disaster.py` uses **TRL (SFT)** on data collected from the same HTTP API, so the training signal matches deployment. A separate evidence path (`scripts/make_evidence_report.py`) can bundle curves and a short **EVIDENCE\_REPORT** for reports or course submissions. Training artifacts and large checkpoints are kept out of the default Git history; the “product” the Space needs is the **app**, not the weights.

## From laptop to a public demo

The server is **Docker**-ready: a single `Dockerfile`, `EXPOSE` aligned with [Hugging Face Spaces Docker](https://huggingface.co/docs/hub/spaces-sdks-docker), and **`PORT`** (default `7860`) so the platform can inject a runtime port. Pushing a **lean** history (no `safetensors` / zips in Git) avoids the Hub’s binary-size hooks for ordinary Git.

The public Space is here:

**[Hugging Face – Disaster Relief Coordinator (Docker)](https://huggingface.co/spaces/prithviipj7/disaster-relief-coordinator)**  

Open the **App** for the **command center UI**; add **`/health`** on the same host for a quick **JSON** liveness check.

## Where to get the code

- **Source & docs:** [github.com/Prithviiraj-7/DisasterReliefLogisticsCoordinator](https://github.com/Prithviiraj-7/DisasterReliefLogisticsCoordinator)
- **Live demo:** [Hugging Face Space (above)](https://huggingface.co/spaces/prithviipj7/disaster-relief-coordinator)

If you are reproducing the stack locally, clone, `pip install -r requirements.txt`, and run Uvicorn as in the README, or `docker build` / `docker run` on port 7860.

## Closing

This project is a **concrete, inspectable** bridge between “LLMs for good” and **reinforced, measurable** behavior in a **disaster logistics** domain. The API stays small; the **hard** part is left to the sim and the models you connect to it.

If you use or extend it, a pointer back to the [repository](https://github.com/Prithviiraj-7/DisasterReliefLogisticsCoordinator) is always appreciated.

---

*The simulation is a research and education prototype—not operational dispatch software.*
