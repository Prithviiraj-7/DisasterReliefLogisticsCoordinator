from __future__ import annotations

import json
import os
from typing import Dict

import requests
from openai import OpenAI


SYSTEM_PROMPT = """You are the agent controlling a disaster relief simulator.
You MUST respond with exactly one command string and no extra text.

Allowed commands:
- get_status
- deploy <resource_type> <amount> to <zone_id>
- check_routes
- wait

Resource types:
- trucks
- medical_teams
- food
- rescue_units

Strategy goals:
1) Prioritize highest severity zones with open roads.
2) Avoid over-supplying zones far beyond requirements.
3) Use check_routes when delivery failures are suspected.
4) Keep all active zones supported; avoid abandoned active zones.
"""


TASKS: Dict[str, str] = {
    "easy": "task_easy_local",
    "medium": "task_medium_city",
    "hard": "task_hard_region",
}


def choose_command(client: OpenAI, model: str, observation: Dict) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(observation)},
        ],
        temperature=0.2,
        max_output_tokens=32,
    )
    text = response.output_text.strip()
    if not text:
        return "get_status"
    return text.splitlines()[0].strip()


def run_episode(task_id: str) -> None:
    base_url = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running inference.")

    client = OpenAI(api_key=api_key)

    reset_resp = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=20)
    reset_resp.raise_for_status()
    payload = reset_resp.json()
    observation = payload["observation"]
    done = payload["done"]
    print(f"Task started: {task_id}")

    while not done:
        command = choose_command(client, model, observation)
        step_resp = requests.post(f"{base_url}/step", json={"action": {"command": command}}, timeout=20)
        step_resp.raise_for_status()
        step_data = step_resp.json()

        observation = step_data["observation"]
        done = step_data["done"]
        reward = step_data["reward"]

        print(
            f"tick={observation['tick']:02d} reward={reward:.3f} score={observation['task_score']:.3f} cmd={command}"
        )

    print("Episode complete.")
    print(f"Final score: {observation['task_score']:.3f}")


if __name__ == "__main__":
    task_alias = os.getenv("DISASTER_TASK", "hard").strip().lower()
    task_id = TASKS.get(task_alias, task_alias)
    run_episode(task_id)
