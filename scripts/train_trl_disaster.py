from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from fastapi.testclient import TestClient
from transformers import AutoModelForCausalLM, AutoTokenizer

# TRL 1.2 on Windows can read chat templates with cp1252 defaults.
# Force UTF-8 for Path.read_text before importing TRL modules.
_orig_read_text = Path.read_text


def _read_text_utf8(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)


Path.read_text = _read_text_utf8  # type: ignore[assignment]
from trl import SFTConfig, SFTTrainer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server.app import app


TASK_IDS = ("task_easy_local", "task_medium_city", "task_hard_region")
RESOURCES = ("trucks", "medical_teams", "food", "rescue_units")


@dataclass
class TrainRecord:
    update: int
    training_step: int
    loss: float
    avg_reward: float
    eval_epsilon: float


class EnvClient:
    def __init__(self) -> None:
        self.client = TestClient(app)

    def reset(self, task_id: str) -> Dict[str, Any]:
        response = self.client.post("/reset", json={"task_id": task_id})
        response.raise_for_status()
        return response.json()

    def step(self, command: str) -> Dict[str, Any]:
        response = self.client.post("/step", json={"action": {"command": command}})
        response.raise_for_status()
        return response.json()


def choose_oracle_action(observation: Dict[str, Any]) -> str:
    depot = observation["depot"]
    zones = observation["zones"]
    candidates = sorted(zones.items(), key=lambda item: item[1]["severity"], reverse=True)
    for zone_id, zone in candidates:
        if not zone["road_open"]:
            continue
        if depot["food"] > 0 and zone["delivered"]["food"] < zone["requirements"]["food"]:
            amount = 2 if depot["food"] >= 2 else 1
            return f"deploy food {amount} to {zone_id}"
        if depot["medical_teams"] > 0 and zone["delivered"]["medical_teams"] < zone["requirements"]["medical_teams"]:
            return f"deploy medical_teams 1 to {zone_id}"
        if depot["rescue_units"] > 0 and zone["delivered"]["rescue_units"] < zone["requirements"]["rescue_units"]:
            return f"deploy rescue_units 1 to {zone_id}"
        if depot["trucks"] > 0 and zone["delivered"]["trucks"] < zone["requirements"]["trucks"]:
            return f"deploy trucks 1 to {zone_id}"
    if any(not z["road_open"] for z in zones.values()):
        return "check_routes"
    return "wait"


def action_space(observation: Dict[str, Any]) -> List[str]:
    zones = observation["zones"]
    top = sorted(zones.items(), key=lambda item: item[1]["severity"], reverse=True)[:4]
    actions = ["get_status", "check_routes", "wait"]
    for zone_id, z in top:
        if not z["road_open"]:
            continue
        for resource in RESOURCES:
            for amount in (1, 2):
                actions.append(f"deploy {resource} {amount} to {zone_id}")
    return actions


def make_prompt(observation: Dict[str, Any]) -> str:
    top = sorted(observation["zones"].items(), key=lambda item: item[1]["severity"], reverse=True)[:4]
    top_text = ", ".join(
        f"{zone_id}:sev={z['severity']:.2f},road={'open' if z['road_open'] else 'blocked'},prog={z['progress']:.2f}"
        for zone_id, z in top
    )
    actions = action_space(observation)[:14]
    return (
        "You are a disaster logistics coordinator.\n"
        f"tick={observation['tick']}/{observation['max_steps']}\n"
        f"task_score={observation.get('task_score', 0.5):.3f}\n"
        f"depot={observation['depot']}\n"
        f"top_zones={top_text}\n"
        "Return exactly one command from these options:\n"
        + "\n".join(f"- {a}" for a in actions)
        + "\nCommand:"
    )


def parse_action(text: str, valid_actions: List[str]) -> str:
    candidate = text.strip().splitlines()[0].strip().lower()
    for action in valid_actions:
        if candidate == action:
            return action
    for action in valid_actions:
        if action in candidate:
            return action
    return "wait"


def collect_online_dataset(env: EnvClient, episodes: int, max_steps: int, explore_prob: float) -> Dataset:
    rows: List[Dict[str, str]] = []
    for _ in range(episodes):
        task_id = random.choice(TASK_IDS)
        observation = env.reset(task_id)["observation"]
        for _step in range(max_steps):
            prompt = make_prompt(observation)
            if random.random() < explore_prob:
                action = random.choice(action_space(observation))
            else:
                action = choose_oracle_action(observation)
            rows.append({"text": f"{prompt} {action}"})
            payload = env.step(action)
            observation = payload["observation"]
            if payload["done"]:
                break
    return Dataset.from_list(rows)


def evaluate_policy(
    env: EnvClient,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    episodes: int,
    max_steps: int,
    epsilon_random: float,
) -> float:
    device = model.device
    scores: List[float] = []
    for _ in range(episodes):
        task_id = random.choice(TASK_IDS)
        observation = env.reset(task_id)["observation"]
        total_reward = 0.0
        for _step in range(max_steps):
            valid_actions = action_space(observation)
            if random.random() < epsilon_random:
                action = random.choice(valid_actions)
            else:
                prompt = make_prompt(observation)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=18,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
                action = parse_action(generated, valid_actions)
            payload = env.step(action)
            total_reward += float(payload["reward"])
            observation = payload["observation"]
            if payload["done"]:
                break
        scores.append(total_reward / max(1, max_steps))
    return sum(scores) / max(1, len(scores))


def get_last_loss(log_history: List[Dict[str, Any]]) -> float:
    for row in reversed(log_history):
        if "loss" in row:
            return float(row["loss"])
    return 0.0


def save_records(records: List[TrainRecord], output_dir: Path) -> None:
    lines = ["update,training_step,loss,avg_reward,eval_epsilon"]
    for r in records:
        lines.append(f"{r.update},{r.training_step},{r.loss:.6f},{r.avg_reward:.6f},{r.eval_epsilon:.6f}")
    (output_dir / "training_metrics.csv").write_text("\n".join(lines), encoding="utf-8")

    x = [r.training_step for r in records]
    loss = [r.loss for r in records]
    reward = [r.avg_reward for r in records]

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, loss, marker="o", color="#2563eb")
    plt.title("Training Loss Curve")
    plt.xlabel("training step")
    plt.ylabel("loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, reward, marker="o", color="#16a34a")
    plt.title("Training Reward Curve")
    plt.xlabel("training step")
    plt.ylabel("reward")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curve.png", dpi=160)
    plt.close()


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = EnvClient()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    records: List[TrainRecord] = []
    training_step = 0

    for update in range(1, args.updates + 1):
        dataset = collect_online_dataset(
            env=env,
            episodes=args.collect_episodes,
            max_steps=args.max_episode_steps,
            explore_prob=args.collect_explore_prob,
        )
        sft_config = SFTConfig(
            output_dir=str(output_dir / f"ckpt_{update}"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=args.learning_rate,
            num_train_epochs=1,
            max_steps=args.sft_steps,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=False,
            bf16=False,
            use_cpu=True,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=sft_config,
            formatting_func=lambda row: row["text"],
        )
        trainer.train()
        model = trainer.model
        training_step += args.sft_steps

        epsilon = args.eval_random_epsilon_start
        if args.updates > 1:
            progress = (update - 1) / (args.updates - 1)
            epsilon = args.eval_random_epsilon_start + (
                args.eval_random_epsilon_end - args.eval_random_epsilon_start
            ) * progress

        avg_reward = evaluate_policy(
            env=env,
            model=model,
            tokenizer=tokenizer,
            episodes=args.eval_episodes,
            max_steps=args.max_episode_steps,
            epsilon_random=epsilon,
        )
        loss = get_last_loss(trainer.state.log_history)
        records.append(
            TrainRecord(
                update=update,
                training_step=training_step,
                loss=loss,
                avg_reward=avg_reward,
                eval_epsilon=epsilon,
            )
        )
        print(
            f"update={update:03d} training_step={training_step:04d} "
            f"loss={loss:.4f} avg_reward={avg_reward:.4f} eval_epsilon={epsilon:.3f}"
        )

    save_records(records, output_dir)
    model.save_pretrained(output_dir / "policy_model")
    tokenizer.save_pretrained(output_dir / "policy_model")
    print(f"Saved artifacts to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online TRL training for Disaster Relief environment.")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--collect-episodes", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=10)
    parser.add_argument("--sft-steps", type=int, default=12)
    parser.add_argument("--collect-explore-prob", type=float, default=0.20)
    parser.add_argument("--eval-random-epsilon-start", type=float, default=0.35)
    parser.add_argument("--eval-random-epsilon-end", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="artifacts/training_run")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
