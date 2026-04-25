from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict


def read_metrics(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "update": float(row["update"]),
                    "training_step": float(row["training_step"]),
                    "loss": float(row["loss"]),
                    "avg_reward": float(row["avg_reward"]),
                }
            )
    if not rows:
        raise ValueError(f"No metric rows found in {csv_path}")
    return rows


def build_report_text(run_dir: Path, rows: List[Dict[str, float]]) -> str:
    first = rows[0]
    last = rows[-1]
    loss_delta = last["loss"] - first["loss"]
    reward_delta = last["avg_reward"] - first["avg_reward"]

    best_reward = max(r["avg_reward"] for r in rows)
    best_reward_step = next(r["training_step"] for r in rows if r["avg_reward"] == best_reward)

    best_loss = min(r["loss"] for r in rows)
    best_loss_step = next(r["training_step"] for r in rows if r["loss"] == best_loss)

    md = []
    md.append("# Training Evidence Report")
    md.append("")
    md.append("This report was auto-generated from a real environment-connected training run.")
    md.append("")
    md.append("## Run Info")
    md.append("")
    md.append(f"- Run directory: `{run_dir}`")
    md.append(f"- Total updates: `{int(last['update'])}`")
    md.append(f"- Final training step: `{int(last['training_step'])}`")
    md.append("")
    md.append("## Metric Summary")
    md.append("")
    md.append(f"- Initial loss: `{first['loss']:.6f}`")
    md.append(f"- Final loss: `{last['loss']:.6f}`")
    md.append(f"- Loss change (final - initial): `{loss_delta:.6f}`")
    md.append(f"- Best (lowest) loss: `{best_loss:.6f}` at step `{int(best_loss_step)}`")
    md.append("")
    md.append(f"- Initial avg reward: `{first['avg_reward']:.6f}`")
    md.append(f"- Final avg reward: `{last['avg_reward']:.6f}`")
    md.append(f"- Reward change (final - initial): `{reward_delta:.6f}`")
    md.append(f"- Best avg reward: `{best_reward:.6f}` at step `{int(best_reward_step)}`")
    md.append("")
    md.append("## Curves")
    md.append("")
    md.append("Loss curve (x-axis: `training step`, y-axis: `loss`):")
    md.append("")
    md.append("![Loss Curve](loss_curve.png)")
    md.append("")
    md.append("Reward curve (x-axis: `training step`, y-axis: `reward`):")
    md.append("")
    md.append("![Reward Curve](reward_curve.png)")
    md.append("")
    md.append("## Raw Metrics")
    md.append("")
    md.append("Source file: `training_metrics.csv`")
    md.append("")
    return "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create markdown evidence report from training artifacts.")
    parser.add_argument("--run-dir", default="artifacts/training_run")
    parser.add_argument("--output-name", default="EVIDENCE_REPORT.md")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "training_metrics.csv"
    loss_plot = run_dir / "loss_curve.png"
    reward_plot = run_dir / "reward_curve.png"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not loss_plot.exists():
        raise FileNotFoundError(f"Missing loss curve: {loss_plot}")
    if not reward_plot.exists():
        raise FileNotFoundError(f"Missing reward curve: {reward_plot}")

    rows = read_metrics(metrics_path)
    report = build_report_text(run_dir, rows)
    output_path = run_dir / args.output_name
    output_path.write_text(report, encoding="utf-8")
    print(f"Evidence report written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
