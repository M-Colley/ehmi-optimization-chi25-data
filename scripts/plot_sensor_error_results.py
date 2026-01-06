"""Plot outputs from bo_sensor_error_simulation.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/plots"))
    return parser.parse_args()


def load_iteration_logs(input_dir: Path) -> pd.DataFrame:
    files = list(input_dir.glob("bo_sensor_error_*_seed*_*.csv"))
    if not files:
        raise FileNotFoundError("No per-iteration logs found in input-dir.")
    frames = [pd.read_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def plot_objectives(df: pd.DataFrame, output_dir: Path) -> None:
    jittered = df[df["error_model"] != "none"]
    grouped = (
        jittered.groupby(["acquisition", "iteration"])[["objective_true", "objective_observed"]]
        .mean()
        .reset_index()
    )
    for acq, data in grouped.groupby("acquisition"):
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=data, x="iteration", y="objective_true", label="Objective (true)")
        sns.lineplot(data=data, x="iteration", y="objective_observed", label="Objective (observed)")
        plt.title(f"Objective trajectory ({acq})")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.tight_layout()
        plt.savefig(output_dir / f"objective_trajectory_{acq}.png", dpi=200)
        plt.close()


def plot_adjustments(input_dir: Path, output_dir: Path) -> None:
    summary_stats = input_dir / "bo_sensor_error_summary_stats.csv"
    if not summary_stats.exists():
        return
    stats = pd.read_csv(summary_stats)
    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=stats,
        x="acquisition",
        y="delta_l2_mean",
        hue="baseline",
    )
    plt.title("Mean parameter adjustment (L2 norm)")
    plt.ylabel("Mean delta L2")
    plt.tight_layout()
    plt.savefig(output_dir / "delta_l2_mean_by_acquisition.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = load_iteration_logs(args.input_dir)
    plot_objectives(logs, args.output_dir)
    plot_adjustments(args.input_dir, args.output_dir)
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()