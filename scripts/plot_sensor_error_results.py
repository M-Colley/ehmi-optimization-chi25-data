"""Plot outputs from bo_sensor_error_simulation.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel


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


def summarize_final_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = (
        df.sort_values("iteration")
        .groupby("run_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    final_rows["baseline"] = final_rows["error_model"] == "none"
    return final_rows


def evaluate_final_outcomes(final_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    baseline = final_df[final_df["baseline"]].copy()
    jittered = final_df[~final_df["baseline"]].copy()
    merged = jittered.merge(
        baseline[
            [
                "acquisition",
                "seed",
                "oracle_model",
                "objective_true",
                "objective_observed",
            ]
        ],
        on=["acquisition", "seed", "oracle_model"],
        how="inner",
        suffixes=("_jitter", "_baseline"),
    )
    if merged.empty:
        return pd.DataFrame()

    rows = []
    group_cols = [
        "acquisition",
        "error_model",
        "jitter_iteration",
        "jitter_std",
        "oracle_model",
    ]
    for keys, group in merged.groupby(group_cols):
        acquisition, error_model, jitter_iteration, jitter_std, oracle_model = keys
        n_runs = len(group)
        mean_true_diff = float(
            (group["objective_true_jitter"] - group["objective_true_baseline"]).mean()
        )
        mean_obs_diff = float(
            (group["objective_observed_jitter"] - group["objective_observed_baseline"]).mean()
        )
        
        # Calculate Cohen's d effect sizes
        def cohens_d(group1, group2):
            diff = group1 - group2
            return diff.mean() / (diff.std() + 1e-10)  # Add small constant to avoid division by zero
        
        cohens_d_true = float(cohens_d(group["objective_true_jitter"], group["objective_true_baseline"]))
        cohens_d_obs = float(cohens_d(group["objective_observed_jitter"], group["objective_observed_baseline"]))
        
        true_p = float("nan")
        obs_p = float("nan")
        if n_runs >= 2:
            true_test = ttest_rel(
                group["objective_true_jitter"],
                group["objective_true_baseline"],
                nan_policy="omit",
            )
            obs_test = ttest_rel(
                group["objective_observed_jitter"],
                group["objective_observed_baseline"],
                nan_policy="omit",
            )
            true_p = float(true_test.pvalue)
            obs_p = float(obs_test.pvalue)
        rows.append(
            {
                "acquisition": acquisition,
                "error_model": error_model,
                "jitter_iteration": jitter_iteration,
                "jitter_std": jitter_std,
                "oracle_model": oracle_model,
                "runs": n_runs,
                "mean_true_diff": mean_true_diff,
                "mean_observed_diff": mean_obs_diff,
                "cohens_d_true": cohens_d_true,
                "cohens_d_observed": cohens_d_obs,
                "p_value_true": true_p,
                "p_value_observed": obs_p,
            }
        )

    stats = pd.DataFrame(rows)
    
    # Apply Bonferroni correction
    n_tests = len(stats) * 2  # multiply by 2 because we test both true and observed
    stats["p_value_true_bonferroni"] = stats["p_value_true"] * n_tests
    stats["p_value_observed_bonferroni"] = stats["p_value_observed"] * n_tests
    # Cap corrected p-values at 1.0
    stats["p_value_true_bonferroni"] = stats["p_value_true_bonferroni"].clip(upper=1.0)
    stats["p_value_observed_bonferroni"] = stats["p_value_observed_bonferroni"].clip(upper=1.0)
    
    # Add significance flags (alpha = 0.05)
    stats["significant_true_uncorrected"] = stats["p_value_true"] < 0.05
    stats["significant_observed_uncorrected"] = stats["p_value_observed"] < 0.05
    stats["significant_true_bonferroni"] = stats["p_value_true_bonferroni"] < 0.05
    stats["significant_observed_bonferroni"] = stats["p_value_observed_bonferroni"] < 0.05
    
    stats_path = output_dir / "final_outcome_significance.csv"
    stats.to_csv(stats_path, index=False)
    
    # Print summary
    print("\nStatistical Testing Summary:")
    print(f"Total number of comparisons: {len(stats)}")
    print(f"Total number of tests: {n_tests}")
    print(f"Bonferroni-corrected alpha: {0.05 / n_tests:.6f}")
    print(f"\nSignificant results (uncorrected, alpha=0.05):")
    print(f"  True objective: {stats['significant_true_uncorrected'].sum()}/{len(stats)}")
    print(f"  Observed objective: {stats['significant_observed_uncorrected'].sum()}/{len(stats)}")
    print(f"\nSignificant results (Bonferroni-corrected, alpha=0.05):")
    print(f"  True objective: {stats['significant_true_bonferroni'].sum()}/{len(stats)}")
    print(f"  Observed objective: {stats['significant_observed_bonferroni'].sum()}/{len(stats)}")
    
    return stats


def plot_final_outcome_significance(stats: pd.DataFrame, output_dir: Path) -> None:
    if stats.empty:
        return
    melted = stats.melt(
        id_vars=[
            "acquisition",
            "error_model",
            "jitter_iteration",
            "jitter_std",
            "oracle_model",
            "runs",
        ],
        value_vars=["p_value_true", "p_value_observed"],
        var_name="metric",
        value_name="p_value",
    )
    for (error_model, jitter_iteration, oracle_model), data in melted.groupby(
        ["error_model", "jitter_iteration", "oracle_model"]
    ):
        plt.figure(figsize=(10, 4))
        sns.scatterplot(
            data=data,
            x="jitter_std",
            y="p_value",
            hue="metric",
            style="acquisition",
        )
        plt.axhline(0.05, color="red", linestyle="--", linewidth=1)
        plt.title(
            "Final outcome significance (paired t-test) "
            f"- {error_model}, {oracle_model}, jitter={jitter_iteration}"
        )
        plt.ylabel("p-value")
        plt.xlabel("jitter_std")
        plt.tight_layout()
        filename = f"final_outcome_pvalues_{error_model}_{oracle_model}_jit{jitter_iteration}.png"
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def plot_objectives(df: pd.DataFrame, output_dir: Path) -> None:
    group_cols = [
        "acquisition",
        "error_model",
        "jitter_std",
        "jitter_iteration",
        "oracle_model",
        "iteration",
    ]
    grouped = (
        df.groupby(group_cols)[["objective_true", "objective_observed"]].mean().reset_index()
    )
    for (acq, error_model, jitter_std, jitter_iteration, oracle_model), data in grouped.groupby(
        ["acquisition", "error_model", "jitter_std", "jitter_iteration", "oracle_model"]
    ):
        plt.figure(figsize=(8, 4))
        sns.lineplot(data=data, x="iteration", y="objective_true", label="Objective (true)")
        sns.lineplot(data=data, x="iteration", y="objective_observed", label="Objective (observed)")
        plt.title(
            "Objective trajectory "
            f"({acq}, {error_model}, {oracle_model}, jitter={jitter_iteration}, std={jitter_std})"
        )
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.tight_layout()
        filename = (
            "objective_trajectory_"
            f"{acq}_{error_model}_{oracle_model}_jit{jitter_iteration}_std{jitter_std}.png"
        )
        plt.savefig(output_dir / filename, dpi=200)
        plt.close()


def plot_adjustments(input_dir: Path, output_dir: Path) -> None:
    summary_stats = input_dir / "bo_sensor_error_summary_stats.csv"
    if not summary_stats.exists():
        return
    stats = pd.read_csv(summary_stats)
    for (error_model, jitter_iteration), data in stats.groupby(
        ["error_model", "jitter_iteration"]
    ):
        plot = sns.catplot(
            data=data,
            x="acquisition",
            y="delta_l2_mean",
            hue="baseline",
            col="jitter_std",
            kind="bar",
            height=4,
            aspect=1.1,
        )
        plot.fig.suptitle(
            f"Mean parameter adjustment (L2 norm) - {error_model} (jitter={jitter_iteration})"
        )
        plot.set_axis_labels("acquisition", "Mean delta L2")
        plot.tight_layout()
        filename = f"delta_l2_mean_{error_model}_jit{jitter_iteration}.png"
        plot.savefig(output_dir / filename, dpi=200)
        plt.close(plot.fig)


def plot_excess_adjustments(input_dir: Path, output_dir: Path) -> None:
    excess_path = input_dir / "bo_sensor_error_excess_summary.csv"
    if not excess_path.exists():
        return
    excess = pd.read_csv(excess_path)
    summary = (
        excess.groupby(["acquisition", "error_model", "jitter_iteration", "jitter_std"])
        .agg(
            delta_excess_l2_mean=("delta_excess_l2_norm", "mean"),
            delta_excess_l2_std=("delta_excess_l2_norm", "std"),
            runs=("delta_excess_l2_norm", "count"),
        )
        .reset_index()
    )
    for (jitter_iteration, jitter_std), data in summary.groupby(
        ["jitter_iteration", "jitter_std"]
    ):
        plt.figure(figsize=(8, 4))
        sns.barplot(
            data=data,
            x="acquisition",
            y="delta_excess_l2_mean",
            hue="error_model",
        )
        plt.title(
            "Mean excess adjustment (L2 norm) "
            f"- jitter={jitter_iteration}, std={jitter_std:.2f}"
        )
        plt.ylabel("Mean delta excess L2")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"delta_excess_l2_mean_jit{jitter_iteration}_std{jitter_std}.png",
            dpi=200,
        )
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = load_iteration_logs(args.input_dir)
    plot_objectives(logs, args.output_dir)
    plot_adjustments(args.input_dir, args.output_dir)
    plot_excess_adjustments(args.input_dir, args.output_dir)
    final_outcomes = summarize_final_outcomes(logs)
    stats = evaluate_final_outcomes(final_outcomes, args.output_dir)
    plot_final_outcome_significance(stats, args.output_dir)
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
