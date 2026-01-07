"""Plot outputs from bo_sensor_error_simulation.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


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


def evaluate_final_outcomes_improved(final_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Comprehensive statistical analysis using ANOVA and appropriate post-hoc tests.
    
    Returns dictionary with:
    - anova_results: Mixed ANOVA results
    - posthoc_results: Tukey HSD post-hoc comparisons
    - effect_sizes: Effect size metrics
    - descriptive_stats: Descriptive statistics by group
    """
    baseline = final_df[final_df["baseline"]].copy()
    jittered = final_df[~final_df["baseline"]].copy()
    
    merged = jittered.merge(
        baseline[["acquisition", "seed", "oracle_model", "objective_true", "objective_observed"]],
        on=["acquisition", "seed", "oracle_model"],
        how="inner",
        suffixes=("_jitter", "_baseline"),
    )
    
    if merged.empty:
        return {}
    
    # Calculate differences for repeated measures
    merged["true_diff"] = merged["objective_true_jitter"] - merged["objective_true_baseline"]
    merged["obs_diff"] = merged["objective_observed_jitter"] - merged["objective_observed_baseline"]
    
    results = {}
    
    # ============================================================================
    # 1. MIXED ANOVA (treating seed as random effect)
    # ============================================================================
    print("\n" + "="*80)
    print("MIXED ANOVA ANALYSIS")
    print("="*80)
    
    for metric in ["true_diff", "obs_diff"]:
        metric_name = "True Objective" if metric == "true_diff" else "Observed Objective"
        print(f"\n{metric_name} Difference:")
        print("-" * 80)
        
        # Check which factors have multiple levels
        factors = {
            'acquisition': merged['acquisition'].nunique(),
            'error_model': merged['error_model'].nunique(),
            'jitter_std': merged['jitter_std'].nunique(),
            'jitter_iteration': merged['jitter_iteration'].nunique(),
            'oracle_model': merged['oracle_model'].nunique(),
        }
        
        print(f"Factor levels: {factors}")
        print(f"Total observations: {len(merged)}")
        
        # Build formula dynamically based on available factors
        valid_factors = [f for f, n in factors.items() if n >= 2]
        
        if len(valid_factors) == 0:
            print("Warning: No factors with 2+ levels. Cannot perform ANOVA.")
            continue
        
        # Check if we have enough observations for interaction terms
        # Rule of thumb: need at least 20 observations per parameter
        total_combinations = np.prod([factors[f] for f in valid_factors])
        min_obs_needed = total_combinations * 2  # At least 2 obs per cell
        
        print(f"Unique factor combinations: {total_combinations}")
        print(f"Minimum observations needed: {min_obs_needed}")
        
        # Start with full model and fall back to simpler models if needed
        models_to_try = []
        
        if len(valid_factors) >= 3 and len(merged) >= min_obs_needed:
            # Try full interaction model
            models_to_try.append(
                (f"{metric} ~ " + " * ".join([f"C({f})" for f in valid_factors]), "full interaction")
            )
        
        if len(valid_factors) >= 2:
            # Try additive model (main effects only)
            models_to_try.append(
                (f"{metric} ~ " + " + ".join([f"C({f})" for f in valid_factors]), "main effects only")
            )
            
            # Try two-way interactions if we have enough data
            if len(valid_factors) == 2 and len(merged) >= min_obs_needed:
                models_to_try.append(
                    (f"{metric} ~ C({valid_factors[0]}) * C({valid_factors[1]})", "two-way interaction")
                )
        
        if len(valid_factors) == 1:
            models_to_try.append(
                (f"{metric} ~ C({valid_factors[0]})", "single factor")
            )
        
        # Try models in order until one works
        anova_success = False
        for formula, description in models_to_try:
            print(f"\nTrying ANOVA: {description}")
            print(f"Formula: {formula}")
            
            try:
                model = ols(formula, data=merged).fit()
                
                # Check for infinite or NaN values
                if not np.all(np.isfinite(model.params)):
                    print("  Warning: Model parameters contain infinite or NaN values. Trying simpler model...")
                    continue
                
                anova_table = anova_lm(model, typ=2)
                
                # Check if ANOVA table is valid
                if anova_table['sum_sq'].isna().all() or not np.all(np.isfinite(anova_table['sum_sq'])):
                    print("  Warning: ANOVA table contains invalid values. Trying simpler model...")
                    continue
                
                # Calculate effect sizes (eta-squared)
                total_ss = anova_table['sum_sq'].sum()
                if total_ss > 0:
                    anova_table['eta_sq'] = anova_table['sum_sq'] / total_ss
                    anova_table['omega_sq'] = (
                        (anova_table['sum_sq'] - anova_table['df'] * model.mse_resid) / 
                        (total_ss + model.mse_resid)
                    )
                else:
                    anova_table['eta_sq'] = np.nan
                    anova_table['omega_sq'] = np.nan
                
                print("\n" + anova_table.to_string())
                
                results[f'anova_{metric}'] = anova_table
                results[f'anova_{metric}_model'] = description
                
                # Interpret main effects
                print("\nEffect Size Interpretation (eta-squared):")
                for idx, row in anova_table.iterrows():
                    if pd.isna(row['PR(>F)']) or pd.isna(row['eta_sq']):
                        continue
                    
                    if row['eta_sq'] >= 0.14:
                        size = "LARGE"
                    elif row['eta_sq'] >= 0.06:
                        size = "MEDIUM"
                    elif row['eta_sq'] >= 0.01:
                        size = "SMALL"
                    else:
                        size = "negligible"
                    
                    if row['PR(>F)'] < 0.001:
                        sig = "***"
                    elif row['PR(>F)'] < 0.01:
                        sig = "**"
                    elif row['PR(>F)'] < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    
                    print(f"  {idx}: η² = {row['eta_sq']:.4f} ({size}) {sig}")
                
                anova_success = True
                break
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        if not anova_success:
            print(f"\nCould not fit any ANOVA model for {metric_name}.")
            print("Consider collecting more data or reducing the number of factors.")
    
    # ============================================================================
    # 2. POST-HOC TESTS (Tukey HSD for pairwise comparisons)
    # ============================================================================
    print("\n" + "="*80)
    print("POST-HOC ANALYSIS (Tukey HSD)")
    print("="*80)
    
    for metric in ["true_diff", "obs_diff"]:
        metric_name = "True Objective" if metric == "true_diff" else "Observed Objective"
        
        # Post-hoc for acquisition strategies
        n_acquisitions = merged['acquisition'].nunique()
        if n_acquisitions >= 2:
            print(f"\n{metric_name} - Pairwise Acquisition Comparisons:")
            print("-" * 80)
            try:
                tukey_acq = pairwise_tukeyhsd(merged[metric], merged['acquisition'])
                print(tukey_acq)
                results[f'tukey_acq_{metric}'] = tukey_acq
            except Exception as e:
                print(f"Could not perform Tukey HSD for acquisitions: {e}")
        else:
            print(f"\n{metric_name} - Skipping acquisition comparisons (only {n_acquisitions} group(s))")
        
        # Post-hoc for error models
        n_error_models = merged['error_model'].nunique()
        if n_error_models >= 2:
            print(f"\n{metric_name} - Pairwise Error Model Comparisons:")
            print("-" * 80)
            try:
                tukey_error = pairwise_tukeyhsd(merged[metric], merged['error_model'])
                print(tukey_error)
                results[f'tukey_error_{metric}'] = tukey_error
            except Exception as e:
                print(f"Could not perform Tukey HSD for error models: {e}")
        else:
            print(f"\n{metric_name} - Skipping error model comparisons (only {n_error_models} group(s))")
        
        # Post-hoc for jitter_std if there are multiple levels
        n_jitter_stds = merged['jitter_std'].nunique()
        if n_jitter_stds >= 2:
            print(f"\n{metric_name} - Pairwise Jitter Std Comparisons:")
            print("-" * 80)
            try:
                tukey_jitter = pairwise_tukeyhsd(merged[metric], merged['jitter_std'])
                print(tukey_jitter)
                results[f'tukey_jitter_{metric}'] = tukey_jitter
            except Exception as e:
                print(f"Could not perform Tukey HSD for jitter_std: {e}")
        else:
            print(f"\n{metric_name} - Skipping jitter_std comparisons (only {n_jitter_stds} level(s))")
    
    # ============================================================================
    # 3. EFFECT SIZES (Cohen's d) for key comparisons
    # ============================================================================
    print("\n" + "="*80)
    print("EFFECT SIZES (Cohen's d)")
    print("="*80)
    
    def cohens_d(group1, group2):
        """Calculate Cohen's d effect size."""
        diff = group1 - group2
        pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
        return diff.mean() / (pooled_std + 1e-10)
    
    effect_sizes = []
    for (acq, error_model, jitter_std, jitter_iter, oracle), group in merged.groupby(
        ['acquisition', 'error_model', 'jitter_std', 'jitter_iteration', 'oracle_model']
    ):
        d_true = cohens_d(group['objective_true_jitter'], group['objective_true_baseline'])
        d_obs = cohens_d(group['objective_observed_jitter'], group['objective_observed_baseline'])
        
        effect_sizes.append({
            'acquisition': acq,
            'error_model': error_model,
            'jitter_std': jitter_std,
            'jitter_iteration': jitter_iter,
            'oracle_model': oracle,
            'cohens_d_true': d_true,
            'cohens_d_obs': d_obs,
            'n': len(group),
        })
    
    effect_df = pd.DataFrame(effect_sizes)
    
    # Interpret effect sizes
    def interpret_d(d):
        abs_d = abs(d)
        if abs_d >= 0.8:
            return "LARGE"
        elif abs_d >= 0.5:
            return "MEDIUM"
        elif abs_d >= 0.2:
            return "SMALL"
        else:
            return "negligible"
    
    effect_df['interpretation_true'] = effect_df['cohens_d_true'].apply(interpret_d)
    effect_df['interpretation_obs'] = effect_df['cohens_d_obs'].apply(interpret_d)
    
    print("\nEffect Sizes by Condition:")
    print(effect_df.to_string(index=False))
    
    results['effect_sizes'] = effect_df
    
    # ============================================================================
    # 4. DESCRIPTIVE STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS")
    print("="*80)
    
    desc_stats = merged.groupby(['acquisition', 'error_model', 'jitter_std']).agg({
        'true_diff': ['mean', 'std', 'sem', 'count'],
        'obs_diff': ['mean', 'std', 'sem', 'count'],
    }).round(4)
    
    print("\nMean Differences by Condition:")
    print(desc_stats.to_string())
    
    results['descriptive_stats'] = desc_stats
    
    # ============================================================================
    # 5. SAVE RESULTS
    # ============================================================================
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ANOVA tables
    for key, value in results.items():
        if 'anova' in key:
            value.to_csv(output_dir / f"{key}.csv")
    
    # Save effect sizes
    effect_df.to_csv(output_dir / "effect_sizes_cohens_d.csv", index=False)
    
    # Save descriptive stats
    desc_stats.to_csv(output_dir / "descriptive_statistics.csv")
    
    # ============================================================================
    # 6. SIMPLE EFFECTS ANALYSIS (if interaction is significant)
    # ============================================================================
    print("\n" + "="*80)
    print("SIMPLE EFFECTS ANALYSIS")
    print("="*80)
    
    # Only perform if we have multiple error models and jitter_std levels
    if merged['error_model'].nunique() >= 2 and merged['jitter_std'].nunique() >= 2:
        # Test effect of error_model at each level of jitter_std
        for jitter_std_val in sorted(merged['jitter_std'].unique()):
            subset = merged[merged['jitter_std'] == jitter_std_val]
            print(f"\nEffect of error_model at jitter_std={jitter_std_val}:")
            
            for metric in ['true_diff', 'obs_diff']:
                groups = [group[metric].dropna().values for name, group in subset.groupby('error_model')]
                # Only perform if we have at least 2 groups with data
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                        print(f"  {metric}: F={f_stat:.4f}, p={p_val:.4f} {sig}")
                    except Exception as e:
                        print(f"  {metric}: Could not compute F-test ({e})")
                else:
                    print(f"  {metric}: Insufficient groups for comparison")
    else:
        print("\nSkipping simple effects analysis - need multiple error models and jitter_std levels")
    
    print("\n" + "="*80)
    print("Analysis complete. Results saved to:", output_dir)
    print("="*80)
    
    return results


# Additional helper function for reporting
def generate_statistical_report(results: dict, output_dir: Path) -> None:
    """Generate a human-readable statistical report."""
    report_path = output_dir / "statistical_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 80 + "\n")
        f.write("1. Mixed ANOVA with Type II sums of squares\n")
        f.write("2. Post-hoc pairwise comparisons using Tukey HSD\n")
        f.write("3. Effect sizes: η² (eta-squared) and Cohen's d\n")
        f.write("4. Significance level: α = 0.05\n\n")
        
        f.write("INTERPRETATION GUIDELINES:\n")
        f.write("-" * 80 + "\n")
        f.write("Effect Size (Cohen's d):\n")
        f.write("  - Small: 0.2 ≤ |d| < 0.5\n")
        f.write("  - Medium: 0.5 ≤ |d| < 0.8\n")
        f.write("  - Large: |d| ≥ 0.8\n\n")
        f.write("Effect Size (η²):\n")
        f.write("  - Small: 0.01 ≤ η² < 0.06\n")
        f.write("  - Medium: 0.06 ≤ η² < 0.14\n")
        f.write("  - Large: η² ≥ 0.14\n\n")
        
        # Summarize key findings
        if 'effect_sizes' in results:
            effect_df = results['effect_sizes']
            large_effects = effect_df[
                (effect_df['interpretation_true'] == 'LARGE') | 
                (effect_df['interpretation_obs'] == 'LARGE')
            ]
            
            f.write("KEY FINDINGS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total conditions tested: {len(effect_df)}\n")
            f.write(f"Conditions with LARGE effects: {len(large_effects)}\n\n")
            
            if not large_effects.empty:
                f.write("Conditions with largest effects:\n")
                top_effects = large_effects.nlargest(10, 'cohens_d_true')
                f.write(top_effects.to_string(index=False))
        
    print(f"\nStatistical report saved to: {report_path}")

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
    stats = evaluate_final_outcomes_improved(final_outcomes, args.output_dir)
    plot_final_outcome_significance(stats, args.output_dir)
    print(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
