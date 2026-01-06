"""
Simulate sensor-error impacts in HITL Bayesian optimization using eHMI data.

Example:
  python scripts/bo_sensor_error_simulation.py \
    --iterations 100 \
    --jitter-iteration 20 \
    --acq all \
    --output-dir /tmp/bo_sensor_error_output
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
import uuid
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "eHMI-bo-participantdata"

PARAM_COLUMNS = [
    "verticalPosition",
    "verticalWidth",
    "horizontalWidth",
    "r",
    "g",
    "b",
    "a",
    "blinkFrequency",
    "volume",
]

OBJECTIVE_MAP = {
    "composite": ["Trust", "Understanding", "PerceivedSafety", "Aesthetics", "Acceptance"],
    "trust": ["Trust"],
    "understanding": ["Understanding"],
    "perceived_safety": ["PerceivedSafety"],
    "aesthetics": ["Aesthetics"],
    "acceptance": ["Acceptance"],
}


@dataclasses.dataclass
class Bounds:
    low: np.ndarray
    high: np.ndarray


@dataclasses.dataclass
class AcquisitionConfig:
    name: str
    xi: float = 0.01
    kappa: float = 2.0


@dataclasses.dataclass
class SimulationConfig:
    iterations: int
    jitter_iteration: int
    jitter_std: float
    initial_samples: int
    candidate_pool: int
    objective: str
    seed: int
    error_model: str
    error_bias: float
    error_drift: float
    error_spike_prob: float
    error_spike_std: float
    dropout_strategy: str
    normalize_objective: bool
    objective_weights: np.ndarray | None


@dataclasses.dataclass
class OracleModel:
    model: RandomForestRegressor
    objective_name: str

    def predict(self, x: np.ndarray) -> float:
        return float(self.model.predict(x.reshape(1, -1))[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--jitter-iteration", type=int, default=20)
    parser.add_argument("--jitter-std", type=float, default=0.2)
    parser.add_argument("--initial-samples", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=1000)
    parser.add_argument("--objective", type=str, default="composite", choices=OBJECTIVE_MAP)
    parser.add_argument("--acq", type=str, default="all", choices=["ei", "pi", "ucb", "all"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds.")
    parser.add_argument("--num-seeds", type=int, default=None, help="Number of sequential seeds.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to eHMI-bo-participantdata (defaults to repo root).",
    )
    parser.add_argument("--baseline-run", action="store_true", default=True)
    parser.add_argument("--no-baseline-run", action="store_false", dest="baseline_run")
    parser.add_argument(
        "--error-model",
        type=str,
        default="gaussian",
        choices=["gaussian", "bias", "drift", "dropout", "spike"],
    )
    parser.add_argument("--error-bias", type=float, default=0.2)
    parser.add_argument("--error-drift", type=float, default=0.02)
    parser.add_argument("--error-spike-prob", type=float, default=0.1)
    parser.add_argument("--error-spike-std", type=float, default=0.5)
    parser.add_argument(
        "--dropout-strategy",
        type=str,
        default="hold_last",
        choices=["hold_last"],
    )
    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--group-id", type=str, default=None)
    parser.add_argument("--normalize-objective", action="store_true", default=False)
    parser.add_argument("--objective-weights", type=str, default=None)
    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=2.0)
    return parser.parse_args()


def load_observations(
    data_dir: Path,
    objective: str,
    user_id: str | None = None,
    group_id: str | None = None,
) -> pd.DataFrame:
    files = list(data_dir.rglob("ObservationsPerEvaluation.csv"))
    if not files:
        raise FileNotFoundError(f"No observation files found in {data_dir}")

    frames = [pd.read_csv(path, sep=";") for path in files]
    df = pd.concat(frames, ignore_index=True)

    for column in PARAM_COLUMNS + OBJECTIVE_MAP[objective] + ["User_ID", "Group_ID"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if user_id is not None:
        df = df[df["User_ID"] == float(user_id)]
    if group_id is not None:
        df = df[df["Group_ID"] == float(group_id)]

    df = df.dropna(subset=PARAM_COLUMNS + OBJECTIVE_MAP[objective])
    if df.empty:
        raise ValueError("No data remaining after applying user/group filters.")
    df = df.reset_index(drop=True)
    return df


def compute_objective(
    df: pd.DataFrame,
    objective: str,
    normalize: bool,
    weights: np.ndarray | None,
) -> pd.Series:
    columns = OBJECTIVE_MAP[objective]
    values = df[columns].to_numpy(dtype=float)
    if normalize:
        min_vals = np.nanmin(values, axis=0)
        max_vals = np.nanmax(values, axis=0)
        ranges = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        values = (values - min_vals) / ranges
    if weights is None:
        return pd.Series(values.mean(axis=1), index=df.index)
    weights = weights / np.sum(weights)
    return pd.Series(values @ weights, index=df.index)


def build_oracle(
    df: pd.DataFrame,
    objective: str,
    seed: int,
    normalize: bool,
    weights: np.ndarray | None,
) -> OracleModel:
    X = df[PARAM_COLUMNS].to_numpy()
    y = compute_objective(df, objective, normalize, weights).to_numpy()
    model = RandomForestRegressor(
        n_estimators=600,
        random_state=seed,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(X, y)
    return OracleModel(model=model, objective_name=objective)


def bounds_from_data(df: pd.DataFrame) -> Bounds:
    low = df[PARAM_COLUMNS].min().to_numpy()
    high = df[PARAM_COLUMNS].max().to_numpy()
    return Bounds(low=low, high=high)


def sample_uniform(bounds: Bounds, rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.uniform(bounds.low, bounds.high, size=(size, len(bounds.low)))


def scale_inputs(bounds: Bounds, X: np.ndarray) -> np.ndarray:
    return (X - bounds.low) / (bounds.high - bounds.low)


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        improvement = mu - best - xi
        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma == 0.0] = 0.0
    return ei


def probability_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        z = (mu - best - xi) / sigma
        pi = norm.cdf(z)
    pi[sigma == 0.0] = 0.0
    return pi


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, kappa: float) -> np.ndarray:
    return mu + kappa * sigma


def acquisition_scores(
    acq: AcquisitionConfig,
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
) -> np.ndarray:
    if acq.name == "ei":
        return expected_improvement(mu, sigma, best, acq.xi)
    if acq.name == "pi":
        return probability_improvement(mu, sigma, best, acq.xi)
    if acq.name == "ucb":
        return upper_confidence_bound(mu, sigma, acq.kappa)
    raise ValueError(f"Unknown acquisition: {acq.name}")


def build_gp(seed: int) -> GaussianProcessRegressor:
    kernel = Matern(nu=2.5, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(
        noise_level=1e-5,
        noise_level_bounds=(1e-8, 1e-1),
    )
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)


def parse_seed_list(seed_arg: str | None, seed: int, num_seeds: int | None) -> list[int]:
    if seed_arg:
        return [int(value.strip()) for value in seed_arg.split(",") if value.strip()]
    if num_seeds:
        return list(range(seed, seed + num_seeds))
    return [seed]


def parse_objective_weights(weights_arg: str | None, objective: str) -> np.ndarray | None:
    if weights_arg is None:
        return None
    values = [float(value.strip()) for value in weights_arg.split(",") if value.strip()]
    expected = len(OBJECTIVE_MAP[objective])
    if len(values) != expected:
        raise ValueError(f"Expected {expected} weights for objective {objective}.")
    return np.array(values, dtype=float)


def apply_sensor_error(
    true_value: float,
    iteration: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    previous_observed: float,
) -> tuple[float, float]:
    if iteration <= config.jitter_iteration:
        return true_value, 0.0

    if config.error_model == "gaussian":
        jitter = rng.normal(0.0, config.jitter_std)
        return true_value + jitter, jitter
    if config.error_model == "bias":
        return true_value + config.error_bias, config.error_bias
    if config.error_model == "drift":
        drift = config.error_drift * (iteration - config.jitter_iteration)
        return true_value + drift, drift
    if config.error_model == "dropout":
        if config.dropout_strategy != "hold_last":
            raise ValueError(f"Unsupported dropout strategy: {config.dropout_strategy}")
        return previous_observed, previous_observed - true_value
    if config.error_model == "spike":
        if rng.random() < config.error_spike_prob:
            spike = rng.normal(0.0, config.error_spike_std)
            return true_value + spike, spike
        return true_value, 0.0
    raise ValueError(f"Unknown error model: {config.error_model}")


def validate_inputs(args: argparse.Namespace) -> None:
    if args.iterations <= 1:
        raise ValueError("iterations must be greater than 1.")
    if args.jitter_iteration < 0 or args.jitter_iteration >= args.iterations:
        raise ValueError("jitter-iteration must be within [0, iterations - 1].")
    if args.initial_samples < 1 or args.initial_samples >= args.iterations:
        raise ValueError("initial-samples must be within [1, iterations - 1].")
    if args.candidate_pool < 1:
        raise ValueError("candidate-pool must be >= 1.")
    if args.error_spike_prob < 0 or args.error_spike_prob > 1:
        raise ValueError("error-spike-prob must be between 0 and 1.")


def run_simulation(
    oracle: OracleModel,
    bounds: Bounds,
    config: SimulationConfig,
    acq: AcquisitionConfig,
    rng: np.random.Generator,
    run_id: str,
    apply_error: bool,
) -> pd.DataFrame:
    X: list[np.ndarray] = []
    y_observed: list[float] = []
    y_true: list[float] = []
    error_magnitudes: list[float] = []
    fit_times: list[float] = []

    gp = build_gp(config.seed)
    previous_observed = None

    for iteration in range(1, config.iterations + 1):
        if iteration <= config.initial_samples or len(X) < 2:
            candidate = sample_uniform(bounds, rng, size=1)[0]
            fit_time = 0.0
        else:
            X_array = np.vstack(X)
            y_array = np.array(y_observed)
            X_scaled = scale_inputs(bounds, X_array)
            fit_start = time.perf_counter()
            gp.fit(X_scaled, y_array)
            fit_time = time.perf_counter() - fit_start

            candidate_pool = sample_uniform(bounds, rng, size=config.candidate_pool)
            pool_scaled = scale_inputs(bounds, candidate_pool)
            mu, std = gp.predict(pool_scaled, return_std=True)
            best = float(np.max(y_array))
            scores = acquisition_scores(acq, mu, std, best)
            candidate = candidate_pool[int(np.argmax(scores))]

        true_value = oracle.predict(candidate)
        if previous_observed is None:
            previous_observed = true_value
        if apply_error:
            observed_value, error_magnitude = apply_sensor_error(
                true_value, iteration, config, rng, previous_observed
            )
        else:
            observed_value, error_magnitude = true_value, 0.0

        X.append(candidate)
        y_true.append(true_value)
        y_observed.append(observed_value)
        error_magnitudes.append(error_magnitude)
        fit_times.append(fit_time)
        previous_observed = observed_value

    results = pd.DataFrame(X, columns=PARAM_COLUMNS)
    results.insert(0, "iteration", np.arange(1, config.iterations + 1))
    results["objective_true"] = y_true
    results["objective_observed"] = y_observed
    results["error_applied"] = apply_error & (results["iteration"] > config.jitter_iteration)
    results["error_magnitude"] = error_magnitudes
    results["acquisition"] = acq.name
    results["fit_time_sec"] = fit_times
    results["seed"] = config.seed
    results["run_id"] = run_id
    results["error_model"] = config.error_model if apply_error else "none"
    results["jitter_std"] = config.jitter_std if apply_error else 0.0
    return results


def summarize_adjustment(results: pd.DataFrame, jitter_iteration: int) -> pd.Series:
    if jitter_iteration >= len(results):
        raise ValueError("jitter_iteration must be less than total iterations")

    current = results.loc[results["iteration"] == jitter_iteration, PARAM_COLUMNS].iloc[0]
    next_iter = results.loc[results["iteration"] == jitter_iteration + 1, PARAM_COLUMNS].iloc[0]
    delta = next_iter - current
    l2_norm = float(np.linalg.norm(delta.to_numpy()))

    summary = {f"delta_{col}": float(delta[col]) for col in PARAM_COLUMNS}
    summary["delta_l2_norm"] = l2_norm
    summary["iteration"] = jitter_iteration
    return pd.Series(summary)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    weights = parse_objective_weights(args.objective_weights, args.objective)
    df = load_observations(args.data_dir, args.objective, args.user_id, args.group_id)
    oracle = build_oracle(df, args.objective, args.seed, args.normalize_objective, weights)
    bounds = bounds_from_data(df)

    base_config = SimulationConfig(
        iterations=args.iterations,
        jitter_iteration=args.jitter_iteration,
        jitter_std=args.jitter_std,
        initial_samples=args.initial_samples,
        candidate_pool=args.candidate_pool,
        objective=args.objective,
        seed=args.seed,
        error_model=args.error_model,
        error_bias=args.error_bias,
        error_drift=args.error_drift,
        error_spike_prob=args.error_spike_prob,
        error_spike_std=args.error_spike_std,
        dropout_strategy=args.dropout_strategy,
        normalize_objective=args.normalize_objective,
        objective_weights=weights,
    )

    if args.acq == "all":
        acquisitions = [
            AcquisitionConfig("ei", xi=args.xi, kappa=args.kappa),
            AcquisitionConfig("pi", xi=args.xi, kappa=args.kappa),
            AcquisitionConfig("ucb", xi=args.xi, kappa=args.kappa),
        ]
    else:
        acquisitions = [AcquisitionConfig(args.acq, xi=args.xi, kappa=args.kappa)]

    output_dir = ensure_output_dir(args.output_dir)
    summaries = []
    runtime_start = time.perf_counter()

    seeds = parse_seed_list(args.seeds, args.seed, args.num_seeds)
    for acq in acquisitions:
        for seed in seeds:
            for is_baseline in [True, False]:
                if is_baseline and not args.baseline_run:
                    continue
                run_id = str(uuid.uuid4())
                run_rng = np.random.default_rng(seed)
                config = dataclasses.replace(base_config, seed=seed)
                apply_error = not is_baseline
                run_start = time.perf_counter()
                results = run_simulation(oracle, bounds, config, acq, run_rng, run_id, apply_error)
                run_runtime = time.perf_counter() - run_start
                run_tag = "baseline" if is_baseline else "jittered"
                results_path = output_dir / f"bo_sensor_error_{acq.name}_seed{seed}_{run_tag}.csv"
                results.to_csv(results_path, index=False)

                summary = summarize_adjustment(results, args.jitter_iteration)
                summary["acquisition"] = acq.name
                summary["objective"] = args.objective
                summary["jitter_std"] = results["jitter_std"].iloc[0]
                summary["iterations"] = args.iterations
                summary["seed"] = seed
                summary["run_id"] = run_id
                summary["error_model"] = results["error_model"].iloc[0]
                summary["baseline"] = is_baseline
                summary["xi"] = acq.xi
                summary["kappa"] = acq.kappa
                summary["runtime_sec"] = run_runtime
                summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "bo_sensor_error_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if args.baseline_run:
        jittered = summary_df[summary_df["baseline"] == False]
        baseline = summary_df[summary_df["baseline"] == True]
        merged = jittered.merge(
            baseline,
            on=["acquisition", "objective", "iterations", "seed", "xi", "kappa"],
            suffixes=("_jitter", "_baseline"),
        )
        excess = {}
        for col in PARAM_COLUMNS:
            excess[f"delta_excess_{col}"] = (
                merged[f"delta_{col}_jitter"] - merged[f"delta_{col}_baseline"]
            )
        merged_excess = pd.DataFrame(excess)
        merged_excess["delta_excess_l2_norm"] = np.linalg.norm(
            merged_excess[[f"delta_excess_{col}" for col in PARAM_COLUMNS]].to_numpy(),
            axis=1,
        )
        merged_excess["acquisition"] = merged["acquisition"]
        merged_excess["seed"] = merged["seed"]
        merged_excess["objective"] = merged["objective"]
        merged_excess["error_model"] = merged["error_model_jitter"]
        merged_excess_path = output_dir / "bo_sensor_error_excess_summary.csv"
        merged_excess.to_csv(merged_excess_path, index=False)

    stats = (
        summary_df.groupby(["acquisition", "baseline", "error_model"])
        .agg(
            delta_l2_mean=("delta_l2_norm", "mean"),
            delta_l2_std=("delta_l2_norm", "std"),
            runs=("delta_l2_norm", "count"),
        )
        .reset_index()
    )
    stats_path = output_dir / "bo_sensor_error_summary_stats.csv"
    stats.to_csv(stats_path, index=False)

    metadata_payload = {
        "args": vars(args),
        "package_versions": {
            "numpy": metadata.version("numpy"),
            "pandas": metadata.version("pandas"),
            "scikit-learn": metadata.version("scikit-learn"),
            "scipy": metadata.version("scipy"),
        },
        "runtime_sec": time.perf_counter() - runtime_start,
    }
    def json_fallback(obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2, default=json_fallback))

    print("Simulation complete.")
    print(f"Results saved to: {output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
