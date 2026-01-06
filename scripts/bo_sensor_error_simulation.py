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
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path("eHMI-bo-participantdata")

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
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser.parse_args()


def load_observations(data_dir: Path, objective: str) -> pd.DataFrame:
    files = list(data_dir.rglob("ObservationsPerEvaluation.csv"))
    if not files:
        raise FileNotFoundError(f"No observation files found in {data_dir}")

    frames = [pd.read_csv(path, sep=";") for path in files]
    df = pd.concat(frames, ignore_index=True)

    for column in PARAM_COLUMNS + OBJECTIVE_MAP[objective]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=PARAM_COLUMNS + OBJECTIVE_MAP[objective])
    df = df.reset_index(drop=True)
    return df


def compute_objective(df: pd.DataFrame, objective: str) -> pd.Series:
    columns = OBJECTIVE_MAP[objective]
    return df[columns].mean(axis=1)


def build_oracle(df: pd.DataFrame, objective: str, seed: int) -> OracleModel:
    X = df[PARAM_COLUMNS].to_numpy()
    y = compute_objective(df, objective).to_numpy()
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
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)


def run_simulation(
    oracle: OracleModel,
    bounds: Bounds,
    config: SimulationConfig,
    acq: AcquisitionConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    X: list[np.ndarray] = []
    y_observed: list[float] = []
    y_true: list[float] = []

    gp = build_gp(config.seed)

    for iteration in range(1, config.iterations + 1):
        if iteration <= config.initial_samples or len(X) < 2:
            candidate = sample_uniform(bounds, rng, size=1)[0]
        else:
            X_array = np.vstack(X)
            y_array = np.array(y_observed)
            X_scaled = scale_inputs(bounds, X_array)
            gp.fit(X_scaled, y_array)

            candidate_pool = sample_uniform(bounds, rng, size=config.candidate_pool)
            pool_scaled = scale_inputs(bounds, candidate_pool)
            mu, std = gp.predict(pool_scaled, return_std=True)
            best = float(np.max(y_array))
            scores = acquisition_scores(acq, mu, std, best)
            candidate = candidate_pool[int(np.argmax(scores))]

        true_value = oracle.predict(candidate)
        jitter = 0.0
        if iteration > config.jitter_iteration:
            jitter = rng.normal(0.0, config.jitter_std)
        observed_value = true_value + jitter

        X.append(candidate)
        y_true.append(true_value)
        y_observed.append(observed_value)

    results = pd.DataFrame(X, columns=PARAM_COLUMNS)
    results.insert(0, "iteration", np.arange(1, config.iterations + 1))
    results["objective_true"] = y_true
    results["objective_observed"] = y_observed
    results["jitter_applied"] = results["iteration"] > config.jitter_iteration
    results["acquisition"] = acq.name
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
    df = load_observations(DATA_DIR, args.objective)
    oracle = build_oracle(df, args.objective, args.seed)
    bounds = bounds_from_data(df)

    config = SimulationConfig(
        iterations=args.iterations,
        jitter_iteration=args.jitter_iteration,
        jitter_std=args.jitter_std,
        initial_samples=args.initial_samples,
        candidate_pool=args.candidate_pool,
        objective=args.objective,
        seed=args.seed,
    )

    if args.acq == "all":
        acquisitions = [AcquisitionConfig("ei"), AcquisitionConfig("pi"), AcquisitionConfig("ucb")]
    else:
        acquisitions = [AcquisitionConfig(args.acq)]

    output_dir = ensure_output_dir(args.output_dir)
    summaries = []

    for index, acq in enumerate(acquisitions):
        run_rng = np.random.default_rng(args.seed + index)
        results = run_simulation(oracle, bounds, config, acq, run_rng)
        results_path = output_dir / f"bo_sensor_error_{acq.name}.csv"
        results.to_csv(results_path, index=False)

        summary = summarize_adjustment(results, args.jitter_iteration)
        summary["acquisition"] = acq.name
        summary["objective"] = args.objective
        summary["jitter_std"] = args.jitter_std
        summary["iterations"] = args.iterations
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "bo_sensor_error_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("Simulation complete.")
    print(f"Results saved to: {output_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
