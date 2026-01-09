"""
Simulate sensor-error impacts in HITL Bayesian optimization using eHMI data.
(BoTorch Implementation + regret metrics)

Example:
  python scripts/bo_sensor_error_simulation.py \
    --iterations 50 \
    --jitter-iterations 20 \
    --jitter-stds 0.1 \
    --acq ei,ucb \
    --output-dir /tmp/botorch_output
"""
from __future__ import annotations


import os

# Set thread limits BEFORE importing numpy/torch/sklearn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import dataclasses
import json
import time
import uuid 
import warnings
from pathlib import Path

# add near imports
import threading
import queue as py_queue

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    LogProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize

torch.set_default_dtype(torch.float64)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "eHMI-bo-participantdata"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

ACQUISITION_CHOICES = ["logei", "logpi", "ucb", "greedy"]
ERROR_MODEL_CHOICES = ["gaussian", "bias", "drift", "dropout", "spike"]
ORACLE_MODEL_CHOICES = [
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost",
    "lightgbm",
]


@dataclasses.dataclass
class Bounds:
    low: np.ndarray
    high: np.ndarray

    @property
    def tensor(self) -> torch.Tensor:
        low_t = torch.from_numpy(self.low)
        high_t = torch.from_numpy(self.high)
        return torch.stack([low_t, high_t])


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
    single_error: bool
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

    # BoTorch optimization controls
    acq_num_restarts: int
    acq_raw_samples: int
    acq_maxiter: int


@dataclasses.dataclass
class OracleModel:
    model: object
    objective_name: str

    def predict(self, x: np.ndarray) -> float:
        return float(self.model.predict(x.reshape(1, -1))[0])

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)

    parser.add_argument("--jitter-iteration", type=int, default=20)
    parser.add_argument("--jitter-std", type=float, default=0.2)
    parser.add_argument(
        "--single-error",
        action="store_true",
        default=False,
        help="Apply sensor error only once at the first iteration after jitter-iteration.",
    )
    # TODO - add more
    parser.add_argument("--jitter-iterations", type=str, default="10,40")
    parser.add_argument("--jitter-stds", type=str, default="0.05,0.5,1,5")

    parser.add_argument("--initial-samples", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=1000)  # kept for compatibility

    parser.add_argument("--objective", type=str, default="composite", choices=OBJECTIVE_MAP)

    parser.add_argument("--acq", type=str, default="all", choices=ACQUISITION_CHOICES + ["all"])
    parser.add_argument("--acq-list", type=str, default=None)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=str, default=None) 
    parser.add_argument("--num-seeds", type=int, default=None) # use this to make statistical data reliable -> use 10+

    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)

    parser.add_argument("--baseline-run", action="store_true", default=True)
    parser.add_argument("--no-baseline-run", action="store_false", dest="baseline_run")

    parser.add_argument("--error-model", type=str, default="gaussian", choices=ERROR_MODEL_CHOICES + ["all"])
    parser.add_argument("--error-models", type=str, default=None)

    parser.add_argument("--error-bias", type=float, default=0.2)
    parser.add_argument("--error-drift", type=float, default=0.02)
    parser.add_argument("--error-spike-prob", type=float, default=0.1)
    parser.add_argument("--error-spike-std", type=float, default=0.5)
    parser.add_argument("--dropout-strategy", type=str, default="hold_last", choices=["hold_last"])

    parser.add_argument("--user-id", type=str, default=None)
    parser.add_argument("--group-id", type=str, default=None)

    parser.add_argument("--normalize-objective", action="store_true", default=False)
    parser.add_argument("--objective-weights", type=str, default=None)

    parser.add_argument("--xi", type=float, default=0.01)
    parser.add_argument("--kappa", type=float, default=2.0)

    parser.add_argument("--oracle-model", type=str, default="xgboost", choices=ORACLE_MODEL_CHOICES + ["all"])
    parser.add_argument("--oracle-models", type=str, default=None)

    # Oracle optimum approximation for regret
    parser.add_argument("--oracle-opt-samples", type=int, default=200_000)
    parser.add_argument("--oracle-opt-batch-size", type=int, default=50_000)
    parser.add_argument("--oracle-opt-seed", type=int, default=10_007)

    # BoTorch acquisition optimization controls
    parser.add_argument("--acq-num-restarts", type=int, default=10)
    parser.add_argument("--acq-raw-samples", type=int, default=512)
    parser.add_argument("--acq-maxiter", type=int, default=200)
    
    parser.add_argument("--parallel",action="store_true", default=False,
    help="Enable parallel processing (auto-enabled for multiple seeds)",)
    parser.add_argument("--n-jobs", type=int, default=-1,
    help="Number of parallel jobs (-1 = all cores, -2 = all but one)",)

    return parser.parse_args()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    return df.reset_index(drop=True)


def compute_objective(
    df: pd.DataFrame,
    objective: str,
    normalize: bool,
    weights: np.ndarray | None,
) -> pd.Series:
    cols = OBJECTIVE_MAP[objective]
    values = df[cols].to_numpy(dtype=float)

    if normalize:
        min_vals = np.nanmin(values, axis=0)
        max_vals = np.nanmax(values, axis=0)
        ranges = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        values = (values - min_vals) / ranges

    if weights is None:
        return pd.Series(values.mean(axis=1), index=df.index)

    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)
    return pd.Series(values @ weights, index=df.index)


def parse_objective_weights(weights_arg: str | None, objective: str) -> np.ndarray | None:
    if weights_arg is None:
        return None
    values = [float(v.strip()) for v in weights_arg.split(",") if v.strip()]
    expected = len(OBJECTIVE_MAP[objective])
    if len(values) != expected:
        raise ValueError(f"Expected {expected} weights for objective={objective}, got {len(values)}.")
    return np.array(values, dtype=float)


def build_oracle(
    df: pd.DataFrame,
    objective: str,
    seed: int,
    normalize: bool,
    weights: np.ndarray | None,
    oracle_model: str,
) -> OracleModel:
    X = df[PARAM_COLUMNS].to_numpy(dtype=float)
    y = compute_objective(df, objective, normalize, weights).to_numpy(dtype=float)

    if oracle_model == "random_forest":
        model = RandomForestRegressor(
            n_estimators=600, random_state=seed, min_samples_leaf=2, n_jobs=1
        )
    elif oracle_model == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=600, random_state=seed, min_samples_leaf=2, n_jobs=1
        )
    elif oracle_model == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=seed
        )
    elif oracle_model == "hist_gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=6, random_state=seed
        )
    elif oracle_model == "xgboost":
        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
        )
    elif oracle_model == "lightgbm":
        model = LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown oracle model: {oracle_model}")

    model.fit(X, y)
    
    # Report oracle performance
    train_score = model.score(X, y)
    y_pred = model.predict(X)
    train_rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"Oracle ({oracle_model}) trained on {len(X)} samples:")
    print(f"  R² score: {train_score:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    
    return OracleModel(model=model, objective_name=objective)


def bounds_from_data(df: pd.DataFrame) -> Bounds:
    low = df[PARAM_COLUMNS].min().to_numpy(dtype=float)
    high = df[PARAM_COLUMNS].max().to_numpy(dtype=float)
    return Bounds(low=low, high=high)


def sample_uniform(bounds: Bounds, rng: np.random.Generator, size: int) -> np.ndarray:
    d = len(bounds.low)
    return rng.uniform(bounds.low, bounds.high, size=(size, d))


def estimate_oracle_optimum(
    oracle: OracleModel,
    bounds: Bounds,
    seed: int,
    n: int,
    batch_size: int,
) -> float:
    rng = np.random.default_rng(seed)
    best = -np.inf
    d = len(bounds.low)

    remaining = int(n)
    while remaining > 0:
        m = min(batch_size, remaining)
        X = rng.uniform(bounds.low, bounds.high, size=(m, d))
        y = oracle.predict_many(X)
        best = max(best, float(np.max(y)))
        remaining -= m

    return best


def parse_seed_list(seed_arg: str | None, seed: int, num_seeds: int | None) -> list[int]:
    if seed_arg:
        values = [int(v.strip()) for v in seed_arg.split(",") if v.strip()]
        if not values:
            raise ValueError("No valid seeds parsed from --seeds.")
        return values
    if num_seeds:
        return list(range(seed, seed + num_seeds))
    return [seed]


def parse_acquisition_list(acq_arg: str, acq_list: str | None) -> list[str]:
    raw = acq_list or acq_arg
    if raw == "all":
        return ACQUISITION_CHOICES
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one acquisition must be specified.")
    unknown = [v for v in values if v not in ACQUISITION_CHOICES]
    if unknown:
        raise ValueError(f"Unknown acquisition(s): {', '.join(unknown)}")
    return values


def parse_error_models(error_model: str, error_models: str | None) -> list[str]:
    raw = error_models or error_model
    if raw == "all":
        return ERROR_MODEL_CHOICES
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one error model must be specified.")
    unknown = [v for v in values if v not in ERROR_MODEL_CHOICES]
    if unknown:
        raise ValueError(f"Unknown error model(s): {', '.join(unknown)}")
    return values


def parse_oracle_models(oracle_model: str, oracle_models: str | None) -> list[str]:
    raw = oracle_models or oracle_model
    if raw == "all":
        return ORACLE_MODEL_CHOICES
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one oracle model must be specified.")
    unknown = [v for v in values if v not in ORACLE_MODEL_CHOICES]
    if unknown:
        raise ValueError(f"Unknown oracle model(s): {', '.join(unknown)}")
    return values


def parse_float_list(value: str | None, default: float) -> list[float]:
    if value is None:
        return [default]
    values = [float(v.strip()) for v in value.split(",") if v.strip()]
    return values or [default]


def parse_int_list(value: str | None, default: int) -> list[int]:
    if value is None:
        return [default]
    values = [int(v.strip()) for v in value.split(",") if v.strip()]
    return values or [default]


def validate_sweeps(jitter_iterations: list[int], jitter_stds: list[float], iterations: int) -> None:
    for j in jitter_iterations:
        if j < 1 or j >= iterations:
            raise ValueError("Each jitter-iteration must be within [1, iterations - 1].")
    for s in jitter_stds:
        if s < 0:
            raise ValueError("Each jitter-std must be >= 0.")


def validate_inputs(args: argparse.Namespace) -> None:
    if args.iterations <= 1:
        raise ValueError("iterations must be greater than 1.")
    if args.initial_samples < 1 or args.initial_samples >= args.iterations:
        raise ValueError("initial-samples must be within [1, iterations - 1].")
    if args.error_spike_prob < 0 or args.error_spike_prob > 1:
        raise ValueError("error-spike-prob must be between 0 and 1.")
    if args.oracle_opt_samples < 10_000:
        raise ValueError("oracle-opt-samples should be reasonably large (>= 10000).")
    if args.oracle_opt_batch_size < 1:
        raise ValueError("oracle-opt-batch-size must be >= 1.")


def apply_sensor_error(
    true_value: float,
    iteration: int,
    config: SimulationConfig,
    rng: np.random.Generator,
    previous_observed: float,
) -> tuple[float, float]:
    if iteration <= config.jitter_iteration:
        return true_value, 0.0
    if config.single_error and iteration != config.jitter_iteration + 1:
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


def get_botorch_candidate(
    gp_model: SingleTaskGP,
    acq_config: AcquisitionConfig,
    bounds_tensor: torch.Tensor,
    best_f: float,
    num_restarts: int,
    raw_samples: int,
    maxiter: int,
) -> torch.Tensor:
    if acq_config.name == "logei":
        acqf = LogExpectedImprovement(model=gp_model, best_f=best_f + acq_config.xi)
    elif acq_config.name == "logpi":
        acqf = LogProbabilityOfImprovement(model=gp_model, best_f=best_f + acq_config.xi)
    elif acq_config.name == "ucb":
        acqf = UpperConfidenceBound(model=gp_model, beta=float(acq_config.kappa**2))
    elif acq_config.name == "greedy":
        acqf = UpperConfidenceBound(model=gp_model, beta=0.0)
    else:
        raise ValueError(f"Unknown acquisition: {acq_config.name}")

    candidate, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds_tensor,
        q=1,
        num_restarts=int(num_restarts),
        raw_samples=int(raw_samples),
        options={"batch_limit": 5, "maxiter": int(maxiter)},
    )
    return candidate.detach()


def run_simulation(
    oracle: OracleModel,
    bounds: Bounds,
    config: SimulationConfig,
    acq: AcquisitionConfig,
    rng: np.random.Generator,
    jitter_rng: np.random.Generator | None,
    run_id: str,
    apply_error: bool,
    oracle_model: str,
    y_opt: float,
) -> pd.DataFrame:
    X_list: list[np.ndarray] = []
    y_observed_list: list[float] = []
    y_true_list: list[float] = []
    error_magnitudes: list[float] = []
    fit_times: list[float] = []

    # Regret tracking (computed on true objective)
    best_true_so_far = -np.inf
    cum_regret = 0.0
    best_true_list: list[float] = []
    regret_inst_list: list[float] = []
    regret_cum_list: list[float] = []
    simple_regret_list: list[float] = []

    bounds_tensor = bounds.tensor
    previous_observed = None

    for iteration in range(1, config.iterations + 1):
        if iteration <= config.initial_samples:
            candidate_np = sample_uniform(bounds, rng, size=1)[0]
            fit_time = 0.0
        else:
            fit_start = time.perf_counter()

            train_X = torch.tensor(np.vstack(X_list), dtype=torch.double)
            train_Y = torch.tensor(np.array(y_observed_list, dtype=float).reshape(-1, 1), dtype=torch.double)

            gp = SingleTaskGP(
                train_X,
                train_Y,
                input_transform=Normalize(d=train_X.shape[-1], bounds=bounds_tensor),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            best_f = train_Y.max().item()

            try:
                candidate_tensor = get_botorch_candidate(
                    gp_model=gp,
                    acq_config=acq,
                    bounds_tensor=bounds_tensor,
                    best_f=float(best_f),
                    num_restarts=config.acq_num_restarts,
                    raw_samples=config.acq_raw_samples,
                    maxiter=config.acq_maxiter,
                )
                candidate_np = candidate_tensor.cpu().numpy().flatten()
            except Exception as e:
                warnings.warn(f"BoTorch optimization failed, falling back to random. Error: {e}")
                candidate_np = sample_uniform(bounds, rng, size=1)[0]

            fit_time = time.perf_counter() - fit_start

        true_value = oracle.predict(candidate_np)

        if previous_observed is None:
            previous_observed = true_value

        if apply_error:
            if jitter_rng is None:
                raise ValueError("jitter_rng must be provided when apply_error is True.")
            observed_value, error_magnitude = apply_sensor_error(
                true_value=true_value,
                iteration=iteration,
                config=config,
                rng=jitter_rng,
                previous_observed=previous_observed,
            )
        else:
            observed_value, error_magnitude = true_value, 0.0

        X_list.append(candidate_np)
        y_true_list.append(true_value)
        y_observed_list.append(observed_value)
        error_magnitudes.append(error_magnitude)
        fit_times.append(fit_time)
        previous_observed = observed_value

        best_true_so_far = max(best_true_so_far, true_value)
        r_t = max(0.0, y_opt - true_value)
        cum_regret += r_t
        s_t = max(0.0, y_opt - best_true_so_far)

        best_true_list.append(best_true_so_far)
        regret_inst_list.append(r_t)
        regret_cum_list.append(cum_regret)
        simple_regret_list.append(s_t)

    results = pd.DataFrame(X_list, columns=PARAM_COLUMNS)
    results.insert(0, "iteration", np.arange(1, config.iterations + 1))

    results["objective_true"] = y_true_list
    results["objective_observed"] = y_observed_list

    if apply_error:
        if config.single_error:
            results["error_applied"] = results["iteration"] == config.jitter_iteration + 1
        else:
            results["error_applied"] = results["iteration"] > config.jitter_iteration
    else:
        results["error_applied"] = False

    results["error_magnitude"] = error_magnitudes
    results["acquisition"] = acq.name
    results["fit_time_sec"] = fit_times
    results["seed"] = config.seed
    results["run_id"] = run_id
    results["error_model"] = config.error_model if apply_error else "none"
    results["jitter_std"] = config.jitter_std if apply_error else 0.0
    results["jitter_iteration"] = config.jitter_iteration
    results["oracle_model"] = oracle_model

    # Regret columns
    results["y_opt"] = float(y_opt)
    results["best_true_so_far"] = best_true_list
    results["regret_inst_true"] = regret_inst_list
    results["regret_cum_true"] = regret_cum_list
    results["simple_regret_true"] = simple_regret_list

    return results


def summarize_adjustment(results: pd.DataFrame, jitter_iteration: int) -> pd.Series:
    max_iter = int(results["iteration"].max())
    if jitter_iteration < 1 or jitter_iteration >= max_iter:
        raise ValueError("jitter_iteration must be within [1, max_iteration - 1].")

    current = results.loc[results["iteration"] == jitter_iteration, PARAM_COLUMNS].iloc[0]
    nxt = results.loc[results["iteration"] == jitter_iteration + 1, PARAM_COLUMNS].iloc[0]
    delta = nxt - current
    l2_norm = float(np.linalg.norm(delta.to_numpy()))

    summary = {f"delta_{col}": float(delta[col]) for col in PARAM_COLUMNS}
    summary["delta_l2_norm"] = l2_norm
    summary["iteration"] = jitter_iteration

    # Add run-level regret summaries (repeated per jitter_iteration row)
    summary["final_best_true"] = float(results["best_true_so_far"].iloc[-1])
    summary["final_simple_regret_true"] = float(results["simple_regret_true"].iloc[-1])
    summary["final_cum_regret_true"] = float(results["regret_cum_true"].iloc[-1])

    sr = results["simple_regret_true"].to_numpy(dtype=float)
    summary["auc_simple_regret_true"] = float(np.trapezoid(sr, dx=1.0))

    return pd.Series(summary)


def run_single_seed(
    seed: int,
    oracle_models: list[str],
    acquisitions: list[AcquisitionConfig],
    error_models: list[str],
    jitter_stds: list[float],
    jitter_iterations: list[int],
    df: pd.DataFrame,
    bounds: Bounds,
    args: argparse.Namespace,
    weights: np.ndarray | None,
    progress_q: object | None = None,          # multiprocessing.Manager().Queue() in parallel mode
    progress_update: callable | None = None,   # tqdm.update in sequential mode
) -> tuple[list[pd.Series], int]:
    """Run all simulations for a single seed."""
    summaries: list[pd.Series] = []
    run_count = 0

    def _tick() -> None:
        nonlocal run_count
        run_count += 1
        if progress_q is not None:
            try:
                progress_q.put(1)
            except Exception:
                pass
        if progress_update is not None:
            try:
                progress_update(1)
            except Exception:
                pass

    for oracle_model in oracle_models:
        oracle = build_oracle(
            df=df,
            objective=args.objective,
            seed=seed,  # use the actual seed for this run
            normalize=args.normalize_objective,
            weights=weights,
            oracle_model=oracle_model,
        )

        y_opt = estimate_oracle_optimum(
            oracle=oracle,
            bounds=bounds,
            seed=args.oracle_opt_seed,
            n=args.oracle_opt_samples,
            batch_size=args.oracle_opt_batch_size,
        )

        base_config = SimulationConfig(
            iterations=args.iterations,
            jitter_iteration=args.jitter_iteration,
            jitter_std=args.jitter_std,
            single_error=args.single_error,
            initial_samples=args.initial_samples,
            candidate_pool=args.candidate_pool,
            objective=args.objective,
            seed=seed,
            error_model=args.error_model,
            error_bias=args.error_bias,
            error_drift=args.error_drift,
            error_spike_prob=args.error_spike_prob,
            error_spike_std=args.error_spike_std,
            dropout_strategy=args.dropout_strategy,
            normalize_objective=args.normalize_objective,
            objective_weights=weights,
            acq_num_restarts=args.acq_num_restarts,
            acq_raw_samples=args.acq_raw_samples,
            acq_maxiter=args.acq_maxiter,
        )

        for acq in acquisitions:
            # Baseline run
            if args.baseline_run:
                baseline_run_id = str(uuid.uuid4())
                run_rng = np.random.default_rng(seed)
                torch.manual_seed(seed)

                config = dataclasses.replace(base_config, seed=seed)
                run_start = time.perf_counter()
                baseline_results = run_simulation(
                    oracle=oracle,
                    bounds=bounds,
                    config=config,
                    acq=acq,
                    rng=run_rng,
                    jitter_rng=None,
                    run_id=baseline_run_id,
                    apply_error=False,
                    oracle_model=oracle_model,
                    y_opt=y_opt,
                )
                baseline_runtime = time.perf_counter() - run_start

                results_path = args.output_dir / f"bo_sensor_error_{acq.name}_seed{seed}_baseline_{oracle_model}.csv"
                baseline_results.to_csv(results_path, index=False)

                for jitter_iteration in jitter_iterations:
                    summary = summarize_adjustment(baseline_results, jitter_iteration)
                    summary["acquisition"] = acq.name
                    summary["objective"] = args.objective
                    summary["jitter_std"] = float(baseline_results["jitter_std"].iloc[0])
                    summary["jitter_iteration"] = int(jitter_iteration)
                    summary["iterations"] = int(args.iterations)
                    summary["seed"] = int(seed)
                    summary["run_id"] = baseline_run_id
                    summary["error_model"] = str(baseline_results["error_model"].iloc[0])
                    summary["oracle_model"] = oracle_model
                    summary["baseline"] = True
                    summary["xi"] = float(acq.xi)
                    summary["kappa"] = float(acq.kappa)
                    summary["runtime_sec"] = float(baseline_runtime)
                    summary["y_opt"] = float(y_opt)
                    summaries.append(summary)

                _tick()

            # Jittered runs
            for error_model in error_models:
                for jitter_std in jitter_stds:
                    for jitter_iteration in jitter_iterations:
                        run_id = str(uuid.uuid4())
                        run_rng = np.random.default_rng(seed)
                        torch.manual_seed(seed)

                        jitter_seed = np.random.SeedSequence(
                            [
                                seed,
                                ACQUISITION_CHOICES.index(acq.name),
                                int(jitter_iteration),
                                int(round(float(jitter_std) * 1_000_000)),
                                ERROR_MODEL_CHOICES.index(error_model),
                            ]
                        )
                        jitter_rng = np.random.default_rng(jitter_seed)

                        config = dataclasses.replace(
                            base_config,
                            seed=seed,
                            error_model=error_model,
                            jitter_std=float(jitter_std),
                            jitter_iteration=int(jitter_iteration),
                        )

                        run_start = time.perf_counter()
                        results = run_simulation(
                            oracle=oracle,
                            bounds=bounds,
                            config=config,
                            acq=acq,
                            rng=run_rng,
                            jitter_rng=jitter_rng,
                            run_id=run_id,
                            apply_error=True,
                            oracle_model=oracle_model,
                            y_opt=y_opt,
                        )
                        run_runtime = time.perf_counter() - run_start

                        results_path = args.output_dir / (
                            f"bo_sensor_error_{acq.name}_seed{seed}_jittered_{oracle_model}_"
                            f"{error_model}_jit{jitter_iteration}_std{jitter_std}.csv"
                        )
                        results.to_csv(results_path, index=False)

                        summary = summarize_adjustment(results, int(jitter_iteration))
                        summary["acquisition"] = acq.name
                        summary["objective"] = args.objective
                        summary["jitter_std"] = float(results["jitter_std"].iloc[0])
                        summary["jitter_iteration"] = int(jitter_iteration)
                        summary["iterations"] = int(args.iterations)
                        summary["seed"] = int(seed)
                        summary["run_id"] = run_id
                        summary["error_model"] = str(results["error_model"].iloc[0])
                        summary["oracle_model"] = oracle_model
                        summary["baseline"] = False
                        summary["xi"] = float(acq.xi)
                        summary["kappa"] = float(acq.kappa)
                        summary["runtime_sec"] = float(run_runtime)
                        summary["y_opt"] = float(y_opt)
                        summaries.append(summary)

                        _tick()

    return summaries, run_count



def main() -> None:
    args = parse_args()
    validate_inputs(args)

    weights = parse_objective_weights(args.objective_weights, args.objective)

    df = load_observations(args.data_dir, args.objective, args.user_id, args.group_id)
    bounds = bounds_from_data(df)

    acquisition_names = parse_acquisition_list(args.acq, args.acq_list)
    acquisitions = [AcquisitionConfig(name=n, xi=args.xi, kappa=args.kappa) for n in acquisition_names]

    error_models = parse_error_models(args.error_model, args.error_models)
    jitter_stds = parse_float_list(args.jitter_stds, args.jitter_std)
    jitter_iterations = parse_int_list(args.jitter_iterations, args.jitter_iteration)
    validate_sweeps(jitter_iterations, jitter_stds, args.iterations)

    oracle_models = parse_oracle_models(args.oracle_model, args.oracle_models)
    seeds = parse_seed_list(args.seeds, args.seed, args.num_seeds)

    output_dir = ensure_output_dir(args.output_dir)
    runtime_start = time.perf_counter()

    baseline_runs = len(acquisitions) * len(seeds) * len(oracle_models) if args.baseline_run else 0
    jittered_runs = (
        len(acquisitions)
        * len(seeds)
        * len(oracle_models)
        * len(error_models)
        * len(jitter_stds)
        * len(jitter_iterations)
    )
    total_runs = baseline_runs + jittered_runs

    use_parallel = args.parallel or (len(seeds) > 1 and not args.parallel)

    if args.n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif args.n_jobs == -2:
        n_jobs = max(1, mp.cpu_count() - 1)
    elif args.n_jobs > 0:
        n_jobs = min(args.n_jobs, mp.cpu_count())
    else:
        n_jobs = 1
        use_parallel = False

    n_jobs = min(n_jobs, len(seeds))

    print(f"Running {len(seeds)} seed(s) with {total_runs} total simulation runs")
    if use_parallel and len(seeds) > 1:
        print(f"Using parallel processing with {n_jobs} worker(s)")
    else:
        print("Using sequential processing")

    summaries: list[pd.Series] = []

    progress = tqdm(total=total_runs, desc="Simulation runs", unit="run")

    if use_parallel and len(seeds) > 1:
        import threading

        manager = mp.Manager()
        progress_q = manager.Queue()

        def _progress_monitor(q, pbar):
            while True:
                msg = q.get()
                if msg is None:
                    break
                try:
                    pbar.update(int(msg))
                except Exception:
                    pass

        monitor = threading.Thread(target=_progress_monitor, args=(progress_q, progress), daemon=True)
        monitor.start()

        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(
                        run_single_seed,
                        seed,
                        oracle_models,
                        acquisitions,
                        error_models,
                        jitter_stds,
                        jitter_iterations,
                        df,
                        bounds,
                        args,
                        weights,
                        progress_q,
                        None,
                    ): seed
                    for seed in seeds
                }

                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        seed_summaries, _ = future.result()
                        summaries.extend(seed_summaries)
                    except Exception as e:
                        print(f"\nError processing seed {seed}: {e}")
                        import traceback
                        traceback.print_exc()
        finally:
            try:
                progress_q.put(None)
            except Exception:
                pass
            try:
                monitor.join(timeout=10)
            except Exception:
                pass
            try:
                progress.close()
            finally:
                try:
                    manager.shutdown()
                except Exception:
                    pass

    else:
        try:
            for seed in seeds:
                seed_summaries, _ = run_single_seed(
                    seed,
                    oracle_models,
                    acquisitions,
                    error_models,
                    jitter_stds,
                    jitter_iterations,
                    df,
                    bounds,
                    args,
                    weights,
                    progress_q=None,
                    progress_update=progress.update,
                )
                summaries.extend(seed_summaries)
        finally:
            progress.close()

    if not summaries:
        print("No simulation runs completed.")
        return

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "bo_sensor_error_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if args.baseline_run:
        jittered = summary_df[summary_df["baseline"] == False]
        baseline = summary_df[summary_df["baseline"] == True]

        merged = jittered.merge(
            baseline,
            on=["acquisition", "objective", "iterations", "jitter_iteration", "seed", "oracle_model", "xi", "kappa"],
            suffixes=("_jitter", "_baseline"),
        )

        excess = {}
        for col in PARAM_COLUMNS:
            excess[f"delta_excess_{col}"] = merged[f"delta_{col}_jitter"] - merged[f"delta_{col}_baseline"]

        merged_excess = pd.DataFrame(excess)
        merged_excess["delta_excess_l2_norm"] = np.linalg.norm(
            merged_excess[[f"delta_excess_{col}" for col in PARAM_COLUMNS]].to_numpy(dtype=float),
            axis=1,
        )
        merged_excess["acquisition"] = merged["acquisition"]
        merged_excess["seed"] = merged["seed"]
        merged_excess["objective"] = merged["objective"]
        merged_excess["oracle_model"] = merged["oracle_model"]
        merged_excess["error_model"] = merged["error_model_jitter"]
        merged_excess["jitter_std"] = merged["jitter_std_jitter"]
        merged_excess["jitter_iteration"] = merged["jitter_iteration"]

        merged_excess["final_simple_regret_excess_true"] = (
            merged["final_simple_regret_true_jitter"] - merged["final_simple_regret_true_baseline"]
        )
        merged_excess["final_cum_regret_excess_true"] = (
            merged["final_cum_regret_true_jitter"] - merged["final_cum_regret_true_baseline"]
        )
        merged_excess["auc_simple_regret_excess_true"] = (
            merged["auc_simple_regret_true_jitter"] - merged["auc_simple_regret_true_baseline"]
        )

        merged_excess_path = output_dir / "bo_sensor_error_excess_summary.csv"
        merged_excess.to_csv(merged_excess_path, index=False)

    stats = (
        summary_df.groupby(["acquisition", "baseline", "error_model", "jitter_iteration", "jitter_std", "oracle_model"])
        .agg(
            delta_l2_mean=("delta_l2_norm", "mean"),
            delta_l2_std=("delta_l2_norm", "std"),
            final_simple_regret_mean=("final_simple_regret_true", "mean"),
            final_simple_regret_std=("final_simple_regret_true", "std"),
            final_cum_regret_mean=("final_cum_regret_true", "mean"),
            final_cum_regret_std=("final_cum_regret_true", "std"),
            auc_simple_regret_mean=("auc_simple_regret_true", "mean"),
            auc_simple_regret_std=("auc_simple_regret_true", "std"),
            runs=("delta_l2_norm", "count"),
        )
        .reset_index()
    )
    stats_path = output_dir / "bo_sensor_error_summary_stats.csv"
    stats.to_csv(stats_path, index=False)

    metadata_payload = {
        "args": vars(args),
        "runtime_sec": float(time.perf_counter() - runtime_start),
        "parallel_execution": bool(use_parallel and len(seeds) > 1),
        "n_workers": int(n_jobs) if (use_parallel and len(seeds) > 1) else 1,
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

    total_runtime = time.perf_counter() - runtime_start
    print("\nSimulation complete.")
    print(f"Total runtime: {total_runtime:.1f}s")
    if use_parallel and len(seeds) > 1:
        print(f"Parallel speedup with {n_jobs} workers")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    # Set multiprocessing start method for Windows compatibility
    if os.name == 'nt':  # Windows
        mp.set_start_method('spawn', force=True)
    
    main()
