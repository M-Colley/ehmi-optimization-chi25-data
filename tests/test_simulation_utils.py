from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bo_sensor_error_simulation.py"

spec = importlib.util.spec_from_file_location("bo_sim", MODULE_PATH)
bo_sim = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bo_sim)


def make_dummy_df(rows: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    data = {
        col: rng.normal(size=rows)
        for col in bo_sim.PARAM_COLUMNS + bo_sim.OBJECTIVE_MAP["composite"]
    }
    data["User_ID"] = rng.integers(1, 3, size=rows)
    data["Group_ID"] = rng.integers(1, 3, size=rows)
    return pd.DataFrame(data)


def test_parse_objective_list_defaults_to_composite_and_multi() -> None:
    objectives = bo_sim.parse_objective_list(None, None)
    assert objectives == ["composite", "multi_objective"]


def test_parse_oracle_models_default_set() -> None:
    models = bo_sim.parse_oracle_models("xgboost", "random_forest,lightgbm,xgboost")
    assert models == ["random_forest", "lightgbm", "xgboost"]


def test_filter_acquisitions_for_objective() -> None:
    single_acqs = ["ei", "ucb", "qei"]
    assert bo_sim.filter_acquisitions_for_objective(single_acqs, "composite") == single_acqs

    multi_acqs = ["qehvi", "qnehvi"]
    assert bo_sim.filter_acquisitions_for_objective(multi_acqs, "multi_objective") == multi_acqs


def test_compute_reference_point_multi_objective() -> None:
    df = make_dummy_df()
    ref = bo_sim.compute_reference_point(df, "multi_objective")
    assert ref is not None
    assert ref.shape[0] == len(bo_sim.OBJECTIVE_MAP["multi_objective"])


def test_apply_sensor_error_vector_bias() -> None:
    config = bo_sim.SimulationConfig(
        iterations=5,
        jitter_iteration=2,
        jitter_std=0.1,
        single_error=False,
        initial_samples=1,
        candidate_pool=10,
        objective="composite",
        objective_columns=bo_sim.OBJECTIVE_MAP["composite"],
        seed=1,
        error_model="bias",
        error_bias=0.5,
        error_drift=0.01,
        error_spike_prob=0.1,
        error_spike_std=0.2,
        dropout_strategy="hold_last",
        normalize_objective=False,
        objective_weights=None,
        acq_num_restarts=2,
        acq_raw_samples=8,
        acq_maxiter=15,
        acq_mc_samples=32,
        ref_point=None,
    )
    rng = np.random.default_rng(0)
    true_value = np.array([1.0, 2.0])
    observed, error = bo_sim.apply_sensor_error(true_value, 3, config, rng, true_value)
    assert np.allclose(observed, np.array([1.5, 2.5]))
    assert np.allclose(error, np.array([0.5, 0.5]))


def test_oracle_builders_for_key_models() -> None:
    df = make_dummy_df()
    for model_name in ["random_forest", "lightgbm", "xgboost"]:
        oracle = bo_sim.build_oracle(
            df=df,
            objective="composite",
            seed=7,
            normalize=False,
            weights=None,
            oracle_model=model_name,
            oracle_augmentation="none",
            oracle_augment_repeats=0,
            oracle_augment_std=0.0,
            oracle_fast=True,
        )
        pred = oracle.predict(df[bo_sim.PARAM_COLUMNS].iloc[0].to_numpy(dtype=float))
        assert pred.shape == (1,)


def test_augment_oracle_data_jitter() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)
    X_aug, y_aug = bo_sim.augment_oracle_data(X, y, rng, "jitter", repeats=2, noise_std=0.01)
    assert X_aug.shape[0] == 30
    assert y_aug.shape[0] == 30
