# Sensor-error simulation for HITL BO

This folder provides a reproducible, data-driven simulation that answers the following
questions using the eHMI study data:

- **Effects of sensor errors in implicit HITL optimization**: inject Gaussian jitter
  into the feedback signal after a specified iteration and observe how the next
  parameter suggestion changes.
- **Testing 100 iterations (or more/less)**: run iterative Bayesian Optimization
  for a configurable number of iterations.
- **After iteration 20, add artificial jitter to feedback values**: the simulation
  supports `--jitter-iteration 20` and `--jitter-std` for magnitude.
- **Testing different acquisition functions**: run `ei`, `pi`, `ucb`, or all.

## What this simulation does

1. Loads all `ObservationsPerEvaluation.csv` files from `../eHMI-bo-participantdata`.
2. Trains a **Random Forest oracle** to map the 9 eHMI parameters to a target objective
   (composite or single-objective).
3. Runs iterative BO with a **Gaussian Process** surrogate.
4. Injects **sensor error** (Gaussian jitter) into the observed feedback after a chosen
   iteration.
5. Writes a per-iteration CSV and a summary of the **parameter adjustment** from
   iteration *N* to *N+1* (e.g., 20 → 21).

## Install (latest compatible versions)

```bash
python -m pip install --upgrade -r scripts/requirements.txt
```

## Run

```bash
python scripts/bo_sensor_error_simulation.py \
  --iterations 100 \
  --jitter-iteration 20 \
  --jitter-std 0.2 \
  --initial-samples 5 \
  --candidate-pool 1000 \
  --objective composite \
  --acq all \
  --seed 7 \
  --output-dir output/bo_sensor_error
```

## Outputs

- `bo_sensor_error_<acq>.csv`
  - Full iteration log: parameter values, true objective, observed (jittered) objective.
- `bo_sensor_error_summary.csv`
  - One row per acquisition function.
  - `delta_<param>`: change in each parameter from iteration *N* to *N+1*.
  - `delta_l2_norm`: L2 norm of the parameter change (overall adjustment magnitude).

## Objective options

Use `--objective` to control which dataset column(s) are optimized:

- `composite` (default): mean of `Trust`, `Understanding`, `PerceivedSafety`,
  `Aesthetics`, `Acceptance`
- `trust`, `understanding`, `perceived_safety`, `aesthetics`, `acceptance`

## Interpreting the parameter adjustment

The summary row answers: **“How much did the suggested parameter vector change right
after the sensor error starts?”**

- The `delta_<param>` values show per-parameter adjustments.
- `delta_l2_norm` provides a single scalar magnitude for the adjustment.

To study stability under sensor error, compare `delta_l2_norm` across acquisition
functions or across multiple seeds.
