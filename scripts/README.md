# Sensor-error simulation for HITL BO

This folder provides a reproducible, data-driven simulation that answers the following
questions using the eHMI study data:

- **Effects of sensor errors in implicit HITL optimization**: inject Gaussian jitter
  into the feedback signal after a specified iteration and observe how the next
  parameter suggestion changes.
- **Testing 100 iterations (or more/less)**: run iterative Bayesian Optimization
  for a configurable number of iterations.
- **After iteration 20, add artificial jitter to feedback values**: the simulation
  supports `--jitter-iteration 20` and `--jitter-std` for magnitude. Use
  `--single-error` to inject only one error after the jitter iteration.
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

### Baseline (no-jitter) comparison

By default the simulation runs **both** a baseline (no jitter) and a jittered run
for each acquisition and seed. This enables an **excess adjustment** calculation
that isolates the impact of sensor error:

- `bo_sensor_error_summary.csv` includes `baseline=true/false`.
- `bo_sensor_error_excess_summary.csv` reports `delta_excess_<param>` and
  `delta_excess_l2_norm` (jittered − baseline).

Disable the baseline with:

```bash
python scripts/bo_sensor_error_simulation.py --no-baseline-run
```

## Outputs

- `bo_sensor_error_<acq>.csv`
  - Full iteration log: parameter values, true objective, observed objective,
    `error_applied`, and `error_magnitude`.
- `bo_sensor_error_<acq>_seed<seed>_baseline.csv`
- `bo_sensor_error_<acq>_seed<seed>_jittered.csv`
- `bo_sensor_error_summary.csv`
  - One row per acquisition function.
  - `delta_<param>`: change in each parameter from iteration *N* to *N+1*.
  - `delta_l2_norm`: L2 norm of the parameter change (overall adjustment magnitude).
- `bo_sensor_error_excess_summary.csv`
  - Per-acquisition/seed **excess change** (jittered − baseline).
- `bo_sensor_error_summary_stats.csv`
  - Mean/std summary by acquisition and baseline flag.
- `run_metadata.json`
  - CLI args, dataset path, package versions, and total runtime.

### How these outputs answer your questions

**1) “Effects of sensor errors in implicit HITL optimization”**  
Sensor error is simulated by adding Gaussian jitter to the feedback after
`--jitter-iteration`. The per-iteration CSVs (`bo_sensor_error_<acq>.csv`)
contain both `objective_true` (oracle signal) and `objective_observed`
jittered values. Comparing these columns after the jitter point shows how
the optimization is driven by noisy feedback rather than the underlying
oracle signal.

**2) “Testing 100 iterations (more or less) and integrating one simulated error as feedback … what is the parameter value adjustment in the following iteration?”**  
Set `--iterations 100` and `--jitter-iteration 20` to match the scenario.
The summary CSV reports the *parameter adjustment* immediately after the
first noisy feedback is injected:

- `delta_<param>` = (parameter at iteration 21) − (parameter at iteration 20)  
- `delta_l2_norm` = overall magnitude of the change

These deltas quantify how the algorithm changes its suggested parameters
in response to the first noisy observation.

**3) “After iteration 20, add artificial jitter to the feedback values”**  
This is the default behavior when `--jitter-iteration 20` is set. All
iterations > 20 use `objective_observed = objective_true + jitter`.
For a single injected error (only iteration 21), add `--single-error`.

**4) “Testing different acquisition functions”**  
Run `--acq all` to generate `bo_sensor_error_ei.csv`,
`bo_sensor_error_pi.csv`, and `bo_sensor_error_ucb.csv`, plus a
summary row for each. Compare `delta_l2_norm` and per-parameter deltas
across acquisitions to see which method reacts most strongly to sensor
error.

## Objective options

Use `--objective` to control which dataset column(s) are optimized:

- `composite` (default): mean of `Trust`, `Understanding`, `PerceivedSafety`,
  `Aesthetics`, `Acceptance`
- `trust`, `understanding`, `perceived_safety`, `aesthetics`, `acceptance`

### Objective normalization and weighting

Use `--normalize-objective` to scale each objective column to [0, 1] before
aggregation. Use `--objective-weights` to apply weights matching the objective
columns, for example:

```bash
python scripts/bo_sensor_error_simulation.py \
  --objective composite \
  --normalize-objective \
  --objective-weights 0.3,0.2,0.2,0.2,0.1
```

## Data location

By default the script looks for `eHMI-bo-participantdata` in the repository root
relative to the script location. If your data lives elsewhere, pass it explicitly:

```bash
python scripts/bo_sensor_error_simulation.py \
  --data-dir /path/to/eHMI-bo-participantdata
```

### Participant or group filtering

Limit the dataset to a single participant or condition group:

```bash
python scripts/bo_sensor_error_simulation.py --user-id 10
python scripts/bo_sensor_error_simulation.py --group-id 1
```

## Interpreting the parameter adjustment

The summary row answers: **“How much did the suggested parameter vector change right
after the sensor error starts?”**

- The `delta_<param>` values show per-parameter adjustments.
- `delta_l2_norm` provides a single scalar magnitude for the adjustment.

To study stability under sensor error, compare `delta_l2_norm` across acquisition
functions or across multiple seeds.

## Multiple seeds

Use `--seeds` to provide an explicit list, or `--num-seeds` to run sequential seeds:

```bash
python scripts/bo_sensor_error_simulation.py --seeds 7,8,9
python scripts/bo_sensor_error_simulation.py --seed 7 --num-seeds 5
```

The summary stats file reports mean and standard deviation across runs.

## Sensor error models

Use `--error-model` to control the error type after `--jitter-iteration`:

- `gaussian` (default): `objective_observed = true + N(0, jitter_std)`
- `bias`: constant offset `+ error_bias`
- `drift`: linearly increasing offset `+ error_drift * (iteration - jitter_iteration)`
- `dropout`: hold the last observed value (`--dropout-strategy hold_last`)
- `spike`: occasional spikes with probability `error_spike_prob`

Example:

```bash
python scripts/bo_sensor_error_simulation.py \
  --error-model spike \
  --error-spike-prob 0.2 \
  --error-spike-std 0.6
```

## Acquisition hyperparameters

Use `--xi` (EI/PI) and `--kappa` (UCB) to control exploration:

```bash
python scripts/bo_sensor_error_simulation.py --xi 0.02 --kappa 2.5
```

## Plotting

Generate plots from the simulation outputs:

```bash
python scripts/plot_sensor_error_results.py \
  --input-dir output/bo_sensor_error \
  --output-dir output/bo_sensor_error/plots
```

This produces objective trajectory plots per acquisition and a bar chart of
mean `delta_l2_norm`.
