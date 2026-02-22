"""
Generate golden test fixtures for Rust e-value engine tests.

These fixtures use the Python reference implementations from
expectation.modules.martingales to produce ground-truth values
that the Rust implementation must match (within 1e-13 tolerance).

Fixtures generated:
  1. golden_one_sided_normal.json   -- One-sided GREATER, ALL_IN combiner
  2. golden_conservative_combiner.json -- Two-sided, CONSERVATIVE combiner (lambda=0.5)
  3. golden_adaptive_combiner.json  -- Two-sided, EMPIRICALLY_ADAPTIVE combiner (gamma=0.5)
  4. golden_less_alternative.json   -- One-sided LESS alternative

References:
  - Time-uniform, nonparametric, nonasymptotic confidence sequences,
    Howard, Ramdas, McAuliffe, Sekhon (2022), Section 3
  - Hypothesis testing with e-values, Ramdas, Wang (2025), Chapter 7
"""

import json
import os
import sys
import numpy as np

# Ensure the project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from expectation.modules.martingales import OneSidedNormalMixture, TwoSidedNormalMixture

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))
N_STEPS = 50
SEED = 42
SHIFT = 0.3


def generate_observations(seed: int, shift: float, n: int) -> np.ndarray:
    """Generate reproducible observations: N(shift, 1) using legacy RandomState."""
    rng = np.random.RandomState(seed=seed)
    return rng.randn(n) + shift


def to_float(val):
    """Convert numpy scalar to Python float for JSON serialization."""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    return val


# ---------------------------------------------------------------------------
# 1. golden_one_sided_normal.json -- One-sided GREATER, ALL_IN combiner
# ---------------------------------------------------------------------------
def generate_one_sided_greater():
    v_opt = 1.0
    alpha_opt = 0.05
    known_variance = 1.0
    null_value = 0.0

    mixture = OneSidedNormalMixture(v_opt=v_opt, alpha_opt=alpha_opt)
    rho = float(mixture.rho)
    observations = generate_observations(SEED, SHIFT, N_STEPS)

    steps = []
    data_sum = 0.0
    prev_log_e_cum = 0.0  # log(1) = 0 at step 0

    for t in range(1, N_STEPS + 1):
        x = observations[t - 1]
        data_sum += x
        count = t
        s = data_sum - null_value * count  # For GREATER: s = data_sum - mu_0 * n
        v = count * known_variance

        log_e_cum = mixture.log_superMG(s, v)
        log_e_sequential = log_e_cum - prev_log_e_cum

        steps.append({
            "t": t,
            "x": to_float(x),
            "data_sum": to_float(data_sum),
            "s": to_float(s),
            "v": to_float(v),
            "log_e_cum": to_float(log_e_cum),
            "log_e_sequential": to_float(log_e_sequential),
        })
        prev_log_e_cum = log_e_cum

    fixture = {
        "description": "One-sided GREATER alternative, ALL_IN combiner, OneSidedNormalMixture, 50 steps",
        "config": {
            "v_opt": v_opt,
            "alpha_opt": alpha_opt,
            "null_value": null_value,
            "known_variance": known_variance,
            "rho": rho,
            "alternative": "GREATER",
            "combiner": "ALL_IN",
        },
        "steps": steps,
    }

    path = os.path.join(FIXTURES_DIR, "golden_one_sided_normal.json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote {path}")
    return fixture


# ---------------------------------------------------------------------------
# 2. golden_conservative_combiner.json -- Two-sided, CONSERVATIVE (lambda=0.5)
# ---------------------------------------------------------------------------
def generate_conservative_combiner():
    v_opt = 1.0
    alpha_opt = 0.05
    known_variance = 1.0
    null_value = 0.0
    lambda_fixed = 0.5

    mixture = TwoSidedNormalMixture(v_opt=v_opt, alpha_opt=alpha_opt)
    rho = float(mixture.rho)
    observations = generate_observations(SEED, SHIFT, N_STEPS)

    steps = []
    data_sum = 0.0
    prev_log_e_cum = 0.0
    log_e_process = 0.0  # Starts at log(1) = 0

    for t in range(1, N_STEPS + 1):
        x = observations[t - 1]
        data_sum += x
        count = t
        s = data_sum - null_value * count
        v = count * known_variance

        log_e_cum = mixture.log_superMG(s, v)
        log_e_sequential = log_e_cum - prev_log_e_cum
        e_value = np.exp(log_e_sequential)

        # Conservative combiner: increment = (1 - lambda) + lambda * E_t
        lambda_t = lambda_fixed
        increment = (1.0 - lambda_t) + lambda_t * e_value
        log_increment = np.log(increment) if increment > 0 else float("-inf")
        log_e_process += log_increment

        steps.append({
            "t": t,
            "x": to_float(x),
            "data_sum": to_float(data_sum),
            "s": to_float(s),
            "v": to_float(v),
            "log_e_cum": to_float(log_e_cum),
            "log_e_sequential": to_float(log_e_sequential),
            "e_value": to_float(e_value),
            "lambda_t": to_float(lambda_t),
            "increment": to_float(increment),
            "log_increment": to_float(log_increment),
            "log_e_process": to_float(log_e_process),
        })
        prev_log_e_cum = log_e_cum

    fixture = {
        "description": "Two-sided normal, CONSERVATIVE combiner (lambda=0.5), 50 steps",
        "config": {
            "v_opt": v_opt,
            "alpha_opt": alpha_opt,
            "null_value": null_value,
            "known_variance": known_variance,
            "rho": rho,
            "alternative": "TWO_SIDED",
            "combiner": "CONSERVATIVE",
            "lambda_fixed": lambda_fixed,
        },
        "steps": steps,
    }

    path = os.path.join(FIXTURES_DIR, "golden_conservative_combiner.json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote {path}")
    return fixture


# ---------------------------------------------------------------------------
# 3. golden_adaptive_combiner.json -- Two-sided, EMPIRICALLY_ADAPTIVE
# ---------------------------------------------------------------------------
def generate_adaptive_combiner():
    v_opt = 1.0
    alpha_opt = 0.05
    known_variance = 1.0
    null_value = 0.0
    gamma = 0.5
    epsilon = 1e-6

    mixture = TwoSidedNormalMixture(v_opt=v_opt, alpha_opt=alpha_opt)
    rho = float(mixture.rho)
    observations = generate_observations(SEED, SHIFT, N_STEPS)

    steps = []
    data_sum = 0.0
    prev_log_e_cum = 0.0
    log_e_process = 0.0  # Starts at log(1) = 0

    # Running sums for adaptive lambda computation (F_{t-1}-measurable)
    S1 = 0.0  # sum of (E_i - 1) for i < t
    S2 = 0.0  # sum of (E_i - 1)^2 for i < t

    for t in range(1, N_STEPS + 1):
        x = observations[t - 1]
        data_sum += x
        count = t
        s = data_sum - null_value * count
        v = count * known_variance

        log_e_cum = mixture.log_superMG(s, v)
        log_e_sequential = log_e_cum - prev_log_e_cum
        e_value = np.exp(log_e_sequential)

        # Compute lambda_t from PREVIOUS steps only (F_{t-1}-measurable)
        if t == 1:
            # No previous data -> lambda_1 = 0
            lambda_t = 0.0
        else:
            # lambda_t = clamp(S1 / (S2 + epsilon), [0, gamma])
            raw_lambda = S1 / (S2 + epsilon)
            lambda_t = max(0.0, min(raw_lambda, gamma))

        # Compute increment and update e-process
        increment = (1.0 - lambda_t) + lambda_t * e_value
        log_increment = np.log(increment) if increment > 0 else float("-inf")
        log_e_process += log_increment

        steps.append({
            "t": t,
            "x": to_float(x),
            "data_sum": to_float(data_sum),
            "s": to_float(s),
            "v": to_float(v),
            "log_e_cum": to_float(log_e_cum),
            "log_e_sequential": to_float(log_e_sequential),
            "e_value": to_float(e_value),
            "lambda_t": to_float(lambda_t),
            "S1_before": to_float(S1),
            "S2_before": to_float(S2),
            "increment": to_float(increment),
            "log_increment": to_float(log_increment),
            "log_e_process": to_float(log_e_process),
        })

        # Update S1, S2 AFTER using them (so they reflect E_1..E_t for next step)
        deviation = e_value - 1.0
        S1 += deviation
        S2 += deviation * deviation

        prev_log_e_cum = log_e_cum

    fixture = {
        "description": "Two-sided normal, EMPIRICALLY_ADAPTIVE combiner (gamma=0.5, epsilon=1e-6), 50 steps",
        "config": {
            "v_opt": v_opt,
            "alpha_opt": alpha_opt,
            "null_value": null_value,
            "known_variance": known_variance,
            "rho": rho,
            "alternative": "TWO_SIDED",
            "combiner": "EMPIRICALLY_ADAPTIVE",
            "gamma": gamma,
            "epsilon": epsilon,
        },
        "steps": steps,
    }

    path = os.path.join(FIXTURES_DIR, "golden_adaptive_combiner.json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote {path}")
    return fixture


# ---------------------------------------------------------------------------
# 4. golden_less_alternative.json -- One-sided LESS alternative
# ---------------------------------------------------------------------------
def generate_one_sided_less():
    v_opt = 1.0
    alpha_opt = 0.05
    known_variance = 1.0
    null_value = 0.0

    mixture = OneSidedNormalMixture(v_opt=v_opt, alpha_opt=alpha_opt)
    rho = float(mixture.rho)
    # Signal below null: observations = randn - 0.3
    observations = generate_observations(SEED, -SHIFT, N_STEPS)

    steps = []
    data_sum = 0.0
    prev_log_e_cum = 0.0

    for t in range(1, N_STEPS + 1):
        x = observations[t - 1]
        data_sum += x
        count = t
        # For LESS direction: negate the centered sum
        s = -(data_sum - null_value * count)
        v = count * known_variance

        log_e_cum = mixture.log_superMG(s, v)
        log_e_sequential = log_e_cum - prev_log_e_cum

        steps.append({
            "t": t,
            "x": to_float(x),
            "data_sum": to_float(data_sum),
            "s": to_float(s),
            "v": to_float(v),
            "log_e_cum": to_float(log_e_cum),
            "log_e_sequential": to_float(log_e_sequential),
        })
        prev_log_e_cum = log_e_cum

    fixture = {
        "description": "One-sided LESS alternative, ALL_IN combiner, OneSidedNormalMixture, 50 steps",
        "config": {
            "v_opt": v_opt,
            "alpha_opt": alpha_opt,
            "null_value": null_value,
            "known_variance": known_variance,
            "rho": rho,
            "alternative": "LESS",
            "combiner": "ALL_IN",
        },
        "steps": steps,
    }

    path = os.path.join(FIXTURES_DIR, "golden_less_alternative.json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=2)
    print(f"Wrote {path}")
    return fixture


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_preview(name, fixture, n_preview=3):
    """Print first few steps from a fixture for verification."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  {fixture['description']}")
    print(f"  rho = {fixture['config']['rho']}")
    print(f"{'='*70}")
    steps = fixture["steps"]
    for step in steps[:n_preview]:
        print(f"  Step {step['t']}:")
        for k, val in step.items():
            if k == "t":
                continue
            print(f"    {k:20s} = {val}")
    print(f"  ... ({len(steps)} total steps)")


if __name__ == "__main__":
    print("Generating golden test fixtures for Rust e-value engine...\n")

    f1 = generate_one_sided_greater()
    f2 = generate_conservative_combiner()
    f3 = generate_adaptive_combiner()
    f4 = generate_one_sided_less()

    print_preview("golden_one_sided_normal.json", f1)
    print_preview("golden_conservative_combiner.json", f2)
    print_preview("golden_adaptive_combiner.json", f3)
    print_preview("golden_less_alternative.json", f4)

    print(f"\nAll fixtures written to: {FIXTURES_DIR}")
    print("Done.")
