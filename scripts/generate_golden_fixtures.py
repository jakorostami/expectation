"""
Generate golden test fixtures from the Python implementation.

These fixtures are used to verify the Rust VoxelField engine produces
identical results to the Python sequential testing pipeline.

Output: tests/rust/fixtures/golden_two_sided_normal.json
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from expectation.modules.martingales import TwoSidedNormalMixture


def generate_single_voxel_golden():
    """Generate golden values for a single voxel over 50 time steps.

    Uses deterministic observations: x_t = sin(t) for reproducibility.
    Known variance = 1.0, null = 0.0, alpha = 0.05.
    """
    v_opt = 1.0
    alpha_opt = 0.05
    null_value = 0.0
    known_variance = 1.0

    mixture = TwoSidedNormalMixture(v_opt, alpha_opt)
    rho = mixture.rho

    steps = []
    data_sum = 0.0
    data_count = 0
    previous_log_e_cum = 0.0

    for t in range(1, 51):
        x = np.sin(float(t))
        data_sum += x
        data_count += 1

        s = data_sum - null_value * data_count
        v = data_count * known_variance

        log_e_cum = float(mixture.log_superMG(s, v))

        steps.append({
            "t": t,
            "x": float(x),
            "data_sum": float(data_sum),
            "s": float(s),
            "v": float(v),
            "log_e_cum": float(log_e_cum),
        })

        previous_log_e_cum = log_e_cum

    return {
        "description": "Single voxel, TwoSidedNormalMixture, known variance, 50 steps",
        "config": {
            "v_opt": v_opt,
            "alpha_opt": alpha_opt,
            "null_value": null_value,
            "known_variance": known_variance,
            "rho": float(rho),
        },
        "steps": steps,
    }


def generate_multi_voxel_golden():
    """Generate golden values for 5 voxels with different null values.

    3 steps of deterministic observations.
    """
    v_opt = 1.0
    alpha_opt = 0.05
    known_variance = 1.0

    mixture = TwoSidedNormalMixture(v_opt, alpha_opt)
    rho = mixture.rho

    n_voxels = 5
    null_values = [0.0, 0.5, -0.5, 1.0, -1.0]

    # Deterministic observations: each voxel gets a different sequence
    observations = [
        [0.3, 0.8, -0.2, 1.5, -0.7],  # t=1
        [-0.1, 0.4, 0.6, -0.3, 0.9],  # t=2
        [0.7, -0.5, 0.1, 0.2, -0.4],  # t=3
    ]

    all_steps = []
    data_sums = [0.0] * n_voxels

    for t_idx, obs in enumerate(observations):
        t = t_idx + 1
        step_data = {"t": t, "observations": obs, "voxels": []}

        for i in range(n_voxels):
            data_sums[i] += obs[i]
            s = data_sums[i] - null_values[i] * t
            v = t * known_variance
            log_e_cum = float(mixture.log_superMG(s, v))

            step_data["voxels"].append({
                "index": i,
                "data_sum": float(data_sums[i]),
                "s": float(s),
                "v": float(v),
                "log_e_cum": float(log_e_cum),
            })

        all_steps.append(step_data)

    return {
        "description": "5 voxels, TwoSidedNormalMixture, known variance, 3 steps",
        "config": {
            "v_opt": v_opt,
            "alpha_opt": alpha_opt,
            "null_values": null_values,
            "known_variance": known_variance,
            "rho": float(rho),
            "n_voxels": n_voxels,
        },
        "steps": all_steps,
    }


def main():
    fixtures_dir = project_root / "tests" / "rust" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    golden = {
        "single_voxel": generate_single_voxel_golden(),
        "multi_voxel": generate_multi_voxel_golden(),
    }

    output_path = fixtures_dir / "golden_two_sided_normal.json"
    with open(output_path, "w") as f:
        json.dump(golden, f, indent=2)

    print(f"Generated golden fixtures: {output_path}")
    print(f"  Single voxel: {len(golden['single_voxel']['steps'])} steps")
    print(f"  Multi voxel: {golden['multi_voxel']['config']['n_voxels']} voxels, "
          f"{len(golden['multi_voxel']['steps'])} steps")


if __name__ == "__main__":
    main()
