# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""
K-sample sequential e-test orchestrator for Bernoulli data.

Based on:
Turner, Ly & Grunwald (2022), "Generic E-Variables for Exact Sequential
k-Sample Tests that allow for Optional Stopping"

Provides the main ``KSampleSequentialTest`` class that:
- Accepts per-block multi-group binary observations.
- Computes per-step RIPr e-values (unrestricted, effect-size restricted,
  or simple alternative).
- Feeds e-values into the existing ``EProcessUpdater`` for temporal
  combination (all-in, conservative, empirically adaptive, log-optimal).
- Checks significance via Ville's inequality at each step.

Architecture follows ``SequentialTesting`` in
``expectation/seqtest/sequential_e_testing.py``.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from expectation.ksample.bernoulli import BernoulliRIPrCalculator
from expectation.ksample.config import KSampleConfig, KSampleStepResult
from expectation.modules.calibrators import EToPCalibrator
from expectation.modules.eprocessupdater import EProcessUpdater
from expectation.modules.hypothesistesting import (
    EProcess,
    EProcessConfig,
)


class KSampleSequentialTest:
    """Sequential k-sample homogeneity test for Bernoulli data.

    Tests H0: theta_1 = theta_2 = ... = theta_k (all groups have the same
    success probability) against alternatives specified by ``config``.

    Parameters
    ----------
    config : KSampleConfig
        Test configuration (number of groups, significance level, prior
        parameter gamma, alternative type, etc.).

    Examples
    --------
    Basic 2-sample test (A/B test):

    >>> config = KSampleConfig(k=2, significance_level=0.05)
    >>> test = KSampleSequentialTest(config)
    >>> result = test.update({0: np.array([1, 0, 1]), 1: np.array([0, 1, 0])})
    >>> result.reject_null
    False

    Effect-size restricted test detecting >= 10% lift:

    >>> config = KSampleConfig(
    ...     k=2,
    ...     alternative_type="effect_size",
    ...     divergence_type="additive",
    ...     min_effect_size=0.1,
    ... )
    >>> test = KSampleSequentialTest(config)

    References
    ----------
    Turner, Ly & Grunwald (2022), "Generic E-Variables for Exact Sequential
    k-Sample Tests that allow for Optional Stopping"
    """

    def __init__(self, config: KSampleConfig) -> None:
        self.config = config
        self.k = config.k

        self._calculator = BernoulliRIPrCalculator(config)

        e_process_config = EProcessConfig(
            significance_level=config.significance_level,
            betting_strategy=config.betting_strategy,
            gamma=config.gamma,
            conservative_lambda=config.conservative_lambda,
        )
        self.e_process = EProcess(config=e_process_config)
        self.e_process_updater = EProcessUpdater(e_process_config)
        self._calibrator = EToPCalibrator()

        self.block_count: int = 0

        self.history: list[dict] = []

        self._buffer: dict[int, list[int]] = {g: [] for g in range(self.k)}

    def update(self, group_data: dict[int, NDArray]) -> KSampleStepResult:
        """Process one block of observations across all k groups.

        Parameters
        ----------
        group_data : dict[int, NDArray]
            Mapping from group id (0..k-1) to a 1-D array of binary
            observations (0 or 1) for this block.
            All k groups must be present. Group sizes may differ.

        Returns
        -------
        KSampleStepResult
            Frozen Pydantic model with e-value, e-process value,
            rejection decision, and diagnostics.

        Raises
        ------
        ValueError
            If not all k groups are present, or if observations are not binary.
        """
        self._validate_group_data(group_data)

        # Extract block statistics
        block_successes: dict[int, int] = {}
        block_sizes: dict[int, int] = {}
        for g in range(self.k):
            arr = np.asarray(group_data[g]).flatten()
            block_successes[g] = int(np.sum(arr))
            block_sizes[g] = len(arr)

        # Compute per-step log e-value (uses cumulative state BEFORE this block)
        log_e_value, theta_estimates, theta_null = self._calculator.compute_log_e_value(
            block_successes, block_sizes
        )

        e_value = np.exp(log_e_value)

        self.e_process_updater.update(self.e_process, e_value)

        self._calculator.update_state(block_successes, block_sizes)
        self.block_count += 1

        reject_null = self.e_process_updater.is_significant(self.e_process)

        current_value = self.e_process_updater.get_current_value(self.e_process)
        log_e_process = (
            self.e_process.log_process_values[-1] if self.e_process.log_process_values else 0.0
        )

        max_value = self.e_process_updater.get_max_value(
            self.e_process
        )  # Anytime-valid p-value (Proposition 2.4, Ramdas & Wang 2025)
        p_value = self._calibrator(max_value)

        # Build cumulative stats for result
        group_counts = dict(self._calculator.cumulative_counts)
        group_means: dict[int, float] = {}
        for g in range(self.k):
            N_g = self._calculator.cumulative_counts[g]
            if N_g > 0:
                group_means[g] = self._calculator.cumulative_successes[g] / N_g
            else:
                group_means[g] = 0.0

        result = KSampleStepResult(
            reject_null=reject_null,
            e_value=float(e_value),
            e_process_value=float(current_value),
            log_e_process=float(log_e_process),
            p_value=float(p_value),
            step=self.block_count,
            group_counts=group_counts,
            group_means=group_means,
            theta_estimates={g: float(v) for g, v in theta_estimates.items()},
            theta_null=float(theta_null),
        )

        self.history.append(
            {
                "step": self.block_count,
                "e_value": float(e_value),
                "e_process_value": float(current_value),
                "log_e_process": float(log_e_process),
                "p_value": float(p_value),
                "reject_null": reject_null,
                "theta_null": float(theta_null),
                **{f"theta_{g}": float(theta_estimates[g]) for g in range(self.k)},
                **{f"count_{g}": group_counts[g] for g in range(self.k)},
                **{f"mean_{g}": group_means[g] for g in range(self.k)},
            }
        )

        return result

    def update_single(
        self, group_id: int, observation: Union[int, float]
    ) -> Optional[KSampleStepResult]:
        """Buffer a single observation. Triggers update() when all k groups have data.

        Parameters
        ----------
        group_id : int
            Group index (0..k-1).
        observation : int or float
            Binary observation (0 or 1).

        Returns
        -------
        KSampleStepResult or None
            Result if a full block was processed, None otherwise.
        """
        if group_id not in range(self.k):
            raise ValueError(f"group_id {group_id} out of range [0, {self.k - 1}]")
        obs = int(observation)
        if obs not in (0, 1):
            raise ValueError(f"Observation must be 0 or 1, got {observation}")

        self._buffer[group_id].append(obs)

        # Check if all groups have at least one buffered observation
        if all(len(self._buffer[g]) > 0 for g in range(self.k)):
            group_data = {g: np.array(self._buffer[g], dtype=np.int64) for g in range(self.k)}
            # Clear buffer
            self._buffer = {g: [] for g in range(self.k)}
            return self.update(group_data)

        return None

    def get_p_process(self) -> list[float]:
        """Compute the p-process (Definition 7.10, Ramdas & Wang 2025).

        Returns the running minimum of calibrated e-process values:
            p_t = min_{s <= t} min(1, 1/M_s)

        Returns
        -------
        list[float]
            Non-increasing sequence of anytime-valid p-values.
        """
        return self.e_process_updater.compute_p_process(self.e_process)

    def get_history_df(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    def get_summary(self) -> dict:
        return {
            "k": self.k,
            "block_count": self.block_count,
            "alternative_type": self.config.alternative_type.value,
            "current_e_process": self.e_process_updater.get_current_value(self.e_process),
            "max_e_process": self.e_process_updater.get_max_value(self.e_process),
            "is_significant": self.e_process_updater.is_significant(self.e_process),
            "stopping_time": self.e_process_updater.get_stopping_time(self.e_process),
            "group_counts": dict(self._calculator.cumulative_counts),
            "group_means": {
                g: (
                    (
                        self._calculator.cumulative_successes[g]
                        / self._calculator.cumulative_counts[g]
                    )
                    if self._calculator.cumulative_counts[g] > 0
                    else 0.0
                )
                for g in range(self.k)
            },
            "betting_strategy": self.config.betting_strategy.value,
        }

    def reset(self) -> None:
        self._calculator.reset()

        e_process_config = EProcessConfig(
            significance_level=self.config.significance_level,
            betting_strategy=self.config.betting_strategy,
            gamma=self.config.gamma,
            conservative_lambda=self.config.conservative_lambda,
        )
        self.e_process = EProcess(config=e_process_config)
        self.e_process_updater = EProcessUpdater(e_process_config)

        self.block_count = 0
        self.history = []
        self._buffer = {g: [] for g in range(self.k)}

    def _validate_group_data(self, group_data: dict[int, NDArray]) -> None:
        # Validate that group_data has all k groups with binary observations.
        for g in range(self.k):
            if g not in group_data:
                raise ValueError(
                    f"Group {g} missing from group_data. " f"All {self.k} groups must be present."
                )
            arr = np.asarray(group_data[g]).flatten()
            if len(arr) == 0:
                raise ValueError(f"Group {g} has no observations.")
            if not np.all(np.isin(arr, [0, 1])):
                raise ValueError(
                    f"Group {g} contains non-binary values. " f"All observations must be 0 or 1."
                )
