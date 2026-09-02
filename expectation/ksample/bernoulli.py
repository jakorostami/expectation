# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""
Bernoulli RIPr e-variable computation for k-sample sequential testing.

Based on:
Turner, Ly & Grunwald (2022), "Generic E-Variables for Exact Sequential
k-Sample Tests that allow for Optional Stopping"

Implements three computation paths:
1. Unrestricted alternative (Proposition 2): closed-form Beta posterior means.
2. Effect-size restricted alternative (Appendix S1): grid-based discretized
   posterior for k=2 with additive or log-odds-ratio constraint.
3. Simple alternative (Eq. 3.2): fixed theta vector, no posterior updating.

All computations are performed in log-space for numerical stability.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import beta as beta_dist

from expectation.ksample.config import (
    DivergenceType,
    KSampleAlternativeType,
    KSampleConfig,
)


class BernoulliRIPrCalculator:
    """Compute per-step RIPr e-values for Bernoulli k-sample testing.

    The RIPr (Reverse Information Projection) construction yields exact
    e-variables for testing homogeneity H0: theta_1 = ... = theta_k
    against various alternatives.

    Parameters
    ----------
    config : KSampleConfig
        Test configuration specifying k, gamma, alternative type, etc.

    Attributes
    ----------
    cumulative_successes : dict[int, int]
        Running total of successes per group.
    cumulative_counts : dict[int, int]
        Running total of observations per group.

    References
    ----------
    Turner, Ly & Grunwald (2022), Proposition 2 (unrestricted),
    Appendix S1 (effect-size restricted), Eq. 3.2 (simple).
    """

    def __init__(self, config: KSampleConfig) -> None:
        self.config = config
        self.k = config.k
        self.gamma = config.gamma

        # Per-group cumulative state
        self.cumulative_successes: dict[int, int] = {g: 0 for g in range(self.k)}
        self.cumulative_counts: dict[int, int] = {g: 0 for g in range(self.k)}

        # Dispatch to computation path
        if config.alternative_type == KSampleAlternativeType.EFFECT_SIZE:
            self._init_grid()

    def _init_grid(self) -> None:
        """Initialize discretized prior grid for effect-size restricted test.

        Implements the grid-based posterior from Appendix S1 of Turner et al.
        (2022). Only valid for k=2.

        The grid spans theta_a values in (K, 1-zeta) where zeta depends on
        the divergence type, and the corresponding theta_b values are derived
        from the effect-size constraint.
        """
        delta = self.config.min_effect_size
        K = self.config.grid_precision

        if self.config.divergence_type == DivergenceType.ADDITIVE:
            # theta_b = theta_a + delta, so theta_a < 1 - delta
            zeta = delta
            self.grid_theta_a = np.arange(K, 1.0 - zeta, K)
            self.grid_theta_b = self.grid_theta_a + delta

            # Filter to ensure both theta_a, theta_b in (0, 1)
            valid = (self.grid_theta_b < 1.0) & (self.grid_theta_a > 0.0)
            self.grid_theta_a = self.grid_theta_a[valid]
            self.grid_theta_b = self.grid_theta_b[valid]

        elif self.config.divergence_type == DivergenceType.LOG_ODDS_RATIO:
            # log(OR) >= delta => theta_b = odds_a * exp(delta) / (1 + odds_a * exp(delta))
            zeta = 0.0
            self.grid_theta_a = np.arange(K, 1.0 - K, K)
            odds_a = self.grid_theta_a / (1.0 - self.grid_theta_a)
            exp_delta = np.exp(delta)
            self.grid_theta_b = (odds_a * exp_delta) / (1.0 + odds_a * exp_delta)

            # Filter to ensure both in (0, 1)
            valid = (
                (self.grid_theta_a > 0.0)
                & (self.grid_theta_a < 1.0)
                & (self.grid_theta_b > 0.0)
                & (self.grid_theta_b < 1.0)
            )
            self.grid_theta_a = self.grid_theta_a[valid]
            self.grid_theta_b = self.grid_theta_b[valid]

        # Compute rho for prior: rho = theta_a / (1 - zeta) maps theta_a to [0, 1]
        if self.config.divergence_type == DivergenceType.ADDITIVE:
            grid_rho = self.grid_theta_a / (1.0 - zeta)
        else:
            # For log odds ratio, use theta_a directly as rho
            grid_rho = self.grid_theta_a

        # Clip rho to avoid beta.logpdf issues at boundaries
        grid_rho = np.clip(grid_rho, 1e-10, 1.0 - 1e-10)

        # Discretized Beta(gamma, gamma) prior on rho-space (normalized)
        log_prior = beta_dist.logpdf(grid_rho, self.gamma, self.gamma)
        log_prior -= logsumexp(log_prior)

        # Initialize log-weights to log-prior
        self.log_weights: NDArray = log_prior.copy()

    def compute_log_e_value(
        self,
        block_successes: dict[int, int],
        block_sizes: dict[int, int],
    ) -> tuple[float, dict[int, float], float]:
        """Compute the per-step log e-value for one block of observations.

        This is the main dispatch method. Posterior means are computed BEFORE
        updating cumulative state (F_{j-1}-measurable requirement).

        Parameters
        ----------
        block_successes : dict[int, int]
            Number of successes per group in this block.
        block_sizes : dict[int, int]
            Number of observations per group in this block.

        Returns
        -------
        log_e_value : float
            Per-step log e-value log(S_j).
        theta_estimates : dict[int, float]
            Posterior mean estimates used for this step.
        theta_null : float
            RIPr mixture parameter theta_0_hat.
        """
        if self.config.alternative_type == KSampleAlternativeType.UNRESTRICTED:
            return self._compute_unrestricted(block_successes, block_sizes)
        elif self.config.alternative_type == KSampleAlternativeType.EFFECT_SIZE:
            return self._compute_effect_size(block_successes, block_sizes)
        elif self.config.alternative_type == KSampleAlternativeType.SIMPLE:
            return self._compute_simple(block_successes, block_sizes)
        else:
            raise ValueError(f"Unknown alternative type: {self.config.alternative_type}")

    def _compute_unrestricted(
        self,
        block_successes: dict[int, int],
        block_sizes: dict[int, int],
    ) -> tuple[float, dict[int, float], float]:
        """Unrestricted alternative (Proposition 2, closed-form).

        Per group g, the posterior mean under Beta(gamma, gamma) prior is:
            theta_g_hat = (U_g + gamma) / (N_g + 2*gamma)

        The RIPr null parameter is:
            theta_0_hat = sum_g (n_g / n) * theta_g_hat

        Per-step log e-value (Eq. 3.4 generalized to k groups):
            log(S_j) = sum_g [ s_g * log(theta_g_hat / theta_0_hat)
                              + (n_g - s_g) * log((1 - theta_g_hat) / (1 - theta_0_hat)) ]
        """
        # Compute posterior means BEFORE updating state (F_{j-1}-measurable)
        theta_estimates: dict[int, float] = {}
        for g in range(self.k):
            U_g = self.cumulative_successes[g]
            N_g = self.cumulative_counts[g]
            theta_estimates[g] = (U_g + self.gamma) / (N_g + 2.0 * self.gamma)

        # RIPr mixture parameter: weighted average of per-group estimates
        total_block_size = sum(block_sizes.values())
        if total_block_size == 0:
            return 0.0, theta_estimates, 0.5

        theta_null = sum(
            (block_sizes[g] / total_block_size) * theta_estimates[g] for g in range(self.k)
        )

        # Compute per-step log e-value
        log_e = self._bernoulli_log_e_value(
            block_successes, block_sizes, theta_estimates, theta_null
        )

        return log_e, theta_estimates, theta_null

    def _compute_effect_size(
        self,
        block_successes: dict[int, int],
        block_sizes: dict[int, int],
    ) -> tuple[float, dict[int, float], float]:
        """Effect-size restricted alternative (Appendix S1, grid-based).

        Only for k=2. Uses a discretized posterior over (theta_a, theta_b)
        pairs satisfying the effect-size constraint d(theta_a, theta_b) >= delta.

        Steps per block:
        1. Compute posterior means from current grid weights (F_{j-1}-measurable).
        2. Compute per-step e-value using the posterior means.
        3. Update grid weights with this block's likelihood (for next step).
        """
        # Step 1: Compute posterior means from current weights
        log_normalizer = logsumexp(self.log_weights)
        posterior = np.exp(self.log_weights - log_normalizer)

        theta_a_hat = float(np.dot(posterior, self.grid_theta_a))
        theta_b_hat = float(np.dot(posterior, self.grid_theta_b))

        theta_estimates = {0: theta_a_hat, 1: theta_b_hat}

        # RIPr null parameter
        total_block_size = block_sizes[0] + block_sizes[1]
        if total_block_size == 0:
            return 0.0, theta_estimates, 0.5

        theta_null = (
            block_sizes[0] / total_block_size * theta_a_hat
            + block_sizes[1] / total_block_size * theta_b_hat
        )

        # Step 2: Compute per-step log e-value
        log_e = self._bernoulli_log_e_value(
            block_successes, block_sizes, theta_estimates, theta_null
        )

        # Step 3: Update grid weights with this block's likelihood
        s_a, n_a = block_successes[0], block_sizes[0]
        s_b, n_b = block_successes[1], block_sizes[1]

        # Vectorized log-likelihood for all grid points
        log_lik = (
            s_a * np.log(self.grid_theta_a)
            + (n_a - s_a) * np.log1p(-self.grid_theta_a)
            + s_b * np.log(self.grid_theta_b)
            + (n_b - s_b) * np.log1p(-self.grid_theta_b)
        )
        self.log_weights = self.log_weights + log_lik

        # Periodically renormalize to prevent underflow drift
        self.log_weights -= logsumexp(self.log_weights)

        return log_e, theta_estimates, theta_null

    def _compute_simple(
        self,
        block_successes: dict[int, int],
        block_sizes: dict[int, int],
    ) -> tuple[float, dict[int, float], float]:
        """Simple alternative (Eq. 3.2): fixed theta vector, no posterior updating."""
        theta_estimates = dict(self.config.simple_theta)

        total_block_size = sum(block_sizes.values())
        if total_block_size == 0:
            return 0.0, theta_estimates, 0.5

        theta_null = sum(
            (block_sizes[g] / total_block_size) * theta_estimates[g] for g in range(self.k)
        )

        log_e = self._bernoulli_log_e_value(
            block_successes, block_sizes, theta_estimates, theta_null
        )

        return log_e, theta_estimates, theta_null

    @staticmethod
    def _bernoulli_log_e_value(
        block_successes: dict[int, int],
        block_sizes: dict[int, int],
        theta_estimates: dict[int, float],
        theta_null: float,
    ) -> float:
        """Compute the Bernoulli per-step log e-value (Eq. 3.4).

        log(S_j) = sum_g [ s_g * log(theta_g_hat / theta_0_hat)
                         + (n_g - s_g) * log((1 - theta_g_hat) / (1 - theta_0_hat)) ]

        Parameters
        ----------
        block_successes : dict[int, int]
            Successes per group in this block.
        block_sizes : dict[int, int]
            Observations per group in this block.
        theta_estimates : dict[int, float]
            Per-group theta estimates (posterior means or fixed).
        theta_null : float
            RIPr null parameter theta_0_hat.

        Returns
        -------
        float
            Per-step log e-value.
        """
        log_e = 0.0

        for g in block_successes:
            s_g = block_successes[g]
            n_g = block_sizes[g]
            theta_g = theta_estimates[g]

            if n_g == 0:
                continue

            # When theta_g == theta_null, this group contributes 0
            if abs(theta_g - theta_null) < 1e-15:
                continue

            # Successes contribute: s_g * log(theta_g / theta_null)
            if s_g > 0:
                log_e += s_g * (np.log(theta_g) - np.log(theta_null))

            # Failures contribute: (n_g - s_g) * log((1 - theta_g) / (1 - theta_null))
            failures = n_g - s_g
            if failures > 0:
                log_e += failures * (np.log1p(-theta_g) - np.log1p(-theta_null))

        return float(log_e)

    def update_state(
        self,
        block_successes: dict[int, int],
        block_sizes: dict[int, int],
    ) -> None:
        """Update cumulative state AFTER computing this step's e-value.

        Must be called after compute_log_e_value() to maintain the
        F_{j-1}-measurability of posterior means.
        """
        for g in range(self.k):
            self.cumulative_successes[g] += block_successes[g]
            self.cumulative_counts[g] += block_sizes[g]

    def reset(self) -> None:
        self.cumulative_successes = {g: 0 for g in range(self.k)}
        self.cumulative_counts = {g: 0 for g in range(self.k)}

        if self.config.alternative_type == KSampleAlternativeType.EFFECT_SIZE:
            self._init_grid()
