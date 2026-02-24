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


    