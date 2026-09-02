# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""
Configuration and result models for k-sample sequential e-testing.

Based on:
Turner, Ly & Grunwald (2022), "Generic E-Variables for Exact Sequential
k-Sample Tests that allow for Optional Stopping"

Implements Pydantic v2 frozen config models following codebase conventions.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from expectation.modules.hypothesistesting import BettingStrategy


class KSampleAlternativeType(str, Enum):
    """Alternative hypothesis type for k-sample testing.

    SIMPLE: Fixed theta vector specified by the user (Eq. 3.2).
    UNRESTRICTED: Any theta_g != theta_g' (Proposition 2, closed-form).
    EFFECT_SIZE: d(theta_a, theta_b) >= delta (Appendix S1, grid-based).
    """

    SIMPLE = "simple"
    UNRESTRICTED = "unrestricted"
    EFFECT_SIZE = "effect_size"


class DivergenceType(str, Enum):
    """Divergence measure for effect-size restricted alternatives (Section 4).

    ADDITIVE: theta_b - theta_a >= delta
    LOG_ODDS_RATIO: log(OR) >= delta
    """

    ADDITIVE = "additive"
    LOG_ODDS_RATIO = "log_odds_ratio"


class KSampleConfig(BaseModel):
    """Configuration for k-sample sequential e-test.

    Parameters
    ----------
    k : int
        Number of groups (must be > 1).
    significance_level : float
        Type-I error level alpha for Ville's inequality.
    gamma : float
        Beta(gamma, gamma) prior parameter. Default 0.18 is REGROW-optimal
        (Turner et al. 2022, Section 5).
    betting_strategy : BettingStrategy
        Strategy for temporal combination of per-step e-values into an
        e-process (Ramdas & Wang 2025, Chapter 7).
    conservative_lambda : float
        Fixed lambda for conservative betting strategy.
    alternative_type : KSampleAlternativeType
        Type of alternative hypothesis.
    divergence_type : DivergenceType, optional
        Required when alternative_type == EFFECT_SIZE.
    min_effect_size : float, optional
        Minimum effect size delta. Required when alternative_type == EFFECT_SIZE.
    grid_precision : float
        Grid step size K for discretized posterior in effect-size restricted
        alternatives (Appendix S1). Default 0.01 gives ~100 grid points.
    simple_theta : dict[int, float], optional
        Fixed theta vector {group_id: theta_g*}. Required when
        alternative_type == SIMPLE.
    """

    model_config = ConfigDict(frozen=True)

    k: int = Field(gt=1, description="Number of groups")
    significance_level: float = Field(gt=0, lt=1, default=0.05, description="Type-I error level")
    gamma: float = Field(
        gt=0,
        default=0.18,
        description="Beta(gamma,gamma) prior parameter; 0.18 is REGROW-optimal (Section 5)",
    )
    betting_strategy: BettingStrategy = Field(
        default=BettingStrategy.ALL_IN,
        description="Temporal combination strategy for e-process",
    )
    conservative_lambda: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Fixed lambda for conservative strategy",
    )
    alternative_type: KSampleAlternativeType = Field(
        default=KSampleAlternativeType.UNRESTRICTED,
        description="Type of alternative hypothesis",
    )
    divergence_type: Optional[DivergenceType] = Field(
        default=None, description="Divergence measure for effect-size restricted tests"
    )
    min_effect_size: Optional[float] = Field(
        default=None, gt=0, description="Minimum effect size delta"
    )
    grid_precision: float = Field(
        default=0.01,
        gt=0,
        lt=1,
        description="Grid step size for discretized posterior (Appendix S1)",
    )
    simple_theta: Optional[dict[int, float]] = Field(
        default=None, description="Fixed theta vector {group_id: theta_g*}"
    )

    @model_validator(mode="after")
    def _validate_alternative_fields(self) -> "KSampleConfig":
        if self.alternative_type == KSampleAlternativeType.EFFECT_SIZE:
            if self.k != 2:
                raise ValueError(
                    "Effect-size restricted alternative is only supported for k=2 "
                    "(2-sample testing). The paper's restricted framework (Appendix S1) "
                    "is defined for two groups."
                )
            if self.divergence_type is None:
                raise ValueError("divergence_type is required when alternative_type == EFFECT_SIZE")
            if self.min_effect_size is None:
                raise ValueError("min_effect_size is required when alternative_type == EFFECT_SIZE")
        if self.alternative_type == KSampleAlternativeType.SIMPLE:
            if self.simple_theta is None:
                raise ValueError("simple_theta is required when alternative_type == SIMPLE")
            if len(self.simple_theta) != self.k:
                raise ValueError(
                    f"simple_theta must have exactly k={self.k} entries, "
                    f"got {len(self.simple_theta)}"
                )
            for gid, theta in self.simple_theta.items():
                if not (0 < theta < 1):
                    raise ValueError(f"simple_theta[{gid}] = {theta} must be in (0, 1)")
        return self


class KSampleStepResult(BaseModel):
    """Result of a single update step in the k-sample sequential e-test.

    Parameters
    ----------
    reject_null : bool
        Whether the null hypothesis of homogeneity is rejected (Ville's inequality).
    e_value : float
        Per-step e-value S_j (Eq. 3.4).
    e_process_value : float
        Current e-process value M_j.
    log_e_process : float
        log(M_j) for numerical monitoring.
    p_value : float
        Anytime-valid p-value: min(1, 1/max(M_1,...,M_j)).
    step : int
        Block index (1-indexed).
    group_counts : dict[int, int]
        Cumulative observation counts per group.
    group_means : dict[int, float]
        Cumulative sample means per group.
    theta_estimates : dict[int, float]
        Posterior mean theta_g_hat per group used for this step's e-value.
    theta_null : float
        RIPr mixture parameter theta_0_hat.
    """

    model_config = ConfigDict(frozen=True)

    reject_null: bool
    e_value: float
    e_process_value: float
    log_e_process: float
    p_value: float
    step: int
    group_counts: dict[int, int]
    group_means: dict[int, float]
    theta_estimates: dict[int, float]
    theta_null: float
