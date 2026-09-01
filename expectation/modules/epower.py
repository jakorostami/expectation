# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

from enum import Enum
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, Field


class EPowerType(str, Enum):
    STANDARD = "standard"
    ALL_OR_NOTHING = "all_or_nothing"
    OPTIMIZED = "optimized"


class EPowerConfig(BaseModel):
    type: EPowerType = Field(default=EPowerType.STANDARD)
    optimize_lambda: bool = Field(default=False)
    grid_size: int = Field(default=100)
    min_lambda: float = Field(default=0.0)
    max_lambda: float = Field(default=1.0)


class EPowerResult(BaseModel):
    e_power: float = Field(description="Computed e-power value")
    is_positive: bool = Field(description="Whether e-power is positive")
    expected_e_value: float = Field(description="Expected e-value")
    optimal_lambda: Optional[float] = Field(default=None, description="Optimal lambda if optimized")
    type: EPowerType = Field(description="Type of e-power calculation used")


class EPowerCalculator:
    """
    Calculator for e-power metrics.
    """

    def __init__(self, config: Optional[EPowerConfig] = None):
        self.config = config or EPowerConfig()

    def compute(
        self, e_values: np.ndarray, alternative_prob: Optional[np.ndarray] = None
    ) -> EPowerResult:
        if alternative_prob is None:
            alternative_prob = np.ones(len(e_values)) / len(e_values)

        if self.config.type == EPowerType.ALL_OR_NOTHING:
            # Convert to all-or-nothing e-values
            e_values = np.where(e_values > 1, 1 / 0.05, 0)  # Using standard α=0.05

        # Compute base e-power
        e_power = np.sum(alternative_prob * np.log(e_values))
        expected_e_value = np.sum(alternative_prob * e_values)

        optimal_lambda = None
        if self.config.optimize_lambda:
            optimal_lambda = self._optimize_lambda(e_values, alternative_prob)
            # Transform e-values using optimal lambda
            e_values = 1 - optimal_lambda + optimal_lambda * e_values
            e_power = np.sum(alternative_prob * np.log(e_values))
            expected_e_value = np.sum(alternative_prob * e_values)

        return EPowerResult(
            e_power=float(e_power),
            is_positive=e_power > 0,
            expected_e_value=float(expected_e_value),
            optimal_lambda=optimal_lambda,
            type=self.config.type,
        )

    def _optimize_lambda(self, e_values: np.ndarray, alternative_prob: np.ndarray) -> float:
        lambdas = np.linspace(self.config.min_lambda, self.config.max_lambda, self.config.grid_size)

        max_e_power = float("-inf")
        optimal_lambda = 0

        for lam in lambdas:
            transformed_e_values = 1 - lam + lam * e_values
            e_power = np.sum(alternative_prob * np.log(transformed_e_values))

            if e_power > max_e_power:
                max_e_power = e_power
                optimal_lambda = lam

        return optimal_lambda
