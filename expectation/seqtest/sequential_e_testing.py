# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

from enum import Enum
from typing import Optional, Union, List, Callable
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field
import pandas as pd
import warnings

from expectation.modules.hypothesistesting import (
    Hypothesis, HypothesisType, EValueConfig, EProcess, EProcessConfig, BettingStrategy,
    LikelihoodRatioEValue
)

from expectation.modules.martingales import (
    BetaBinomialMixture, OneSidedNormalMixture,
    TwoSidedNormalMixture, GammaExponentialMixture
)

from expectation.modules.orderstatistics import (
    StaticOrderStatistics
)
from expectation.modules.quantiletest import (
    QuantileABTest
)

from expectation.modules.epower import EPowerCalculator, EPowerConfig, EPowerType

from expectation.modules.eprocessupdater import EProcessUpdater

from expectation.modules import boundaries

class TestType(str, Enum):
    MEAN = "mean"
    QUANTILE = "quantile"
    VARIANCE = "variance"
    PROPORTION = "proportion"

class AlternativeType(str, Enum):
    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"

class BoundaryType(str, Enum):
    NORMAL = "normal"
    BETA_BINOMIAL = "beta_binomial"
    GAMMA_EXPONENTIAL = "gamma_exponential"
    GAMMA_POISSON = "gamma_poisson"
    POLY_STITCHING = "poly_stitching"
    EMPIRICAL_PROCESS_LIL = "empirical_process_lil"
    DOUBLE_STITCHING = "double_stitching"

class BoundaryConfig(BaseModel):
    boundary_type: BoundaryType = Field(
        default=BoundaryType.NORMAL,
        description="Type of boundary to use for confidence sequences"
    )
    v_opt: Optional[float] = Field(
        default=None,
        description="Optimal intrinsic time for tuning the boundary (required for most boundaries)"
    )
    alpha_opt: float = Field(
        default=0.05,
        description="Tuning parameter for the boundary (affects tightness)"
    )
    c: Optional[float] = Field(
        default=None,
        description="Parameter for gamma-exponential/gamma-poisson bounds (tail behavior)"
    )
    g: Optional[float] = Field(
        default=None,
        description="First parameter for beta-binomial bounds"
    )
    h: Optional[float] = Field(
        default=None,
        description="Second parameter for beta-binomial bounds"
    )
    v_min: Optional[float] = Field(
        default=None,
        description="Minimum intrinsic time for poly-stitching bounds"
    )
    s: float = Field(
        default=1.4,
        description="Parameter for poly-stitching bounds"
    )
    eta: float = Field(
        default=2.0,
        description="Parameter for poly-stitching/double-stitching bounds"
    )
    t_min: Optional[float] = Field(
        default=None,
        description="Minimum time for empirical process LIL bounds"
    )
    A: float = Field(
        default=0.85,
        description="Parameter for empirical process LIL bounds"
    )
    delta: float = Field(
        default=0.5,
        description="Parameter for double-stitching bounds"
    )

class SequentialTestResult(BaseModel):
    reject_null: bool
    e_value: float
    e_process: EProcess
    sample_size: int
    p_value: Optional[float] = None
    confidence_bounds: Optional[tuple[float, float]] = None
    test_type: TestType
    alternative: AlternativeType
    timestamp: float = Field(default_factory=lambda: np.datetime64('now').astype(float))
    e_power: Optional[float] = None
    e_power_is_positive: Optional[bool] = None
    optimal_lambda: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True
    


class SequentialTesting:
    def __init__(
        self,
        test_type: Union[TestType, str],
        null_value: float,
        alternative: Union[AlternativeType, str] = "two_sided",
        quantile: Optional[float] = None,
        config: Optional[EValueConfig] = None,
        e_power_config: Optional[EPowerConfig] = None,
        betting_strategy: Optional[str] = None,
        gamma: Optional[float] = None,
        conservative_lambda: Optional[float] = None,
        log_optimal_expectation: Optional[Callable] = None,
        known_variance: Optional[float] = None,
        min_samples_for_update: int = 1,
        use_empirical_variance: bool = True,
        variance_bound: Optional[float] = None,
        boundary_config: Optional[BoundaryConfig] = None
    ):
        """
        Initialize sequential test with full configuration.
        
        Args:
            test_type: Type of test (mean, proportion, variance, quantile)
            null_value: Value under null hypothesis
            alternative: Alternative hypothesis direction
            quantile: Quantile level for quantile tests
            config: E-value configuration
            e_power_config: E-power calculation configuration
            betting_strategy: Strategy for combining e-values
            gamma: Upper bound for adaptive strategy
            conservative_lambda: Fixed lambda for conservative strategy
            log_optimal_expectation: Function for log-optimal strategy
            known_variance: Known variance for mean tests (if available)
            min_samples_for_update: Minimum samples before updating parameters
            use_empirical_variance: Whether to use empirical variance estimation
            variance_bound: Upper bound on variance for bounded distributions
            boundary_config: Configuration for confidence sequence boundaries
        """
        self.test_type = TestType(test_type) if isinstance(test_type, str) else test_type
        self.null_value = null_value
        self.alternative = AlternativeType(alternative) if isinstance(alternative, str) else alternative
        self.quantile = quantile
        
        self.known_variance = known_variance
        self.min_samples_for_update = min_samples_for_update
        self.use_empirical_variance = use_empirical_variance
        self.variance_bound = variance_bound
        
        self.e_power_config = e_power_config or EPowerConfig()
        self.e_power_calculator = EPowerCalculator(self.e_power_config)
        
        if betting_strategy:
            self.config = EProcessConfig(
                significance_level=config.significance_level if config else 0.05,
                allow_infinite=config.allow_infinite if config else False,
                betting_strategy=BettingStrategy(betting_strategy),
                gamma=gamma or 0.5,
                conservative_lambda=conservative_lambda or 0.5
            )
        else:
            self.config = config or EValueConfig()
        
        self.e_process = EProcess(config=self.config)
        
        self.e_process_updater = EProcessUpdater(self.config)
        
        if log_optimal_expectation:
            self.e_process_updater.set_log_optimal_expectation(log_optimal_expectation)
        
        self.data_sum = 0.0
        self.data_sum_squares = 0.0
        self.data_count = 0
        self.all_data = []

        self.intrinsic_time = 0.0
        self.sum_centered_squares = 0.0
        self.empirical_variance = 0.0

        # Track previous cumulative log e-value to compute sequential e-values
        # Sequential e-value: E_t = E_{1:t} / E_{1:t-1} = exp(log_e_t - log_e_{t-1})
        self.previous_log_e_cumulative = 0.0  # log(E_{1:0}) = log(1) = 0

        self.boundary_config = boundary_config or self._get_default_boundary_config()

        self.v_opt = self.boundary_config.v_opt or 1.0
        self.alpha_opt = self.boundary_config.alpha_opt
        self.mixture = None
        
        self._setup_evaluator()

        self.e_power_history = []
        self.lambda_history = []
        self.rejection_times = []
        self.history = []

    def _get_default_boundary_config(self) -> BoundaryConfig:
        alpha = self.config.significance_level if hasattr(self, 'config') else 0.05

        if self.test_type == TestType.MEAN:
            return BoundaryConfig(
                boundary_type=BoundaryType.NORMAL,
                v_opt=1.0,
                alpha_opt=alpha
            )
        elif self.test_type == TestType.PROPORTION:
            return BoundaryConfig(
                boundary_type=BoundaryType.BETA_BINOMIAL,
                v_opt=0.25,  # p*(1-p) with p=0.5 as default
                alpha_opt=alpha,
                g=0.5,  # Symmetric prior
                h=0.5
            )
        elif self.test_type == TestType.VARIANCE:
            return BoundaryConfig(
                boundary_type=BoundaryType.GAMMA_EXPONENTIAL,
                v_opt=1.0,
                alpha_opt=alpha,
                c=1.0  # Default tail parameter
            )
        elif self.test_type == TestType.QUANTILE:
            return BoundaryConfig(
                boundary_type=BoundaryType.DOUBLE_STITCHING,
                v_opt=float(self.min_samples_for_update),
                alpha_opt=alpha,
                delta=0.5,
                eta=2.0
            )
        else:
            return BoundaryConfig(
                boundary_type=BoundaryType.NORMAL,
                v_opt=1.0,
                alpha_opt=alpha
            )

    def _setup_evaluator(self):
        if self.test_type == TestType.MEAN:
            self._setup_mean_test()
        elif self.test_type == TestType.PROPORTION:
            self._setup_proportion_test()
        elif self.test_type == TestType.VARIANCE:
            self._setup_variance_test()
        elif self.test_type == TestType.QUANTILE:
            self._setup_quantile_test()
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")
    
    def _setup_mean_test(self):
        if self.alternative == AlternativeType.TWO_SIDED:
            self.mixture = TwoSidedNormalMixture(self.v_opt, self.alpha_opt)
        else:
            self.mixture = OneSidedNormalMixture(self.v_opt, self.alpha_opt)

        def e_calculator(data):
            ### filtration step
            n = len(data)
            cumsum = np.sum(data)

            self.data_sum += cumsum
            self.data_sum_squares += np.sum(data**2)
            self.data_count += n

            s = self.data_sum - self.null_value * self.data_count # centered sum process
            if self.alternative == AlternativeType.LESS:
                s = -s

            if self.known_variance is not None:
                v = self.data_count * self.known_variance
            elif self.use_empirical_variance and self.data_count > max(1, self.min_samples_for_update):
                sample_mean = self.data_sum / self.data_count
                var_estimate = (self.data_sum_squares / self.data_count - sample_mean**2) * (self.data_count / (self.data_count - 1))

                if self.variance_bound:
                    var_estimate = min(var_estimate, self.variance_bound)

                v = max(self.data_count * var_estimate, 0.01)

                if self.data_count >= 2 * self.min_samples_for_update:
                    self.v_opt = v / self.data_count  # Update optimal intrinsic time
                    if self.alternative == AlternativeType.TWO_SIDED:
                        self.mixture = TwoSidedNormalMixture(self.v_opt, self.alpha_opt)
                    else:
                        self.mixture = OneSidedNormalMixture(self.v_opt, self.alpha_opt)
            else:
                v = self.data_count * 1.0

            log_e_cumulative = self.mixture.log_superMG(s, v)

            if np.isnan(log_e_cumulative) or np.isinf(log_e_cumulative):
                if log_e_cumulative > 0:
                    log_e_cumulative = np.log(1e10)
                else:
                    log_e_cumulative = np.log(1e-10)

            # Compute sequential e-value: E_t = E_{1:t} / E_{1:t-1}
            # In log space: log(E_t) = log(E_{1:t}) - log(E_{1:t-1})
            log_e_sequential = log_e_cumulative - self.previous_log_e_cumulative
            self.previous_log_e_cumulative = log_e_cumulative

            return np.exp(log_e_sequential)

        self.e_calculator = e_calculator
    
    def _setup_proportion_test(self):
        if not 0 < self.null_value < 1:
            raise ValueError(f"Null proportion must be in (0,1), got {self.null_value}")

        t_opt = max(100, 1 / (self.null_value * (1 - self.null_value)))  # Adaptive t_opt
        is_one_sided = self.alternative != AlternativeType.TWO_SIDED

        self.mixture = BetaBinomialMixture(
            t_opt * self.null_value * (1 - self.null_value),
            self.alpha_opt,
            self.null_value,
            1 - self.null_value,
            is_one_sided
        )

        def e_calculator(data):
            if not np.all(np.isin(data, [0, 1])):
                raise ValueError("Proportion test requires binary (0/1) data")

            successes = np.sum(data)
            trials = len(data)

            self.data_sum += successes
            self.data_count += trials

            if self.alternative == AlternativeType.LESS:
                s = self.data_count * self.null_value - self.data_sum
                v = self.data_count * self.null_value * (1 - self.null_value)
            elif self.alternative == AlternativeType.GREATER:
                s = self.data_sum - self.data_count * self.null_value
                v = self.data_count * self.null_value * (1 - self.null_value)
            else:  ## two-sided
                s = abs(self.data_sum - self.data_count * self.null_value)
                v = self.data_count * self.null_value * (1 - self.null_value)

            log_e_cumulative = self.mixture.log_superMG(s, v)

            log_e_sequential = log_e_cumulative - self.previous_log_e_cumulative
            self.previous_log_e_cumulative = log_e_cumulative

            return np.exp(log_e_sequential)

        self.e_calculator = e_calculator
    
    def _setup_variance_test(self):
        # TODO: validate this part, i am unsure if this is valid/optimal
        self.mixture = GammaExponentialMixture(100, self.alpha_opt, c=np.sqrt(2))

        def e_calculator(data):
            n = len(data)

            self.data_sum += np.sum(data)
            self.data_sum_squares += np.sum(data**2)
            self.data_count += n

            if self.data_count <= 1:
                return 1.0

            sample_mean = self.data_sum / self.data_count
            sample_var = (self.data_sum_squares - self.data_count * sample_mean**2) / (self.data_count - 1)

            # H0: variance ≤ null_value vs H1: variance > null_value
            chi_squared_stat = (self.data_count - 1) * sample_var / self.null_value

            s = chi_squared_stat
            v = self.data_count - 1

            # Compute cumulative log e-value
            log_e_cumulative = self.mixture.log_superMG(s, v)

            # Compute sequential e-value: E_t = E_{1:t} / E_{1:t-1}
            log_e_sequential = log_e_cumulative - self.previous_log_e_cumulative
            self.previous_log_e_cumulative = log_e_cumulative

            return np.exp(log_e_sequential)

        self.e_calculator = e_calculator
    
    def _setup_quantile_test(self):
        if self.quantile is None:
            raise ValueError("Quantile must be specified for quantile tests")

        if not 0 < self.quantile < 1:
            raise ValueError(f"Quantile must be in (0,1), got {self.quantile}")

        t_opt = max(100, 1 / (self.quantile * (1 - self.quantile)))
        is_one_sided = self.alternative != AlternativeType.TWO_SIDED

        self.mixture = BetaBinomialMixture(
            t_opt * self.quantile * (1 - self.quantile),
            self.alpha_opt,
            self.quantile,
            1 - self.quantile,
            is_one_sided
        )

        self.order_stats = None

        def e_calculator(data):
            flat_data = np.asarray(data).flatten().tolist()
            self.all_data.extend(flat_data)
            n = len(self.all_data)

            if n < 2:
                return 1.0

            self.order_stats = StaticOrderStatistics(self.all_data)

            count_below = self.order_stats.count_less(self.null_value)
            prop_below = count_below / n

            if self.alternative == AlternativeType.GREATER:
                s = (prop_below - self.quantile) * n
            elif self.alternative == AlternativeType.LESS:
                s = (self.quantile - prop_below) * n
            else:  # TWO_SIDED
                s = abs(prop_below - self.quantile) * n

            v = self.quantile * (1 - self.quantile) * n

            log_e_cumulative = self.mixture.log_superMG(s, v)

            log_e_sequential = log_e_cumulative - self.previous_log_e_cumulative
            self.previous_log_e_cumulative = log_e_cumulative

            return np.exp(log_e_sequential)

        self.e_calculator = e_calculator
    
    def update(self, new_data: Union[float, List[float], NDArray]) -> SequentialTestResult:
        data = np.asarray(new_data).flatten()
        
        if len(data) == 0:
            raise ValueError("No data provided")
        
        try:
            e_value = self.e_calculator(data)
        except Exception as e:
            raise ValueError(f"Error calculating e-value: {str(e)}")
        
        self.e_process_updater.update(self.e_process, e_value)
        
        reject_null = self.e_process_updater.is_significant(self.e_process)
        stopping_time = self.e_process_updater.get_stopping_time(self.e_process)
        current_value = self.e_process_updater.get_current_value(self.e_process)
        
        if reject_null and stopping_time and stopping_time not in self.rejection_times:
            self.rejection_times.append(stopping_time)
        
        e_power_result = None
        e_power = None
        e_power_is_positive = None
        
        if self.e_power_config and self.e_process.values:
            e_power_result = self.e_power_calculator.compute(
                e_values=np.array(self.e_process.values),
                alternative_prob=None
            )
            e_power = e_power_result.e_power
            e_power_is_positive = e_power_result.is_positive
        
            self.e_power_history.append(e_power)
        
        p_value = min(1.0, 1.0/current_value) if current_value > 0 else 1.0
        
        confidence_bounds = self._compute_confidence_bounds()
        
        optimal_lambda = None
        if self.e_process.lambdas:
            optimal_lambda = self.e_process.lambdas[-1]
            self.lambda_history.append(optimal_lambda)

        self.history.append({
            'step': len(self.history) + 1,
            'observations': data.tolist(),
            'e_value': e_value,  # raw e-value from this update (not cumulative)
            'cumulative_e_value': current_value,  # cumulative e-process value
            'reject_null': reject_null,
            'p_value': p_value,
            'sample_size': self.data_count,
            'confidence_lower': confidence_bounds[0] if confidence_bounds else None,
            'confidence_upper': confidence_bounds[1] if confidence_bounds else None,
            'stopping_time': stopping_time,
            'max_e_value': self.e_process_updater.get_max_value(self.e_process),
            'timestamp': np.datetime64('now').astype(float),
            'e_power': e_power,
            'e_power_is_positive': e_power_is_positive,
            'optimal_lambda': optimal_lambda
        })

        return SequentialTestResult(
            reject_null=reject_null,
            e_value=e_value,  # sequential e-value from this update
            e_process=self.e_process,
            sample_size=self.data_count,
            p_value=p_value,
            confidence_bounds=confidence_bounds,
            test_type=self.test_type,
            alternative=self.alternative,
            e_power=e_power,
            e_power_is_positive=e_power_is_positive,
            optimal_lambda=optimal_lambda,
            e_power_result=e_power_result
        )
    
    def _compute_confidence_bounds(self) -> Optional[tuple[float, float]]:
        if self.data_count == 0:
            return None

        alpha = self.config.significance_level

        if self.test_type == TestType.MEAN:
            return self._compute_mean_confidence_bounds(alpha)

        elif self.test_type == TestType.PROPORTION:
            return self._compute_proportion_confidence_bounds(alpha)

        elif self.test_type == TestType.VARIANCE:
            return self._compute_variance_confidence_bounds(alpha)

        elif self.test_type == TestType.QUANTILE:
            return self._compute_quantile_confidence_bounds(alpha)

        return None

    def _compute_mean_confidence_bounds(self, alpha: float) -> Optional[tuple[float, float]]:
        if self.data_count < 1:
            return None

        mean_estimate = self.data_sum / self.data_count

        # Update intrinsic time (variance-adjusted time)
        if self.known_variance is not None:
            self.intrinsic_time = self.data_count * self.known_variance
        else:
            if self.data_count > 1:
                # Compute empirical variance using Welford's online algorithm
                self.empirical_variance = (
                    self.data_sum_squares - self.data_sum**2 / self.data_count
                ) / (self.data_count - 1)
                self.intrinsic_time = self.data_count * max(self.empirical_variance, 0.01)
            else:
                self.intrinsic_time = self.variance_bound if self.variance_bound else 1.0

        if self.boundary_config.boundary_type == BoundaryType.NORMAL:
            # Normal mixture bound for sub-Gaussian data
            is_one_sided = self.alternative != AlternativeType.TWO_SIDED
            radius = boundaries.normal_mixture_bound(
                v=np.array([self.intrinsic_time]),
                alpha=alpha,
                v_opt=self.boundary_config.v_opt or self.intrinsic_time,
                alpha_opt=self.boundary_config.alpha_opt,
                is_one_sided=is_one_sided
            )[0]
        elif self.boundary_config.boundary_type == BoundaryType.GAMMA_EXPONENTIAL:
            # Gamma-exponential for heavier tails
            radius = boundaries.gamma_exponential_mixture_bound(
                v=np.array([self.intrinsic_time]),
                alpha=alpha,
                v_opt=self.boundary_config.v_opt or self.intrinsic_time,
                c=self.boundary_config.c or 1.0,
                alpha_opt=self.boundary_config.alpha_opt
            )[0]
        elif self.boundary_config.boundary_type == BoundaryType.POLY_STITCHING:
            # Poly-stitching for very heavy tails
            radius = boundaries.poly_stitching_bound(
                v=np.array([self.intrinsic_time]),
                alpha=alpha,
                v_min=self.boundary_config.v_min or 0.5,
                c=0,  # For mean, centered process
                s=self.boundary_config.s,
                eta=self.boundary_config.eta
            )[0]
        else:
            is_one_sided = self.alternative != AlternativeType.TWO_SIDED
            radius = boundaries.normal_mixture_bound(
                v=np.array([self.intrinsic_time]),
                alpha=alpha,
                v_opt=self.intrinsic_time,
                alpha_opt=alpha,
                is_one_sided=is_one_sided
            )[0]

        # CRITICAL FIX: The boundary gives a bound for the sum, not the mean
        # So we divide by n to get the bound for the mean
        scaled_radius = radius / self.data_count

        return (mean_estimate - scaled_radius, mean_estimate + scaled_radius)

    def _compute_proportion_confidence_bounds(self, alpha: float) -> Optional[tuple[float, float]]:
        if self.data_count < 1:
            return None

        # Use exact Bernoulli confidence interval
        if self.boundary_config.boundary_type == BoundaryType.BETA_BINOMIAL:
            lower, upper = boundaries.bernoulli_confidence_interval(
                num_successes=self.data_sum,
                num_trials=self.data_count,
                alpha=alpha,
                t_opt=self.boundary_config.v_opt or float(self.min_samples_for_update),
                alpha_opt=self.boundary_config.alpha_opt
            )
            return (lower, upper)
        else:
            # Fallback to beta-binomial mixture bound
            prop_estimate = self.data_sum / self.data_count

            # Intrinsic time for Bernoulli is n * p * (1-p)
            self.intrinsic_time = self.data_count * prop_estimate * (1 - prop_estimate)

            # Use beta-binomial mixture
            g = self.boundary_config.g or prop_estimate
            h = self.boundary_config.h or (1 - prop_estimate)

            radius = boundaries.beta_binomial_mixture_bound(
                v=np.array([self.intrinsic_time]),
                alpha=alpha,
                v_opt=self.boundary_config.v_opt or 0.25 * self.data_count,
                g=g,
                h=h,
                alpha_opt=self.boundary_config.alpha_opt,
                is_one_sided=(self.alternative != AlternativeType.TWO_SIDED)
            )[0]

            scaled_radius = radius / self.data_count
            lower = max(0, prop_estimate - scaled_radius)
            upper = min(1, prop_estimate + scaled_radius)

            return (lower, upper)

    def _compute_quantile_confidence_bounds(self, alpha: float) -> Optional[tuple[float, float]]:
        if not hasattr(self, 'order_stats') or self.order_stats is None:
            return None

        n = self.order_stats.size()
        if n < 2:
            return None

        # Use double-stitching bound for quantile confidence sequences
        if self.boundary_config.boundary_type == BoundaryType.DOUBLE_STITCHING:
            radius = boundaries.double_stitching_bound(
                quantile_p=self.quantile,
                t=float(n),
                alpha=alpha,
                t_opt=self.boundary_config.v_opt or float(self.min_samples_for_update),
                delta=self.boundary_config.delta,
                s=self.boundary_config.s,
                eta=self.boundary_config.eta
            )
        else:
            # Fallback to empirical process LIL bound
            if n >= (self.boundary_config.t_min or 5):
                radius = boundaries.empirical_process_lil_bound(
                    t=float(n),
                    alpha=alpha,
                    t_min=self.boundary_config.t_min or 5.0,
                    A=self.boundary_config.A
                ) * np.sqrt(n)
            else:
                radius = np.sqrt(n * np.log(2/alpha))

        # Convert radius to order statistics indices
        k = int(self.quantile * n)
        lower_idx = max(1, k - int(radius))
        upper_idx = min(n, k + int(radius))

        lower_bound = self.order_stats.get_order_statistic(lower_idx)
        upper_bound = self.order_stats.get_order_statistic(upper_idx)

        return (lower_bound, upper_bound)

    def _compute_variance_confidence_bounds(self, alpha: float) -> Optional[tuple[float, float]]:
        if self.data_count < 2:
            return None

        sample_mean = self.data_sum / self.data_count
        sample_variance = (
            self.data_sum_squares - self.data_count * sample_mean**2
        ) / (self.data_count - 1)

        # TODO: validate below
        # For variance, we use chi-squared type bounds
        # Intrinsic time is related to the fourth moment
        centered_data_sq = self.data_sum_squares - 2 * sample_mean * self.data_sum + self.data_count * sample_mean**2
        self.intrinsic_time = centered_data_sq

        if self.boundary_config.boundary_type == BoundaryType.GAMMA_EXPONENTIAL:
            # Gamma-exponential is appropriate for chi-squared type statistics
            radius = boundaries.gamma_exponential_mixture_bound(
                v=np.array([self.intrinsic_time]),
                alpha=alpha/2,  # Two-sided for variance
                v_opt=self.boundary_config.v_opt or self.intrinsic_time,
                c=self.boundary_config.c or np.sqrt(2),  # Chi-squared has c=sqrt(2)
                alpha_opt=self.boundary_config.alpha_opt
            )[0]
        elif self.boundary_config.boundary_type == BoundaryType.GAMMA_POISSON:
            # Alternative for discrete-like variance
            radius = boundaries.gamma_poisson_mixture_bound(
                v=np.array([self.intrinsic_time]),
                alpha=alpha/2,
                v_opt=self.boundary_config.v_opt or self.intrinsic_time,
                c=self.boundary_config.c or 1.0,
                alpha_opt=self.boundary_config.alpha_opt
            )[0]
        else:
            radius = boundaries.gamma_exponential_mixture_bound(
                v=np.array([self.intrinsic_time]),
                alpha=alpha/2,
                v_opt=self.intrinsic_time,
                c=np.sqrt(2),
                alpha_opt=alpha
            )[0]

        # Scale the radius appropriately for variance bounds
        df = self.data_count - 1
        scaled_radius = radius * sample_variance / np.sqrt(2 * df)

        lower = max(0, sample_variance - scaled_radius)
        upper = sample_variance + scaled_radius

        return (lower, upper)
    
    def get_summary(self) -> dict:
        summary = {
            "test_type": self.test_type.value,
            "null_value": self.null_value,
            "alternative": self.alternative.value,
            "sample_size": self.data_count,
        
            "current_e_value": self.e_process_updater.get_current_value(self.e_process),
            "max_e_value": self.e_process_updater.get_max_value(self.e_process),
            "is_significant": self.e_process_updater.is_significant(self.e_process),
            "stopping_time": self.e_process_updater.get_stopping_time(self.e_process),
            "p_value": min(1.0, 1.0/self.e_process_updater.get_current_value(self.e_process)) 
                       if self.e_process_updater.get_current_value(self.e_process) > 0 else 1.0,
            
            "empirical_e_power": self.e_process_updater.compute_empirical_e_power(self.e_process),
            "asymptotic_growth_rate": self.e_process_updater.compute_asymptotic_growth_rate(self.e_process),
        }
        
        if self.test_type == TestType.MEAN and self.data_count > 0:
            summary["sample_mean"] = self.data_sum / self.data_count
            if self.data_count > 1:
                summary["sample_variance"] = (self.data_sum_squares - self.data_sum**2 / self.data_count) / (self.data_count - 1)
            summary["v_opt"] = self.v_opt
            
        elif self.test_type == TestType.PROPORTION and self.data_count > 0:
            summary["sample_proportion"] = self.data_sum / self.data_count
            
        elif self.test_type == TestType.VARIANCE and self.data_count > 1:
            sample_mean = self.data_sum / self.data_count
            summary["sample_variance"] = (self.data_sum_squares - self.data_count * sample_mean**2) / (self.data_count - 1)
            
        elif self.test_type == TestType.QUANTILE and self.all_data:
            summary["empirical_quantile"] = np.quantile(self.all_data, self.quantile)
            summary["n_observations"] = len(self.all_data)
        
        if isinstance(self.config, EProcessConfig):
            summary["betting_strategy"] = self.config.betting_strategy.value
            if self.e_process.lambdas:
                summary["mean_lambda"] = np.mean(self.e_process.lambdas)
                summary["current_lambda"] = self.e_process.lambdas[-1]
        
        if self.e_power_config:
            summary["e_power_type"] = self.e_power_config.type.value
            if self.e_power_history:
                summary["mean_e_power"] = np.mean(self.e_power_history)
                summary["max_e_power"] = np.max(self.e_power_history)
        
        if self.rejection_times:
            summary["first_rejection_time"] = min(self.rejection_times)
            summary["n_rejections"] = len(self.rejection_times)
        
        return summary

    def get_history_df(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    def reset(self):
        self.data_sum = 0.0
        self.data_sum_squares = 0.0
        self.data_count = 0
        self.all_data = []
        self.order_stats = None
        
        self.e_process = EProcess(config=self.config)
        
        self.e_process_updater = EProcessUpdater(self.config)
        
        self.v_opt = 1.0
        
        self.e_power_history = []
        self.lambda_history = []
        self.rejection_times = []
        self.history = []
        
        self._setup_evaluator()
