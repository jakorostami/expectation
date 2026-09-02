# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""
Parallel sequential hypothesis testing with Rust acceleration.

Provides a Pythonic interface for massively parallel sequential hypothesis
testing over 300K+ independent tests using e-values and e-processes.

The engine delegates the per-test hot loop and cross-test multiple testing
corrections to Rust (rayon parallelism, SoA memory layout) while exposing
Pydantic-typed configuration and results on the Python side.

When ``global_merge`` is configured, the engine additionally merges all K
per-step e-values into a single merged e-value (Vovk & Wang 2024, Corollary 1)
and accumulates it temporally into an e-process for the intersection
hypothesis (Ramdas & Wang 2025, Definition 7.21).

Based on these papers:

Hypothesis testing with e-values, A. Ramdas, R. Wang (2025)
    - Ch. 4: Multiple testing with e-values (e-BH, e-Bonferroni, e-Holm)
    - Ch. 7: E-processes and sequential e-values (Definition 7.21, Proposition 7.20)
    - Ch. 8: Merging e-values (Definitions 8.1, 8.5, 8.9, 8.10, Theorem 8.4)

Merging sequential e-values via martingales, V. Vovk, R. Wang (2024)
    - Section 4: Theorem 1, Corollary 1

Time-uniform, nonparametric, nonasymptotic confidence sequences,
S.R. Howard, A. Ramdas, J. McAuliffe, J. Sekhon (2022)
    - Section 3: Normal mixture supermartingale

Estimating means of bounded random variables by betting,
I. Waudby-Smith, A. Ramdas (2024)
    - ONS-based adaptive betting (Theorem 7.22 in Ramdas & Wang 2025)
"""

from enum import Enum
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from expectation._rust import PyParallelSequentialTest


class MartingaleType(str, Enum):
    TWO_SIDED_NORMAL = "two_sided_normal"
    ONE_SIDED_NORMAL = "one_sided_normal"


class AlternativeDirection(str, Enum):
    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"


class CombinerStrategy(str, Enum):
    ALL_IN = "all_in"
    CONSERVATIVE = "conservative"
    EMPIRICALLY_ADAPTIVE = "empirically_adaptive"


class VarianceMode(str, Enum):
    """How variance is determined for each test.

    KNOWN_HOMOGENEOUS: Single known variance shared across all tests.
    KNOWN_HETEROGENEOUS: Per-test known variances (array input).
    EMPIRICAL: Online Welford estimation after min_samples observations.
    """

    KNOWN_HOMOGENEOUS = "known_homogeneous"
    KNOWN_HETEROGENEOUS = "known_heterogeneous"
    EMPIRICAL = "empirical"


class MultipleTestingMethod(str, Enum):
    E_BONFERRONI = "e_bonferroni"
    E_BH = "e_bh"
    E_HOLM = "e_holm"


class AdjusterType(str, Enum):
    LOOKBACK = "lookback"
    SQRT = "sqrt"


class MergingMethod(str, Enum):
    ARITHMETIC_MEAN = "arithmetic_mean"
    U_STATISTIC = "u_statistic"
    LAMBDA_PRODUCT = "lambda_product"
    SEGMENT_PRODUCT = "segment_product"
    PRODUCT = "product"


class ParallelTestConfig(BaseModel):
    """Configuration for the parallel sequential testing engine.

    Follows the codebase convention of frozen Pydantic models for all
    configuration objects (cf. EValueConfig, EProcessConfig).

    Parameters
    ----------
    n_tests : int
        Number of simultaneous hypothesis tests.
    alpha : float
        Significance level for per-test Ville rejection
        (Ramdas & Wang 2025, Definition 2.4).
    martingale_type : MartingaleType
        Which mixture supermartingale to use.
    v_opt : float
        Optimal intrinsic time for the mixing parameter rho
        (Howard et al. 2022, Section 3).
    alpha_opt : float
        Optimal alpha for computing rho.
    variance_mode : VarianceMode
        How variance is determined per test.
    min_samples : int
        Minimum observations before using empirical variance (Welford).
    alternative : AlternativeDirection
        Alternative hypothesis direction (Ramdas & Wang 2025, Section 2.1).
    combiner : CombinerStrategy
        How sequential e-values are combined (Ramdas & Wang 2025, Definition 7.21).
    conservative_lambda : float
        Fixed lambda for conservative combiner. Must be in (0, 1).
    gamma : float
        Cap for adaptive combiner lambda. Must be in (0, 1].
    epsilon : float
        Regularization for adaptive combiner. Must be > 0.
    global_merge : MergingMethod, optional
        Merging function for intersection hypothesis testing.
        None (default) disables merging. (Vovk & Wang 2024, Corollary 1)
    merge_u_order : int
        U-statistic order n for U_STATISTIC merge. Default 1.
    merge_lambda_param : float
        Lambda parameter for LAMBDA_PRODUCT merge. Default 0.5.
    merge_segments : list of int, optional
        Segment boundaries for SEGMENT_PRODUCT merge.
    merge_combiner : CombinerStrategy
        Temporal combiner for the merged stream. Default ALL_IN.
    merge_conservative_lambda : float
        Lambda for conservative merge temporal combiner. Default 0.5.
    merge_gamma : float
        Cap for adaptive merge temporal combiner. Default 0.5.
    merge_epsilon : float
        Regularization for adaptive merge temporal combiner. Default 1e-6.
    merge_include_rejected : bool
        Include rejected tests in merge. Default True.
    """

    n_tests: int = Field(gt=0, description="Number of simultaneous hypothesis tests")
    alpha: float = Field(gt=0, lt=1, default=0.05, description="Significance level")
    martingale_type: MartingaleType = Field(default=MartingaleType.TWO_SIDED_NORMAL)
    v_opt: float = Field(gt=0, default=1.0, description="Optimal intrinsic time")
    alpha_opt: float = Field(gt=0, lt=1, default=0.05, description="Optimal alpha for rho")
    variance_mode: VarianceMode = Field(default=VarianceMode.KNOWN_HOMOGENEOUS)
    min_samples: int = Field(ge=1, default=30, description="Min samples for empirical variance")
    alternative: AlternativeDirection = Field(default=AlternativeDirection.TWO_SIDED)
    combiner: CombinerStrategy = Field(default=CombinerStrategy.ALL_IN)
    conservative_lambda: float = Field(
        gt=0, lt=1, default=0.5, description="Lambda for conservative combiner"
    )
    gamma: float = Field(gt=0, le=1, default=0.5, description="Cap for adaptive combiner lambda")
    epsilon: float = Field(gt=0, default=1e-6, description="Regularization for adaptive combiner")

    # ── Merge configuration (V&W 2024, R&W 2025 Ch. 8) ──
    global_merge: Optional[MergingMethod] = Field(
        default=None,
        description="Merging function for intersection hypothesis (None = disabled)",
    )
    merge_u_order: int = Field(default=1, ge=0, description="U-statistic order n")
    merge_lambda_param: float = Field(
        default=0.5, gt=0, le=1, description="Lambda for lambda-product merge"
    )
    merge_segments: Optional[List[int]] = Field(
        default=None, description="Segment boundaries for segment-product merge"
    )
    merge_combiner: CombinerStrategy = Field(
        default=CombinerStrategy.ALL_IN, description="Temporal combiner for merged stream"
    )
    merge_conservative_lambda: float = Field(
        default=0.5, gt=0, lt=1, description="Lambda for conservative merge combiner"
    )
    merge_gamma: float = Field(
        default=0.5, gt=0, le=1, description="Cap for adaptive merge combiner"
    )
    merge_epsilon: float = Field(
        default=1e-6, gt=0, description="Regularization for adaptive merge combiner"
    )
    merge_include_rejected: bool = Field(
        default=True, description="Include rejected tests in merge"
    )

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_merge(self) -> "ParallelTestConfig":
        if self.global_merge == MergingMethod.SEGMENT_PRODUCT:
            if self.merge_segments is None:
                raise ValueError("merge_segments is required when global_merge is SEGMENT_PRODUCT")
        if self.global_merge == MergingMethod.U_STATISTIC:
            if self.merge_u_order > self.n_tests:
                raise ValueError(
                    f"merge_u_order ({self.merge_u_order}) must be <= n_tests ({self.n_tests})"
                )
        return self


class StepResult(BaseModel):
    """Result of processing one time step across all tests.

    Returned by ``ParallelSequentialTest.step()``.
    Mirrors the Rust ``StepResult`` struct.

    Attributes
    ----------
    time_step : int
        Number of time steps processed (1-indexed).
    n_rejected : int
        Tests rejected by Ville's inequality at this step (no correction).
    n_tests : int
        Total number of tests.
    n_newly_rejected : int
        Number of tests newly rejected in this step.
    merged_e_value : float, optional
        Spatially merged e-value F(E_1^t, ..., E_K^t).
        Only populated when global_merge is configured.
    log_merged_e_value : float, optional
        Log of the spatially merged e-value.
    merged_e_process : float, optional
        Current temporal merged e-process M_t.
    log_merged_e_process : float, optional
        Log of the temporal merged e-process.
    merged_rejected : bool, optional
        Whether M_t >= 1/alpha (Ville's inequality on merged e-process).
    merged_p_value : float, optional
        Merged p-value: min(1, exp(-log M_t)).
    merged_lambda : float, optional
        Current merged temporal betting fraction.
    """

    time_step: int = Field(ge=1)
    n_rejected: int = Field(ge=0)
    n_tests: int = Field(gt=0)
    n_newly_rejected: int = Field(ge=0)

    # Merged fields (populated only when global_merge is configured)
    merged_e_value: Optional[float] = None
    log_merged_e_value: Optional[float] = None
    merged_e_process: Optional[float] = None
    log_merged_e_process: Optional[float] = None
    merged_rejected: Optional[bool] = None
    merged_p_value: Optional[float] = None
    merged_lambda: Optional[float] = None

    model_config = ConfigDict(frozen=True)


class MultipleTestingResult(BaseModel):
    """Result of a cross-test multiple testing correction.

    Returned by ``e_bonferroni()``, ``e_bh()``, ``e_holm()``.

    Attributes
    ----------
    rejected : NDArray[np.bool_]
        Per-test rejection flags after correction.
    n_rejected : int
        Total number of rejections.
    method : MultipleTestingMethod
        Which correction was applied.
    alpha : float
        Target error level used.

    References
    ----------
    Ramdas & Wang (2025), Ch. 4: Multiple testing with e-values.
    """

    rejected: NDArray[np.bool_]
    n_rejected: int = Field(ge=0)
    method: MultipleTestingMethod
    alpha: float = Field(gt=0, lt=1)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class ParallelSequentialTest:
    """High-performance parallel engine for massively concurrent sequential tests.

    Processes 300K+ tests simultaneously using Rust + rayon parallelism
    with Structure-of-Arrays memory layout. Each test runs an independent
    sequential e-process with anytime-valid guarantees.

    The engine uses Ville's inequality (Ramdas & Wang 2025, Theorem 2.5)
    for per-test stopping and supports cross-test error control via
    e-Bonferroni (FWER), e-BH (FDR), and e-Holm (FWER).

    When ``global_merge`` is configured, the engine additionally merges
    all K per-step e-values into a single merged e-value (Vovk & Wang 2024)
    and accumulates it temporally into an e-process for the intersection
    hypothesis (Ramdas & Wang 2025, Definition 7.21).

    Parameters
    ----------
    config : ParallelTestConfig
        Engine configuration (frozen Pydantic model).
    null_values : float or NDArray[np.float64]
        Per-test null hypothesis values. Scalar is broadcast to all tests.
    variance : float, NDArray[np.float64], or None
        Known variance. Scalar for homogeneous, array for heterogeneous.
        None for empirical variance estimation (Welford's method).

    References
    ----------
    - Ramdas & Wang (2025). Hypothesis testing with e-values, Ch. 4, 7 & 8.
    - Vovk & Wang (2024). Merging sequential e-values via martingales.
    - Howard, Ramdas, McAuliffe, Sekhon (2022). Time-uniform confidence
      sequences, Section 3.
    - Waudby-Smith & Ramdas (2024). Estimating means of bounded random
      variables by betting.
    """

    def __init__(
        self,
        config: ParallelTestConfig,
        null_values: Union[float, NDArray[np.float64]],
        variance: Optional[Union[float, NDArray[np.float64]]] = None,
    ):
        self._config = config

        # Broadcast scalar null_values
        if isinstance(null_values, (int, float)):
            null_values = np.full(config.n_tests, float(null_values))
        null_arr = np.asarray(null_values, dtype=np.float64)

        # Convert to Python lists for clean PyO3 extraction
        var_arg = variance
        if var_arg is not None and not isinstance(var_arg, (int, float)):
            var_arg = np.asarray(var_arg, dtype=np.float64).tolist()

        # Auto-select martingale: GREATER/LESS -> one_sided_normal
        martingale_type = config.martingale_type.value
        if config.alternative in (AlternativeDirection.GREATER, AlternativeDirection.LESS):
            if config.martingale_type == MartingaleType.TWO_SIDED_NORMAL:
                martingale_type = MartingaleType.ONE_SIDED_NORMAL.value

        # Build merge kwargs
        merge_kwargs = {}
        if config.global_merge is not None:
            merge_kwargs["global_merge"] = config.global_merge.value
            merge_kwargs["merge_u_order"] = config.merge_u_order
            merge_kwargs["merge_lambda_param"] = config.merge_lambda_param
            if config.merge_segments is not None:
                merge_kwargs["merge_segments"] = list(config.merge_segments)
            merge_kwargs["merge_combiner"] = config.merge_combiner.value
            merge_kwargs["merge_conservative_lambda"] = config.merge_conservative_lambda
            merge_kwargs["merge_gamma"] = config.merge_gamma
            merge_kwargs["merge_epsilon"] = config.merge_epsilon
            merge_kwargs["merge_include_rejected"] = config.merge_include_rejected

        self._inner = PyParallelSequentialTest(
            n_tests=config.n_tests,
            null_values=null_arr.tolist(),
            alpha=config.alpha,
            martingale_type=martingale_type,
            v_opt=config.v_opt,
            alpha_opt=config.alpha_opt,
            variance=var_arg,
            min_samples=config.min_samples,
            alternative=config.alternative.value,
            combiner=config.combiner.value,
            conservative_lambda=config.conservative_lambda,
            gamma=config.gamma,
            epsilon=config.epsilon,
            **merge_kwargs,
        )

    @property
    def config(self) -> ParallelTestConfig:
        """Engine configuration (frozen)."""
        return self._config

    def step(self, observations: NDArray[np.float64]) -> StepResult:
        """Process one observation per test for this time step.

        Parameters
        ----------
        observations : NDArray[np.float64]
            One observation per test, shape (n_tests,).

        Returns
        -------
        StepResult
            Frozen Pydantic model with time_step, n_rejected, n_tests,
            n_newly_rejected, and optional merged fields.
        """
        observations = np.asarray(observations, dtype=np.float64)
        raw = self._inner.step(observations)
        return StepResult(
            time_step=raw["time_step"],
            n_rejected=raw["n_rejected"],
            n_tests=raw["n_tests"],
            n_newly_rejected=raw["n_newly_rejected"],
            merged_e_value=raw.get("merged_e_value"),
            log_merged_e_value=raw.get("log_merged_e_value"),
            merged_e_process=raw.get("merged_e_process"),
            log_merged_e_process=raw.get("log_merged_e_process"),
            merged_rejected=raw.get("merged_rejected"),
            merged_p_value=raw.get("merged_p_value"),
            merged_lambda=raw.get("merged_lambda"),
        )

    def log_e_processes(self) -> NDArray[np.float64]:
        """Current log e-process values, shape (n_tests,).

        Each value is log M_t where M_t is the running e-process
        (Ramdas & Wang 2025, Definition 7.21).
        """
        return np.asarray(self._inner.log_e_processes())

    def rejected(self) -> NDArray[np.bool_]:
        """Per-test rejection flags via Ville's inequality (no correction).

        A test is rejected if max_{s<=t} M_s >= 1/alpha
        (Ramdas & Wang 2025, Theorem 2.5).
        """
        return np.asarray(self._inner.rejected())

    def log_e_sequential(self) -> NDArray[np.float64]:
        """Per-step sequential log e-values: log(E_t).

        E_t = exp(log_e_cum_t - log_e_cum_{t-1}).
        (Ramdas & Wang 2025, Ch. 7).
        """
        return np.asarray(self._inner.log_e_sequential())

    def p_values(self) -> NDArray[np.float64]:
        """Per-test p-values: min(1, exp(-log_e_process)).

        (Ramdas & Wang 2025, Proposition 2.2).
        """
        return np.asarray(self._inner.p_values())

    def stopping_times(self) -> NDArray[np.uint64]:
        """Per-test stopping times (first rejection step, 0 = not stopped)."""
        return np.asarray(self._inner.stopping_times())

    def lambdas(self) -> NDArray[np.float64]:
        """Per-test current betting fractions (lambda)."""
        return np.asarray(self._inner.lambdas())

    # ── Merge accessors ───────────────────────────────────────────────

    def merged_e_value(self) -> Optional[float]:
        """Current merged e-value (spatial). None if merge not configured."""
        return self._inner.merged_e_value()

    def log_merged_e_process(self) -> Optional[float]:
        """Current log merged e-process (temporal). None if merge not configured."""
        return self._inner.log_merged_e_process()

    def merged_rejected(self) -> Optional[bool]:
        """Whether the intersection null has been rejected. None if merge not configured."""
        return self._inner.merged_rejected()

    def merged_p_value(self) -> Optional[float]:
        """Merged p-value. None if merge not configured."""
        return self._inner.merged_p_value()

    def merged_stopping_time(self) -> Optional[int]:
        """Merged stopping time (0 = not stopped). None if merge not configured."""
        val = self._inner.merged_stopping_time()
        return int(val) if val is not None else None

    def merged_lambda(self) -> Optional[float]:
        """Current merged temporal lambda. None if merge not configured."""
        return self._inner.merged_lambda()

    # ── Multiple testing corrections ──────────────────────────────────

    def e_bonferroni(self, alpha: Optional[float] = None) -> MultipleTestingResult:
        """Apply e-Bonferroni correction for FWER control.

        WARNING: Not carefree. Rejections can disappear with more data.
        For FWER-sup control with monotone rejections, use
        ``adjusted_e_bonferroni()``.

        Reference: Ramdas & Wang (2025), Section 4.1, Proposition 4.1.
        See also: Tavyrikov, Goeman & de Heide (2025), arXiv:2501.19360v2.
        """
        used_alpha = alpha if alpha is not None else self._config.alpha
        raw = self._inner.e_bonferroni(alpha=alpha)
        return MultipleTestingResult(
            rejected=np.asarray(raw["rejected"]),
            n_rejected=raw["n_rejected"],
            method=MultipleTestingMethod.E_BONFERRONI,
            alpha=used_alpha,
        )

    def e_bh(self, alpha: Optional[float] = None) -> MultipleTestingResult:
        """Apply e-BH procedure for FDR control.

        WARNING: Not carefree. Rejections can disappear with more data.
        For FDR-sup control with monotone rejections, use
        ``adjusted_e_bh()``.

        Reference: Ramdas & Wang (2025), Section 4.2, Theorem 4.2.
        See also: Tavyrikov, Goeman & de Heide (2025), arXiv:2501.19360v2.
        """
        used_alpha = alpha if alpha is not None else self._config.alpha
        raw = self._inner.e_bh(alpha=alpha)
        return MultipleTestingResult(
            rejected=np.asarray(raw["rejected"]),
            n_rejected=raw["n_rejected"],
            method=MultipleTestingMethod.E_BH,
            alpha=used_alpha,
        )

    def e_holm(self, alpha: Optional[float] = None) -> MultipleTestingResult:
        """Apply e-Holm step-down procedure for FWER control.

        WARNING: Not carefree. Rejections can disappear with more data.
        For FWER-sup control with monotone rejections, use
        ``adjusted_e_holm()``.

        Reference: Ramdas & Wang (2025), Section 4.1, Proposition 4.3.
        See also: Tavyrikov, Goeman & de Heide (2025), arXiv:2501.19360v2.
        """
        used_alpha = alpha if alpha is not None else self._config.alpha
        raw = self._inner.e_holm(alpha=alpha)
        return MultipleTestingResult(
            rejected=np.asarray(raw["rejected"]),
            n_rejected=raw["n_rejected"],
            method=MultipleTestingMethod.E_HOLM,
            alpha=used_alpha,
        )

    # ── Running maxima accessor ────────────────────────────────────

    def max_log_m(self) -> NDArray[np.float64]:
        """Per-test running maxima: log(max_{s<=t} M_s), shape (n_tests,).

        Used by adjusted multiple testing procedures for carefree error
        control. Each value tracks the maximum log e-process seen so far
        for each test.

        Reference: Tavyrikov, Goeman & de Heide (2025), Section 2.
        """
        return np.asarray(self._inner.max_log_m())

    # ── Adjusted (carefree) multiple testing corrections ─────────

    def adjusted_e_bh(
        self,
        alpha: Optional[float] = None,
        adjuster: str = "lookback",
    ) -> MultipleTestingResult:
        """Apply adjusted e-BH for carefree FDR control.

        Applies an admissible adjuster to running maxima of e-processes,
        then runs e-BH. Controls FDR-sup at level K₀α/K, yielding
        monotonically non-decreasing rejections over time.

        Parameters
        ----------
        alpha : float, optional
            Target FDR level (default: engine alpha).
        adjuster : str
            Which admissible adjuster: "lookback" or "sqrt".
            Default "lookback".

        Returns
        -------
        MultipleTestingResult

        References
        ----------
        Tavyrikov, Goeman & de Heide (2025). Carefree multiple testing
        with e-processes. arXiv:2501.19360v2, Theorem 1.
        """
        used_alpha = alpha if alpha is not None else self._config.alpha
        raw = self._inner.adjusted_e_bh(alpha=alpha, adjuster=adjuster)
        return MultipleTestingResult(
            rejected=np.asarray(raw["rejected"]),
            n_rejected=raw["n_rejected"],
            method=MultipleTestingMethod.E_BH,
            alpha=used_alpha,
        )

    def adjusted_e_bonferroni(
        self,
        alpha: Optional[float] = None,
        adjuster: str = "lookback",
    ) -> MultipleTestingResult:
        """Apply adjusted e-Bonferroni for carefree FWER control.

        Applies an admissible adjuster to running maxima of e-processes,
        then runs e-Bonferroni.

        Parameters
        ----------
        alpha : float, optional
            Target FWER level (default: engine alpha).
        adjuster : str
            Which admissible adjuster: "lookback" or "sqrt".
            Default "lookback".

        Returns
        -------
        MultipleTestingResult

        References
        ----------
        Tavyrikov, Goeman & de Heide (2025). Carefree multiple testing
        with e-processes. arXiv:2501.19360v2, Theorem 1.
        """
        used_alpha = alpha if alpha is not None else self._config.alpha
        raw = self._inner.adjusted_e_bonferroni(
            alpha=alpha,
            adjuster=adjuster,
        )
        return MultipleTestingResult(
            rejected=np.asarray(raw["rejected"]),
            n_rejected=raw["n_rejected"],
            method=MultipleTestingMethod.E_BONFERRONI,
            alpha=used_alpha,
        )

    def adjusted_e_holm(
        self,
        alpha: Optional[float] = None,
        adjuster: str = "lookback",
    ) -> MultipleTestingResult:
        """Apply adjusted e-Holm for carefree FWER control.

        Applies an admissible adjuster to running maxima of e-processes,
        then runs e-Holm.

        Parameters
        ----------
        alpha : float, optional
            Target FWER level (default: engine alpha).
        adjuster : str
            Which admissible adjuster: "lookback" or "sqrt".
            Default "lookback".

        Returns
        -------
        MultipleTestingResult

        References
        ----------
        Tavyrikov, Goeman & de Heide (2025). Carefree multiple testing
        with e-processes. arXiv:2501.19360v2, Theorem 1.
        """
        used_alpha = alpha if alpha is not None else self._config.alpha
        raw = self._inner.adjusted_e_holm(alpha=alpha, adjuster=adjuster)
        return MultipleTestingResult(
            rejected=np.asarray(raw["rejected"]),
            n_rejected=raw["n_rejected"],
            method=MultipleTestingMethod.E_HOLM,
            alpha=used_alpha,
        )

    @property
    def n_tests(self) -> int:
        """Number of tests."""
        return self._inner.n_tests

    @property
    def time_step(self) -> int:
        """Current time step (number of observations processed)."""
        return self._inner.time_step

    @property
    def alpha(self) -> float:
        """Significance level."""
        return self._inner.alpha
