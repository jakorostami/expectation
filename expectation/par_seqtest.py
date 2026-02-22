"""
Parallel sequential hypothesis testing with Rust acceleration.

Provides a Pythonic interface for massively parallel sequential hypothesis
testing over 300K+ independent tests using e-values and e-processes.

The engine delegates the per-test hot loop and cross-test multiple testing
corrections to Rust (rayon parallelism, SoA memory layout) while exposing
Pydantic-typed configuration and results on the Python side.

Based on these papers:

Hypothesis testing with e-values, A. Ramdas, R. Wang (2025)
    - Ch. 4: Multiple testing with e-values (e-BH, e-Bonferroni, e-Holm)
    - Ch. 7: E-processes and sequential e-values (Proposition 7.20: all-in strategy)

Time-uniform, nonparametric, nonasymptotic confidence sequences,
S.R. Howard, A. Ramdas, J. McAuliffe, J. Sekhon (2022)
    - Section 3: Normal mixture supermartingale
"""

from enum import Enum
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from expectation._rust import PyParallelSequentialTest


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MartingaleType(str, Enum):
    """Mixture supermartingale types for the parallel engine.

    TWO_SIDED_NORMAL: Two-sided normal mixture (Howard et al. 2022, Section 3).
        log M(s, v) = 0.5 * ln(rho / (v + rho)) + s^2 / (2 * (v + rho))
    """
    TWO_SIDED_NORMAL = "two_sided_normal"


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
    """Cross-test error control procedures.

    E_BONFERRONI: FWER control via e-value Bonferroni
        (Ramdas & Wang 2025, Section 4.1, Proposition 4.1).
    E_BH: FDR control via e-value Benjamini-Hochberg
        (Ramdas & Wang 2025, Section 4.2, Theorem 4.2).
    E_HOLM: FWER control via e-value Holm step-down
        (Ramdas & Wang 2025, Section 4.1, Proposition 4.3).
    """
    E_BONFERRONI = "e_bonferroni"
    E_BH = "e_bh"
    E_HOLM = "e_holm"


# ---------------------------------------------------------------------------
# Pydantic config models (frozen=True)
# ---------------------------------------------------------------------------

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
    """
    n_tests: int = Field(gt=0, description="Number of simultaneous hypothesis tests")
    alpha: float = Field(gt=0, lt=1, default=0.05, description="Significance level")
    martingale_type: MartingaleType = Field(default=MartingaleType.TWO_SIDED_NORMAL)
    v_opt: float = Field(gt=0, default=1.0, description="Optimal intrinsic time")
    alpha_opt: float = Field(gt=0, lt=1, default=0.05, description="Optimal alpha for rho")
    variance_mode: VarianceMode = Field(default=VarianceMode.KNOWN_HOMOGENEOUS)
    min_samples: int = Field(ge=1, default=30, description="Min samples for empirical variance")

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Pydantic result models (frozen=True)
# ---------------------------------------------------------------------------

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
    """
    time_step: int = Field(ge=1)
    n_rejected: int = Field(ge=0)
    n_tests: int = Field(gt=0)

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


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ParallelSequentialTest:
    """High-performance parallel engine for massively concurrent sequential tests.

    Processes 300K+ tests simultaneously using Rust + rayon parallelism
    with Structure-of-Arrays memory layout. Each test runs an independent
    sequential e-process with anytime-valid guarantees.

    The engine uses Ville's inequality (Ramdas & Wang 2025, Theorem 2.5)
    for per-test stopping and supports cross-test error control via
    e-Bonferroni (FWER), e-BH (FDR), and e-Holm (FWER).

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
    - Ramdas & Wang (2025). Hypothesis testing with e-values, Ch. 4 & 7.
    - Howard, Ramdas, McAuliffe, Sekhon (2022). Time-uniform confidence
      sequences, Section 3.

    Example
    -------
    >>> import numpy as np
    >>> from expectation.par_seqtest import ParallelSequentialTest, ParallelTestConfig
    >>>
    >>> config = ParallelTestConfig(
    ...     n_tests=300_000,
    ...     alpha=0.05,
    ...     martingale_type="two_sided_normal",
    ...     v_opt=1.0,
    ...     alpha_opt=0.05,
    ... )
    >>> pst = ParallelSequentialTest(config=config, null_values=0.0, variance=1.0)
    >>> for t in range(100):
    ...     obs = np.random.randn(300_000)
    ...     result = pst.step(obs)
    ...     bh = pst.e_bh()
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
        # (avoids NumPy deprecation warning on ndim>0 to scalar conversion)
        var_arg = variance
        if var_arg is not None and not isinstance(var_arg, (int, float)):
            var_arg = np.asarray(var_arg, dtype=np.float64).tolist()

        self._inner = PyParallelSequentialTest(
            n_tests=config.n_tests,
            null_values=null_arr.tolist(),
            alpha=config.alpha,
            martingale_type=config.martingale_type.value,
            v_opt=config.v_opt,
            alpha_opt=config.alpha_opt,
            variance=var_arg,
            min_samples=config.min_samples,
        )

    @property
    def config(self) -> ParallelTestConfig:
        """Engine configuration (frozen)."""
        return self._config

    def step(self, observations: NDArray[np.float64]) -> StepResult:
        """Process one observation per test for this time step.

        Applies the mixture supermartingale update to each test in parallel
        (Proposition 7.20 in Ramdas & Wang 2025: all-in e-process).

        Parameters
        ----------
        observations : NDArray[np.float64]
            One observation per test, shape (n_tests,).

        Returns
        -------
        StepResult
            Frozen Pydantic model with time_step, n_rejected, n_tests.
        """
        observations = np.asarray(observations, dtype=np.float64)
        raw = self._inner.step(observations)
        return StepResult(
            time_step=raw["time_step"],
            n_rejected=raw["n_rejected"],
            n_tests=raw["n_tests"],
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

    def e_bonferroni(self, alpha: Optional[float] = None) -> MultipleTestingResult:
        """Apply e-Bonferroni correction for FWER control.

        Rejects test i if e_i >= m / alpha where m is the number of tests.
        Controls familywise error rate at level alpha.

        Reference: Ramdas & Wang (2025), Section 4.1, Proposition 4.1.

        Parameters
        ----------
        alpha : float, optional
            Target FWER level. Defaults to construction alpha.

        Returns
        -------
        MultipleTestingResult
            Frozen model with rejected mask, n_rejected, method, alpha.
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

        Sorts e-values descending and finds k* = max{k : e_{(k)} >= m/(k*alpha)}.
        Rejects the top k* tests. Controls false discovery rate at level alpha.

        Reference: Ramdas & Wang (2025), Section 4.2, Theorem 4.2.

        Parameters
        ----------
        alpha : float, optional
            Target FDR level. Defaults to construction alpha.

        Returns
        -------
        MultipleTestingResult
            Frozen model with rejected mask, n_rejected, method, alpha.
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

        Sorts e-values descending and rejects while e_{(k)} >= (m-k+1)/alpha.
        Tighter than e-Bonferroni via step-down rejection.

        Reference: Ramdas & Wang (2025), Section 4.1, Proposition 4.3.

        Parameters
        ----------
        alpha : float, optional
            Target FWER level. Defaults to construction alpha.

        Returns
        -------
        MultipleTestingResult
            Frozen model with rejected mask, n_rejected, method, alpha.
        """
        used_alpha = alpha if alpha is not None else self._config.alpha
        raw = self._inner.e_holm(alpha=alpha)
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
