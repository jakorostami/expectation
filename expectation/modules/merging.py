"""
Merging sequential e-values via martingales.

Based on these papers:

Merging sequential e-values via martingales, V. Vovk, R. Wang (2024)
    - https://arxiv.org/pdf/2007.06382
    - Section 4: Theorem 1, Corollary 1, Eq. (12)-(13)

Hypothesis testing with e-values, A. Ramdas, R. Wang (2025)
    - Definitions 8.1, 8.5, 8.9, 8.10
    - Theorem 8.4, 8.12
    - Proposition 8.16

Key result: All admissible se-merging functions are martingale merging
functions (Corollary 1 of V&W 2024). Every martingale merging function
has the form:

    S_K(e) = prod_{k=1}^{K} (1 + s_k(e_{(k-1)})(e_k - 1))

where s_k is a gambling system with s_k in [0, 1] (Eq. 4).

This module implements batch merging (combine K e-values at once) and
exposes the gambling system for each named function, bridging to the
sequential e-process framework in martingales.py.
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from scipy.special import comb as _sp_comb
from typing import Optional, List
from numpy.typing import NDArray
from pydantic import BaseModel, Field, ConfigDict, model_validator


def _comb(n: int, k: int) -> int:
    """Exact integer binomial coefficient via scipy."""
    return int(_sp_comb(n, k, exact=True))


class MergingFunction(str, Enum):
    ARITHMETIC_MEAN = "arithmetic_mean"
    U_STATISTIC = "u_statistic"
    LAMBDA_PRODUCT = "lambda_product"
    SEGMENT_PRODUCT = "segment_product"
    PRODUCT = "product"


class MergingConfig(BaseModel):
    """
    Configuration for e-value merging.

    Parameters
    ----------
    merging_function : MergingFunction
        Which merging function to use.
    K : int, optional
        Total number of e-values (required for streaming; inferred in batch).
    lambda_param : float
        Hedging parameter for LAMBDA_PRODUCT, in (0, 1]. Default 0.5.
    u_order : int
        Order n for U_STATISTIC. U_0=1, U_1=mean, U_K=product. Default 1.
    segments : list of int, optional
        Segment boundaries for SEGMENT_PRODUCT. Each entry is the index
        where a new segment starts. Must be strictly increasing, all > 0
        and < K.
    """
    merging_function: MergingFunction
    K: Optional[int] = Field(default=None, ge=1)
    lambda_param: float = Field(default=0.5, gt=0, le=1)
    u_order: int = Field(default=1, ge=0)
    segments: Optional[List[int]] = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_segments(self) -> "MergingConfig":
        if self.segments is not None:
            if len(self.segments) == 0:
                raise ValueError("segments must be non-empty")
            for i in range(len(self.segments)):
                if self.segments[i] < 1:
                    raise ValueError("segment boundaries must be >= 1")
                if i > 0 and self.segments[i] <= self.segments[i - 1]:
                    raise ValueError("segment boundaries must be strictly increasing")
            if self.K is not None and self.segments[-1] >= self.K:
                raise ValueError("segment boundaries must be < K")
        return self


class MergingResult(BaseModel):
    merged_e_value: float
    log_merged_e_value: float
    K: int
    merging_function: MergingFunction
    is_valid: bool

    model_config = ConfigDict(frozen=True)


class EValueMerger(ABC):
    """
    Abstract base class for e-value merging functions.

    Provides both batch merging and the gambling system representation
    from Theorem 2 of Vovk & Wang (2024).
    """

    @abstractmethod
    def merge(self, e_values: NDArray) -> MergingResult:
        """
        Batch-merge K e-values into a single merged e-value.

        Parameters
        ----------
        e_values : NDArray
            Array of K e-values (each >= 0).

        Returns
        -------
        MergingResult
        """
        pass

    @abstractmethod
    def gambling_system(self, past_e_values: List[float], k: int) -> float:
        """
        Return the gambling fraction s_k in [0, 1] for step k+1.

        From Eq. (4) of Vovk & Wang (2024):
            S_K(e) = prod_{k=1}^{K} (1 + s_k * (e_k - 1))

        Parameters
        ----------
        past_e_values : list of float
            E-values seen so far: e_1, ..., e_{k}.
        k : int
            Current step index (0-based). s_0 uses no past values,
            s_1 uses e_1, etc.

        Returns
        -------
        float
            Gambling fraction in [0, 1].
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def _validate(self, e_values: NDArray) -> bool:
        return bool(np.all(e_values >= 0))
    

class ArithmeticMeanMerger(EValueMerger):
    """
    Arithmetic mean merging: F(e) = (e_1 + ... + e_K) / K.

    The most conservative admissible merging function. Under the null,
    E[F] = 1 exactly when each E[e_k] = 1.

    Gambling system: s_k = 1 / (K * S_k) where S_k is the running
    merged value after k steps.

    References
    ----------
    Vovk & Wang (2024) Section 4 p.9; Ramdas & Wang (2025) Proposition 8.3.
    """

    def __init__(self, K: int):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K

    def merge(self, e_values: NDArray) -> MergingResult:
        e_values = np.asarray(e_values, dtype=np.float64)
        if len(e_values) == 0:
            raise ValueError("e_values must be non-empty")
        is_valid = self._validate(e_values)
        merged = float(np.mean(e_values))
        log_merged = float(np.log(merged)) if merged > 0 else -np.inf
        return MergingResult(
            merged_e_value=merged,
            log_merged_e_value=log_merged,
            K=len(e_values),
            merging_function=MergingFunction.ARITHMETIC_MEAN,
            is_valid=is_valid,
        )

    def gambling_system(self, past_e_values: List[float], k: int) -> float:
        # s_k = 1 / (K * S_k) where S_k is the running merged value.
        # At step 0 (before any e-values), S_0 = 1, so s_0 = 1/K.
        # After k e-values, S_k = (sum(past) + (K - k)) / K.
        if k == 0:
            return 1.0 / self.K
        running_sum = sum(past_e_values[:k])
        # S_k via the martingale representation:
        # The running product S_k = prod_{j=1}^{k} (1 + s_j*(e_j - 1))
        # For arithmetic mean, S_k = (sum_{j=1}^k e_j + (K - k)) / K
        s_k = (running_sum + (self.K - k)) / self.K
        if s_k <= 0:
            return 0.0
        return min(1.0, 1.0 / (self.K * s_k))

    def reset(self) -> None:
        pass