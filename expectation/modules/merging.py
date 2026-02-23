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

class UStatisticMerger(EValueMerger):
    """
    U-statistic merging of order n: F(e) = U_n(e_1, ..., e_K).

    U_n = (1 / C(K, n)) * sum_{|A|=n} prod_{k in A} e_k

    Computed via elementary symmetric polynomials (ESP) in O(K*n) time.
    Special cases: U_0 = 1, U_1 = arithmetic mean, U_K = product.

    References
    ----------
    Vovk & Wang (2024) Section 4 Eq. (13); Ramdas & Wang (2025) Definition 8.9.
    """

    def __init__(self, n: int, K: int):
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if n > K:
            raise ValueError(f"n must be <= K, got n={n}, K={K}")
        self.n = n
        self.K = K

    def merge(self, e_values: NDArray) -> MergingResult:
        e_values = np.asarray(e_values, dtype=np.float64)
        if len(e_values) == 0:
            raise ValueError("e_values must be non-empty")
        if self.n > len(e_values):
            raise ValueError(
                f"n={self.n} exceeds number of e-values K={len(e_values)}"
            )
        is_valid = self._validate(e_values)
        merged = self._compute_u_statistic(e_values, self.n)
        log_merged = float(np.log(merged)) if merged > 0 else -np.inf
        return MergingResult(
            merged_e_value=merged,
            log_merged_e_value=log_merged,
            K=len(e_values),
            merging_function=MergingFunction.U_STATISTIC,
            is_valid=is_valid,
        )

    @staticmethod
    def _compute_u_statistic(e_values: NDArray, n: int) -> float:
        """
        Compute U_n via elementary symmetric polynomials.

        Recurrence: p_j(e_1,...,e_k) = p_j(e_1,...,e_{k-1}) + e_k * p_{j-1}(...)
        Result: U_n = p_n / C(K, n)

        The inner update ``p[1:n+1] += e * p[0:n]`` produces the same result
        as the scalar backward traversal because ``e * p[0:n]`` allocates a
        temporary that snapshots all old values before ``+=`` writes any.
        Both ensure each p[j] receives e * old_p[j-1].

        Time O(K*n), space O(n).
        """
        K = len(e_values)
        if n == 0:
            return 1.0
        if n == K:
            return float(np.prod(e_values))

        p = np.zeros(n + 1)
        p[0] = 1.0

        for e in e_values:
            p[1:n + 1] += e * p[0:n]

        return float(p[n] / _comb(K, n))

    def gambling_system(self, past_e_values: List[float], k: int) -> float:
        # Derived from the ESP representation. After observing k values,
        # the running merged value is S_k = U_n(e_0,...,e_{k-1}, 1^{K-k}).
        # The martingale decomposition gives:
        #   S_{k+1} = S_k * (1 + s_{k+1} * (e_k - 1))
        # where s_{k+1} = B / (A + B) with:
        #   q[j] = ESP_j(e_0, ..., e_{k-1})   (ESP from past values)
        #   m = K - k - 1                      (remaining ones after e_k)
        #   A = sum_i C(m, i) * q[n-i]
        #   B = sum_i C(m, i) * q[n-1-i]
        # This is purely F_k-measurable (uses only past values).
        # Reference: V&W (2024) Eq.(13), derived from ESP recurrence.

        n = self.n

        # Compute ESP q[0..n] from past values (vectorized inner loop)
        q = np.zeros(n + 1)
        q[0] = 1.0
        for e in past_e_values[:k]:
            q[1:n + 1] += e * q[0:n]

        m = self.K - k - 1  # remaining ones after the next e-value

        # Build binomial coefficient vector once: C(m, 0), C(m, 1), ...
        # exact=False returns float64 — exact for integer inputs up to 2^53
        len_a = min(n, m) + 1
        i_a = np.arange(len_a)
        coeffs_a = _sp_comb(m, i_a)

        # A = sum_i C(m, i) * q[n - i]  (dot product)
        A = float(np.dot(coeffs_a, q[n - i_a]))

        # B = sum_i C(m, i) * q[n - 1 - i]  (dot product, requires n >= 1)
        B = 0.0
        if n >= 1:
            len_b = min(n - 1, m) + 1
            # coeffs_b is a prefix of coeffs_a
            i_b = np.arange(len_b)
            B = float(np.dot(coeffs_a[:len_b], q[n - 1 - i_b]))

        denom = A + B
        if denom <= 0:
            return 0.0
        return float(np.clip(B / denom, 0.0, 1.0))

    def reset(self) -> None:
        pass

class LambdaProductMerger(EValueMerger):
    """
    Lambda-product merging: F(e) = prod_{k=1}^K (1 - lambda + lambda * e_k).

    Interpolates between no betting (lambda -> 0, F -> 1) and full product
    (lambda = 1, F = prod e_k). Computed in log-space for stability.

    Gambling system: constant s_k = lambda for all k.

    References
    ----------
    Vovk & Wang (2024) Section 4 p.10; Ramdas & Wang (2025) Definition 8.5.
    """

    def __init__(self, lambda_param: float = 0.5):
        if not (0 < lambda_param <= 1):
            raise ValueError(
                f"lambda_param must be in (0, 1], got {lambda_param}"
            )
        self.lambda_param = lambda_param

    def merge(self, e_values: NDArray) -> MergingResult:
        e_values = np.asarray(e_values, dtype=np.float64)
        if len(e_values) == 0:
            raise ValueError("e_values must be non-empty")
        is_valid = self._validate(e_values)

        # Compute in log-space: sum log(1 - lambda + lambda * e_k)
        terms = (1.0 - self.lambda_param) + self.lambda_param * e_values
        # Handle any non-positive terms
        if np.any(terms <= 0):
            log_merged = -np.inf
            merged = 0.0
        else:
            log_merged = float(np.sum(np.log(terms)))
            merged = float(np.exp(log_merged))

        return MergingResult(
            merged_e_value=merged,
            log_merged_e_value=log_merged,
            K=len(e_values),
            merging_function=MergingFunction.LAMBDA_PRODUCT,
            is_valid=is_valid,
        )

    def gambling_system(self, past_e_values: List[float], k: int) -> float:
        return self.lambda_param

    def reset(self) -> None:
        pass

class SegmentProductMerger(EValueMerger):
    """
    Segment-product merging: partition e-values into segments, average
    within each segment, then multiply the segment averages.

    F(e) = prod_{i=1}^{m} mean(e_{a_i}, ..., e_{b_i})

    Interpolates between arithmetic mean (single segment) and product
    (K singletons).

    Gambling system: within each segment, the gambling fraction follows the
    arithmetic-mean pattern for that segment's size.

    References
    ----------
    Vovk & Wang (2024) Section 4 p.10; Ramdas & Wang (2025) Definition 8.10.
    """

    def __init__(self, segments: List[int], K: int):
        """
        Parameters
        ----------
        segments : list of int
            Indices where new segments start (0-based). E.g., [3, 7] means
            three segments: [0,3), [3,7), [7,K).
        K : int
            Total number of e-values.
        """
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if not segments:
            raise ValueError("segments must be non-empty")
        for i, s in enumerate(segments):
            if s < 1:
                raise ValueError(f"segment boundaries must be >= 1, got {s}")
            if i > 0 and s <= segments[i - 1]:
                raise ValueError("segment boundaries must be strictly increasing")
        if segments[-1] >= K:
            raise ValueError(
                f"last segment boundary {segments[-1]} must be < K={K}"
            )
        self.segments = segments
        self.K = K

    def _get_segment_ranges(self, K: int) -> List[tuple]:
        boundaries = [0] + list(self.segments) + [K]
        return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    def merge(self, e_values: NDArray) -> MergingResult:
        e_values = np.asarray(e_values, dtype=np.float64)
        if len(e_values) == 0:
            raise ValueError("e_values must be non-empty")
        is_valid = self._validate(e_values)
        K = len(e_values)

        ranges = self._get_segment_ranges(K)
        log_merged = 0.0
        for start, end in ranges:
            seg_mean = float(np.mean(e_values[start:end]))
            if seg_mean <= 0:
                log_merged = -np.inf
                break
            log_merged += np.log(seg_mean)

        merged = float(np.exp(log_merged)) if log_merged > -np.inf else 0.0

        return MergingResult(
            merged_e_value=merged,
            log_merged_e_value=log_merged,
            K=K,
            merging_function=MergingFunction.SEGMENT_PRODUCT,
            is_valid=is_valid,
        )

    def gambling_system(self, past_e_values: List[float], k: int) -> float:
        # Within each segment, the gambling fraction follows the
        # arithmetic-mean pattern for that segment's size.
        # s_k = 1 / (seg_size * S_k^{seg}) where S_k^{seg} is the running
        # average within the current segment.
        ranges = self._get_segment_ranges(self.K)

        # Find which segment k belongs to
        for start, end in ranges:
            if k < end:
                seg_size = end - start
                # Position within this segment
                pos_in_seg = k - start
                if pos_in_seg == 0:
                    return 1.0 / seg_size

                seg_values = past_e_values[start:k]
                running_sum = sum(seg_values)
                remaining = seg_size - pos_in_seg
                s_seg = (running_sum + remaining) / seg_size
                if s_seg <= 0:
                    return 0.0
                return min(1.0, 1.0 / (seg_size * s_seg))

        return 0.0

    def reset(self) -> None:
        pass

class ProductMerger(EValueMerger):
    """
    Product merging: F(e) = e_1 * e_2 * ... * e_K.

    The most aggressive admissible merging function. Equivalent to
    all-in betting (s_k = 1 for all k). Highest variance among
    precise se-merging functions (Proposition 2 / R&W Prop 8.16).

    References
    ----------
    Vovk & Wang (2024) Section 4 Eq. (12); Ramdas & Wang (2025) Theorem 8.4.
    """

    def merge(self, e_values: NDArray) -> MergingResult:
        e_values = np.asarray(e_values, dtype=np.float64)
        if len(e_values) == 0:
            raise ValueError("e_values must be non-empty")
        is_valid = self._validate(e_values)

        # Compute in log-space for stability
        if np.any(e_values == 0):
            log_merged = -np.inf
            merged = 0.0
        else:
            log_merged = float(np.sum(np.log(e_values)))
            merged = float(np.exp(log_merged))

        return MergingResult(
            merged_e_value=merged,
            log_merged_e_value=log_merged,
            K=len(e_values),
            merging_function=MergingFunction.PRODUCT,
            is_valid=is_valid,
        )

    def gambling_system(self, past_e_values: List[float], k: int) -> float:
        return 1.0

    def reset(self) -> None:
        pass


def create_merger(config: MergingConfig) -> EValueMerger:
    """
    Create an EValueMerger from a MergingConfig.

    Parameters
    ----------
    config : MergingConfig
        Configuration specifying which merging function and its parameters.

    Returns
    -------
    EValueMerger
    """
    func = config.merging_function

    if func == MergingFunction.ARITHMETIC_MEAN:
        if config.K is None:
            raise ValueError("K is required for ArithmeticMeanMerger")
        return ArithmeticMeanMerger(K=config.K)

    elif func == MergingFunction.U_STATISTIC:
        if config.K is None:
            raise ValueError("K is required for UStatisticMerger")
        return UStatisticMerger(n=config.u_order, K=config.K)

    elif func == MergingFunction.LAMBDA_PRODUCT:
        return LambdaProductMerger(lambda_param=config.lambda_param)

    elif func == MergingFunction.SEGMENT_PRODUCT:
        if config.K is None:
            raise ValueError("K is required for SegmentProductMerger")
        if config.segments is None:
            raise ValueError("segments is required for SegmentProductMerger")
        return SegmentProductMerger(segments=list(config.segments), K=config.K)

    elif func == MergingFunction.PRODUCT:
        return ProductMerger()

    else:
        raise ValueError(f"Unknown merging function: {func}")



def arithmetic_mean_merge(e_values: NDArray) -> float:
    """
    Merge e-values by arithmetic mean.

    Parameters
    ----------
    e_values : NDArray
        Array of e-values.

    Returns
    -------
    float
        The merged e-value (arithmetic mean).
    """
    e_values = np.asarray(e_values, dtype=np.float64)
    return float(np.mean(e_values))


def u_statistic_merge(e_values: NDArray, n: int) -> float:
    """
    Merge e-values via the U-statistic of order n.

    Parameters
    ----------
    e_values : NDArray
        Array of K e-values.
    n : int
        Order of the U-statistic. U_0=1, U_1=mean, U_K=product.

    Returns
    -------
    float
        The U_n merged e-value.
    """
    e_values = np.asarray(e_values, dtype=np.float64)
    return UStatisticMerger._compute_u_statistic(e_values, n)


def lambda_product_merge(e_values: NDArray, lambda_param: float = 0.5) -> float:
    """
    Merge e-values via the lambda-product.

    Parameters
    ----------
    e_values : NDArray
        Array of e-values.
    lambda_param : float
        Hedging parameter in (0, 1]. Default 0.5.

    Returns
    -------
    float
        The lambda-product merged e-value.
    """
    merger = LambdaProductMerger(lambda_param=lambda_param)
    result = merger.merge(e_values)
    return result.merged_e_value


def segment_product_merge(e_values: NDArray, segments: List[int]) -> float:
    """
    Merge e-values via segment-product.

    Parameters
    ----------
    e_values : NDArray
        Array of K e-values.
    segments : list of int
        Indices where new segments start.

    Returns
    -------
    float
        The segment-product merged e-value.
    """
    e_values = np.asarray(e_values, dtype=np.float64)
    K = len(e_values)
    merger = SegmentProductMerger(segments=segments, K=K)
    result = merger.merge(e_values)
    return result.merged_e_value