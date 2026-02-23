"""
Admissible e-value adjusters for carefree multiple testing.

Implements two adjusters from Tavyrikov, Goeman & de Heide (2025),
"Carefree multiple testing with e-processes" (arXiv:2501.19360v2)

References
----------
Tavyrikov, Goeman & de Heide (2025). Carefree multiple testing with
    e-processes. arXiv:2501.19360v2.
Dawid, Ryter, Vovk, de Heide (2011a). Prequential probability.
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict


class AdjusterFunction(str, Enum):
    LOOKBACK = "lookback"
    SQRT = "sqrt"


class AdjusterConfig(BaseModel):
    adjuster: AdjusterFunction

    model_config = ConfigDict(frozen=True)


class Adjuster(ABC):
    """Abstract base class for admissible e-value adjusters.

    An admissible adjuster A: [1, inf) -> [0, inf) satisfies
    ``∫1^inf A(E) / E^2 dE = 1`` (calibration condition).

    Subclasses must implement ``adjust`` (natural scale) and
    ``log_adjust`` (log scale). Both accept scalars or arrays.
    """

    @abstractmethod
    def adjust(
        self, e: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Compute A(E) on the natural scale.

        Parameters
        ----------
        e : float or NDArray
            E-value(s). Must be >= 0.

        Returns
        -------
        float or NDArray
            Adjusted e-value(s). Returns 0 for E <= 1.
        """
        pass

    @abstractmethod
    def log_adjust(
        self, log_e: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        """Compute ln(A(exp(log_e))) in log space.

        Parameters
        ----------
        log_e : float or NDArray
            Log e-value(s).

        Returns
        -------
        float or NDArray
            Log adjusted e-value(s). Returns -inf for log_e <= 0.
        """
        pass


# Taylor threshold: for |x| < 1e-4, x^8/40320 < 1e-32
_TAYLOR_THRESHOLD_SQ = 1e-8

#. TODO: where to put?
# def _lookback_from_log_scalar(x: float) -> float:
#     """Core lookback computation for a single x = ln(E) > 0.

#     A_1 = (e^x - 1 - x) / x^2

#     Uses Taylor for small x: 1/2 + x/6 + x^2/24 + x^3/120
#     Uses expm1 for larger x to avoid cancellation.
#     """
#     x_sq = x * x
#     if x_sq < _TAYLOR_THRESHOLD_SQ:
#         return 0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0)))
#     else:
#         return (np.expm1(x) - x) / x_sq


def _lookback_from_log_array(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized lookback computation for array x = ln(E).

    Uses Taylor branch where x^2 < threshold, direct branch elsewhere.
    """
    result = np.empty_like(x)
    x_sq = x * x
    small = x_sq < _TAYLOR_THRESHOLD_SQ

    # Taylor branch
    xs = x[small]
    result[small] = 0.5 + xs * (1.0 / 6.0 + xs * (1.0 / 24.0 + xs * (1.0 / 120.0)))

    # Direct branch
    big = ~small
    xb = x[big]
    result[big] = (np.expm1(xb) - xb) / (xb * xb)

    return result


class LookbackAdjuster(Adjuster):
    """Lookback adjuster: A_1(E) = (E - 1 - ln E) / (ln E)^2.

    Also known as the Dawid et al. (2011a) adjuster. Has a removable
    singularity at E = 1 with limit A_1(1) = 1/2.

    Numerically stabilized via Taylor expansion for small ln(E) and
    ``expm1`` for larger values.

    Reference: Tavyrikov, Goeman & de Heide (2025), Eq. (5), first line.
    """

    def adjust(
        self, e: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        scalar = np.isscalar(e)
        e = np.atleast_1d(np.asarray(e, dtype=np.float64))
        result = np.zeros_like(e)
        mask = e > 1.0
        if np.any(mask):
            x = np.log(e[mask])
            result[mask] = _lookback_from_log_array(x)
        if scalar:
            return float(result[0])
        return result

    def log_adjust(
        self, log_e: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        scalar = np.isscalar(log_e)
        log_e = np.atleast_1d(np.asarray(log_e, dtype=np.float64))
        result = np.full_like(log_e, -np.inf)
        mask = log_e > 0.0
        if np.any(mask):
            vals = _lookback_from_log_array(log_e[mask])
            # vals should be > 0 for log_e > 0
            with np.errstate(divide="ignore"):
                result[mask] = np.where(vals > 0, np.log(vals), -np.inf)
        if scalar:
            return float(result[0])
        return result



class SqrtAdjuster(Adjuster):
    """Sqrt adjuster: A_2(E) = sqrt(E) - 1.

    Simple and computationally cheap. Uses ``expm1`` in log space
    for stability.

    Reference: Tavyrikov, Goeman & de Heide (2025), Eq. (5), second line.
    """

    def adjust(
        self, e: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        scalar = np.isscalar(e)
        e = np.atleast_1d(np.asarray(e, dtype=np.float64))
        result = np.zeros_like(e)
        mask = e > 1.0
        if np.any(mask):
            result[mask] = np.sqrt(e[mask]) - 1.0
        if scalar:
            return float(result[0])
        return result

    def log_adjust(
        self, log_e: Union[float, NDArray[np.float64]]
    ) -> Union[float, NDArray[np.float64]]:
        scalar = np.isscalar(log_e)
        log_e = np.atleast_1d(np.asarray(log_e, dtype=np.float64))
        result = np.full_like(log_e, -np.inf)
        mask = log_e > 0.0
        if np.any(mask):
            # sqrt(E) - 1 = exp(log_e/2) - 1 = expm1(log_e/2)
            vals = np.expm1(log_e[mask] / 2.0)
            with np.errstate(divide="ignore"):
                result[mask] = np.where(vals > 0, np.log(vals), -np.inf)
        if scalar:
            return float(result[0])
        return result


def create_adjuster(config: AdjusterConfig) -> Adjuster:
    """Create an Adjuster from an AdjusterConfig.

    Parameters
    ----------
    config : AdjusterConfig
        Configuration specifying which adjuster.

    Returns
    -------
    Adjuster
    """
    if config.adjuster == AdjusterFunction.LOOKBACK:
        return LookbackAdjuster()
    elif config.adjuster == AdjusterFunction.SQRT:
        return SqrtAdjuster()
    else:
        raise ValueError(f"Unknown adjuster: {config.adjuster}")


def lookback_adjust(
    e: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """Apply the lookback adjuster: A_1(E) = (E - 1 - ln E) / (ln E)^2.

    Parameters
    ----------
    e : float or NDArray
        E-value(s).

    Returns
    -------
    float or NDArray
        Adjusted e-value(s). Returns 0 for E <= 1.

    Reference: Tavyrikov, Goeman & de Heide (2025), Eq. (5).
    """
    return LookbackAdjuster().adjust(e)


def sqrt_adjust(
    e: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """Apply the sqrt adjuster: A_2(E) = sqrt(E) - 1.

    Parameters
    ----------
    e : float or NDArray
        E-value(s).

    Returns
    -------
    float or NDArray
        Adjusted e-value(s). Returns 0 for E <= 1.

    Reference: Tavyrikov, Goeman & de Heide (2025), Eq. (5).
    """
    return SqrtAdjuster().adjust(e)
