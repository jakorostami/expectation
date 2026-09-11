# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""
Stein Skeptic: sequential goodness of fit for a Forecaster who announces only a score.

Based on these papers:

Sequential Kernelized Stein Discrepancy, D. Martinez-Taboada, A. Ramdas (2025), AISTATS,
PMLR 258 - https://arxiv.org/pdf/2409.17505
    - Eq. (5): normalised payoff g_t(x) = sum_{i<t} h_p(X_i, x) / sum_{i<t} M_p(X_i) >= -1
    - Theorem 1: K_t = prod (1 + lambda_t g_t(X_t)), lambda_t in [0, 1] predictable, is a
      test martingale under H0 : X_t | F_{t-1} ~ p
    - Theorem 2: exponential growth under the alternative (LBOW strategy)
    - Definition 1 (aGRAPA) and Definition 2 (LBOW): the betting strategies they use
    - Section 1, footnote 1: MCMC autocorrelation and burn-in caveats

Hypothesis testing with e-values, A. Ramdas, R. Wang (2025) - https://arxiv.org/pdf/2410.23614
    - Definition 7.21(iii): the empirically adaptive e-process (the library's
      ``EmpiricallyAdaptiveCombiner``), which bets on E_t - 1 = g_t exactly as aGRAPA-type
      rules do

Skeptic emits E_t = 1 + g_t(X_t) in [0, inf), a sequential e-value with conditional mean
one under H0, so the library's combiners bet (1 - lambda) + lambda E_t = 1 + lambda g_t.
The target is a ``ScoreTarget`` (score function plus Lipschitz information); the bound
M_p comes from ``kernels.py`` in closed form.  Null caveat: the hypothesis is about the
conditional law given the past; autocorrelated MCMC output violates it (thin, or use one
draw per independent replica).
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from expectation.modules.protocol import SkepticStrategy
from expectation.stein.kernels import SteinKernel


class ScoreTarget(ABC):
    """
    A target density known through its score s = grad log p and Lipschitz information.

    The natural interface for Boltzmann distributions p ∝ exp(-beta U) (s = -beta grad U)
    and energy-based models.  ``lipschitz`` is a global Lipschitz constant of s (needed by
    globally supported kernels); ``local_lipschitz(y, r)`` a constant valid on the ball of
    radius r around y (enough for compactly supported kernels, and finite for polynomial
    potentials whose score is not globally Lipschitz).
    """

    name: str = "score_target"

    def __init__(
        self,
        dim: int,
        lipschitz: Optional[float] = None,
        local_lipschitz: Optional[Callable[[NDArray, float], float]] = None,
    ):
        if dim < 1:
            raise ValueError("dim must be a positive integer")
        if lipschitz is None and local_lipschitz is None:
            raise ValueError(
                "Provide a global lipschitz constant and/or a local_lipschitz function"
            )
        if lipschitz is not None and (not np.isfinite(lipschitz) or lipschitz < 0):
            raise ValueError("lipschitz must be finite and nonnegative")
        self.dim = int(dim)
        self.lipschitz = None if lipschitz is None else float(lipschitz)
        self.local_lipschitz = local_lipschitz

    @abstractmethod
    def _score(self, x: NDArray) -> NDArray:
        """score(x) for x of shape (n, d), returning shape (n, d)."""
        pass

    def score(self, x: NDArray) -> NDArray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.dim)
        s = np.asarray(self._score(x), dtype=np.float64).reshape(x.shape[0], self.dim)
        if not np.all(np.isfinite(s)):
            raise ValueError("score returned non-finite values")
        return s

    def lipschitz_constants(self, Y: NDArray, radius: float) -> NDArray:
        """Per-point Lipschitz constants of the score valid on B(y, radius)."""
        Y = np.asarray(Y, dtype=np.float64).reshape(-1, self.dim)
        if np.isinf(radius) or self.local_lipschitz is None:
            if self.lipschitz is None:
                raise ValueError(
                    f"target {self.name!r} has no global Lipschitz constant; use "
                    "WendlandSteinKernel with a local_lipschitz function"
                )
            return np.full(Y.shape[0], self.lipschitz)
        out = np.array([float(self.local_lipschitz(y, radius)) for y in Y])
        if np.any(out < 0) or not np.all(np.isfinite(out)):
            raise ValueError("Lipschitz constants must be finite and nonnegative")
        return out


class CallableScoreTarget(ScoreTarget):
    """A target given by an arbitrary score callable."""

    def __init__(
        self,
        score: Callable[[NDArray], NDArray],
        dim: int,
        lipschitz: Optional[float] = None,
        local_lipschitz: Optional[Callable[[NDArray, float], float]] = None,
        name: str = "score_target",
    ):
        super().__init__(dim, lipschitz, local_lipschitz)
        self.score_fn = score
        self.name = name

    def _score(self, x: NDArray) -> NDArray:
        return np.asarray(self.score_fn(x), dtype=np.float64)


class GaussianTarget(ScoreTarget):
    """N(mean, cov): score -cov^{-1}(x - mean), Lipschitz constant ||cov^{-1}||_op."""

    name = "gaussian"

    def __init__(self, mean: NDArray, cov: Optional[NDArray] = None):
        self.mean = np.asarray(mean, dtype=np.float64).ravel()
        d = len(self.mean)
        self.cov = np.eye(d) if cov is None else np.asarray(cov, dtype=np.float64).reshape(d, d)
        precision = np.linalg.inv(self.cov)
        self.precision = 0.5 * (precision + precision.T)
        super().__init__(d, lipschitz=float(np.linalg.eigvalsh(self.precision).max()))

    def _score(self, x: NDArray) -> NDArray:
        return np.asarray(-np.einsum("nd,de->ne", x - self.mean, self.precision))


class BoltzmannTarget(ScoreTarget):
    """
    p ∝ exp(-beta U) given grad U and a global or local Lipschitz constant of grad U.

    Parameters
    ----------
    grad_potential : callable
        grad U(x) for x of shape (n, d).
    beta : float
        Inverse temperature.
    dim : int
    lipschitz_grad_potential : float, optional
        Global Lipschitz constant of grad U.
    local_lipschitz_grad_potential : callable, optional
        (y, r) -> Lipschitz constant of grad U on B(y, r).
    """

    name = "boltzmann"

    def __init__(
        self,
        grad_potential: Callable[[NDArray], NDArray],
        beta: float,
        dim: int,
        lipschitz_grad_potential: Optional[float] = None,
        local_lipschitz_grad_potential: Optional[Callable[[NDArray, float], float]] = None,
    ):
        if not beta > 0:
            raise ValueError("beta must be positive")
        self.beta = float(beta)
        self.grad_potential = grad_potential
        lipschitz = (
            None if lipschitz_grad_potential is None else self.beta * lipschitz_grad_potential
        )
        local = None
        if local_lipschitz_grad_potential is not None:
            local = lambda y, r: self.beta * float(
                local_lipschitz_grad_potential(y, r)
            )  # noqa: E731
        super().__init__(dim, lipschitz, local)

    def _score(self, x: NDArray) -> NDArray:
        return -self.beta * np.asarray(self.grad_potential(x), dtype=np.float64)


class SteinSkeptic(SkepticStrategy):
    """
    Kernel Stein betting against a score-only target (Martinez-Taboada & Ramdas 2025).

    Parameters
    ----------
    target : ScoreTarget
    kernel : SteinKernel
        Base kernel of matching dimension.
    max_reference_size : int, optional
        Keep only the first ``max_reference_size`` observations as reference points
        (a predictable rule) to bound the per-round cost.
    """

    name = "stein"

    def __init__(
        self, target: ScoreTarget, kernel: SteinKernel, max_reference_size: Optional[int] = None
    ):
        if target.dim != kernel.dim:
            raise ValueError(f"target dim {target.dim} != kernel dim {kernel.dim}")
        if kernel.requires_global_lipschitz and target.lipschitz is None:
            raise ValueError(
                "this kernel needs a global Lipschitz constant of the score; "
                "use WendlandSteinKernel with a local_lipschitz function instead"
            )
        if max_reference_size is not None and max_reference_size < 1:
            raise ValueError("max_reference_size must be >= 1")
        self.target = target
        self.kernel = kernel
        self.max_reference_size = max_reference_size
        self.reference_points: NDArray = np.empty((0, target.dim))
        self.reference_scores: NDArray = np.empty((0, target.dim))
        self.reference_bounds: NDArray = np.empty(0)
        self._last: dict[str, Any] = {}

    @property
    def n_reference(self) -> int:
        return self.reference_points.shape[0]

    def payoff(self, x: NDArray) -> float:
        """g_t(x) of Eq. (5), in [-1, inf); zero before any reference point exists."""
        if self.n_reference == 0:
            return 0.0
        x = np.asarray(x, dtype=np.float64).reshape(1, self.target.dim)
        s_x = self.target.score(x)
        h = self.kernel.stein_kernel(x[0], self.reference_points, s_x[0], self.reference_scores)
        return float(np.sum(h) / np.sum(self.reference_bounds))

    def bet(self, reality_move: Any, announcement: Any = None) -> float:
        g = self.payoff(reality_move)
        if g < -1.0 - 1e-9:
            raise RuntimeError(
                f"payoff {g} below -1: the certified lower bound was violated (check the "
                "Lipschitz constant supplied for the target)"
            )
        g = max(g, -1.0)
        self._last = {"payoff": g, "n_reference": self.n_reference}
        return 1.0 + g

    def observe(self, reality_move: Any, announcement: Any = None) -> None:
        x = np.asarray(reality_move, dtype=np.float64).reshape(1, self.target.dim)
        if not np.all(np.isfinite(x)):
            raise ValueError("observation must be finite")
        if self.max_reference_size is not None and self.n_reference >= self.max_reference_size:
            return
        s = self.target.score(x)
        L = self.target.lipschitz_constants(x, self.kernel.support_radius)
        M = self.kernel.lower_bound(np.linalg.norm(s, axis=1), L)
        self.reference_points = np.vstack([self.reference_points, x])
        self.reference_scores = np.vstack([self.reference_scores, s])
        self.reference_bounds = np.concatenate([self.reference_bounds, M])

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last)

    def reset(self) -> None:
        self.reference_points = np.empty((0, self.target.dim))
        self.reference_scores = np.empty((0, self.target.dim))
        self.reference_bounds = np.empty(0)
        self._last = {}
