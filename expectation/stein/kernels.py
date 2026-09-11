# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""
Radial Stein kernels with certified pointwise lower bounds.

Based on these papers:

Sequential Kernelized Stein Discrepancy, D. Martinez-Taboada, A. Ramdas (2025), AISTATS,
PMLR 258 - https://arxiv.org/pdf/2409.17505
    - Eq. (1): the Stein kernel h_p(x, y)
    - Section 4: the pointwise bound M_p(y) >= -inf_x h_p(x, y) that makes the payoff >= -1
    - Section 5.1: the general bounding approach (terms (i)-(iii)); Section 5.2: IMQ examples

A Kernelized Stein Discrepancy for Goodness-of-fit Tests, Q. Liu, J. Lee, M. Jordan (2016), ICML
A Kernel Test of Goodness of Fit, K. Chwialkowski, H. Strathmann, A. Gretton (2016), ICML
    - the kernelized Stein discrepancy and the Stein identity E_p[h_p(X, y)] = 0
      (Chwialkowski et al. Lemma 5.1, as cited by Martinez-Taboada & Ramdas)

Measuring Sample Quality with Kernels, J. Gorham, L. Mackey (2017), ICML
    - the inverse multiquadric (IMQ) base kernel and its tail sensitivity

Piecewise polynomial, positive definite and compactly supported radial functions of minimal
degree, H. Wendland (1995), Advances in Computational Mathematics 4, 389-396
    - the C^2 Wendland functions phi_l(rho) = (1 - rho)_+^(l+1) ((l+1) rho + 1), l = floor(d/2)+2

For a target with score s = grad log p and a radial base kernel k(x, y) = psi(||x - y||),

    h_p(x, y) = <s(x), s(y)> psi(u) - (psi'(u) / u) <s(x) - s(y), x - y>
                - [ psi''(u) + (d - 1) psi'(u) / u ],        u = ||x - y||.

Martinez-Taboada & Ramdas derive M_p by hand per model (their Section 5.2).  Here M_p is
obtained generically from a Lipschitz constant L of the score, following the structure of
their Section 5.1 with Gamma(u) = L u:

    term (i)   >= -||s(y)||^2 sup psi - L ||s(y)|| sup_u u psi(u)
    term (ii)  >= -L sup_u u |psi'(u)|
    term (iii) >= inf_u -[psi''(u) + (d-1) psi'(u)/u]

so M_p(y) = ||s(y)||^2 K0 + L ||s(y)|| c_i + L c_ii + c_iii with closed-form kernel constants:

- IMQ, psi(u) = (1 + u^2)^(-1/2): K0 = 1, c_i = 1, c_ii = 2 / 3^(3/2),
  c_iii = 2 ((3 - d) / 5)^(5/2) for d < 3 else 0.  Needs a *global* Lipschitz constant.
- Wendland, psi(u) = phi_l(u / r): K0 = 1, c_i = r max_rho rho phi(rho),
  c_ii = 4 (l+1)/(l+2) (l/(l+2))^l, c_iii = (l+1)(l+2) ((l-1)/(l+d))^(l-1) / r^2.  Because
  h_p(., y) vanishes outside the ball of radius r, the Stein identity holds for every C^1
  positive density by the divergence theorem with no moment condition, and only a *local*
  Lipschitz constant on that ball is needed.  Trade-off: compact support forgoes the slow
  decay that makes IMQ sensitive to tail discrepancies (Gorham & Mackey 2017).
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class SteinKernel(ABC):
    """
    Radial base kernel psi(||x - y||) with its Stein kernel and certified lower bound.

    Subclasses set ``support_radius`` (inf for global support) and
    ``requires_global_lipschitz`` in ``__init__`` and implement the radial profile and
    the four bound constants (K0, c_i, c_ii, c_iii).
    """

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError("dim must be a positive integer")
        self.dim = int(dim)
        self.support_radius: float = np.inf
        self.requires_global_lipschitz: bool = True

    @abstractmethod
    def psi(self, u: NDArray) -> NDArray:
        """psi(u)."""
        pass

    @abstractmethod
    def dpsi_over_u(self, u: NDArray) -> NDArray:
        """psi'(u) / u (finite at u = 0)."""
        pass

    @abstractmethod
    def d2psi(self, u: NDArray) -> NDArray:
        """psi''(u)."""
        pass

    @abstractmethod
    def bound_constants(self) -> tuple[float, float, float, float]:
        """(K0, c_i, c_ii, c_iii): sup psi, sup u psi, sup u |psi'|, -inf of term (iii)."""
        pass

    def stein_kernel(self, x: NDArray, Y: NDArray, s_x: NDArray, s_Y: NDArray) -> NDArray:
        """h_p(x, Y_j) for one point x and reference points Y, shape (m,)."""
        x = np.asarray(x, dtype=np.float64).reshape(1, self.dim)
        Y = np.asarray(Y, dtype=np.float64).reshape(-1, self.dim)
        s_x = np.asarray(s_x, dtype=np.float64).reshape(1, self.dim)
        s_Y = np.asarray(s_Y, dtype=np.float64).reshape(-1, self.dim)
        diff = x - Y
        u = np.sqrt(np.sum(diff**2, axis=1))
        term_i = np.einsum("md,d->m", s_Y, s_x[0]) * self.psi(u)
        term_ii = -self.dpsi_over_u(u) * np.sum((s_x - s_Y) * diff, axis=1)
        term_iii = -(self.d2psi(u) + (self.dim - 1) * self.dpsi_over_u(u))
        return np.asarray(term_i + term_ii + term_iii, dtype=np.float64)

    def lower_bound(self, score_norm_y: NDArray, lipschitz: NDArray) -> NDArray:
        """M_p(y) >= -inf_x h_p(x, y) from ||s(y)|| and a (per-point or scalar) Lipschitz constant."""
        n = np.asarray(score_norm_y, dtype=np.float64)
        L = np.broadcast_to(np.asarray(lipschitz, dtype=np.float64), n.shape)
        if np.any(L < 0) or not np.all(np.isfinite(L)):
            raise ValueError("Lipschitz constants must be finite and nonnegative")
        k0, c_i, c_ii, c_iii = self.bound_constants()
        return n**2 * k0 + L * n * c_i + L * c_ii + c_iii


class IMQSteinKernel(SteinKernel):
    """Inverse multiquadric base kernel (1 + ||x - y||^2)^(-1/2) (Gorham & Mackey 2017)."""

    def psi(self, u: NDArray) -> NDArray:
        return (1.0 + u**2) ** (-0.5)

    def dpsi_over_u(self, u: NDArray) -> NDArray:
        return -((1.0 + u**2) ** (-1.5))

    def d2psi(self, u: NDArray) -> NDArray:
        return -((1.0 + u**2) ** (-1.5)) + 3.0 * u**2 * (1.0 + u**2) ** (-2.5)

    def bound_constants(self) -> tuple[float, float, float, float]:
        # term (iii) = (1 + u^2)^(-5/2) [d + (d - 3) u^2]; its minimum for d < 3 is at
        # u^2 = (d + 2) / (3 - d) and equals -2 ((3 - d) / 5)^(5/2); it is >= 0 for d >= 3.
        c_iii = 2.0 * ((3.0 - self.dim) / 5.0) ** 2.5 if self.dim < 3 else 0.0
        return 1.0, 1.0, float(2.0 / 3.0**1.5), float(c_iii)


class WendlandSteinKernel(SteinKernel):
    """Compactly supported Wendland C^2 base kernel phi_l(||x - y|| / r), l = floor(d/2) + 2."""

    def __init__(self, dim: int, radius: float):
        super().__init__(dim)
        if not radius > 0:
            raise ValueError("radius must be positive")
        self.radius = float(radius)
        self.order = self.dim // 2 + 2
        self.support_radius = self.radius
        self.requires_global_lipschitz = False
        self._max_rho_phi = _max_rho_phi(self.order)

    def _rho(self, u: NDArray) -> NDArray:
        return np.asarray(u, dtype=np.float64) / self.radius

    def phi(self, rho: NDArray) -> NDArray:
        l = self.order
        return np.where(rho < 1.0, (1.0 - rho).clip(0.0) ** (l + 1) * ((l + 1) * rho + 1.0), 0.0)

    def dphi_over_rho(self, rho: NDArray) -> NDArray:
        l = self.order
        return np.where(rho < 1.0, -(l + 1) * (l + 2) * (1.0 - rho).clip(0.0) ** l, 0.0)

    def d2phi(self, rho: NDArray) -> NDArray:
        l = self.order
        return np.where(
            rho < 1.0,
            -(l + 1) * (l + 2) * (1.0 - rho).clip(0.0) ** (l - 1) * (1.0 - (l + 1) * rho),
            0.0,
        )

    def psi(self, u: NDArray) -> NDArray:
        return self.phi(self._rho(u))

    def dpsi_over_u(self, u: NDArray) -> NDArray:
        return self.dphi_over_rho(self._rho(u)) / self.radius**2

    def d2psi(self, u: NDArray) -> NDArray:
        return self.d2phi(self._rho(u)) / self.radius**2

    def bound_constants(self) -> tuple[float, float, float, float]:
        l, d, r = self.order, self.dim, self.radius
        c_i = r * self._max_rho_phi
        c_ii = 4.0 * (l + 1) / (l + 2) * (l / (l + 2)) ** l
        c_iii = (l + 1) * (l + 2) * ((l - 1) / (l + d)) ** (l - 1) / r**2
        return 1.0, float(c_i), float(c_ii), float(c_iii)


def _max_rho_phi(order: int) -> float:
    """max over [0, 1] of rho (1 - rho)^(l+1) ((l+1) rho + 1) via the roots of its derivative."""
    one_minus = np.polynomial.Polynomial([1.0, -1.0])
    rho = np.polynomial.Polynomial([0.0, 1.0])
    g = rho * one_minus ** (order + 1) * ((order + 1) * rho + 1)
    roots = g.deriv().roots()
    candidates = np.concatenate([[0.0, 1.0], roots[np.isreal(roots)].real])
    candidates = candidates[(candidates >= 0) & (candidates <= 1)]
    return float(np.max(g(candidates)))
