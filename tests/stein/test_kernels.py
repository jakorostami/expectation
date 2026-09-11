# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""Exact checks: kernel derivatives, Stein identity by quadrature, certified lower bounds."""

import numpy as np
import pytest
from scipy import integrate

from expectation.stein.kernels import IMQSteinKernel, SteinKernel, WendlandSteinKernel
from expectation.stein.skeptic import BoltzmannTarget, CallableScoreTarget, GaussianTarget

KERNELS_1D = [IMQSteinKernel(dim=1), WendlandSteinKernel(dim=1, radius=2.5)]


def _double_well(beta: float = 1.0) -> BoltzmannTarget:
    # U(x) = (x^2 - 1)^2 ; U'(x) = 4 x (x^2 - 1) ; U''(x) = 12 x^2 - 4  (not globally Lipschitz)
    def grad_u(x):
        return 4.0 * x * (x**2 - 1.0)

    def local_lip(y, r):
        m = abs(float(y[0])) + r
        return max(12.0 * m**2 - 4.0, 4.0)

    return BoltzmannTarget(grad_u, beta=beta, dim=1, local_lipschitz_grad_potential=local_lip)


def _double_well_density():
    beta = 1.0
    unnorm = lambda x: np.exp(-beta * (x**2 - 1.0) ** 2)  # noqa: E731
    Z, _ = integrate.quad(unnorm, -6, 6)
    return lambda x: unnorm(x) / Z


@pytest.mark.parametrize("kernel", KERNELS_1D, ids=["imq", "wendland"])
def test_radial_derivatives_match_finite_differences(kernel: SteinKernel):
    u = np.array([0.05, 0.3, 0.9, 1.7, 2.4])
    h = 1e-5
    dpsi_fd = (kernel.psi(u + h) - kernel.psi(u - h)) / (2 * h)
    d2psi_fd = (kernel.psi(u + h) - 2 * kernel.psi(u) + kernel.psi(u - h)) / h**2
    assert np.allclose(kernel.dpsi_over_u(u) * u, dpsi_fd, atol=1e-6)
    assert np.allclose(kernel.d2psi(u), d2psi_fd, atol=1e-4)
    # psi'(u)/u is continuous at zero
    assert np.isfinite(kernel.dpsi_over_u(np.array([0.0]))[0])


@pytest.mark.parametrize("kernel", KERNELS_1D, ids=["imq", "wendland"])
def test_stein_identity_gaussian_by_quadrature(kernel: SteinKernel):
    target = GaussianTarget(mean=[0.3], cov=[[1.5]])
    density = lambda x: np.exp(-((x - 0.3) ** 2) / 3.0) / np.sqrt(3.0 * np.pi)  # noqa: E731
    for y in (-1.0, 0.3, 2.2):
        s_y = target.score(np.array([[y]]))[0]

        def integrand(x):
            s_x = target.score(np.array([[x]]))[0]
            return kernel.stein_kernel(np.array([x]), np.array([[y]]), s_x, s_y)[0] * density(x)

        value, err = integrate.quad(integrand, -12, 12, limit=400)
        assert abs(value) < 1e-7 + 10 * err


@pytest.mark.parametrize("kernel", KERNELS_1D, ids=["imq", "wendland"])
def test_stein_identity_double_well_boltzmann_by_quadrature(kernel: SteinKernel):
    target = _double_well()
    density = _double_well_density()
    for y in (-1.2, 0.0, 0.8):
        s_y = target.score(np.array([[y]]))[0]

        def integrand(x):
            s_x = target.score(np.array([[x]]))[0]
            return kernel.stein_kernel(np.array([x]), np.array([[y]]), s_x, s_y)[0] * density(x)

        value, err = integrate.quad(integrand, -6, 6, limit=400)
        assert abs(value) < 1e-7 + 10 * err


def test_stein_identity_2d_gaussian_monte_carlo():
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.4], [0.4, 0.7]])
    target = GaussianTarget(mean=[0.0, 0.5], cov=cov)
    chol = np.linalg.cholesky(cov)
    X = target.mean + np.einsum("nd,ed->ne", rng.standard_normal((200_000, 2)), chol)
    S = target.score(X)
    for kernel in (IMQSteinKernel(dim=2), WendlandSteinKernel(dim=2, radius=2.0)):
        y = np.array([0.4, -0.2])
        s_y = target.score(y[None, :])[0]
        # h_p(X_i, y) for all samples: symmetric kernel, so evaluate with x = y and Y = X
        h = kernel.stein_kernel(y, X, s_y, S)
        se = h.std() / np.sqrt(len(h))
        assert abs(h.mean()) < 4 * se


class TestCertifiedLowerBounds:
    def test_kernel_constants_dominate_grid_maxima(self):
        u = np.linspace(1e-6, 12, 200_001)
        for kernel in (IMQSteinKernel(dim=1), IMQSteinKernel(dim=2), IMQSteinKernel(dim=5)):
            assert kernel.bound_constants()[0] >= kernel.psi(u).max() - 1e-12
            assert kernel.bound_constants()[1] >= (u * kernel.psi(u)).max() - 1e-12
            assert (
                kernel.bound_constants()[2] >= (u * np.abs(kernel.dpsi_over_u(u) * u)).max() - 1e-12
            )
            term_iii = -(kernel.d2psi(u) + (kernel.dim - 1) * kernel.dpsi_over_u(u))
            assert -kernel.bound_constants()[3] <= term_iii.min() + 1e-12
        for kernel in (
            WendlandSteinKernel(dim=1, radius=1.0),
            WendlandSteinKernel(dim=2, radius=0.7),
            WendlandSteinKernel(dim=3, radius=2.0),
            WendlandSteinKernel(dim=6, radius=1.3),
        ):
            uu = np.linspace(1e-6, kernel.radius, 200_001)
            assert kernel.bound_constants()[0] >= kernel.psi(uu).max() - 1e-12
            assert kernel.bound_constants()[1] >= (uu * kernel.psi(uu)).max() - 1e-9
            assert (
                kernel.bound_constants()[2]
                >= (uu * np.abs(kernel.dpsi_over_u(uu) * uu)).max() - 1e-9
            )
            term_iii = -(kernel.d2psi(uu) + (kernel.dim - 1) * kernel.dpsi_over_u(uu))
            assert -kernel.bound_constants()[3] <= term_iii.min() + 1e-9

    @pytest.mark.parametrize("kernel", KERNELS_1D, ids=["imq", "wendland"])
    def test_lower_bound_holds_for_gaussian_target(self, kernel: SteinKernel):
        target = GaussianTarget(mean=[0.0], cov=[[1.0]])
        xs = np.linspace(-15, 15, 60_001)
        S_x = target.score(xs[:, None])
        for y in (-3.0, -0.5, 0.0, 1.7, 4.0):
            s_y = target.score(np.array([[y]]))
            L = target.lipschitz_constants(np.array([[y]]), kernel.support_radius)
            M = kernel.lower_bound(np.linalg.norm(s_y, axis=1), L)[0]
            h = kernel.stein_kernel(np.array([y]), xs[:, None], s_y[0], S_x)
            assert h.min() >= -M - 1e-9

    def test_local_lipschitz_bound_holds_for_double_well_with_compact_kernel(self):
        kernel = WendlandSteinKernel(dim=1, radius=1.5)
        target = _double_well()
        xs = np.linspace(-6, 6, 120_001)
        S_x = target.score(xs[:, None])
        for y in (-2.0, -1.0, 0.0, 0.7, 2.3):
            s_y = target.score(np.array([[y]]))
            L = target.lipschitz_constants(np.array([[y]]), kernel.support_radius)
            M = kernel.lower_bound(np.linalg.norm(s_y, axis=1), L)[0]
            h = kernel.stein_kernel(np.array([y]), xs[:, None], s_y[0], S_x)
            assert h.min() >= -M - 1e-9

    def test_imq_requires_global_lipschitz(self):
        target = _double_well()
        with pytest.raises(ValueError):
            target.lipschitz_constants(np.array([[0.0]]), IMQSteinKernel(dim=1).support_radius)

    def test_imq_constant_matches_skssd_gaussian_example_but_tighter(self):
        # Martinez-Taboada & Ramdas (2025) use M(y) = |y|(1 + |y|) + 3 for N(0,1) with IMQ.
        kernel = IMQSteinKernel(dim=1)
        y = np.array([1.3])
        ours = kernel.lower_bound(y, 1.0)[0]
        theirs = y[0] * (1 + y[0]) + 3.0
        assert ours <= theirs
        assert ours == pytest.approx(y[0] ** 2 + y[0] + 2 / 3**1.5 + 2 * (0.4) ** 2.5)


def test_score_target_validation():
    with pytest.raises(ValueError):
        CallableScoreTarget(lambda x: x, dim=1)
    with pytest.raises(ValueError):
        CallableScoreTarget(lambda x: x, dim=1, lipschitz=-1.0)
    with pytest.raises(ValueError):
        BoltzmannTarget(lambda x: x, beta=0.0, dim=1, lipschitz_grad_potential=1.0)
    target = CallableScoreTarget(lambda x: np.full_like(x, np.nan), dim=1, lipschitz=1.0)
    with pytest.raises(ValueError):
        target.score(np.zeros((1, 1)))
