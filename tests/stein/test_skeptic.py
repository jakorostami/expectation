# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""Stein Skeptic through the betting protocol: validity, power, predictability."""

import numpy as np
import pytest
from scipy import integrate

from expectation.modules.hypothesistesting import BettingStrategy, EProcessConfig
from expectation.modules.protocol import BettingProtocol
from expectation.stein.kernels import IMQSteinKernel, WendlandSteinKernel
from expectation.stein.skeptic import SteinSkeptic  # noqa: F401
from expectation.stein.skeptic import BoltzmannTarget, GaussianTarget


def _adaptive(alpha=0.05):
    # empirically adaptive combiner = exact GRAPA on the payoff g = E - 1 over lambda in [0, 1]
    return EProcessConfig(
        significance_level=alpha, betting_strategy=BettingStrategy.EMPIRICALLY_ADAPTIVE, gamma=1.0
    )


def _binomial_upper(alpha, n):
    return alpha * n + 3 * np.sqrt(alpha * (1 - alpha) * n)


def _double_well_target():
    def grad_u(x):
        return 4.0 * x * (x**2 - 1.0)

    def local_lip(y, r):
        m = abs(float(y[0])) + r
        return max(12.0 * m**2 - 4.0, 4.0)

    return BoltzmannTarget(grad_u, beta=1.0, dim=1, local_lipschitz_grad_potential=local_lip)


def _double_well_sampler(rng, size, shift=0.0):
    """Exact iid draws from p ∝ exp(-(x^2-1)^2) shifted by `shift`, by inverse CDF on a grid."""
    grid = np.linspace(-4, 4, 20_001)
    dens = np.exp(-((grid**2 - 1.0) ** 2))
    cdf = integrate.cumulative_trapezoid(dens, grid, initial=0.0)
    cdf /= cdf[-1]
    return np.interp(rng.random(size), cdf, grid) + shift


@pytest.mark.parametrize(
    "kernel",
    [IMQSteinKernel(dim=1), WendlandSteinKernel(dim=1, radius=2.0)],
    ids=["imq", "wendland"],
)
def test_type_one_error_time_uniform_under_null(kernel):
    rng = np.random.default_rng(0)
    target = GaussianTarget(mean=[0.0], cov=[[1.0]])
    n_reps, T, alpha = 100, 60, 0.1
    rejections, finals = 0, []
    for _ in range(n_reps):
        protocol = BettingProtocol(SteinSkeptic(target, kernel), _adaptive(alpha))
        rejected = False
        for x in rng.standard_normal(T):
            result = protocol.update(np.array([x]))
            rejected |= result.reject_null
        rejections += rejected
        finals.append(result.e_process_value)
    finals = np.array(finals)
    assert rejections <= _binomial_upper(alpha, n_reps)
    assert finals.mean() <= 1.0 + 3 * finals.std() / np.sqrt(n_reps)


@pytest.mark.parametrize(
    "kernel",
    [IMQSteinKernel(dim=1), WendlandSteinKernel(dim=1, radius=2.0)],
    ids=["imq", "wendland"],
)
def test_power_against_mean_shift(kernel):
    rng = np.random.default_rng(1)
    target = GaussianTarget(mean=[0.0], cov=[[1.0]])
    protocol = BettingProtocol(SteinSkeptic(target, kernel), _adaptive(0.05))
    for x in rng.standard_normal(400) + 1.0:
        result = protocol.update(np.array([x]))
        if result.reject_null:
            break
    assert result.reject_null
    assert protocol.stopping_time is not None and protocol.stopping_time < 400


def test_power_against_heavier_tails_imq():
    rng = np.random.default_rng(2)
    target = GaussianTarget(mean=[0.0], cov=[[1.0]])
    protocol = BettingProtocol(SteinSkeptic(target, IMQSteinKernel(dim=1)), _adaptive(0.05))
    for x in rng.standard_t(df=2.0, size=600):
        result = protocol.update(np.array([x]))
        if result.reject_null:
            break
    assert result.reject_null


def test_boltzmann_double_well_with_local_lipschitz_compact_kernel():
    """Score-only Boltzmann target whose score is not globally Lipschitz."""
    rng = np.random.default_rng(3)
    target = _double_well_target()
    kernel = WendlandSteinKernel(dim=1, radius=1.5)

    # null: exact draws from the double well -> expected capital <= 1
    finals, rejections = [], 0
    for _ in range(40):
        protocol = BettingProtocol(SteinSkeptic(target, kernel), _adaptive(0.1))
        rejected = False
        for x in _double_well_sampler(rng, 50):
            result = protocol.update(np.array([x]))
            rejected |= result.reject_null
        finals.append(result.e_process_value)
        rejections += rejected
    finals = np.array(finals)
    assert finals.mean() <= 1.0 + 3 * finals.std() / np.sqrt(len(finals))
    assert rejections <= _binomial_upper(0.1, 40)

    # alternative: the sampler is biased (shifted well) -> rejects
    protocol = BettingProtocol(SteinSkeptic(target, kernel), _adaptive(0.05))
    for x in _double_well_sampler(rng, 600, shift=0.6):
        result = protocol.update(np.array([x]))
        if result.reject_null:
            break
    assert result.reject_null


def test_imq_rejects_target_without_global_lipschitz():
    with pytest.raises(ValueError):
        SteinSkeptic(_double_well_target(), IMQSteinKernel(dim=1))


def test_payoff_is_predictable_and_bounded_below():
    rng = np.random.default_rng(4)
    target = GaussianTarget(mean=[0.0, 0.0])
    skeptic = SteinSkeptic(target, IMQSteinKernel(dim=2))
    assert skeptic.bet(np.zeros(2)) == 1.0  # no reference yet: no bet
    for x in rng.standard_normal((30, 2)) * 3:
        skeptic.observe(x)
    probe = rng.standard_normal((500, 2)) * 4
    e_values = np.array([skeptic.bet(x) for x in probe])
    assert np.all(e_values >= 0.0)
    # bet does not change state
    assert skeptic.n_reference == 30
    before = skeptic.bet(probe[0])
    skeptic.bet(probe[1])
    assert skeptic.bet(probe[0]) == before


def test_reference_cap_and_reset():
    rng = np.random.default_rng(5)
    target = GaussianTarget(mean=[0.0])
    skeptic = SteinSkeptic(target, IMQSteinKernel(dim=1), max_reference_size=7)
    for x in rng.standard_normal(20):
        skeptic.observe(np.array([x]))
    assert skeptic.n_reference == 7
    skeptic.reset()
    assert skeptic.n_reference == 0 and skeptic.bet(np.array([0.0])) == 1.0


def test_dimension_mismatch_and_invalid_observation():
    target = GaussianTarget(mean=[0.0, 0.0])
    with pytest.raises(ValueError):
        SteinSkeptic(target, IMQSteinKernel(dim=1))
    skeptic = SteinSkeptic(target, IMQSteinKernel(dim=2))
    with pytest.raises(ValueError):
        skeptic.observe(np.array([np.nan, 0.0]))
