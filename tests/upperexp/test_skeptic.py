# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""Sequential validity and power of upper-expectation Skeptics through the protocol."""

import numpy as np
import pytest

from expectation.modules.hypothesistesting import EProcessConfig
from expectation.modules.protocol import BettingProtocol
from expectation.upperexp.nulls import (
    CHSHLocalRealismNull,
    ContaminationNull,
    PolytopeNull,
    WassersteinNull,
)
from expectation.upperexp.skeptic import (
    UpperExpectationConfig,
    UpperExpectationSkeptic,
    WassersteinDistanceSequence,
)

ALPHA = 0.1
GRID = np.array([0.0, 1.0, 2.0, 3.0])
COST = np.abs(GRID[:, None] - GRID[None, :])
P0 = np.array([0.4, 0.3, 0.2, 0.1])


def _adversarial_run(null, n_steps, rng, config=None):
    """Reality picks the least favourable member of the set after seeing Skeptic's bet."""
    skeptic = UpperExpectationSkeptic(null, config)
    protocol = BettingProtocol(skeptic, EProcessConfig(significance_level=ALPHA))
    rejected = False
    for _ in range(n_steps):
        e = skeptic.e_variable()
        P = null.worst_case_member(e)
        x = rng.choice(len(P), p=P)
        result = protocol.update(x)
        rejected |= result.reject_null
    return result.e_process_value, rejected


def _binomial_upper(alpha, n):
    return alpha * n + 3 * np.sqrt(alpha * (1 - alpha) * n)


@pytest.mark.parametrize(
    "null",
    [
        ContaminationNull(P0, 0.1),
        WassersteinNull(P0, COST, radius=0.3),
        PolytopeNull(
            np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.1, 0.4, 0.4], [0.25, 0.25, 0.25, 0.25]])
        ),
    ],
    ids=["contamination", "wasserstein", "polytope"],
)
def test_supermartingale_under_adaptive_adversary(null):
    rng = np.random.default_rng(11)
    n_reps, n_steps = 80, 12
    finals, rejections = [], 0
    for _ in range(n_reps):
        final, rejected = _adversarial_run(null, n_steps, rng)
        finals.append(final)
        rejections += rejected
    finals = np.array(finals)
    # E[M_T] <= 1 up to Monte Carlo error; time-uniform rejection <= alpha
    assert finals.mean() <= 1.0 + 3 * finals.std() / np.sqrt(n_reps)
    assert rejections <= _binomial_upper(ALPHA, n_reps)


def test_power_outside_contamination_set():
    null = ContaminationNull(P0, 0.05)
    truth = np.array([0.05, 0.05, 0.3, 0.6])  # far outside the neighbourhood
    rng = np.random.default_rng(5)
    protocol = BettingProtocol(
        UpperExpectationSkeptic(null), EProcessConfig(significance_level=0.05)
    )
    result = None
    for _ in range(150):
        result = protocol.update(rng.choice(4, p=truth))
    assert result is not None and result.reject_null
    assert protocol.stopping_time is not None and protocol.stopping_time < 150


def test_wasserstein_skeptic_rejects_only_meaningful_deviation():
    rng = np.random.default_rng(7)
    truth = np.array([0.1, 0.2, 0.3, 0.4])
    w1 = WassersteinNull(P0, COST, 1.0).wasserstein_distance(truth)
    tolerant = WassersteinNull(P0, COST, radius=w1 * 1.2)  # truth inside: never rejects w.h.p.
    strict = WassersteinNull(P0, COST, radius=w1 * 0.3)  # truth outside: rejects
    outcomes = rng.choice(4, p=truth, size=250)

    p_tolerant = BettingProtocol(
        UpperExpectationSkeptic(tolerant), EProcessConfig(significance_level=0.05)
    )
    p_strict = BettingProtocol(
        UpperExpectationSkeptic(strict), EProcessConfig(significance_level=0.05)
    )
    for x in outcomes:
        r_tol = p_tolerant.update(x)
        r_strict = p_strict.update(x)
    assert r_strict.reject_null
    assert not r_tol.reject_null


def test_time_varying_announcement_bid_ask_spread():
    """Forecaster announces a fresh interval null every round (bid-ask on a binary event)."""
    rng = np.random.default_rng(3)
    skeptic = UpperExpectationSkeptic(config=UpperExpectationConfig(alternative=[0.3, 0.7]))
    protocol = BettingProtocol(skeptic, EProcessConfig(significance_level=0.05))
    log_capital = 0.0
    for t in range(200):
        bid = 0.55 + 0.05 * np.sin(t / 10.0)
        spread = PolytopeNull(np.array([[1 - bid, bid], [1 - bid - 0.05, bid + 0.05]]))
        x = rng.choice(2, p=[0.3, 0.7])  # truth: P(event) = 0.7, above the ask
        result = protocol.update(x, announcement=spread)
        log_capital = result.log_e_process
    assert result.reject_null
    assert log_capital > 0


def test_belief_inside_spread_never_bets():
    skeptic = UpperExpectationSkeptic(config=UpperExpectationConfig(alternative=[0.42, 0.58]))
    spread = PolytopeNull(np.array([[0.6, 0.4], [0.4, 0.6]]))
    assert skeptic.bet(1, announcement=spread) == pytest.approx(1.0)
    assert skeptic.diagnostics()["belief_in_null"] is True


def test_chsh_quantum_data_grows_at_kl_rate_and_local_data_does_not():
    null = CHSHLocalRealismNull()
    q = null.singlet_distribution()
    kl = null.reverse_information_projection(q).kl_divergence
    rng = np.random.default_rng(2)
    n_steps = 1500

    quantum = BettingProtocol(
        UpperExpectationSkeptic(null, UpperExpectationConfig(alternative=q.tolist())),
        EProcessConfig(significance_level=0.01),
    )
    for x in rng.choice(16, p=q, size=n_steps):
        result = quantum.update(x)
    rate = result.log_e_process / n_steps
    assert result.reject_null
    assert rate == pytest.approx(kl, rel=0.25)  # LLN: growth rate -> KL(q || p*)

    # a local-realist experiment with the same Skeptic: expected capital <= 1
    local = null.vertices[5]
    finals = []
    for _ in range(100):
        p = BettingProtocol(
            UpperExpectationSkeptic(null, UpperExpectationConfig(alternative=q.tolist())),
            EProcessConfig(significance_level=0.01),
        )
        for x in rng.choice(16, p=local, size=30):
            r = p.update(x)
        finals.append(r.e_process_value)
    finals = np.array(finals)
    assert finals.mean() <= 1.0 + 3 * finals.std() / np.sqrt(len(finals))


def test_plug_in_skeptic_detects_singlet_without_knowing_it():
    null = CHSHLocalRealismNull()
    q = null.singlet_distribution()
    rng = np.random.default_rng(9)
    protocol = BettingProtocol(
        UpperExpectationSkeptic(null), EProcessConfig(significance_level=0.05)
    )
    for x in rng.choice(16, p=q, size=3000):
        result = protocol.update(x)
    assert result.reject_null


class TestWassersteinDistanceSequence:
    radii = np.linspace(0.0, 1.4, 8)

    def test_lower_bound_coverage_is_time_uniform(self):
        truth = np.array([0.1, 0.2, 0.3, 0.4])
        rho_star = WassersteinNull(P0, COST, 1.0).wasserstein_distance(truth)
        rho_plus = self.radii[self.radii >= rho_star][0]
        rng = np.random.default_rng(21)
        n_reps, n_steps, alpha = 80, 30, 0.1
        violations = 0
        for _ in range(n_reps):
            seq = WassersteinDistanceSequence(P0, COST, self.radii, significance_level=alpha)
            violated = False
            for x in rng.choice(4, p=truth, size=n_steps):
                res = seq.update(x)
                violated |= res.lower_bound > rho_plus  # the true-null radius was rejected
            violations += violated
        assert violations <= _binomial_upper(alpha, n_reps)

    def test_lower_bound_rises_toward_true_distance(self):
        truth = np.array([0.1, 0.2, 0.3, 0.4])
        rho_star = WassersteinNull(P0, COST, 1.0).wasserstein_distance(truth)
        rng = np.random.default_rng(22)
        seq = WassersteinDistanceSequence(P0, COST, self.radii, significance_level=0.05)
        for x in rng.choice(4, p=truth, size=600):
            res = seq.update(x)
        assert res.lower_bound > 0.5 * rho_star
        assert res.lower_bound <= rho_star + (self.radii[1] - self.radii[0])
        assert res.wasserstein_estimate == pytest.approx(rho_star, abs=0.15)

    def test_reset_and_validation(self):
        seq = WassersteinDistanceSequence(P0, COST, self.radii)
        seq.update(0)
        seq.reset()
        assert seq.step == 0 and all(p.process_values == [1.0] for p in seq.e_processes)
        with pytest.raises(ValueError):
            WassersteinDistanceSequence(P0, COST, [0.2, 0.1])
        with pytest.raises(ValueError):
            seq.update(7)


def test_invalid_outcome_leaves_state_unchanged():
    null = ContaminationNull(P0, 0.1)
    skeptic = UpperExpectationSkeptic(null)
    protocol = BettingProtocol(skeptic)
    with pytest.raises(ValueError):
        protocol.update(9)
    assert skeptic.n_observations == 0 and protocol.step == 0


def test_reset_is_fresh():
    null = ContaminationNull(P0, 0.1)
    skeptic = UpperExpectationSkeptic(null)
    protocol = BettingProtocol(skeptic)
    for x in [0, 1, 2, 3, 3, 3]:
        protocol.update(x)
    protocol.reset()
    assert skeptic.n_observations == 0
    assert np.allclose(skeptic.belief(), 0.25)


def test_averaging_two_skeptics_is_a_valid_skeptic():
    """Averaging Skeptics' capitals (arithmetic-mean merge, valid under any dependence)
    composes with the protocol: E[(E1 + E2)/2 | F_{t-1}] <= 1 whenever both are e-values."""
    from expectation.modules.merging import ArithmeticMeanMerger
    from expectation.modules.protocol import SkepticStrategy

    class AveragedSkeptic(SkepticStrategy):
        def __init__(self, a, b):
            self.a, self.b = a, b
            self._merger = ArithmeticMeanMerger(2)

        @property
        def name(self):
            return "averaged"

        def bet(self, x, announcement=None):
            return self._merger.merge(
                np.array([self.a.bet(x, announcement), self.b.bet(x, announcement)])
            ).merged_e_value

        def observe(self, x, announcement=None):
            self.a.observe(x, announcement)
            self.b.observe(x, announcement)

        def reset(self):
            self.a.reset()
            self.b.reset()

    null = ContaminationNull(P0, 0.1)
    rng = np.random.default_rng(31)
    finals = []
    for _ in range(60):
        plug_in = UpperExpectationSkeptic(null)
        fixed = UpperExpectationSkeptic(
            null, UpperExpectationConfig(alternative=[0.1, 0.1, 0.3, 0.5])
        )
        protocol = BettingProtocol(
            AveragedSkeptic(plug_in, fixed), EProcessConfig(significance_level=ALPHA)
        )
        for _ in range(10):
            e_avg = protocol.strategy.bet(0)  # any outcome; adversary reacts to the vector
            e_vec = 0.5 * (plug_in.e_variable() + fixed.e_variable())
            P = null.worst_case_member(e_vec)
            result = protocol.update(rng.choice(4, p=P))
        finals.append(result.e_process_value)
    finals = np.array(finals)
    assert finals.mean() <= 1.0 + 3 * finals.std() / np.sqrt(len(finals))
