# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""Exact checks for upper-expectation nulls on finite outcome spaces."""

import itertools

import numpy as np
import pytest
from scipy import optimize

from expectation.upperexp.nulls import (
    CHSHLocalRealismNull,
    ContaminationNull,
    PolytopeNull,
    UpperExpectationNull,
    WassersteinNull,
)


def _growth(q, e):
    mask = q > 0
    return float(np.sum(q[mask] * np.log(e[mask])))


def _generic_gro(null: UpperExpectationNull, q: np.ndarray) -> float:
    """Brute-force GRO value: maximise sum q log e s.t. support_function(e) <= 1."""
    K = len(q)

    def objective(z):
        return -_growth(q, np.exp(z))

    def constraint(z):
        return 1.0 - null.support_function(np.exp(z))

    res = optimize.minimize(
        objective,
        x0=np.zeros(K),
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": constraint}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    assert res.success, res.message
    return -res.fun


class TestContaminationNull:
    p0 = np.array([0.4, 0.3, 0.2, 0.1])
    eps = 0.1

    def test_support_function_matches_point_mass_adversary(self):
        null = ContaminationNull(self.p0, self.eps)
        rng = np.random.default_rng(0)
        for _ in range(20):
            e = rng.uniform(0, 3, size=4)
            brute = max(
                (1 - self.eps) * self.p0 @ e + self.eps * e[j] for j in range(4)
            )  # H = point mass at outcome j is the worst case by linearity
            assert null.support_function(e) == pytest.approx(brute, abs=1e-12)

    def test_certified_e_variable_is_valid_for_every_member(self):
        null = ContaminationNull(self.p0, self.eps)
        e = null.certify(np.array([5.0, 0.1, 2.0, 0.7]))
        assert null.support_function(e) <= 1.0 + 1e-12
        # every contaminant H on the simplex: sample many, exact bound must hold
        rng = np.random.default_rng(1)
        for _ in range(200):
            H = rng.dirichlet(np.ones(4))
            Q = (1 - self.eps) * self.p0 + self.eps * H
            assert Q @ e <= 1.0 + 1e-12

    def test_optimal_e_variable_recovers_huber_strassen_clipping(self):
        null = ContaminationNull(self.p0, self.eps)
        q = np.array([0.1, 0.1, 0.3, 0.5])
        e = null.optimal_e_variable(q)
        assert null.support_function(e) <= 1.0 + 1e-10
        # Clipped likelihood-ratio form: e = min(r / kappa, m) with r = q / p0.
        r = q / self.p0
        m = e.max()
        unclipped = e < m - 1e-9
        assert unclipped.any()
        kappas = r[unclipped] / e[unclipped]
        assert np.allclose(kappas, kappas[0], rtol=1e-6)
        clipped = ~unclipped
        assert np.all(r[clipped] / kappas[0] >= m - 1e-6)

    def test_optimal_growth_matches_generic_convex_programme(self):
        null = ContaminationNull(self.p0, self.eps)
        q = np.array([0.1, 0.1, 0.3, 0.5])
        e = null.optimal_e_variable(q)
        assert _growth(q, e) == pytest.approx(_generic_gro(null, q), abs=1e-6)

    def test_belief_inside_set_gives_trivial_e_variable(self):
        null = ContaminationNull(self.p0, self.eps)
        H = np.array([0.0, 0.0, 0.5, 0.5])
        q = (1 - self.eps) * self.p0 + self.eps * H
        assert null.contains(q)
        assert np.allclose(null.optimal_e_variable(q), 1.0)

    def test_validation(self):
        with pytest.raises(ValueError):
            ContaminationNull(np.array([0.5, 0.5, 0.0]), 0.1)  # p0 must be positive
        with pytest.raises(ValueError):
            ContaminationNull(self.p0, 1.0)
        with pytest.raises(ValueError):
            ContaminationNull(np.array([0.6, 0.6]), 0.1)


class TestWassersteinNull:
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    cost = np.abs(grid[:, None] - grid[None, :])
    p0 = np.array([0.25, 0.25, 0.25, 0.25])

    def test_two_point_support_function_closed_form(self):
        cost = np.array([[0.0, 2.0], [2.0, 0.0]])
        p0 = np.array([0.7, 0.3])
        rho = 0.4  # may move 0.2 mass between the two points
        null = WassersteinNull(p0, cost, rho)
        e = np.array([1.0, 3.0])
        # move mass toward the larger e: Q = (0.5, 0.5)
        assert null.support_function(e) == pytest.approx(0.5 * 1.0 + 0.5 * 3.0, abs=1e-9)
        e = np.array([3.0, 1.0])
        assert null.support_function(e) == pytest.approx(0.9 * 3.0 + 0.1 * 1.0, abs=1e-9)

    def test_distance_and_potential_agree_kantorovich_rubinstein(self):
        null = WassersteinNull(self.p0, self.cost, radius=0.3)
        q = np.array([0.1, 0.2, 0.3, 0.4])
        w1 = null.wasserstein_distance(q)
        # exact 1-D formula: integral of |F_q - F_p0| over the grid spacing
        cdf_gap = np.cumsum(q) - np.cumsum(self.p0)
        assert w1 == pytest.approx(np.sum(np.abs(cdf_gap[:-1]) * np.diff(self.grid)), abs=1e-9)
        h = null.kantorovich_potential(q)
        diffs = np.abs(h[:, None] - h[None, :])
        assert np.all(diffs <= self.cost + 1e-9)  # 1-Lipschitz w.r.t. the cost
        assert (q - self.p0) @ h == pytest.approx(w1, abs=1e-9)  # attains the distance

    def test_witness_e_variable_is_valid_on_the_whole_ball(self):
        null = WassersteinNull(self.p0, self.cost, radius=0.3)
        q = np.array([0.1, 0.2, 0.3, 0.4])
        e = null.optimal_e_variable(q)
        assert np.all(e >= 0)
        assert null.support_function(e) <= 1.0 + 1e-9

    def test_growth_positive_iff_belief_outside_ball(self):
        q = np.array([0.1, 0.2, 0.3, 0.4])
        w1 = WassersteinNull(self.p0, self.cost, 1.0).wasserstein_distance(q)
        inside = WassersteinNull(self.p0, self.cost, radius=w1 * 1.05)
        outside = WassersteinNull(self.p0, self.cost, radius=w1 * 0.5)
        assert inside.contains(q)
        assert not outside.contains(q)
        assert np.allclose(inside.optimal_e_variable(q), 1.0)
        e_out = outside.optimal_e_variable(q)
        assert _growth(q, e_out) > 1e-4

    def test_witness_growth_close_to_generic_gro_on_small_space(self):
        cost = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        p0 = np.array([0.5, 0.3, 0.2])
        null = WassersteinNull(p0, cost, radius=0.2)
        q = np.array([0.2, 0.2, 0.6])
        witness_growth = _growth(q, null.optimal_e_variable(q))
        gro = _generic_gro(null, q)
        assert witness_growth > 0
        # The potential-witness family attains the GRO value to solver precision
        # (SLSQP with a nested-LP constraint is the less accurate side here).
        assert witness_growth == pytest.approx(gro, abs=1e-4)

    def test_validation(self):
        with pytest.raises(ValueError):
            WassersteinNull(self.p0, self.cost[:3, :3], 0.1)
        with pytest.raises(ValueError):
            WassersteinNull(self.p0, -self.cost, 0.1)
        with pytest.raises(ValueError):
            WassersteinNull(self.p0, self.cost, -0.1)


class TestPolytopeNull:
    vertices = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )

    def test_support_function_is_vertex_maximum(self):
        null = PolytopeNull(self.vertices)
        e = np.array([1.0, 2.0, 0.5])
        assert null.support_function(e) == pytest.approx((self.vertices @ e).max())

    def test_contains_uses_convex_hull(self):
        null = PolytopeNull(self.vertices)
        w = np.array([0.2, 0.5, 0.3])
        assert null.contains(w @ self.vertices)
        assert not null.contains(np.array([0.0, 0.0, 1.0]))

    def test_ripr_matches_direct_kl_minimisation(self):
        null = PolytopeNull(self.vertices)
        q = np.array([0.05, 0.05, 0.9])
        ripr = null.reverse_information_projection(q)

        def kl(w):
            p = w @ self.vertices
            return float(np.sum(q * np.log(q / p)))

        res = optimize.minimize(
            kl,
            x0=np.ones(3) / 3,
            method="SLSQP",
            bounds=[(0, 1)] * 3,
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
            options={"ftol": 1e-14, "maxiter": 1000},
        )
        assert ripr.kl_divergence == pytest.approx(res.fun, abs=1e-7)
        assert ripr.converged

    def test_certified_numeraire_satisfies_every_vertex(self):
        null = PolytopeNull(self.vertices)
        q = np.array([0.05, 0.05, 0.9])
        e = null.optimal_e_variable(q)
        assert np.all(self.vertices @ e <= 1.0 + 1e-12)
        assert _growth(q, e) == pytest.approx(
            null.reverse_information_projection(q).kl_divergence, abs=1e-6
        )
        assert _growth(q, e) == pytest.approx(_generic_gro(null, q), abs=1e-5)

    def test_belief_in_hull_gives_trivial_e_variable(self):
        null = PolytopeNull(self.vertices)
        q = np.array([0.2, 0.5, 0.3]) @ self.vertices
        assert np.allclose(null.optimal_e_variable(q), 1.0, atol=1e-8)

    def test_validation(self):
        with pytest.raises(ValueError):
            PolytopeNull(np.array([[0.5, 0.6]]))
        with pytest.raises(ValueError):
            PolytopeNull(np.array([[1.0, 0.0], [1.0, 0.0]]))  # union support must be full


class TestCHSH:
    null = CHSHLocalRealismNull()

    def test_local_vertices_have_chsh_value_two(self):
        vertices = self.null.vertices
        assert vertices.shape == (16, 16)
        assert np.allclose(vertices.sum(axis=1), 1.0)
        for v in vertices:
            assert abs(CHSHLocalRealismNull.chsh_value(v)) == pytest.approx(2.0)

    def test_local_mixtures_respect_bell_bound(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            w = rng.dirichlet(np.ones(16))
            assert abs(self.null.chsh_value(w @ self.null.vertices)) <= 2.0 + 1e-12

    def test_singlet_reaches_tsirelson_bound(self):
        q = self.null.singlet_distribution()
        assert q.sum() == pytest.approx(1.0)
        assert abs(self.null.chsh_value(q)) == pytest.approx(2 * np.sqrt(2), abs=1e-12)

    def test_singlet_is_outside_local_polytope_with_positive_kl(self):
        q = self.null.singlet_distribution()
        assert not self.null.contains(q)
        ripr = self.null.reverse_information_projection(q)
        assert ripr.kl_divergence > 0.01
        e = self.null.optimal_e_variable(q)
        assert np.all(self.null.vertices @ e <= 1.0 + 1e-12)
        assert _growth(q, e) == pytest.approx(ripr.kl_divergence, abs=1e-6)

    def test_local_mixture_has_zero_kl(self):
        rng = np.random.default_rng(4)
        w = rng.dirichlet(np.ones(16))
        q = w @ self.null.vertices
        assert self.null.contains(q)
        assert np.allclose(self.null.optimal_e_variable(q), 1.0, atol=1e-8)

    def test_exhaustive_deterministic_strategies_match_vertices(self):
        seen = set()
        for A0, A1, B0, B1 in itertools.product([0, 1], repeat=4):
            outcome_support = tuple(
                sorted((a, b, [A0, A1][a], [B0, B1][b]) for a in (0, 1) for b in (0, 1))
            )
            seen.add(outcome_support)
        assert len(seen) == 16
        assert all(np.count_nonzero(v) == 4 for v in self.null.vertices)

    def test_setting_probabilities_validation(self):
        with pytest.raises(ValueError):
            CHSHLocalRealismNull(setting_probs=[0.5, 0.5, 0.0, 0.0])
        with pytest.raises(ValueError):
            CHSHLocalRealismNull.outcome_index(2, 0, 0, 0)
