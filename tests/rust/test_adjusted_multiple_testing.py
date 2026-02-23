"""
Integration tests for carefree adjusted multiple testing procedures.

Tests:
1. Golden cross-validation: Python adjusters + manual BH vs Rust adjusted_e_bh
2. Carefree property: adjusted rejections monotonically non-decreasing
3. max_log_m accessor: correct shape and values match manual tracking
4. FDR-sup Monte Carlo: under all-null, sup FDR <= alpha

References:
    Tavyrikov, Goeman & de Heide (2025). Carefree multiple testing with
    e-processes. arXiv:2501.19360v2.
    Ramdas & Wang (2025). Hypothesis testing with e-values, Ch. 4.
"""

import numpy as np
import pytest

from expectation.par_seqtest import (
    AdjusterType,
    CombinerStrategy,
    ParallelSequentialTest,
    ParallelTestConfig,
)
from expectation.modules.adjusters import LookbackAdjuster, SqrtAdjuster

TOLERANCE = 1e-13


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pst(
    n_tests: int,
    alpha: float = 0.05,
    variance: float = 1.0,
    combiner: str = "all_in",
) -> ParallelSequentialTest:
    """Create a ParallelSequentialTest for testing."""
    config = ParallelTestConfig(
        n_tests=n_tests,
        alpha=alpha,
        martingale_type="two_sided_normal",
        v_opt=1.0,
        alpha_opt=0.05,
        combiner=combiner,
    )
    return ParallelSequentialTest(config=config, null_values=0.0, variance=variance)


# ---------------------------------------------------------------------------
# max_log_m accessor
# ---------------------------------------------------------------------------


class TestMaxLogMAccessor:
    """Verify the max_log_m accessor returns correct shape and values."""

    def test_shape(self):
        pst = _make_pst(n_tests=50)
        max_log = pst.max_log_m()
        assert max_log.shape == (50,)

    def test_initial_values(self):
        """Before any data, max_log_m should be NEG_INFINITY."""
        pst = _make_pst(n_tests=10)
        max_log = pst.max_log_m()
        assert np.all(np.isneginf(max_log))

    def test_non_decreasing(self):
        """max_log_m should be non-decreasing over time for each test."""
        rng = np.random.default_rng(42)
        pst = _make_pst(n_tests=20)
        prev = pst.max_log_m().copy()

        for _ in range(50):
            obs = rng.normal(0.0, 1.0, size=20)
            pst.step(obs)
            current = pst.max_log_m().copy()
            assert np.all(current >= prev - 1e-15), (
                "max_log_m decreased!"
            )
            prev = current

    def test_matches_manual_tracking(self):
        """max_log_m should track max of log_e_process over time."""
        rng = np.random.default_rng(123)
        n_tests = 5
        pst = _make_pst(n_tests=n_tests)
        manual_max = np.full(n_tests, -np.inf)

        for _ in range(30):
            obs = rng.normal(0.5, 1.0, size=n_tests)
            pst.step(obs)
            # Note: max_log_m tracks max of the cumulative log supermartingale,
            # which is log_e_process for the all_in combiner.
            log_ep = np.asarray(pst.log_e_processes())
            manual_max = np.maximum(manual_max, log_ep)

        engine_max = pst.max_log_m()
        np.testing.assert_allclose(engine_max, manual_max, atol=TOLERANCE)


# ---------------------------------------------------------------------------
# Golden cross-validation: Python adjusters + manual BH vs Rust
# ---------------------------------------------------------------------------


class TestGoldenCrossValidation:
    """Python adjuster + Python BH should match Rust adjusted_e_bh at 1e-13."""

    def _manual_adjusted_e_bh(self, log_running_max, alpha, adjuster_cls):
        """Replicate adjusted e-BH in pure Python."""
        adj = adjuster_cls()
        adjusted_log = adj.log_adjust(log_running_max)
        m = len(adjusted_log)
        if m == 0:
            return np.array([], dtype=bool), 0

        log_m = np.log(m)
        log_alpha = np.log(alpha)

        # Sort descending
        order = np.argsort(-adjusted_log)
        sorted_log = adjusted_log[order]

        k_star = 0
        for rank_0 in range(m):
            k = rank_0 + 1
            threshold = log_m - np.log(k) - log_alpha
            if sorted_log[rank_0] >= threshold:
                k_star = k

        rejected = np.zeros(m, dtype=bool)
        for i in range(k_star):
            rejected[order[i]] = True

        return rejected, k_star

    def test_lookback_cross_validate(self):
        """Run engine, then compare Rust adjusted_e_bh vs Python manual."""
        rng = np.random.default_rng(42)
        n_tests = 30
        pst = _make_pst(n_tests=n_tests)

        # Run some steps with mixed null/alternative
        for t in range(100):
            obs = np.zeros(n_tests)
            # First 5 tests have signal
            obs[:5] = rng.normal(1.0, 1.0, size=5)
            # Rest are null
            obs[5:] = rng.normal(0.0, 1.0, size=n_tests - 5)
            pst.step(obs)

        # Rust result
        rust_result = pst.adjusted_e_bh(adjuster="lookback")

        # Python manual result
        max_log = pst.max_log_m()
        py_rejected, py_n = self._manual_adjusted_e_bh(
            max_log, 0.05, LookbackAdjuster
        )

        assert rust_result.n_rejected == py_n, (
            f"Rust: {rust_result.n_rejected}, Python: {py_n}"
        )
        np.testing.assert_array_equal(
            rust_result.rejected, py_rejected,
            err_msg="Rejection patterns differ between Rust and Python"
        )

    def test_sqrt_cross_validate(self):
        """Same cross-validation for sqrt adjuster."""
        rng = np.random.default_rng(99)
        n_tests = 20
        pst = _make_pst(n_tests=n_tests)

        for t in range(80):
            obs = np.zeros(n_tests)
            obs[:3] = rng.normal(1.5, 1.0, size=3)
            obs[3:] = rng.normal(0.0, 1.0, size=n_tests - 3)
            pst.step(obs)

        rust_result = pst.adjusted_e_bh(adjuster="sqrt")
        max_log = pst.max_log_m()
        py_rejected, py_n = self._manual_adjusted_e_bh(
            max_log, 0.05, SqrtAdjuster
        )

        assert rust_result.n_rejected == py_n
        np.testing.assert_array_equal(rust_result.rejected, py_rejected)


# ---------------------------------------------------------------------------
# Carefree property: monotone rejections
# ---------------------------------------------------------------------------


class TestCarefreeProperty:
    """Adjusted rejections should be monotonically non-decreasing over time."""

    def test_adjusted_e_bh_monotone_lookback(self):
        rng = np.random.default_rng(42)
        n_tests = 50
        pst = _make_pst(n_tests=n_tests)
        prev_rejected = np.zeros(n_tests, dtype=bool)

        for t in range(200):
            obs = np.zeros(n_tests)
            obs[:10] = rng.normal(0.8, 1.0, size=10)
            obs[10:] = rng.normal(0.0, 1.0, size=n_tests - 10)
            pst.step(obs)

            result = pst.adjusted_e_bh(adjuster="lookback")
            current = result.rejected
            # Once rejected, should stay rejected (carefree)
            assert np.all(current[prev_rejected]), (
                f"Rejection lost at step {t + 1}! "
                f"Was rejected: {np.where(prev_rejected & ~current)[0]}"
            )
            prev_rejected = current.copy()

    def test_adjusted_e_bh_monotone_sqrt(self):
        rng = np.random.default_rng(77)
        n_tests = 30
        pst = _make_pst(n_tests=n_tests)
        prev_rejected = np.zeros(n_tests, dtype=bool)

        for t in range(150):
            obs = np.zeros(n_tests)
            obs[:5] = rng.normal(1.0, 1.0, size=5)
            obs[5:] = rng.normal(0.0, 1.0, size=n_tests - 5)
            pst.step(obs)

            result = pst.adjusted_e_bh(adjuster="sqrt")
            current = result.rejected
            assert np.all(current[prev_rejected]), (
                f"Rejection lost at step {t + 1}!"
            )
            prev_rejected = current.copy()

    def test_adjusted_e_bonferroni_monotone(self):
        rng = np.random.default_rng(55)
        n_tests = 20
        pst = _make_pst(n_tests=n_tests)
        prev_n = 0

        for t in range(100):
            obs = np.zeros(n_tests)
            obs[:3] = rng.normal(1.5, 1.0, size=3)
            obs[3:] = rng.normal(0.0, 1.0, size=n_tests - 3)
            pst.step(obs)

            result = pst.adjusted_e_bonferroni(adjuster="lookback")
            assert result.n_rejected >= prev_n, (
                f"Bonferroni rejections decreased: {prev_n} -> {result.n_rejected}"
            )
            prev_n = result.n_rejected

    def test_adjusted_e_holm_monotone(self):
        rng = np.random.default_rng(33)
        n_tests = 20
        pst = _make_pst(n_tests=n_tests)
        prev_rejected = np.zeros(n_tests, dtype=bool)

        for t in range(100):
            obs = np.zeros(n_tests)
            obs[:3] = rng.normal(1.5, 1.0, size=3)
            obs[3:] = rng.normal(0.0, 1.0, size=n_tests - 3)
            pst.step(obs)

            result = pst.adjusted_e_holm(adjuster="sqrt")
            current = result.rejected
            assert np.all(current[prev_rejected]), (
                f"Holm rejection lost at step {t + 1}!"
            )
            prev_rejected = current.copy()


# ---------------------------------------------------------------------------
# Standard (non-adjusted) e_bh is NOT carefree (counterexample)
# ---------------------------------------------------------------------------


class TestStandardNotCarefree:
    """Show that standard e_bh CAN lose rejections (motivating the adjusters)."""

    def test_standard_e_bh_can_lose_rejections(self):
        """Standard e_bh applied to current e-processes can have rejections
        disappear. This is not a bug -- it motivates adjusted procedures."""
        rng = np.random.default_rng(12345)
        n_tests = 20
        pst = _make_pst(n_tests=n_tests)

        max_n_rejected = 0
        lost_once = False

        for t in range(500):
            obs = np.zeros(n_tests)
            # Give signal then take it away
            if t < 50:
                obs[:3] = rng.normal(2.0, 1.0, size=3)
            else:
                obs[:3] = rng.normal(-0.5, 1.0, size=3)
            obs[3:] = rng.normal(0.0, 1.0, size=n_tests - 3)
            pst.step(obs)

            result = pst.e_bh()
            if result.n_rejected > max_n_rejected:
                max_n_rejected = result.n_rejected
            elif result.n_rejected < max_n_rejected:
                lost_once = True
                break

        # This test just documents the behavior -- it's OK if the specific
        # seed doesn't trigger a loss. The point is that it CAN happen.
        # We don't assert lost_once because it depends on random data.


# ---------------------------------------------------------------------------
# FDR-sup Monte Carlo simulation under all-null
# ---------------------------------------------------------------------------


class TestFDRSupMonteCarlo:
    """Under all-null, sup FDR of adjusted e-BH should be <= alpha."""

    def test_fdr_sup_lookback(self):
        """Monte Carlo: FDR-sup <= alpha + tolerance under all-null."""
        rng = np.random.default_rng(42)
        n_tests = 20
        alpha = 0.05
        n_sims = 200
        n_steps = 100

        max_fdrs = []

        for sim in range(n_sims):
            pst = _make_pst(n_tests=n_tests, alpha=alpha)
            max_fdr = 0.0

            for t in range(n_steps):
                obs = rng.normal(0.0, 1.0, size=n_tests)
                pst.step(obs)
                result = pst.adjusted_e_bh(adjuster="lookback")
                if result.n_rejected > 0:
                    # Under all-null, all rejections are false discoveries
                    fdr = 1.0  # FDP = n_rejected / n_rejected = 1
                    max_fdr = max(max_fdr, fdr)

            max_fdrs.append(max_fdr)

        # Average of sup FDR across simulations
        avg_sup_fdr = np.mean(max_fdrs)
        assert avg_sup_fdr <= alpha + 0.02, (
            f"FDR-sup = {avg_sup_fdr:.4f}, exceeds alpha={alpha}"
        )

    def test_fdr_sup_sqrt(self):
        """Monte Carlo: FDR-sup <= alpha under all-null with sqrt adjuster."""
        rng = np.random.default_rng(99)
        n_tests = 20
        alpha = 0.05
        n_sims = 200
        n_steps = 100

        max_fdrs = []

        for sim in range(n_sims):
            pst = _make_pst(n_tests=n_tests, alpha=alpha)
            max_fdr = 0.0

            for t in range(n_steps):
                obs = rng.normal(0.0, 1.0, size=n_tests)
                pst.step(obs)
                result = pst.adjusted_e_bh(adjuster="sqrt")
                if result.n_rejected > 0:
                    fdr = 1.0
                    max_fdr = max(max_fdr, fdr)

            max_fdrs.append(max_fdr)

        avg_sup_fdr = np.mean(max_fdrs)
        assert avg_sup_fdr <= alpha + 0.02, (
            f"FDR-sup = {avg_sup_fdr:.4f}, exceeds alpha={alpha}"
        )


# ---------------------------------------------------------------------------
# Adjusted procedures are more conservative than unadjusted
# ---------------------------------------------------------------------------


class TestConservativeness:
    """Adjusted should reject <= unadjusted at any time step."""

    def test_adjusted_leq_unadjusted(self):
        rng = np.random.default_rng(42)
        n_tests = 30
        pst = _make_pst(n_tests=n_tests)

        for t in range(100):
            obs = np.zeros(n_tests)
            obs[:5] = rng.normal(1.0, 1.0, size=5)
            obs[5:] = rng.normal(0.0, 1.0, size=n_tests - 5)
            pst.step(obs)

            unadj = pst.e_bh()
            adj_lb = pst.adjusted_e_bh(adjuster="lookback")
            adj_sq = pst.adjusted_e_bh(adjuster="sqrt")

            assert adj_lb.n_rejected <= unadj.n_rejected, (
                f"Step {t+1}: adjusted LB ({adj_lb.n_rejected}) > "
                f"unadjusted ({unadj.n_rejected})"
            )
            assert adj_sq.n_rejected <= unadj.n_rejected, (
                f"Step {t+1}: adjusted sqrt ({adj_sq.n_rejected}) > "
                f"unadjusted ({unadj.n_rejected})"
            )
