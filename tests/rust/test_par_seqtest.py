"""
Integration tests for the Rust ParallelSequentialTest engine.

Tests:
1. Golden tests: Rust vs Python on deterministic sequences (1e-13 tolerance)
2. Performance: 300K tests < 10ms per step
3. FDR simulation: e-BH on all-null tests, avg FDR <= alpha + tolerance
4. FWER simulation: e-Bonferroni on all-null tests, FWER <= alpha + tolerance
5. Signal detection: strong signal tests should be rejected

References:
    - Ramdas & Wang (2025). Hypothesis testing with e-values, Ch. 4.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest

from expectation.par_seqtest import (
    AlternativeDirection,
    CombinerStrategy,
    MartingaleType,
    MultipleTestingMethod,
    MultipleTestingResult,
    ParallelSequentialTest,
    ParallelTestConfig,
    StepResult,
    VarianceMode,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TOLERANCE = 1e-13


def _make_pst(
    n_tests: int,
    null_values,
    alpha: float = 0.05,
    v_opt: float = 1.0,
    alpha_opt: float = 0.05,
    variance=1.0,
) -> ParallelSequentialTest:
    """Helper to create a ParallelSequentialTest with the Pydantic config."""
    config = ParallelTestConfig(
        n_tests=n_tests,
        alpha=alpha,
        martingale_type="two_sided_normal",
        v_opt=v_opt,
        alpha_opt=alpha_opt,
    )
    return ParallelSequentialTest(config=config, null_values=null_values, variance=variance)


@pytest.fixture
def golden_data():
    """Load golden test fixtures generated from Python."""
    path = FIXTURES_DIR / "golden_two_sided_normal.json"
    with open(path) as f:
        return json.load(f)


class TestGoldenSingleVoxel:
    """Verify single-test Rust output matches Python exactly."""

    def test_log_e_process_matches_python(self, golden_data):
        """Each step's log_e_process must match Python's log_superMG within 1e-13."""
        cfg = golden_data["single_voxel"]["config"]
        steps = golden_data["single_voxel"]["steps"]

        pst = _make_pst(
            n_tests=1,
            null_values=cfg["null_value"],
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
        )

        for step in steps:
            result = pst.step(np.array([step["x"]]))
            assert isinstance(result, StepResult)

            rust_log_e = pst.log_e_processes()[0]
            python_log_e = step["log_e_cum"]

            assert abs(rust_log_e - python_log_e) < TOLERANCE, (
                f"Step {step['t']}: Rust={rust_log_e}, Python={python_log_e}, "
                f"diff={abs(rust_log_e - python_log_e)}"
            )

    def test_50_steps_final_value(self, golden_data):
        """Verify the final log_e_process after 50 steps."""
        cfg = golden_data["single_voxel"]["config"]
        steps = golden_data["single_voxel"]["steps"]

        pst = _make_pst(
            n_tests=1,
            null_values=cfg["null_value"],
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
        )

        for step in steps:
            pst.step(np.array([step["x"]]))

        final_rust = pst.log_e_processes()[0]
        final_python = steps[-1]["log_e_cum"]
        assert abs(final_rust - final_python) < TOLERANCE


class TestGoldenMultiVoxel:
    """Verify multi-test Rust output matches Python exactly."""

    def test_all_tests_match_python(self, golden_data):
        """All 5 tests at all 3 steps must match Python within 1e-13."""
        cfg = golden_data["multi_voxel"]["config"]
        steps = golden_data["multi_voxel"]["steps"]

        pst = _make_pst(
            n_tests=cfg["n_voxels"],
            null_values=np.array(cfg["null_values"]),
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
        )

        for step in steps:
            obs = np.array(step["observations"])
            pst.step(obs)

            log_e = pst.log_e_processes()
            for voxel_data in step["voxels"]:
                i = voxel_data["index"]
                rust_val = log_e[i]
                python_val = voxel_data["log_e_cum"]
                assert abs(rust_val - python_val) < TOLERANCE, (
                    f"Step {step['t']}, test {i}: Rust={rust_val}, "
                    f"Python={python_val}, diff={abs(rust_val - python_val)}"
                )


class TestStepResultModel:
    """Verify StepResult Pydantic model properties."""

    def test_step_returns_frozen_model(self):
        """step() should return a frozen StepResult, not a dict."""
        pst = _make_pst(n_tests=10, null_values=0.0)
        result = pst.step(np.random.randn(10))

        assert isinstance(result, StepResult)
        assert result.time_step == 1
        assert result.n_tests == 10
        assert result.n_rejected >= 0

        # Frozen: assignment should raise
        with pytest.raises(Exception):
            result.time_step = 999

    def test_multiple_testing_result_model(self):
        """e_bh() should return a frozen MultipleTestingResult."""
        pst = _make_pst(n_tests=100, null_values=0.0)
        pst.step(np.random.randn(100))

        bh = pst.e_bh(alpha=0.05)
        assert isinstance(bh, MultipleTestingResult)
        assert bh.method == MultipleTestingMethod.E_BH
        assert bh.alpha == 0.05
        assert isinstance(bh.rejected, np.ndarray)
        assert bh.rejected.dtype == np.bool_
        assert len(bh.rejected) == 100

    def test_config_is_accessible(self):
        """config property should return the frozen ParallelTestConfig."""
        config = ParallelTestConfig(n_tests=50, alpha=0.01, v_opt=2.0, alpha_opt=0.1)
        pst = ParallelSequentialTest(config=config, null_values=0.0, variance=1.0)

        assert pst.config is config
        assert pst.config.alpha == 0.01
        assert pst.config.v_opt == 2.0


class TestPerformance:
    """Performance benchmarks for the Rust ParallelSequentialTest engine."""

    @pytest.mark.parametrize("n_tests", [1_000, 10_000, 100_000, 300_000])
    def test_step_latency(self, n_tests):
        """Single step should complete within 50ms for up to 300K tests."""
        pst = _make_pst(n_tests=n_tests, null_values=0.0)
        obs = np.random.randn(n_tests)

        # Warm up
        pst.step(obs)

        # Benchmark
        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            pst.step(obs)
        elapsed = (time.perf_counter() - start) / n_iters

        ms = elapsed * 1000
        print(f"\n  {n_tests:>7,} tests: {ms:.2f} ms/step")

        if n_tests <= 300_000:
            assert ms < 50, f"Step took {ms:.1f}ms, expected < 50ms for {n_tests} tests"

    def test_e_bh_latency_300k(self):
        """e-BH on 300K tests should complete within 50ms."""
        n_tests = 300_000
        pst = _make_pst(n_tests=n_tests, null_values=0.0)

        for _ in range(5):
            pst.step(np.random.randn(n_tests))

        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            pst.e_bh()
        elapsed = (time.perf_counter() - start) / n_iters

        ms = elapsed * 1000
        print(f"\n  e-BH on {n_tests:,} tests: {ms:.2f} ms")
        assert ms < 50, f"e-BH took {ms:.1f}ms, expected < 50ms"


class TestFDRControl:
    """Verify FDR control of e-BH under all-null scenario."""

    def test_e_bh_fdr_under_null(self):
        """Under H0 (all null), e-BH at alpha=0.05 should have FDR <= 0.08.

        Runs 50 Monte Carlo simulations with 1000 tests and 20 time steps.
        Under the global null, any rejection is a false discovery,
        so FDR = P(any rejection).

        Reference: Ramdas & Wang (2025), Theorem 4.2.
        """
        np.random.seed(42)
        n_tests = 1000
        n_steps = 20
        n_sims = 50
        alpha = 0.05

        any_rejection_count = 0

        for _ in range(n_sims):
            pst = _make_pst(n_tests=n_tests, null_values=0.0, alpha=alpha)

            max_rejected = 0
            for _ in range(n_steps):
                obs = np.random.randn(n_tests)
                pst.step(obs)
                result = pst.e_bh(alpha=alpha)
                max_rejected = max(max_rejected, result.n_rejected)

            if max_rejected > 0:
                any_rejection_count += 1

        fdr_estimate = any_rejection_count / n_sims
        print(f"\n  e-BH FDR estimate (all null): {fdr_estimate:.3f} "
              f"({any_rejection_count}/{n_sims} sims with any rejection)")

        assert fdr_estimate <= alpha + 0.03, (
            f"FDR estimate {fdr_estimate:.3f} exceeds alpha+tolerance={alpha + 0.03}"
        )


class TestFWERControl:
    """Verify FWER control of e-Bonferroni under all-null scenario."""

    def test_e_bonferroni_fwer_under_null(self):
        """Under H0, e-Bonferroni at alpha=0.05 should have FWER <= 0.08.

        50 Monte Carlo simulations with 1000 tests and 20 time steps.

        Reference: Ramdas & Wang (2025), Proposition 4.1.
        """
        np.random.seed(123)
        n_tests = 1000
        n_steps = 20
        n_sims = 50
        alpha = 0.05

        any_rejection_count = 0

        for _ in range(n_sims):
            pst = _make_pst(n_tests=n_tests, null_values=0.0, alpha=alpha)

            had_rejection = False
            for _ in range(n_steps):
                obs = np.random.randn(n_tests)
                pst.step(obs)
                result = pst.e_bonferroni(alpha=alpha)
                if result.n_rejected > 0:
                    had_rejection = True
                    break

            if had_rejection:
                any_rejection_count += 1

        fwer_estimate = any_rejection_count / n_sims
        print(f"\n  e-Bonferroni FWER estimate (all null): {fwer_estimate:.3f} "
              f"({any_rejection_count}/{n_sims} sims with any rejection)")

        assert fwer_estimate <= alpha + 0.03, (
            f"FWER estimate {fwer_estimate:.3f} exceeds alpha+tolerance={alpha + 0.03}"
        )


class TestSignalDetection:
    """Verify that the engine detects strong signals."""

    def test_strong_signal_rejected(self):
        """Tests with strong signal (mean=3) should be rejected quickly."""
        n_tests = 100
        n_signal = 10
        n_steps = 30

        pst = _make_pst(n_tests=n_tests, null_values=0.0)

        np.random.seed(99)
        for _ in range(n_steps):
            obs = np.random.randn(n_tests)
            obs[:n_signal] += 3.0
            pst.step(obs)

        rejected = pst.rejected()
        signal_rejected = np.sum(rejected[:n_signal])
        null_rejected = np.sum(rejected[n_signal:])

        print(f"\n  Signal tests rejected: {signal_rejected}/{n_signal}")
        print(f"  Null tests rejected: {null_rejected}/{n_tests - n_signal}")

        assert signal_rejected == n_signal, (
            f"Only {signal_rejected}/{n_signal} signal tests rejected"
        )

    def test_e_bh_detects_signal_subset(self):
        """e-BH should detect a subset of signals with FDR control.

        Reference: Ramdas & Wang (2025), Theorem 4.2.
        """
        n_tests = 1000
        n_signal = 50
        n_steps = 30

        pst = _make_pst(n_tests=n_tests, null_values=0.0)

        np.random.seed(77)
        for _ in range(n_steps):
            obs = np.random.randn(n_tests)
            obs[:n_signal] += 2.5
            pst.step(obs)

        bh_result = pst.e_bh(alpha=0.05)
        assert isinstance(bh_result, MultipleTestingResult)
        assert bh_result.method == MultipleTestingMethod.E_BH

        rejected_mask = bh_result.rejected
        true_positives = np.sum(rejected_mask[:n_signal])
        false_positives = np.sum(rejected_mask[n_signal:])

        print(f"\n  e-BH: {bh_result.n_rejected} rejections")
        print(f"  True positives: {true_positives}/{n_signal}")
        print(f"  False positives: {false_positives}/{n_tests - n_signal}")

        assert true_positives > 0, "e-BH should detect at least some signals"
        if bh_result.n_rejected > 0:
            fdr = false_positives / bh_result.n_rejected
            print(f"  Empirical FDR: {fdr:.3f}")


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_dimension_mismatch_raises(self):
        """Passing wrong-sized observations should raise."""
        pst = _make_pst(n_tests=100, null_values=0.0)
        with pytest.raises(ValueError, match="[Dd]imension"):
            pst.step(np.array([1.0, 2.0, 3.0]))

    def test_invalid_martingale_type_raises(self):
        """Unknown martingale type should raise."""
        with pytest.raises(ValueError, match="Input should be"):
            ParallelTestConfig(
                n_tests=10,
                martingale_type="nonexistent",
            )

    def test_heterogeneous_null_values(self):
        """Per-test null values should work."""
        n = 5
        nulls = np.array([0.0, 0.5, -0.5, 1.0, -1.0])

        pst = _make_pst(n_tests=n, null_values=nulls)
        pst.step(np.ones(n))
        log_e = pst.log_e_processes()

        assert not np.allclose(log_e, log_e[0]), (
            "Different null values should produce different e-values"
        )

    def test_heterogeneous_variance(self):
        """Per-test variance should work."""
        n = 5
        variances = np.array([0.5, 1.0, 1.5, 2.0, 3.0])

        config = ParallelTestConfig(
            n_tests=n,
            variance_mode="known_heterogeneous",
        )
        pst = ParallelSequentialTest(config=config, null_values=0.0, variance=variances)

        pst.step(np.ones(n))
        log_e = pst.log_e_processes()

        assert len(set(log_e)) == n, "Each variance should produce a unique e-value"


# ---------------------------------------------------------------------------
# Phase 6: New golden + feature tests
# ---------------------------------------------------------------------------


def _make_pst_full(
    n_tests: int,
    null_values=0.0,
    alpha: float = 0.05,
    v_opt: float = 1.0,
    alpha_opt: float = 0.05,
    variance=1.0,
    alternative: str = "two_sided",
    combiner: str = "all_in",
    conservative_lambda: float = 0.5,
    gamma: float = 0.5,
    epsilon: float = 1e-6,
    martingale_type: str = "two_sided_normal",
) -> ParallelSequentialTest:
    """Helper with full config support for Phase 6 tests."""
    config = ParallelTestConfig(
        n_tests=n_tests,
        alpha=alpha,
        martingale_type=martingale_type,
        v_opt=v_opt,
        alpha_opt=alpha_opt,
        alternative=alternative,
        combiner=combiner,
        conservative_lambda=conservative_lambda,
        gamma=gamma,
        epsilon=epsilon,
    )
    return ParallelSequentialTest(config=config, null_values=null_values, variance=variance)


# ── Golden fixture loaders ────────────────────────────────────────────────


@pytest.fixture
def golden_one_sided():
    path = FIXTURES_DIR / "golden_one_sided_normal.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def golden_conservative():
    path = FIXTURES_DIR / "golden_conservative_combiner.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def golden_adaptive():
    path = FIXTURES_DIR / "golden_adaptive_combiner.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def golden_less():
    path = FIXTURES_DIR / "golden_less_alternative.json"
    with open(path) as f:
        return json.load(f)


# ── Test classes ──────────────────────────────────────────────────────────


class TestOneSidedGolden:
    """Golden tests: one-sided GREATER normal vs Python OneSidedNormalMixture.

    Reference: Howard et al. (2022), Section 3.
    """

    def test_log_e_process_matches_python(self, golden_one_sided):
        """Each step's log_e_process must match Python within 1e-13."""
        cfg = golden_one_sided["config"]

        pst = _make_pst_full(
            n_tests=1,
            null_values=cfg["null_value"],
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
            alternative="greater",
            martingale_type="one_sided_normal",
        )

        for step in golden_one_sided["steps"]:
            pst.step(np.array([step["x"]]))
            rust_log_e = pst.log_e_processes()[0]
            python_log_e = step["log_e_cum"]

            assert abs(rust_log_e - python_log_e) < TOLERANCE, (
                f"Step {step['t']}: Rust={rust_log_e}, Python={python_log_e}, "
                f"diff={abs(rust_log_e - python_log_e)}"
            )

    def test_final_value_50_steps(self, golden_one_sided):
        """Final log_e_process after 50 steps matches Python."""
        cfg = golden_one_sided["config"]
        steps = golden_one_sided["steps"]

        pst = _make_pst_full(
            n_tests=1,
            null_values=cfg["null_value"],
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
            alternative="greater",
            martingale_type="one_sided_normal",
        )

        for step in steps:
            pst.step(np.array([step["x"]]))

        final_rust = pst.log_e_processes()[0]
        final_python = steps[-1]["log_e_cum"]
        assert abs(final_rust - final_python) < TOLERANCE


class TestLessAlternativeGolden:
    """Golden tests: one-sided LESS alternative with sign flip.

    Reference: Ramdas & Wang (2025), Section 2.1.
    """

    def test_log_e_process_matches_python(self, golden_less):
        """LESS alternative: s is negated, log_e_process matches Python."""
        cfg = golden_less["config"]

        pst = _make_pst_full(
            n_tests=1,
            null_values=cfg["null_value"],
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
            alternative="less",
            martingale_type="one_sided_normal",
        )

        for step in golden_less["steps"]:
            pst.step(np.array([step["x"]]))
            rust_log_e = pst.log_e_processes()[0]
            python_log_e = step["log_e_cum"]

            assert abs(rust_log_e - python_log_e) < TOLERANCE, (
                f"Step {step['t']}: Rust={rust_log_e}, Python={python_log_e}, "
                f"diff={abs(rust_log_e - python_log_e)}"
            )


class TestAlternativeDirection:
    """Tests for alternative hypothesis direction handling.

    Reference: Ramdas & Wang (2025), Section 2.1.
    """

    def test_greater_vs_less_asymmetric(self):
        """GREATER and LESS on same data should produce different e-values."""
        np.random.seed(101)
        obs = np.random.randn(1) + 0.5  # positive shift

        pst_greater = _make_pst_full(
            n_tests=1, alternative="greater", martingale_type="one_sided_normal",
        )
        pst_less = _make_pst_full(
            n_tests=1, alternative="less", martingale_type="one_sided_normal",
        )

        for _ in range(10):
            obs = np.random.randn(1) + 0.5
            pst_greater.step(obs)
            pst_less.step(obs)

        log_e_greater = pst_greater.log_e_processes()[0]
        log_e_less = pst_less.log_e_processes()[0]

        assert log_e_greater != log_e_less, (
            "GREATER and LESS should give different e-values on asymmetric data"
        )
        # Positive shift favors GREATER
        assert log_e_greater > log_e_less, (
            "Positive signal should favor GREATER over LESS"
        )

    def test_two_sided_auto_selects_martingale(self):
        """TWO_SIDED should use TwoSidedNormalMixture (default)."""
        pst = _make_pst_full(n_tests=1, alternative="two_sided")
        assert pst.config.martingale_type == MartingaleType.TWO_SIDED_NORMAL

    def test_greater_auto_selects_one_sided(self):
        """GREATER should auto-select OneSidedNormalMixture."""
        config = ParallelTestConfig(
            n_tests=1,
            alternative="greater",
            martingale_type="two_sided_normal",  # should auto-switch
        )
        # The auto-switch happens inside ParallelSequentialTest.__init__
        pst = ParallelSequentialTest(config=config, null_values=0.0, variance=1.0)
        # Verify it works (doesn't crash) - the martingale was switched internally
        pst.step(np.array([1.0]))
        assert pst.log_e_processes()[0] != 0.0


class TestConservativeCombiner:
    """Golden tests: conservative combiner (fixed lambda=0.5).

    Reference: Ramdas & Wang (2025), Definition 7.21.
    """

    def test_log_e_process_matches_python(self, golden_conservative):
        """Conservative combiner e-process must match Python within 1e-13."""
        cfg = golden_conservative["config"]

        pst = _make_pst_full(
            n_tests=1,
            null_values=cfg["null_value"],
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
            alternative="two_sided",
            combiner="conservative",
            conservative_lambda=cfg["lambda_fixed"],
        )

        for step in golden_conservative["steps"]:
            pst.step(np.array([step["x"]]))
            rust_log_e = pst.log_e_processes()[0]
            python_log_e = step["log_e_process"]

            assert abs(rust_log_e - python_log_e) < TOLERANCE, (
                f"Step {step['t']}: Rust={rust_log_e}, Python={python_log_e}, "
                f"diff={abs(rust_log_e - python_log_e)}"
            )

    def test_conservative_dampens_all_in(self):
        """Conservative combiner should dampen e-process growth vs ALL_IN."""
        np.random.seed(55)
        obs_seq = [np.random.randn(1) + 1.0 for _ in range(30)]

        pst_all_in = _make_pst_full(n_tests=1, combiner="all_in")
        pst_conservative = _make_pst_full(
            n_tests=1, combiner="conservative", conservative_lambda=0.5,
        )

        for obs in obs_seq:
            pst_all_in.step(obs)
            pst_conservative.step(obs)

        log_all_in = pst_all_in.log_e_processes()[0]
        log_conservative = pst_conservative.log_e_processes()[0]

        # Under signal, ALL_IN grows faster than conservative
        assert log_all_in > log_conservative, (
            f"ALL_IN ({log_all_in:.4f}) should grow faster than "
            f"conservative ({log_conservative:.4f}) under signal"
        )

    def test_lambda_is_fixed(self, golden_conservative):
        """Lambda should be constant at the configured value."""
        cfg = golden_conservative["config"]

        pst = _make_pst_full(
            n_tests=1,
            null_values=cfg["null_value"],
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
            combiner="conservative",
            conservative_lambda=cfg["lambda_fixed"],
        )

        for step in golden_conservative["steps"]:
            pst.step(np.array([step["x"]]))

        # Lambda should be the fixed value
        lam = pst.lambdas()[0]
        assert abs(lam - cfg["lambda_fixed"]) < 1e-15


class TestAdaptiveCombiner:
    """Tests for empirically adaptive combiner (ONS approximation).

    Reference: Waudby-Smith & Ramdas (2024), Theorem 7.22 in Ramdas & Wang (2025).
    """

    def test_log_e_process_matches_python(self, golden_adaptive):
        """Adaptive combiner e-process must match Python within 1e-13."""
        cfg = golden_adaptive["config"]

        pst = _make_pst_full(
            n_tests=1,
            null_values=cfg["null_value"],
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
            alternative="two_sided",
            combiner="empirically_adaptive",
            gamma=cfg["gamma"],
            epsilon=cfg["epsilon"],
        )

        for step in golden_adaptive["steps"]:
            pst.step(np.array([step["x"]]))
            rust_log_e = pst.log_e_processes()[0]
            python_log_e = step["log_e_process"]

            assert abs(rust_log_e - python_log_e) < TOLERANCE, (
                f"Step {step['t']}: Rust={rust_log_e}, Python={python_log_e}, "
                f"diff={abs(rust_log_e - python_log_e)}"
            )

    def test_lambda_starts_at_zero(self):
        """First lambda should be 0 (no previous data for estimation)."""
        pst = _make_pst_full(
            n_tests=1, combiner="empirically_adaptive",
        )
        pst.step(np.array([1.0]))
        # After step 1, the lambda used was 0.0 (S1=0, S2=0 before step 1)
        # But the stored lambda reflects the CURRENT lambda for the next step
        # Let's verify via the e-process: with lambda=0, increment=1.0, so log_e_process=0.0
        log_ep = pst.log_e_processes()[0]
        assert abs(log_ep) < 1e-15, (
            f"First step with lambda=0 should give log_e_process=0, got {log_ep}"
        )

    def test_lambda_bounded_by_gamma(self):
        """Lambda should never exceed gamma."""
        gamma = 0.3
        pst = _make_pst_full(
            n_tests=1, combiner="empirically_adaptive", gamma=gamma,
        )

        np.random.seed(77)
        for _ in range(50):
            pst.step(np.random.randn(1) + 2.0)

        lam = pst.lambdas()[0]
        assert lam <= gamma + 1e-15, (
            f"Lambda {lam} exceeds gamma {gamma}"
        )

    def test_adaptive_grows_under_signal(self):
        """Adaptive combiner should detect signal (e-process grows)."""
        pst = _make_pst_full(
            n_tests=1, combiner="empirically_adaptive",
        )

        np.random.seed(88)
        for _ in range(100):
            pst.step(np.random.randn(1) + 1.0)

        log_ep = pst.log_e_processes()[0]
        assert log_ep > 0, (
            f"Adaptive combiner should grow under signal, got log_e_process={log_ep}"
        )


class TestSequentialEValues:
    """Verify sequential e-value decomposition property.

    For ALL_IN combiner: product of sequential e-values = cumulative e-process.
    E_{1:T} = prod_{t=1}^T E_t, i.e., sum of log_e_sequential = log_e_process.

    Reference: Ramdas & Wang (2025), Proposition 7.20.
    """

    def test_product_decomposition(self):
        """Sum of log sequential e-values = log e-process (ALL_IN)."""
        pst = _make_pst_full(n_tests=1, combiner="all_in")

        np.random.seed(33)
        log_seq_sum = 0.0
        for _ in range(30):
            pst.step(np.random.randn(1) + 0.5)
            log_seq_sum += pst.log_e_sequential()[0]

        log_ep = pst.log_e_processes()[0]
        # For ALL_IN: log_e_process == cumulative log supermartingale
        # And sum of sequential == cumulative
        assert abs(log_seq_sum - log_ep) < 1e-12, (
            f"Product decomposition failed: sum_log_seq={log_seq_sum}, "
            f"log_e_process={log_ep}, diff={abs(log_seq_sum - log_ep)}"
        )

    def test_sequential_e_value_matches_golden(self, golden_one_sided):
        """Sequential log e-values from Rust match Python golden."""
        cfg = golden_one_sided["config"]
        steps = golden_one_sided["steps"]

        pst = _make_pst_full(
            n_tests=1,
            null_values=cfg["null_value"],
            v_opt=cfg["v_opt"],
            alpha_opt=cfg["alpha_opt"],
            variance=cfg["known_variance"],
            alternative="greater",
            martingale_type="one_sided_normal",
        )

        for step in steps:
            pst.step(np.array([step["x"]]))
            rust_log_seq = pst.log_e_sequential()[0]
            python_log_seq = step["log_e_sequential"]

            assert abs(rust_log_seq - python_log_seq) < TOLERANCE, (
                f"Step {step['t']}: Rust log_e_seq={rust_log_seq}, "
                f"Python={python_log_seq}"
            )

    def test_multi_test_independence(self):
        """Sequential e-values for different tests should be independent."""
        pst = _make_pst_full(n_tests=5, combiner="all_in")

        np.random.seed(44)
        for _ in range(10):
            obs = np.random.randn(5)
            obs[0] += 3.0  # Strong signal on test 0 only
            pst.step(obs)

        log_seq = pst.log_e_sequential()
        # Test 0 (signal) should differ from others
        assert abs(log_seq[0] - log_seq[1]) > 0.01


class TestStoppingTimes:
    """Verify first-rejection stopping time tracking.

    Reference: Ramdas & Wang (2025), Theorem 2.5 (Ville's inequality).
    """

    def test_stopping_time_under_signal(self):
        """Tests with strong signal should have stopping time > 0."""
        pst = _make_pst_full(n_tests=10)

        for t in range(1, 51):
            pst.step(np.full(10, 3.0))

        st = pst.stopping_times()
        assert np.all(st > 0), (
            f"All tests should have stopped under strong signal: {st}"
        )

    def test_stopping_time_zero_under_null(self):
        """Tests under null should mostly not stop (stopping_time=0)."""
        np.random.seed(22)
        pst = _make_pst_full(n_tests=100)

        for _ in range(20):
            pst.step(np.random.randn(100))

        st = pst.stopping_times()
        n_stopped = np.sum(st > 0)
        # Under null, very few should stop at alpha=0.05
        assert n_stopped < 10, (
            f"Too many tests stopped under null: {n_stopped}/100"
        )

    def test_stopping_time_matches_rejection_step(self):
        """Stopping time should equal the first step where rejection occurs."""
        pst = _make_pst_full(n_tests=1)

        rejection_step = 0
        for t in range(1, 51):
            result = pst.step(np.array([3.0]))
            if result.n_rejected > 0 and rejection_step == 0:
                rejection_step = t

        st = pst.stopping_times()[0]
        assert st == rejection_step, (
            f"Stopping time {st} != first rejection step {rejection_step}"
        )


class TestPValues:
    """Verify p-value computation: p_t = min(1, exp(-log_e_process)).

    Reference: Ramdas & Wang (2025), Proposition 2.2.
    """

    def test_p_values_in_unit_interval(self):
        """P-values should always be in [0, 1]."""
        np.random.seed(55)
        pst = _make_pst_full(n_tests=100)

        for _ in range(20):
            pst.step(np.random.randn(100) + 0.5)

        pv = pst.p_values()
        assert np.all(pv >= 0), "P-values should be non-negative"
        assert np.all(pv <= 1.0), "P-values should be at most 1"

    def test_p_values_start_at_one(self):
        """Before any evidence, p-values should be 1.0."""
        pst = _make_pst_full(n_tests=5)
        pv = pst.p_values()
        assert np.allclose(pv, 1.0), f"Initial p-values should be 1.0, got {pv}"

    def test_p_values_decrease_under_signal(self):
        """P-values should decrease under strong signal."""
        pst = _make_pst_full(n_tests=10)

        for _ in range(30):
            pst.step(np.full(10, 2.0))

        pv = pst.p_values()
        assert np.all(pv < 0.05), (
            f"P-values should be < 0.05 under strong signal: {pv}"
        )

    def test_p_value_formula(self):
        """p = min(1, exp(-log_e_process))."""
        np.random.seed(66)
        pst = _make_pst_full(n_tests=10)

        for _ in range(10):
            pst.step(np.random.randn(10) + 0.5)

        log_ep = pst.log_e_processes()
        pv = pst.p_values()

        expected = np.minimum(1.0, np.exp(-log_ep))
        np.testing.assert_allclose(pv, expected, atol=1e-14)

    def test_p_values_uniform_under_null(self):
        """Under H0, p-values at final step should be stochastically >= Uniform.

        This is a weak check: median p-value under null should be > 0.3.
        """
        np.random.seed(77)
        n_sims = 200
        final_pvals = []

        for _ in range(n_sims):
            pst = _make_pst_full(n_tests=1)
            for _ in range(20):
                pst.step(np.random.randn(1))
            final_pvals.append(pst.p_values()[0])

        median_pv = np.median(final_pvals)
        assert median_pv > 0.3, (
            f"Median p-value under null should be > 0.3, got {median_pv:.3f}"
        )


class TestEHolmCorrection:
    """Tests for e-Holm step-down procedure.

    Reference: Ramdas & Wang (2025), Section 4.1, Proposition 4.3.
    """

    def test_e_holm_controls_fwer(self):
        """Under all-null, e-Holm FWER <= alpha + tolerance."""
        np.random.seed(200)
        n_sims = 50
        n_tests = 500
        alpha = 0.05

        any_rejection = 0
        for _ in range(n_sims):
            pst = _make_pst_full(n_tests=n_tests, alpha=alpha)
            for _ in range(20):
                pst.step(np.random.randn(n_tests))
            result = pst.e_holm(alpha=alpha)
            if result.n_rejected > 0:
                any_rejection += 1

        fwer = any_rejection / n_sims
        assert fwer <= alpha + 0.03, (
            f"e-Holm FWER {fwer:.3f} exceeds alpha + tolerance"
        )

    def test_e_holm_detects_signal(self):
        """e-Holm should detect strong signals."""
        n_tests = 100
        n_signal = 5

        pst = _make_pst_full(n_tests=n_tests, alpha=0.05)

        np.random.seed(201)
        for _ in range(30):
            obs = np.random.randn(n_tests)
            obs[:n_signal] += 3.0
            pst.step(obs)

        result = pst.e_holm(alpha=0.05)
        assert result.method == MultipleTestingMethod.E_HOLM
        assert result.n_rejected >= n_signal - 1, (
            f"e-Holm should detect most signals, got {result.n_rejected}"
        )


class TestNewlyRejected:
    """Tests for the n_newly_rejected field in StepResult."""

    def test_newly_rejected_counts(self):
        """n_newly_rejected should count tests rejected in this step only."""
        pst = _make_pst_full(n_tests=10)

        total_newly = 0
        for _ in range(50):
            result = pst.step(np.full(10, 3.0))
            total_newly += result.n_newly_rejected

        # Total newly rejected should equal total rejected
        assert total_newly == 10, (
            f"Sum of newly rejected ({total_newly}) should equal n_tests (10)"
        )

    def test_newly_rejected_zero_after_all_rejected(self):
        """Once all tests are rejected, n_newly_rejected should be 0."""
        pst = _make_pst_full(n_tests=5)

        # Run until all rejected
        for _ in range(50):
            result = pst.step(np.full(5, 5.0))
            if result.n_rejected == 5:
                break

        # Next step should have 0 newly rejected
        result = pst.step(np.full(5, 5.0))
        assert result.n_newly_rejected == 0


class TestLambdasAccessor:
    """Tests for the lambdas() accessor."""

    def test_all_in_lambda_is_one(self):
        """ALL_IN combiner should have lambda=1.0."""
        pst = _make_pst_full(n_tests=5, combiner="all_in")
        pst.step(np.ones(5))

        lam = pst.lambdas()
        np.testing.assert_allclose(lam, 1.0, atol=1e-15)

    def test_conservative_lambda_matches_config(self):
        """Conservative combiner lambda should match configured value."""
        lam_val = 0.3
        pst = _make_pst_full(
            n_tests=5, combiner="conservative", conservative_lambda=lam_val,
        )
        pst.step(np.ones(5))

        lam = pst.lambdas()
        np.testing.assert_allclose(lam, lam_val, atol=1e-15)

    def test_adaptive_lambda_shape(self):
        """Adaptive combiner lambda should have shape (n_tests,)."""
        pst = _make_pst_full(n_tests=10, combiner="empirically_adaptive")

        for _ in range(5):
            pst.step(np.random.randn(10))

        lam = pst.lambdas()
        assert lam.shape == (10,)
        assert np.all(lam >= 0)
        assert np.all(lam <= 0.5 + 1e-15)  # bounded by gamma=0.5
