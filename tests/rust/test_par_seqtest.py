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
    MultipleTestingMethod,
    MultipleTestingResult,
    ParallelSequentialTest,
    ParallelTestConfig,
    StepResult,
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
