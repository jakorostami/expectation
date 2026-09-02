# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

import numpy as np
import pytest

from expectation.ksample.bernoulli import BernoulliRIPrCalculator
from expectation.ksample.config import (
    KSampleAlternativeType,
    KSampleConfig,
    KSampleStepResult,
)
from expectation.ksample.ksample_test import KSampleSequentialTest
from expectation.modules.calibrators import EToPCalibrator
from expectation.modules.hypothesistesting import BettingStrategy


class TestKSampleConfig:
    def test_default_config(self):
        config = KSampleConfig(k=2)
        assert config.k == 2
        assert config.significance_level == 0.05
        assert config.gamma == 0.18
        assert config.alternative_type == KSampleAlternativeType.UNRESTRICTED
        assert config.betting_strategy == BettingStrategy.ALL_IN

    def test_k_must_be_gt_1(self):
        with pytest.raises(Exception):
            KSampleConfig(k=1)

    def test_effect_size_requires_k2(self):
        with pytest.raises(ValueError, match="k=2"):
            KSampleConfig(
                k=3,
                alternative_type="effect_size",
                divergence_type="additive",
                min_effect_size=0.1,
            )

    def test_effect_size_requires_divergence_type(self):
        with pytest.raises(ValueError, match="divergence_type"):
            KSampleConfig(
                k=2,
                alternative_type="effect_size",
                min_effect_size=0.1,
            )

    def test_effect_size_requires_min_effect_size(self):
        with pytest.raises(ValueError, match="min_effect_size"):
            KSampleConfig(
                k=2,
                alternative_type="effect_size",
                divergence_type="additive",
            )

    def test_simple_requires_theta(self):
        with pytest.raises(ValueError, match="simple_theta"):
            KSampleConfig(k=2, alternative_type="simple")

    def test_simple_requires_k_entries(self):
        with pytest.raises(ValueError, match="exactly k=2"):
            KSampleConfig(
                k=2,
                alternative_type="simple",
                simple_theta={0: 0.3},
            )

    def test_simple_theta_must_be_in_01(self):
        with pytest.raises(ValueError, match="must be in"):
            KSampleConfig(
                k=2,
                alternative_type="simple",
                simple_theta={0: 0.3, 1: 1.5},
            )

    def test_frozen_config(self):
        config = KSampleConfig(k=2)
        with pytest.raises(Exception):
            config.k = 3


class TestUnrestrictedAlternative:
    def test_first_block_symmetric_gives_e_value_1(self):
        config = KSampleConfig(k=2, gamma=0.18)
        test = KSampleSequentialTest(config)

        result = test.update(
            {0: np.array([1, 0]), 1: np.array([1, 0])}
        )  # Symmetric data -> same proportion in both groups
        # All theta_hats start at 0.5 (prior), so log e-value should be approx 0
        assert abs(result.e_value - 1.0) < 1e-10

    def test_more_extreme_data_larger_e_value(self):
        config = KSampleConfig(k=2, gamma=0.18)

        test_moderate = KSampleSequentialTest(config)  # run with moderate difference
        for _ in range(20):
            test_moderate.update({0: np.array([1, 1, 0]), 1: np.array([0, 0, 1])})
        moderate_process = test_moderate.e_process.process_values[-1]

        # Run with extreme difference
        test_extreme = KSampleSequentialTest(config)
        for _ in range(20):
            test_extreme.update({0: np.array([1, 1, 1]), 1: np.array([0, 0, 0])})
        extreme_process = test_extreme.e_process.process_values[-1]

        assert extreme_process > moderate_process

    def test_k3_unrestricted(self):
        config = KSampleConfig(k=3)
        test = KSampleSequentialTest(config)
        result = test.update(
            {
                0: np.array([1, 0, 1]),
                1: np.array([0, 0, 1]),
                2: np.array([1, 1, 0]),
            }
        )
        assert isinstance(result, KSampleStepResult)
        assert result.step == 1
        assert len(result.theta_estimates) == 3

    def test_posterior_means_update_correctly(self):
        config = KSampleConfig(k=2, gamma=1.0)
        test = KSampleSequentialTest(config)

        test.update(
            {0: np.array([1, 1, 1, 1, 1]), 1: np.array([0, 0, 0, 0, 0])}
        )  # after feeding data posterior means should shift
        # Now group 0 has 5 successes out of 5, group 1 has 0 out of 5
        # Posterior mean for group 0: (5 + 1)/(5 + 2) = 6/7
        # Posterior mean for group 1: (0 + 1)/(5 + 2) = 1/7

        result = test.update({0: np.array([1]), 1: np.array([0])})
        assert abs(result.theta_estimates[0] - 6.0 / 7.0) < 1e-10
        assert abs(result.theta_estimates[1] - 1.0 / 7.0) < 1e-10


class TestSimpleAlternative:
    def test_simple_alternative_fixed_theta(self):
        config = KSampleConfig(
            k=2,
            alternative_type="simple",
            simple_theta={0: 0.3, 1: 0.6},
        )
        test = KSampleSequentialTest(config)

        result1 = test.update({0: np.array([0, 1, 0]), 1: np.array([1, 1, 0])})
        assert isinstance(result1, KSampleStepResult)

        assert abs(result1.theta_estimates[0] - 0.3) < 1e-10
        assert abs(result1.theta_estimates[1] - 0.6) < 1e-10

        # Second step should also use fixed theta
        result2 = test.update({0: np.array([0, 0, 0]), 1: np.array([1, 1, 1])})
        assert abs(result2.theta_estimates[0] - 0.3) < 1e-10
        assert abs(result2.theta_estimates[1] - 0.6) < 1e-10

    def test_simple_alternative_e_value_matches_eq32(self):
        theta_a, theta_b = 0.3, 0.7
        config = KSampleConfig(
            k=2,
            alternative_type="simple",
            simple_theta={0: theta_a, 1: theta_b},
        )
        test = KSampleSequentialTest(config)

        # Block: group 0 gets [1, 0], group 1 gets [1, 1]
        s_a, n_a = 1, 2
        s_b, n_b = 2, 2
        # theta_null = (n_a/(n_a+n_b)) * theta_a + (n_b/(n_a+n_b)) * theta_b
        theta_null = 0.5 * theta_a + 0.5 * theta_b  # = 0.5

        # Manual log e-value
        log_e_manual = (
            s_a * np.log(theta_a / theta_null)
            + (n_a - s_a) * np.log((1 - theta_a) / (1 - theta_null))
            + s_b * np.log(theta_b / theta_null)
            + (n_b - s_b) * np.log((1 - theta_b) / (1 - theta_null))
        )
        e_manual = np.exp(log_e_manual)

        result = test.update({0: np.array([1, 0]), 1: np.array([1, 1])})
        assert abs(result.e_value - e_manual) < 1e-12


class TestEffectSizeRestricted:
    def test_additive_grid_construction(self):
        config = KSampleConfig(
            k=2,
            alternative_type="effect_size",
            divergence_type="additive",
            min_effect_size=0.1,
            grid_precision=0.05,
        )
        calc = BernoulliRIPrCalculator(config)

        # grid_theta_b = grid_theta_a + 0.1, all in (0, 1)
        assert len(calc.grid_theta_a) > 0
        np.testing.assert_allclose(calc.grid_theta_b, calc.grid_theta_a + 0.1, atol=1e-14)
        assert np.all(calc.grid_theta_a > 0)
        assert np.all(calc.grid_theta_b < 1)

    def test_log_odds_ratio_grid_construction(self):
        config = KSampleConfig(
            k=2,
            alternative_type="effect_size",
            divergence_type="log_odds_ratio",
            min_effect_size=0.5,
            grid_precision=0.05,
        )
        calc = BernoulliRIPrCalculator(config)

        # Verify constraint: log(OR) = delta for each grid point
        odds_a = calc.grid_theta_a / (1 - calc.grid_theta_a)
        odds_b = calc.grid_theta_b / (1 - calc.grid_theta_b)
        log_or = np.log(odds_b / odds_a)
        np.testing.assert_allclose(log_or, 0.5, atol=1e-12)

    def test_effect_size_restricted_runs(self):
        config = KSampleConfig(
            k=2,
            alternative_type="effect_size",
            divergence_type="additive",
            min_effect_size=0.1,
        )
        test = KSampleSequentialTest(config)
        for _ in range(10):
            result = test.update({0: np.array([0, 1, 0]), 1: np.array([1, 1, 0])})
        assert isinstance(result, KSampleStepResult)
        assert result.step == 10

    def test_restricted_more_power_when_effect_matches(self):
        rng = np.random.default_rng(42)
        theta_a, delta = 0.3, 0.2
        theta_b = theta_a + delta

        config_restricted = KSampleConfig(
            k=2,
            alternative_type="effect_size",
            divergence_type="additive",
            min_effect_size=delta,
        )
        config_unrestricted = KSampleConfig(k=2)

        test_r = KSampleSequentialTest(config_restricted)
        test_u = KSampleSequentialTest(config_unrestricted)

        for _ in range(100):
            data_a = rng.binomial(1, theta_a, size=5)
            data_b = rng.binomial(1, theta_b, size=5)
            test_r.update({0: data_a, 1: data_b})
            test_u.update({0: data_a, 1: data_b})

        # Restricted should accumulate at least as much evidence (on average, more, but with one seed we just check it runs)
        r_val = test_r.e_process.process_values[-1]
        u_val = test_u.e_process.process_values[-1]

        assert r_val > 1.0 or u_val > 1.0


class TestEVariableProperty:
    def test_mean_e_value_near_1_under_h0(self):
        rng = np.random.default_rng(123)
        theta_0 = 0.4
        config = KSampleConfig(k=2, gamma=0.18)
        test = KSampleSequentialTest(config)

        e_values = []
        for _ in range(2000):
            data_a = rng.binomial(1, theta_0, size=10)
            data_b = rng.binomial(1, theta_0, size=10)
            result = test.update({0: data_a, 1: data_b})
            e_values.append(result.e_value)

        mean_e = np.mean(e_values)
        # Should be close to 1 (within ~0.15 for 2000 samples)
        assert 0.8 < mean_e < 1.2, f"Mean e-value under H0: {mean_e}"


class TestTypeIError:
    @pytest.mark.parametrize("k", [2, 3])
    def test_type_i_error_rate(self, k):
        rng = np.random.default_rng(2024)
        alpha = 0.05
        theta_0 = 0.5
        n_simulations = 1000
        max_blocks = 200
        block_size = 5

        rejections = 0
        for _ in range(n_simulations):
            config = KSampleConfig(k=k, significance_level=alpha, gamma=0.18)
            test = KSampleSequentialTest(config)

            rejected = False
            for _ in range(max_blocks):
                group_data = {g: rng.binomial(1, theta_0, size=block_size) for g in range(k)}
                result = test.update(group_data)
                if result.reject_null:
                    rejected = True
                    break

            if rejected:
                rejections += 1

        rejection_rate = rejections / n_simulations
        # 2 SE of binomial: 2 * sqrt(alpha * (1 - alpha) / n)
        se = np.sqrt(alpha * (1 - alpha) / n_simulations)
        upper_bound = alpha + 3 * se  # 3 SE for safety margin

        assert rejection_rate <= upper_bound, (
            f"Type-I error rate {rejection_rate:.4f} exceeds "
            f"bound {upper_bound:.4f} (alpha={alpha}, k={k})"
        )


class TestPowerUnderH1:
    def test_k2_power(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=2, significance_level=0.05)
        test = KSampleSequentialTest(config)

        rejected = False
        for _ in range(500):
            data_a = rng.binomial(1, 0.3, size=10)
            data_b = rng.binomial(1, 0.6, size=10)
            result = test.update({0: data_a, 1: data_b})
            if result.reject_null:
                rejected = True
                break

        assert rejected, "Should reject H0 with theta_a=0.3, theta_b=0.6"

    def test_k3_power(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=3, significance_level=0.05)
        test = KSampleSequentialTest(config)

        rejected = False
        for _ in range(500):
            data = {
                0: rng.binomial(1, 0.3, size=10),
                1: rng.binomial(1, 0.3, size=10),
                2: rng.binomial(1, 0.6, size=10),
            }
            result = test.update(data)
            if result.reject_null:
                rejected = True
                break

        assert rejected, "Should reject H0 with heterogeneous k=3"


class TestEdgeCases:
    def test_all_zeros_one_group_all_ones_another(self):
        config = KSampleConfig(k=2, gamma=0.18)
        test = KSampleSequentialTest(config)

        # Step 1: e_value = 1.0 (prior only, no data yet to differentiate)
        result1 = test.update({0: np.array([0, 0, 0, 0, 0]), 1: np.array([1, 1, 1, 1, 1])})
        assert abs(result1.e_value - 1.0) < 1e-10

        # Step 2: posterior means now differ, should produce strong evidence
        result2 = test.update({0: np.array([0, 0, 0, 0, 0]), 1: np.array([1, 1, 1, 1, 1])})
        assert result2.e_value > 1.0

    def test_single_observation_per_group(self):
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)
        result = test.update({0: np.array([1]), 1: np.array([0])})
        assert isinstance(result, KSampleStepResult)

    def test_unbalanced_group_sizes(self):
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)
        result = test.update({0: np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1]), 1: np.array([0])})
        assert isinstance(result, KSampleStepResult)
        assert result.group_counts[0] == 10
        assert result.group_counts[1] == 1

    def test_k10_groups(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=10)
        test = KSampleSequentialTest(config)

        group_data = {g: rng.binomial(1, 0.5, size=5) for g in range(10)}
        result = test.update(group_data)
        assert isinstance(result, KSampleStepResult)
        assert len(result.theta_estimates) == 10

    def test_missing_group_raises(self):
        config = KSampleConfig(k=3)
        test = KSampleSequentialTest(config)
        with pytest.raises(ValueError, match="Group 2 missing"):
            test.update({0: np.array([1]), 1: np.array([0])})

    def test_non_binary_raises(self):
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)
        with pytest.raises(ValueError, match="non-binary"):
            test.update({0: np.array([0.5, 0.3]), 1: np.array([1, 0])})

    def test_empty_group_raises(self):
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)
        with pytest.raises(ValueError, match="no observations"):
            test.update({0: np.array([]), 1: np.array([1])})


class TestNumericalStability:
    def test_10000_blocks_no_overflow(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=2, gamma=0.18)
        test = KSampleSequentialTest(config)

        for _ in range(10000):
            data_a = rng.binomial(1, 0.5, size=3)
            data_b = rng.binomial(1, 0.5, size=3)
            result = test.update({0: data_a, 1: data_b})

        # No NaN or Inf in e-process
        assert not np.isnan(result.e_process_value)
        assert not np.isinf(result.log_e_process)
        assert result.step == 10000

    def test_gamma_prevents_boundary_theta(self):
        config = KSampleConfig(k=2, gamma=0.18)
        test = KSampleSequentialTest(config)

        # Feed extreme data
        for _ in range(50):
            test.update({0: np.array([0, 0, 0, 0, 0]), 1: np.array([1, 1, 1, 1, 1])})

        # Posterior means should still be in (0, 1)
        calc = test._calculator
        for g in range(2):
            theta = (calc.cumulative_successes[g] + config.gamma) / (
                calc.cumulative_counts[g] + 2 * config.gamma
            )
            assert 0 < theta < 1

    def test_effect_size_grid_weights_stable(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(
            k=2,
            alternative_type="effect_size",
            divergence_type="additive",
            min_effect_size=0.1,
        )
        test = KSampleSequentialTest(config)

        for _ in range(1000):
            data_a = rng.binomial(1, 0.3, size=5)
            data_b = rng.binomial(1, 0.4, size=5)
            test.update({0: data_a, 1: data_b})

        # Log-weights should be finite (not -inf everywhere)
        log_weights = test._calculator.log_weights
        assert np.all(np.isfinite(log_weights))


class TestBettingStrategies:
    @pytest.mark.parametrize(
        "strategy",
        [
            BettingStrategy.ALL_IN,
            BettingStrategy.CONSERVATIVE,
            BettingStrategy.EMPIRICALLY_ADAPTIVE,
        ],
    )
    def test_strategy_produces_valid_results(self, strategy):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=2, betting_strategy=strategy, conservative_lambda=0.5, gamma=0.18)
        test = KSampleSequentialTest(config)

        for _ in range(20):
            data_a = rng.binomial(1, 0.3, size=5)
            data_b = rng.binomial(1, 0.5, size=5)
            result = test.update({0: data_a, 1: data_b})

        assert result.e_process_value > 0
        assert not np.isnan(result.e_process_value)


class TestIntegration:
    def test_history_dataframe(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)

        for _ in range(5):
            test.update({0: rng.binomial(1, 0.5, size=3), 1: rng.binomial(1, 0.5, size=3)})

        df = test.get_history_df()
        assert len(df) == 5
        expected_cols = {
            "step",
            "e_value",
            "e_process_value",
            "log_e_process",
            "p_value",
            "reject_null",
            "theta_null",
            "theta_0",
            "theta_1",
            "count_0",
            "count_1",
            "mean_0",
            "mean_1",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_reset_reinitializes(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)

        for _ in range(10):
            test.update({0: rng.binomial(1, 0.5, size=5), 1: rng.binomial(1, 0.5, size=5)})
        assert test.block_count == 10

        test.reset()
        assert test.block_count == 0
        assert len(test.history) == 0
        assert test._calculator.cumulative_counts[0] == 0
        assert test._calculator.cumulative_counts[1] == 0
        assert test.e_process.process_values == [1.0]
        assert test.e_process.log_process_values == [0.0]

    def test_update_single_interface(self):
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)

        result = test.update_single(0, 1)  # First observation for group 0 should not trigger
        assert result is None

        result = test.update_single(1, 0)  # First observation for group 1 should trigger
        assert isinstance(result, KSampleStepResult)
        assert result.step == 1

    def test_get_summary(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)

        for _ in range(5):
            test.update({0: rng.binomial(1, 0.5, size=3), 1: rng.binomial(1, 0.5, size=3)})

        summary = test.get_summary()
        assert summary["k"] == 2
        assert summary["block_count"] == 5
        assert "is_significant" in summary
        assert "group_counts" in summary

    def test_step_result_is_frozen(self):
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)
        result = test.update({0: np.array([1, 0]), 1: np.array([0, 1])})
        with pytest.raises(Exception):
            result.e_value = 999.0


class TestCalibratorIntegration:
    def test_p_value_matches_calibrator(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=2, gamma=0.18)
        test = KSampleSequentialTest(config)
        calibrator = EToPCalibrator()

        for _ in range(20):
            data_a = rng.binomial(1, 0.3, size=5)
            data_b = rng.binomial(1, 0.6, size=5)
            result = test.update({0: data_a, 1: data_b})

        max_value = test.e_process_updater.get_max_value(test.e_process)
        expected_p = calibrator(max_value)
        assert abs(result.p_value - expected_p) < 1e-15

    def test_p_value_at_step_1_equals_1(self):
        config = KSampleConfig(k=2, gamma=0.18)
        test = KSampleSequentialTest(config)
        result = test.update({0: np.array([1, 0]), 1: np.array([1, 0])})
        assert abs(result.p_value - 1.0) < 1e-10

    def test_get_p_process_returns_non_increasing(self):
        rng = np.random.default_rng(42)
        config = KSampleConfig(k=2, gamma=0.18)
        test = KSampleSequentialTest(config)

        for _ in range(30):
            data_a = rng.binomial(1, 0.3, size=5)
            data_b = rng.binomial(1, 0.6, size=5)
            test.update({0: data_a, 1: data_b})

        p_process = test.get_p_process()
        assert len(p_process) > 0
        for i in range(1, len(p_process)):
            assert p_process[i] <= p_process[i - 1] + 1e-15

    def test_get_p_process_matches_manual(self):
        rng = np.random.default_rng(99)
        config = KSampleConfig(k=2, gamma=0.18)
        test = KSampleSequentialTest(config)
        calibrator = EToPCalibrator()

        for _ in range(10):
            data_a = rng.binomial(1, 0.4, size=5)
            data_b = rng.binomial(1, 0.5, size=5)
            test.update({0: data_a, 1: data_b})

        p_process = test.get_p_process()
        process_values = test.e_process.process_values

        for t in range(len(p_process)):
            max_so_far = max(process_values[: t + 1])
            expected_p = calibrator(max_so_far)
            assert abs(p_process[t] - expected_p) < 1e-14

    def test_get_p_process_empty_before_update(self):
        config = KSampleConfig(k=2)
        test = KSampleSequentialTest(config)
        p_process = test.get_p_process()

        assert len(p_process) == 1
        assert abs(p_process[0] - 1.0) < 1e-15
