import pytest
import numpy as np
from expectation.seqtest.sequential_e_testing import (
    SequentialTesting, TestType, AlternativeType, BoundaryConfig, BoundaryType
)
from expectation.modules.hypothesistesting import EValueConfig


class TestEProcessUpdaterIntegration:
    def test_basic_eprocess_tracking(self):
        np.random.seed(42)

        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        data = np.random.normal(0.5, 1.0, 50)
        result = test.update(data)

        current_val = test.e_process_updater.get_current_value(test.e_process)
        max_val = test.e_process_updater.get_max_value(test.e_process)
        is_sig = test.e_process_updater.is_significant(test.e_process)

        assert current_val > 0, "Current e-value should be positive"
        assert max_val >= current_val, "Max e-value should be >= current"
        assert isinstance(is_sig, bool), "Significance should be boolean"

        assert len(test.e_process.values) > 0, "E-values should be tracked"
        assert test.e_process.cumulative_value == current_val

    def test_betting_strategies(self):
        np.random.seed(42)

        strategies = [
            ("all_in", None, None),
            ("conservative", None, 0.3),
            ("empirically_adaptive", 0.5, None)
        ]

        results = {}

        for strategy_name, gamma, conservative_lambda in strategies:
            np.random.seed(42)

            config = EValueConfig(significance_level=0.05)
            test = SequentialTesting(
                test_type=TestType.MEAN,
                null_value=0.0,
                alternative=AlternativeType.TWO_SIDED,
                config=config,
                betting_strategy=strategy_name,
                gamma=gamma,
                conservative_lambda=conservative_lambda
            )

            data = np.random.normal(0.5, 1.0, 50)
            result = test.update(data)

            current_val = test.e_process_updater.get_current_value(test.e_process)
            results[strategy_name] = current_val

            if strategy_name != "all_in":  # all_in doesn't track lambdas
                assert hasattr(test.e_process, 'lambdas'), f"Lambdas should be tracked for {strategy_name}"
                if test.e_process.lambdas:
                    assert all(0 <= l <= 1 for l in test.e_process.lambdas), "Lambdas should be in [0,1]"

        assert len(set(results.values())) > 1, "Different strategies should produce different e-values"

    def test_summary_statistics(self):
        np.random.seed(42)

        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.GREATER,
            config=config
        )

        for _ in range(5):
            batch = np.random.normal(0.3, 1.0, 20)
            test.update(batch)

        summary = test.get_summary()

        assert "current_e_value" in summary
        assert "max_e_value" in summary
        assert "is_significant" in summary
        assert "stopping_time" in summary
        assert "p_value" in summary
        assert "empirical_e_power" in summary
        assert "asymptotic_growth_rate" in summary

        assert summary["current_e_value"] > 0, "E-value should be positive"
        assert summary["max_e_value"] >= summary["current_e_value"]
        assert 0 <= summary["p_value"] <= 1, "P-value should be in [0,1]"
        assert isinstance(summary["is_significant"], bool)

        # Empirical e-power can be negative when e-values < 1
        # This is mathematically correct: mean(log(E_t)) < 0 when E_t < 1
        # Just check it is a valid number
        assert isinstance(summary["empirical_e_power"], (int, float)), "E-power should be a number"
        assert not np.isnan(summary["empirical_e_power"]), "E-power should not be NaN"

    def test_p_process_computation(self):
        np.random.seed(42)

        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        data = np.random.normal(0.0, 1.0, 50)
        test.update(data)

        p_process = test.e_process_updater.compute_p_process(test.e_process)

        assert len(p_process) == len(test.e_process.process_values), "P-process should match process_values length"
        assert len(p_process) == len(test.e_process.values) + 1, "P-process should be len(values) + 1"
        assert all(0 <= p <= 1 for p in p_process), "All p-values should be in [0,1]"

        assert p_process[0] == 1.0, "First p-value should be 1.0"

        for t in range(len(p_process)):
            values_up_to_t = test.e_process.process_values[:t+1]
            expected_p = min(1.0, min(1.0/v for v in values_up_to_t if v > 0))
            assert abs(p_process[t] - expected_p) < 1e-10, f"P-value at t={t} should be minimum inverse"

    def test_stopping_time_detection(self):
        np.random.seed(42)

        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        stopping_detected = False
        for i in range(10):
            batch = np.random.normal(1.0, 0.5, 10)
            result = test.update(batch)

            stopping_time = test.e_process_updater.get_stopping_time(test.e_process)
            if stopping_time is not None:
                stopping_detected = True
                assert stopping_time > 0, "Stopping time should be positive"
                assert stopping_time <= len(test.e_process.values), "Stopping time should be <= current time"
                break

        assert stopping_detected, "Should detect stopping time with strong signal"

    def test_growth_rate_computation(self):
        np.random.seed(123)

        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        for _ in range(10):
            batch = np.random.normal(0.0, 1.0, 20)
            test.update(batch)

        summary = test.get_summary()

        assert len(test.e_process.values) == 10
        assert summary["asymptotic_growth_rate"] is None, "Should return None when len(values) < min_samples"

        for _ in range(10):
            batch = np.random.normal(0.0, 1.0, 20)
            test.update(batch)

        summary = test.get_summary()

        assert len(test.e_process.values) == 20
        growth_rate = summary["asymptotic_growth_rate"]
        assert growth_rate is not None, "Should return a value when len(values) >= min_samples"
        assert isinstance(growth_rate, (int, float)), "Should return a number"
        assert not np.isnan(growth_rate), "Should not return NaN"

        assert isinstance(summary["empirical_e_power"], (int, float))
        assert not np.isnan(summary["empirical_e_power"])

    @pytest.mark.parametrize("test_type,null_value,data_generator", [
        (TestType.MEAN, 0.0, lambda: np.random.normal(0.5, 1.0, 50)),
        (TestType.PROPORTION, 0.5, lambda: np.random.binomial(1, 0.6, 50)),
        (TestType.VARIANCE, 1.0, lambda: np.random.normal(0, np.sqrt(2), 50)),
    ])
    def test_different_test_types(self, test_type, null_value, data_generator):
        np.random.seed(42)

        config = EValueConfig(significance_level=0.05)

        if test_type == TestType.QUANTILE:
            test = SequentialTesting(
                test_type=test_type,
                null_value=null_value,
                quantile=0.5,
                alternative=AlternativeType.TWO_SIDED,
                config=config
            )
        else:
            test = SequentialTesting(
                test_type=test_type,
                null_value=null_value,
                alternative=AlternativeType.TWO_SIDED,
                config=config
            )

        data = data_generator()
        result = test.update(data)

        assert result.e_value > 0, f"E-value should be positive for {test_type}"
        assert test.e_process_updater.get_current_value(test.e_process) == result.e_value

        summary = test.get_summary()
        assert summary["test_type"] == test_type.value
        assert summary["current_e_value"] > 0

    def test_rejection_times_tracking(self):
        np.random.seed(42)

        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        for i in range(10):
            batch = np.random.normal(2.0, 0.5, 20)
            result = test.update(batch)

            if result.reject_null:
                assert len(test.rejection_times) > 0, "Rejection times should be tracked"
                assert test.rejection_times[0] == test.e_process_updater.get_stopping_time(test.e_process)
                break
        else:
            pytest.fail("Should have rejected null with strong signal")

    def test_with_boundary_config(self):
        np.random.seed(42)

        boundary_configs = [
            BoundaryConfig(boundary_type=BoundaryType.NORMAL, v_opt=1.0, alpha_opt=0.05),
            BoundaryConfig(boundary_type=BoundaryType.GAMMA_EXPONENTIAL, v_opt=1.0, alpha_opt=0.05, c=1.0),
            BoundaryConfig(boundary_type=BoundaryType.POLY_STITCHING, v_min=0.5, s=1.4, eta=2.0)
        ]

        for boundary_config in boundary_configs:
            config = EValueConfig(significance_level=0.05)
            test = SequentialTesting(
                test_type=TestType.MEAN,
                null_value=0.0,
                alternative=AlternativeType.TWO_SIDED,
                config=config,
                boundary_config=boundary_config
            )

            data = np.random.normal(0.5, 1.0, 50)
            result = test.update(data)

            assert result.confidence_bounds is not None, f"Should have bounds for {boundary_config.boundary_type}"

            assert test.e_process_updater.get_current_value(test.e_process) > 0

            summary = test.get_summary()
            assert "current_e_value" in summary
            assert "empirical_e_power" in summary