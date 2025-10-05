import pytest
import numpy as np
from expectation.seqtest.sequential_e_testing import (
    SequentialTesting, TestType, AlternativeType, BoundaryType, BoundaryConfig, SequentialTestResult
)
from expectation.modules.hypothesistesting import EValueConfig


class TestSequentialTesting:
    def test_initialization_mean_test(self):
        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.GREATER,
            config=config
        )

        assert test.test_type == TestType.MEAN
        assert test.alternative == AlternativeType.GREATER
        assert test.null_value == 0.0
        assert test.data_count == 0
        assert test.data_sum == 0.0
        assert test.e_process is not None
        assert test.e_process_updater is not None
        assert test.boundary_config is not None
        assert test.boundary_config.boundary_type == BoundaryType.NORMAL  # Default for mean

    def test_initialization_proportion_test(self):
        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.PROPORTION,
            null_value=0.5,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        assert test.test_type == TestType.PROPORTION
        assert test.null_value == 0.5
        assert test.boundary_config.boundary_type == BoundaryType.BETA_BINOMIAL  # Default for proportion

    def test_initialization_variance_test(self):
        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.VARIANCE,
            null_value=1.0,
            alternative=AlternativeType.GREATER,
            config=config
        )

        assert test.test_type == TestType.VARIANCE
        assert test.null_value == 1.0
        assert test.boundary_config.boundary_type == BoundaryType.GAMMA_EXPONENTIAL  # Default for variance

    def test_initialization_quantile_test(self):
        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.QUANTILE,
            null_value=0.0,
            quantile=0.5,  ## median
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        assert test.test_type == TestType.QUANTILE
        assert test.quantile == 0.5
        assert test.null_value == 0.0
        assert test.boundary_config.boundary_type == BoundaryType.DOUBLE_STITCHING  # Default for quantile

    def test_invalid_parameters(self):
        config = EValueConfig(significance_level=0.05)

        with pytest.raises(ValueError):
            SequentialTesting(
                test_type="invalid_type",
                null_value=0.0,
                config=config
            )

        with pytest.raises(ValueError):
            SequentialTesting(
                test_type=TestType.MEAN,
                null_value=0.0,
                alternative="invalid_alternative",
                config=config
            )

        with pytest.raises(ValueError):
            SequentialTesting(
                test_type=TestType.QUANTILE,
                null_value=0.0,
                config=config
            )

        with pytest.raises(ValueError):
            SequentialTesting(
                test_type=TestType.PROPORTION,
                null_value=1.5,  # should be in (0,1)
                config=config
            )

    def test_mean_test_update(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        data = np.random.normal(0.0, 1.0, 50)
        result = test.update(data)

        assert isinstance(result, SequentialTestResult)
        assert result.e_value > 0
        assert result.sample_size == 50
        assert result.test_type == TestType.MEAN
        assert result.alternative == AlternativeType.TWO_SIDED
        assert result.p_value is not None
        assert 0 <= result.p_value <= 1
        assert result.confidence_bounds is not None
        assert len(result.confidence_bounds) == 2

        # Check e-process tracking
        assert len(test.e_process.values) == 1
        assert test.e_process.values[0] == result.e_value

    def test_proportion_test_update(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.PROPORTION,
            null_value=0.5,
            alternative=AlternativeType.GREATER,
            config=config
        )

        data = np.random.binomial(1, 0.5, 100)
        result = test.update(data)

        assert result.e_value > 0
        assert result.sample_size == 100
        assert result.confidence_bounds is not None

        with pytest.raises(ValueError, match="binary"):
            test.update([0.5, 1.5, 2.0])

    def test_sequential_updates(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.GREATER,
            config=config
        )

        batch_sizes = [10, 20, 30]
        total_samples = 0

        for batch_size in batch_sizes:
            data = np.random.normal(0.0, 1.0, batch_size)
            result = test.update(data)
            total_samples += batch_size

            assert result.sample_size == total_samples
            assert len(test.e_process.values) == len(batch_sizes[:batch_sizes.index(batch_size) + 1])

        assert len(test.e_process.values) == len(batch_sizes)

    def test_mean_confidence_bounds(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        boundary_config = BoundaryConfig(
            boundary_type=BoundaryType.NORMAL,
            v_opt=1.0,
            alpha_opt=0.05
        )

        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config,
            boundary_config=boundary_config
        )

        true_mean = 0.5
        data = np.random.normal(true_mean, 1.0, 100)
        result = test.update(data)

        lower, upper = result.confidence_bounds
        sample_mean = test.data_sum / test.data_count

        assert lower <= sample_mean <= upper

        center = (lower + upper) / 2
        assert abs(center - sample_mean) < 0.01

    def test_proportion_confidence_bounds(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        boundary_config = BoundaryConfig(
            boundary_type=BoundaryType.BETA_BINOMIAL,
            v_opt=0.25,
            alpha_opt=0.05,
            g=0.5,
            h=0.5
        )

        test = SequentialTesting(
            test_type=TestType.PROPORTION,
            null_value=0.5,
            alternative=AlternativeType.TWO_SIDED,
            config=config,
            boundary_config=boundary_config
        )

        true_p = 0.6
        data = np.random.binomial(1, true_p, 200)
        result = test.update(data)

        lower, upper = result.confidence_bounds
        sample_proportion = test.data_sum / test.data_count

        assert lower <= sample_proportion <= upper

        assert 0 <= lower <= 1
        assert 0 <= upper <= 1

    def test_different_boundary_types(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        boundary_types = [
            (BoundaryType.NORMAL, {"v_opt": 1.0, "alpha_opt": 0.05}),
            (BoundaryType.GAMMA_EXPONENTIAL, {"v_opt": 1.0, "alpha_opt": 0.05, "c": 1.0}),
            (BoundaryType.POLY_STITCHING, {"v_min": 0.5, "s": 1.4, "eta": 2.0})
        ]

        data = np.random.normal(0.5, 1.0, 100)
        results = {}

        for boundary_type, params in boundary_types:
            boundary_config = BoundaryConfig(boundary_type=boundary_type, **params)

            test = SequentialTesting(
                test_type=TestType.MEAN,
                null_value=0.0,
                alternative=AlternativeType.TWO_SIDED,
                config=config,
                boundary_config=boundary_config
            )

            result = test.update(data)
            results[boundary_type] = result.confidence_bounds

            assert result.confidence_bounds is not None
            lower, upper = result.confidence_bounds
            assert lower < upper

            sample_mean = test.data_sum / test.data_count
            assert lower <= sample_mean <= upper

        widths = {k: v[1] - v[0] for k, v in results.items()}
        assert len(set(widths.values())) > 1  # At least some should differ

    def test_betting_strategies(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        strategies = [
            ("all_in", None, None),
            ("conservative", None, 0.3),
            ("empirically_adaptive", 0.5, None)
        ]

        results = {}

        for strategy_name, gamma, conservative_lambda in strategies:
            np.random.seed(42)

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

            results[strategy_name] = result.e_value

            if strategy_name != "all_in":
                assert hasattr(test.e_process, 'lambdas')
                if test.e_process.lambdas:
                    assert all(0 <= l <= 1 for l in test.e_process.lambdas)

        assert len(set(results.values())) > 1

    def test_optimal_lambda_tracking(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config,
            betting_strategy="conservative",
            conservative_lambda=0.3
        )

        for _ in range(5):
            data = np.random.normal(0.3, 1.0, 20)
            result = test.update(data)

            if result.optimal_lambda is not None:
                assert 0 <= result.optimal_lambda <= 1
                assert len(test.lambda_history) > 0

    def test_eprocess_updater_integration(self):
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

        assert current_val == result.e_value
        assert max_val >= current_val
        assert isinstance(is_sig, bool)

        expected_p = min(1.0, 1.0/current_val) if current_val > 0 else 1.0
        assert abs(result.p_value - expected_p) < 1e-10

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
        for _ in range(10):
            batch = np.random.normal(2.0, 0.5, 10)
            result = test.update(batch)

            if result.reject_null:
                stopping_time = test.e_process_updater.get_stopping_time(test.e_process)
                assert stopping_time is not None
                assert stopping_time > 0
                assert stopping_time <= len(test.e_process.values)
                stopping_detected = True
                break

        assert stopping_detected, "Should detect stopping with strong signal"

    def test_rejection_times_tracking(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        for _ in range(10):
            batch = np.random.normal(3.0, 0.5, 20)
            result = test.update(batch)

            if result.reject_null:
                assert len(test.rejection_times) > 0
                assert test.rejection_times[0] == test.e_process_updater.get_stopping_time(test.e_process)
                break
        else:
            pytest.fail("Should have rejected null with strong signal")

    def test_get_summary(self):
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

        required_fields = [
            "test_type", "null_value", "alternative",
            "current_e_value", "max_e_value", "is_significant",
            "stopping_time", "p_value", "sample_size",
            "empirical_e_power", "asymptotic_growth_rate"
        ]

        for field in required_fields:
            assert field in summary, f"Missing field: {field}"

        assert summary["test_type"] == TestType.MEAN.value
        assert summary["null_value"] == 0.0
        assert summary["alternative"] == AlternativeType.GREATER.value
        assert summary["current_e_value"] > 0
        assert summary["max_e_value"] >= summary["current_e_value"]
        assert 0 <= summary["p_value"] <= 1
        assert summary["sample_size"] == 100
        assert isinstance(summary["is_significant"], bool)

    def test_e_process_tracking(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        batch_sizes = [10, 20, 30]
        raw_e_values = []
        cumulative_e_values = []
        for batch_size in batch_sizes:
            data = np.random.normal(0.0, 1.0, batch_size)
            result = test.update(data)
            cumulative_e_values.append(result.e_value)

        assert len(test.e_process.values) == len(batch_sizes)
        raw_e_values = test.e_process.values

        assert len(test.e_process.process_values) == len(batch_sizes) + 1
        assert test.e_process.process_values[0] == 1.0  # Initial value
        
        expected_product = 1.0
        for i, raw_e in enumerate(raw_e_values):
            expected_product *= raw_e
            assert abs(test.e_process.process_values[i+1] - expected_product) < 1e-10
            assert abs(cumulative_e_values[i] - expected_product) < 1e-10

        assert abs(test.e_process.cumulative_value - expected_product) < 1e-10

    def test_reset_functionality(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config
        )

        data = np.random.normal(0.5, 1.0, 50)
        test.update(data)

        assert test.data_count > 0
        assert len(test.e_process.values) > 0

        test.reset()

        assert test.data_count == 0
        assert test.data_sum == 0.0
        assert test.data_sum_squares == 0.0
        assert len(test.e_process.values) == 0
        assert test.e_process.cumulative_value == 1.0
        assert len(test.rejection_times) == 0

    def test_empty_data_error(self):
        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            config=config
        )

        with pytest.raises(ValueError, match="No data provided"):
            test.update([])

    def test_variance_test_with_single_observation(self):
        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.VARIANCE,
            null_value=1.0,
            config=config
        )

        result = test.update([1.0])
        assert result.e_value == 1.0

    def test_known_variance_option(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        known_var = 2.0
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config,
            known_variance=known_var
        )

        data = np.random.normal(0.0, np.sqrt(known_var), 50)
        _ = test.update(data)

        assert test.intrinsic_time == test.data_count * known_var

    def test_variance_bound_option(self):
        np.random.seed(42)
        config = EValueConfig(significance_level=0.05)

        var_bound = 0.5
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=AlternativeType.TWO_SIDED,
            config=config,
            variance_bound=var_bound,
            use_empirical_variance=True,
            min_samples_for_update=1
        )

        # Generate data with large variance
        data = np.random.normal(0.0, 2.0, 100)
        _ = test.update(data)

        assert test.data_count == 100
        assert hasattr(test, 'empirical_variance')

    @pytest.mark.parametrize("test_type,null_value,data_generator", [
        (TestType.MEAN, 0.0, lambda: np.random.normal(0.5, 1.0, 50)),
        (TestType.PROPORTION, 0.5, lambda: np.random.binomial(1, 0.6, 50)),
        (TestType.VARIANCE, 1.0, lambda: np.random.normal(0, np.sqrt(2), 50)),
    ])
    def test_all_test_types(self, test_type, null_value, data_generator):
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

        assert result.e_value > 0
        assert result.test_type == test_type
        assert result.confidence_bounds is not None

        summary = test.get_summary()
        assert summary["test_type"] == test_type.value

    @pytest.mark.parametrize("alternative", [
        AlternativeType.TWO_SIDED,
        AlternativeType.GREATER,
        AlternativeType.LESS
    ])
    def test_all_alternatives(self, alternative):
        np.random.seed(42)

        config = EValueConfig(significance_level=0.05)
        test = SequentialTesting(
            test_type=TestType.MEAN,
            null_value=0.0,
            alternative=alternative,
            config=config
        )

        if alternative == AlternativeType.GREATER:
            data = np.random.normal(0.5, 1.0, 50)  # Mean > null
        elif alternative == AlternativeType.LESS:
            data = np.random.normal(-0.5, 1.0, 50)  # Mean < null
        else:
            data = np.random.normal(0.0, 1.0, 50)  # Mean = null

        result = test.update(data)

        assert result.e_value > 0
        assert result.alternative == alternative