import numpy as np
import pytest

from expectation.modules.hypothesistesting import (
    EProcess,
    EValue,
    EValueConfig,
    Hypothesis,
    HypothesisType,
    LikelihoodRatioEValue,
)


@pytest.fixture
def basic_config():
    return EValueConfig(significance_level=0.05, allow_infinite=False)


@pytest.fixture
def normal_evaluator():
    null_hypothesis = Hypothesis(
        name="Standard Normal", description="N(0,1) distribution", type=HypothesisType.SIMPLE
    )

    def null_density(x):
        return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

    def alt_density(x):
        return np.exp(-0.5 * (x - 1) * (x - 1)) / np.sqrt(2 * np.pi)

    return LikelihoodRatioEValue(
        null_hypothesis=null_hypothesis, null_density=null_density, alt_density=alt_density
    )


class TestEValue:
    def test_config_validation(self):
        config = EValueConfig(significance_level=0.05, allow_infinite=False)
        assert config.significance_level == 0.05

        with pytest.raises(ValueError):
            EValueConfig(significance_level=1.5)
        with pytest.raises(ValueError):
            EValueConfig(significance_level=-0.1)

    def test_likelihood_ratio_properties(self, normal_evaluator):
        np.random.seed(42)
        null_data = np.random.normal(0, 1, 100)
        null_result = normal_evaluator.test(null_data)

        alt_data = np.random.normal(1, 1, 100)
        alt_result = normal_evaluator.test(alt_data)

        assert null_result.value >= 0
        assert alt_result.value >= 0

        assert alt_result.value > null_result.value

    def test_e_process_properties(self, normal_evaluator):
        from expectation.modules.eprocessupdater import EProcessUpdater

        config = EValueConfig(significance_level=0.05, allow_infinite=False)
        e_process = EProcess(config=config)  # Use config directly, not fixture
        updater = EProcessUpdater(config)

        np.random.seed(42)
        data_batches = np.array_split(np.random.normal(0, 1, 100), 10)

        cumulative_values = []
        for batch in data_batches:
            result = normal_evaluator.test(batch)
            updater.update(e_process, result.value)
            cumulative_values.append(e_process.cumulative_value)

        assert all(v >= 0 for v in cumulative_values)  # Non-negativity
        assert len(cumulative_values) == len(data_batches)  # Correct length

        assert updater.is_significant(e_process, 0.05) == (e_process.cumulative_value >= 20)


class TestHypothesis:
    def test_hypothesis_creation(self):
        h = Hypothesis(name="Test", description="Test hypothesis", type=HypothesisType.SIMPLE)
        assert h.name == "Test"
        assert h.type == HypothesisType.SIMPLE

        with pytest.raises(ValueError):
            Hypothesis(name="Test", type="invalid")
