import pytest
import numpy as np
import pandas as pd
from expectation.seqtest.sequential_e_testing import SequentialTest, TestType, AlternativeType

@pytest.fixture
def mean_test():
    """Fixture for mean test."""
    return SequentialTest(
        test_type="mean",
        null_value=0,
        alternative="greater"
    )

class TestSequentialTest:
    def test_initialization(self):
        """Test proper initialization of SequentialTest."""
        test = SequentialTest(
            test_type="mean",
            null_value=0,
            alternative="greater"
        )
        
        assert test.test_type == TestType.MEAN
        assert test.alternative == AlternativeType.GREATER
        assert test.null_value == 0
        assert len(test.history) == 0
        
    def test_invalid_parameters(self):
        """Test parameter validation."""
        # Invalid test type
        with pytest.raises(ValueError):
            SequentialTest(test_type="invalid", null_value=0)
            
        # Invalid alternative
        with pytest.raises(ValueError):
            SequentialTest(test_type="mean", null_value=0, alternative="invalid")
            
        # Missing quantile for quantile test
        with pytest.raises(ValueError):
            SequentialTest(test_type="quantile", null_value=0)
            
        # Invalid proportion value
        with pytest.raises(ValueError):
            SequentialTest(test_type="proportion", null_value=1.5)
    
    def test_mean_test_behavior(self, mean_test):
        """Test behavior of mean test under null and alternative."""
        # Under null hypothesis (μ = 0)
        np.random.seed(42)
        null_data = np.random.normal(0, 1, 100)
        
        for i in range(0, len(null_data), 10):
            result = mean_test.update(null_data[i:i+10])
            assert result.e_value >= 0  # E-values are non-negative
            
        # Reset test
        mean_test.reset()
        
        # Under alternative (μ = 1)
        alt_data = np.random.normal(1, 1, 100)
        e_values = []
        
        for i in range(0, len(alt_data), 10):
            result = mean_test.update(alt_data[i:i+10])
            e_values.append(result.e_value)
        
        # E-values should tend to be larger under alternative
        assert np.mean(e_values) > 1
    
    def test_history_tracking(self, mean_test):
        """Test history tracking functionality."""
        data_batches = [
            np.array([1.0, 2.0]),
            np.array([0.5, 1.5]),
            np.array([1.2, 0.8])
        ]
        
        for batch in data_batches:
            mean_test.update(batch)
        
        history_df = mean_test.get_history_df()
        
        assert len(history_df) == len(data_batches)
        assert all(col in history_df.columns for col in 
                  ['step', 'observations', 'eValue', 'cumulativeEValue'])
        
    def test_reset_functionality(self, mean_test):
        """Test reset functionality."""
        data = np.array([1.0, 2.0])
        mean_test.update(data)
        
        initial_history_len = len(mean_test.history)
        assert initial_history_len > 0
        assert mean_test.n_samples > 0
        
        mean_test.reset()
        
        # History should remain unchanged after reset
        assert len(mean_test.history) == initial_history_len
        # But other state should be reset
        assert mean_test.n_samples == 0
        assert mean_test.e_process.cumulative_value == 1.0
    
    def test_e_value_consistency(self, mean_test):
        """Test consistency properties of e-values."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        e_values = []
        
        for i in range(0, len(data), 10):
            result = mean_test.update(data[i:i+10])
            e_values.append(result.e_value)
        
        # Test properties of e-values
        assert all(e >= 0 for e in e_values)  # Non-negativity
        assert np.mean(e_values) <= 2  # Approximately bounded mean under null