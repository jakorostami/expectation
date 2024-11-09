import pytest
import numpy as np
from expectation.modules.quantiletest import QuantileABTest
from expectation.modules.orderstatistics import StaticOrderStatistics

@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    np.random.seed(42)
    return {
        'arm1': sorted(np.random.normal(0, 1, 100)),
        'arm2': sorted(np.random.normal(0.5, 1, 100))
    }

@pytest.fixture
def quantile_test(sample_data):
    """Fixture providing initialized quantile test."""
    return QuantileABTest(
        quantile_p=0.5,  # median test
        t_opt=100,
        alpha_opt=0.05,
        arm1_os=StaticOrderStatistics(sample_data['arm1']),
        arm2_os=StaticOrderStatistics(sample_data['arm2'])
    )

class TestQuantileABTest:
    def test_initialization(self, sample_data):
        """Test proper initialization of QuantileABTest."""
        # Valid initialization
        test = QuantileABTest(
            quantile_p=0.5,
            t_opt=100,
            alpha_opt=0.05,
            arm1_os=StaticOrderStatistics(sample_data['arm1']),
            arm2_os=StaticOrderStatistics(sample_data['arm2'])
        )
        assert test.quantile_p == 0.5
        
        # Invalid quantile
        with pytest.raises(AssertionError):
            QuantileABTest(
                quantile_p=1.5,  # invalid probability
                t_opt=100,
                alpha_opt=0.05,
                arm1_os=StaticOrderStatistics(sample_data['arm1']),
                arm2_os=StaticOrderStatistics(sample_data['arm2'])
            )
    
    def test_p_value_properties(self, quantile_test):
        """Test properties of computed p-values."""
        p_value = quantile_test.p_value()
        
        # P-value should be between 0 and 1
        assert 0 <= p_value <= 1
        
        # P-value should be smaller when there's a clear difference
        # Create test with more separated data
        np.random.seed(42)
        data1 = sorted(np.random.normal(0, 1, 100))
        data2 = sorted(np.random.normal(2, 1, 100))  # larger separation
        
        separated_test = QuantileABTest(
            quantile_p=0.5,
            t_opt=100,
            alpha_opt=0.05,
            arm1_os=StaticOrderStatistics(data1),
            arm2_os=StaticOrderStatistics(data2)
        )
        
        assert separated_test.p_value() < p_value
    
    def test_log_superMG_properties(self, quantile_test):
        """Test properties of log supermartingale."""
        log_bound = quantile_test.log_superMG_lower_bound()
        
        # Should be finite
        assert not np.isinf(log_bound)
        assert not np.isnan(log_bound)
        
        # Should relate to p-value correctly
        p_value = quantile_test.p_value()
        assert np.abs(p_value - min(1.0, np.exp(-log_bound))) < 1e-10
    
    def test_arm_specific_calculations(self, quantile_test):
        """Test arm-specific calculations."""
        # Test G function properties
        G1, x1_lower, x1_upper = quantile_test.get_G_fn(1)
        G2, x2_lower, x2_upper = quantile_test.get_G_fn(2)
        
        # Bounds should be ordered correctly
        assert x1_lower <= x1_upper
        assert x2_lower <= x2_upper
        
        # G function should be callable and return finite values
        test_points = np.linspace(x1_lower, x1_upper, 10)
        for x in test_points:
            g1_value = G1(x)
            assert not np.isinf(g1_value)
            assert not np.isnan(g1_value)
    
    @pytest.mark.parametrize("quantile", [0.25, 0.5, 0.75])
    def test_different_quantiles(self, sample_data, quantile):
        """Test behavior with different quantiles."""
        test = QuantileABTest(
            quantile_p=quantile,
            t_opt=100,
            alpha_opt=0.05,
            arm1_os=StaticOrderStatistics(sample_data['arm1']),
            arm2_os=StaticOrderStatistics(sample_data['arm2'])
        )
        
        p_value = test.p_value()
        assert 0 <= p_value <= 1
    
    def test_edge_cases(self, sample_data):
        """Test edge cases and extreme scenarios."""
        # Test with identical data
        identical_test = QuantileABTest(
            quantile_p=0.5,
            t_opt=100,
            alpha_opt=0.05,
            arm1_os=StaticOrderStatistics(sample_data['arm1']),
            arm2_os=StaticOrderStatistics(sample_data['arm1'])  # same data
        )
        
        # P-value should be large (null hypothesis is true)
        assert identical_test.p_value() > 0.05
        
        # Test with very different data
        large_diff_data = np.array(sample_data['arm1']) + 1000
        different_test = QuantileABTest(
            quantile_p=0.5,
            t_opt=100,
            alpha_opt=0.05,
            arm1_os=StaticOrderStatistics(sample_data['arm1']),
            arm2_os=StaticOrderStatistics(large_diff_data)
        )
        
        # P-value should be very small
        assert different_test.p_value() < 0.05