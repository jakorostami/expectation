import pytest
import numpy as np
from expectation.modules.epower import (
    EPowerCalculator, EPowerConfig, EPowerType, EPowerResult
)

class TestEPower:
    @pytest.fixture
    def calculator(self):
        """Fixture for basic EPowerCalculator."""
        return EPowerCalculator()

    def test_basic_epower_calculation(self, calculator):
        """Test basic e-power computation."""
        e_values = np.array([1.5, 2.0, 1.8])
        result = calculator.compute(e_values)
        
        # Test result structure
        assert isinstance(result, EPowerResult)
        assert isinstance(result.e_power, float)
        assert isinstance(result.expected_e_value, float)
        assert result.type == EPowerType.STANDARD
    
    def test_epower_config_validation(self):
        """Test EPowerConfig validation."""
        # Valid config
        config = EPowerConfig(
            type=EPowerType.STANDARD,
            grid_size=50
        )
        assert config.type == EPowerType.STANDARD
        assert config.grid_size == 50
    
    def test_optimized_epower(self):
        """Test e-power with optimization enabled."""
        config = EPowerConfig(optimize_lambda=True)
        calculator = EPowerCalculator(config)
        
        e_values = np.array([1.5, 2.0, 1.8])
        result = calculator.compute(e_values)
        
        assert result.optimal_lambda is not None
        assert 0 <= result.optimal_lambda <= 1
    
    def test_alternative_probabilities(self, calculator):
        """Test e-power with custom alternative probabilities."""
        e_values = np.array([1.5, 2.0, 1.8])
        alt_prob = np.array([0.3, 0.4, 0.3])
        
        result = calculator.compute(e_values, alt_prob)
        assert result.e_power is not None
        assert result.expected_e_value > 0