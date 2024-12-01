import pytest
import numpy as np
from expectation.conformal.cusum import ConformalCUSUM, CUSUMResult, EfficiencyAnalyzer

class TestCUSUM:
    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data."""
        np.random.seed(42)
        return np.random.normal(0, 1, 100)
    
    def test_initialization(self):
        """Test proper initialization."""
        with pytest.raises(ValueError):
            ConformalCUSUM(threshold=0.5)  # Should require threshold > 1
            
        detector = ConformalCUSUM(threshold=5.0)
        assert len(detector._stats_history) == 0
        assert len(detector._alarms) == 0
    
    def test_cusum_detector(self, sample_data):
        """Test CUSUM detector functionality."""
        detector = ConformalCUSUM(threshold=2.0)  # Lower threshold for testing
        
        # First, accumulate some statistics
        result = detector.update(1.5)
        assert isinstance(result, CUSUMResult)
        assert len(result.alarms) == 0
        assert result.statistic > 0
        
        # Continue accumulating until threshold
        result = detector.update(1.5)
        current_stat = result.statistic
        
        # Add value that should trigger alarm
        result = detector.update(2.0)
        assert len(result.alarms) == 1
        assert result.n_alarms == 1
        assert result.statistic < current_stat  # Should reset after alarm
        
        # Verify statistics history
        assert len(result.all_stats) == 3
        assert all(stat >= 0 for stat in result.all_stats)

    def test_sequential_alarms(self):
        """Test multiple sequential alarms."""
        detector = ConformalCUSUM(threshold=2.0)
        
        # Generate sequence that should trigger multiple alarms
        sequence = [1.5, 1.5, 2.0,  # First alarm
                   1.5, 1.5, 2.0]   # Second alarm
        
        results = []
        for value in sequence:
            results.append(detector.update(value))
        
        # Check alarm pattern
        assert len(results[-1].alarms) == 2
        assert results[-1].n_alarms == 2
        
        # Verify alarm times
        assert results[-1].alarms[0] < results[-1].alarms[1]
        
    def test_threshold_behavior(self):
        """Test behavior with different thresholds."""
        low_detector = ConformalCUSUM(threshold=1.5)
        high_detector = ConformalCUSUM(threshold=5.0)
        
        # Same sequence should trigger alarm in low but not high
        sequence = [1.2, 1.2, 1.2]
        
        for value in sequence:
            low_result = low_detector.update(value)
            high_result = high_detector.update(value)
        
        assert low_result.n_alarms > 0
        assert high_result.n_alarms == 0