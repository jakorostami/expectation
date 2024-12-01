import pytest
import numpy as np
from expectation.conformal.cusum import ConformalCUSUM, CUSUMResult, EfficiencyAnalyzer

class TestCUSUM:
    @pytest.fixture
    def detector(self):
        """Fixture providing basic CUSUM detector."""
        return ConformalCUSUM(threshold=3.0)

    def test_initialization(self):
        """Test proper initialization."""
        with pytest.raises(ValueError):
            ConformalCUSUM(threshold=0.5)  # Should require threshold > 1
            
        detector = ConformalCUSUM(threshold=3.0)
        assert detector._last_alarm == 0
        assert detector._cusum_stat == 0.0
        
    def test_basic_update(self, detector):
        """Test basic update without alarm."""
        result = detector.update(1.5)
        assert isinstance(result, CUSUMResult)
        assert len(result.alarms) == 0
        assert result.statistic == 1.5  # Should accumulate directly
        assert result.n_alarms == 0
        
        # Verify stat history
        assert len(result.all_stats) == 1
        assert result.all_stats[0] == 1.5
        
    def test_stat_accumulation(self, detector):
        """Test proper statistic accumulation."""
        # Update with sequence
        result = detector.update(1.0)
        result = detector.update(1.0)
        
        # Should multiply e-values
        assert np.isclose(result.statistic, 2.0, rtol=1e-10)
        assert len(result.alarms) == 0
        
    def test_alarm_generation(self):
        """Test alarm generation with controlled sequence."""
        detector = ConformalCUSUM(threshold=2.0)
        
        # Generate sequence exceeding threshold
        result = detector.update(1.5)  # stat = 1.5
        result = detector.update(1.5)  # stat = 1.5 * 1.5 = 2.25 > threshold
        
        assert len(result.alarms) == 1
        assert result.n_alarms == 1
        assert result.statistic == 0.0  # Should reset after alarm
        
    def test_multiple_alarms(self):
        """Test multiple alarm generations."""
        detector = ConformalCUSUM(threshold=2.0)
        
        # First alarm sequence
        detector.update(1.5)
        result = detector.update(1.5)  # Should trigger first alarm
        assert len(result.alarms) == 1
        
        # Second alarm sequence
        detector.update(1.5)
        result = detector.update(1.5)  # Should trigger second alarm
        assert len(result.alarms) == 2
        assert result.alarms[1] > result.alarms[0]
        
    def test_threshold_behavior(self):
        """Test behavior with different thresholds."""
        low = ConformalCUSUM(threshold=2.0)
        high = ConformalCUSUM(threshold=4.0)
        
        # Update both detectors with same sequence
        for _ in range(3):
            low_result = low.update(1.5)
            high_result = high.update(1.5)
        
        # Low threshold should alarm, high shouldn't
        assert low_result.n_alarms >= 1
        assert high_result.n_alarms == 0