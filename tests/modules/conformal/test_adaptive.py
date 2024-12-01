import pytest
import numpy as np
from expectation.conformal.adaptivethreshold import AdaptiveThresholdHandler


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.normal(0, 1, 100)

class TestAdaptiveThreshold:
    def test_threshold_adaptation(self):
        handler = AdaptiveThresholdHandler(target_false_alarm_rate=0.05)
        initial_threshold = handler.threshold
        
        # Test adaptation to high false alarm rate
        handler.record_alarm(True)
        handler.record_alarm(True)
        new_threshold = handler.update(handler.get_current_rate())
        
        # Compare with initial threshold instead of current
        assert new_threshold > initial_threshold
        
        # Test reset
        handler.reset()
        assert len(handler.alarm_history) == 0
        assert handler.threshold == 10.0
    
    def test_threshold_bounds(self):
        handler = AdaptiveThresholdHandler(
            target_false_alarm_rate=0.05,
            min_threshold=5.0,
            max_threshold=15.0
        )
        
        # Force threshold to approach bounds
        # High false alarm rate should increase threshold
        for _ in range(5):
            handler.record_alarm(True)
            handler.update(handler.get_current_rate())
        assert handler.threshold <= handler.max_threshold
        
        # Reset and test lower bound
        handler.reset()
        # Low false alarm rate should decrease threshold
        for _ in range(5):
            handler.record_alarm(False)
            handler.update(handler.get_current_rate())
        assert handler.threshold >= handler.min_threshold