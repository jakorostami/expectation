import pytest
import numpy as np
from expectation.conformal.adaptivethreshold import AdaptiveThresholdHandler


class TestAdaptiveThreshold:
    def test_threshold_adaptation(self):
        """Test adaptive threshold behavior."""
        handler = AdaptiveThresholdHandler(target_false_alarm_rate=0.05)
        
        # Test adaptation to high false alarm rate
        handler.record_alarm(True)
        handler.record_alarm(True)
        new_threshold = handler.update(handler.get_current_rate())
        assert new_threshold > handler.threshold
        
        # Test reset
        handler.reset()
        assert len(handler.alarm_history) == 0
        assert handler.threshold == 10.0