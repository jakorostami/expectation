import pytest
import numpy as np
from expectation.conformal.cusum import ConformalCUSUM, CUSUMResult, EfficiencyAnalyzer

class TestCUSUM:
    def test_cusum_basic(self):
        """Test basic CUSUM functionality."""
        detector = ConformalCUSUM(threshold=2.0)
        
        # Should not trigger alarm
        result = detector.update(1.0)
        assert result.statistic == 1.0
        assert len(result.alarms) == 0
        
        # Should trigger alarm (1.0 * 2.0 = 2.0)
        result = detector.update(2.0)
        assert result.statistic == 0  # Reset after alarm
        assert len(result.alarms) == 1
        assert result.n_alarms == 1
