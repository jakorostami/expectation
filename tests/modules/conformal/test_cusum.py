import pytest
import numpy as np
from expectation.conformal.cusum import ConformalCUSUM, CUSUMResult, EfficiencyAnalyzer

# TODO: fix test because it is not proper
class TestCUSUM:
    def test_cusum_basic(self):
        detector = ConformalCUSUM(threshold=2.0)
        

        result = detector.update(1.0)
        print(result.statistic)
        assert result.statistic == 1e-10
        assert len(result.alarms) == 0
        
        result = detector.update(2.0)
        assert result.statistic == 2e-10  
