import pytest
import numpy as np
from expectation.conformal.cusum import ConformalCUSUM, CUSUMResult, EfficiencyAnalyzer


@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.normal(0, 1, 100)

class TestCUSUM:
    def test_cusum_detector(self, sample_data):
        detector = ConformalCUSUM(threshold=5.0)
        
        # Test update and alarm generation
        result = detector.update(3.0)
        assert isinstance(result, CUSUMResult)
        assert len(result.alarms) == 0
        
        # Generate alarm
        result = detector.update(10.0)
        assert len(result.alarms) == 1
        assert result.statistic == 0.0  # Reset after alarm
    
    def test_efficiency_analyzer(self):
        analyzer = EfficiencyAnalyzer()
        detector = ConformalCUSUM(threshold=10.0)
        
        metrics = analyzer.compute_efficiency_metrics(
            detector, n_pre=50, n_post=50, n_trials=5
        )
        
        assert isinstance(metrics, dict)
        assert 0 <= metrics['false_alarm_rate'] <= 1
