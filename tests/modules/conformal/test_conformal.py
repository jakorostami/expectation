import pytest
import numpy as np
from expectation.conformal.conformal import ConformalEValue, ConformalEPseudomartingale, TruncatedEPseudomartingale

@pytest.fixture
def sample_data():
    np.random.seed(42)
    return np.random.normal(0, 1, 100)

class TestConformal:
    def test_conformal_evalue(self, sample_data):
        evaluator = ConformalEValue(nonconformity_type="normal", is_one_sided=True)
        
        # Test updates and score computation
        score = evaluator.update(sample_data[:10])
        assert score > 0 and not np.isinf(score)
        
        # Test invalid initialization
        with pytest.raises(ValueError):
            ConformalEValue(nonconformity_type="invalid")
    
    def test_pseudomartingale(self):
        mart = ConformalEPseudomartingale(initial_capital=1.0)
        
        # Test capital evolution
        capital, max_cap = mart.update(2.0)
        assert capital == 2.0 and max_cap == 2.0
        
        capital, max_cap = mart.update(0.5)
        assert capital == 1.0 and max_cap == 2.0
    
    def test_truncated_martingale(self):
        mart = TruncatedEPseudomartingale(initial_capital=1.0, min_capital=0.1)
        
        # Test truncation
        capital, _ = mart.update(0.05)
        assert capital == 0.1