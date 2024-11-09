import pytest
import numpy as np
from expectation.modules.martingales import (
    BetaBinomialMixture, OneSidedNormalMixture, 
    TwoSidedNormalMixture, GammaExponentialMixture
)

class TestMixtureMartingales:
    def test_normal_mixture(self):
        """Test normal mixture supermartingale properties."""
        # Test one-sided
        one_sided = OneSidedNormalMixture(v_opt=1.0, alpha_opt=0.05)
        
        # Test two-sided
        two_sided = TwoSidedNormalMixture(v_opt=1.0, alpha_opt=0.05)
        
        # Generate some data
        np.random.seed(42)
        s_values = np.random.normal(0, 1, 100)
        v_values = np.ones_like(s_values)
        
        # Test log superMG values
        one_sided_values = [one_sided.log_superMG(s, v) 
                           for s, v in zip(s_values, v_values)]
        two_sided_values = [two_sided.log_superMG(s, v) 
                           for s, v in zip(s_values, v_values)]
        
        # Properties that should hold
        assert np.mean(np.exp(one_sided_values)) <= 2  # Approximate martingale property
        assert np.mean(np.exp(two_sided_values)) <= 2
        
    def test_beta_binomial_mixture(self):
        """Test beta-binomial mixture properties."""
        mixture = BetaBinomialMixture(
            v_opt=1.0,
            alpha_opt=0.05,
            g=0.5,
            h=0.5,
            is_one_sided=True
        )
        
        # Test with various inputs
        s_values = np.linspace(-1, 1, 100)
        v_values = np.ones_like(s_values)
        
        log_values = [mixture.log_superMG(s, v) 
                     for s, v in zip(s_values, v_values)]
        
        # Check properties
        assert all(not np.isnan(v) for v in log_values)
        assert all(not np.isinf(v) for v in log_values)
        
        # Test bound method
        bound = mixture.bound(1.0, np.log(20))  # For α = 0.05
        assert bound > 0
        
    def test_gamma_exponential_mixture(self):
        """Test gamma-exponential mixture properties."""
        mixture = GammaExponentialMixture(
            v_opt=1.0,
            alpha_opt=0.05,
            c=1.0
        )
        
        # Test with various inputs
        s_values = np.linspace(0, 2, 100)
        v_values = np.ones_like(s_values)
        
        log_values = [mixture.log_superMG(s, v) 
                     for s, v in zip(s_values, v_values)]
        
        # Check properties
        assert all(not np.isnan(v) for v in log_values)
        assert all(not np.isinf(v) for v in log_values)
        
    def test_mixture_bounds(self):
        """Test bound computation for all mixtures."""
        mixtures = [
            OneSidedNormalMixture(1.0, 0.05),
            TwoSidedNormalMixture(1.0, 0.05),
            BetaBinomialMixture(1.0, 0.05, 0.5, 0.5, True),
            GammaExponentialMixture(1.0, 0.05, 1.0)
        ]
        
        v_values = [0.5, 1.0, 2.0]
        log_threshold = np.log(20)  # For α = 0.05
        
        for mixture in mixtures:
            for v in v_values:
                bound = mixture.bound(v, log_threshold)
                assert bound > 0
                assert not np.isnan(bound)
                assert not np.isinf(bound)

@pytest.mark.parametrize("v_opt,alpha_opt", [
    (0.5, 0.05),
    (1.0, 0.01),
    (2.0, 0.1)
])
def test_mixture_parameters(v_opt, alpha_opt):
    """Test mixture behavior with different parameters."""
    # One-sided normal
    one_sided = OneSidedNormalMixture(v_opt, alpha_opt)
    # Two-sided normal
    two_sided = TwoSidedNormalMixture(v_opt, alpha_opt)
    
    s, v = 0.0, 1.0
    
    # Test log superMG values
    one_sided_value = one_sided.log_superMG(s, v)
    two_sided_value = two_sided.log_superMG(s, v)
    
    assert not np.isnan(one_sided_value)
    assert not np.isnan(two_sided_value)