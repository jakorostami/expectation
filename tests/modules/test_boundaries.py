import pytest
import numpy as np
from expectation.modules.boundaries import (
    normal_log_mixture, normal_mixture_bound,
    gamma_exponential_log_mixture, gamma_exponential_mixture_bound,
    beta_binomial_log_mixture, beta_binomial_mixture_bound,
    poly_stitching_bound, empirical_process_lil_bound,
    double_stitching_bound, log_beta, log_incomplete_beta
)

class TestBoundaryFunctions:
    @pytest.mark.parametrize("is_one_sided", [True, False])
    def test_normal_mixture(self, is_one_sided):
        """Test normal mixture functions."""
        s = np.array([0.0, 0.5, 1.0])
        v = np.ones_like(s)
        v_opt = 1.0
        alpha_opt = 0.05
        
        # Test log mixture
        log_values = normal_log_mixture(s, v, v_opt, alpha_opt, is_one_sided)
        assert len(log_values) == len(s)
        assert not np.any(np.isnan(log_values))
        
        # Test bound
        bounds = normal_mixture_bound(v, 0.05, v_opt, alpha_opt, is_one_sided)
        assert len(bounds) == len(v)
        assert np.all(bounds > 0)
    
    def test_gamma_exponential_mixture(self):
        """Test gamma-exponential mixture functions."""
        s = np.array([0.0, 0.5, 1.0])
        v = np.ones_like(s)
        v_opt = 1.0
        c = 1.0
        alpha_opt = 0.05
        
        # Test log mixture
        log_values = gamma_exponential_log_mixture(s, v, v_opt, c, alpha_opt)
        assert len(log_values) == len(s)
        assert not np.any(np.isnan(log_values))
        
        # Test bound
        bounds = gamma_exponential_mixture_bound(v, 0.05, v_opt, c, alpha_opt)
        assert len(bounds) == len(v)
        assert np.all(bounds > 0)
    
    def test_beta_binomial_mixture(self):
        """Test beta-binomial mixture functions."""
        s = np.array([0.0, 0.5, 1.0])
        v = np.ones_like(s)
        v_opt = 1.0
        g = 0.5
        h = 0.5
        alpha_opt = 0.05
        
        # Test both one-sided and two-sided
        for is_one_sided in [True, False]:
            log_values = beta_binomial_log_mixture(
                s, v, v_opt, g, h, alpha_opt, is_one_sided
            )
            assert len(log_values) == len(s)
            assert not np.any(np.isnan(log_values))
            
            bounds = beta_binomial_mixture_bound(
                v, 0.05, v_opt, g, h, alpha_opt, is_one_sided
            )
            assert len(bounds) == len(v)
            assert np.all(bounds > 0)
    
    def test_poly_stitching_bound(self):
        """Test polynomial stitching bound."""
        v = np.array([1.0, 2.0, 5.0])
        alpha = 0.05
        v_min = 0.5
        
        bounds = poly_stitching_bound(v, alpha, v_min)
        assert len(bounds) == len(v)
        assert np.all(bounds > 0)
        
        # Test monotonicity
        assert np.all(np.diff(bounds) >= 0)  # should increase with v
    
    def test_empirical_process_lil_bound(self):
        """Test empirical process LIL bound."""
        t_values = [10, 100, 1000]
        alpha = 0.05
        t_min = 5
        
        for t in t_values:
            bound = empirical_process_lil_bound(t, alpha, t_min)
            assert bound > 0
            assert not np.isnan(bound)
    
    def test_double_stitching_bound(self):
        """Test double stitching bound."""
        test_cases = [
            (0.5, 100, 0.05, 50),  # median, moderate sample
            (0.75, 200, 0.01, 100),  # upper quartile, larger sample
            (0.25, 50, 0.1, 25)  # lower quartile, smaller sample
        ]
        
        for quantile_p, t, alpha, t_opt in test_cases:
            bound = double_stitching_bound(quantile_p, t, alpha, t_opt)
            assert bound > 0
            assert not np.isnan(bound)
            
            # Test monotonicity in t
            bound_larger_t = double_stitching_bound(quantile_p, t * 2, alpha, t_opt)
            assert bound_larger_t >= bound
    
    def test_log_beta(self):
        """Test log beta function."""
        test_cases = [
            (1.0, 1.0),  # uniform
            (0.5, 0.5),  # jeffreys
            (2.0, 3.0)   # asymmetric
        ]
        
        for a, b in test_cases:
            value = log_beta(a, b)
            assert not np.isnan(value)
            assert not np.isinf(value)
            
            # Test symmetry
            assert np.abs(log_beta(a, b) - log_beta(b, a)) < 1e-10
    
    def test_log_incomplete_beta(self):
        """Test log incomplete beta function."""
        test_cases = [
            (1.0, 1.0, 0.5),  # symmetric case
            (2.0, 2.0, 0.7),  # symmetric parameters
            (0.5, 1.5, 0.3)   # asymmetric case
        ]
        
        for a, b, x in test_cases:
            value = log_incomplete_beta(a, b, x)
            assert not np.isnan(value)
            assert not np.isinf(value)
            
            # Test boundary case
            full_value = log_incomplete_beta(a, b, 1.0)
            assert full_value == log_beta(a, b)
    
    @pytest.mark.parametrize("bound_fn,params", [
        (normal_mixture_bound, {'v_opt': 1.0, 'alpha_opt': 0.05, 'is_one_sided': True}),
        (gamma_exponential_mixture_bound, {'v_opt': 1.0, 'c': 1.0, 'alpha_opt': 0.05}),
        (poly_stitching_bound, {'v_min': 0.5, 'c': 0, 's': 1.4, 'eta': 2})
    ])
    def test_bound_properties(self, bound_fn, params):
        """Test general properties of boundary functions."""
        v_values = np.array([0.5, 1.0, 2.0, 5.0])
        alpha = 0.05
        
        bounds = bound_fn(v_values, alpha, **params)
        
        # Basic properties
        assert len(bounds) == len(v_values)
        assert np.all(bounds > 0)
        assert not np.any(np.isnan(bounds))
        assert not np.any(np.isinf(bounds))
        
        # Monotonicity in v
        assert np.all(np.diff(bounds) >= 0)
        
        # Monotonicity in alpha
        bounds_larger_alpha = bound_fn(v_values, alpha * 2, **params)
        assert np.all(bounds_larger_alpha <= bounds)
    
    def test_extreme_cases(self):
        """Test boundary functions with extreme inputs."""
        # Very small values
        v_small = np.array([1e-10, 1e-9, 1e-8])
        # Very large values
        v_large = np.array([1e8, 1e9, 1e10])
        
        # Test each boundary function with extreme values
        for v in [v_small, v_large]:
            # Normal mixture
            bounds = normal_mixture_bound(v, 0.05, 1.0, 0.05, True)
            assert np.all(np.isfinite(bounds))
            
            # Gamma exponential
            bounds = gamma_exponential_mixture_bound(v, 0.05, 1.0, 1.0, 0.05)
            assert np.all(np.isfinite(bounds))
            
            # Poly stitching
            bounds = poly_stitching_bound(v, 0.05, min(v)/2)
            assert np.all(np.isfinite(bounds))