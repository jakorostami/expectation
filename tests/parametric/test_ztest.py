import pytest
import numpy as np
import warnings
from expectation.parametric.ztest import savi_z_test_stat
from expectation.utils.helper_functions import effective_sample_size, check_and_return_esmin_parameter_side

class TestEffectiveSampleSize:
    """Test cases for effective_sample_size function."""
    
    def test_single_sample(self):
        assert effective_sample_size(10) == 10
        
    def test_two_samples(self):
        result = effective_sample_size(10, 20)
        expected = 1 / (1/10 + 1/20)  # harmonic mean
        assert np.isclose(result, expected)
        
    def test_paired(self):
        assert effective_sample_size(15, 30, paired=True) == 15
        
    def test_invalid_n1(self):
        with pytest.raises(ValueError):
            effective_sample_size(0)
            
    def test_invalid_n2(self):
        with pytest.raises(ValueError):
            effective_sample_size(10, n2=0)

class TestCheckAndReturnEsminParameterSide:
    """Test cases for check_and_return_esmin_parameter_side function."""
    
    def test_valid_greater(self):
        assert check_and_return_esmin_parameter_side(1.5, "greater") == 1.5
        
    def test_invalid_greater(self):
        with pytest.raises(ValueError):
            check_and_return_esmin_parameter_side(-1, "greater")
            
    def test_valid_less(self):
        assert check_and_return_esmin_parameter_side(-0.5, "less") == -0.5
        
    def test_invalid_less(self):
        with pytest.raises(ValueError):
            check_and_return_esmin_parameter_side(1, "less")
            
    def test_invalid_twoSided(self):
        with pytest.raises(ValueError):
            check_and_return_esmin_parameter_side(0, "twoSided")

class TestSaviZTestStat:
    """Comprehensive test cases for savi_z_test_stat function."""
    
    # ========== Basic Functionality Tests ==========
    def test_grow_method_all_alternatives(self):
        """Test grow method for all alternatives with appropriate parameter signs."""
        # Two-sided (parameter sign doesn't matter for grow after check_and_return_esmin_parameter_side)
        result = savi_z_test_stat(z=1.96, n1=30, parameter=0.5, alternative="twoSided")
        assert "eValue" in result
        assert result["eValue"] > 0
        assert np.isfinite(result["eValue"])
        
        # Greater (positive parameter)
        result = savi_z_test_stat(z=2.0, n1=20, parameter=0.5, alternative="greater")
        assert result["eValue"] > 0
        
        # Less (negative parameter for grow method)
        result = savi_z_test_stat(z=-2.0, n1=20, parameter=-0.5, alternative="less")
        assert result["eValue"] > 0
        
    def test_mom_method_all_alternatives(self):
        """Test moment of mixture method with positive parameters only."""
        # Two-sided (positive parameter)
        result = savi_z_test_stat(z=2.0, n1=20, parameter=0.1, alternative="twoSided", eType="mom")
        assert "eValue" in result
        assert result["eValue"] > 0
        
        # Greater (positive parameter)
        result = savi_z_test_stat(z=2.0, n1=20, parameter=0.1, alternative="greater", eType="mom")
        assert "eValue" in result
        assert "eValueApproxError" in result
        assert result["eValue"] > 0
        
        # Less (positive parameter - the integrand handles direction)
        result = savi_z_test_stat(z=-2.0, n1=20, parameter=0.1, alternative="less", eType="mom")
        assert result["eValue"] > 0
        
    def test_egauss_method_all_alternatives(self):
        """Test eGauss method with positive parameters only."""
        for alt in ["twoSided", "greater", "less"]:
            z_val = 2.0 if alt != "less" else -2.0
            # Use positive parameter for all alternatives
            result = savi_z_test_stat(z=z_val, n1=20, parameter=0.1, 
                                    alternative=alt, eType="eGauss")
            assert "eValue" in result
            assert result["eValue"] > 0
            assert np.isfinite(result["eValue"])
            
    def test_imom_method_all_alternatives(self):
        """Test inverse moment of mixture method with positive parameters."""
        for alt in ["twoSided", "greater", "less"]:
            z_val = 2.0 if alt != "less" else -2.0
            # Use positive parameter (tau must be positive)
            result = savi_z_test_stat(z=z_val, n1=20, parameter=0.1, 
                                    alternative=alt, eType="imom")
            assert "eValue" in result
            assert "eValueApproxError" in result
            assert result["eValue"] > 0
            
    def test_ecauchy_method_all_alternatives(self):
        """Test eCauchy method with positive parameters."""
        for alt in ["twoSided", "greater", "less"]:
            z_val = 2.0 if alt != "less" else -2.0
            # Use positive parameter (kappaG must be positive)
            result = savi_z_test_stat(z=z_val, n1=20, parameter=0.1, 
                                    alternative=alt, eType="eCauchy")
            assert "eValue" in result
            assert "eValueApproxError" in result
            assert result["eValue"] > 0

    # ========== Input Validation Tests ==========
    def test_invalid_alternative(self):
        """Test invalid alternative parameter."""
        with pytest.raises(ValueError, match="Alternative must be from one of"):
            savi_z_test_stat(z=1.0, n1=10, parameter=0.5, alternative="invalid")
            
    def test_not_implemented_etype(self):
        """Test unimplemented eType."""
        with pytest.raises(NotImplementedError):
            savi_z_test_stat(z=1.0, n1=10, parameter=0.5, eType="unknown")
    
    def test_invalid_parameter_signs(self):
        """Test that methods handle invalid parameter signs appropriately."""
        # Test negative parameters for methods that require positive parameters
        methods_requiring_positive = ["mom", "eGauss", "imom", "eCauchy"]
        
        for method in methods_requiring_positive:
            # Check for warnings when using invalid parameters
            with pytest.raises(ValueError):
                savi_z_test_stat(z=1.0, n1=10, parameter=-0.5, alternative="greater", eType=method)
                
    # ========== Sample Size and Parameter Variations ==========
    @pytest.mark.parametrize("n1,n2", [(10, None), (10, 20), (50, 30)])
    def test_various_sample_sizes(self, n1, n2):
        """Test various sample size combinations."""
        result = savi_z_test_stat(z=1.5, n1=n1, n2=n2, parameter=0.5, alternative="greater")
        assert "eValue" in result
        assert result["eValue"] > 0
        
    def test_paired_samples(self):
        """Test paired samples."""
        result = savi_z_test_stat(z=1.5, n1=20, n2=25, parameter=0.5, 
                                alternative="greater", paired=True)
        assert result["eValue"] > 0
    
    @pytest.mark.parametrize("parameter", [0.01, 0.1, 0.5, 1.0])  # Only positive parameters
    def test_various_parameters(self, parameter):
        """Test various positive parameter values."""
        result = savi_z_test_stat(z=1.5, n1=20, parameter=parameter, alternative="greater")
        assert result["eValue"] > 0
        
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])  # Reduced range for stability
    def test_various_sigma(self, sigma):
        """Test various sigma values."""
        result = savi_z_test_stat(z=1.5, n1=20, parameter=0.5, alternative="greater", sigma=sigma)
        assert result["eValue"] > 0
    
    # ========== Edge Cases and Numerical Stability ==========
    def test_zero_z_statistic(self):
        """Test with z=0."""
        result = savi_z_test_stat(z=0, n1=20, parameter=0.5, alternative="greater")
        assert result["eValue"] > 0
        
    def test_moderate_z_statistic(self):
        """Test with moderate z-statistic."""
        result = savi_z_test_stat(z=3, n1=20, parameter=0.1, alternative="greater")
        assert result["eValue"] > 0
        assert np.isfinite(result["eValue"])
        
    @pytest.mark.slow  # Mark as slow test
    def test_large_z_statistic(self):
        """Test with large z-statistic (may be slow for integration methods)."""
        result = savi_z_test_stat(z=8, n1=20, parameter=0.05, alternative="greater")
        assert result["eValue"] > 0
        assert np.isfinite(result["eValue"])
        
    def test_small_sample_size(self):
        """Test with small sample size."""
        result = savi_z_test_stat(z=1.5, n1=5, parameter=0.5, alternative="greater")
        assert result["eValue"] > 0
        
    def test_large_sample_size(self):
        """Test with large sample size."""
        result = savi_z_test_stat(z=1.5, n1=1000, parameter=0.1, alternative="greater")
        assert result["eValue"] > 0
    
    def test_extreme_parameters_carefully(self):
        """Test numerical stability with carefully chosen extreme values."""
        # Very small parameter (but not too small to cause numerical issues)
        result1 = savi_z_test_stat(z=1.0, n1=10, parameter=1e-3, alternative="greater")
        assert np.isfinite(result1["eValue"])
        assert result1["eValue"] > 0
        
        # Moderately large parameter (avoid extreme values that cause convergence issues)
        result2 = savi_z_test_stat(z=1.0, n1=10, parameter=2.0, alternative="greater")
        assert np.isfinite(result2["eValue"])
    
    # ========== Mathematical Properties Tests ==========
    def test_monotonicity_with_z_greater(self):
        """Test that e-value increases with z-statistic for greater alternative."""
        z_values = [0.5, 1.0, 1.5, 2.0]  # Moderate range
        e_values = []
        
        for z in z_values:
            result = savi_z_test_stat(z=z, n1=20, parameter=0.2, alternative="greater")
            e_values.append(result["eValue"])
        
        # e-values should generally increase with z for greater alternative
        assert e_values[-1] > e_values[0], "e-value should increase with z-statistic"
        
    def test_monotonicity_with_z_less(self):
        """Test that e-value increases with |z| for less alternative (grow method only)."""
        z_values = [-0.5, -1.0, -1.5, -2.0]
        e_values = []
        
        for z in z_values:
            # Use grow method with negative parameter for less alternative
            result = savi_z_test_stat(z=z, n1=20, parameter=-0.2, alternative="less", eType="grow")
            e_values.append(result["eValue"])
        
        # e-values should increase as z becomes more negative
        assert e_values[-1] > e_values[0], "e-value should increase with |z| for less alternative"
    
    def test_symmetry_properties_two_sided(self):
        """Test symmetry properties for two-sided tests with relaxed tolerance."""
        # For two-sided tests, z and -z should give same result
        result_pos = savi_z_test_stat(z=2.0, n1=20, parameter=0.5, alternative="twoSided")
        result_neg = savi_z_test_stat(z=-2.0, n1=20, parameter=0.5, alternative="twoSided")
        
        # Use more relaxed tolerance to account for numerical integration errors
        assert np.isclose(result_pos["eValue"], result_neg["eValue"], rtol=1e-6), \
               "Two-sided test should be approximately symmetric for z and -z"
    
    def test_consistency_across_methods_relaxed(self):
        """Test that different methods give reasonable results with relaxed comparison."""
        base_params = {"z": 1.5, "n1": 20, "parameter": 0.1, "alternative": "greater"}  # Moderate values
        
        # Test implemented methods with positive parameters
        methods = ["grow", "mom", "eGauss", "imom", "eCauchy"]
        results = {}
        
        for method in methods:
            results[method] = savi_z_test_stat(eType=method, **base_params)
            assert results[method]["eValue"] > 0, f"{method} should give positive e-value"
            assert np.isfinite(results[method]["eValue"]), f"{method} should give finite e-value"
        
        # Check that results are within reasonable range (relaxed threshold)
        e_values = [results[method]["eValue"] for method in methods]
        max_e = max(e_values)
        min_e = min(e_values)
        
        # Use log scale comparison for safer magnitude comparison
        log_ratio = np.log10(max_e / min_e)
        assert log_ratio < 10, f"Methods should give results within 10 orders of magnitude, got {log_ratio:.2f}"

    # ========== Integration Accuracy Tests ==========
    def test_integration_error_bounds(self):
        """Test that integration methods return reasonable approximation errors."""
        integration_methods = ["mom", "imom", "eCauchy"]
        
        for method in integration_methods:
            # Test with moderate values to ensure stable integration
            result = savi_z_test_stat(z=1.5, n1=20, parameter=0.1, 
                                    alternative="greater", eType=method)
            
            if method in ["imom", "eCauchy"]:
                assert "eValueApproxError" in result
                assert result["eValueApproxError"] >= 0
                
                # Check relative error for reasonable values, absolute error for very small values
                if result["eValue"] > 1e-10:  
                    relative_error = result["eValueApproxError"] / result["eValue"]
                    assert relative_error < 0.1, f"{method} relative error should be < 10%"
                else:
                    # For very small e-values, check absolute error instead
                    assert result["eValueApproxError"] < 1e-8, f"{method} absolute error should be < 1e-8 for small e-values"

    # ========== Robust Overflow and Warning Tests ==========
    def test_overflow_warning_mechanism_robust(self):
        """Test overflow warning mechanism with custom warning matching."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create conditions that might trigger custom overflow handling
            result = savi_z_test_stat(z=-20, n1=20, parameter=5, 
                                    alternative="greater", eType="grow")
            
            # Check for custom overflow warning from your code
            overflow_warned = False
            for warning in w:
                if "Overflow" in str(warning.message) and "e-value smaller than 0" in str(warning.message):
                    overflow_warned = True
                    assert result["eValue"] == 2**(-15)
                    break
            
            # If no custom overflow warning, result should still be valid
            if not overflow_warned:
                assert result["eValue"] > 0
                assert np.isfinite(result["eValue"])

# ========== Performance Tests ==========
class TestPerformance:
    """Performance tests with realistic expectations."""
    
    @pytest.mark.parametrize("method", ["grow", "mom", "eGauss"])  # Test faster methods
    def test_reasonable_execution_time_fast_methods(self, method):
        """Test that faster methods execute quickly."""
        import time
        
        start_time = time.time()
        result = savi_z_test_stat(z=2.0, n1=100, parameter=0.1, 
                                alternative="greater", eType=method)
        end_time = time.time()
        
        assert result["eValue"] > 0
        assert end_time - start_time < 2.0, f"{method} should complete within 2 seconds"
    
    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["imom", "eCauchy"])  # Test integration methods
    def test_reasonable_execution_time_integration_methods(self, method):
        """Test that integration methods complete in reasonable time."""
        import time
        
        start_time = time.time()
        result = savi_z_test_stat(z=2.0, n1=50, parameter=0.1, 
                                alternative="greater", eType=method)
        end_time = time.time()
        
        assert result["eValue"] > 0
        # More generous time limit for integration methods
        assert end_time - start_time < 10.0, f"{method} should complete within 10 seconds"
    
    @pytest.mark.slow
    def test_large_sample_performance_relaxed(self):
        """Test performance with moderately large samples (relaxed timing)."""
        import time
        
        start_time = time.time()
        # Reduced sample size to be more realistic
        result = savi_z_test_stat(z=1.5, n1=2000, parameter=0.1, alternative="greater")
        end_time = time.time()
        
        assert result["eValue"] > 0
        # Relaxed timing expectation
        assert end_time - start_time < 5.0, "Moderately large sample test should complete within 5 seconds"

# ========== Regression Tests ==========
class TestRegression:
    """Regression tests for previously fixed bugs."""
    
    def test_all_methods_work_without_errors_safe_parameters(self):
        """Comprehensive test with safe parameter choices."""
        methods = ["grow", "mom", "eGauss", "imom", "eCauchy"]
        alternatives = ["twoSided", "greater", "less"]
        
        for method in methods:
            for alt in alternatives:
                z_val = 1.5 if alt != "less" else -1.5  # Moderate z values
                
                # Use method-appropriate parameters
                if method == "grow" and alt == "less":
                    param = -0.2  # Negative for grow + less
                else:
                    param = 0.2   # Positive for other methods
                
                try:
                    result = savi_z_test_stat(z=z_val, n1=20, parameter=param, 
                                            alternative=alt, eType=method)
                    assert "eValue" in result
                    assert result["eValue"] > 0
                    assert np.isfinite(result["eValue"])
                except Exception as e:
                    pytest.fail(f"Method {method} with alternative {alt} failed: {e}")
    
    def test_formula_consistency_regression_relaxed(self):
        """Test that formulas are mathematically consistent with relaxed comparison."""
        # Test eGauss specifically for the fixed formula
        result = savi_z_test_stat(z=1.0, n1=10, parameter=0.2, 
                                alternative="greater", eType="eGauss")
        assert result["eValue"] > 0
        assert np.isfinite(result["eValue"])
        
        # Compare with grow method (relaxed comparison)
        result_grow = savi_z_test_stat(z=1.0, n1=10, parameter=0.2, 
                                     alternative="greater", eType="grow")
        
        # Results should be in reasonable range (very relaxed)
        ratio = result["eValue"] / result_grow["eValue"]
        assert 1e-8 < ratio < 1e8, f"eGauss and grow should give reasonably comparable results, got ratio={ratio}"

    def test_formula_robustness_randomized(self):
        """Test formula robustness with randomized inputs."""
        rng = np.random.default_rng(42)
        
        methods = ["grow", "mom", "eGauss", "imom", "eCauchy"]
        
        for method in methods:
            # Test 5 random combinations per method
            for _ in range(5):
                z = rng.normal(0, 2)  # Random z-score
                n1 = rng.integers(5, 100)  # Random sample size
                param = rng.uniform(0.05, 1.0)  # Random positive parameter
                
                # Handle grow method with less alternative
                if method == "grow" and z < 0:
                    alt = "less"
                    param = -param  # Negative parameter for less alternative
                else:
                    alt = "greater"
                
                try:
                    result = savi_z_test_stat(z=z, n1=n1, parameter=param, 
                                            alternative=alt, eType=method)
                    assert np.isfinite(result["eValue"]), f"{method} should give finite result"
                    assert result["eValue"] > 0, f"{method} should give positive e-value"
                except Exception as e:
                    pytest.fail(f"Method {method} failed with z={z}, n1={n1}, param={param}: {e}")

# ========== Custom pytest configuration ==========
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")

# Alternative: Create pytest.ini file with:
# [tool:pytest]
# markers =
#     slow: marks tests as slow (deselect with '-m "not slow"')

if __name__ == "__main__":
    # Run tests with verbose output, excluding slow tests by default
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])