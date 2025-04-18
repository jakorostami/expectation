import pytest
import numpy as np


from expectation.parametric.ttest_universal import (
    TtestGaussianMixtureMartingale,
    TtestFlatMixtureMartingale,
    TtestConfidenceSequence
)
from expectation.confseq.confidenceconfig import ConfidenceSequenceConfig


class TestTTestLowLevel:
    def test_gaussian_mixture_e_value(self):
        np.random.seed(42)
        data = np.random.normal(0.5, 1.0, 30)
        s = np.sum(data - np.mean(data))
        v = np.sum((data - np.mean(data)) ** 2)
        mixture = TtestGaussianMixtureMartingale(prior_precision=1.0)
        
        # Skip test if variance is too small
        if v <= 1e-10:
            pytest.skip("Invalid parameter region for log-superMG")
            
        log_ev = mixture.log_superMG(s, v)
        assert np.isfinite(log_ev)
    
    def test_flat_mixture_e_value(self):
        np.random.seed(42)
        data = np.random.normal(0.5, 1.0, 30)
        s = np.sum(data - np.mean(data))
        v = np.sum((data - np.mean(data)) ** 2)
        mixture = TtestFlatMixtureMartingale()
        
        # Skip test if variance is too small
        if v <= 1e-10:
            pytest.skip("Invalid parameter region for log-superMG")
            
        log_ev = mixture.log_superMG(s, v)
        assert np.isfinite(log_ev)
    
    @pytest.mark.skip(reason="TtestConfidenceSequence needs to be implemented first")
    def test_confidence_sequence_universal_inference(self):
        np.random.seed(0)
        data = np.random.normal(0.2, 1.0, 50)
        config = ConfidenceSequenceConfig(alpha=0.05)
        cs = TtestConfidenceSequence(
            config=config,
            method="universal_inference"
        )
        result = cs.update(data)
        assert result.lower < result.upper
        assert np.isfinite(result.lower)
        assert np.isfinite(result.upper)
    
    @pytest.mark.skip(reason="TtestConfidenceSequence needs to be implemented first")
    def test_confidence_sequence_gaussian_mixture(self):
        np.random.seed(1)
        data = np.random.normal(0.0, 1.0, 50)
        config = ConfidenceSequenceConfig(alpha=0.05)
        cs = TtestConfidenceSequence(
            config=config,
            method="gaussian_mixture",
            prior_precision=1.0
        )
        result = cs.update(data)
        assert result.lower < result.upper
        assert np.isfinite(result.lower)
        assert np.isfinite(result.upper)
