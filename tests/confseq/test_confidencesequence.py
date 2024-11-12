import pytest
import numpy as np
from expectation.confseq.confidencesequence import (
    ConfidenceSequence,
    EmpiricalBernsteinConfidenceSequence,
    ConfidenceSequenceState,
    ConfidenceSequenceResult
)
from expectation.confseq.confidenceconfig import (
    ConfidenceSequenceConfig,
    EmpiricalBernsteinConfig,
    BoundaryType,
    EstimandType
)

@pytest.fixture
def basic_config():
    """Fixture providing basic configuration."""
    return ConfidenceSequenceConfig(
        alpha=0.05,
        alpha_opt=0.05,
        v_opt=1.0,
        c=1.0,
        boundary_type=BoundaryType.NORMAL_MIXTURE
    )

@pytest.fixture
def bernstein_config():
    """Fixture providing Empirical Bernstein configuration."""
    return EmpiricalBernsteinConfig(
        alpha=0.05,
        alpha_opt=0.05,
        v_opt=1.0,
        c=1.0,
        boundary_type=BoundaryType.NORMAL_MIXTURE,
        lower_bound=0.0,
        upper_bound=1.0,
        rho=2.0
    )

@pytest.fixture
def confidence_sequence(basic_config):
    """Fixture providing initialized confidence sequence."""
    return ConfidenceSequence(
        config=basic_config,
        estimand=EstimandType.MEAN
    )

@pytest.fixture
def bernstein_sequence(bernstein_config):
    """Fixture providing initialized Empirical Bernstein confidence sequence."""
    return EmpiricalBernsteinConfidenceSequence(
        config=bernstein_config,
        estimand=EstimandType.MEAN
    )

class TestConfidenceSequenceState:
    """Tests for ConfidenceSequenceState."""
    
    def test_initialization(self):
        """Test proper initialization of state."""
        state = ConfidenceSequenceState()
        assert state.n_samples == 0
        assert state.sum == 0.0
        assert state.sum_squares == 0.0
        assert state.running_mean == 0.0
        assert state.intrinsic_time == 0.0
        assert state.variance_estimate is None
    
    def test_immutability(self):
        """Test that state is immutable."""
        state = ConfidenceSequenceState()
        with pytest.raises(Exception):
            state.n_samples = 1

class TestConfidenceSequence:
    """Tests for base ConfidenceSequence."""
    
    def test_initialization(self, basic_config):
        """Test proper initialization."""
        cs = ConfidenceSequence(
            config=basic_config,
            estimand=EstimandType.MEAN
        )
        assert cs.state.n_samples == 0
        assert cs.estimand == EstimandType.MEAN
        assert cs.config == basic_config
    
    def test_update_properties(self, confidence_sequence):
        """Test properties of update method."""
        # Test with single observation
        data = np.array([1.0])
        result = confidence_sequence.update(data)
        
        assert isinstance(result, ConfidenceSequenceResult)
        assert result.lower <= result.upper
        assert result.sample_size == 1
        assert result.state.running_mean == 1.0
        
        # Test with multiple observations
        data = np.array([0.0, 1.0, 2.0])
        result = confidence_sequence.update(data)
        
        assert result.sample_size == 4  # 1 + 3 from previous update
        assert result.lower <= np.mean(data) <= result.upper
        assert result.state.variance_estimate is not None
    
    def test_reset(self, confidence_sequence):
        """Test reset functionality."""
        # Update with some data
        data = np.array([1.0, 2.0])
        confidence_sequence.update(data)
        
        # Reset
        confidence_sequence.reset()
        
        # Check state is reset
        assert confidence_sequence.state.n_samples == 0
        assert confidence_sequence.state.sum == 0.0
        assert confidence_sequence.state.running_mean == 0.0
    
    def test_coverage_property(self, confidence_sequence):
        """Test coverage property under null distribution."""
        np.random.seed(42)
        n_trials = 1000
        covered = 0
        true_mean = 0.0
        
        for _ in range(n_trials):
            confidence_sequence.reset()
            data = np.random.normal(true_mean, 1, 10)
            result = confidence_sequence.update(data)
            if result.lower <= true_mean <= result.upper:
                covered += 1
        
        coverage = covered / n_trials
        assert coverage >= 0.93  # Should be close to 0.95 (1-alpha)
    
    def test_boundary_consistency(self, confidence_sequence):
        """Test consistency of confidence bounds."""
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        
        prev_width = float('inf')
        for i in range(10, len(data), 10):
            result = confidence_sequence.update(data[i-10:i])
            width = result.upper - result.lower
            
            # Width should generally decrease with more samples
            assert width <= prev_width * 1.1  # Allow small fluctuations
            prev_width = width

class TestEmpiricalBernsteinConfidenceSequence:
    """Tests for Empirical Bernstein confidence sequence."""
    
    def test_initialization(self, bernstein_config):
        """Test proper initialization."""
        cs = EmpiricalBernsteinConfidenceSequence(
            config=bernstein_config,
            estimand=EstimandType.MEAN
        )
        assert cs.config.lower_bound == 0.0
        assert cs.config.upper_bound == 1.0
    
    def test_bounds_validation(self, bernstein_sequence):
        """Test validation of observation bounds."""
        # Valid data
        valid_data = np.array([0.2, 0.5, 0.8])
        result = bernstein_sequence.update(valid_data)
        assert result.lower >= bernstein_sequence.config.lower_bound
        assert result.upper <= bernstein_sequence.config.upper_bound
        
        # Invalid data - below lower bound
        invalid_data = np.array([-0.1, 0.5])
        with pytest.raises(ValueError):
            bernstein_sequence.update(invalid_data)
        
        # Invalid data - above upper bound
        invalid_data = np.array([0.5, 1.1])
        with pytest.raises(ValueError):
            bernstein_sequence.update(invalid_data)
    
    def test_coverage_property(self, bernstein_sequence):
        """Test coverage property with bounded observations."""
        np.random.seed(42)
        n_trials = 1000
        covered = 0
        true_mean = 0.5  # Center of [0,1] interval
        
        for _ in range(n_trials):
            bernstein_sequence.reset()
            # Generate bounded data using beta distribution
            data = np.random.beta(5, 5, 10)  # Beta(5,5) is symmetric around 0.5
            result = bernstein_sequence.update(data)
            if result.lower <= true_mean <= result.upper:
                covered += 1
        
        coverage = covered / n_trials
        assert coverage >= 0.93  # Should be close to 0.95 (1-alpha)
    
    @pytest.mark.parametrize("data_size", [1, 10, 100])
    def test_sample_size_scaling(self, bernstein_sequence, data_size):
        """Test scaling of confidence bounds with sample size."""
        np.random.seed(42)
        data = np.random.beta(5, 5, data_size)
        result = bernstein_sequence.update(data)
        
        # Basic properties that should hold regardless of sample size
        assert result.lower >= bernstein_sequence.config.lower_bound
        assert result.upper <= bernstein_sequence.config.upper_bound
        assert result.sample_size == data_size
        
        if data_size > 1:
            assert result.state.variance_estimate is not None

@pytest.mark.parametrize("boundary_type", [
    BoundaryType.NORMAL_MIXTURE,
    BoundaryType.BETA_BINOMIAL,
    BoundaryType.GAMMA_EXPONENTIAL,
    BoundaryType.POLY_STITCHING
])
def test_boundary_types(basic_config, boundary_type):
    """Test confidence sequence with different boundary types."""
    config = ConfidenceSequenceConfig(
        **{**basic_config.model_dump(), "boundary_type": boundary_type}
    )
    cs = ConfidenceSequence(config=config, estimand=EstimandType.MEAN)
    
    data = np.random.normal(0, 1, 10)
    result = cs.update(data)
    
    assert result.boundary_type == boundary_type
    assert result.lower <= result.upper