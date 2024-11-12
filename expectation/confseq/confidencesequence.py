import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from numpy.typing import NDArray
from scipy import special, optimize
from typing import Callable, Optional
from expectation.modules import boundaries
from expectation.confseq.confidenceconfig import (BoundaryType, 
                                                  EstimandType, 
                                                  ConfidenceSequenceConfig, 
                                                  EmpiricalBernsteinConfig)

class ConfidenceSequenceState(BaseModel):
    """Current state of confidence sequence."""
    n_samples: int = Field(default=0, ge=0)
    sum: float = Field(default=0.0)
    sum_squares: float = Field(default=0.0)
    running_mean: float = Field(default=0.0)
    intrinsic_time: float = Field(default=0.0)
    variance_estimate: Optional[float] = None
    model_config = ConfigDict(frozen=True)  # State is immutable

class ConfidenceSequenceResult(BaseModel):
    """Result returned after each confidence sequence update."""
    lower: float = Field(description="Lower confidence bound")
    upper: float = Field(description="Upper confidence bound")
    state: ConfidenceSequenceState = Field(description="Current state after update")
    sample_size: int = Field(gt=0, description="Total number of samples processed")
    estimand: EstimandType = Field(description="Type of parameter being estimated")
    boundary_type: BoundaryType = Field(description="Type of boundary used")
    timestamp: float = Field(
        default_factory=lambda: np.datetime64('now').astype(float),
        description="Timestamp of update"
    )
    model_config = ConfigDict(frozen=True)

class ConfidenceSequence(BaseModel):
    """Base confidence sequence implementation using existing boundaries."""
    config: ConfidenceSequenceConfig
    state: ConfidenceSequenceState = Field(default_factory=ConfidenceSequenceState)
    estimand: EstimandType = Field(default=EstimandType.MEAN)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def update(self, new_data: NDArray[np.float_]) -> ConfidenceSequenceResult:
        """Update confidence sequence with new observations."""
        data = np.asarray(new_data)
        n_new = len(data)
        
        # Calculate new state values
        new_sum = self.state.sum + np.sum(data)
        new_n_samples = self.state.n_samples + n_new
        new_running_mean = new_sum / new_n_samples
        
        # Update empirical variance estimate
        if new_n_samples > 1:
            new_sum_squares = (self.state.sum_squares + 
                             np.sum((data - new_running_mean) * 
                                  (data - self.state.running_mean)))
            variance_estimate = new_sum_squares / (new_n_samples - 1)
            intrinsic_time = new_n_samples * variance_estimate
        else:
            new_sum_squares = 0
            variance_estimate = None
            intrinsic_time = new_n_samples
            
        # Create new state (immutable)
        new_state = ConfidenceSequenceState(
            n_samples=new_n_samples,
            sum=new_sum,
            sum_squares=new_sum_squares,
            running_mean=new_running_mean,
            intrinsic_time=intrinsic_time,
            variance_estimate=variance_estimate
        )
        
        # Use existing boundary implementations
        radius = (1.0 / new_n_samples * boundaries.gamma_exponential_mixture_bound(
            intrinsic_time, 
            self.config.alpha/2,
            self.config.v_opt,
            self.config.c,
            alpha_opt=self.config.alpha_opt/2
        ))

        # Update instance state
        self.state = new_state
        
        return ConfidenceSequenceResult(
            lower=new_running_mean - radius,
            upper=new_running_mean + radius,
            state=new_state,
            sample_size=new_n_samples,
            estimand=self.estimand,  # Use estimand from class field
            boundary_type=self.config.boundary_type,  # Use boundary type from config
        )
    
    def reset(self) -> None:
        """Reset the confidence sequence state."""
        self.state = ConfidenceSequenceState()

class EmpiricalBernsteinConfidenceSequence(ConfidenceSequence):
    """Empirical Bernstein confidence sequence for bounded observations."""
    config: EmpiricalBernsteinConfig
    
    def update(self, new_data: NDArray[np.float_]) -> ConfidenceSequenceResult:
        """Update confidence sequence with empirical variance estimate."""
        data = np.asarray(new_data)
        
        # Validate data is within bounds
        if (data < self.config.lower_bound).any() or (data > self.config.upper_bound).any():
            raise ValueError(
                f"All observations must be within [{self.config.lower_bound}, "
                f"{self.config.upper_bound}]"
            )
            
        # Use base class update with range scaling
        result = super().update(data)
        range_width = self.config.upper_bound - self.config.lower_bound
        
        return ConfidenceSequenceResult(
            lower=max(self.config.lower_bound, 
                     result.lower * range_width),
            upper=min(self.config.upper_bound, 
                     result.upper * range_width),
            state=result.state,
            sample_size=result.sample_size,
            estimand=self.estimand,  # Use estimand from parent class field
            boundary_type=self.config.boundary_type,  # Use boundary type from config
            timestamp=result.timestamp
        )