from typing import Optional, Callable, Sequence, ClassVar
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from numpy.typing import NDArray
from dataclasses import dataclass
from enum import Enum

class HypothesisType(str, Enum):
    """Types of statistical hypotheses."""
    SIMPLE = "simple"
    COMPOSITE = "composite"
    POINT = "point"

class Hypothesis(BaseModel):
    """Base model for statistical hypotheses."""
    name: str
    description: Optional[str] = None
    type: HypothesisType
    model_config = ConfigDict(frozen=True)

class EValueConfig(BaseModel):
    """Configuration settings for e-values."""
    significance_level: float = Field(gt=0, lt=1, default=0.05)
    allow_infinite: bool = Field(default=False)
    model_config = ConfigDict(frozen=True)

class EValueResult(BaseModel):
    """Result model for e-value computations."""
    value: float = Field(ge=0)
    significant: bool
    sample_size: int = Field(gt=0)
    hypothesis: Hypothesis
    config: EValueConfig
    timestamp: float = Field(default_factory=lambda: np.datetime64('now').astype(float))

class EValue(ABC):
    """Abstract base class for e-values using Pydantic."""
    
    config: ClassVar[EValueConfig] = EValueConfig()
    
    def __init__(self, 
                 null_hypothesis: Hypothesis,
                 config: Optional[EValueConfig] = None):
        self.null_hypothesis = null_hypothesis
        self.config = config or self.config
        self._result: Optional[EValueResult] = None
    
    @abstractmethod
    def compute(self, data: NDArray) -> float:
        """Compute the e-value for given data."""
        pass
    
    def test(self, data: NDArray) -> EValueResult:
        """Compute e-value and return detailed results."""
        value = self.compute(data)
        
        if not self.config.allow_infinite and np.isinf(value):
            raise ValueError("Infinite e-value detected and not allowed by config")
            
        result = EValueResult(
            value=float(value),
            significant=value >= 1/self.config.significance_level,
            sample_size=len(data),
            hypothesis=self.null_hypothesis,
            config=self.config
        )
        self._result = result
        return result
    
    @property
    def result(self) -> Optional[EValueResult]:
        """Get the latest test result if available."""
        return self._result

class LikelihoodRatioEValue(EValue):
    """E-value based on likelihood ratio with Pydantic models."""
    
    class Config(BaseModel):
        """Configuration for likelihood ratio computation."""
        log_space: bool = Field(default=True, description="Compute in log space for numerical stability")
        model_config = ConfigDict(frozen=True)
    
    def __init__(self,
                 null_hypothesis: Hypothesis,
                 null_density: Callable[[NDArray], NDArray],
                 alt_density: Callable[[NDArray], NDArray],
                 config: Optional[EValueConfig] = None,
                 lr_config: Optional[Config] = None):
        super().__init__(null_hypothesis, config)
        self.null_density = null_density
        self.alt_density = alt_density
        self.lr_config = lr_config or self.Config()
    
    def compute(self, data: NDArray) -> float:
        if self.lr_config.log_space:
            log_ratios = np.log(self.alt_density(data)) - np.log(self.null_density(data))
            return float(np.exp(np.sum(log_ratios)))
        else:
            ratios = self.alt_density(data) / self.null_density(data)
            return float(np.prod(ratios))

class EProcess(BaseModel):
    """E-process model for sequential testing."""
    
    values: list[float] = Field(default_factory=list)
    cumulative_value: float = Field(default=1.0)
    total_samples: int = Field(default=0)
    config: EValueConfig
    
    def update(self, e_value: float) -> float:
        """Update e-process with new e-value."""
        self.values.append(e_value)
        self.cumulative_value *= e_value
        self.total_samples += 1
        return self.cumulative_value
    
    def is_significant(self, alpha: Optional[float] = None) -> bool:
        """Check if current cumulative e-value is significant."""
        alpha = alpha or self.config.significance_level
        return self.cumulative_value >= 1/alpha

class UniversalEValue(EValue):
    """Universal inference e-value with Pydantic."""
    
    class Config(BaseModel):
        """Configuration for universal inference."""
        split_ratio: float = Field(gt=0, lt=1, default=0.5)
        min_samples: int = Field(ge=2, default=4)
        model_config = ConfigDict(frozen=True)
    
    def __init__(self,
                 null_hypothesis: Hypothesis,
                 null_mle: Callable[[NDArray], Callable[[NDArray], NDArray]],
                 alt_mle: Callable[[NDArray], Callable[[NDArray], NDArray]],
                 config: Optional[EValueConfig] = None,
                 ui_config: Optional[Config] = None):
        super().__init__(null_hypothesis, config)
        self.null_mle = null_mle
        self.alt_mle = alt_mle
        self.ui_config = ui_config or self.Config()
    
    def compute(self, data: NDArray) -> float:
        if len(data) < self.ui_config.min_samples:
            raise ValueError(f"Need at least {self.ui_config.min_samples} samples")
            
        split_idx = int(len(data) * self.ui_config.split_ratio)
        D0, D1 = data[:split_idx], data[split_idx:]
        
        q1_hat = self.alt_mle(D1)
        p0_hat = self.null_mle(D0)
        
        ratios = q1_hat(D0) / p0_hat(D0)
        return float(np.prod(ratios))

# Example usage
if __name__ == "__main__":
    from scipy.stats import norm
    
    # Define a simple null hypothesis
    null = Hypothesis(
        name="Standard Normal",
        description="Normal(0,1) distribution",
        type=HypothesisType.SIMPLE
    )
    
    # Create custom config
    config = EValueConfig(
        significance_level=0.05,
        allow_infinite=False
    )
    
    # Create density functions
    def null_density(x): return norm.pdf(x, loc=0, scale=1)
    def alt_density(x): return norm.pdf(x, loc=1, scale=1)
    
    # Initialize e-value calculator
    e_val = LikelihoodRatioEValue(
        null_hypothesis=null,
        null_density=null_density,
        alt_density=alt_density,
        config=config,
        lr_config=LikelihoodRatioEValue.Config(log_space=True)
    )
    
    # Generate test data
    np.random.seed(42)
    data = np.random.normal(loc=0.5, scale=1, size=10)
    
    # Run test
    result = e_val.test(data)
    print(f"E-value: {result.value:.4f}")
    print(f"Significant: {result.significant}")
    
    # Create e-process
    e_process = EProcess(config=config)
    
    # Update sequentially
    for x in data:
        new_result = e_val.test(np.array([x]))
        cumulative = e_process.update(new_result.value)
        print(f"Observation: {x:.2f}, "
              f"E-value: {new_result.value:.4f}, "
              f"Cumulative: {cumulative:.4f}")