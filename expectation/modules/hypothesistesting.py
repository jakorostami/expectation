"""
Based on these papers:

Hypothesis testing with e-values, A. Ramdas, R. Wang (2024) - https://arxiv.org/pdf/2410.23614

Safe Testing, P. Grünwald, R. de Heide, W.M Koolen (2019) - https://arxiv.org/pdf/1906.07801
"""

from typing import Optional, Callable, Sequence, ClassVar, List, Union, Tuple
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
from numpy.typing import NDArray
from enum import Enum

from expectation.modules.epower import (
    EPowerCalculator, EPowerConfig, EPowerType, EPowerResult
)

class SymmetryType(str, Enum):
    FISHER = "fisher"
    SIGN = "sign"
    WILCOXON = "wilcoxon"

class HypothesisType(str, Enum):
    SIMPLE = "simple"
    COMPOSITE = "composite"
    POINT = "point"

class Hypothesis(BaseModel):
    name: str
    description: Optional[str] = None
    type: HypothesisType
    model_config = ConfigDict(frozen=True)

class EValueConfig(BaseModel):
    significance_level: float = Field(gt=0, lt=1, default=0.05)
    allow_infinite: bool = Field(default=False)
    model_config = ConfigDict(frozen=True)

class EValueResult(BaseModel):
    value: float = Field(ge=0)
    significant: bool
    sample_size: int = Field(gt=0)
    hypothesis: Hypothesis
    config: EValueConfig
    timestamp: float = Field(default_factory=lambda: np.datetime64('now').astype(float))

class EValue(ABC):
    config: ClassVar[EValueConfig] = EValueConfig()
    
    def __init__(self, 
                 null_hypothesis: Hypothesis,
                 config: Optional[EValueConfig] = None):
        self.null_hypothesis = null_hypothesis
        self.config = config or self.config
        self._result: Optional[EValueResult] = None
    
    @abstractmethod
    def compute(self, data: NDArray) -> float:
        """
        Compute the e-value for given data.
        """
        pass
    
    def test(self, data: NDArray) -> EValueResult:
        """
        Compute e-value and return detailed results.
        """
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
        """
        Get the latest test result if available.
        """
        return self._result

class LikelihoodRatioEValue(EValue):
    class Config(BaseModel):
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
    values: list[float] = Field(default_factory=list)
    cumulative_value: float = Field(default=1.0)
    total_samples: int = Field(default=0)
    config: EValueConfig
    
    def update(self, e_value: float) -> float:
        self.values.append(e_value)
        self.cumulative_value *= e_value
        self.total_samples += 1
        return self.cumulative_value
    
    def is_significant(self, alpha: Optional[float] = None) -> bool:
        alpha = alpha or self.config.significance_level
        return self.cumulative_value >= 1/alpha

class UniversalEValue(EValue):
    class Config(BaseModel):
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
    

class SymmetryETest:
    """Implementation of nonparametric e-tests of symmetry from the paper:
    Nonparametric E-tests of symmetry, Vovk and R. Wang (2024) - https://doi.org/10.51387/24-NEJSDS60
    
    This class implements the three symmetry tests discussed in the paper:
    1. Fisher-type test (based on sum of observations)
    2. Sign test (based on number of positive observations)
    3. Wilcoxon signed-rank test (based on ranks of positive observations)
    """
    
    def __init__(self, 
                 test_type: SymmetryType = SymmetryType.FISHER,
                 config: Optional[EValueConfig] = None,
                 lambda_value: float = 0.5,
                 e_power_config: Optional[EPowerConfig] = None):
        self.null_hypothesis = Hypothesis(
            name="Symmetry",
            description="Distribution is symmetric around 0",
            type=HypothesisType.COMPOSITE
        )
        
        self.config = config or EValueConfig()
        self.test_type = test_type
        self.lambda_value = lambda_value
        self.e_power_calculator = EPowerCalculator(e_power_config)
        self._result = None
        
    def compute(self, data: NDArray) -> float:
        if self.test_type == SymmetryType.FISHER:
            return self._compute_fisher_e_value(data)
        elif self.test_type == SymmetryType.SIGN:
            return self._compute_sign_e_value(data)
        elif self.test_type == SymmetryType.WILCOXON:
            return self._compute_wilcoxon_e_value(data)
        else:
            raise ValueError(f"Unknown test type: {self.test_type}")
            
    def _compute_fisher_e_value(self, data: NDArray) -> float:
        """
        Compute Fisher-type e-value (Section 4 in the paper).
        
        This implements equation (4.4) from the paper
        """
        lambda_val = self.lambda_value
        numerator = np.exp(lambda_val * data)
        denominator = 0.5 * (np.exp(lambda_val * data) + np.exp(-lambda_val * data))
        
        e_value = np.prod(numerator / denominator)
        
        return float(e_value)
    
    def _compute_sign_e_value(self, data: NDArray) -> float:
        """
        Compute Sign e-value (Section 7 in the paper).
        
        This implements equation (7.3) from the paper where k is the number of psitive observations
        """
        lambda_val = self.lambda_value

        k = np.sum(data > 0)
        n = len(data)

        e_value = np.exp(lambda_val * k) * (2 / (1 + np.exp(lambda_val)))**n
        
        return float(e_value)
    
    def _compute_wilcoxon_e_value(self, data: NDArray) -> float:
        """
        Compute Wilcoxon signed-rank e-value (Section 8 in the paper).
        
        This implements equation (8.3) from the paper where V_n is the sum of ranks of positive observations.
        """
        lambda_val = self.lambda_value
        n = len(data)

        abs_data = np.abs(data)
        ranks = np.argsort(np.argsort(abs_data)) + 1
 
        V_n = np.sum(ranks[data > 0])
        
        numerator = np.exp(lambda_val * V_n)
        denominator_factors = np.array([1 + np.exp(lambda_val * i) for i in range(1, n+1)])
        denominator = np.prod(2 / denominator_factors)
        
        e_value = numerator * denominator
        
        return float(e_value)

    def test(self, data: NDArray) -> EValueResult:
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
        return self._result
    
    def compute_e_power(self, 
                        alternative_data: NDArray, 
                        e_power_config: Optional[EPowerConfig] = None) -> EPowerResult:
        """
        Compute e-power against alternative data using existing EPowerCalculator.
        
        Args:
            alternative_data: Data from alternative distribution
            e_power_config: Override default e-power configuration

        Returns:
            EPowerResult from the e-power calculator
        """
        # Compute e-values for the alternative data
        if alternative_data.ndim == 1:
            # Single sample
            e_values = np.array([self.compute(alternative_data)])
        else:
            # Multiple samples
            e_values = np.array([self.compute(sample) for sample in alternative_data])
        
        # Use the e-power calculator from epower.py
        calculator = EPowerCalculator(e_power_config or self.e_power_calculator.config)
        return calculator.compute(e_values)
    
    def get_asy_efficiency(self) -> float:
        # Return the known asymptotic efficiencies from the paper
        if self.test_type == SymmetryType.FISHER:
            return 1.0  # Fisher's test has efficiency 1 (Section 6)
        elif self.test_type == SymmetryType.SIGN:
            return 2/np.pi  # Sign test has efficiency 2/pi
        elif self.test_type == SymmetryType.WILCOXON:
            return 3/np.pi  # Wilcoxon test has efficiency 3/pi
        else:
            return 0.0
