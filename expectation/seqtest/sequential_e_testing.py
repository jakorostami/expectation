from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, List, Dict
import numpy as np
from pydantic import BaseModel, Field
import pandas as pd

from expectation.modules.hypothesistesting import (
    Hypothesis, HypothesisType, EValueConfig, EProcess, 
    LikelihoodRatioEValue
)

from expectation.modules.orderstatistics import (
    StaticOrderStatistics
)
from expectation.modules.quantiletest import (
    QuantileABTest
)

class TestType(str, Enum):
    """Types of sequential tests available."""
    MEAN = "mean"
    QUANTILE = "quantile"
    VARIANCE = "variance"
    PROPORTION = "proportion"

class AlternativeType(str, Enum):
    """Types of alternative hypotheses."""
    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"

class SequentialTestResult(BaseModel):
    """Results from a sequential test."""
    reject_null: bool
    e_value: float
    e_process: EProcess
    sample_size: int
    p_value: Optional[float] = None
    confidence_bounds: Optional[tuple[float, float]] = None
    test_type: TestType
    alternative: AlternativeType
    timestamp: float = Field(default_factory=lambda: np.datetime64('now').astype(float))
    
    class Config:
        arbitrary_types_allowed = True

class SequentialTest:
    """
    User-friendly interface for sequential testing using e-values and e-processes.
    
    Examples:
    --------
    # Mean test
    >>> test = SequentialTest(test_type="mean", 
                             null_value=0, 
                             alternative="greater")
    >>> result = test.update([1.2, 0.8, 1.5])
    >>> print(f"Reject null: {result.reject_null}")
    
    # Proportion test with custom config
    >>> config = EValueConfig(significance_level=0.01, allow_infinite=False)
    >>> prop_test = SequentialTest(test_type="proportion",
                                  null_value=0.5,
                                  alternative="two_sided",
                                  config=config)
    >>> result = prop_test.update([1, 0, 1, 1, 0])
    
    # Quantile test
    >>> quant_test = SequentialTest(test_type="quantile",
                                   quantile=0.5,
                                   null_value=10)
    >>> result = quant_test.update([8, 12, 9, 11])
    """
    
    def __init__(
        self,
        test_type: Union[TestType, str],
        null_value: float,
        alternative: Union[AlternativeType, str] = "two_sided",
        quantile: Optional[float] = None,
        config: Optional[EValueConfig] = None
    ):
        """
        Initialize sequential test.
        
        Parameters:
        -----------
        test_type : str
            Type of test to perform ("mean", "quantile", "variance", "proportion")
        null_value : float
            Null hypothesis value
        alternative : str, optional
            Alternative hypothesis type ("two_sided", "greater", "less")
        quantile : float, optional
            Quantile to test (required for quantile tests)
        config : EValueConfig, optional
            Configuration for e-values and testing
        """
        self.test_type = TestType(test_type)
        self.alternative = AlternativeType(alternative)
        self.null_value = null_value
        self.quantile = quantile
        self.config = config or EValueConfig()
        
        # Validate parameters
        if self.test_type == TestType.QUANTILE and quantile is None:
            raise ValueError("Quantile parameter required for quantile tests")
        if self.test_type == TestType.PROPORTION and not (0 <= null_value <= 1):
            raise ValueError("Null value must be between 0 and 1 for proportion tests")
        
        # Create null hypothesis
        self.null_hypothesis = Hypothesis(
            name=f"{self.test_type.value.title()} Test",
            description=f"H0: {self.test_type.value} = {self.null_value}",
            type=HypothesisType.SIMPLE
        )
        
        # Initialize e-value calculator and e-process
        self._setup_evaluator()
        self.e_process = EProcess(config=self.config)
        
        # Initialize state
        self._reset_state()
    
    def _reset_state(self):
        """Reset test state."""
        self.data = []
        self.n_samples = 0
    
    def _setup_evaluator(self):
        """Set up appropriate e-value calculator for the test type."""
        if self.test_type == TestType.MEAN:
            # Define densities for likelihood ratio
            def null_density(x): 
                return stats.norm.pdf(x, loc=self.null_value)
            
            def alt_density(x):
                # Alternative based on direction
                if self.alternative == AlternativeType.TWO_SIDED:
                    return 0.5 * (stats.norm.pdf(x, loc=self.null_value - 1) + 
                                stats.norm.pdf(x, loc=self.null_value + 1))
                elif self.alternative == AlternativeType.GREATER:
                    return stats.norm.pdf(x, loc=self.null_value + 1)
                else:  # LESS
                    return stats.norm.pdf(x, loc=self.null_value - 1)
            
            self.e_calculator = LikelihoodRatioEValue(
                null_hypothesis=self.null_hypothesis,
                null_density=null_density,
                alt_density=alt_density,
                config=self.config
            )
            
        elif self.test_type == TestType.PROPORTION:
            def null_density(x):
                return stats.bernoulli.pmf(x, self.null_value)
            
            def alt_density(x):
                if self.alternative == AlternativeType.TWO_SIDED:
                    p1 = max(0.1, self.null_value - 0.2)
                    p2 = min(0.9, self.null_value + 0.2)
                    return 0.5 * (stats.bernoulli.pmf(x, p1) + stats.bernoulli.pmf(x, p2))
                elif self.alternative == AlternativeType.GREATER:
                    p = min(0.9, self.null_value + 0.2)
                    return stats.bernoulli.pmf(x, p)
                else:  # LESS
                    p = max(0.1, self.null_value - 0.2)
                    return stats.bernoulli.pmf(x, p)
            
            self.e_calculator = LikelihoodRatioEValue(
                null_hypothesis=self.null_hypothesis,
                null_density=null_density,
                alt_density=alt_density,
                config=self.config
            )
            
        elif self.test_type == TestType.QUANTILE:
            # Will be initialized with first batch of data
            self.e_calculator = None
            
        elif self.test_type == TestType.VARIANCE:
            def null_density(x):
                return stats.norm.pdf(x, scale=np.sqrt(self.null_value))
            
            def alt_density(x):
                if self.alternative == AlternativeType.TWO_SIDED:
                    return 0.5 * (stats.norm.pdf(x, scale=np.sqrt(self.null_value * 0.5)) + 
                                stats.norm.pdf(x, scale=np.sqrt(self.null_value * 2)))
                elif self.alternative == AlternativeType.GREATER:
                    return stats.norm.pdf(x, scale=np.sqrt(self.null_value * 2))
                else:  # LESS
                    return stats.norm.pdf(x, scale=np.sqrt(self.null_value * 0.5))
            
            self.e_calculator = LikelihoodRatioEValue(
                null_hypothesis=self.null_hypothesis,
                null_density=null_density,
                alt_density=alt_density,
                config=self.config
            )
    
    def update(self, new_data: Union[List, np.ndarray, pd.Series]) -> SequentialTestResult:
        """
        Update test with new data.
        
        Parameters:
        -----------
        new_data : array-like
            New observations to include in the test
            
        Returns:
        --------
        SequentialTestResult
            Updated test results
        """
        # Convert input to numpy array
        new_data = np.asarray(new_data)
        
        # Update state
        self.data.extend(new_data)
        self.n_samples += len(new_data)
        
        # Special handling for quantile test
        if self.test_type == TestType.QUANTILE:
            if self.e_calculator is None:
                # Initialize quantile test with first batch
                self.e_calculator = QuantileABTest(
                    quantile_p=self.quantile,
                    t_opt=len(new_data),
                    alpha_opt=self.config.significance_level,
                    arm1_os=StaticOrderStatistics(new_data),
                    arm2_os=StaticOrderStatistics([self.null_value])
                )
            
            # Compute e-value using quantile test
            e_value = np.exp(-self.e_calculator.log_superMG_lower_bound())
        else:
            # Compute e-value using likelihood ratio
            result = self.e_calculator.test(new_data)
            e_value = result.value
        
        # Update e-process
        self.e_process.update(e_value)
        
        return SequentialTestResult(
            reject_null=self.e_process.is_significant(),
            e_value=e_value,
            e_process=self.e_process,
            sample_size=self.n_samples,
            p_value=1/self.e_process.cumulative_value if self.e_process.cumulative_value > 1 else 1.0,
            test_type=self.test_type,
            alternative=self.alternative
        )
    
    def reset(self):
        """Reset test to initial state."""
        self._reset_state()
        self.e_process = EProcess(config=self.config)