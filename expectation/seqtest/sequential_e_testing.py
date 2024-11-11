from enum import Enum
from typing import Optional, Union, List
import numpy as np
from pydantic import BaseModel, Field
import pandas as pd

from expectation.modules.hypothesistesting import (
    Hypothesis, HypothesisType, EValueConfig, EProcess, 
    LikelihoodRatioEValue
)

from expectation.modules.martingales import (
    BetaBinomialMixture, OneSidedNormalMixture, 
    TwoSidedNormalMixture, GammaExponentialMixture
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
        self.history = []
        
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
            # Using normal mixture directly
            mixture = (TwoSidedNormalMixture if self.alternative == AlternativeType.TWO_SIDED 
                    else OneSidedNormalMixture)(
                v_opt=1.0,  # Can be optimized
                alpha_opt=self.config.significance_level
            )
            
            def calculator(data):
                # For mean test:
                n = len(data)
                batch_mean = np.mean(data)
                # s should be √n * (mean - null_value) for proper scaling
                s = np.sqrt(n) * (batch_mean - self.null_value)
                v = 1.0  # Using standard normal scaling
                return np.exp(mixture.log_superMG(s, v))
                
            self.e_calculator = calculator
                
        elif self.test_type == TestType.PROPORTION:
            # Using beta-binomial mixture
            mixture = BetaBinomialMixture(
                v_opt=self.null_value * (1 - self.null_value),
                alpha_opt=self.config.significance_level,
                g=self.null_value,
                h=1 - self.null_value,
                is_one_sided=self.alternative != AlternativeType.TWO_SIDED
            )
            
            def calculator(data):
                s = np.sum(data - self.null_value)
                v = len(data) * self.null_value * (1 - self.null_value)
                return np.exp(mixture.log_superMG(s, v))
                
            self.e_calculator = calculator

        # TODO: This works but is it properly done? Version 1.    
        elif self.test_type == TestType.VARIANCE:
            # Using gamma-exponential mixture
            mixture = GammaExponentialMixture(
                v_opt=self.null_value,
                alpha_opt=self.config.significance_level,
                c=np.sqrt(self.null_value)
            )
            
            def calculator(data):
                n = len(data)
                centered_data = data - np.mean(data)
                s = np.sum(centered_data**2) - self.null_value * n
                if self.alternative == AlternativeType.TWO_SIDED:
                    return max(np.exp(mixture.log_superMG(s, n)), 
                            np.exp(mixture.log_superMG(-s, n)))
                elif self.alternative == AlternativeType.LESS:
                    return np.exp(mixture.log_superMG(-s, n))
                else:  # GREATER
                    return np.exp(mixture.log_superMG(s, n))
                    
            self.e_calculator = calculator
                
        elif self.test_type == TestType.QUANTILE:
            self.e_calculator = None
            
        # TODO: Not clear which edge cases below is preferable for. Version 1 above uses direct mixture.    
        # elif self.test_type == TestType.VARIANCE:
        #     # Testing variance with gamma-exponential mixture
        #     mixture = GammaExponentialMixture(
        #         v_opt=self.null_value,  # baseline variance
        #         alpha_opt=self.config.significance_level,
        #         c=np.sqrt(self.null_value)  # scale parameter
        #     )
            
        #     def null_density(x):
        #         n = len(x)
        #         centered_data = x - np.mean(x)
        #         s = np.sum(centered_data**2) - self.null_value * n
        #         return np.exp(mixture.log_superMG(s, n))
            
        #     def alt_density(x):
        #         if self.alternative == AlternativeType.TWO_SIDED:
        #             n = len(x)
        #             centered_data = x - np.mean(x)
        #             s_upper = np.sum(centered_data**2) - self.null_value * n
        #             s_lower = -s_upper
        #             return max(np.exp(mixture.log_superMG(s_upper, n)), 
        #                     np.exp(mixture.log_superMG(s_lower, n)))
        #         elif self.alternative == AlternativeType.GREATER:
        #             n = len(x)
        #             centered_data = x - np.mean(x)
        #             s = np.sum(centered_data**2) - self.null_value * n
        #             return np.exp(mixture.log_superMG(s, n))
        #         else:  # LESS
        #             n = len(x)
        #             centered_data = x - np.mean(x)
        #             s = -(np.sum(centered_data**2) - self.null_value * n)
        #             return np.exp(mixture.log_superMG(s, n))
            
        #     evaluator = LikelihoodRatioEValue(
        #         null_hypothesis=self.null_hypothesis,
        #         null_density=null_density,
        #         alt_density=alt_density,
        #         config=self.config
        #     )

        #     def calculator(data):
        #         result = evaluator.test(data)
        #         return result.value
            
        #     self.e_calculator = calculator
    
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
        new_data = np.asarray(new_data)
        
        # Update state
        self.data.extend(new_data)
        self.n_samples += len(new_data)
        
        # Special handling for quantile test
        if self.test_type == TestType.QUANTILE:
            if self.e_calculator is None:
                self.e_calculator = QuantileABTest(
                    quantile_p=self.quantile,
                    t_opt=len(new_data),
                    alpha_opt=self.config.significance_level,
                    arm1_os=StaticOrderStatistics(new_data),
                    arm2_os=StaticOrderStatistics([self.null_value])
                )
            e_value = np.exp(-self.e_calculator.log_superMG_lower_bound())
        else:
            # Directly compute e-value using mixture calculator
            e_value = self.e_calculator(new_data)
        
        # Update e-process
        self.e_process.update(e_value)
        
        history_entry = {
            'step': len(self.history) + 1,
            'observations': new_data.tolist(),
            'eValue': e_value,
            'cumulativeEValue': self.e_process.cumulative_value,
            'rejectNull': self.e_process.is_significant(),
            'timestamp': np.datetime64('now').astype(float)
        }
        self.history.append(history_entry)
        
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

    def get_history_df(self) -> pd.DataFrame:
        """Get test history as a pandas DataFrame."""
        return pd.DataFrame(self.history)