from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import scipy.stats
from expectation.agentic.models import TestState, AgentConfig, ECalibrationMethod
from expectation.agentic.agents import EvaluationAgent
from expectation.modules.epower import EPowerCalculator

class StandardEvaluatorAgent(EvaluationAgent):
    """
    Standard implementation of the evaluator agent.
    
    This agent evaluates test results and forms conclusions, using
    various methods for combining e-values.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.e_power_calculator = EPowerCalculator()
    
    def process(self, state: TestState, data: Dict[str, pd.DataFrame]) -> Tuple[bool, float, str]:

        if not state.results:
            return None, 0.0, "No test results available for evaluation"
        
        e_values = [result.e_value for result in state.results]
        
        combined_e_value = self._combine_e_values(e_values)
        
        e_power_result = self.e_power_calculator.compute(np.array(e_values))
        
        alpha = self.config.significance_level
        is_significant = combined_e_value >= 1/alpha
        
        confidence = 1.0 - (1.0 / combined_e_value) if combined_e_value > 1.0 else 0.0
        
        method_description = self._get_method_description()
        
        if is_significant:
            conclusion = True
            reasoning = (
                f"The combined evidence ({method_description}, e-value: {combined_e_value:.4f}) "
                f"from {len(state.results)} tests is sufficient to reject the null hypothesis "
                f"with {confidence:.2%} confidence. The e-power of {e_power_result.e_power:.4f} "
                f"indicates strong evidence growth."
            )
        elif state.iteration >= self.config.max_iterations:
            conclusion = False
            reasoning = (
                f"After {state.iteration} iterations, the combined evidence "
                f"({method_description}, e-value: {combined_e_value:.4f}) is insufficient "
                f"to reject the null hypothesis. The e-power of {e_power_result.e_power:.4f} "
                f"indicates limited evidence growth."
            )
        else:
            conclusion = None
            reasoning = (
                f"Current evidence ({method_description}, e-value: {combined_e_value:.4f}) "
                f"is inconclusive. Continuing testing with e-power of {e_power_result.e_power:.4f}."
            )
        
        return conclusion, confidence, reasoning
    
    def _combine_e_values(self, e_values: List[float]) -> float:

        if not e_values:
            return 1.0
            
        if self.config.combine_method == "product":
            return np.prod(e_values)
            
        elif self.config.combine_method == "fisher":
            # Fisher's method for combining p-values, converted to e-value
            p_values = [min(1.0, 1.0/e) for e in e_values]
            chi_square = -2 * np.sum(np.log(p_values))
            degrees_freedom = 2 * len(p_values)
            
            # Calculate the combined p-value from the chi-square distribution
            combined_p_value = 1.0 - scipy.stats.chi2.cdf(chi_square, degrees_freedom)
            
            # Convert back to e-value (ensuring no division by zero)
            return 1.0 / max(combined_p_value, 1e-10)
            
        else:  # e_calibrator
            p_values = [min(1.0, 1.0/e) for e in e_values]
            
            if self.config.e_calibrator_method == ECalibrationMethod.KAPPA:
                # Kappa-calibrator (simpler, but parameter-dependent)
                kappa = self.config.kappa
                e_calibrated = [kappa * (p ** (kappa - 1)) for p in p_values]
                return np.prod(e_calibrated)
                
            else:  # integral calibrator
                # Integral-calibrator (more sophisticated)
                # Formula: (1 - p + p*log(p))/(p*(-log(p))^2)
                e_calibrated = []
                for p in p_values:
                    if p <= 0 or p >= 1:
                        e_calibrated.append(1.0)  # Default for boundary cases
                    else:
                        numerator = 1 - p + p * np.log(p)
                        denominator = p * ((-np.log(p))**2)
                        e_calibrated.append(numerator / denominator)
                
                return np.prod(e_calibrated)
    
    def _get_method_description(self) -> str:

        if self.config.combine_method == "product":
            return "product method"
        elif self.config.combine_method == "fisher":
            return "Fisher's method"
        else:  # e_calibrator
            if self.config.e_calibrator_method == ECalibrationMethod.KAPPA:
                return f"κ-calibrator (κ={self.config.kappa})"
            else:
                return "integral e-calibrator"