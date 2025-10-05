"""
Hypothesis testing with e-values by Ramdas & Wang (2025)
"""

from typing import Optional, Union, List, Callable
import numpy as np

from expectation.modules.martingales import (
    SequentialEValueCombiner,
    AllInCombiner,
    EmpiricallyAdaptiveCombiner,
    ConservativeCombiner,
    LogOptimalCombiner
)

from expectation.modules.hypothesistesting import EValueConfig, EProcess, EProcessConfig, BettingStrategy

class EProcessUpdater:
    def __init__(self, config: Union[EValueConfig, EProcessConfig]):
        """
        Initialize updater with appropriate combiner based on config.
        
        Args:
            config: Configuration specifying betting strategy
        """
        self.config = config
        self.combiner = self._create_combiner(config)
    
    def _create_combiner(self, config: Union[EValueConfig, EProcessConfig]) -> SequentialEValueCombiner:
        if not isinstance(config, EProcessConfig):
            return AllInCombiner()  # backward compatibility
        
        strategy = config.betting_strategy
        
        if strategy == BettingStrategy.ALL_IN:
            return AllInCombiner()
        elif strategy == BettingStrategy.EMPIRICALLY_ADAPTIVE:
            return EmpiricallyAdaptiveCombiner(gamma=config.gamma)
        elif strategy == BettingStrategy.CONSERVATIVE:
            return ConservativeCombiner(lambda_fixed=config.conservative_lambda)
        elif strategy == BettingStrategy.LOG_OPTIMAL:
            # Will be set later via set_log_optimal_expectation
            return None
        else:
            return AllInCombiner()
    
    def set_log_optimal_expectation(self, expectation_func: Callable[[float, List[float], int], float]):
        """
        Set expectation function for log-optimal strategy.
        
        Args:
            expectation_func: function(lambda, past_e_values, t) -> E_Q[log((1-lambda) + lambdaE_t) | F_{t-1}]
        """
        if isinstance(self.config, EProcessConfig) and self.config.betting_strategy == BettingStrategy.LOG_OPTIMAL:
            self.combiner = LogOptimalCombiner(expectation_func)
    
    def update(self, process: EProcess, e_value: float) -> None:
        """
        Update the e-process with a new sequential e-value.
        
        Implements: M_t = M_{t-1} * ((1 - lambda_t) + lambda_t * E_t)
        """
        if not self.config.allow_infinite and np.isinf(e_value):
            raise ValueError(f"Infinite e-value {e_value} not allowed by config")
        
        process.values.append(e_value)
        process.total_samples += 1
        
        if self.combiner is None:
            raise ValueError("Combiner not initialized. Call set_log_optimal_expectation if using LOG_OPTIMAL strategy.")
        
        t = len(process.values)
        lambda_t = self.combiner.compute_lambda(process.values[:-1], t)
        process.lambdas.append(lambda_t)
        
        increment = self.combiner.compute_increment(e_value, lambda_t)
        
        current_value = process.process_values[-1] if process.process_values else 1.0
        new_value = current_value * increment
        process.process_values.append(new_value)
        process.cumulative_value = new_value
        
        if increment > 0:
            log_increment = self.combiner.compute_log_increment(e_value, lambda_t)
            current_log = process.log_process_values[-1] if process.log_process_values else 0.0
            process.log_process_values.append(current_log + log_increment)
        else:
            process.log_process_values.append(-np.inf)
    
    def get_current_value(self, process: EProcess) -> float:
        return process.process_values[-1] if process.process_values else 1.0
    
    def get_max_value(self, process: EProcess) -> float:
        return max(process.process_values) if process.process_values else 1.0
    
    def is_significant(self, process: EProcess, alpha: Optional[float] = None) -> bool:
        """
        Test significance using Ville's inequality.
        """
        alpha = alpha or process.config.significance_level
        threshold = 1 / alpha
        return self.get_max_value(process) >= threshold
    
    def get_stopping_time(self, process: EProcess, alpha: Optional[float] = None) -> Optional[int]:
        """
        Find first time the e-process crosses 1/α.
        """
        alpha = alpha or process.config.significance_level
        threshold = 1 / alpha
        
        for t, value in enumerate(process.process_values[1:], 1):
            if value >= threshold:
                return t
        return None
    
    def compute_p_process(self, process: EProcess) -> List[float]:
        """
        Compute the p-process from Definition 7.10.
        """
        if not process.process_values:
            return []
        
        p_values = []
        for t in range(len(process.process_values)):
            values_up_to_t = process.process_values[:t+1]
            min_inv = min(1/m if m > 0 else 1.0 for m in values_up_to_t)
            p_values.append(min(1.0, min_inv))
        
        return p_values
    
    def compute_empirical_e_power(self, process: EProcess) -> float:
        """
        Compute empirical e-power from Definition 3.11.
        """
        if not process.values:
            return 0.0
        
        log_terms = [np.log(e) for e in process.values if e > 0]
        return np.mean(log_terms) if log_terms else -np.inf
    
    def compute_asymptotic_growth_rate(self, process: EProcess, min_samples: int = 20) -> Optional[float]:
        """
        Estimate asymptotic growth rate from Section 7.3.2.
        """
        if len(process.values) < min_samples:
            return None
        
        t = len(process.values)
        if process.log_process_values:
            return process.log_process_values[-1] / t
        else:
            if process.cumulative_value > 0:
                return np.log(process.cumulative_value) / t
            return -np.inf
    
    def get_summary(self, process: EProcess) -> dict:
        return {
            "n_samples": process.total_samples,
            "current_value": self.get_current_value(process),
            "max_value": self.get_max_value(process),
            "is_significant": self.is_significant(process),
            "stopping_time": self.get_stopping_time(process),
            "empirical_e_power": self.compute_empirical_e_power(process),
            "asymptotic_growth_rate": self.compute_asymptotic_growth_rate(process),
            "mean_lambda": np.mean(process.lambdas) if process.lambdas else None,
            "mean_e_value": np.mean(process.values) if process.values else None,
            "combiner_type": self.combiner.__class__.__name__ if self.combiner else "None"
        }