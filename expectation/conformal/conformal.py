"""
This is an experimental module of the library trying to implement the conformal prediction framework but for e-values.
The module is not yet complete and is still under heavy development so use it with caution as no unit tests exist yet.

Conformal e-testing (2024) - Vladimir Vovk, Ilia Nouretdinov, and Alex Gammerman
https://www.alrw.net/articles/29.pdf

"""


from typing import Optional, List, Tuple, Callable, Union
import numpy as np
from numpy.typing import ArrayLike

from expectation.modules.martingales import (
    BetaBinomialMixture, OneSidedNormalMixture, 
    TwoSidedNormalMixture, GammaExponentialMixture
)

class ConformalEValue:
    """
    Implementation of conformal e-values using proper nonconformity e-measures.
    Fixed to properly handle scaling and normalization.
    """
    def __init__(self, 
                 nonconformity_type: str = "normal",
                 is_one_sided: bool = True,
                 v_opt: float = 1.0,
                 alpha_opt: float = 0.05,
                 allow_infinite: bool = False):
        self.allow_infinite = allow_infinite
        self.v_opt = v_opt
        self.alpha_opt = alpha_opt
        
        # Initialize mixture martingale
        if nonconformity_type == "normal":
            self.mixture = (OneSidedNormalMixture if is_one_sided 
                          else TwoSidedNormalMixture)(v_opt, alpha_opt)
        else:
            raise ValueError(f"Unsupported nonconformity type: {nonconformity_type}")
            
        self.reset()
    
    def reset(self):
        self._data: List[float] = []
        self._running_mean = 0.0
        self._running_var = 1.0
        self._n_samples = 0
        
    def _update_statistics(self, new_data: np.ndarray):
        for x in new_data:
            self._n_samples += 1
            delta = x - self._running_mean
            self._running_mean += delta / self._n_samples
            if self._n_samples > 1:
                delta2 = x - self._running_mean
                self._running_var = ((self._n_samples - 2) * self._running_var + 
                                   delta * delta2) / (self._n_samples - 1)
    
    def compute_nonconformity_score(self, data: np.ndarray) -> float:
        batch_size = len(data)
        
        if self._n_samples == 0:
            # First batch
            s = np.sqrt(batch_size) * np.mean(data)
            v = self.v_opt
        else:
            # Compute standardized difference
            batch_mean = np.mean(data)
            s = np.sqrt(batch_size) * (batch_mean - self._running_mean) 
            s /= np.sqrt(self._running_var + 1e-8)  # Add small constant for stability
            v = self.v_opt * (1 + 1/np.sqrt(self._n_samples))
        
        # Compute e-score using mixture
        log_e_score = self.mixture.log_superMG(s, v)
        e_score = np.exp(log_e_score)
        
        if not self.allow_infinite and np.isinf(e_score):
            raise ValueError("Infinite e-value detected and not allowed")
            
        return e_score
        
    def update(self, new_data: ArrayLike) -> float:
        new_data = np.asarray(new_data)
        
        # Compute e-value before updating statistics
        e_value = self.compute_nonconformity_score(new_data)
        
        # Update running statistics
        self._update_statistics(new_data)
        
        return e_value
    
    @property
    def n_samples(self) -> int:
        """
        
        Number of samples processed.
        
        """
        return self._n_samples
    

class ConformalEPseudomartingale:
    """
    Implementation of conformal e-pseudomartingales as described in Section 3 of the paper.
    Tracks the product of conformal e-values and implements compound betting strategies.
    """
    def __init__(self, 
                 initial_capital: float = 1.0,
                 allow_infinite: bool = False):
        self.initial_capital = initial_capital
        self.allow_infinite = allow_infinite
        self.reset()
        
    def reset(self):
        self._capital = self.initial_capital  # Current capital Sₙ
        self._e_values: List[float] = []  # History of e-values
        self._capital_history: List[float] = [self.initial_capital]  # History of Sₙ
        self._max_capital = self.initial_capital  # Running maximum S*_∞
        
    def update(self, e_value: float) -> Tuple[float, float]:

        if not self.allow_infinite and np.isinf(e_value):
            raise ValueError("Infinite e-value detected and not allowed")
            
        # Update capital by multiplying with e-value
        self._capital *= e_value
        
        # Update histories
        self._e_values.append(e_value)
        self._capital_history.append(self._capital)
        
        # Update running maximum
        self._max_capital = max(self._max_capital, self._capital)
        
        return self._capital, self._max_capital
    
    def compound_bet(self, e_values: ArrayLike) -> float:
        e_values = np.asarray(e_values)
        return float(self.initial_capital * np.prod(e_values))
    
    @property
    def capital(self) -> float:
        return self._capital
    
    @property
    def max_capital(self) -> float:
        return self._max_capital
    
    @property
    def n_steps(self) -> int:
        return len(self._e_values)
    
    def get_history(self) -> Tuple[np.ndarray, np.ndarray]:
        return (np.array(self._e_values), 
                np.array(self._capital_history))
    
    def test_threshold(self, threshold: float, use_max: bool = True) -> bool:
        test_value = self._max_capital if use_max else self._capital
        return test_value >= threshold

class TruncatedEPseudomartingale(ConformalEPseudomartingale):
    def __init__(self,
                 initial_capital: float = 1.0,
                 min_capital: float = 1e-10,
                 allow_infinite: bool = False):
        super().__init__(initial_capital, allow_infinite)
        self.min_capital = min_capital
        
    def update(self, e_value: float) -> Tuple[float, float]:
        capital, max_cap = super().update(e_value)
        
        # Apply truncation
        if capital < self.min_capital:
            self._capital = self.min_capital
            self._capital_history[-1] = self.min_capital
            
        return self._capital, self._max_capital