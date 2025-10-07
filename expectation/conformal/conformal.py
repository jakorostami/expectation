"""

This is an experimental module of the library trying to implement the conformal prediction framework but for e-values.
The module is not yet complete and is still under heavy development so use it with caution as no unit tests exist yet.

This module implements two approaches:
1. Mixture martingales (calibration=False)

2. Conformal normalized likelihood ratios (calibration=True)

References:
Conformal e-testing (2024) - Vovk, Nouretdinov, Gammerman
https://www.alrw.net/articles/29.pdf

Conformal e-prediction (2025) - Vovk
https://alrw.net/articles/26.pdf

"""

from typing import Optional, List, Tuple, Callable
import numpy as np
from numpy.typing import ArrayLike
import warnings

warnings.warn("The conformal module is experimental and may change in future releases.\nIt comes with no validation and is not tested by the core team.", UserWarning)

from expectation.modules.martingales import (
    OneSidedNormalMixture, 
    TwoSidedNormalMixture
)


class ConformalEValue:
    """
    EXPERIMENTIAL MODULE
    
    Implementation of conformal e-values using proper nonconformity e-measures with optional conformal calibration.
    
    Two modes of operation:
    
    Mode 1: Mixture Martingales (use_conformal_calibration=False, default)
        
    Mode 2: Conformal Calibration (use_conformal_calibration=True)
    """
    
    def __init__(self, 
                 nonconformity_type: str = "normal",
                 is_one_sided: bool = True,
                 v_opt: float = 1.0,
                 alpha_opt: float = 0.05,
                 use_conformal_calibration: bool = False,
                 null_density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 alt_density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 allow_infinite: bool = False):
        """
        Initialize conformal e-value generator.
        
        Args:
            nonconformity_type: Type of mixture ("normal", "beta_binomial", etc.)
                               Only used when use_conformal_calibration=False
            is_one_sided: One-sided vs two-sided test
            v_opt: Variance parameter for mixture optimization
            alpha_opt: Significance level for mixture optimization
            use_conformal_calibration: Use conformal normalization (equation 12)
            null_density: Density function for null hypothesis p_0(x)
                         Required when use_conformal_calibration=True
            alt_density: Density function for alternative hypothesis p_1(x)
                        Required when use_conformal_calibration=True
            allow_infinite: Allow infinite e-values
            
        Example (Mixture Martingales):
            >>> conf = ConformalEValue(nonconformity_type="normal")
            >>> e_value = conf.update(np.random.normal(0, 1, 10))
            
        Example (Conformal Calibration):
            >>> import scipy.stats
            >>> null = lambda x: scipy.stats.norm.pdf(x, 0, 1)
            >>> alt = lambda x: scipy.stats.norm.pdf(x, 0.5, 1)
            >>> conf = ConformalEValue(
            ...     use_conformal_calibration=True,
            ...     null_density=null,
            ...     alt_density=alt
            ... )
            >>> e_value = conf.update(np.random.normal(0.5, 1, 10))
        """
        warnings.warn("EXPERIMENTAL MODULE - USE WITH CAUTION.", UserWarning, stacklevel=2)

        self.use_conformal_calibration = use_conformal_calibration
        self.allow_infinite = allow_infinite
        self.v_opt = v_opt
        self.alpha_opt = alpha_opt
        
        if use_conformal_calibration:
            if null_density is None or alt_density is None:
                raise ValueError(
                    "When use_conformal_calibration=True, you must provide:\n"
                    "  - null_density: Callable[[np.ndarray], np.ndarray]\n"
                    "  - alt_density: Callable[[np.ndarray], np.ndarray]\n"
                    "Example:\n"
                    "  import scipy.stats\n"
                    "  null_density = lambda x: scipy.stats.norm.pdf(x, 0, 1)\n"
                    "  alt_density = lambda x: scipy.stats.norm.pdf(x, 0.5, 1)"
                )
            self.null_density = null_density
            self.alt_density = alt_density
            self._likelihood_ratios: List[float] = []
        else:
            if nonconformity_type == "normal":
                self.mixture = (OneSidedNormalMixture if is_one_sided 
                              else TwoSidedNormalMixture)(v_opt, alpha_opt)
            elif nonconformity_type == "beta_binomial":
                raise NotImplementedError("beta_binomial not yet implemented")
            else:
                raise ValueError(f"Unsupported nonconformity type: {nonconformity_type}")
        
        self.reset()
    
    def reset(self):
        if self.use_conformal_calibration:
            self._likelihood_ratios = []
        else:
            self._data: List[float] = []
            self._running_mean = 0.0
            self._running_var = 1.0
            self._n_samples = 0
    
    def _update_with_mixture(self, data: np.ndarray) -> float:
        batch_size = len(data)
        
        if self._n_samples == 0:
            # First batch - compare to null
            s = np.sqrt(batch_size) * np.mean(data)
            v = self.v_opt
        else:
            # Compare to running statistics
            batch_mean = np.mean(data)
            s = np.sqrt(batch_size) * (batch_mean - self._running_mean) 
            s /= np.sqrt(self._running_var + 1e-8)
            v = self.v_opt * (1 + 1/np.sqrt(self._n_samples))
        
        log_e_score = self.mixture.log_superMG(s, v)
        e_score = np.exp(log_e_score)
        
        # Update running statistics for next iteration
        for x in data:
            self._n_samples += 1
            delta = x - self._running_mean
            self._running_mean += delta / self._n_samples
            if self._n_samples > 1:
                delta2 = x - self._running_mean
                self._running_var = ((self._n_samples - 2) * self._running_var + 
                                   delta * delta2) / (self._n_samples - 1)
        
        return e_score
    
    def _update_with_calibration(self, data: np.ndarray) -> float:
        """
        Mode 2: Use conformal normalized likelihood ratios.
        
        Implements equation (12) from Vovk et al. (2024)
        """
        try:
            null_probs = self.null_density(data)
            alt_probs = self.alt_density(data)
            
            null_probs = np.maximum(null_probs, 1e-300)
            
            # Likelihood ratio: L_n = prod(alt / null)
            # TODO: use LikelihoodRatioEValue from hypothesistesting.py instead
            likelihood_ratio = float(np.prod(alt_probs / null_probs))
            
            likelihood_ratio = np.clip(likelihood_ratio, 1e-300, 1e300)
            
        except Exception as e:
            warnings.warn(f"Error computing likelihood ratio: {e}. Returning 1.0")
            likelihood_ratio = 1.0
        
        self._likelihood_ratios.append(likelihood_ratio)
        
        # Apply conformal normalization (equation 12)
        # E_n = L_n / mean(L_1,...,L_n)
        mean_lr = np.mean(self._likelihood_ratios)
        conformal_e_value = likelihood_ratio / max(mean_lr, 1e-10)
        
        return float(conformal_e_value)
    
    def update(self, new_data: ArrayLike) -> float:
        new_data = np.asarray(new_data).flatten()
        
        if len(new_data) == 0:
            raise ValueError("Data batch cannot be empty")
        
        if self.use_conformal_calibration:
            e_value = self._update_with_calibration(new_data)
        else:
            e_value = self._update_with_mixture(new_data)
        
        if not self.allow_infinite and np.isinf(e_value):
            raise ValueError(
                f"Infinite e-value detected: {e_value}\n"
                f"This may indicate numerical instability or extreme data.\n"
                f"Set allow_infinite=True to allow infinite values."
            )
        
        return e_value
    
    @property
    def n_samples(self) -> int:
        if self.use_conformal_calibration:
            return len(self._likelihood_ratios)
        else:
            return self._n_samples
    
    def get_likelihood_ratio_history(self) -> Optional[np.ndarray]:
        if self.use_conformal_calibration:
            return np.array(self._likelihood_ratios)
        return None
    
    def get_calibration_diagnostic(self) -> Optional[dict]:
        if not self.use_conformal_calibration:
            return None
        
        if len(self._likelihood_ratios) == 0:
            return {"n_samples": 0, "mean_lr": None, "std_lr": None}
        
        lrs = np.array(self._likelihood_ratios)
        
        return {
            "n_samples": len(lrs),
            "mean_lr": float(np.mean(lrs)),
            "std_lr": float(np.std(lrs)),
            "min_lr": float(np.min(lrs)),
            "max_lr": float(np.max(lrs)),
            "admissibility_check": abs(np.mean(lrs) - 1.0) < 0.1,  # Should be ~1
        }


class ConformalEPseudomartingale:
    """
    Implementation of conformal e-pseudomartingales (Section 3).
    
    Tracks the product S_n = E_1 * E_2 * ... * E_n and running maximum S_inf.
    Works with e-values from either calibration mode.
    """
    
    def __init__(self, 
                 initial_capital: float = 1.0,
                 allow_infinite: bool = False):
        self.initial_capital = initial_capital
        self.allow_infinite = allow_infinite
        self.reset()
        
    def reset(self):
        self._capital = self.initial_capital
        self._e_values: List[float] = []
        self._capital_history: List[float] = [self.initial_capital]
        self._max_capital = self.initial_capital
        
    def update(self, e_value: float) -> Tuple[float, float]:
        if not self.allow_infinite and np.isinf(e_value):
            raise ValueError("Infinite e-value detected and not allowed")
            
        self._capital *= e_value
        
        self._e_values.append(e_value)
        self._capital_history.append(self._capital)
        
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
        """Test if capital exceeds threshold."""
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