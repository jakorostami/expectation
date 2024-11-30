"""
This is an experimental module of the library trying to implement the conformal prediction framework but for e-values.
The module is not yet complete and is still under heavy development so use it with caution as no unit tests exist yet.

Conformal e-testing (2024) - Vladimir Vovk, Ilia Nouretdinov, and Alex Gammerman
https://www.alrw.net/articles/29.pdf

"""


from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import ArrayLike

class ConformalEValue:
    """
    Implementation of conformal e-values as defined in Section 2 of the paper.
    Handles the computation of nonconformity e-scores and conformal e-values.
    """
    def __init__(self, 
                 nonconformity_measure: Optional[callable] = None,
                 allow_infinite: bool = False):
        """
        Initialize conformal e-value calculator.
        
        Args:
            nonconformity_measure: Optional custom nonconformity measure.
                If None, uses default based on likelihood ratios.
            allow_infinite: Whether to allow infinite e-values.
        """
        self._nonconformity_measure = nonconformity_measure
        self.allow_infinite = allow_infinite
        self._scores: List[float] = []
        self._n_samples = 0
        
    def compute_nonconformity_scores(self, data: ArrayLike) -> np.ndarray:
        """
        Compute nonconformity e-scores for given data.
        Ensures sum of scores <= 1 (admissibility condition).
        """
        data = np.asarray(data)
        if self._nonconformity_measure is not None:
            scores = self._nonconformity_measure(data)
        else:
            # Default measure based on empirical distribution
            scores = np.abs(data - np.median(data))
        
        # Normalize to ensure admissibility (sum <= 1)
        scores = scores / (np.sum(scores) + 1e-10)
        return scores
        
    def update(self, new_data: ArrayLike) -> float:
        """
        Update with new observations and compute conformal e-value.
        
        Args:
            new_data: New observations to process
            
        Returns:
            Conformal e-value for the new observations
        """
        new_data = np.asarray(new_data)
        n_new = len(new_data)
        
        # Compute nonconformity scores including new data
        all_data = np.concatenate([self._scores, new_data]) if self._scores else new_data
        scores = self.compute_nonconformity_scores(all_data)
        
        # Get e-value for new observations
        e_value = scores[-n_new:].mean()
        
        # Store scores and update count
        self._scores = scores.tolist()
        self._n_samples += n_new
        
        if not self.allow_infinite and np.isinf(e_value):
            raise ValueError("Infinite e-value detected and not allowed")
            
        return e_value
        
    def reset(self):
        """Reset internal state."""
        self._scores = []
        self._n_samples = 0
        
    @property 
    def n_samples(self) -> int:
        """Number of samples processed."""
        return self._n_samples

    def is_admissible(self) -> bool:
        """Check if current state satisfies admissibility condition."""
        return np.sum(self._scores) <= 1.0
    

class ConformalEPseudomartingale:
    """
    Implementation of conformal e-pseudomartingales as described in Section 3 of the paper.
    Tracks the product of conformal e-values and implements compound betting strategies.
    """
    def __init__(self, 
                 initial_capital: float = 1.0,
                 allow_infinite: bool = False):
        """
        Initialize conformal e-pseudomartingale.
        
        Args:
            initial_capital: Initial capital (S₀), defaults to 1
            allow_infinite: Whether to allow infinite values
        """
        self.initial_capital = initial_capital
        self.allow_infinite = allow_infinite
        self.reset()
        
    def reset(self):
        """Reset to initial state."""
        self._capital = self.initial_capital  # Current capital Sₙ
        self._e_values: List[float] = []  # History of e-values
        self._capital_history: List[float] = [self.initial_capital]  # History of Sₙ
        self._max_capital = self.initial_capital  # Running maximum S*_∞
        
    def update(self, e_value: float) -> Tuple[float, float]:
        """
        Update pseudomartingale with new e-value.
        
        Args:
            e_value: New conformal e-value
            
        Returns:
            Tuple of (current capital, maximum capital so far)
        """
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
        """
        Compute compound betting result for sequence of e-values.
        
        Args:
            e_values: Sequence of e-values to compound
            
        Returns:
            Final capital after compounding all e-values
        """
        e_values = np.asarray(e_values)
        return float(self.initial_capital * np.prod(e_values))
    
    @property
    def capital(self) -> float:
        """Current capital Sₙ."""
        return self._capital
    
    @property
    def max_capital(self) -> float:
        """Maximum capital achieved S*_∞."""
        return self._max_capital
    
    @property
    def n_steps(self) -> int:
        """Number of updates performed."""
        return len(self._e_values)
    
    def get_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get history of e-values and capitals.
        
        Returns:
            Tuple of (e_values array, capital history array)
        """
        return (np.array(self._e_values), 
                np.array(self._capital_history))
    
    def test_threshold(self, threshold: float, use_max: bool = True) -> bool:
        """
        Test if capital exceeds threshold.
        
        Args:
            threshold: Threshold value c > 1 to test against
            use_max: Whether to use maximum capital (True) or current capital (False)
            
        Returns:
            True if capital exceeds threshold
        """
        test_value = self._max_capital if use_max else self._capital
        return test_value >= threshold

class TruncatedEPseudomartingale(ConformalEPseudomartingale):
    """
    Implementation of truncated version that forces minimum capital level.
    This helps prevent numerical underflow and implements truncation 
    as discussed in the paper.
    """
    def __init__(self,
                 initial_capital: float = 1.0,
                 min_capital: float = 1e-10,
                 allow_infinite: bool = False):
        """
        Initialize truncated pseudomartingale.
        
        Args:
            initial_capital: Initial capital S₀
            min_capital: Minimum allowed capital level
            allow_infinite: Whether to allow infinite values
        """
        super().__init__(initial_capital, allow_infinite)
        self.min_capital = min_capital
        
    def update(self, e_value: float) -> Tuple[float, float]:
        """
        Update with truncation at minimum capital level.
        """
        capital, max_cap = super().update(e_value)
        
        # Apply truncation
        if capital < self.min_capital:
            self._capital = self.min_capital
            self._capital_history[-1] = self.min_capital
            
        return self._capital, self._max_capital