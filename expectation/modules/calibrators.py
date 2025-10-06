"""
Based on:
Hypothesis Testing with E-values, Section 2.3 (Ramdas & Wang, 2025)

Implements p-to-e and e-to-p calibrators as described in chapter 2. 
These are explicit formulas from the book.

There are more calibrators in the book but there are certain requirements associated with them which
requires abit more complexity. Feel free to extend this module by raising an issue and submitting a PR.
"""

from enum import Enum
from typing import Union
import numpy as np
from numpy.typing import NDArray


class PToECalibratorType(str, Enum):
    POWER = "power"
    MIXTURE = "mixture"
    LINEAR = "linear"
    SHAFER = "shafer"
    LOGARITHMIC = "logarithmic"


class EToPCalibrator:
    """
    E-to-p calibrator: g(e) = min(1, 1/e)
    
    This is the ONLY admissible e-to-p calibrator (Proposition 2.4).
    Follows from Markov inequality.
    
    Reference:
        Section 2.3, Proposition 2.4
    
    Examples:
        >>> calibrator = EToPCalibrator()
        >>> calibrator(20)  # e-value of 20
        0.05
        >>> calibrator(np.array([10, 20, 100]))
        array([0.1 , 0.05, 0.01])
    """
    
    def __call__(self, e: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Convert e-value to p-value.
        
        Args:
            e: E-value in [0, inf]
            
        Returns:
            P-value in [0, 1]
        """
        e = np.asarray(e)
        result = np.minimum(1.0, 1.0 / np.maximum(e, 1e-300))
        return result if result.shape else float(result)


class PToECalibrator:
    """
    P-to-e calibrator with multiple types from Section 2.3.
    
    Implements the 5 admissible p-to-e calibrators from equations 2.1-2.5:
    - POWER: k * p^(k-1) for k in (0,1)
    - MIXTURE: (1 - p + p log p) / (p(−log p)^2) 
    - LINEAR: 2(1 - p)
    - SHAFER: p^(-1/2) - 1
    - LOGARITHMIC: -log(p)
    
    Reference:
        Section 2.3, Equations (2.1)-(2.5)
        Proposition 2.5
    
    Examples:
        >>> # Default is Shafer's calibrator (recommended in book)
        >>> calibrator = PToECalibrator()
        >>> calibrator(0.01)  # p-value of 0.01
        9.0
        
        >>> # Use power calibrator with custom kappa
        >>> calibrator = PToECalibrator(calibrator_type=PToECalibratorType.POWER, kappa=0.5)
        >>> calibrator(0.01)
        50.0
        
        >>> # Apply to arrays
        >>> calibrator = PToECalibrator(calibrator_type=PToECalibratorType.LINEAR)
        >>> calibrator(np.array([0.01, 0.05, 0.1]))
        array([1.98, 1.9 , 1.8 ])
    """
    
    def __init__(
        self, 
        calibrator_type: PToECalibratorType = PToECalibratorType.SHAFER,
        kappa: float = 0.5
    ):
        """
        Initialize p-to-e calibrator.
        
        Args:
            calibrator_type: Type of calibrator to use
            kappa: Power parameter only for POWER type, must be in (0, 1) and is considered to be a
            tuning parameter that controls how aggressively the p-values are converted to e-values. 
        """
        self.calibrator_type = calibrator_type
        self.kappa = kappa
        
        if calibrator_type == PToECalibratorType.POWER and not (0 < kappa < 1):
            raise ValueError(f"kappa must be in (0, 1), got {kappa}")
        
        self._calibrator_methods = {
            PToECalibratorType.POWER: self._power,
            PToECalibratorType.MIXTURE: self._mixture,
            PToECalibratorType.LINEAR: self._linear,
            PToECalibratorType.SHAFER: self._shafer,
            PToECalibratorType.LOGARITHMIC: self._logarithmic,
        }
        
        self._compute_method = self._calibrator_methods[calibrator_type]
    
    def __call__(self, p: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Apply p-to-e calibrator.
        
        Args:
            p: P-value in [0, inf)
            
        Returns:
            E-value in [0, inf]
        """
        return self._compute_method(p)
    
    def _power(self, p: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Power form: f(p) = k * p^(k-1) for k in (0,1)
        
        Reference: Equation (2.1)
        """
        p = np.asarray(p)
        result = np.where(
            p <= 1,
            self.kappa * np.power(np.maximum(p, 1e-300), self.kappa - 1),
            0.0
        )
        result = np.where(p == 0, np.inf, result)
        return result if result.shape else float(result)
    
    def _mixture(self, p: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Mixture of powers: f(p) = (1 - p + p log p) / (p(−log p)^2)
        
        Reference: Equation (2.2)
        """
        p = np.asarray(p)
        eps = 1e-300
        p_safe = np.maximum(p, eps)
        
        log_p = np.log(p_safe)
        numerator = 1 - p + p * log_p
        denominator = p_safe * (-log_p)**2
        
        result = np.where(p <= 1, numerator / denominator, 0.0) # TODO: this can be unstable for p close to 1, need to fix
        result = np.where(p == 0, np.inf, result)
        return result if result.shape else float(result)
    
    def _linear(self, p: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Linear: f(p) = 2(1 - p)
        
        Reference: Equation (2.3)
        """
        p = np.asarray(p)
        result = np.where(p <= 1, 2.0 * (1 - p), 0.0)
        return result if result.shape else float(result)
    
    def _shafer(self, p: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Shafers calibrator: f(p) = p^(-1/2) - 1
        
        Reference: Equation (2.4), Table 2.19
        """
        p = np.asarray(p)
        eps = 1e-300
        p_safe = np.maximum(p, eps)
        
        result = np.where(p <= 1, np.power(p_safe, -0.5) - 1, 0.0)
        result = np.where(p == 0, np.inf, result)
        return result if result.shape else float(result)
    
    def _logarithmic(self, p: Union[float, NDArray]) -> Union[float, NDArray]:
        """
        Logarithmic: f(p) = -log(p)
        
        Reference: Equation (2.5)
        """
        p = np.asarray(p)
        eps = 1e-300
        p_safe = np.maximum(p, eps)
        
        result = np.where(p <= 1, -np.log(p_safe), 0.0)
        result = np.where(p == 0, np.inf, result)
        return result if result.shape else float(result)