'''
Compute E-values Based on Z-statistic
Computes e-values using the z-statistic and the sample sizes only based on the test defining parameter phiS.

Paper references:
    - Grünwald, P., de Heide, R., & Koolen, W. (2024). Safe Testing.
    - Ly, A., et al. (2024). Safe Tests and Always-Valid Confidence Intervals.
'''
import numpy as np
from scipy import stats, integrate
from typing import Union, Tuple, Optional
import warnings

from expectation.seqtest.sequential_e_testing import AlternativeType

def CheckAndReturnsEsMinParameterSide(paramToCheck, alternative: str, es_min_name: str, 
                                      param_domain: Optional[str] = None):
    '''
    Checks consistency between the sided of the hypothesis and the minimal 
    clinically relevant effect size or savi test defining parameter. Throws an error if the one-sided hypothesis is incongruent

    Args:
    param_to_check: Parameter value to check
    es_min_name: Provides the name of the effect size. Either "meanDiffMin" for the z-test, "deltaMin" for
    the t-test, or "hrMin" for the logrank test
    paramDomain: Domain of the paramToCheck, typically, positive number

    Returns:
    float: Adjusted parameter value
    Raises:
    ValueError: If parameter constraints are violated

    '''
    if not isinstance(alternative, AlternativeType):
        raise ValueError(f"alternative must be an AlternativeType enum, got {type(alternative)}")
    
    valid_es_name = ["noName", "meanDiffMin", "phiS","deltaMin", "deltaS",
                     "hrMin", "thetaS", "deltaTrue", "g", "kappaG"]
    if es_min_name not in valid_es_name:
        raise ValueError(f"Invalid es_min_name. Must be one of the {valid_es_name}")

    if alternative == AlternativeType.TWO_SIDED:
        if es_min_name in ["meanDiffMin", "deltaMin", "deltaTrue"]:
            return abs(paramToCheck)
        return paramToCheck
    
    if es_min_name == "noName":
        param_name = None
    else:
        param_name = es_min_name

    error = None

    if param_name is None:
        param_name = "the savi test defining parameter"
        hyp_param_name = "test relevant parameter"
        param_domain = "unknown"

    elif param_name in ["phiS", "meanDiffMin"]: 
        hyp_param_name = "meanDiff"
        param_domain = "realNumbers" 

    elif param_name in ["deltaS", "deltaMin", "deltaTrue"]:  
        hyp_param_name = "delta"
        param_domain = "realNumbers"  
    
    elif param_name in ["thetaS", "hrMin"]:  
        hyp_param_name = "theta"
        param_domain = "positiveNumbers"  
        if paramToCheck < 0:
            error = "thetaS and hrMin must be positive"
    
    elif param_name == "g":
        hyp_param_name = "g"
        param_domain = "positiveNumbers"
        if paramToCheck < 0:
            error = "The parameter g must be positive"

    elif param_name == "kappaG":
        hyp_param_name = "kappaG"
        param_domain = "positiveNumbers"  
        if paramToCheck < 0:
            error = "The parameter kappaG must be positive"
    
    else:
        hyp_param_name = "testRelevantParameter"
    
    if error is not None:
        raise ValueError(error)
    
    if param_domain == "unknown":
        
        if alternative == AlternativeType.GREATER and paramToCheck < 0:
            warnings.warn(
                'The savi test defining parameter is incongruent with alternative "greater". '
                'This savi test parameter is made positive to compare H+: '
                'test-relevant parameter > 0 against H0 : test-relevant parameter = 0'
            )
            paramToCheck = -paramToCheck
        
        elif alternative == AlternativeType.LESS and paramToCheck > 0:
            warnings.warn(
                'The savi test defining parameter is incongruent with alternative "less". '
                'This savi test parameter is made negative to compare H-: '
                'test-relevant parameter < 0 against H0 : test-relevant parameter = 0'
            )
            paramToCheck = -paramToCheck
    
    elif param_domain == "realNumbers":
        if alternative == AlternativeType.GREATER and paramToCheck < 0:
            warnings.warn(
                f'{param_name} incongruent with alternative "greater". '
                f'{param_name} set to -{param_name} > 0 in order to compare H+: '
                f'{hyp_param_name} > 0 against H0 : {hyp_param_name} = 0'
            )
            paramToCheck = -paramToCheck
        
        elif alternative == AlternativeType.LESS and paramToCheck > 0:
            warnings.warn(
                f'{param_name} incongruent with alternative "less". '
                f'{param_name} set to -{param_name} < 0 in order to compare H-: '
                f'{hyp_param_name} < 0 against H0 : {hyp_param_name} = 0'
            )
            paramToCheck = -paramToCheck
    
    elif param_domain == "positiveNumbers":
        if alternative == AlternativeType.GREATER and paramToCheck < 1:
            warnings.warn(
                f'{param_name} incongruent with alternative "greater". '
                f'{param_name} set to 1/{param_name} > 1 in order to compare H+: '
                f'{hyp_param_name} > 1 against H0 : {hyp_param_name} = 1'
            )
            paramToCheck = 1 / paramToCheck
        
        elif alternative == AlternativeType.LESS and paramToCheck > 1:
            warnings.warn(
                f'{param_name} incongruent with alternative "less". '
                f'{param_name} set to 1/{param_name} < 1 in order to compare H-: '
                f'{hyp_param_name} < 1 against H0 : {hyp_param_name} = 1'
            )
            paramToCheck = 1 / paramToCheck
    
    return float(paramToCheck)
    


class SaviZtestStat:
    def __init__(self, parameter: float, n1: int, 
                 paired: bool = False,sigma: float = 1.0,  n2: Optional[int] = None, 
                 etype: str = "ecauchy", 
                 alternative: Union[AlternativeType, str] = AlternativeType.TWO_SIDED,):
        '''
        Args:
            parameter: Test Parameter
            n1: Sample size (or first sample size)
            n2: Optional second sample size
            alternative: AlternativeType enum or string
            paired: Paired test flag
            sigma: Known population standard deviation
        '''
        self.parameter = parameter
        self.n1 = n1
        self.n2 = n2
        self.paired = paired
        self.sigma = sigma
        self.etype = etype  
        self.alternative = alternative.lower()              
    # Compute effective sample size
        self.nEff = self.compute_n_Eff()

    def compute_n_Eff(self):
        '''
        Compute effective sample size
        '''
        if self.paired is True or self.n2 is None:
            return self.n1
        else:
            return 1.0/(1/self.n1 + 1/self.n2)         
    
    def compute_e_value(self, z: float):
        '''
        Compute e-value from z-statistic based on specified eType
        '''
        if self.etype == "grow":
            self.compute_grow(z)
        elif self.etype == "egauss":
            return self.compute_egauss(z)
        elif self.etype == "mom":
            return self.compute_mom(z)
        elif self.etype == "imom":
            return self.compute_imom(z)
        elif self.etype == "ecauchy":
            return self.compute_ecauchy(z)
        else:
            raise ValueError(f"Unknown etype: {self.etype}")
    
    def compute_ecauchy(self, z: float) -> Tuple[float,float]:
        kappaG = self.parameter
        nEff = self.nEff

        if self.alternative == "two_sided":
            def integrand(g):
                log_term = (
                    -0.5 * np.log(1 + nEff * g) + 
                    nEff * g * z**2 / (2 * (1 + nEff * g)) - 
                    2 * np.log(g) + 
                    stats.gamma.logpdf(1/g, a=0.5, scale=2/(kappaG**2))
                )
                return np.exp(log_term)
        else:
            lower_tail = (self.alternative == "less")
            
            def integrand(g):
                threshold = -np.sqrt(g * nEff / (1 + nEff * g)) * z
                pnorm_term = (stats.norm.cdf(threshold) if lower_tail 
                            else stats.norm.sf(threshold))
                
                log_term = (
                    -0.5 * np.log(1 + nEff * g) + 
                    nEff * g * z**2 / (2 * (1 + nEff * g)) - 
                    2 * np.log(g) + 
                    stats.gamma.logpdf(1/g, a=0.5, scale=2/(kappaG**2))
                )
                return 2 * np.exp(log_term) * pnorm_term
        
        result: float
        error: float
        result, error = integrate.quad(integrand, 0, np.inf)
        return float(result), float(error)
    
    def compute_mom(self, z: float) -> Tuple[float, Optional[float]]:
        """Mixture of Means method formula."""
        g = self.parameter
        nEff = self.nEff
        
        if self.alternative == "two_sided":
            log_result = (
                -1.5 * np.log(1 + nEff * g) + 
                np.log(1 + nEff * g / (1 + nEff * g) * z**2) + 
                nEff * g / (1 + nEff * g) * z**2 / 2
            )
            return float(np.exp(log_result)), None
        else:
            def integrand(delta):
                return (
                    2 * g**(-1) * delta**2 * 
                    np.exp(
                        np.sqrt(nEff) * z * delta - 
                        nEff / 2 * delta**2 + 
                        stats.norm.logpdf(delta, loc=0, scale=np.sqrt(g))
                    )
                )
            
            lower = 0 if self.alternative == "greater" else -np.inf
            upper = np.inf if self.alternative == "greater" else 0
            
            result: float
            error: float
            result, error = integrate.quad(integrand, lower, upper)
            return float(result), float(error)

    def compute_imom(self, z: float) -> Tuple[float, float]:
        """Inverse Mixture of Means method formula."""
        tau = self.parameter
        nEff = self.nEff
        
        some_constant = 1 if self.alternative == "two_sided" else 2
        
        def integrand(delta):
            return some_constant * np.exp(
                np.sqrt(nEff) * z * delta - 
                nEff / 2 * delta**2 + 
                0.5 * np.log(tau) - 
                np.log(np.abs(delta)**2) - 
                tau / (delta**2) -
                np.log(np.sqrt(2 * np.pi))
            )
        
        if self.alternative == "less":
            lower, upper = -np.inf, 0
        elif self.alternative == "greater":
            lower, upper = 0, np.inf
        else:
            lower, upper = -np.inf, np.inf
        
        result, error = integrate.quad(integrand, lower, upper)
        return float(result), float(error)

    def compute_egauss(self, z: float)-> float:
        """e-Gaussian method formula"""
        g = self.parameter
        nEff = self.nEff

        log_result = (-0.5 * np.log(1 + nEff * g) + 
                      nEff * g * z**2 / (2 * (1 + nEff * g)))
        
        if self.alternative == "two_sided":
            result = np.exp(log_result)

        elif self.alternative == "greater":
            threshold = -np.sqrt(g * nEff / (1 + nEff * g)) * z
            result = 2 * np.exp(log_result) * stats.norm.sf(threshold)
        
        else:
            threshold = -np.sqrt(g * nEff / (1 + nEff * g)) * z
            result = 2 * np.exp(log_result) * stats.norm.cdf(threshold)
        
        return float(result)

    def compute_grow(self, z: float) -> float:
        phiS = 1 
        nEff = self.nEff
        sigma = self.sigma
        
        if self.alternative == "two_sided":
            exponent = -(nEff * phiS**2) / (2 * sigma**2)
            cosh_term = np.cosh(np.sqrt(nEff) * phiS / sigma * z)
            result = np.exp(exponent) * cosh_term
        else:
            result = np.exp(
                np.sqrt(nEff) * phiS / sigma * z - 
                nEff * phiS**2 / (2 * sigma**2)
            )
        
        if result < 0 or not np.isfinite(result):
            result = 2**(-15)
        
        return float(result)