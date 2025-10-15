'''
Compute E-values Based on Z-statistic
'''
import numpy as np
from scipy import stats, integrate
from typing import Union, Tuple, Optional

from expectation.seqtest.sequential_e_testing import AlternativeType

class saviZTestStat:
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
        self.nEff = self.compute_nEff()

    def compute_nEff(self):
        '''
        Compute effective sample size
        '''
        if self.paired is True or self.n2 is None:
            return self.n1
        else:
            return 1.0/(1/self.n1 + 1/self.n2)         
    
    def compute_e_value(self, z: float):
        if self.etype == "grow":
            pass
        elif self.etype == "egauss":
            pass
        elif self.etype == "mom":
            pass
        elif self.etype == "imom":
            pass
        elif self.etype == "ecauchy":
            pass
        else:
            raise ValueError(f"Unknown etype: {self.etype}")
    
    def compute_ecauchy(self, z: float):
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
        
        result, error = integrate.quad(integrand, 0, np.inf)
        return float(result), float(error)

