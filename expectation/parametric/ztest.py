import numpy as np
import math
import warnings
from scipy.stats import norm ,gamma
from scipy.special import gammaln
from scipy.integrate import quad
from expectation.utils.helper_functions import effective_sample_size, check_and_return_esmin_parameter_side

def savi_z_test_stat(z: float, n1: int, n2: int = None, parameter: float = 1.0, alternative: str = "twoSided", 
                     paired: bool = False, sigma: float = 1.0, eType: str = "grow"):
    """
    Compute the e value for z statistic using the grow method.

    Parameters
    --------------
    z : float
        Observed z-statistic from the test.
    n1 : int
        Sample size of group 1.
    n2 : int, optional
        Sample size of group 2. Default is None.
    parameter : float
        Effect size parameter (phiS).
    alternative : str, optional
        Type of test: 'twoSided', 'less', or 'greater'. Default is 'twoSided'.
    paired : bool, optional
        Whether the test is paired. Default is False.
    sigma : float, optional
        Standard deviation under null. Default is 1.
    eType : str, optional
        Type of e-value method. Currently only 'grow' is implemented.

    Returns
    -------
    dict
        Dictionary with key "eValue".
    """
    valid_alternative = ['twoSided', 'greater', 'less']
    if alternative not in valid_alternative:
        raise ValueError(f"Alternative must be from one of {valid_alternative}")
    nEff = effective_sample_size(n1,n2,paired)
    if eType == 'grow':
        phiS = check_and_return_esmin_parameter_side(parameter, alternative)

        if alternative == "twoSided":
            e_val = np.exp(-nEff*phiS**2 / (2*sigma**2)) * np.cosh(np.sqrt(nEff) * phiS / sigma * z)
        else:
            e_val = np.exp(np.sqrt(nEff) * phiS / sigma * z - nEff * phiS**2 / (2 * sigma**2))

        if e_val < 0:
            warnings.warn("Overflow: e-value smaller than 0. Resetting to 2^-15")
            e_val = 2**(-15)

    elif eType == "mom":
        g = parameter
        if g <= 0:
            raise ValueError("For eType 'mom', parameter g must be strictly positive.")
        if alternative == "twoSided":
            logResult = -1.5*np.log(1+nEff*g) + np.log((1+(nEff*g)/(1+nEff*g))*z**2) + (nEff*g/(1+nEff*g))*z**2/2
            e_val = np.exp(logResult)
            return {"eValue": e_val}
        elif alternative in ["greater", "less"]:
            def mom_integrand(delta):
                 return (
                2 * g**(-1) * delta**2 *
                np.exp(
                    np.sqrt(nEff) * z * delta
                    - (nEff / 2) * delta**2
                    + norm.logpdf(delta, loc=0, scale=np.sqrt(g))
                )
            )
            lowerBound, upperBound = (0,np.inf) if alternative == "greater" else (-np.inf, 0)
            e_val, abs_error = quad(mom_integrand, lowerBound, upperBound, limit = 200)
            return {"eValue": e_val, "eValueApproxError": abs_error}
    elif eType == "eGauss":
        g = parameter
        if g <= 0:
            raise ValueError("For eType 'eGauss', parameter 'g' must be > 0.")
        logResult = -(1/2)*np.log(1+nEff*g) + nEff*g*z**2/(2*(1+nEff*g))
        if alternative == "twoSided":
            e_val = np.exp(logResult)
        elif alternative == "greater":
            e_val = 2*np.exp(logResult)*norm.sf(-np.sqrt(g * nEff / (1 + nEff * g))*z)
        elif alternative == "less":
            e_val = 2*np.exp(logResult)*norm.cdf(-np.sqrt(g*nEff / (1+ nEff*g))*z)
                                                

    elif eType == "imom":
        tau = parameter
        if tau <= 0:
            raise ValueError("For eType 'imom', parameter 'tau' must be > 0.")
        someConstant = 1 if alternative == "twoSided" else 2
        def iMomIntegrand(delta):
            return someConstant * np.exp(
            np.sqrt(nEff) * z * delta
            - 0.5 * nEff * delta**2
            + 0.5 * np.log(tau)
            - gammaln(0.5)       # lgamma(1/2)
            - np.log(delta**2)
            - tau / (delta**2)
        )

        # Integration bounds
        if alternative == "less":
            upperBound, lowerBound = 0, -np.inf
        elif alternative == "greater":
            upperBound, lowerBound = np.inf, 0
        else:
            upperBound, lowerBound = np.inf, -np.inf
        
        e_val, abs_error = quad(iMomIntegrand, lowerBound, upperBound, limit=200)
        return  {"eValue": e_val, "eValueApproxError": abs_error}

    elif eType == "eCauchy":
        kappaG = parameter
        if kappaG <= 0:
            raise ValueError("eCauchy requires parameter kappaG > 0.")
        
        if alternative == "twoSided":
            def integrand(g):
                log_density = gamma.logpdf(1/g, a=0.5, scale=2/(kappaG**2))
                base = -0.5 * np.log(1 + nEff * g) + (nEff * g * z**2) / (2 * (1 + nEff * g)) - 2 * np.log(g)
                return np.exp(base + log_density)

        elif alternative in ["greater", "less"]:
            wantLowerTail = (alternative == "less")
            def integrand(g):  
                log_density = gamma.logpdf(1/g, a=0.5, scale=2/(kappaG**2))
                base = -0.5 * np.log(1 + nEff * g) + (nEff * g * z**2) / (2 * (1 + nEff * g)) - 2 * np.log(g)
                return 2*np.exp(base + log_density)*(norm.cdf(-np.sqrt(g * nEff / (1 + nEff * g)) * z) 
                                                     if wantLowerTail 
                                                     else norm.sf(-np.sqrt(g * nEff / (1 + nEff * g)) * z))
            
        # Perform integration
        e_val, abs_error = quad(integrand, 0, np.inf, limit=200)
        return {"eValue": e_val, "eValueApproxError": abs_error}
    else:
        raise NotImplementedError(f"eType '{eType} not implemented yet. Some bug in the code") 
    
    return {"eValue":e_val}