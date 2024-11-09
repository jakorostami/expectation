# Translatend from https://github.com/gostevehoward/confseq boundaries.cpp

# Using numpy and scipy for math functions
# Boosts special functions -> Scipy equivalents
# Numpy arrays for vectorized calculations
# Scipy optimize for root finding

import numpy as np
from scipy import special, optimize
from typing import Tuple, Union
from numpy.typing import ArrayLike
from expectation.modules.martingales import (BetaBinomialMixture, 
                                                                     OneSidedNormalMixture, 
                                                                     TwoSidedNormalMixture, 
                                                                     GammaExponentialMixture, 
                                                                     GammaPoissonMixture, 
                                                                     PolyStitchingBound,
                                                                     EmpiricalProcessLILBound)


def log_beta(a: float, b: float) -> float:
    """Compute logarithm of beta function."""
    return special.gammaln(a) + special.gammaln(b) - special.gammaln(a + b)

def log_incomplete_beta(a: float, b: float, x: float) -> float:
    """Compute logarithm of incomplete beta function."""
    if x == 1:
        return log_beta(a, b)
    return np.log(special.betainc(a, b, x)) + log_beta(a, b)

# Simple interface functions
def normal_log_mixture(s: ArrayLike, v: ArrayLike, v_opt: float, 
                      alpha_opt: float = 0.05, is_one_sided: bool = True) -> np.ndarray:
    """Calculate normal log mixture."""
    s, v = np.asarray(s), np.asarray(v)
    mixture = OneSidedNormalMixture(v_opt, alpha_opt) if is_one_sided else TwoSidedNormalMixture(v_opt, alpha_opt)
    return np.vectorize(mixture.log_superMG)(s, v)


def normal_mixture_bound(v: ArrayLike, alpha: float, v_opt: float,
                        alpha_opt: float = 0.05, is_one_sided: bool = True) -> np.ndarray:
    """Calculate normal mixture bound."""
    v = np.asarray(v)
    mixture = OneSidedNormalMixture(v_opt, alpha_opt) if is_one_sided else TwoSidedNormalMixture(v_opt, alpha_opt)
    return np.vectorize(lambda x: mixture.bound(x, np.log(1/alpha)))(v)

def gamma_exponential_log_mixture(s: ArrayLike, v: ArrayLike, v_opt: float,
                                c: float, alpha_opt: float = 0.05) -> np.ndarray:
    """Calculate gamma exponential log mixture."""
    s, v = np.asarray(s), np.asarray(v)
    mixture = GammaExponentialMixture(v_opt, alpha_opt, c)
    return np.vectorize(mixture.log_superMG)(s, v)

def gamma_exponential_mixture_bound(v: ArrayLike, alpha: float, v_opt: float,
                                  c: float, alpha_opt: float = 0.05) -> np.ndarray:
    """Calculate gamma exponential mixture bound."""
    v = np.asarray(v)
    mixture = GammaExponentialMixture(v_opt, alpha_opt, c)
    return np.vectorize(lambda x: mixture.bound(x, np.log(1/alpha)))(v)

def gamma_poisson_log_mixture(s: ArrayLike, v: ArrayLike, v_opt: float,
                            c: float, alpha_opt: float = 0.05) -> np.ndarray:
    """Calculate gamma Poisson log mixture."""
    s, v = np.asarray(s), np.asarray(v)
    mixture = GammaPoissonMixture(v_opt, alpha_opt, c)
    return np.vectorize(mixture.log_superMG)(s, v)

def gamma_poisson_mixture_bound(v: ArrayLike, alpha: float, v_opt: float,
                              c: float, alpha_opt: float = 0.05) -> np.ndarray:
    """Calculate gamma Poisson mixture bound."""
    v = np.asarray(v)
    mixture = GammaPoissonMixture(v_opt, alpha_opt, c)
    return np.vectorize(lambda x: mixture.bound(x, np.log(1/alpha)))(v)

def beta_binomial_log_mixture(s: ArrayLike, v: ArrayLike, v_opt: float,
                            g: float, h: float, alpha_opt: float = 0.05,
                            is_one_sided: bool = True) -> np.ndarray:
    """Calculate beta binomial log mixture."""
    s, v = np.asarray(s), np.asarray(v)
    mixture = BetaBinomialMixture(v_opt, alpha_opt, g, h, is_one_sided)
    return np.vectorize(mixture.log_superMG)(s, v)

def beta_binomial_mixture_bound(v: ArrayLike, alpha: float, v_opt: float,
                              g: float, h: float, alpha_opt: float = 0.05,
                              is_one_sided: bool = True) -> np.ndarray:
    """Calculate beta binomial mixture bound."""
    v = np.asarray(v)
    mixture = BetaBinomialMixture(v_opt, alpha_opt, g, h, is_one_sided)
    return np.vectorize(lambda x: mixture.bound(x, np.log(1/alpha)))(v)

def poly_stitching_bound(v: ArrayLike, alpha: float, v_min: float,
                        c: float = 0, s: float = 1.4, eta: float = 2) -> np.ndarray:
    """Calculate polynomial stitching bound."""
    v = np.asarray(v)
    bound = PolyStitchingBound(v_min, c, s, eta)
    return np.vectorize(lambda x: bound(x, alpha))(v)

def empirical_process_lil_bound(t: Union[int, float], alpha: float,
                               t_min: float, A: float = 0.85) -> float:
    """Calculate empirical process LIL bound."""
    bound = EmpiricalProcessLILBound(alpha, t_min, A)
    return bound(t)

def double_stitching_bound(quantile_p: float, t: float, alpha: float,
                          t_opt: float, delta: float = 0.5,
                          s: float = 1.4, eta: float = 2) -> float:
    """Calculate double stitching bound."""
    t_max_m = max(t, t_opt)
    
    def logit(p: float) -> float:
        return np.log(p / (1 - p))
    
    def expit(l: float) -> float:
        return 1 / (1 + np.exp(-l))
    
    r = (quantile_p if quantile_p >= 0.5 else 
         min(0.5, expit(logit(quantile_p) + 2 * delta * np.sqrt(t_opt * eta / t_max_m))))
    
    sigma_sq = r * (1 - r)
    j = np.sqrt(t_max_m / t_opt) * abs(logit(quantile_p)) / (2 * delta) + 1
    zeta_s = special.zeta(s)
    
    ell = (s * np.log(np.log(eta * t_max_m / t_opt)) + s * np.log(j) +
           np.log(2 * zeta_s * (2 * zeta_s + 1) / (alpha * np.power(np.log(eta), s))))
    
    cp = (1 - 2 * quantile_p) / 3
    k1 = (np.power(eta, 0.25) + np.power(eta, -0.25)) / np.sqrt(2)
    k2 = (np.sqrt(eta) + 1) / 2
    term2 = k2 * cp * ell
    
    return (delta * np.sqrt(eta * t_max_m * sigma_sq / t_opt) +
            np.sqrt(k1 * k1 * sigma_sq * t_max_m * ell + term2 * term2) + term2)

def bernoulli_confidence_interval(num_successes: Union[float, int],
                                num_trials: Union[float, int],
                                alpha: float,
                                t_opt: float,
                                alpha_opt: float = 0.05) -> Tuple[float, float]:
    """Calculate Bernoulli confidence interval."""
    threshold = np.log(1/alpha)
    empirical_p = float(num_successes) / num_trials
    
    def objective(p: float) -> float:
        if p <= 0 or p >= 1:
            return np.inf
        mixture = BetaBinomialMixture(p * (1 - p) * t_opt, alpha_opt, p, 1 - p, False)
        s = (empirical_p - p) * num_trials
        v = p * (1 - p) * num_trials
        return mixture.log_superMG(s, v) - threshold
    
    # Find lower bound
    lower_bound = 0.0
    if empirical_p > 0:
        lower_bound = optimize.bisect(
            lambda p: objective(p),
            0.0, empirical_p,
            xtol=2**-40
        )
    
    # Find upper bound
    upper_bound = 1.0
    if empirical_p < 1:
        upper_bound = optimize.bisect(
            lambda p: -objective(p),
            empirical_p, 1.0,
            xtol=2**-40
        )
    
    return lower_bound, upper_bound