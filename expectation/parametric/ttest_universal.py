"""
Sequential t-tests and confidence sequences for Gaussian means with unknown variance.

Based on the paper: "Anytime-valid t-tests and confidence sequences for Gaussian means
with unknown variance" by Hongjian Wang and Aaditya Ramdas (2024).

This module implements two main approaches:
1. Universal inference t-test e-processes (Theorem 3.2)
2. Scale invariant likelihood ratio t-test martingales (Theorem 4.9)


-- MIXTURE BASED T-TEST --
Null: mu = 0
Alternative: mu != 0 or mu > 0
Base martingales: scale invariant likelihood ratio
Parametrized by: theta = mu / sigma, the alternative standardized mean
Test process: log supermartingale in this code

TODO: Look at Alexander Ly's R package "safestats" for the Bayesian connection via
Grunwald's paper "Safe testing". The package implements safe t-test and z-test with confidence sequences.
Some parts to look at for the "Safe testing" concept:

- Extended to general Bayes factors with tuned priors for optimality
- Test supermartingales appear via conditional e-values called "safe under optional continuation"
- Design plan for testing

Ping: @alexanderly good entrypoint

"""

import numpy as np
from typing import Union, Tuple

from expectation.modules.martingales import MixtureSupermartingale
from expectation.seqtest.sequential_e_testing import SequentialTest, TestType, AlternativeType
from expectation.confseq.confidencesequence import ConfidenceSequence, ConfidenceSequenceConfig


class TtestGaussianMixtureMartingale(MixtureSupermartingale):
    """
    Implementation of t-test martingale using Gaussian mixture of scale invariant likelihood ratios.
    
    Based on Theorem 4.9 from Wang and Ramdas (2024): "Anytime-valid t-tests and confidence 
    sequences for Gaussian means with unknown variance"
    """
    
    def __init__(self, prior_precision: float = 1.0):
        """
        Initialize t-test martingale with Gaussian mixture.
        
        Args:
            prior_precision: Precision parameter c² for the Gaussian prior N(0,c⁻²)
        """
        self.c_squared = prior_precision**2
        
    def log_superMG(self, s: float, v: float) -> float:
        n = 1
        if hasattr(s, "__len__"):
            n = len(s)
        
        # Scale invariant t-test martingale from Theorem 4.9
        # G_n^(c) = sqrt(c^2 / (n+c^2)) * ((n + c^2) * V_n / ((n + c^2) * V_n - S_n^2))^(n/2)
        
        log_first_term = 0.5 * np.log(self.c_squared / (n + self.c_squared))
        log_second_term = (n/2) * np.log((n + self.c_squared) * v / ((n + self.c_squared) * v - s**2)) # -inf problem
        
        return (log_first_term + log_second_term) # Log t-test martingale
    
    def s_upper_bound(self, v: float) -> float:
        n = 1
        if hasattr(v, "__len__"):
            n = len(v)
        return np.sqrt((n + self.c_squared) * v)
    
    def bound(self, v: float, log_threshold: float) -> float:
        """
        Calculate bound given v and log threshold.
        
        This computes the confidence sequence radius from Theorem 4.9.
        """
        n = 1
        if hasattr(v, "__len__"):
            n = len(v)
        
        alpha = np.exp(-log_threshold)
        
        # Compute the radius formula from Theorem 4.9
        # TODO: move to confidence sequence module
        denominator = (alpha**(2*self.c_squared/(n+self.c_squared))**(1/n)) * (n + self.c_squared) - self.c_squared
        
        if denominator <= 0:
            return float('inf')
            
        radius_squared = (n + self.c_squared) * (1 - (alpha**(2*self.c_squared/(n+self.c_squared)))**(1/n)) / denominator * v
        
        return np.sqrt(radius_squared)


class TtestFlatMixtureMartingale(MixtureSupermartingale):
    """
    Implementation of t-test extended martingale using flat mixture of scale invariant likelihood ratios.
    
    Based on Theorem 4.7 from Wang and Ramdas (2024): "Anytime-valid t-tests and confidence 
    sequences for Gaussian means with unknown variance"
    """
    
    def log_superMG(self, s: float, v: float) -> float:
        n = 1
        if hasattr(s, "__len__"):
            n = len(s)
        
        # Compute the extended martingale from Theorem 4.7
        # H_n = sqrt(2 * pi / n) * ((n * V_n) / (n * V_n - S_n^2))^(n/2)
        
        log_first_term = 0.5 * np.log((2 * np.pi) / n)
        log_second_term = (n/2) * np.log( (n * v )/ (n * v - s**2))
        
        return (log_first_term + log_second_term) # Log t-test extended martingale
    
    def s_upper_bound(self, v: float) -> float:
        n = 1
        if hasattr(v, "__len__"):
            n = len(v)
        return np.sqrt(n * v)
    
    def bound(self, v: float, log_threshold: float) -> float:
        """
        Calculate bound given v and log threshold.
        
        This computes the confidence sequence radius based on Lai's t-CS (Theorem 4.1).
        """
        n = 1
        if hasattr(v, "__len__"):
            n = len(v)
        m = max(2, n)  # Starting time m ≥ 2
        
        # Calculate alpha from log_threshold
        alpha = np.exp(-log_threshold)
        
        # We need to solve for 'a' from the equation 2(1 - F_{m-1}(a) + af_{m-1}(a)) = alpha
        # This is a numerical approximation based on eq. (21) in the paper
        a = alpha**(-1/(m-1))
        
        # Calculate b and ξ_n from eq. (16) and (17)
        b = (1/m) * (1 + a/2 * (m-1)/m)
        radius = np.sqrt(v) * ((b*n)**(1/n) - 1)
        
        return radius

# TODO: this is not usable yet, need to update the TestType config in sequential_e_testing.py
def setup_ttest(test: SequentialTest):

    original_setup = test._setup_evaluator
    
    def new_setup_evaluator():
        if test.test_type == TestType.TTEST:

            ttest_method = test.config.get("ttest_method", "universal_inference")
            mixture_type = test.config.get("mixture_type", "gaussian")
            prior_precision = test.config.get("prior_precision", 1.0)
            

            test._mu_estimators = [0.0]
            test._sigma_estimators = [1.0]
            
            if ttest_method == "universal_inference":
                def calculator(data):
                    
                    mean = np.mean(data)
                    std = max(np.std(data, ddof=1), 1e-6)
                    test._mu_estimators.append(mean)
                    test._sigma_estimators.append(std)
                    
                    
                    data_centered = data - test.null_value
                    s = np.sum(data_centered)
                    v = np.sum(data_centered**2)
                    n = len(data)
                    
                    # Calculate e-value based on Theorem 3.2 or 3.4
                    if test.alternative == AlternativeType.TWO_SIDED:
                        
                        x_squared_bar = v / n
                        
                        # Calculate e-process using the formula from Theorem 3.2
                        log_process = (n/2) * np.log(x_squared_bar) + (n/2)
                        
                        
                        log_process -= np.sum(np.log(test._sigma_estimators[-n-1:-1]))
                        
                        
                        standardized_terms = np.zeros(n)
                        for i in range(n):
                            standardized_terms[i] = -0.5 * ((data_centered[i] - test._mu_estimators[-n-1+i])/test._sigma_estimators[-n-1+i])**2
                        
                        log_process += np.sum(standardized_terms)
                        return np.exp(log_process)
                    
                    else:  # One-sided test
                        # Implement the one-sided test (Theorem 3.4)
                        x_squared_bar = v / n
                        mu_bar = np.mean(data_centered)
                        mu_bar_neg = min(0, mu_bar)
                        adjusted_x_squared_bar = x_squared_bar - mu_bar_neg**2
                        
                        log_process = (n/2) * np.log(adjusted_x_squared_bar) + (n/2)
                        
                        
                        log_process -= np.sum(np.log(test._sigma_estimators[-n-1:-1]))
                        
                        
                        standardized_terms = np.zeros(n)
                        for i in range(n):
                            standardized_terms[i] = -0.5 * ((data_centered[i] - test._mu_estimators[-n-1+i])/test._sigma_estimators[-n-1+i])**2
                        
                        log_process += np.sum(standardized_terms)
                        
                        
                        if test.alternative == AlternativeType.LESS:
                            return np.exp(log_process.conjugate() if np.iscomplexobj(log_process) else log_process)
                        return np.exp(log_process)
                
                test.e_calculator = calculator
                
            else:  # scale_invariant
                if mixture_type == "gaussian":
                    mixture = TtestGaussianMixtureMartingale(prior_precision=prior_precision)
                else:
                    mixture = TtestFlatMixtureMartingale()
                
                def calculator(data):
                    data_centered = data - test.null_value
                    s = np.sum(data_centered)
                    v = np.sum(data_centered**2)
                    
                    
                    if test.alternative == AlternativeType.TWO_SIDED:
                        return np.exp(mixture.log_superMG(s, v))
                    
                    elif test.alternative == AlternativeType.GREATER:
                        # For greater alternative, we need the semi-one-sided form (Theorem 4.11)
                        neg_s = min(0, s)
                        full_value = np.exp(mixture.log_superMG(s, v))
                        neg_term = np.exp(mixture.log_superMG(neg_s, v))
                        return 2 * (full_value - neg_term)
                    
                    else:  # AlternativeType.LESS
                        pos_s = max(0, s)
                        full_value = np.exp(mixture.log_superMG(-s, v))
                        pos_term = np.exp(mixture.log_superMG(pos_s, v))
                        return 2 * (full_value - pos_term)
                
                test.e_calculator = calculator
        else:
            original_setup()
    
    
    test._setup_evaluator = new_setup_evaluator
    
    test._setup_evaluator()
    
    return test

# TODO: move to confidence sequence module
class TtestConfidenceSequence(ConfidenceSequence):
    def __init__(self, config: ConfidenceSequenceConfig, method: str = "universal_inference", 
                 prior_precision: float = 1.0):
        super().__init__(config)
        self.method = method
        self.prior_precision = prior_precision
        
        self.mu_estimators = [0.0]
        self.sigma_estimators = [1.0]
        
        if self.method == "gaussian_mixture":
            self.mixture = TtestGaussianMixtureMartingale(prior_precision=prior_precision)
        elif self.method == "flat_mixture":
            self.mixture = TtestFlatMixtureMartingale()
    
    def update(self, new_data: np.ndarray) -> Tuple[float, float]:
        result = super().update(new_data)
        
        if self.method == "universal_inference":
            mean = np.mean(new_data)
            std = max(np.std(new_data, ddof=1), 1e-6)
            self.mu_estimators.append(mean)
            self.sigma_estimators.append(std)
            
            # Compute confidence bounds using Theorem 3.2
            n = self.state.n_samples
            mu_bar = self.state.running_mean
            x_squared_bar = self.state.sum_squares / n
            
            # Compute W_n from equation (10) in Theorem 3.2
            log_sum_terms = 0
            for i in range(min(n, len(self.mu_estimators)-1)):
                log_sum_terms += np.log(self.sigma_estimators[i]**2) + ((new_data[i] - self.mu_estimators[i])/self.sigma_estimators[i])**2
            
            W = (1/self.config.alpha**(2/n)) * np.exp(log_sum_terms/n)
            
            radius = np.sqrt(max(0, mu_bar**2 - x_squared_bar + W))
            
        else:  # scale_invariant methods
            # Compute radius using mixture bounds
            n = self.state.n_samples
            s_squared = self.state.variance_estimate or 0
            
            log_threshold = np.log(1/self.config.alpha)
            radius = self.mixture.bound(s_squared * n, log_threshold)
        
        # Return confidence sequence (interval)
        return mu_bar - radius, mu_bar + radius


def create_ttest(null_value: float = 0.0, 
                alternative: Union[str, AlternativeType] = "two_sided",
                method: str = "universal_inference",
                mixture_type: str = "gaussian",
                prior_precision: float = 1.0,
                alpha: float = 0.05) -> SequentialTest:
    config = {
        "ttest_method": method,
        "mixture_type": mixture_type,
        "prior_precision": prior_precision
    }

    test = SequentialTest(
        test_type=TestType.TTEST,
        null_value=null_value,
        alternative=alternative,
        config=config
    )

    return setup_ttest(test)


def create_ttest_confidence_sequence(alpha: float = 0.05,
                                    method: str = "universal_inference",
                                    prior_precision: float = 1.0) -> TtestConfidenceSequence:
    config = ConfidenceSequenceConfig(alpha=alpha)

    return TtestConfidenceSequence(config, method=method, prior_precision=prior_precision)