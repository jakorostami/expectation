# SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
# Copyright (c) Jako Rostami 2024-present
# Project: expectation
#
# Licensed under GPL-3.0 with additional restrictions per Section 7(b).
# Use of this code for AI/ML model training is strictly prohibited.
# See LICENSE for full terms.

"""
Based on these papers:

Time-uniform, nonparametric, nonasymptotic confidence sequences, S.R Howard, A. Ramdas, J. McAuliffe, J. Sekhon (2022) - https://arxiv.org/pdf/1810.08240

Merging sequential e-values via martingales, V. Vovk, R. Wang (2024) - https://arxiv.org/pdf/2007.06382

Time-uniform central limit theory and asymptotic confidence sequences, I. Waudby-Smith, D. Arbour, R. Sinha, E.H Kennedy, A. Ramdas (2021) - https://arxiv.org/pdf/2103.06476

Hypothesis testing with e-values - A. Ramdas, R. Wang (2025) -
"""

# Translatend from https://github.com/gostevehoward/confseq boundaries.cpp
# Howard, S. R., Waudby-Smith, I. and Ramdas, A. (2019-), ConfSeq: software for confidence sequences and uniform boundaries, https://github.com/gostevehoward/confseq [Online; accessed ].

# Using numpy and scipy for math functions
# Boosts special functions -> Scipy equivalents
# Numpy arrays for vectorized calculations
# Scipy optimize for root finding


from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
from scipy import optimize, special, stats


class MixtureSupermartingale(ABC):
    """
    Abstract base class for mixture supermartingales.
    """

    @abstractmethod
    def log_superMG(self, s: float, v: float) -> float:
        pass

    @abstractmethod
    def s_upper_bound(self, v: float) -> float:
        pass

    @abstractmethod
    def bound(self, v: float, log_threshold: float) -> float:
        pass


class SequentialEValueCombiner(ABC):
    """
    Abstract base class for combining sequential e-values into e-processes.

    Following Definition 7.21, an e-process built on sequential e-values (E_t) is:
        M_t = prod{s=1}^t ((1 - lambda_s) + lambda_s E_s)
    where lambda_t is F_{t-1}-measurable (depends only on past).
    """

    @abstractmethod
    def compute_lambda(self, past_e_values: List[float], t: int) -> float:
        """
        Compute the betting fraction lambda_t based on E_1, ..., E_{t-1}.
        Must be F_{t-1}-measurable (use only past values).
        """
        pass

    def compute_increment(self, e_value: float, lambda_t: float) -> float:
        return (1 - lambda_t) + lambda_t * e_value

    def compute_log_increment(self, e_value: float, lambda_t: float) -> float:
        increment = self.compute_increment(e_value, lambda_t)
        return np.log(increment) if increment > 0 else -np.inf


def find_s_upper_bound(mixture: MixtureSupermartingale, v: float, log_threshold: float) -> float:
    trial_upper_bound = float(v)
    for _ in range(50):
        if mixture.log_superMG(trial_upper_bound, v) > log_threshold:
            return trial_upper_bound
        trial_upper_bound *= 2
    raise RuntimeError("Failed to find upper limit for mixture bound")


def find_mixture_bound(mixture: MixtureSupermartingale, v: float, log_threshold: float) -> float:
    def root_fn(s: float) -> float:
        return mixture.log_superMG(s, v) - log_threshold

    s_upper = mixture.s_upper_bound(v)
    if np.isinf(s_upper):
        s_upper = find_s_upper_bound(mixture, v, log_threshold)

    if root_fn(s_upper) < 0:
        return s_upper

    result = optimize.bisect(root_fn, 0.0, s_upper, xtol=2**-40)
    return result


class TwoSidedNormalMixture(MixtureSupermartingale):
    def __init__(self, v_opt: float, alpha_opt: float):
        assert v_opt > 0
        self.rho = self.best_rho(v_opt, alpha_opt)

    def log_superMG(self, s: float, v: float) -> float:
        return 0.5 * np.log(self.rho / (v + self.rho)) + s * s / (2 * (v + self.rho))

    def s_upper_bound(self, v: float) -> float:
        return np.inf

    def bound(self, v: float, log_threshold: float) -> float:
        return np.sqrt((v + self.rho) * (np.log(1 + v / self.rho) + 2 * log_threshold))

    @staticmethod
    def best_rho(v: float, alpha: float) -> float:
        assert 0 < alpha < 1
        return v / (2 * np.log(1 / alpha) + np.log(1 + 2 * np.log(1 / alpha)))


class OneSidedNormalMixture(MixtureSupermartingale):
    def __init__(self, v_opt: float, alpha_opt: float):
        self.rho = self.best_rho(v_opt, alpha_opt)

    def log_superMG(self, s: float, v: float) -> float:
        return (
            0.5 * np.log(4 * self.rho / (v + self.rho))
            + s * s / (2 * (v + self.rho))
            + np.log(stats.norm.cdf(s / np.sqrt(v + self.rho)))
        )

    def s_upper_bound(self, v: float) -> float:
        return np.inf

    def bound(self, v: float, log_threshold: float) -> float:
        return find_mixture_bound(self, v, log_threshold)

    @staticmethod
    def best_rho(v: float, alpha: float) -> float:
        return TwoSidedNormalMixture.best_rho(v, 2 * alpha)


class GammaExponentialMixture(MixtureSupermartingale):
    def __init__(self, v_opt: float, alpha_opt: float, c: float):
        self.rho = OneSidedNormalMixture.best_rho(v_opt, alpha_opt)
        self.c = c
        self.leading_constant = self._get_leading_constant()

    def _get_leading_constant(self) -> float:
        rho_c_sq = self.rho / (self.c * self.c)
        return (
            rho_c_sq * np.log(rho_c_sq)
            - special.gammaln(rho_c_sq)
            - np.log(special.gammainc(rho_c_sq, rho_c_sq))
        )

    def log_superMG(self, s: float, v: float) -> float:
        c_sq = self.c * self.c
        cs_v_csq = (self.c * s + v) / c_sq
        v_rho_csq = (v + self.rho) / c_sq

        return (
            self.leading_constant
            + special.gammaln(v_rho_csq)
            + np.log(special.gammainc(v_rho_csq, cs_v_csq + self.rho / c_sq))
            - v_rho_csq * np.log(cs_v_csq + self.rho / c_sq)
            + cs_v_csq
        )

    def s_upper_bound(self, v: float) -> float:
        return np.inf

    def bound(self, v: float, log_threshold: float) -> float:
        return find_mixture_bound(self, v, log_threshold)


class GammaPoissonMixture(MixtureSupermartingale):
    def __init__(self, v_opt: float, alpha_opt: float, c: float):
        self.rho = OneSidedNormalMixture.best_rho(v_opt, alpha_opt)
        self.c = c
        self.leading_constant = self._get_leading_constant()

    def _get_leading_constant(self) -> float:
        rho_c_sq = self.rho / (self.c * self.c)
        return (
            rho_c_sq * np.log(rho_c_sq)
            - special.gammaln(rho_c_sq)
            - np.log(special.gammaincc(rho_c_sq, rho_c_sq))
        )

    def log_superMG(self, s: float, v: float) -> float:
        c_sq = self.c * self.c
        v_rho_csq = (v + self.rho) / c_sq
        cs_v_rho_csq = s / self.c + v_rho_csq

        return (
            self.leading_constant
            + special.gammaln(cs_v_rho_csq)
            + np.log(special.gammaincc(cs_v_rho_csq, v_rho_csq))
            - cs_v_rho_csq * np.log(v_rho_csq)
            + v / c_sq
        )

    def s_upper_bound(self, v: float) -> float:
        return np.inf

    def bound(self, v: float, log_threshold: float) -> float:
        return find_mixture_bound(self, v, log_threshold)


class BetaBinomialMixture(MixtureSupermartingale):
    def __init__(self, v_opt: float, alpha_opt: float, g: float, h: float, is_one_sided: bool):
        assert g > 0 and h > 0
        self.g = g
        self.h = h
        self.is_one_sided = is_one_sided
        self.r = self._optimal_r(v_opt, alpha_opt)
        self.normalizer = self._compute_normalizer()

    def _optimal_r(self, v_opt: float, alpha_opt: float) -> float:
        rho = (OneSidedNormalMixture if self.is_one_sided else TwoSidedNormalMixture).best_rho(
            v_opt, alpha_opt
        )
        return max(rho - self.g * self.h, 1e-3 * self.g * self.h)

    def _compute_normalizer(self) -> float:
        x = self.h / (self.g + self.h) if self.is_one_sided else 1
        return log_incomplete_beta(
            self.r / (self.g * (self.g + self.h)), self.r / (self.h * (self.g + self.h)), x
        )

    def log_superMG(self, s: float, v: float) -> float:
        x = self.h / (self.g + self.h) if self.is_one_sided else 1
        return (
            v / (self.g * self.h) * np.log(self.g + self.h)
            - ((v + self.h * s) / (self.h * (self.g + self.h))) * np.log(self.g)
            - ((v - self.g * s) / (self.g * (self.g + self.h))) * np.log(self.h)
            + log_incomplete_beta(
                (self.r + v - self.g * s) / (self.g * (self.g + self.h)),
                (self.r + v + self.h * s) / (self.h * (self.g + self.h)),
                x,
            )
            - self.normalizer
        )

    def s_upper_bound(self, v: float) -> float:
        return v / self.g

    def bound(self, v: float, log_threshold: float) -> float:
        return find_mixture_bound(self, v, log_threshold)


class PolyStitchingBound:
    def __init__(self, v_min: float, c: float, s: float, eta: float):
        assert v_min > 0
        self.v_min = v_min
        self.c = c
        self.s = s
        self.eta = eta
        self.k1 = (np.power(eta, 0.25) + np.power(eta, -0.25)) / np.sqrt(2)
        self.k2 = (np.sqrt(eta) + 1) / 2
        self.A = np.log(special.zeta(s) / np.power(np.log(eta), s))

    def __call__(self, v: float, alpha: float) -> float:
        use_v = max(v, self.v_min)
        ell = self.s * np.log(np.log(self.eta * use_v / self.v_min)) + self.A + np.log(1 / alpha)
        term2 = self.k2 * self.c * ell
        return np.sqrt(self.k1 * self.k1 * use_v * ell + term2 * term2) + term2


class EmpiricalProcessLILBound:
    def __init__(self, alpha: float, t_min: float, A: float):
        assert A > 1 / np.sqrt(2)
        assert t_min >= 1
        assert 0 < alpha < 1
        self.t_min = t_min
        self.A = A
        self.C = self._find_optimal_C(alpha)

    def __call__(self, t: float) -> float:
        if t < self.t_min:
            return np.inf
        return self.A * np.sqrt((np.log(1 + np.log(t / self.t_min)) + self.C) / t)

    def _find_optimal_C(self, alpha: float) -> float:
        def error_bound(C: float, eta: float) -> float:
            gamma_sq = 2 / eta * np.power(self.A - np.sqrt(2 * (eta - 1) / C), 2)
            if gamma_sq <= 1:
                return np.inf
            return 4 * np.exp(-gamma_sq * C) * (1 + 1 / ((gamma_sq - 1) * np.log(eta)))

        def optimize_eta(C: float) -> float:
            def objective(eta: float) -> float:
                return np.sqrt(eta / 2) + np.sqrt(2 * (eta - 1) / C) - self.A

            eta_result = optimize.bisect(objective, 1.0, 2 * self.A * self.A)
            return optimize.minimize_scalar(
                lambda eta: error_bound(C, eta), bounds=(1.0, eta_result), method="bounded"
            ).fun

        def objective(C: float) -> float:
            return optimize_eta(C) - alpha

        C_result = optimize.root_scalar(objective, bracket=(5.0, 100.0), method="bisect")
        return C_result.root


def log_beta(a: float, b: float) -> float:
    return special.gammaln(a) + special.gammaln(b) - special.gammaln(a + b)


def log_incomplete_beta(a: float, b: float, x: float) -> float:
    if x == 1:
        return log_beta(a, b)
    return np.log(special.betainc(a, b, x)) + log_beta(a, b)


### combiners
class AllInCombiner(SequentialEValueCombiner):
    """All-in betting: lambda_t = 1 for all t (Proposition 7.20)"""

    def compute_lambda(self, past_e_values: List[float], t: int) -> float:
        return 1.0


class ConservativeCombiner(SequentialEValueCombiner):
    """Conservative betting with fixed lambda < 1"""

    def __init__(self, lambda_fixed: float = 0.5):
        if not 0 < lambda_fixed <= 1:
            raise ValueError(f"lambda_fixed must be in (0, 1], got {lambda_fixed}")
        self.lambda_fixed = lambda_fixed

    def compute_lambda(self, past_e_values: List[float], t: int) -> float:
        return self.lambda_fixed


class EmpiricallyAdaptiveCombiner(SequentialEValueCombiner):
    """
    Empirically adaptive betting from Definition 7.21(iii).
    - lambda_1 = 0 (start conservatively)
    - lambda_t maximizes empirical e-power
    - lambda_t = 0 iff empirical mean ≤ 1 (Theorem 3.14)
    - Asymptotically log-optimal (Theorem 7.22)
    """

    def __init__(self, gamma: float = 0.5):
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self.gamma = gamma

    def compute_lambda(self, past_e_values: List[float], t: int) -> float:
        # lambda_1 = 0 as specified
        if t == 1 or not past_e_values:
            return 0.0

        # Theorem 3.14: lambda_t = 0 iff empirical mean ≤ 1
        empirical_mean = np.mean(past_e_values)
        if empirical_mean <= 1:
            return 0.0

        # Optimize empirical e-power
        def negative_empirical_epower(lam: float) -> float:
            log_terms = []
            for e in past_e_values:
                increment = (1 - lam) + lam * e
                if increment > 0:
                    log_terms.append(np.log(increment))

            if not log_terms:
                return np.inf

            return -np.mean(log_terms)

        result = optimize.minimize_scalar(
            negative_empirical_epower,
            bounds=(0, self.gamma),
            method="bounded",
            options={"xatol": 1e-8},
        )

        return result.x


class LogOptimalCombiner(SequentialEValueCombiner):
    """
    Q-log-optimal betting from Definition 7.21(ii).
    Maximizes E_Q[log((1-lambda) + lambdaE_t) | F_{t-1}] when Q is known.
    """

    def __init__(self, expectation_function: Callable[[float, List[float], int], float]):
        """
        Args:
            expectation_function: Function(lambda, past_e_values, t) -> E_Q[log((1-lambda) + lambdaE_t) | F_{t-1}]
        """
        self.expectation_function = expectation_function

    def compute_lambda(self, past_e_values: List[float], t: int) -> float:
        def negative_expected_epower(lam: float) -> float:
            return -self.expectation_function(lam, past_e_values, t)

        result = optimize.minimize_scalar(negative_expected_epower, bounds=(0, 1), method="bounded")

        return result.x
