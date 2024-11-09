import numpy as np
from typing import Tuple, Callable
from scipy import optimize
from expectation.modules.orderstatistics import OrderStatisticInterface
from expectation.modules.martingales import BetaBinomialMixture


class QuantileABTest:
    """Implementation of quantile A/B testing."""
    
    def __init__(self, quantile_p: float, t_opt: int, alpha_opt: float,
                 arm1_os: OrderStatisticInterface, arm2_os: OrderStatisticInterface):
        assert 0 < quantile_p < 1
        self.quantile_p = quantile_p
        self.mixture = BetaBinomialMixture(
            t_opt * quantile_p * (1 - quantile_p),
            alpha_opt, quantile_p, 1 - quantile_p, False
        )
        self.arm1_os = arm1_os
        self.arm2_os = arm2_os
    
    def p_value(self) -> float:
        """Calculate p-value."""
        return min(1.0, np.exp(-self.log_superMG_lower_bound()))
    
    def log_superMG_lower_bound(self) -> float:
        """Calculate lower bound of log supermartingale."""
        arm1_G = self.get_G_fn(1)
        arm2_G = self.get_G_fn(2)
        
        if arm1_G[1] <= arm2_G[1]:  # Compare minimum_end_x
            return self.find_log_superMG_lower_bound(arm1_G, arm2_G, 2)
        else:
            return self.find_log_superMG_lower_bound(arm2_G, arm1_G, 1)
    
    def get_G_fn(self, arm: int) -> Tuple[Callable[[float], float], float, float]:
        """Get G function for given arm."""
        def objective(a: float) -> float:
            return self.arm_log_superMG(arm, a)
        
        # Find minimizer using Brent's method
        minimizer = optimize.minimize_scalar(objective, bounds=(0, 1), method='bounded').x
        
        N = self.order_stats(arm).size()
        x_lower = self.order_stats(arm).get_order_statistic(int(np.ceil(minimizer * N)))
        x_upper = self.order_stats(arm).get_order_statistic(int(np.floor(minimizer * N)) + 1)
        
        def G_callable(x: float) -> float:
            if x < x_lower:
                prop_below = self.order_stats(arm).count_less_or_equal(x) / N
            elif x > x_upper:
                prop_below = self.order_stats(arm).count_less(x) / N
            else:
                prop_below = minimizer
            return self.arm_log_superMG(arm, prop_below)
        
        return G_callable, x_lower, x_upper
    
    def find_log_superMG_lower_bound(
        self,
        first_arm_G: Tuple[Callable[[float], float], float, float],
        second_arm_G: Tuple[Callable[[float], float], float, float],
        second_arm: int
    ) -> float:
        """Find lower bound of log supermartingale."""
        G1, _, _ = first_arm_G
        G2, x2_lower, x2_upper = second_arm_G
        
        def objective(x: float) -> float:
            return G1(x) + G2(x)
        
        # Calculate minimum value at endpoints
        min_value = min(objective(x2_lower), objective(x2_upper))
        
        # Check all order statistics between x2_lower and x2_upper
        start_index = self.order_stats(second_arm).count_less_or_equal(x2_lower)
        end_index = self.order_stats(second_arm).count_less_or_equal(x2_upper)
        
        for i in range(max(1, start_index), end_index + 1):
            x = self.order_stats(second_arm).get_order_statistic(i)
            value = objective(x)
            min_value = min(min_value, value)
        
        return min_value
    
    def arm_log_superMG(self, arm: int, prop_below: float) -> float:
        """Calculate arm log supermartingale."""
        N = self.order_stats(arm).size()
        s = (prop_below - self.quantile_p) * N
        v = self.quantile_p * (1 - self.quantile_p) * N
        return self.mixture.log_superMG(s, v)
    
    def order_stats(self, arm: int) -> OrderStatisticInterface:
        """Get order statistics for given arm."""
        assert arm in (1, 2)
        return self.arm1_os if arm == 1 else self.arm2_os