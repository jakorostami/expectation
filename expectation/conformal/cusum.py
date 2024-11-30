"""
This is an experimental module of the library trying to implement the conformal prediction framework but for e-values.
The module is not yet complete and is still under heavy development so use it with caution as no unit tests exist yet.

Conformal e-testing (2024) - Vladimir Vovk, Ilia Nouretdinov, and Alex Gammerman
https://www.alrw.net/articles/29.pdf

"""

from typing import List, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from dataclasses import dataclass

@dataclass
class CUSUMResult:
    """Results from CUSUM procedure."""
    statistic: float  # Current CUSUM statistic
    alarms: List[int]  # Time points of alarms
    alarm_stats: List[float]  # CUSUM statistics at alarm points
    all_stats: List[float]  # Full history of statistics
    n_alarms: int  # Total number of alarms raised

class ConformalCUSUM:
    """
    Implementation of the conformal CUSUM e-procedure as described in Section 5.
    Includes both standard and reverse Shiryaev-Roberts modifications.
    """
    def __init__(self, 
                 threshold: float = 20.0,
                 use_sr: bool = False,
                 truncate: bool = True,
                 min_value: float = 1e-10):
        """
        Initialize CUSUM e-procedure.
        
        Args:
            threshold: Detection threshold c > 1
            use_sr: Whether to use Shiryaev-Roberts modification
            truncate: Whether to use truncation
            min_value: Minimum value for truncation
        """
        if threshold <= 1:
            raise ValueError("Threshold must be > 1")
        
        self.threshold = threshold
        self.use_sr = use_sr
        self.truncate = truncate
        self.min_value = min_value
        
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self._last_alarm = 0  # Time of last alarm
        self._cusum_stat = 0.0  # Current CUSUM statistic
        self._stats_history = []  # History of statistics
        self._alarms = []  # Alarm times
        self._alarm_stats = []  # Statistics at alarm times
        
    def update(self, e_value: float) -> CUSUMResult:
        """
        Update CUSUM statistic with new e-value.
        
        Args:
            e_value: New conformal e-value
            
        Returns:
            CUSUMResult with current state
        """
        # Update time step
        t = len(self._stats_history) + 1
        
        # Calculate CUSUM statistic
        if self.use_sr:
            # Shiryaev-Roberts modification
            sum_stat = sum(e_value * stat for stat in 
                         self._stats_history[self._last_alarm:])
            self._cusum_stat = max(sum_stat, self.min_value if self.truncate else 0)
        else:
            # Standard CUSUM
            self._cusum_stat = max(
                self._cusum_stat * e_value,
                self.min_value if self.truncate else 0
            )
        
        # Store statistic
        self._stats_history.append(self._cusum_stat)
        
        # Check for alarm
        if self._cusum_stat >= self.threshold:
            self._alarms.append(t)
            self._alarm_stats.append(self._cusum_stat)
            self._last_alarm = t
            self._cusum_stat = 0.0  # Reset after alarm
            
        return CUSUMResult(
            statistic=self._cusum_stat,
            alarms=self._alarms.copy(),
            alarm_stats=self._alarm_stats.copy(),
            all_stats=self._stats_history.copy(),
            n_alarms=len(self._alarms)
        )
    
    def get_alarm_rate(self) -> float:
        """
        Calculate empirical alarm rate.
        
        Returns:
            Number of alarms divided by number of observations
        """
        t = len(self._stats_history)
        return len(self._alarms) / t if t > 0 else 0.0

class EfficiencyAnalyzer:
    """
    Tools for analyzing efficiency of conformal e-testing procedures
    as described in Section 6 of the paper.
    """
    def __init__(self, 
                 pre_dist: Optional[callable] = None,
                 post_dist: Optional[callable] = None):
        """
        Initialize analyzer.
        
        Args:
            pre_dist: Pre-change distribution sampler
            post_dist: Post-change distribution sampler
        """
        self.pre_dist = pre_dist or (lambda n: np.random.normal(0, 1, n))
        self.post_dist = post_dist or (lambda n: np.random.normal(0.5, 1, n))
        
    def compute_likelihood_ratios(self, data: ArrayLike) -> np.ndarray:
        """
        Compute likelihood ratios for efficiency analysis.
        """
        data = np.asarray(data)
        # Using normal distributions as default
        pre_likelihood = np.exp(-0.5 * data**2) / np.sqrt(2 * np.pi)
        post_likelihood = np.exp(-0.5 * (data - 0.5)**2) / np.sqrt(2 * np.pi)
        return post_likelihood / pre_likelihood
    
    def analyze_decay(self, 
                     e_values: ArrayLike,
                     change_point: int) -> Tuple[float, float]:
        """
        Analyze decay rate of e-values after change point.
        
        Args:
            e_values: Sequence of e-values
            change_point: Known change point
            
        Returns:
            Tuple of (decay rate, standard error)
        """
        e_values = np.asarray(e_values)
        post_change = e_values[change_point:]
        
        if len(post_change) < 2:
            return 0.0, float('inf')
            
        # Compute log ratios
        log_ratios = np.log(post_change[1:] / post_change[:-1])
        
        # Estimate decay rate
        decay_rate = -np.mean(log_ratios)
        decay_se = np.std(log_ratios) / np.sqrt(len(log_ratios))
        
        return decay_rate, decay_se
    
    def compute_efficiency_metrics(self, 
                                 detector: ConformalCUSUM,
                                 n_pre: int = 1000,
                                 n_post: int = 1000,
                                 n_trials: int = 100) -> dict:
        """
        Compute comprehensive efficiency metrics.
        
        Args:
            detector: CUSUM detector to evaluate
            n_pre: Pre-change sample size
            n_post: Post-change sample size
            n_trials: Number of Monte Carlo trials
            
        Returns:
            Dictionary of efficiency metrics
        """
        detection_delays = []
        false_alarms = []
        decay_rates = []
        
        for _ in range(n_trials):
            # Generate data
            pre_data = self.pre_dist(n_pre)
            post_data = self.post_dist(n_post)
            data = np.concatenate([pre_data, post_data])
            
            # Process with detector
            detector.reset()
            e_values = []
            detected = False
            
            for x in data:
                # Compute e-value (simplified)
                e_value = np.exp(-0.5 * (x - 0.5)**2 + 0.5 * x**2)
                e_values.append(e_value)
                
                # Update detector
                result = detector.update(e_value)
                
                # Check for first detection after change point
                if not detected and result.n_alarms > 0:
                    last_alarm = result.alarms[-1]
                    if last_alarm > n_pre:
                        detection_delays.append(last_alarm - n_pre)
                        detected = True
                    else:
                        false_alarms.append(last_alarm)
            
            # Analyze decay
            decay_rate, _ = self.analyze_decay(e_values, n_pre)
            decay_rates.append(decay_rate)
        
        return {
            'mean_detection_delay': np.mean(detection_delays),
            'detection_delay_std': np.std(detection_delays),
            'false_alarm_rate': len(false_alarms) / n_trials,
            'mean_decay_rate': np.mean(decay_rates),
            'decay_rate_std': np.std(decay_rates)
        }