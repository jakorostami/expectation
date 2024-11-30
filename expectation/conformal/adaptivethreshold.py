class AdaptiveThresholdHandler:
    """
    Handles adaptive thresholding for conformal e-testing by dynamically adjusting the threshold to control the false alarm rate.
    """
    def __init__(self, target_false_alarm_rate: float = 0.05, learning_rate: float = 0.01, min_threshold: float = 1.0, max_threshold: float = 50.0):
        """
        Initialize the adaptive threshold handler.
        
        Args:
            target_false_alarm_rate (float): Target rate for false alarms (between 0 and 1).
            learning_rate (float): Rate at which the threshold adjusts in response to observed false alarms.
            min_threshold (float): Minimum value for the threshold to prevent going below a useful range.
            max_threshold (float): Maximum value for the threshold to prevent it from becoming too large.
        """
        self.target_rate = target_false_alarm_rate
        self.learning_rate = learning_rate
        self.threshold = 10.0  # Start with a higher initial threshold for sensitivity
        self.alarm_history = []  # A list of booleans to track if alarms were false or true
        self.min_threshold = min_threshold  # Ensure that the threshold doesn't drop below this level
        self.max_threshold = max_threshold  # Ensure that the threshold doesn't exceed this level
        self.cumulative_error = 0.0  # Track cumulative error for smoother adaptation
        
    def update(self, current_rate: float) -> float:
        """
        Update the threshold based on the observed false alarm rate.
        The threshold increases if the observed false alarm rate is below the target, and decreases otherwise.
        
        Args:
            current_rate (float): The current observed false alarm rate.
        
        Returns:
            float: The updated threshold value.
        """
        # Calculate the difference between the observed rate and the target rate
        error = current_rate - self.target_rate
        
        # Apply a cumulative weighted average to smooth the adaptation
        self.cumulative_error = 0.9 * self.cumulative_error + 0.1 * error
        
        # Apply a non-linear adjustment factor to make adaptation more responsive to large deviations
        if self.cumulative_error > 0:
            adjustment_factor = 1 + (self.learning_rate * abs(self.cumulative_error))  # Increase more conservatively if below target
        else:
            adjustment_factor = 1 / (1 + (self.learning_rate * abs(self.cumulative_error)))  # Decrease more aggressively if above target
        
        # Update the threshold based on the adjustment factor
        self.threshold *= adjustment_factor
        
        # Ensure the threshold does not fall below the minimum value or exceed the maximum value
        self.threshold = max(self.min_threshold, min(self.threshold, self.max_threshold))
        
        return self.threshold
    
    def record_alarm(self, is_false_alarm: bool):
        """
        Record whether an alarm was a false alarm.
        
        Args:
            is_false_alarm (bool): True if the alarm was a false alarm, False otherwise.
        """
        self.alarm_history.append(is_false_alarm)
        
    def get_current_rate(self) -> float:
        """
        Calculate the current false alarm rate based on recorded alarms.
        
        Returns:
            float: The current false alarm rate. Returns 0.0 if no alarms have occurred.
        """
        if not self.alarm_history:
            return 0.0
        return sum(self.alarm_history) / len(self.alarm_history)
    
    def reset(self):
        """
        Reset the alarm history and threshold to start afresh.
        """
        self.alarm_history = []
        self.threshold = 10.0  # Reset to a reasonable initial value
        self.cumulative_error = 0.0  # Reset cumulative error
