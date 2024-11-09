import numpy as np
from typing import List
from abc import ABC, abstractmethod

class OrderStatisticInterface(ABC):
    """Abstract base class for order statistics."""
    
    @abstractmethod
    def get_order_statistic(self, order_index: int) -> float:
        """Get order statistic at given index."""
        pass
    
    @abstractmethod
    def count_less(self, value: float) -> int:
        """Count values less than given value."""
        pass
    
    @abstractmethod
    def count_less_or_equal(self, value: float) -> int:
        """Count values less than or equal to given value."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get size of the sample."""
        pass

class StaticOrderStatistics:
    """Implementation of static order statistics."""
    
    def __init__(self, values: List[float]):
        self.sorted_values = sorted(values)
    
    def get_order_statistic(self, order_index: int) -> float:
        """Get order statistic at given index."""
        if order_index < 1 or order_index > len(self.sorted_values):
            raise IndexError(f"Order index {order_index} out of bounds for length {len(self.sorted_values)}")
        return self.sorted_values[order_index - 1]
    
    def count_less(self, value: float) -> int:
        """Count values less than given value."""
        return np.searchsorted(self.sorted_values, value, side='left')
    
    def count_less_or_equal(self, value: float) -> int:
        """Count values less than or equal to given value."""
        return np.searchsorted(self.sorted_values, value, side='right')
    
    def size(self) -> int:
        """Get size of the sample."""
        return len(self.sorted_values)
