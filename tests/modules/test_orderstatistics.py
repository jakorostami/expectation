import pytest
import numpy as np
from expectation.modules.orderstatistics import StaticOrderStatistics

class TestStaticOrderStatistics:
    @pytest.fixture
    def sample_data(self):
        return [1.0, 5.0, 2.0, 8.0, 3.0]
    
    @pytest.fixture
    def ordered_stats(self, sample_data):
        return StaticOrderStatistics(sample_data)
    
    def test_initialization(self, sample_data):
        stats = StaticOrderStatistics(sample_data)
        assert len(stats.sorted_values) == len(sample_data)
        assert all(x <= y for x, y in zip(stats.sorted_values[:-1], 
                                        stats.sorted_values[1:]))
    
    def test_get_order_statistic(self, ordered_stats):
        # First order statistic should be minimum
        assert ordered_stats.get_order_statistic(1) == min(ordered_stats.sorted_values)
        
        # Last order statistic should be maximum
        n = len(ordered_stats.sorted_values)
        assert ordered_stats.get_order_statistic(n) == max(ordered_stats.sorted_values)
        
        # Invalid indices should raise IndexError
        with pytest.raises(IndexError):
            ordered_stats.get_order_statistic(0)  # Too small
        
        with pytest.raises(IndexError):
            ordered_stats.get_order_statistic(len(ordered_stats.sorted_values) + 1)  # Too large
    
    def test_count_less(self, ordered_stats):

        max_val = max(ordered_stats.sorted_values)
        assert ordered_stats.count_less(max_val + 1) == len(ordered_stats.sorted_values)
        
        min_val = min(ordered_stats.sorted_values)
        assert ordered_stats.count_less(min_val) == 0
        
        middle_val = ordered_stats.sorted_values[len(ordered_stats.sorted_values)//2]
        count = ordered_stats.count_less(middle_val)
        assert 0 < count < len(ordered_stats.sorted_values)
    
    def test_count_less_or_equal(self, ordered_stats):

        max_val = max(ordered_stats.sorted_values)
        assert ordered_stats.count_less_or_equal(max_val) == len(ordered_stats.sorted_values)
        
        min_val = min(ordered_stats.sorted_values)
        min_count = sum(1 for x in ordered_stats.sorted_values if x == min_val)
        assert ordered_stats.count_less_or_equal(min_val) == min_count
    
    def test_size(self, ordered_stats, sample_data):
        assert ordered_stats.size() == len(sample_data)
        
        empty_stats = StaticOrderStatistics([])
        assert empty_stats.size() == 0
    
    def test_consistency(self, ordered_stats):
        for i in range(1, ordered_stats.size() + 1):
            value = ordered_stats.get_order_statistic(i)
            # Count of values less than order statistic i should be i-1
            assert ordered_stats.count_less(value) <= i-1
            # Count of values less or equal should be at least i
            assert ordered_stats.count_less_or_equal(value) >= i
    
    @pytest.mark.parametrize("data", [
        [1.0],  # single value
        [1.0, 1.0, 1.0],  # repeated values
        list(range(100)),  # many values
        [-np.inf, 0, np.inf]  # extreme values
    ])
    def test_special_cases(self, data):
        stats = StaticOrderStatistics(data)
        
        assert stats.size() == len(data)
        assert stats.get_order_statistic(1) == min(data)
        assert stats.get_order_statistic(len(data)) == max(data)