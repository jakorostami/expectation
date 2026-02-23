"""
Tests for e-value merging functions.

Based on:
    Vovk & Wang (2024) "Merging sequential e-values via martingales"
    Ramdas & Wang (2025) "Hypothesis testing with e-values", Chapter 8
"""

import pytest
import numpy as np
from math import comb

from expectation.modules.merging import (
    MergingFunction,
    MergingConfig,
    MergingResult,
    ArithmeticMeanMerger,
    UStatisticMerger,
    LambdaProductMerger,
    SegmentProductMerger,
    ProductMerger,
    create_merger,
    arithmetic_mean_merge,
    u_statistic_merge,
    lambda_product_merge,
    segment_product_merge,
)


class TestArithmeticMeanMerger:
    def test_basic_merge(self):
        merger = ArithmeticMeanMerger(K=3)
        e = np.array([2.0, 4.0, 6.0])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(4.0)
        assert result.K == 3
        assert result.merging_function == MergingFunction.ARITHMETIC_MEAN
        assert result.is_valid is True

    def test_unit_e_values(self):
        """Under null, E[e_k] = 1, so mean = 1."""
        merger = ArithmeticMeanMerger(K=5)
        e = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(1.0)

    def test_single_e_value(self):
        merger = ArithmeticMeanMerger(K=1)
        e = np.array([3.5])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(3.5)

    def test_gambling_system_consistency(self):
        """Sequential application of gambling system reproduces batch result."""
        K = 5
        merger = ArithmeticMeanMerger(K=K)
        e_values = [2.0, 0.5, 3.0, 1.5, 4.0]

        # Batch
        batch_result = merger.merge(np.array(e_values))

        # Sequential via gambling system
        S = 1.0
        for k in range(K):
            s_k = merger.gambling_system(e_values[:k], k)
            S *= (1.0 + s_k * (e_values[k] - 1.0))

        assert S == pytest.approx(batch_result.merged_e_value, rel=1e-12)

    def test_log_merged(self):
        merger = ArithmeticMeanMerger(K=2)
        e = np.array([2.0, 4.0])
        result = merger.merge(e)
        assert result.log_merged_e_value == pytest.approx(np.log(3.0))

    def test_invalid_K(self):
        with pytest.raises(ValueError):
            ArithmeticMeanMerger(K=0)


class TestUStatisticMerger:
    def test_u0_equals_one(self):
        """U_0(e) = 1 for any e-values."""
        merger = UStatisticMerger(n=0, K=4)
        e = np.array([2.0, 3.0, 5.0, 7.0])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(1.0)

    def test_u1_equals_mean(self):
        """U_1(e) = arithmetic mean."""
        merger = UStatisticMerger(n=1, K=4)
        e = np.array([2.0, 4.0, 6.0, 8.0])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(5.0)

    def test_uK_equals_product(self):
        """U_K(e) = product."""
        e = np.array([2.0, 3.0, 5.0])
        merger = UStatisticMerger(n=3, K=3)
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(30.0)

    def test_u2_manual(self):
        """Manual verification: U_2([2, 3, 4]) = (2*3 + 2*4 + 3*4) / C(3,2) = (6+8+12)/3 = 26/3."""
        merger = UStatisticMerger(n=2, K=3)
        e = np.array([2.0, 3.0, 4.0])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(26.0 / 3.0)

    def test_monotonicity_in_order(self):
        """For e > 1, U_n is monotonically increasing in n (Proposition 8.16)."""
        e = np.array([2.0, 3.0, 4.0, 5.0])
        K = len(e)
        u_values = []
        for n in range(K + 1):
            merger = UStatisticMerger(n=n, K=K)
            result = merger.merge(e)
            u_values.append(result.merged_e_value)

        for i in range(len(u_values) - 1):
            assert u_values[i] <= u_values[i + 1] + 1e-10

    def test_n_exceeds_K_raises(self):
        with pytest.raises(ValueError, match="n must be <= K"):
            UStatisticMerger(n=5, K=3)

    def test_gambling_system_consistency(self):
        """Sequential gambling system reproduces batch U_2."""
        K = 4
        n = 2
        merger = UStatisticMerger(n=n, K=K)
        e_values = [2.0, 3.0, 4.0, 5.0]

        batch_result = merger.merge(np.array(e_values))

        S = 1.0
        for k in range(K):
            s_k = merger.gambling_system(e_values[:k], k)
            S *= (1.0 + s_k * (e_values[k] - 1.0))

        assert S == pytest.approx(batch_result.merged_e_value, rel=1e-10)


class TestLambdaProductMerger:
    def test_lambda_one_equals_product(self):
        """lambda=1 should equal the standard product."""
        merger = LambdaProductMerger(lambda_param=1.0)
        e = np.array([2.0, 3.0, 5.0])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(30.0)

    def test_small_lambda_shrinks_toward_one(self):
        """Small lambda hedges toward 1."""
        e = np.array([2.0, 3.0, 5.0])
        merger_small = LambdaProductMerger(lambda_param=0.01)
        merger_large = LambdaProductMerger(lambda_param=0.99)

        result_small = merger_small.merge(e)
        result_large = merger_large.merge(e)

        # Small lambda should be closer to 1
        assert abs(result_small.merged_e_value - 1.0) < abs(
            result_large.merged_e_value - 1.0
        )

    def test_mathematical_identity(self):
        """prod(1 - lambda + lambda * e_k) manual check."""
        lam = 0.3
        e = np.array([2.0, 4.0])
        merger = LambdaProductMerger(lambda_param=lam)
        result = merger.merge(e)

        expected = (1 - 0.3 + 0.3 * 2.0) * (1 - 0.3 + 0.3 * 4.0)
        assert result.merged_e_value == pytest.approx(expected)

    def test_constant_gambling_system(self):
        """Gambling system is constant lambda."""
        merger = LambdaProductMerger(lambda_param=0.7)
        for k in range(10):
            assert merger.gambling_system([1.0] * k, k) == pytest.approx(0.7)

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError):
            LambdaProductMerger(lambda_param=0.0)
        with pytest.raises(ValueError):
            LambdaProductMerger(lambda_param=1.5)
        with pytest.raises(ValueError):
            LambdaProductMerger(lambda_param=-0.1)

    def test_log_space_stability(self):
        """Log-space computation should handle large products."""
        merger = LambdaProductMerger(lambda_param=0.5)
        e = np.array([10.0] * 100)
        result = merger.merge(e)
        expected_log = 100 * np.log(1 - 0.5 + 0.5 * 10.0)
        assert result.log_merged_e_value == pytest.approx(expected_log, rel=1e-12)


class TestSegmentProductMerger:
    def test_singletons_equal_product(self):
        """Segments of size 1 = standard product."""
        e = np.array([2.0, 3.0, 5.0, 7.0])
        # Singletons: segments at [1, 2, 3] -> 4 segments of size 1
        merger = SegmentProductMerger(segments=[1, 2, 3], K=4)
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(2.0 * 3.0 * 5.0 * 7.0)

    def test_two_segment_manual(self):
        """[2,4,6,8] with segments=[2] -> mean([2,4]) * mean([6,8]) = 3 * 7 = 21."""
        e = np.array([2.0, 4.0, 6.0, 8.0])
        merger = SegmentProductMerger(segments=[2], K=4)
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(21.0)

    def test_three_segments(self):
        """Three segments manual check."""
        e = np.array([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
        # segments = [2, 4] -> [0,2), [2,4), [4,6)
        merger = SegmentProductMerger(segments=[2, 4], K=6)
        result = merger.merge(e)
        expected = np.mean([1.0, 3.0]) * np.mean([5.0, 2.0]) * np.mean([4.0, 6.0])
        assert result.merged_e_value == pytest.approx(expected)

    def test_invalid_segments_raises(self):
        with pytest.raises(ValueError):
            SegmentProductMerger(segments=[], K=5)
        with pytest.raises(ValueError):
            SegmentProductMerger(segments=[0], K=5)
        with pytest.raises(ValueError):
            SegmentProductMerger(segments=[3, 2], K=5)
        with pytest.raises(ValueError):
            SegmentProductMerger(segments=[5], K=5)

    def test_gambling_system_consistency(self):
        """Sequential application reproduces batch result."""
        K = 6
        segments = [2, 4]
        merger = SegmentProductMerger(segments=segments, K=K)
        e_values = [2.0, 4.0, 3.0, 5.0, 1.5, 2.5]

        batch_result = merger.merge(np.array(e_values))

        S = 1.0
        for k in range(K):
            s_k = merger.gambling_system(e_values[:k], k)
            S *= (1.0 + s_k * (e_values[k] - 1.0))

        assert S == pytest.approx(batch_result.merged_e_value, rel=1e-10)


class TestProductMerger:
    def test_basic_product(self):
        merger = ProductMerger()
        e = np.array([2.0, 3.0, 5.0])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(30.0)
        assert result.merging_function == MergingFunction.PRODUCT

    def test_gambling_system_always_one(self):
        merger = ProductMerger()
        for k in range(20):
            assert merger.gambling_system([1.0] * k, k) == 1.0

    def test_log_product(self):
        merger = ProductMerger()
        e = np.array([2.0, 3.0, 5.0])
        result = merger.merge(e)
        assert result.log_merged_e_value == pytest.approx(
            np.log(2.0) + np.log(3.0) + np.log(5.0)
        )

    def test_zero_e_value(self):
        merger = ProductMerger()
        e = np.array([2.0, 0.0, 5.0])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(0.0)
        assert result.log_merged_e_value == -np.inf

    def test_single_e_value(self):
        merger = ProductMerger()
        e = np.array([7.0])
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(7.0)


class TestProposition2:
    """
    Proposition 2 of V&W (2024) / Proposition 8.16 of R&W (2025):
    Among all precise se-merging functions, the product has the largest
    variance. Monte Carlo verification.
    """

    def test_product_largest_variance(self):
        np.random.seed(42)
        K = 5
        n_sim = 5000

        # Generate i.i.d. e-values under null: E[e] = 1 (exponential(1))
        e_matrix = np.random.exponential(1.0, size=(n_sim, K))

        product_merger = ProductMerger()
        mean_merger = ArithmeticMeanMerger(K=K)
        u2_merger = UStatisticMerger(n=2, K=K)
        lambda_merger = LambdaProductMerger(lambda_param=0.5)

        mergers = {
            "product": product_merger,
            "mean": mean_merger,
            "u2": u2_merger,
            "lambda": lambda_merger,
        }

        variances = {}
        for name, merger in mergers.items():
            merged_values = [merger.merge(e_matrix[i]).merged_e_value for i in range(n_sim)]
            variances[name] = np.var(merged_values)

        # Product should have the largest variance
        for name in ["mean", "u2", "lambda"]:
            assert variances["product"] > variances[name], (
                f"Product variance {variances['product']:.4f} should exceed "
                f"{name} variance {variances[name]:.4f}"
            )

    def test_null_mean_bounded(self):
        """Under null (E[e_k]=1), all merging functions have E[F] <= 1."""
        np.random.seed(123)
        K = 4
        n_sim = 10000

        e_matrix = np.random.exponential(1.0, size=(n_sim, K))

        mergers = [
            ArithmeticMeanMerger(K=K),
            UStatisticMerger(n=2, K=K),
            LambdaProductMerger(lambda_param=0.5),
        ]

        for merger in mergers:
            merged = [merger.merge(e_matrix[i]).merged_e_value for i in range(n_sim)]
            mean_merged = np.mean(merged)
            # E[F] should be close to 1 (or <=1 for sub-merging functions)
            # Allow some Monte Carlo noise
            assert mean_merged < 1.3, (
                f"{merger.__class__.__name__}: E[F] = {mean_merged:.4f} exceeds 1.3"
            )


class TestMergingConfigFactory:
    def test_factory_arithmetic_mean(self):
        config = MergingConfig(
            merging_function=MergingFunction.ARITHMETIC_MEAN, K=10
        )
        merger = create_merger(config)
        assert isinstance(merger, ArithmeticMeanMerger)

    def test_factory_u_statistic(self):
        config = MergingConfig(
            merging_function=MergingFunction.U_STATISTIC, K=10, u_order=3
        )
        merger = create_merger(config)
        assert isinstance(merger, UStatisticMerger)
        assert merger.n == 3

    def test_factory_lambda_product(self):
        config = MergingConfig(
            merging_function=MergingFunction.LAMBDA_PRODUCT, lambda_param=0.3
        )
        merger = create_merger(config)
        assert isinstance(merger, LambdaProductMerger)
        assert merger.lambda_param == 0.3

    def test_factory_segment_product(self):
        config = MergingConfig(
            merging_function=MergingFunction.SEGMENT_PRODUCT,
            K=10,
            segments=[3, 7],
        )
        merger = create_merger(config)
        assert isinstance(merger, SegmentProductMerger)

    def test_factory_product(self):
        config = MergingConfig(merging_function=MergingFunction.PRODUCT)
        merger = create_merger(config)
        assert isinstance(merger, ProductMerger)

    def test_config_is_frozen(self):
        config = MergingConfig(merging_function=MergingFunction.PRODUCT)
        with pytest.raises(Exception):
            config.merging_function = MergingFunction.ARITHMETIC_MEAN

    def test_result_is_frozen(self):
        result = MergingResult(
            merged_e_value=1.0,
            log_merged_e_value=0.0,
            K=1,
            merging_function=MergingFunction.PRODUCT,
            is_valid=True,
        )
        with pytest.raises(Exception):
            result.merged_e_value = 2.0

    def test_factory_missing_K_raises(self):
        config = MergingConfig(
            merging_function=MergingFunction.ARITHMETIC_MEAN
        )
        with pytest.raises(ValueError, match="K is required"):
            create_merger(config)

    def test_factory_missing_segments_raises(self):
        config = MergingConfig(
            merging_function=MergingFunction.SEGMENT_PRODUCT, K=10
        )
        with pytest.raises(ValueError, match="segments is required"):
            create_merger(config)


class TestConvenienceFunctions:
    def test_arithmetic_mean(self):
        e = np.array([2.0, 4.0, 6.0])
        assert arithmetic_mean_merge(e) == pytest.approx(4.0)

    def test_u_statistic(self):
        e = np.array([2.0, 3.0, 4.0])
        # U_2 = (6 + 8 + 12) / 3 = 26/3
        assert u_statistic_merge(e, n=2) == pytest.approx(26.0 / 3.0)

    def test_lambda_product(self):
        e = np.array([2.0, 3.0])
        lam = 0.5
        expected = (0.5 + 0.5 * 2.0) * (0.5 + 0.5 * 3.0)
        assert lambda_product_merge(e, lambda_param=lam) == pytest.approx(expected)

    def test_segment_product(self):
        e = np.array([2.0, 4.0, 6.0, 8.0])
        result = segment_product_merge(e, segments=[2])
        assert result == pytest.approx(21.0)

    def test_convenience_matches_class(self):
        """Convenience functions match class-based API."""
        e = np.array([1.5, 2.5, 3.5, 4.5])

        merger = ArithmeticMeanMerger(K=4)
        assert arithmetic_mean_merge(e) == pytest.approx(
            merger.merge(e).merged_e_value
        )

        merger = UStatisticMerger(n=2, K=4)
        assert u_statistic_merge(e, n=2) == pytest.approx(
            merger.merge(e).merged_e_value
        )

        merger = LambdaProductMerger(lambda_param=0.4)
        assert lambda_product_merge(e, lambda_param=0.4) == pytest.approx(
            merger.merge(e).merged_e_value
        )

        merger = SegmentProductMerger(segments=[2], K=4)
        assert segment_product_merge(e, segments=[2]) == pytest.approx(
            merger.merge(e).merged_e_value
        )


class TestEdgeCases:
    def test_empty_array_raises(self):
        mergers = [
            ArithmeticMeanMerger(K=3),
            UStatisticMerger(n=1, K=3),
            LambdaProductMerger(lambda_param=0.5),
            ProductMerger(),
        ]
        for merger in mergers:
            with pytest.raises(ValueError, match="non-empty"):
                merger.merge(np.array([]))

    def test_zero_e_values(self):
        """Merging functions should handle zero e-values gracefully."""
        e = np.array([0.0, 1.0, 2.0])

        # Arithmetic mean
        merger = ArithmeticMeanMerger(K=3)
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(1.0)
        assert result.is_valid is True

        # Product
        merger = ProductMerger()
        result = merger.merge(e)
        assert result.merged_e_value == pytest.approx(0.0)

    def test_large_e_values_numerical_stability(self):
        """Large e-values should not cause overflow in log-space methods."""
        e = np.array([1e100, 1e100])

        # Lambda product in log-space
        merger = LambdaProductMerger(lambda_param=0.5)
        result = merger.merge(e)
        expected_log = 2 * np.log(0.5 + 0.5 * 1e100)
        assert result.log_merged_e_value == pytest.approx(expected_log, rel=1e-10)

    def test_small_e_values(self):
        """Very small e-values should not underflow."""
        e = np.array([1e-100, 1e-100, 1e-100])

        merger = ArithmeticMeanMerger(K=3)
        result = merger.merge(e)
        assert result.merged_e_value > 0

    def test_negative_e_values_flagged(self):
        """Negative e-values should set is_valid=False."""
        e = np.array([-1.0, 2.0, 3.0])

        merger = ArithmeticMeanMerger(K=3)
        result = merger.merge(e)
        assert result.is_valid is False

    def test_merging_function_enum_values(self):
        """All enum values are accessible."""
        assert MergingFunction.ARITHMETIC_MEAN.value == "arithmetic_mean"
        assert MergingFunction.U_STATISTIC.value == "u_statistic"
        assert MergingFunction.LAMBDA_PRODUCT.value == "lambda_product"
        assert MergingFunction.SEGMENT_PRODUCT.value == "segment_product"
        assert MergingFunction.PRODUCT.value == "product"

    def test_reset_is_noop(self):
        """Reset should not fail on stateless mergers."""
        mergers = [
            ArithmeticMeanMerger(K=3),
            UStatisticMerger(n=1, K=3),
            LambdaProductMerger(lambda_param=0.5),
            SegmentProductMerger(segments=[1, 2], K=3),
            ProductMerger(),
        ]
        for merger in mergers:
            merger.reset()  # Should not raise
