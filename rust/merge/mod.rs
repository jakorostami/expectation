//! Spatial merging of K e-values + temporal accumulation into an e-process.
//!
//! Implements five merging functions from Vovk & Wang (2024) and
//! Ramdas & Wang (2025) Ch. 8, plus temporal combiners to accumulate
//! the merged stream into an anytime-valid e-process for the
//! intersection hypothesis (all K nulls true).
//!
//! # Merging functions (spatial, per time step)
//!
//! All are martingale merging functions (V&W 2024, Corollary 1):
//! - **ArithmeticMean**: F(e) = mean(e_k)  [Proposition 8.3]
//! - **UStatistic**: F(e) = U_n(e) via ESP  [Definition 8.9]
//! - **LambdaProduct**: F(e) = prod(1-λ+λ*e_k)  [Definition 8.5]
//! - **SegmentProduct**: partition → mean per seg → product  [Definition 8.10]
//! - **Product**: F(e) = prod(e_k)  [Theorem 8.4]
//!
//! # Temporal combiners (accumulate merged stream over time)
//!
//! Same combiner strategies as the per-test engine (R&W 2025, Def. 7.21):
//! - AllIn: M_t = prod E_merged_s (cumulative supermartingale)
//! - Conservative: M_t = prod((1-λ) + λ*E_merged_s)
//! - EmpiricallyAdaptive: ONS-based λ_t = clamp(S1/(S2+ε), [0, γ])
//!
//! References:
//! - Vovk & Wang (2024). Merging sequential e-values via martingales.
//! - Ramdas & Wang (2025). Hypothesis testing with e-values, Ch. 7-8.

pub mod state;

pub use state::MergeState;

/// Which merging function to apply spatially across K e-values.
///
/// Reference: Vovk & Wang (2024), Section 4.
#[derive(Debug, Clone)]
pub enum MergeFunction {
    /// F(e) = mean(e_k). Most conservative. [R&W 2025, Proposition 8.3]
    ArithmeticMean,
    /// F(e) = U_n(e). ESP-based, O(K*n). [R&W 2025, Definition 8.9]
    UStatistic { n: usize },
    /// F(e) = prod(1-λ+λ*e_k). Interpolates mean↔product. [R&W 2025, Definition 8.5]
    LambdaProduct { lambda: f64 },
    /// Partition → mean per segment → product. [R&W 2025, Definition 8.10]
    SegmentProduct { segments: Vec<usize> },
    /// F(e) = prod(e_k). Most aggressive. [R&W 2025, Theorem 8.4]
    Product,
}

/// Temporal combiner for the merged stream (same strategies as per-test).
///
/// Reference: Ramdas & Wang (2025), Definition 7.21.
#[derive(Debug, Clone, Copy)]
pub enum MergeCombinerType {
    /// λ_t = 1 for all t. E-process = cumulative product of merged e-values.
    AllIn,
    /// Fixed λ ∈ (0, 1). E-process = Π((1-λ) + λ·E_merged_t).
    Conservative { lambda: f64 },
    /// ONS-based adaptive: λ_t = clamp(S1/(S2+ε), [0, γ]).
    EmpiricallyAdaptive { gamma: f64, epsilon: f64 },
}

/// Configuration for the global merge feature.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Which spatial merging function to apply.
    pub function: MergeFunction,
    /// How to accumulate merged e-values temporally.
    pub combiner: MergeCombinerType,
    /// Whether to include rejected tests in the merge.
    /// When false, rejected tests are replaced with 1.0 (neutral element).
    /// V&W Eq. 4 requires fixed K for the martingale representation.
    pub include_rejected: bool,
}

// ── Spatial merge functions ─────────────────────────────────────────────

/// Apply the configured merge function to K e-values.
///
/// All functions take e-values (not log e-values) and return a single
/// merged e-value.
pub fn merge_e_values(e_values: &[f64], function: &MergeFunction) -> f64 {
    match function {
        MergeFunction::ArithmeticMean => merge_arithmetic_mean(e_values),
        MergeFunction::UStatistic { n } => merge_u_statistic(e_values, *n),
        MergeFunction::LambdaProduct { lambda } => merge_lambda_product(e_values, *lambda),
        MergeFunction::SegmentProduct { segments } => merge_segment_product(e_values, segments),
        MergeFunction::Product => merge_product(e_values),
    }
}

/// Arithmetic mean: F(e) = (e_1 + ... + e_K) / K.
///
/// Reference: R&W (2025), Proposition 8.3.
#[inline]
fn merge_arithmetic_mean(e_values: &[f64]) -> f64 {
    let k = e_values.len() as f64;
    if k == 0.0 {
        return 1.0;
    }
    let sum: f64 = e_values.iter().sum();
    sum / k
}

/// U-statistic of order n: U_n = p_n / C(K, n).
///
/// Computed via elementary symmetric polynomials (ESP) recurrence.
/// p_j(e_1,...,e_k) = p_j(e_1,...,e_{k-1}) + e_k * p_{j-1}(...)
///
/// Time O(K*n), space O(n). Same algorithm as Python `merging.py`.
///
/// Special cases: U_0 = 1, U_1 = arithmetic mean, U_K = product.
///
/// Reference: V&W (2024), Section 4, Eq. (13); R&W (2025), Definition 8.9.
#[inline]
fn merge_u_statistic(e_values: &[f64], n: usize) -> f64 {
    let k = e_values.len();
    if n == 0 {
        return 1.0;
    }
    if n > k {
        return 0.0;
    }
    if n == k {
        return merge_product(e_values);
    }

    // ESP recurrence: O(K*n)
    let mut p = vec![0.0_f64; n + 1];
    p[0] = 1.0;

    for &e in e_values {
        // Traverse backwards to avoid overwriting values needed for update
        for j in (1..=n).rev() {
            p[j] += e * p[j - 1];
        }
    }

    // C(K, n) via integer arithmetic
    let binom = binomial_coeff(k, n);
    p[n] / binom
}

/// Lambda-product: F(e) = prod(1 - λ + λ * e_k).
///
/// Computed in log-space for numerical stability.
///
/// Reference: V&W (2024), Section 4; R&W (2025), Definition 8.5.
#[inline]
fn merge_lambda_product(e_values: &[f64], lambda: f64) -> f64 {
    let mut log_sum = 0.0_f64;
    for &e in e_values {
        let term = (1.0 - lambda) + lambda * e;
        if term <= 0.0 {
            return 0.0;
        }
        log_sum += term.ln();
    }
    log_sum.exp()
}

/// Segment-product: partition → mean per segment → product of segment means.
///
/// `segments` contains the start indices of each segment (except the first
/// which implicitly starts at 0). E.g., segments=[3, 7] with K=10 gives
/// three segments: [0,3), [3,7), [7,10).
///
/// Reference: V&W (2024), Section 4; R&W (2025), Definition 8.10.
#[inline]
fn merge_segment_product(e_values: &[f64], segments: &[usize]) -> f64 {
    let k = e_values.len();
    if k == 0 {
        return 1.0;
    }

    // Build segment boundaries: [0, segments[0], segments[1], ..., K]
    let mut boundaries = Vec::with_capacity(segments.len() + 2);
    boundaries.push(0usize);
    boundaries.extend_from_slice(segments);
    boundaries.push(k);

    let mut log_merged = 0.0_f64;
    for window in boundaries.windows(2) {
        let start = window[0];
        let end = window[1];
        if end <= start {
            continue;
        }
        let seg_len = (end - start) as f64;
        let seg_sum: f64 = e_values[start..end].iter().sum();
        let seg_mean = seg_sum / seg_len;
        if seg_mean <= 0.0 {
            return 0.0;
        }
        log_merged += seg_mean.ln();
    }
    log_merged.exp()
}

/// Product: F(e) = e_1 * e_2 * ... * e_K.
///
/// Computed in log-space for stability.
///
/// Reference: V&W (2024), Eq. (12); R&W (2025), Theorem 8.4.
#[inline]
fn merge_product(e_values: &[f64]) -> f64 {
    let mut log_sum = 0.0_f64;
    for &e in e_values {
        if e <= 0.0 {
            return 0.0;
        }
        log_sum += e.ln();
    }
    log_sum.exp()
}

/// Exact binomial coefficient C(n, k) as f64.
fn binomial_coeff(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    // Use the smaller of k, n-k for efficiency
    let k = k.min(n - k);
    let mut result = 1.0_f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

// ── Temporal update ─────────────────────────────────────────────────────

/// Apply spatial merge + temporal accumulation after `step_parallel()`.
///
/// This is a standalone function (not a method on ParallelSequentialTest)
/// to avoid borrow-checker issues with `&mut self`.
///
/// Steps:
/// 1. Read `log_e_sequential` → convert to e-values via exp()
/// 2. Optionally mask rejected tests with 1.0 (neutral element)
/// 3. Apply merge function → single merged e-value
/// 4. Apply temporal combiner → update running merged e-process
/// 5. Check Ville's inequality on merged e-process
pub fn apply_merge(
    log_e_sequential: &[f64],
    rejected: &[bool],
    merge_config: &MergeConfig,
    merge_state: &mut MergeState,
    log_threshold: f64,
    time_step: u64,
) {
    let k = log_e_sequential.len();

    // Step 1-2: Convert to e-values, optionally masking rejected tests
    let mut e_values = Vec::with_capacity(k);
    for i in 0..k {
        if !merge_config.include_rejected && rejected[i] {
            e_values.push(1.0); // Neutral element
        } else {
            e_values.push(log_e_sequential[i].exp());
        }
    }

    // Step 3: Spatial merge
    let merged_e_value = merge_e_values(&e_values, &merge_config.function);
    let log_merged_e_value = if merged_e_value > 0.0 {
        merged_e_value.ln()
    } else {
        f64::NEG_INFINITY
    };
    merge_state.log_merged_e_value = log_merged_e_value;

    // Step 4: Temporal accumulation
    update_merged_temporal(
        merge_state,
        merged_e_value,
        &merge_config.combiner,
        log_threshold,
        time_step,
    );
}

/// Update the temporal e-process with a new merged e-value.
///
/// Same combiner patterns as the per-test engine (update.rs):
/// - AllIn: log M_t += log(E_merged_t)
/// - Conservative: log M_t += log((1-λ) + λ*E_merged_t)
/// - EmpiricallyAdaptive: ONS-based λ_t, then log((1-λ_t) + λ_t*E_merged_t)
fn update_merged_temporal(
    state: &mut MergeState,
    merged_e_value: f64,
    combiner: &MergeCombinerType,
    log_threshold: f64,
    time_step: u64,
) {
    match combiner {
        MergeCombinerType::AllIn => {
            // M_t = prod E_merged_s → log M_t += log E_merged_t
            let log_e = if merged_e_value > 0.0 {
                merged_e_value.ln()
            } else {
                f64::NEG_INFINITY
            };
            state.log_merged_e_process += log_e;
            state.merged_lambda = 1.0;
        }
        MergeCombinerType::Conservative { lambda } => {
            // M_t = prod((1-λ) + λ*E_merged_s)
            let term = (1.0 - lambda) + lambda * merged_e_value;
            state.log_merged_e_process += term.ln();
            state.merged_lambda = *lambda;
        }
        MergeCombinerType::EmpiricallyAdaptive { gamma, epsilon } => {
            // λ_t from previous-step stats (F_{t-1}-measurable)
            let lambda_t = (state.sum_e_minus_1 / (state.sum_e_minus_1_sq + epsilon))
                .clamp(0.0, *gamma);
            state.merged_lambda = lambda_t;

            // log M_t += log((1-λ_t) + λ_t*E_merged_t)
            let term = (1.0 - lambda_t) + lambda_t * merged_e_value;
            state.log_merged_e_process += term.ln();

            // Update ONS stats for next step
            let e_minus_1 = merged_e_value - 1.0;
            state.sum_e_minus_1 += e_minus_1;
            state.sum_e_minus_1_sq += e_minus_1 * e_minus_1;
        }
    }

    // Step 5: P-value and Ville's inequality
    state.merged_p_value = 1.0_f64.min((-state.log_merged_e_process).exp());

    if state.log_merged_e_process > state.max_log_merged {
        state.max_log_merged = state.log_merged_e_process;
    }
    if state.max_log_merged >= log_threshold && !state.merged_rejected {
        state.merged_rejected = true;
        state.merged_stopping_time = time_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic_mean_basic() {
        let e = vec![2.0, 3.0, 5.0];
        let result = merge_arithmetic_mean(&e);
        assert!((result - 10.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_arithmetic_mean_unit() {
        // All e-values = 1 → mean = 1 (null property)
        let e = vec![1.0; 10];
        let result = merge_arithmetic_mean(&e);
        assert!((result - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_u_statistic_order_0() {
        let e = vec![2.0, 3.0, 5.0];
        assert!((merge_u_statistic(&e, 0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_u_statistic_order_1_is_mean() {
        let e = vec![2.0, 3.0, 5.0];
        let mean = merge_arithmetic_mean(&e);
        let u1 = merge_u_statistic(&e, 1);
        assert!((u1 - mean).abs() < 1e-14);
    }

    #[test]
    fn test_u_statistic_order_k_is_product() {
        let e = vec![2.0, 3.0, 5.0];
        let prod = merge_product(&e);
        let uk = merge_u_statistic(&e, 3);
        assert!((uk - prod).abs() < 1e-12);
    }

    #[test]
    fn test_lambda_product_lambda_1_is_product() {
        let e = vec![2.0, 3.0, 5.0];
        let prod = merge_product(&e);
        let lp = merge_lambda_product(&e, 1.0);
        assert!((lp - prod).abs() < 1e-12);
    }

    #[test]
    fn test_lambda_product_unit() {
        let e = vec![1.0; 5];
        let result = merge_lambda_product(&e, 0.5);
        assert!((result - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_segment_product_singletons_is_product() {
        // Each element in its own segment = product
        let e = vec![2.0, 3.0, 5.0];
        let segments = vec![1, 2]; // segments: [0,1), [1,2), [2,3)
        let sp = merge_segment_product(&e, &segments);
        let prod = merge_product(&e);
        assert!((sp - prod).abs() < 1e-12);
    }

    #[test]
    fn test_segment_product_single_segment_is_mean() {
        // One big segment = arithmetic mean
        let e = vec![2.0, 3.0, 5.0];
        let segments: Vec<usize> = vec![]; // single segment [0, 3)
        let sp = merge_segment_product(&e, &segments);
        let mean = merge_arithmetic_mean(&e);
        assert!((sp - mean).abs() < 1e-14);
    }

    #[test]
    fn test_product_basic() {
        let e = vec![2.0, 3.0, 5.0];
        let result = merge_product(&e);
        assert!((result - 30.0).abs() < 1e-12);
    }

    #[test]
    fn test_product_unit() {
        let e = vec![1.0; 10];
        let result = merge_product(&e);
        assert!((result - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_binomial_coeff() {
        assert!((binomial_coeff(5, 2) - 10.0).abs() < 1e-14);
        assert!((binomial_coeff(10, 3) - 120.0).abs() < 1e-12);
        assert!((binomial_coeff(0, 0) - 1.0).abs() < 1e-14);
        assert!((binomial_coeff(5, 0) - 1.0).abs() < 1e-14);
        assert!((binomial_coeff(5, 5) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_temporal_all_in() {
        let mut state = MergeState::new();
        let combiner = MergeCombinerType::AllIn;
        let log_threshold = (1.0 / 0.05_f64).ln();

        // Step 1: merged e-value = 2.0
        update_merged_temporal(&mut state, 2.0, &combiner, log_threshold, 1);
        assert!((state.log_merged_e_process - 2.0_f64.ln()).abs() < 1e-14);
        assert!((state.merged_lambda - 1.0).abs() < 1e-14);

        // Step 2: merged e-value = 3.0  → log M = ln(2) + ln(3) = ln(6)
        update_merged_temporal(&mut state, 3.0, &combiner, log_threshold, 2);
        assert!((state.log_merged_e_process - 6.0_f64.ln()).abs() < 1e-14);
    }

    #[test]
    fn test_temporal_conservative() {
        let mut state = MergeState::new();
        let combiner = MergeCombinerType::Conservative { lambda: 0.5 };
        let log_threshold = (1.0 / 0.05_f64).ln();

        // Step 1: merged e-value = 2.0 → log((1-0.5) + 0.5*2) = log(1.5)
        update_merged_temporal(&mut state, 2.0, &combiner, log_threshold, 1);
        let expected = 1.5_f64.ln();
        assert!((state.log_merged_e_process - expected).abs() < 1e-14);
    }

    #[test]
    fn test_apply_merge_basic() {
        let log_e_seq = vec![1.0_f64.ln(), 2.0_f64.ln(), 3.0_f64.ln()];
        let rejected = vec![false, false, false];
        let config = MergeConfig {
            function: MergeFunction::ArithmeticMean,
            combiner: MergeCombinerType::AllIn,
            include_rejected: true,
        };
        let mut state = MergeState::new();
        let log_threshold = (1.0 / 0.05_f64).ln();

        apply_merge(&log_e_seq, &rejected, &config, &mut state, log_threshold, 1);

        // Mean of [1, 2, 3] = 2.0
        let expected_merged = 2.0;
        let expected_log = expected_merged.ln();
        assert!((state.log_merged_e_value - expected_log).abs() < 1e-13);
        assert!((state.log_merged_e_process - expected_log).abs() < 1e-13);
    }

    #[test]
    fn test_include_rejected_false() {
        let log_e_seq = vec![2.0_f64.ln(), 5.0_f64.ln(), 3.0_f64.ln()];
        let rejected = vec![false, true, false]; // middle test rejected
        let config = MergeConfig {
            function: MergeFunction::ArithmeticMean,
            combiner: MergeCombinerType::AllIn,
            include_rejected: false,
        };
        let mut state = MergeState::new();
        let log_threshold = (1.0 / 0.05_f64).ln();

        apply_merge(&log_e_seq, &rejected, &config, &mut state, log_threshold, 1);

        // Rejected test replaced with 1.0: mean of [2, 1, 3] = 2.0
        let expected_merged = 2.0;
        assert!((state.log_merged_e_value - expected_merged.ln()).abs() < 1e-13);
    }
}
