//! Structure-of-Arrays (SoA) state for parallel sequential tests.
//!
//! SoA layout ensures each cache line (64B = 8 x f64) carries 8 consecutive
//! values of the *same* field, enabling sequential access patterns in the
//! parallel hot loop and SIMD auto-vectorization of arithmetic.
//!
//! Total memory for 300K tests: ~13.5 MB (fits in L3 cache).

/// Per-test mutable state stored in Structure-of-Arrays layout.
///
/// Each `Vec` is accessed sequentially in the hot loop, maximizing
/// cache line utilization and enabling auto-vectorization.
pub struct ParTestState {
    /// Running sum of observations: sum_{i=1}^{n} x_i
    pub data_sum: Vec<f64>,
    /// Running sum of squared observations: sum_{i=1}^{n} x_i^2
    pub data_sum_sq: Vec<f64>,
    /// Number of observations processed
    pub count: Vec<u32>,
    /// Previous cumulative log e-value (for sequential e-value computation)
    pub prev_log_e_cum: Vec<f64>,
    /// Current log e-process value (running product in log space)
    pub log_e_process: Vec<f64>,
    /// Maximum log martingale value seen (for Ville's inequality stopping)
    pub max_log_m: Vec<f64>,
    /// Per-test rejection flag (Ville's inequality: max M >= 1/alpha)
    pub rejected: Vec<bool>,
}

impl ParTestState {
    /// Allocate zeroed state for `n` tests.
    pub fn zeros(n: usize) -> Self {
        Self {
            data_sum: vec![0.0; n],
            data_sum_sq: vec![0.0; n],
            count: vec![0; n],
            prev_log_e_cum: vec![0.0; n],
            log_e_process: vec![0.0; n],
            max_log_m: vec![f64::NEG_INFINITY; n],
            rejected: vec![false; n],
        }
    }

    /// Number of tests.
    #[inline]
    pub fn n_tests(&self) -> usize {
        self.data_sum.len()
    }
}
