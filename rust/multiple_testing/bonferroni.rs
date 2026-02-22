//! e-Bonferroni procedure for FWER control.
//!
//! The simplest multiple testing correction for e-values.
//! Reject hypothesis i if e_i >= m / alpha, where m is the number of tests.
//!
//! In log space: reject if log_e[i] >= ln(m / alpha) = ln(m) + ln(1/alpha).
//!
//! FWER control follows directly from Markov's inequality applied to the
//! average e-value (Ramdas & Wang 2025, Ch. 4, Section 4.1).
//!
//! Trivially parallel: each comparison is independent.

use rayon::prelude::*;

/// Result of e-Bonferroni multiple testing.
pub struct BonferroniResult {
    /// Per-hypothesis rejection flags.
    pub rejected: Vec<bool>,
    /// Number of rejections.
    pub n_rejected: usize,
}

/// Apply e-Bonferroni correction to log e-values.
///
/// Rejects hypothesis i if `log_e_values[i] >= ln(m/alpha)`.
///
/// # Arguments
/// * `log_e_values` - Log e-values (one per hypothesis)
/// * `alpha` - Target FWER level
pub fn e_bonferroni(log_e_values: &[f64], alpha: f64) -> BonferroniResult {
    let m = log_e_values.len() as f64;
    let log_threshold = (m / alpha).ln();

    let rejected: Vec<bool> = log_e_values
        .par_iter()
        .map(|&log_e| log_e >= log_threshold)
        .collect();

    let n_rejected = rejected.par_iter().filter(|&&r| r).count();

    BonferroniResult {
        rejected,
        n_rejected,
    }
}