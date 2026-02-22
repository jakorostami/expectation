//! e-BH procedure for FDR control.
//!
//! The e-value analogue of the Benjamini-Hochberg procedure.
//! Controls the False Discovery Rate (FDR) at level alpha.
//!
//! Algorithm (Ramdas & Wang 2025, Ch. 4, Section 4.2):
//!   1. Sort e-values in descending order: e_{(1)} >= e_{(2)} >= ... >= e_{(m)}
//!   2. Find k* = max{k : e_{(k)} >= m / (k * alpha)}
//!   3. Reject the top k* hypotheses (those with the largest e-values)
//!
//! In log space:
//!   Find k* = max{k : log_e_{(k)} >= ln(m) - ln(k) - ln(alpha)}
//!
//! The sort dominates cost: rayon's par_sort_unstable on 300K f64s ≈ 1ms.

use rayon::prelude::*;

/// Result of e-BH multiple testing.
pub struct BhResult {
    /// Per-hypothesis rejection flags (original order).
    pub rejected: Vec<bool>,
    /// Number of rejections (k*).
    pub n_rejected: usize,
}

/// Apply e-BH procedure to log e-values for FDR control.
///
/// # Arguments
/// * `log_e_values` - Log e-values (one per hypothesis)
/// * `alpha` - Target FDR level
pub fn e_bh(log_e_values: &[f64], alpha: f64) -> BhResult {
    let m = log_e_values.len();
    if m == 0 {
        return BhResult {
            rejected: vec![],
            n_rejected: 0,
        };
    }

    let log_m = (m as f64).ln();
    let log_alpha = alpha.ln();

    // Create (index, log_e) pairs and sort descending by log_e
    let mut indexed: Vec<(usize, f64)> = log_e_values.iter().copied().enumerate().collect();
    indexed.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Step-up: find k* = max{k : log_e_{(k)} >= ln(m) - ln(k) - ln(alpha)}
    // k is 1-indexed: the k-th largest e-value
    let mut k_star = 0usize;
    for (rank_0, &(_orig_idx, log_e)) in indexed.iter().enumerate() {
        let k = rank_0 + 1; // 1-indexed rank
        // threshold_k = m / (k * alpha), in log: ln(m/(k*alpha)) = ln(m) - ln(k) - ln(alpha)
        let log_threshold_k = log_m - (k as f64).ln() - log_alpha;
        if log_e >= log_threshold_k {
            k_star = k;
        } else {
            break; // Since sorted descending, once we fail, all subsequent fail too
        }
    }

    // Reject the top k* hypotheses
    let mut rejected = vec![false; m];
    for &(orig_idx, _) in &indexed[..k_star] {
        rejected[orig_idx] = true;
    }

    BhResult {
        rejected,
        n_rejected: k_star,
    }
}