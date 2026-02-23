// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! e-Holm step-down procedure for FWER control.
//!
//! Tighter than e-Bonferroni: uses the step-down principle to adapt
//! thresholds as hypotheses are rejected.
//!
//! Algorithm (Ramdas & Wang 2025, Ch. 4, Section 4.1):
//!   1. Sort e-values descending: e_{(1)} >= e_{(2)} >= ... >= e_{(m)}
//!   2. Step-down: reject while e_{(k)} >= (m - k + 1) / alpha
//!      i.e., reject the k-th largest if it exceeds the threshold for
//!      the remaining (m - k + 1) hypotheses.
//!   3. Stop at the first non-rejection.
//!
//! In log space:
//!   Reject while log_e_{(k)} >= ln(m - k + 1) - ln(alpha)

use rayon::prelude::*;

/// Result of e-Holm multiple testing.
pub struct HolmResult {
    /// Per-hypothesis rejection flags (original order).
    pub rejected: Vec<bool>,
    /// Number of rejections.
    pub n_rejected: usize,
}

/// Apply e-Holm step-down procedure for FWER control.
///
/// # Arguments
/// * `log_e_values` - Log e-values (one per hypothesis)
/// * `alpha` - Target FWER level
pub fn e_holm(log_e_values: &[f64], alpha: f64) -> HolmResult {
    let m = log_e_values.len();
    if m == 0 {
        return HolmResult {
            rejected: vec![],
            n_rejected: 0,
        };
    }

    let log_inv_alpha = (1.0 / alpha).ln();

    // Sort descending by log_e, keeping original indices
    let mut indexed: Vec<(usize, f64)> = log_e_values.iter().copied().enumerate().collect();
    indexed.par_sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Step-down: reject while log_e_{(k)} >= ln(m - k + 1) + ln(1/alpha)
    // k is 1-indexed
    let mut k_star = 0usize;
    for (rank_0, &(_orig_idx, log_e)) in indexed.iter().enumerate() {
        let k = rank_0 + 1;
        let remaining = (m - k + 1) as f64;
        let log_threshold_k = remaining.ln() + log_inv_alpha;
        if log_e >= log_threshold_k {
            k_star = k;
        } else {
            break; // Step-down: stop at first non-rejection
        }
    }

    let mut rejected = vec![false; m];
    for &(orig_idx, _) in &indexed[..k_star] {
        rejected[orig_idx] = true;
    }

    HolmResult {
        rejected,
        n_rejected: k_star,
    }
}