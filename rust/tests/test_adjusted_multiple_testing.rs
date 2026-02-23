// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

use crate::adjusters::AdjusterType;
use crate::multiple_testing::adjusted::{adjusted_e_bh, adjusted_e_bonferroni, adjusted_e_holm};
use crate::multiple_testing::bh::e_bh;

#[test]
fn test_all_null_zero_log_max() {
    // All max_log_m = 0 (i.e., max E = 1) → adjusters return 0 → 0 rejections
    let log_max = vec![0.0; 50];
    let result = adjusted_e_bh(&log_max, 0.05, AdjusterType::Lookback);
    assert_eq!(result.n_rejected, 0);
}

#[test]
fn test_all_null_neg_inf() {
    // Initial state before any data: max_log_m = NEG_INFINITY
    let log_max = vec![f64::NEG_INFINITY; 50];
    let result = adjusted_e_bh(&log_max, 0.05, AdjusterType::Sqrt);
    assert_eq!(result.n_rejected, 0);
}

#[test]
fn test_strong_signal_bh() {
    let mut log_max = vec![0.0; 10];
    log_max[0] = 100.0;
    log_max[1] = 80.0;
    let result = adjusted_e_bh(&log_max, 0.05, AdjusterType::Lookback);
    assert!(result.n_rejected >= 2);
    assert!(result.rejected[0]);
    assert!(result.rejected[1]);
}

#[test]
fn test_strong_signal_bonferroni() {
    let mut log_max = vec![0.0; 5];
    log_max[0] = 100.0;
    let result = adjusted_e_bonferroni(&log_max, 0.05, AdjusterType::Sqrt);
    assert!(result.n_rejected >= 1);
    assert!(result.rejected[0]);
}

#[test]
fn test_strong_signal_holm() {
    let mut log_max = vec![0.0; 5];
    log_max[0] = 100.0;
    let result = adjusted_e_holm(&log_max, 0.05, AdjusterType::Lookback);
    assert!(result.n_rejected >= 1);
}

#[test]
fn test_adjusted_leq_unadjusted() {
    // Adjusted procedures should reject <= unadjusted (adjusters shrink values)
    let log_max: Vec<f64> = (0..20)
        .map(|i| if i < 5 { 5.0 } else { 0.5 })
        .collect();

    let unadjusted = e_bh(&log_max, 0.05);
    let adj_lb = adjusted_e_bh(&log_max, 0.05, AdjusterType::Lookback);
    let adj_sq = adjusted_e_bh(&log_max, 0.05, AdjusterType::Sqrt);

    assert!(adj_lb.n_rejected <= unadjusted.n_rejected);
    assert!(adj_sq.n_rejected <= unadjusted.n_rejected);
}

#[test]
fn test_empty_input() {
    let result = adjusted_e_bh(&[], 0.05, AdjusterType::Lookback);
    assert_eq!(result.n_rejected, 0);
    assert!(result.rejected.is_empty());
}

#[test]
fn test_both_adjusters_agree_on_zero() {
    // Both should give 0 rejections for all-null
    let log_max = vec![0.0; 20];
    let lb = adjusted_e_bh(&log_max, 0.05, AdjusterType::Lookback);
    let sq = adjusted_e_bh(&log_max, 0.05, AdjusterType::Sqrt);
    assert_eq!(lb.n_rejected, 0);
    assert_eq!(sq.n_rejected, 0);
}
