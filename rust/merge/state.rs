// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Scalar state for the merged e-process.
//!
//! Tracks the running merged e-process from spatial merging of K
//! per-test sequential e-values + temporal accumulation.
//!
//! References:
//! - Vovk & Wang (2024), Corollary 1: admissible merging functions.
//! - Ramdas & Wang (2025), Definition 7.21: e-process from sequential e-values.

/// Scalar state for the merged e-process.
///
/// Allocated only when `global_merge` is configured (zero overhead otherwise).
#[derive(Debug, Clone)]
pub struct MergeState {
    /// Current step's log merged e-value (spatial merge output).
    pub log_merged_e_value: f64,
    /// Running log merged e-process (temporal accumulation).
    pub log_merged_e_process: f64,
    /// Max log merged e-process seen (for Ville's inequality).
    pub max_log_merged: f64,
    /// Whether the intersection null has been rejected.
    pub merged_rejected: bool,
    /// First rejection time step (0 = not stopped).
    pub merged_stopping_time: u64,
    /// Merged p-value: min(1, exp(-log_merged_e_process)).
    pub merged_p_value: f64,
    /// Current temporal betting fraction for the merged stream.
    pub merged_lambda: f64,
    /// ONS stat: cumulative sum of (merged_E_t - 1) for adaptive temporal combiner.
    pub sum_e_minus_1: f64,
    /// ONS stat: cumulative sum of (merged_E_t - 1)^2 for adaptive temporal combiner.
    pub sum_e_minus_1_sq: f64,
}

// `MergeState` is always constructed via `new()`; a Default impl would be dead
// code, so the lint is allowed rather than satisfied with an unused trait impl.
#[allow(clippy::new_without_default)]
impl MergeState {
    /// Create a new zeroed merge state.
    pub fn new() -> Self {
        Self {
            log_merged_e_value: 0.0,
            log_merged_e_process: 0.0,
            max_log_merged: 0.0,
            merged_rejected: false,
            merged_stopping_time: 0,
            merged_p_value: 1.0,
            merged_lambda: 0.0,
            sum_e_minus_1: 0.0,
            sum_e_minus_1_sq: 0.0,
        }
    }
}
