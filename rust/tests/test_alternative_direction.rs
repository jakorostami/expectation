// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Tests for alternative hypothesis direction handling.
//!
//! Reference: Ramdas & Wang (2025), Section 2.1.

use crate::martingale::{OneSidedNormalMixture, TwoSidedNormalMixture};
use crate::par_seqtest::state::ParTestState;
use crate::par_seqtest::update::{
    step_parallel, AlternativeDirection, CombinerType, VarianceConfig,
};

#[test]
fn test_less_negates_sign() {
    // Under LESS alternative with positive data, s is negated.
    // With data_sum > 0 and null=0, s = -(data_sum) < 0 for LESS.
    // For one-sided normal, negative s produces low e-values.
    let n = 1;
    let mut state_greater = ParTestState::zeros(n);
    let mut state_less = ParTestState::zeros(n);
    let m = OneSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    // Positive data favors GREATER, not LESS
    let obs = vec![1.0];
    step_parallel(
        &mut state_greater,
        &obs,
        &null_values,
        log_threshold,
        &variance,
        CombinerType::AllIn,
        AlternativeDirection::Greater,
        1,
        &m,
    )
    .unwrap();

    step_parallel(
        &mut state_less,
        &obs,
        &null_values,
        log_threshold,
        &variance,
        CombinerType::AllIn,
        AlternativeDirection::Less,
        1,
        &m,
    )
    .unwrap();

    // GREATER should have higher e-value with positive data
    assert!(
        state_greater.log_e_process[0] > state_less.log_e_process[0],
        "GREATER ({}) should have higher log_e_process than LESS ({}) with positive data",
        state_greater.log_e_process[0],
        state_less.log_e_process[0]
    );
}

#[test]
fn test_less_detects_negative_signal() {
    // Under LESS alternative with strongly negative data, tests should reject.
    let n = 5;
    let mut state = ParTestState::zeros(n);
    let m = OneSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0; n];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    for t in 1..=30 {
        let obs = vec![-3.0; n]; // Strong negative signal
        step_parallel(
            &mut state,
            &obs,
            &null_values,
            log_threshold,
            &variance,
            CombinerType::AllIn,
            AlternativeDirection::Less,
            t,
            &m,
        )
        .unwrap();
    }

    assert!(
        state.rejected.iter().all(|&r| r),
        "All tests should reject under strong negative signal with LESS alternative"
    );
}

#[test]
fn test_two_sided_symmetric() {
    // Two-sided normal is symmetric in s: data_sum and -data_sum give same log_e_process.
    let n = 1;
    let mut state_pos = ParTestState::zeros(n);
    let mut state_neg = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    step_parallel(
        &mut state_pos,
        &[1.5],
        &null_values,
        log_threshold,
        &variance,
        CombinerType::AllIn,
        AlternativeDirection::TwoSided,
        1,
        &m,
    )
    .unwrap();

    step_parallel(
        &mut state_neg,
        &[-1.5],
        &null_values,
        log_threshold,
        &variance,
        CombinerType::AllIn,
        AlternativeDirection::TwoSided,
        1,
        &m,
    )
    .unwrap();

    assert!(
        (state_pos.log_e_process[0] - state_neg.log_e_process[0]).abs() < 1e-14,
        "Two-sided should be symmetric: pos={}, neg={}",
        state_pos.log_e_process[0],
        state_neg.log_e_process[0]
    );
}

#[test]
fn test_greater_direction_with_two_sided_martingale() {
    // GREATER direction should still work with TwoSidedNormalMixture
    // (just applies sign=1.0, same as TwoSided).
    let n = 1;
    let mut state_greater = ParTestState::zeros(n);
    let mut state_two_sided = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    let obs = [0.5];
    step_parallel(
        &mut state_greater,
        &obs,
        &null_values,
        log_threshold,
        &variance,
        CombinerType::AllIn,
        AlternativeDirection::Greater,
        1,
        &m,
    )
    .unwrap();

    step_parallel(
        &mut state_two_sided,
        &obs,
        &null_values,
        log_threshold,
        &variance,
        CombinerType::AllIn,
        AlternativeDirection::TwoSided,
        1,
        &m,
    )
    .unwrap();

    // With TwoSidedNormalMixture, GREATER and TwoSided should give same result
    // (both use sign=1.0, and TwoSided uses s² so sign doesn't matter)
    assert!(
        (state_greater.log_e_process[0] - state_two_sided.log_e_process[0]).abs() < 1e-15,
        "GREATER and TwoSided should be identical with TwoSidedNormalMixture"
    );
}
