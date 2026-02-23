// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Tests for the conservative combiner (fixed lambda).
//!
//! Reference: Ramdas & Wang (2025), Definition 7.21.

use crate::martingale::TwoSidedNormalMixture;
use crate::par_seqtest::state::ParTestState;
use crate::par_seqtest::update::{
    step_parallel, AlternativeDirection, CombinerType, VarianceConfig,
};

#[test]
fn test_conservative_dampens_growth() {
    // Conservative combiner with lambda=0.5 should produce lower e-process
    // than ALL_IN under signal.
    let n = 1;
    let mut state_all_in = ParTestState::zeros(n);
    let mut state_conservative = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    for t in 1..=20 {
        let obs = vec![1.0];
        step_parallel(
            &mut state_all_in,
            &obs,
            &null_values,
            log_threshold,
            &variance,
            CombinerType::AllIn,
            AlternativeDirection::TwoSided,
            t,
            &m,
        )
        .unwrap();

        step_parallel(
            &mut state_conservative,
            &obs,
            &null_values,
            log_threshold,
            &variance,
            CombinerType::Conservative { lambda: 0.5 },
            AlternativeDirection::TwoSided,
            t,
            &m,
        )
        .unwrap();
    }

    assert!(
        state_all_in.log_e_process[0] > state_conservative.log_e_process[0],
        "ALL_IN ({}) should grow faster than conservative ({})",
        state_all_in.log_e_process[0],
        state_conservative.log_e_process[0]
    );
}

#[test]
fn test_conservative_manual_arithmetic() {
    // Verify conservative combiner formula manually for a single step.
    // E-process: log M = log((1-λ) + λ·E_t) where E_t = exp(log_e_seq)
    let n = 1;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let rho = m.rho();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);
    let lambda = 0.5_f64;

    let x = 0.7;
    step_parallel(
        &mut state,
        &[x],
        &null_values,
        log_threshold,
        &variance,
        CombinerType::Conservative { lambda },
        AlternativeDirection::TwoSided,
        1,
        &m,
    )
    .unwrap();

    // Manual computation:
    // s = 0.7, v = 1.0 * 1.0 = 1.0
    let s = x;
    let v = 1.0;
    let v_plus_rho = v + rho;
    let log_e_cum = 0.5 * (rho / v_plus_rho).ln() + (s * s) / (2.0 * v_plus_rho);
    let e_t = log_e_cum.exp(); // For first step: log_e_seq = log_e_cum - 0 = log_e_cum
    let increment = (1.0 - lambda) + lambda * e_t;
    let expected_log_ep = increment.ln();

    assert!(
        (state.log_e_process[0] - expected_log_ep).abs() < 1e-13,
        "Manual: {}, Rust: {}, diff: {}",
        expected_log_ep,
        state.log_e_process[0],
        (state.log_e_process[0] - expected_log_ep).abs()
    );
}

#[test]
fn test_conservative_lambda_stored() {
    // Lambda should be stored correctly for conservative combiner.
    let n = 3;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0; n];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);
    let lambda = 0.3;

    step_parallel(
        &mut state,
        &[1.0; 3],
        &null_values,
        log_threshold,
        &variance,
        CombinerType::Conservative { lambda },
        AlternativeDirection::TwoSided,
        1,
        &m,
    )
    .unwrap();

    assert!(state.lambda.iter().all(|&l| (l - lambda).abs() < 1e-15));
}

#[test]
fn test_conservative_still_rejects_strong_signal() {
    // Even conservative combiner should reject under very strong signal.
    let n = 5;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0; n];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    for t in 1..=50 {
        step_parallel(
            &mut state,
            &[5.0; 5],
            &null_values,
            log_threshold,
            &variance,
            CombinerType::Conservative { lambda: 0.3 },
            AlternativeDirection::TwoSided,
            t,
            &m,
        )
        .unwrap();
    }

    assert!(
        state.rejected.iter().all(|&r| r),
        "Conservative combiner should still reject under very strong signal"
    );
}
