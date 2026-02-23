// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Tests for the empirically adaptive (ONS) combiner.
//!
//! Reference: Waudby-Smith & Ramdas (2024), Theorem 7.22 in Ramdas & Wang (2025).

use crate::martingale::TwoSidedNormalMixture;
use crate::par_seqtest::state::ParTestState;
use crate::par_seqtest::update::{
    step_parallel, AlternativeDirection, CombinerType, VarianceConfig,
};

#[test]
fn test_adaptive_first_lambda_zero() {
    // First step lambda should be 0 (no previous data to estimate from).
    // With lambda=0, increment=1.0, so log_e_process=0.0.
    let n = 1;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    step_parallel(
        &mut state,
        &[1.0],
        &null_values,
        log_threshold,
        &variance,
        CombinerType::EmpiricallyAdaptive {
            gamma: 0.5,
            epsilon: 1e-6,
        },
        AlternativeDirection::TwoSided,
        1,
        &m,
    )
    .unwrap();

    // lambda for step 1 was 0 (S1=0, S2=0 before step 1)
    // So increment = (1-0) + 0*E_1 = 1.0
    // log_e_process = log(1.0) = 0.0
    assert!(
        state.log_e_process[0].abs() < 1e-15,
        "First step should have log_e_process=0 with lambda=0, got {}",
        state.log_e_process[0]
    );
}

#[test]
fn test_adaptive_lambda_bounded_by_gamma() {
    // Lambda should never exceed gamma.
    let n = 1;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);
    let gamma = 0.3;

    for t in 1..=50 {
        step_parallel(
            &mut state,
            &[2.0],
            &null_values,
            log_threshold,
            &variance,
            CombinerType::EmpiricallyAdaptive {
                gamma,
                epsilon: 1e-6,
            },
            AlternativeDirection::TwoSided,
            t,
            &m,
        )
        .unwrap();
    }

    assert!(
        state.lambda[0] <= gamma + 1e-15,
        "Lambda {} should not exceed gamma {}",
        state.lambda[0],
        gamma
    );
}

#[test]
fn test_adaptive_lambda_non_negative() {
    // Lambda should always be >= 0.
    let n = 1;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    // Use data that might make S1 negative (data consistent with null)
    let observations = [0.1, -0.2, 0.05, -0.15, 0.0, -0.1, 0.02, -0.08, 0.01, -0.05];

    for (t, &x) in observations.iter().enumerate() {
        step_parallel(
            &mut state,
            &[x],
            &null_values,
            log_threshold,
            &variance,
            CombinerType::EmpiricallyAdaptive {
                gamma: 0.5,
                epsilon: 1e-6,
            },
            AlternativeDirection::TwoSided,
            (t + 1) as u64,
            &m,
        )
        .unwrap();

        assert!(
            state.lambda[0] >= -1e-15,
            "Lambda should be non-negative at step {}, got {}",
            t + 1,
            state.lambda[0]
        );
    }
}

#[test]
fn test_adaptive_manual_two_steps() {
    // Manually verify the ONS combiner for two steps.
    let n = 1;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let rho = m.rho();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);
    let gamma = 0.5;
    let epsilon = 1e-6;

    // Step 1: lambda=0 (S1=0, S2=0)
    let x1 = 1.0_f64;
    step_parallel(
        &mut state,
        &[x1],
        &null_values,
        log_threshold,
        &variance,
        CombinerType::EmpiricallyAdaptive { gamma, epsilon },
        AlternativeDirection::TwoSided,
        1,
        &m,
    )
    .unwrap();

    // After step 1: log_e_process = 0.0 (lambda was 0)
    assert!(state.log_e_process[0].abs() < 1e-15);

    // Compute E_1 manually
    let s1_val = x1;
    let v1 = 1.0;
    let log_e_cum_1 = 0.5 * (rho / (v1 + rho)).ln() + (s1_val * s1_val) / (2.0 * (v1 + rho));
    let e1 = log_e_cum_1.exp(); // E_1 (sequential)

    // After step 1, S1 = E_1 - 1, S2 = (E_1 - 1)^2
    let s1_after_1 = e1 - 1.0;
    let s2_after_1 = (e1 - 1.0) * (e1 - 1.0);

    // Step 2: lambda = clamp(S1 / (S2 + epsilon), [0, gamma])
    let expected_lambda_2 = (s1_after_1 / (s2_after_1 + epsilon)).clamp(0.0, gamma);

    let x2 = 0.5_f64;
    step_parallel(
        &mut state,
        &[x2],
        &null_values,
        log_threshold,
        &variance,
        CombinerType::EmpiricallyAdaptive { gamma, epsilon },
        AlternativeDirection::TwoSided,
        2,
        &m,
    )
    .unwrap();

    // Compute E_2 manually
    let data_sum_2 = x1 + x2;
    let s2_val = data_sum_2;
    let v2 = 2.0;
    let log_e_cum_2 = 0.5 * (rho / (v2 + rho)).ln() + (s2_val * s2_val) / (2.0 * (v2 + rho));
    let log_e_seq_2 = log_e_cum_2 - log_e_cum_1;
    let e2 = log_e_seq_2.exp();

    // Expected e-process after step 2:
    // log_ep = 0 + log((1-lambda_2) + lambda_2 * E_2)
    let increment_2 = (1.0 - expected_lambda_2) + expected_lambda_2 * e2;
    let expected_log_ep = increment_2.ln();

    assert!(
        (state.log_e_process[0] - expected_log_ep).abs() < 1e-13,
        "Step 2 manual: expected={}, got={}, diff={}",
        expected_log_ep,
        state.log_e_process[0],
        (state.log_e_process[0] - expected_log_ep).abs()
    );
}

#[test]
fn test_adaptive_grows_under_signal() {
    // Under signal, adaptive combiner should eventually produce positive log_e_process.
    let n = 1;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    for t in 1..=100 {
        step_parallel(
            &mut state,
            &[1.0],
            &null_values,
            log_threshold,
            &variance,
            CombinerType::EmpiricallyAdaptive {
                gamma: 0.5,
                epsilon: 1e-6,
            },
            AlternativeDirection::TwoSided,
            t,
            &m,
        )
        .unwrap();
    }

    assert!(
        state.log_e_process[0] > 0.0,
        "Adaptive combiner should grow under signal, got {}",
        state.log_e_process[0]
    );
}

#[test]
fn test_adaptive_ons_stats_update() {
    // Verify that sum_e_minus_1 and sum_e_minus_1_sq accumulate correctly.
    let n = 1;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let rho = m.rho();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    let observations = [0.5, -0.2, 0.8];

    let mut expected_s1 = 0.0;
    let mut expected_s2 = 0.0;
    let mut data_sum = 0.0;
    let mut prev_log_e_cum = 0.0;

    for (t, &x) in observations.iter().enumerate() {
        step_parallel(
            &mut state,
            &[x],
            &null_values,
            log_threshold,
            &variance,
            CombinerType::EmpiricallyAdaptive {
                gamma: 0.5,
                epsilon: 1e-6,
            },
            AlternativeDirection::TwoSided,
            (t + 1) as u64,
            &m,
        )
        .unwrap();

        // Manually compute E_t
        data_sum += x;
        let s = data_sum;
        let v = (t + 1) as f64;
        let log_e_cum = 0.5 * (rho / (v + rho)).ln() + (s * s) / (2.0 * (v + rho));
        let log_e_seq = log_e_cum - prev_log_e_cum;
        let e_t = log_e_seq.exp();
        prev_log_e_cum = log_e_cum;

        expected_s1 += e_t - 1.0;
        expected_s2 += (e_t - 1.0) * (e_t - 1.0);
    }

    assert!(
        (state.sum_e_minus_1[0] - expected_s1).abs() < 1e-13,
        "S1: expected={}, got={}",
        expected_s1,
        state.sum_e_minus_1[0]
    );
    assert!(
        (state.sum_e_minus_1_sq[0] - expected_s2).abs() < 1e-13,
        "S2: expected={}, got={}",
        expected_s2,
        state.sum_e_minus_1_sq[0]
    );
}
