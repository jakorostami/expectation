// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Tests for OneSidedNormalMixture.
//!
//! Golden values cross-validated against Python `OneSidedNormalMixture.log_superMG`.
//!
//! Reference:
//!   Howard, Ramdas, McAuliffe, Sekhon (2022), Section 3.

use crate::martingale::{MixtureSuperMartingale, OneSidedNormalMixture, TwoSidedNormalMixture};
use crate::math::log_ndtr;

const TOLERANCE: f64 = 1e-13;

#[test]
fn test_best_rho_uses_double_alpha() {
    // OneSidedNormalMixture::best_rho(v, alpha) == TwoSidedNormalMixture::best_rho(v, 2*alpha)
    let v = 1.0;
    let alpha = 0.05;
    let one_sided = OneSidedNormalMixture::best_rho(v, alpha).unwrap();
    let two_sided = TwoSidedNormalMixture::best_rho(v, 2.0 * alpha).unwrap();
    assert!(
        (one_sided - two_sided).abs() < TOLERANCE,
        "one_sided_rho={one_sided}, two_sided_rho(2α)={two_sided}"
    );
}

#[test]
fn test_log_super_mg_formula() {
    // Verify against manual computation of the formula:
    // log M(s, v) = 0.5 * ln(4ρ/(v+ρ)) + s²/(2(v+ρ)) + ln(Φ(s/√(v+ρ)))
    let m = OneSidedNormalMixture::new(1.0, 0.05).unwrap();
    let rho = m.rho();

    let s = 1.5;
    let v = 3.0;
    let v_plus_rho = v + rho;

    let expected = 0.5 * (4.0 * rho / v_plus_rho).ln()
        + (s * s) / (2.0 * v_plus_rho)
        + log_ndtr(s / v_plus_rho.sqrt());

    let result = m.log_super_mg(s, v);
    assert!(
        (result - expected).abs() < TOLERANCE,
        "result={result}, expected={expected}"
    );
}

#[test]
fn test_not_symmetric() {
    // Unlike two-sided, one-sided is NOT symmetric
    let m = OneSidedNormalMixture::new(1.0, 0.05).unwrap();
    let pos = m.log_super_mg(1.0, 2.0);
    let neg = m.log_super_mg(-1.0, 2.0);
    assert!(
        (pos - neg).abs() > 0.1,
        "One-sided should be asymmetric: pos={pos}, neg={neg}"
    );
    // Positive direction should give higher log M
    assert!(
        pos > neg,
        "Positive s should give higher log M: pos={pos}, neg={neg}"
    );
}

#[test]
fn test_positive_direction_larger_than_two_sided() {
    // For s > 0, one-sided log M should be larger than two-sided
    // (because it uses 2*alpha for rho and adds the Φ term)
    let one = OneSidedNormalMixture::new(1.0, 0.05).unwrap();
    let two = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();

    let s = 3.0;
    let v = 5.0;
    let one_val = one.log_super_mg(s, v);
    let two_val = two.log_super_mg(s, v);
    assert!(
        one_val > two_val,
        "One-sided should be more powerful in positive direction: {one_val} <= {two_val}"
    );
}

#[test]
fn test_at_s_zero() {
    let m = OneSidedNormalMixture::new(1.0, 0.05).unwrap();
    let val = m.log_super_mg(0.0, 1.0);
    // At s=0, Φ(0) = 0.5, so:
    // log M = 0.5*ln(4ρ/(v+ρ)) + 0 + ln(0.5)
    //       = 0.5*ln(ρ/(v+ρ)) + 0.5*ln(4) + ln(0.5)
    //       = 0.5*ln(ρ/(v+ρ)) + ln(2) + ln(0.5)
    //       = 0.5*ln(ρ/(v+ρ))   (since ln(2)+ln(0.5)=0)
    // This equals the two-sided value at s=0!
    let two = TwoSidedNormalMixture::from_rho(m.rho()).unwrap();
    let two_val = two.log_super_mg(0.0, 1.0);
    assert!(
        (val - two_val).abs() < TOLERANCE,
        "At s=0, one-sided should equal two-sided: one={val}, two={two_val}"
    );
}

#[test]
fn test_bound_positive() {
    let m = OneSidedNormalMixture::new(1.0, 0.05).unwrap();
    let log_threshold = (1.0_f64 / 0.05).ln();
    let b = m.bound(1.0, log_threshold);
    assert!(b > 0.0, "Bound should be positive, got {b}");
    // Verify the bound actually crosses the threshold
    let val = m.log_super_mg(b, 1.0);
    assert!(
        (val - log_threshold).abs() < 1e-10,
        "Bound should satisfy log M(b,v) ≈ log_threshold: {val} vs {log_threshold}"
    );
}

#[test]
fn test_bound_consistency() {
    // For any v, bound(v, threshold) should satisfy log_super_mg(bound, v) ≈ threshold
    let m = OneSidedNormalMixture::new(1.0, 0.05).unwrap();
    let log_threshold = (1.0_f64 / 0.05).ln();
    for &v in &[1.0, 5.0, 10.0, 50.0] {
        let b = m.bound(v, log_threshold);
        let val = m.log_super_mg(b, v);
        assert!(
            (val - log_threshold).abs() < 1e-8,
            "v={v}: bound={b}, log_super_mg(bound,v)={val}, threshold={log_threshold}"
        );
    }
}

#[test]
fn test_invalid_parameters() {
    assert!(OneSidedNormalMixture::new(0.0, 0.05).is_err());
    assert!(OneSidedNormalMixture::new(-1.0, 0.05).is_err());
    assert!(OneSidedNormalMixture::new(1.0, 0.0).is_err());
    assert!(OneSidedNormalMixture::new(1.0, 0.6).is_err()); // > 0.5
}

#[test]
fn test_golden_sequence_100_steps() {
    // Deterministic sequence, manually verify formula
    let m = OneSidedNormalMixture::new(1.0, 0.05).unwrap();
    let rho = m.rho();

    let mut data_sum = 0.0_f64;
    let mut count = 0u32;

    for t in 1..=100 {
        let x = 0.1 + 0.01 * (t as f64).sin(); // slight positive drift
        data_sum += x;
        count += 1;

        let s = data_sum;
        let v = count as f64;
        let v_plus_rho = v + rho;

        let result = m.log_super_mg(s, v);
        let expected = 0.5 * (4.0 * rho / v_plus_rho).ln()
            + (s * s) / (2.0 * v_plus_rho)
            + log_ndtr(s / v_plus_rho.sqrt());

        assert!(
            (result - expected).abs() < TOLERANCE,
            "Step {t}: result={result}, expected={expected}"
        );
    }
}
