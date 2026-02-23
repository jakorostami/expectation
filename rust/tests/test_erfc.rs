// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Tests for the fdlibm erfc implementation and derived normal CDF.
//!
//! Reference values from Python's math.erfc (which uses the same fdlibm code).

use crate::math::erfc::{erfc, log_ndtr, ndtr};

// ── erfc tests ─────────────────────────────────────────────────────────

#[test]
fn test_erfc_at_zero() {
    assert!((erfc(0.0) - 1.0).abs() < 1e-15);
}

#[test]
fn test_erfc_symmetry() {
    // erfc(-x) = 2 - erfc(x)
    for &x in &[0.1, 0.3, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0] {
        let lhs = erfc(-x);
        let rhs = 2.0 - erfc(x);
        assert!(
            (lhs - rhs).abs() < 1e-14,
            "symmetry failed at x={x}: erfc(-x)={lhs}, 2-erfc(x)={rhs}"
        );
    }
}

#[test]
fn test_erfc_known_values() {
    // erfc(0.5) ≈ 0.4795001221869535
    assert!((erfc(0.5) - 0.4795001221869535).abs() < 1e-14);
    // erfc(1.0) ≈ 0.15729920705028513
    assert!((erfc(1.0) - 0.15729920705028513).abs() < 1e-14);
    // erfc(2.0) ≈ 0.004677734981047266
    assert!((erfc(2.0) - 0.004677734981047266).abs() < 1e-14);
    // erfc(3.0) ≈ 2.2090496998585438e-05 (relative tolerance)
    let e3 = erfc(3.0);
    let expected_3 = 2.209_049_699_858_544e-05;
    assert!(
        (e3 - expected_3).abs() / expected_3 < 1e-8,
        "erfc(3.0)={e3}, expected={expected_3}"
    );
}

#[test]
fn test_erfc_region_boundaries() {
    // Test continuity at region boundaries: 0.84375 and 1.25
    let eps = 1e-7;

    // Boundary at 0.84375
    let e1 = erfc(0.84375 - eps);
    let e2 = erfc(0.84375 + eps);
    assert!(
        (e1 - e2).abs() < 1e-4,
        "discontinuity at 0.84375: {e1} vs {e2}"
    );

    // Boundary at 1.25
    let e3 = erfc(1.25 - eps);
    let e4 = erfc(1.25 + eps);
    assert!(
        (e3 - e4).abs() < 1e-5,
        "discontinuity at 1.25: {e3} vs {e4}"
    );
}

#[test]
fn test_erfc_large_argument() {
    // erfc(5.0) ≈ 1.5374597944280349e-12 (relative tolerance)
    let e5 = erfc(5.0);
    let expected_5 = 1.537_459_794_428_035e-12;
    assert!(
        (e5 - expected_5).abs() / expected_5 < 1e-8,
        "erfc(5.0)={e5}, expected={expected_5}"
    );
    // erfc(20.0) should be extremely small but positive
    let val = erfc(20.0);
    assert!(val > 0.0);
    assert!(val < 1e-170);
}

#[test]
fn test_erfc_special_cases() {
    assert!(erfc(f64::NAN).is_nan());
    assert_eq!(erfc(f64::INFINITY), 0.0);
    assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
}

// ── ndtr tests ─────────────────────────────────────────────────────────

#[test]
fn test_ndtr_at_zero() {
    assert!((ndtr(0.0) - 0.5).abs() < 1e-15);
}

#[test]
fn test_ndtr_symmetry() {
    // Φ(-x) = 1 - Φ(x)
    for &x in &[0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0] {
        let lhs = ndtr(-x);
        let rhs = 1.0 - ndtr(x);
        assert!(
            (lhs - rhs).abs() < 1e-14,
            "ndtr symmetry failed at x={x}: Φ(-x)={lhs}, 1-Φ(x)={rhs}"
        );
    }
}

#[test]
fn test_ndtr_known_values() {
    // Φ(1) ≈ 0.8413447460685429
    assert!((ndtr(1.0) - 0.8413447460685429).abs() < 1e-14);
    // Φ(2) ≈ 0.9772498680518208
    assert!((ndtr(2.0) - 0.9772498680518208).abs() < 1e-14);
    // Φ(3) ≈ 0.9986501019683699
    assert!((ndtr(3.0) - 0.9986501019683699).abs() < 1e-13);
    // Φ(-1) ≈ 0.15865525393145707
    assert!((ndtr(-1.0) - 0.15865525393145707).abs() < 1e-14);
    // Φ(-2) ≈ 0.022750131948179216
    assert!((ndtr(-2.0) - 0.022750131948179216).abs() < 1e-14);
}

#[test]
fn test_ndtr_extreme_tails() {
    // Φ(6) ≈ 0.9999999990134123
    let n6 = ndtr(6.0);
    assert!(
        (n6 - 0.9999999990134123).abs() < 1e-13,
        "ndtr(6.0)={n6}"
    );
    // Φ(-6) ≈ 9.86587645e-10 (relative tolerance)
    let nm6 = ndtr(-6.0);
    let expected_nm6 = 9.865_876_450_377e-10;
    assert!(
        (nm6 - expected_nm6).abs() / expected_nm6 < 1e-10,
        "ndtr(-6.0)={nm6}, expected={expected_nm6}"
    );
}

#[test]
fn test_ndtr_special_cases() {
    assert!(ndtr(f64::NAN).is_nan());
    assert_eq!(ndtr(f64::INFINITY), 1.0);
    assert_eq!(ndtr(f64::NEG_INFINITY), 0.0);
}

// ── log_ndtr tests ─────────────────────────────────────────────────────

#[test]
fn test_log_ndtr_at_zero() {
    // ln(0.5) ≈ -0.6931471805599453
    assert!((log_ndtr(0.0) - (-0.6931471805599453)).abs() < 1e-14);
}

#[test]
fn test_log_ndtr_matches_direct() {
    // For moderate x, log_ndtr(x) ≈ ln(ndtr(x))
    for &x in &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
        let direct = ndtr(x).ln();
        let stable = log_ndtr(x);
        assert!(
            (direct - stable).abs() < 1e-12,
            "x={x}: direct={direct}, stable={stable}"
        );
    }
}

#[test]
fn test_log_ndtr_large_negative() {
    // For x = -30, ndtr(x) underflows but log_ndtr should be finite
    let val = log_ndtr(-30.0);
    assert!(val.is_finite());
    assert!(val < -400.0); // very negative log
}

#[test]
fn test_log_ndtr_large_positive() {
    // For x = 10, Φ(x) ≈ 1, so log Φ(x) ≈ 0
    let val = log_ndtr(10.0);
    // ln(1.0) = 0.0 exactly (Φ(10) rounds to 1.0 in double precision)
    assert!(val <= 0.0);
    assert!(val > -1e-10);
}
