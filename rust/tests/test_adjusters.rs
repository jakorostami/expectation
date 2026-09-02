// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

use crate::adjusters::AdjusterType;

// ── Boundary tests ──────────────────────────────────────────────────────

#[test]
fn test_lookback_boundary_at_one() {
    assert_eq!(AdjusterType::Lookback.adjust(1.0), 0.0);
}

#[test]
fn test_lookback_boundary_below_one() {
    assert_eq!(AdjusterType::Lookback.adjust(0.5), 0.0);
}

#[test]
fn test_sqrt_boundary_at_one() {
    assert_eq!(AdjusterType::Sqrt.adjust(1.0), 0.0);
}

#[test]
fn test_log_lookback_boundary() {
    assert_eq!(AdjusterType::Lookback.log_adjust(0.0), f64::NEG_INFINITY);
    assert_eq!(AdjusterType::Lookback.log_adjust(-1.0), f64::NEG_INFINITY);
}

#[test]
fn test_log_sqrt_boundary() {
    assert_eq!(AdjusterType::Sqrt.log_adjust(0.0), f64::NEG_INFINITY);
    assert_eq!(AdjusterType::Sqrt.log_adjust(-5.0), f64::NEG_INFINITY);
}

// ── Known values ────────────────────────────────────────────────────────

#[test]
fn test_lookback_at_e() {
    // E = e, x = 1: A_1 = (e - 1 - 1)/1 = e - 2
    let result = AdjusterType::Lookback.adjust(std::f64::consts::E);
    let expected = std::f64::consts::E - 2.0;
    assert!((result - expected).abs() < 1e-14);
}

#[test]
fn test_sqrt_known_values() {
    assert!((AdjusterType::Sqrt.adjust(4.0) - 1.0).abs() < 1e-14);
    assert!((AdjusterType::Sqrt.adjust(9.0) - 2.0).abs() < 1e-14);
    assert!((AdjusterType::Sqrt.adjust(100.0) - 9.0).abs() < 1e-13);
}

// ── Taylor accuracy ─────────────────────────────────────────────────────

#[test]
fn test_taylor_accuracy_at_1e_6() {
    let log_e = 1e-6_f64;
    let e = log_e.exp();
    let result = AdjusterType::Lookback.adjust(e);
    // Taylor: 0.5 + x/6 + x^2/24 + x^3/120
    let x = log_e;
    let reference = 0.5 + x / 6.0 + x * x / 24.0 + x * x * x / 120.0;
    assert!(
        (result - reference).abs() < 1e-13,
        "At log_e=1e-6: result={}, reference={}",
        result,
        reference
    );
}

// ── Lookback limit ──────────────────────────────────────────────────────

#[test]
fn test_lookback_limit_approaches_half() {
    // A_1(1+eps) → 1/2
    let e = 1.0 + 1e-12;
    let result = AdjusterType::Lookback.adjust(e);
    assert!(
        (result - 0.5).abs() < 1e-4,
        "Lookback limit: {}, expected ~0.5",
        result
    );
}

// ── Monotonicity ────────────────────────────────────────────────────────

#[test]
fn test_lookback_strictly_increasing() {
    let mut prev = 0.0_f64;
    for i in 1..500 {
        let log_e = i as f64 * 0.04;
        let val = AdjusterType::Lookback.adjust(log_e.exp());
        assert!(val >= prev, "Not monotone at log_e={}", log_e);
        prev = val;
    }
}

#[test]
fn test_sqrt_strictly_increasing() {
    let mut prev = 0.0_f64;
    for i in 1..500 {
        let log_e = i as f64 * 0.04;
        let val = AdjusterType::Sqrt.adjust(log_e.exp());
        assert!(val >= prev, "Not monotone at log_e={}", log_e);
        prev = val;
    }
}

// ── Log-space consistency ───────────────────────────────────────────────

#[test]
fn test_log_vs_natural_consistency() {
    for &log_e in &[0.001_f64, 0.01, 0.1, 1.0, 3.0, 10.0] {
        for adj in &[AdjusterType::Lookback, AdjusterType::Sqrt] {
            let natural = adj.adjust(log_e.exp());
            let from_log = adj.log_adjust(log_e).exp();
            assert!(
                (natural - from_log).abs() < 1e-12,
                "{:?} inconsistency at log_e={}: natural={}, from_log={}",
                adj,
                log_e,
                natural,
                from_log
            );
        }
    }
}
