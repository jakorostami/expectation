use crate::martingale::{MixtureSuperMartingale, TwoSidedNormalMixture};

const TOLERANCE: f64 = 1e-13;

#[test]
fn test_best_rho_matches_python() {
    let rho = TwoSidedNormalMixture::best_rho(1.0, 0.05).unwrap();
    let log_20 = (1.0_f64 / 0.05).ln();
    let expected = 1.0 / (2.0 * log_20 + (1.0 + 2.0 * log_20).ln());
    assert!(
        (rho - expected).abs() < TOLERANCE,
        "rho={rho}, expected={expected}"
    );
}

#[test]
fn test_log_super_mg_matches_python() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let result = m.log_super_mg(0.5, 1.0);

    let rho = m.rho();
    let v_plus_rho = 1.0 + rho;
    let expected = 0.5 * (rho / v_plus_rho).ln() + 0.25 / (2.0 * v_plus_rho);
    assert!(
        (result - expected).abs() < TOLERANCE,
        "log_super_mg={result}, expected={expected}"
    );
}

#[test]
fn test_symmetry() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let pos = m.log_super_mg(0.7, 2.0);
    let neg = m.log_super_mg(-0.7, 2.0);
    assert!(
        (pos - neg).abs() < TOLERANCE,
        "Symmetry failed: {pos} != {neg}"
    );
}

#[test]
fn test_monotone_in_s_squared() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let v = 5.0;
    let val_small = m.log_super_mg(0.1, v);
    let val_large = m.log_super_mg(1.0, v);
    assert!(
        val_large > val_small,
        "Should be monotone in s^2: {val_large} <= {val_small}"
    );
}

#[test]
fn test_at_s_zero() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let val = m.log_super_mg(0.0, 1.0);
    assert!(val < 0.0, "log M(0, v) should be negative, got {val}");
}

#[test]
fn test_bound_positive() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let b = m.bound(1.0, (1.0_f64 / 0.05).ln());
    assert!(b > 0.0, "Bound should be positive, got {b}");
}

#[test]
fn test_invalid_v_opt() {
    assert!(TwoSidedNormalMixture::new(0.0, 0.05).is_err());
    assert!(TwoSidedNormalMixture::new(-1.0, 0.05).is_err());
}

#[test]
fn test_invalid_alpha() {
    assert!(TwoSidedNormalMixture::new(1.0, 0.0).is_err());
    assert!(TwoSidedNormalMixture::new(1.0, 1.0).is_err());
    assert!(TwoSidedNormalMixture::new(1.0, -0.1).is_err());
}

#[test]
fn test_golden_sequence_1000_steps() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let rho = m.rho();

    let mut data_sum = 0.0_f64;
    let mut count = 0u32;

    for t in 1..=1000 {
        let x = (t as f64).sin();
        data_sum += x;
        count += 1;

        let s = data_sum;
        let v = count as f64;

        let result = m.log_super_mg(s, v);
        let v_plus_rho = v + rho;
        let expected = 0.5 * (rho / v_plus_rho).ln() + (s * s) / (2.0 * v_plus_rho);

        assert!(
            (result - expected).abs() < TOLERANCE,
            "Step {t}: result={result}, expected={expected}"
        );
    }
}
