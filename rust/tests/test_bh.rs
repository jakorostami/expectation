use crate::multiple_testing::bh::e_bh;
use crate::multiple_testing::bonferroni::e_bonferroni;

#[test]
fn test_all_null_no_rejections() {
    let log_e_values = vec![0.0; 100];
    let result = e_bh(&log_e_values, 0.05);
    assert_eq!(result.n_rejected, 0);
}

#[test]
fn test_empty() {
    let result = e_bh(&[], 0.05);
    assert_eq!(result.n_rejected, 0);
}

#[test]
fn test_single_strong_signal() {
    let m = 100;
    let alpha = 0.05;
    let mut log_e_values = vec![0.0; m];
    log_e_values[50] = 10.0;
    let result = e_bh(&log_e_values, alpha);
    assert_eq!(result.n_rejected, 1);
    assert!(result.rejected[50]);
}

#[test]
fn test_multiple_signals() {
    let m = 10;
    let alpha = 0.05;
    let log_m = (m as f64).ln();
    let log_inv_alpha = (1.0_f64 / alpha).ln();

    let mut log_e_values = vec![0.0; m];
    log_e_values[0] = 6.0;
    log_e_values[1] = 5.0;
    log_e_values[2] = 4.5;

    let result = e_bh(&log_e_values, alpha);
    assert_eq!(result.n_rejected, 3);
    assert!(result.rejected[0]);
    assert!(result.rejected[1]);
    assert!(result.rejected[2]);

    let t1 = log_m + log_inv_alpha;
    let t2 = log_m - (2.0_f64).ln() + log_inv_alpha;
    let t3 = log_m - (3.0_f64).ln() + log_inv_alpha;
    assert!(6.0 >= t1, "6.0 should >= {t1}");
    assert!(5.0 >= t2, "5.0 should >= {t2}");
    assert!(4.5 >= t3, "4.5 should >= {t3}");
}

#[test]
fn test_bh_more_liberal_than_bonferroni() {
    let m = 100;
    let alpha = 0.05;
    let mut log_e_values = vec![0.0; m];

    for i in 0..10 {
        log_e_values[i] = 6.0 + i as f64 * 0.5;
    }

    let bh_result = e_bh(&log_e_values, alpha);
    let bonf_result = e_bonferroni(&log_e_values, alpha);

    assert!(
        bh_result.n_rejected >= bonf_result.n_rejected,
        "e-BH ({}) should reject >= e-Bonferroni ({})",
        bh_result.n_rejected,
        bonf_result.n_rejected
    );
}

/// Regression test for the premature-break bug.
///
/// Crafts a scenario where rank k=1 FAILS its threshold but rank k=2
/// PASSES its (lower) threshold. Before the fix, the `break` at k=1
/// would cause the k=2 rejection to be missed.
///
/// With m=10, alpha=0.05:
///   threshold(k=1) = ln(10) - ln(1) - ln(0.05) = 2.303 + 2.996 = 5.299
///   threshold(k=2) = ln(10) - ln(2) - ln(0.05) = 2.303 - 0.693 + 2.996 = 4.606
///
/// We set:
///   e_{(1)} = 5.0  (fails threshold 5.299)
///   e_{(2)} = 4.7  (passes threshold 4.606)
///
/// After fix: k* = 2 (reject both), before fix: k* = 0 (reject none).
#[test]
fn test_bh_no_premature_break() {
    let m = 10;
    let alpha = 0.05;

    let mut log_e_values = vec![0.0; m];
    log_e_values[0] = 5.0;  // rank 1: fails its threshold (5.299)
    log_e_values[1] = 4.7;  // rank 2: passes its threshold (4.606)

    let result = e_bh(&log_e_values, alpha);

    // Both should be rejected: k* = 2
    assert_eq!(
        result.n_rejected, 2,
        "e-BH should reject 2 hypotheses (regression: premature break would give 0)"
    );
    assert!(result.rejected[0]);
    assert!(result.rejected[1]);
}
