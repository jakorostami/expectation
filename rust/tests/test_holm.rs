use crate::multiple_testing::bonferroni::e_bonferroni;
use crate::multiple_testing::holm::e_holm;

#[test]
fn test_all_null_no_rejections() {
    let log_e_values = vec![0.0; 100];
    let result = e_holm(&log_e_values, 0.05);
    assert_eq!(result.n_rejected, 0);
}

#[test]
fn test_empty() {
    let result = e_holm(&[], 0.05);
    assert_eq!(result.n_rejected, 0);
}

#[test]
fn test_single_strong_signal() {
    let m = 100;
    let alpha = 0.05;
    let mut log_e_values = vec![0.0; m];
    log_e_values[50] = 10.0;
    let result = e_holm(&log_e_values, alpha);
    assert_eq!(result.n_rejected, 1);
    assert!(result.rejected[50]);
}

#[test]
fn test_holm_tighter_than_bonferroni() {
    let m = 100;
    let alpha = 0.05;
    let mut log_e_values = vec![0.0; m];

    for i in 0..5 {
        log_e_values[i] = 8.0 + i as f64;
    }

    let holm_result = e_holm(&log_e_values, alpha);
    let bonf_result = e_bonferroni(&log_e_values, alpha);

    assert!(
        holm_result.n_rejected >= bonf_result.n_rejected,
        "e-Holm ({}) should reject >= e-Bonferroni ({})",
        holm_result.n_rejected,
        bonf_result.n_rejected
    );
}

#[test]
fn test_step_down_stops_correctly() {
    let m = 5;
    let alpha = 0.05;
    let log_inv_alpha = (1.0_f64 / alpha).ln();

    let t1 = (5.0_f64).ln() + log_inv_alpha;
    let t2 = (4.0_f64).ln() + log_inv_alpha;

    let mut log_e_values = vec![0.0; m];
    log_e_values[0] = t1 + 0.1;
    log_e_values[1] = t2 - 0.1;

    let result = e_holm(&log_e_values, alpha);
    assert_eq!(result.n_rejected, 1);
    assert!(result.rejected[0]);
}
