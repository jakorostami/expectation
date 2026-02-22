use crate::multiple_testing::bonferroni::e_bonferroni;

#[test]
fn test_all_null_no_rejections() {
    let log_e_values = vec![0.0; 100];
    let result = e_bonferroni(&log_e_values, 0.05);
    assert_eq!(result.n_rejected, 0);
}

#[test]
fn test_strong_signal_rejects() {
    let mut log_e_values = vec![0.0; 100];
    log_e_values[42] = 10.0;
    let result = e_bonferroni(&log_e_values, 0.05);
    assert_eq!(result.n_rejected, 1);
    assert!(result.rejected[42]);
}

#[test]
fn test_threshold_exact() {
    let m = 10;
    let alpha = 0.05;
    let threshold = (m as f64 / alpha).ln();

    let mut log_e_values = vec![0.0; m];
    log_e_values[0] = threshold;
    let result = e_bonferroni(&log_e_values, alpha);
    assert_eq!(result.n_rejected, 1);

    log_e_values[0] = threshold - 1e-10;
    let result = e_bonferroni(&log_e_values, alpha);
    assert_eq!(result.n_rejected, 0);
}
