use crate::martingale::TwoSidedNormalMixture;
use crate::par_seqtest::state::ParTestState;
use crate::par_seqtest::update::{step_parallel, CombinerType, VarianceConfig};

#[test]
fn test_step_parallel_basic() {
    let n = 100;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0; n];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    let obs = vec![0.5; n];
    step_parallel(
        &mut state,
        &obs,
        &null_values,
        log_threshold,
        &variance,
        CombinerType::AllIn,
        &m,
    )
    .unwrap();

    assert!(state.count.iter().all(|&c| c == 1));
    assert!(state.data_sum.iter().all(|&s| (s - 0.5).abs() < 1e-15));

    let first = state.log_e_process[0];
    assert!(state.log_e_process.iter().all(|&v| (v - first).abs() < 1e-15));
}

#[test]
fn test_dimension_mismatch() {
    let mut state = ParTestState::zeros(100);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0; 100];
    let obs = vec![0.5; 50];

    let result = step_parallel(
        &mut state,
        &obs,
        &null_values,
        3.0,
        &VarianceConfig::KnownHomogeneous(1.0),
        CombinerType::AllIn,
        &m,
    );
    assert!(result.is_err());
}

#[test]
fn test_ville_rejection_under_signal() {
    let n = 10;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0; n];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    for _ in 0..20 {
        let obs = vec![3.0; n];
        step_parallel(
            &mut state,
            &obs,
            &null_values,
            log_threshold,
            &variance,
            CombinerType::AllIn,
            &m,
        )
        .unwrap();
    }

    assert!(
        state.rejected.iter().all(|&r| r),
        "All tests should reject under strong signal"
    );
}

#[test]
fn test_single_test_matches_manual() {
    let mut state = ParTestState::zeros(1);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let rho = m.rho();
    let null_values = vec![0.0];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHomogeneous(1.0);

    let observations = [0.3, -0.1, 0.7, 0.2, -0.5];

    let mut data_sum = 0.0_f64;
    for (t, &x) in observations.iter().enumerate() {
        data_sum += x;
        let s = data_sum;
        let v = (t + 1) as f64;
        let v_plus_rho = v + rho;
        let expected_log_e = 0.5 * (rho / v_plus_rho).ln() + (s * s) / (2.0 * v_plus_rho);

        step_parallel(
            &mut state,
            &[x],
            &null_values,
            log_threshold,
            &variance,
            CombinerType::AllIn,
            &m,
        )
        .unwrap();

        assert!(
            (state.log_e_process[0] - expected_log_e).abs() < 1e-13,
            "Step {}: got {}, expected {}",
            t + 1,
            state.log_e_process[0],
            expected_log_e
        );
    }
}

#[test]
fn test_heterogeneous_variance() {
    let n = 3;
    let mut state = ParTestState::zeros(n);
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let null_values = vec![0.0; n];
    let log_threshold = (1.0_f64 / 0.05).ln();
    let variance = VarianceConfig::KnownHeterogeneous(vec![0.5, 1.0, 2.0]);

    let obs = vec![1.0; n];
    step_parallel(
        &mut state,
        &obs,
        &null_values,
        log_threshold,
        &variance,
        CombinerType::AllIn,
        &m,
    )
    .unwrap();

    assert!(
        (state.log_e_process[0] - state.log_e_process[1]).abs() > 1e-10,
        "Different variances should produce different e-values"
    );
}
