use crate::martingale::TwoSidedNormalMixture;
use crate::par_seqtest::{CombinerType, VarianceConfig, ParallelSequentialTest};

#[test]
fn test_par_seqtest_creation() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let pst = ParallelSequentialTest::new(
        1000,
        vec![0.0; 1000],
        0.05,
        VarianceConfig::KnownHomogeneous(1.0),
        CombinerType::AllIn,
        m,
    )
    .unwrap();

    assert_eq!(pst.n_tests(), 1000);
    assert_eq!(pst.time_step(), 0);
    assert_eq!(pst.alpha(), 0.05);
}

#[test]
fn test_par_seqtest_step() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let mut pst = ParallelSequentialTest::new(
        100,
        vec![0.0; 100],
        0.05,
        VarianceConfig::KnownHomogeneous(1.0),
        CombinerType::AllIn,
        m,
    )
    .unwrap();

    let result = pst.step(&vec![0.5; 100]).unwrap();
    assert_eq!(result.time_step, 1);
    assert_eq!(result.n_tests, 100);
}

#[test]
fn test_par_seqtest_batch() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let mut pst = ParallelSequentialTest::new(
        50,
        vec![0.0; 50],
        0.05,
        VarianceConfig::KnownHomogeneous(1.0),
        CombinerType::AllIn,
        m,
    )
    .unwrap();

    let batch: Vec<Vec<f64>> = (0..10).map(|_| vec![0.3; 50]).collect();
    let results = pst.step_batch(&batch).unwrap();
    assert_eq!(results.len(), 10);
    assert_eq!(results[9].time_step, 10);
}

#[test]
fn test_null_values_mismatch() {
    let m = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let result = ParallelSequentialTest::new(
        100,
        vec![0.0; 50],
        0.05,
        VarianceConfig::KnownHomogeneous(1.0),
        CombinerType::AllIn,
        m,
    );
    assert!(result.is_err());
}
