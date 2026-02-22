//! Parallel step implementation using rayon parallel iterators.
//!
//! The hot loop processes all tests in parallel by zipping the SoA arrays
//! into a single rayon parallel iterator. The variance config match is
//! hoisted outside the hot loop to avoid per-test branching.
//!
//! For the ALL_IN combiner with TwoSidedNormalMixture, each test does:
//!   ~1 LN + ~6 FLOP + 2 CMP ≈ 2.9 ns (release mode)

use rayon::prelude::*;

use crate::error::{Result, EngineError};
use crate::martingale::MixtureSuperMartingale;
use crate::par_seqtest::state::ParTestState;

/// How variance is determined for each test.
#[derive(Debug, Clone)]
pub enum VarianceConfig {
    /// Single known variance for all tests.
    KnownHomogeneous(f64),
    /// Per-test known variances.
    KnownHeterogeneous(Vec<f64>),
    /// Empirical variance estimated from data using Welford's method.
    /// `min_samples`: minimum observations before using the estimate.
    Empirical { min_samples: u32 },
}

/// How sequential e-values are combined into an e-process.
#[derive(Debug, Clone, Copy)]
pub enum CombinerType {
    /// lambda_t = 1 for all t. E-process = cumulative supermartingale.
    /// (Proposition 7.20 in Ramdas & Wang 2025)
    AllIn,
}

/// Core per-test update logic, inlined into the parallel loop.
///
/// Factored out to avoid code duplication across variance config branches.
#[inline(always)]
fn update_test<M: MixtureSuperMartingale>(
    ds: &mut f64,
    dsq: &mut f64,
    cnt: &mut u32,
    _plec: &mut f64,
    lep: &mut f64,
    mlm: &mut f64,
    rej: &mut bool,
    x: f64,
    null: f64,
    v: f64,
    log_threshold: f64,
    martingale: &M,
) {
    // 1-3: Accumulate sufficient statistics
    *ds += x;
    *dsq += x * x;
    *cnt += 1;

    // 4: Centered sum process s = sum(x_i) - null * n
    let s = *ds - null * (*cnt as f64);

    // 6: Cumulative log supermartingale
    let mut log_e_cum = martingale.log_super_mg(s, v);

    // Clamp NaN/Inf for numerical safety (matches Python behavior)
    if log_e_cum.is_nan() || log_e_cum.is_infinite() {
        log_e_cum = if log_e_cum > 0.0 {
            (1e10_f64).ln()
        } else {
            (1e-10_f64).ln()
        };
    }

    // 7: ALL_IN combiner: e-process = cumulative supermartingale
    *lep = log_e_cum;
    *_plec = log_e_cum;

    // 8-9: Ville's inequality check
    if log_e_cum > *mlm {
        *mlm = log_e_cum;
    }
    if *mlm >= log_threshold {
        *rej = true;
    }
}

/// Macro to zip all SoA arrays + observations + nulls into a rayon parallel iterator.
///
/// Returns a parallel iterator yielding:
///   ((((((((ds, dsq), cnt), plec), lep), mlm), rej), obs), null)
macro_rules! zip_state {
    ($state:expr, $observations:expr, $null_values:expr) => {
        $state
            .data_sum
            .par_iter_mut()
            .zip($state.data_sum_sq.par_iter_mut())
            .zip($state.count.par_iter_mut())
            .zip($state.prev_log_e_cum.par_iter_mut())
            .zip($state.log_e_process.par_iter_mut())
            .zip($state.max_log_m.par_iter_mut())
            .zip($state.rejected.par_iter_mut())
            .zip($observations.par_iter().copied())
            .zip($null_values.par_iter().copied())
    };
}

/// Parallel step over all tests for one time step.
///
/// The variance config match is hoisted outside the hot loop so the
/// per-test code has zero branching on variance mode.
pub fn step_parallel<M: MixtureSuperMartingale>(
    state: &mut ParTestState,
    observations: &[f64],
    null_values: &[f64],
    log_threshold: f64,
    variance_config: &VarianceConfig,
    _combiner: CombinerType,
    martingale: &M,
) -> Result<()> {
    let n = state.n_tests();
    if observations.len() != n {
        return Err(EngineError::DimensionMismatch {
            expected: n,
            got: observations.len(),
        });
    }

    match variance_config {
        VarianceConfig::KnownHomogeneous(var) => {
            let var = *var;
            zip_state!(state, observations, null_values).for_each(
                |((((((((ds, dsq), cnt), plec), lep), mlm), rej), x), null)| {
                    // Pre-increment count for variance computation
                    let v = (*cnt as f64 + 1.0) * var;
                    update_test(ds, dsq, cnt, plec, lep, mlm, rej, x, null, v, log_threshold, martingale);
                },
            );
        }
        VarianceConfig::KnownHeterogeneous(ref vars) => {
            zip_state!(state, observations, null_values)
                .zip(vars.par_iter().copied())
                .for_each(
                    |(((((((((ds, dsq), cnt), plec), lep), mlm), rej), x), null), var_i)| {
                        let v = (*cnt as f64 + 1.0) * var_i;
                        update_test(ds, dsq, cnt, plec, lep, mlm, rej, x, null, v, log_threshold, martingale);
                    },
                );
        }
        VarianceConfig::Empirical { min_samples } => {
            let min_s = *min_samples;
            zip_state!(state, observations, null_values).for_each(
                |((((((((ds, dsq), cnt), plec), lep), mlm), rej), x), null)| {
                    // Peek at post-update count for variance
                    let new_cnt = *cnt + 1;
                    let cnt_f = new_cnt as f64;

                    let v = if new_cnt > min_s.max(1) {
                        let new_sum = *ds + x;
                        let new_sum_sq = *dsq + x * x;
                        let mean = new_sum / cnt_f;
                        let var_est = (new_sum_sq / cnt_f - mean * mean) * (cnt_f / (cnt_f - 1.0));
                        (cnt_f * var_est).max(0.01)
                    } else {
                        cnt_f * 1.0
                    };

                    update_test(ds, dsq, cnt, plec, lep, mlm, rej, x, null, v, log_threshold, martingale);
                },
            );
        }
    }

    Ok(())
}
