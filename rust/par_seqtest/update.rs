// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Parallel step implementation using rayon parallel iterators.
//!
//! The hot loop processes all tests in parallel by zipping the SoA arrays
//! into a single rayon parallel iterator. The variance config and combiner
//! type matches are hoisted outside the hot loop to avoid per-test branching.
//!
//! References:
//! - Ramdas & Wang (2025). Hypothesis testing with e-values, Ch. 7.
//! - Waudby-Smith & Ramdas (2024). Estimating means of bounded random
//!   variables by betting.

// The nine step specializations below (3 combiners x 3 variance branches) each
// take the Structure-of-Arrays state as explicit parallel slices so the whole
// body monomorphizes and inlines into the rayon hot loop. Grouping them into a
// struct would defeat that, so `too_many_arguments` is intentional here.
#![allow(clippy::too_many_arguments)]

use rayon::prelude::*;

use crate::error::{EngineError, Result};
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

/// Alternative hypothesis direction.
///
/// Controls the sign applied to the centered sum process s.
/// Reference: Ramdas & Wang (2025), Section 2.1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlternativeDirection {
    /// Two-sided test: s used as-is (s² in TwoSidedNormalMixture makes it symmetric).
    TwoSided,
    /// One-sided test for mean > null: s = data_sum - null * count (positive direction).
    Greater,
    /// One-sided test for mean < null: s = -(data_sum - null * count) (negated).
    Less,
}

/// How sequential e-values are combined into an e-process.
///
/// Reference: Ramdas & Wang (2025), Definition 7.21.
#[derive(Debug, Clone, Copy)]
pub enum CombinerType {
    /// λ_t = 1 for all t. E-process = cumulative supermartingale.
    /// (Proposition 7.20 in Ramdas & Wang 2025)
    AllIn,
    /// Fixed λ ∈ (0, 1). E-process = Π((1-λ) + λ·E_t).
    /// More conservative but more robust to model misspecification.
    Conservative { lambda: f64 },
    /// ONS-based adaptive betting: λ_t = clamp(S1/(S2+ε), [0, γ]).
    /// S1 = Σ(E_s - 1), S2 = Σ(E_s - 1)².
    /// (Waudby-Smith & Ramdas 2024, Theorem 7.22 in Ramdas & Wang 2025)
    EmpiricallyAdaptive { gamma: f64, epsilon: f64 },
}

/// Core per-test update logic for the ALL_IN combiner.
///
/// E-process = cumulative supermartingale (Proposition 7.20).
#[inline(always)]
fn update_test_all_in<M: MixtureSuperMartingale>(
    ds: &mut f64,
    dsq: &mut f64,
    cnt: &mut u32,
    plec: &mut f64,
    lep: &mut f64,
    mlm: &mut f64,
    rej: &mut bool,
    les: &mut f64,
    pv: &mut f64,
    st: &mut u64,
    lam: &mut f64,
    x: f64,
    null: f64,
    v: f64,
    sign: f64,
    log_threshold: f64,
    time_step: u64,
    martingale: &M,
) {
    *ds += x;
    *dsq += x * x;
    *cnt += 1;

    let s = sign * (*ds - null * (*cnt as f64));
    let mut log_e_cum = martingale.log_super_mg(s, v);

    if log_e_cum.is_nan() || log_e_cum.is_infinite() {
        log_e_cum = if log_e_cum > 0.0 {
            (1e10_f64).ln()
        } else {
            (1e-10_f64).ln()
        };
    }

    // Sequential e-value: log(E_t)
    *les = log_e_cum - *plec;

    // ALL_IN: e-process = cumulative supermartingale
    *lep = log_e_cum;
    *plec = log_e_cum;
    *lam = 1.0;

    // P-value
    *pv = 1.0_f64.min((-*lep).exp());

    // Ville's inequality
    if log_e_cum > *mlm {
        *mlm = log_e_cum;
    }
    if *mlm >= log_threshold && !*rej {
        *rej = true;
        *st = time_step;
    }
}

/// Core per-test update logic for the CONSERVATIVE combiner.
#[inline(always)]
fn update_test_conservative<M: MixtureSuperMartingale>(
    ds: &mut f64,
    dsq: &mut f64,
    cnt: &mut u32,
    plec: &mut f64,
    lep: &mut f64,
    mlm: &mut f64,
    rej: &mut bool,
    les: &mut f64,
    pv: &mut f64,
    st: &mut u64,
    lam: &mut f64,
    x: f64,
    null: f64,
    v: f64,
    sign: f64,
    fixed_lambda: f64,
    log_threshold: f64,
    time_step: u64,
    martingale: &M,
) {
    *ds += x;
    *dsq += x * x;
    *cnt += 1;

    let s = sign * (*ds - null * (*cnt as f64));
    let mut log_e_cum = martingale.log_super_mg(s, v);

    if log_e_cum.is_nan() || log_e_cum.is_infinite() {
        log_e_cum = if log_e_cum > 0.0 {
            (1e10_f64).ln()
        } else {
            (1e-10_f64).ln()
        };
    }

    let log_e_t = log_e_cum - *plec;
    *les = log_e_t;
    *plec = log_e_cum;

    // Conservative: log M_t += log((1-λ) + λ·E_t)
    let e_t = log_e_t.exp();
    *lep += ((1.0 - fixed_lambda) + fixed_lambda * e_t).ln();
    *lam = fixed_lambda;

    *pv = 1.0_f64.min((-*lep).exp());

    if *lep > *mlm {
        *mlm = *lep;
    }
    if *mlm >= log_threshold && !*rej {
        *rej = true;
        *st = time_step;
    }
}

/// Core per-test update logic for the EMPIRICALLY_ADAPTIVE (ONS) combiner.
///
/// λ_t = clamp(S1 / (S2 + ε), [0, γ]) where S1,S2 are F_{t-1}-measurable.
#[inline(always)]
fn update_test_adaptive<M: MixtureSuperMartingale>(
    ds: &mut f64,
    dsq: &mut f64,
    cnt: &mut u32,
    plec: &mut f64,
    lep: &mut f64,
    mlm: &mut f64,
    rej: &mut bool,
    les: &mut f64,
    pv: &mut f64,
    st: &mut u64,
    s1: &mut f64,
    s2: &mut f64,
    lam: &mut f64,
    x: f64,
    null: f64,
    v: f64,
    sign: f64,
    gamma: f64,
    epsilon: f64,
    log_threshold: f64,
    time_step: u64,
    martingale: &M,
) {
    *ds += x;
    *dsq += x * x;
    *cnt += 1;

    let s = sign * (*ds - null * (*cnt as f64));
    let mut log_e_cum = martingale.log_super_mg(s, v);

    if log_e_cum.is_nan() || log_e_cum.is_infinite() {
        log_e_cum = if log_e_cum > 0.0 {
            (1e10_f64).ln()
        } else {
            (1e-10_f64).ln()
        };
    }

    let log_e_t = log_e_cum - *plec;
    *les = log_e_t;
    *plec = log_e_cum;

    // λ_t from previous-step stats (F_{t-1}-measurable)
    let lambda_t = (*s1 / (*s2 + epsilon)).clamp(0.0, gamma);
    *lam = lambda_t;

    // E-process: log M_t += log((1-λ_t) + λ_t·E_t)
    let e_t = log_e_t.exp();
    *lep += ((1.0 - lambda_t) + lambda_t * e_t).ln();

    // Update ONS stats for next step
    let e_minus_1 = e_t - 1.0;
    *s1 += e_minus_1;
    *s2 += e_minus_1 * e_minus_1;

    *pv = 1.0_f64.min((-*lep).exp());

    if *lep > *mlm {
        *mlm = *lep;
    }
    if *mlm >= log_threshold && !*rej {
        *rej = true;
        *st = time_step;
    }
}

/// Macro to zip core SoA arrays + extended arrays + observations + nulls.
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
            .zip($state.log_e_sequential.par_iter_mut())
            .zip($state.p_value.par_iter_mut())
            .zip($state.stopping_time.par_iter_mut())
            .zip($state.lambda.par_iter_mut())
            .zip($observations.par_iter().copied())
            .zip($null_values.par_iter().copied())
    };
}

/// Macro to zip all SoA arrays including ONS stats for adaptive combiner.
macro_rules! zip_state_adaptive {
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
            .zip($state.log_e_sequential.par_iter_mut())
            .zip($state.p_value.par_iter_mut())
            .zip($state.stopping_time.par_iter_mut())
            .zip($state.sum_e_minus_1.par_iter_mut())
            .zip($state.sum_e_minus_1_sq.par_iter_mut())
            .zip($state.lambda.par_iter_mut())
            .zip($observations.par_iter().copied())
            .zip($null_values.par_iter().copied())
    };
}

/// Parallel step over all tests for one time step.
///
/// Returns the number of newly rejected tests in this step.
pub fn step_parallel<M: MixtureSuperMartingale>(
    state: &mut ParTestState,
    observations: &[f64],
    null_values: &[f64],
    log_threshold: f64,
    variance_config: &VarianceConfig,
    combiner: CombinerType,
    alternative: AlternativeDirection,
    time_step: u64,
    martingale: &M,
) -> Result<u64> {
    let n = state.n_tests();
    if observations.len() != n {
        return Err(EngineError::DimensionMismatch {
            expected: n,
            got: observations.len(),
        });
    }

    let sign = match alternative {
        AlternativeDirection::TwoSided | AlternativeDirection::Greater => 1.0,
        AlternativeDirection::Less => -1.0,
    };

    let prev_rejected: usize = state.rejected.iter().filter(|&&r| r).count();

    match combiner {
        CombinerType::AllIn => {
            step_combiner_all_in(state, observations, null_values, log_threshold,
                variance_config, sign, time_step, martingale)?;
        }
        CombinerType::Conservative { lambda } => {
            step_combiner_conservative(state, observations, null_values, log_threshold,
                variance_config, sign, lambda, time_step, martingale)?;
        }
        CombinerType::EmpiricallyAdaptive { gamma, epsilon } => {
            step_combiner_adaptive(state, observations, null_values, log_threshold,
                variance_config, sign, gamma, epsilon, time_step, martingale)?;
        }
    }

    let cur_rejected: usize = state.rejected.iter().filter(|&&r| r).count();
    Ok((cur_rejected - prev_rejected) as u64)
}

// ── Combiner-specific step functions (variance match hoisted) ──────────

fn step_combiner_all_in<M: MixtureSuperMartingale>(
    state: &mut ParTestState,
    observations: &[f64],
    null_values: &[f64],
    log_threshold: f64,
    variance_config: &VarianceConfig,
    sign: f64,
    time_step: u64,
    martingale: &M,
) -> Result<()> {
    match variance_config {
        VarianceConfig::KnownHomogeneous(var) => {
            let var = *var;
            zip_state!(state, observations, null_values).for_each(
                |((((((((((((ds, dsq), cnt), plec), lep), mlm), rej), les), pv), st), lam), x), null)| {
                    let v = (*cnt as f64 + 1.0) * var;
                    update_test_all_in(ds, dsq, cnt, plec, lep, mlm, rej, les, pv, st, lam,
                        x, null, v, sign, log_threshold, time_step, martingale);
                },
            );
        }
        VarianceConfig::KnownHeterogeneous(ref vars) => {
            zip_state!(state, observations, null_values)
                .zip(vars.par_iter().copied())
                .for_each(
                    |(((((((((((((ds, dsq), cnt), plec), lep), mlm), rej), les), pv), st), lam), x), null), var_i)| {
                        let v = (*cnt as f64 + 1.0) * var_i;
                        update_test_all_in(ds, dsq, cnt, plec, lep, mlm, rej, les, pv, st, lam,
                            x, null, v, sign, log_threshold, time_step, martingale);
                    },
                );
        }
        VarianceConfig::Empirical { min_samples } => {
            let min_s = *min_samples;
            zip_state!(state, observations, null_values).for_each(
                |((((((((((((ds, dsq), cnt), plec), lep), mlm), rej), les), pv), st), lam), x), null)| {
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
                    update_test_all_in(ds, dsq, cnt, plec, lep, mlm, rej, les, pv, st, lam,
                        x, null, v, sign, log_threshold, time_step, martingale);
                },
            );
        }
    }
    Ok(())
}

fn step_combiner_conservative<M: MixtureSuperMartingale>(
    state: &mut ParTestState,
    observations: &[f64],
    null_values: &[f64],
    log_threshold: f64,
    variance_config: &VarianceConfig,
    sign: f64,
    fixed_lambda: f64,
    time_step: u64,
    martingale: &M,
) -> Result<()> {
    match variance_config {
        VarianceConfig::KnownHomogeneous(var) => {
            let var = *var;
            zip_state!(state, observations, null_values).for_each(
                |((((((((((((ds, dsq), cnt), plec), lep), mlm), rej), les), pv), st), lam), x), null)| {
                    let v = (*cnt as f64 + 1.0) * var;
                    update_test_conservative(ds, dsq, cnt, plec, lep, mlm, rej, les, pv, st, lam,
                        x, null, v, sign, fixed_lambda, log_threshold, time_step, martingale);
                },
            );
        }
        VarianceConfig::KnownHeterogeneous(ref vars) => {
            zip_state!(state, observations, null_values)
                .zip(vars.par_iter().copied())
                .for_each(
                    |(((((((((((((ds, dsq), cnt), plec), lep), mlm), rej), les), pv), st), lam), x), null), var_i)| {
                        let v = (*cnt as f64 + 1.0) * var_i;
                        update_test_conservative(ds, dsq, cnt, plec, lep, mlm, rej, les, pv, st, lam,
                            x, null, v, sign, fixed_lambda, log_threshold, time_step, martingale);
                    },
                );
        }
        VarianceConfig::Empirical { min_samples } => {
            let min_s = *min_samples;
            zip_state!(state, observations, null_values).for_each(
                |((((((((((((ds, dsq), cnt), plec), lep), mlm), rej), les), pv), st), lam), x), null)| {
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
                    update_test_conservative(ds, dsq, cnt, plec, lep, mlm, rej, les, pv, st, lam,
                        x, null, v, sign, fixed_lambda, log_threshold, time_step, martingale);
                },
            );
        }
    }
    Ok(())
}

fn step_combiner_adaptive<M: MixtureSuperMartingale>(
    state: &mut ParTestState,
    observations: &[f64],
    null_values: &[f64],
    log_threshold: f64,
    variance_config: &VarianceConfig,
    sign: f64,
    gamma: f64,
    epsilon: f64,
    time_step: u64,
    martingale: &M,
) -> Result<()> {
    match variance_config {
        VarianceConfig::KnownHomogeneous(var) => {
            let var = *var;
            zip_state_adaptive!(state, observations, null_values).for_each(
                |((((((((((((((ds, dsq), cnt), plec), lep), mlm), rej), les), pv), st), s1), s2), lam), x), null)| {
                    let v = (*cnt as f64 + 1.0) * var;
                    update_test_adaptive(ds, dsq, cnt, plec, lep, mlm, rej, les, pv, st, s1, s2, lam,
                        x, null, v, sign, gamma, epsilon, log_threshold, time_step, martingale);
                },
            );
        }
        VarianceConfig::KnownHeterogeneous(ref vars) => {
            zip_state_adaptive!(state, observations, null_values)
                .zip(vars.par_iter().copied())
                .for_each(
                    |(((((((((((((((ds, dsq), cnt), plec), lep), mlm), rej), les), pv), st), s1), s2), lam), x), null), var_i)| {
                        let v = (*cnt as f64 + 1.0) * var_i;
                        update_test_adaptive(ds, dsq, cnt, plec, lep, mlm, rej, les, pv, st, s1, s2, lam,
                            x, null, v, sign, gamma, epsilon, log_threshold, time_step, martingale);
                    },
                );
        }
        VarianceConfig::Empirical { min_samples } => {
            let min_s = *min_samples;
            zip_state_adaptive!(state, observations, null_values).for_each(
                |((((((((((((((ds, dsq), cnt), plec), lep), mlm), rej), les), pv), st), s1), s2), lam), x), null)| {
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
                    update_test_adaptive(ds, dsq, cnt, plec, lep, mlm, rej, les, pv, st, s1, s2, lam,
                        x, null, v, sign, gamma, epsilon, log_threshold, time_step, martingale);
                },
            );
        }
    }
    Ok(())
}
