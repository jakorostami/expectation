// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Parallel sequential testing engine for massively parallel hypothesis tests.
//!
//! Processes 300K+ tests simultaneously using Structure-of-Arrays layout
//! and rayon parallel iteration. Each time step updates all tests in parallel.
//!
//! # Architecture
//!
//! `ParallelSequentialTest<M>` is generic over the martingale type `M`, so the
//! compiler monomorphizes the entire hot loop -- `log_super_mg` is inlined with
//! zero vtable overhead. The PyO3 boundary uses enum dispatch (one match per
//! call, not per test).
//!
//! # Global merge (optional)
//!
//! When `merge_config` is provided, after each parallel step the engine:
//! 1. Reads `log_e_sequential` → converts to e-values via exp()
//! 2. Applies a spatial merge function (V&W 2024, Corollary 1) → single merged e-value
//! 3. Accumulates temporally into a running merged e-process (R&W 2025, Def. 7.21)
//! 4. Checks Ville's inequality on the merged e-process
//!
//! This gives an anytime-valid test of the intersection hypothesis (all K nulls true).
//!
//! # Performance
//!
//! At ~2.9 ns/call for `log_super_mg` and rayon parallelism:
//! - 300K tests per step: ~1.5 ms (8 cores)
//! - Memory: ~28 MB with full state (fits L3 cache)
//!
//! # References
//!
//! - Ramdas & Wang (2025), Hypothesis testing with e-values, Ch. 4, 7 & 8
//! - Vovk & Wang (2024), Merging sequential e-values via martingales
//! - Waudby-Smith & Ramdas (2024), Estimating means of bounded random
//!   variables by betting

pub mod state;
pub mod update;

use crate::error::{EngineError, Result};
use crate::martingale::MixtureSuperMartingale;
use crate::merge::{self, MergeConfig, MergeState};
use state::ParTestState;
pub use update::{AlternativeDirection, CombinerType, VarianceConfig};

/// Parallel engine for massively concurrent sequential hypothesis tests.
///
/// Each test runs an independent sequential e-process with its own sufficient
/// statistics. The `step()` method processes one observation per test
/// in parallel using rayon.
///
/// When `merge_config` is `Some`, the engine additionally merges all K
/// per-step e-values into a single merged e-value and accumulates it
/// temporally into a merged e-process for the intersection hypothesis.
pub struct ParallelSequentialTest<M: MixtureSuperMartingale> {
    /// SoA per-test state
    pub state: ParTestState,
    /// Per-test null hypothesis values
    null_values: Vec<f64>,
    /// ln(1/alpha) threshold for Ville's inequality
    log_threshold: f64,
    /// Significance level
    alpha: f64,
    /// How variance is computed
    variance_config: VarianceConfig,
    /// How e-values are combined into e-processes
    combiner: CombinerType,
    /// Alternative hypothesis direction
    alternative: AlternativeDirection,
    /// The shared mixture supermartingale
    martingale: M,
    /// Number of time steps processed
    time_step: u64,
    /// Optional merge configuration (None = no merging, backward compatible)
    merge_config: Option<MergeConfig>,
    /// Optional merge state (allocated only when merge_config is Some)
    merge_state: Option<MergeState>,
}

impl<M: MixtureSuperMartingale> ParallelSequentialTest<M> {
    /// Create a new ParallelSequentialTest.
    ///
    /// # Arguments
    /// * `n_tests` - Number of simultaneous hypothesis tests
    /// * `null_values` - Per-test null hypothesis values (length must be n_tests)
    /// * `alpha` - Significance level for per-test Ville rejection
    /// * `variance_config` - How variance is determined
    /// * `combiner` - How sequential e-values are combined
    /// * `alternative` - Alternative hypothesis direction
    /// * `martingale` - The mixture supermartingale (shared across tests)
    /// * `merge_config` - Optional merge configuration for intersection testing
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if null_values length != n_tests.
    /// Returns `InvalidParameter` if alpha not in (0, 1).
    pub fn new(
        n_tests: usize,
        null_values: Vec<f64>,
        alpha: f64,
        variance_config: VarianceConfig,
        combiner: CombinerType,
        alternative: AlternativeDirection,
        martingale: M,
        merge_config: Option<MergeConfig>,
    ) -> Result<Self> {
        if null_values.len() != n_tests {
            return Err(EngineError::DimensionMismatch {
                expected: n_tests,
                got: null_values.len(),
            });
        }
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(EngineError::InvalidParameter(
                "alpha must be in (0, 1)".into(),
            ));
        }

        let merge_state = merge_config.as_ref().map(|_| MergeState::new());

        Ok(Self {
            state: ParTestState::zeros(n_tests),
            null_values,
            log_threshold: (1.0 / alpha).ln(),
            alpha,
            variance_config,
            combiner,
            alternative,
            martingale,
            time_step: 0,
            merge_config,
            merge_state,
        })
    }

    /// Number of tests.
    #[inline]
    pub fn n_tests(&self) -> usize {
        self.state.n_tests()
    }

    /// Current time step (number of observations processed).
    #[inline]
    pub fn time_step(&self) -> u64 {
        self.time_step
    }

    /// Significance level.
    #[inline]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Reference to the martingale.
    #[inline]
    pub fn martingale(&self) -> &M {
        &self.martingale
    }

    /// Process one observation per test.
    ///
    /// Returns a `StepResult` with the number of newly rejected tests
    /// and optional merged e-process fields.
    ///
    /// # Errors
    /// Returns `DimensionMismatch` if observations length != n_tests.
    pub fn step(&mut self, observations: &[f64]) -> Result<StepResult> {
        self.time_step += 1;

        let n_newly_rejected = update::step_parallel(
            &mut self.state,
            observations,
            &self.null_values,
            self.log_threshold,
            &self.variance_config,
            self.combiner,
            self.alternative,
            self.time_step,
            &self.martingale,
        )?;

        // Apply merge if configured
        if let (Some(ref config), Some(ref mut state)) =
            (&self.merge_config, &mut self.merge_state)
        {
            merge::apply_merge(
                &self.state.log_e_sequential,
                &self.state.rejected,
                config,
                state,
                self.log_threshold,
                self.time_step,
            );
        }

        let n_rejected = self.state.rejected.iter().filter(|&&r| r).count();

        // Build StepResult with optional merged fields
        let (merged_e_value, log_merged_e_value, merged_e_process,
             log_merged_e_process, merged_rejected, merged_p_value, merged_lambda) =
            match &self.merge_state {
                Some(ms) => (
                    Some(ms.log_merged_e_value.exp()),
                    Some(ms.log_merged_e_value),
                    Some(ms.log_merged_e_process.exp()),
                    Some(ms.log_merged_e_process),
                    Some(ms.merged_rejected),
                    Some(ms.merged_p_value),
                    Some(ms.merged_lambda),
                ),
                None => (None, None, None, None, None, None, None),
            };

        Ok(StepResult {
            time_step: self.time_step,
            n_rejected,
            n_tests: self.n_tests(),
            n_newly_rejected,
            merged_e_value,
            log_merged_e_value,
            merged_e_process,
            log_merged_e_process,
            merged_rejected,
            merged_p_value,
            merged_lambda,
        })
    }

    /// Process a batch of time steps (one observation per test per step).
    ///
    /// `observations` shape: T rows, each of length n_tests.
    pub fn step_batch(&mut self, observations: &[Vec<f64>]) -> Result<Vec<StepResult>> {
        let mut results = Vec::with_capacity(observations.len());
        for obs in observations {
            results.push(self.step(obs)?);
        }
        Ok(results)
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Current log e-process values (one per test).
    pub fn log_e_processes(&self) -> &[f64] {
        &self.state.log_e_process
    }

    /// Per-test rejection flags (Ville's inequality, no multiple testing correction).
    pub fn rejected(&self) -> &[bool] {
        &self.state.rejected
    }

    /// Per-test running maxima: log(max_{s<=t} M_s) for each test.
    ///
    /// Used by adjusted multiple testing procedures for carefree error control.
    /// Reference: Tavyrikov, Goeman & de Heide (2025), Section 2.
    pub fn max_log_m(&self) -> &[f64] {
        &self.state.max_log_m
    }

    /// Per-step sequential log e-values: log(E_t) = log_e_cum_t - prev_log_e_cum.
    pub fn log_e_sequential(&self) -> &[f64] {
        &self.state.log_e_sequential
    }

    /// Per-test p-values: min(1, exp(-log_e_process)).
    pub fn p_values(&self) -> &[f64] {
        &self.state.p_value
    }

    /// Per-test stopping times (first rejection step, 0 = not stopped).
    pub fn stopping_times(&self) -> &[u64] {
        &self.state.stopping_time
    }

    /// Per-test current betting fractions (lambda).
    pub fn lambdas(&self) -> &[f64] {
        &self.state.lambda
    }

    /// The log threshold ln(1/alpha).
    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }

    // ── Merge accessors ───────────────────────────────────────────────

    /// Current merged e-value (spatial merge output). None if merge not configured.
    pub fn merged_e_value(&self) -> Option<f64> {
        self.merge_state.as_ref().map(|ms| ms.log_merged_e_value.exp())
    }

    /// Current log merged e-process (temporal). None if merge not configured.
    pub fn log_merged_e_process(&self) -> Option<f64> {
        self.merge_state.as_ref().map(|ms| ms.log_merged_e_process)
    }

    /// Whether the intersection null has been rejected. None if merge not configured.
    pub fn merged_rejected(&self) -> Option<bool> {
        self.merge_state.as_ref().map(|ms| ms.merged_rejected)
    }

    /// Merged p-value. None if merge not configured.
    pub fn merged_p_value(&self) -> Option<f64> {
        self.merge_state.as_ref().map(|ms| ms.merged_p_value)
    }

    /// Merged stopping time (0 = not stopped). None if merge not configured.
    pub fn merged_stopping_time(&self) -> Option<u64> {
        self.merge_state.as_ref().map(|ms| ms.merged_stopping_time)
    }

    /// Current merged temporal lambda. None if merge not configured.
    pub fn merged_lambda(&self) -> Option<f64> {
        self.merge_state.as_ref().map(|ms| ms.merged_lambda)
    }
}

/// Result of a single step over all tests.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub time_step: u64,
    pub n_rejected: usize,
    pub n_tests: usize,
    /// Number of tests newly rejected in this step.
    pub n_newly_rejected: u64,
    // ── Merged fields (populated only when merge is configured) ──
    /// Spatially merged e-value F(E_1^t, ..., E_K^t).
    pub merged_e_value: Option<f64>,
    /// Log of the spatially merged e-value.
    pub log_merged_e_value: Option<f64>,
    /// Current temporal merged e-process M_t.
    pub merged_e_process: Option<f64>,
    /// Log of the temporal merged e-process.
    pub log_merged_e_process: Option<f64>,
    /// Whether the intersection null has been rejected (Ville's inequality on M_t).
    pub merged_rejected: Option<bool>,
    /// Merged p-value: min(1, exp(-log M_t)).
    pub merged_p_value: Option<f64>,
    /// Current merged temporal betting fraction.
    pub merged_lambda: Option<f64>,
}
