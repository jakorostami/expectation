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
//! # Performance
//!
//! At ~2.9 ns/call for `log_super_mg` and rayon parallelism:
//! - 300K tests per step: ~1.5 ms (8 cores)
//! - Memory: ~13.5 MB (fits L3 cache)
//!
//! # References
//!
//! - Ramdas & Wang (2025), Hypothesis testing with e-values, Ch. 4 & 7

pub mod state;
pub mod update;

use crate::error::{Result, EngineError};
use crate::martingale::MixtureSuperMartingale;
use state::ParTestState;
pub use update::{CombinerType, VarianceConfig};

/// Parallel engine for massively concurrent sequential hypothesis tests.
///
/// Each test runs an independent sequential e-process with its own sufficient
/// statistics. The `step()` method processes one observation per test
/// in parallel using rayon.
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
    /// The shared mixture supermartingale
    martingale: M,
    /// Number of time steps processed
    time_step: u64,
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
    /// * `martingale` - The mixture supermartingale (shared across tests)
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
        martingale: M,
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

        Ok(Self {
            state: ParTestState::zeros(n_tests),
            null_values,
            log_threshold: (1.0 / alpha).ln(),
            alpha,
            variance_config,
            combiner,
            martingale,
            time_step: 0,
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
    /// # Errors
    /// Returns `DimensionMismatch` if observations length != n_tests.
    pub fn step(&mut self, observations: &[f64]) -> Result<StepResult> {
        update::step_parallel(
            &mut self.state,
            observations,
            &self.null_values,
            self.log_threshold,
            &self.variance_config,
            self.combiner,
            &self.martingale,
        )?;

        self.time_step += 1;

        let n_rejected = self.state.rejected.iter().filter(|&&r| r).count();
        Ok(StepResult {
            time_step: self.time_step,
            n_rejected,
            n_tests: self.n_tests(),
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

    /// Access the current log e-process values (one per test).
    pub fn log_e_processes(&self) -> &[f64] {
        &self.state.log_e_process
    }

    /// Access per-test rejection flags (Ville's inequality, no multiple testing correction).
    pub fn rejected(&self) -> &[bool] {
        &self.state.rejected
    }

    /// Access the log threshold.
    pub fn log_threshold(&self) -> f64 {
        self.log_threshold
    }
}

/// Result of a single step over all tests.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub time_step: u64,
    pub n_rejected: usize,
    pub n_tests: usize,
}
