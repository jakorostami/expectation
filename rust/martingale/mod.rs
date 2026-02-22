//! Mixture supermartingale trait and implementations.
//!
//! Based on:
//! - Time-uniform, nonparametric, nonasymptotic confidence sequences,
//!   S.R Howard, A. Ramdas, J. McAuliffe, J. Sekhon (2022)
//! - Hypothesis testing with e-values, A. Ramdas, R. Wang (2025), Ch. 7

pub mod two_sided_normal;

pub use two_sided_normal::TwoSidedNormalMixture;

/// Core trait for mixture supermartingales.
///
/// Implementors provide `log_super_mg(s, v)` which computes the log of the
/// mixture supermartingale at process value `s` and intrinsic time `v`.
///
/// `Send + Sync` bounds enable safe use across rayon parallel iterators.
pub trait MixtureSuperMartingale: Send + Sync {
    /// Log of the mixture supermartingale M(s, v).
    ///
    /// For the two-sided normal mixture:
    ///   log M = 0.5 * ln(rho / (v + rho)) + s^2 / (2 * (v + rho))
    fn log_super_mg(&self, s: f64, v: f64) -> f64;

    /// Upper bound on s for a given v (used by root-finding for confidence bounds).
    fn s_upper_bound(&self, v: f64) -> f64;

    /// Confidence bound: smallest s such that log_super_mg(s, v) >= log_threshold.
    fn bound(&self, v: f64, log_threshold: f64) -> f64;
}
