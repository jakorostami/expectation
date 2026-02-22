//! Two-sided normal mixture supermartingale.
//!
//! Implements the two-sided normal mixture from:
//! - Time-uniform, nonparametric, nonasymptotic confidence sequences,
//!   S.R Howard, A. Ramdas, J. McAuliffe, J. Sekhon (2022), Section 3
//! - Hypothesis testing with e-values, A. Ramdas, R. Wang (2025), Ch. 7
//!
//! The log supermartingale is:
//!   log M(s, v) = 0.5 * ln(rho / (v + rho)) + s^2 / (2 * (v + rho))
//!
//! where rho is the optimal mixing parameter computed from (v_opt, alpha_opt).

use crate::error::{Result, VoxelError};
use crate::martingale::MixtureSuperMartingale;

/// Two-sided normal mixture supermartingale.
///
/// Stores the precomputed mixing parameter `rho` for O(1) evaluation.
/// This struct is `Send + Sync` (all fields are `f64`), enabling safe
/// parallel use across rayon worker threads.
#[derive(Debug, Clone, Copy)]
pub struct TwoSidedNormalMixture {
    rho: f64,
}

impl TwoSidedNormalMixture {
    /// Create a new mixture with optimal rho derived from (v_opt, alpha_opt).
    ///
    /// # Errors
    /// Returns `InvalidParameter` if v_opt <= 0 or alpha_opt not in (0, 1).
    pub fn new(v_opt: f64, alpha_opt: f64) -> Result<Self> {
        if v_opt <= 0.0 {
            return Err(VoxelError::InvalidParameter(
                "v_opt must be positive".into(),
            ));
        }
        let rho = Self::best_rho(v_opt, alpha_opt)?;
        Ok(Self { rho })
    }

    /// Create directly from a known rho value (for testing / deserialization).
    ///
    /// # Errors
    /// Returns `InvalidParameter` if rho <= 0.
    pub fn from_rho(rho: f64) -> Result<Self> {
        if rho <= 0.0 {
            return Err(VoxelError::InvalidParameter(
                "rho must be positive".into(),
            ));
        }
        Ok(Self { rho })
    }

    /// Optimal mixing parameter rho for the two-sided normal mixture.
    ///
    /// Formula (Howard et al. 2022):
    ///   rho = v / (2 * ln(1/alpha) + ln(1 + 2 * ln(1/alpha)))
    pub fn best_rho(v: f64, alpha: f64) -> Result<f64> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(VoxelError::InvalidParameter(
                "alpha must be in (0, 1)".into(),
            ));
        }
        let log_inv_alpha = (1.0 / alpha).ln();
        Ok(v / (2.0 * log_inv_alpha + (1.0 + 2.0 * log_inv_alpha).ln()))
    }

    /// Access the precomputed rho.
    #[inline(always)]
    pub fn rho(&self) -> f64 {
        self.rho
    }
}

impl MixtureSuperMartingale for TwoSidedNormalMixture {
    #[inline(always)]
    fn log_super_mg(&self, s: f64, v: f64) -> f64 {
        let v_plus_rho = v + self.rho;
        // 0.5 * ln(rho / (v + rho)) + s^2 / (2 * (v + rho))
        0.5 * (self.rho / v_plus_rho).ln() + (s * s) / (2.0 * v_plus_rho)
    }

    fn s_upper_bound(&self, _v: f64) -> f64 {
        f64::INFINITY
    }

    fn bound(&self, v: f64, log_threshold: f64) -> f64 {
        let v_plus_rho = v + self.rho;
        (v_plus_rho * ((1.0 + v / self.rho).ln() + 2.0 * log_threshold)).sqrt()
    }
}