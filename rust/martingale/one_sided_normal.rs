// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! One-sided normal mixture supermartingale.
//!
//! Implements the one-sided normal mixture from:
//! - Time-uniform, nonparametric, nonasymptotic confidence sequences,
//!   S.R Howard, A. Ramdas, J. McAuliffe, J. Sekhon (2022), Section 3
//! - Hypothesis testing with e-values, A. Ramdas, R. Wang (2025), Ch. 7
//!
//! The log supermartingale is:
//!   log M(s, v) = 0.5 * ln(4ρ / (v + ρ)) + s² / (2(v + ρ)) + ln(Φ(s / √(v + ρ)))
//!
//! where ρ = TwoSidedNormalMixture::best_rho(v_opt, 2 * alpha_opt) and
//! Φ is the standard normal CDF.
//!
//! Unlike the two-sided case, `bound()` has no closed form and requires
//! bisection root-finding.

use crate::error::{EngineError, Result};
use crate::martingale::MixtureSuperMartingale;
use crate::martingale::TwoSidedNormalMixture;
use crate::math::log_ndtr;

/// One-sided normal mixture supermartingale.
///
/// Stores the precomputed mixing parameter `rho` for O(1) evaluation.
/// This struct is `Send + Sync` (all fields are `f64`), enabling safe
/// parallel use across rayon worker threads.
#[derive(Debug, Clone, Copy)]
pub struct OneSidedNormalMixture {
    rho: f64,
}

impl OneSidedNormalMixture {
    /// Create a new one-sided mixture with optimal rho derived from (v_opt, alpha_opt).
    ///
    /// Uses double-alpha trick: ρ = TwoSidedNormalMixture::best_rho(v_opt, 2 * alpha_opt).
    /// This is because the one-sided test uses half the error budget.
    ///
    /// # Errors
    /// Returns `InvalidParameter` if v_opt <= 0 or alpha_opt not in (0, 0.5].
    pub fn new(v_opt: f64, alpha_opt: f64) -> Result<Self> {
        if v_opt <= 0.0 {
            return Err(EngineError::InvalidParameter(
                "v_opt must be positive".into(),
            ));
        }
        if !(0.0 < alpha_opt && alpha_opt <= 0.5) {
            return Err(EngineError::InvalidParameter(
                "alpha_opt must be in (0, 0.5] for one-sided tests".into(),
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
            return Err(EngineError::InvalidParameter(
                "rho must be positive".into(),
            ));
        }
        Ok(Self { rho })
    }

    /// Optimal mixing parameter rho for the one-sided normal mixture.
    ///
    /// Uses the two-sided formula with doubled alpha:
    ///   ρ = TwoSidedNormalMixture::best_rho(v, 2 * alpha)
    ///
    /// Reference: Howard et al. (2022), Section 3.
    pub fn best_rho(v: f64, alpha: f64) -> Result<f64> {
        TwoSidedNormalMixture::best_rho(v, 2.0 * alpha)
    }

    /// Access the precomputed rho.
    #[inline(always)]
    pub fn rho(&self) -> f64 {
        self.rho
    }
}

impl MixtureSuperMartingale for OneSidedNormalMixture {
    /// Log of the one-sided normal mixture supermartingale.
    ///
    /// log M(s, v) = 0.5 * ln(4ρ / (v + ρ)) + s² / (2(v + ρ)) + ln(Φ(s / √(v + ρ)))
    ///
    /// where Φ is the standard normal CDF.
    ///
    /// For numerical stability with large negative s/√(v+ρ), we use log_ndtr
    /// which employs an asymptotic expansion to avoid underflow.
    #[inline(always)]
    fn log_super_mg(&self, s: f64, v: f64) -> f64 {
        let v_plus_rho = v + self.rho;
        let sqrt_v_plus_rho = v_plus_rho.sqrt();
        let z = s / sqrt_v_plus_rho;

        // 0.5 * ln(4ρ / (v + ρ)) + s² / (2(v + ρ)) + ln(Φ(z))
        0.5 * (4.0 * self.rho / v_plus_rho).ln()
            + (s * s) / (2.0 * v_plus_rho)
            + log_ndtr(z)
    }

    fn s_upper_bound(&self, _v: f64) -> f64 {
        f64::INFINITY
    }

    /// Confidence bound via bisection root-finding.
    ///
    /// Finds the smallest s >= 0 such that log_super_mg(s, v) >= log_threshold.
    /// Unlike the two-sided case, there is no closed-form solution because
    /// of the Φ(s/√(v+ρ)) term.
    ///
    /// Uses bisection with tolerance 2^{-52} (machine epsilon for f64).
    fn bound(&self, v: f64, log_threshold: f64) -> f64 {
        // If log_super_mg at s=0 is already above threshold, bound is 0
        if self.log_super_mg(0.0, v) >= log_threshold {
            return 0.0;
        }

        // Find upper bracket: double s until log_super_mg exceeds threshold
        let mut s_hi = 1.0;
        while self.log_super_mg(s_hi, v) < log_threshold {
            s_hi *= 2.0;
            if s_hi > 1e15 {
                return f64::INFINITY;
            }
        }

        // Bisection
        let mut s_lo = 0.0_f64;
        for _ in 0..200 {
            let s_mid = 0.5 * (s_lo + s_hi);
            if s_mid == s_lo || s_mid == s_hi {
                break;
            }
            if self.log_super_mg(s_mid, v) >= log_threshold {
                s_hi = s_mid;
            } else {
                s_lo = s_mid;
            }
        }

        s_hi
    }
}
