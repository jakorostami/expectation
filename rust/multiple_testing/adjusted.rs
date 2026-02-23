//! Adjusted multiple testing procedures for carefree error control.
//!
//! Applies admissible adjusters to running maxima of e-processes before
//! passing to standard e-BH / e-Bonferroni / e-Holm procedures.
//!
//! The key insight from Tavyrikov, Goeman & de Heide (2025) is that applying
//! standard multiple testing procedures to running maxima of e-processes
//! (which gives monotone rejections) does NOT preserve FDR/FWER control
//! (Proposition 1, Corollary 2). However, applying an admissible adjuster A
//! to the running maxima first restores validity:
//!
//!     adjusted_e[k] = A(max_{s<=t} E_s^k)
//!
//! Theorem 1: adjusted e-BH controls FDR-sup at level K₀α/K.
//!
//! References:
//! - Tavyrikov, Goeman & de Heide (2025). Carefree multiple testing with
//!   e-processes. arXiv:2501.19360v2.
//! - Ramdas & Wang (2025). Hypothesis testing with e-values, Ch. 4.

use rayon::prelude::*;

use crate::adjusters::AdjusterType;
use crate::multiple_testing::bh::{self, BhResult};
use crate::multiple_testing::bonferroni::{self, BonferroniResult};
use crate::multiple_testing::holm::{self, HolmResult};

/// Apply an adjuster to log running maxima in parallel, producing adjusted log e-values.
///
/// For each test k: adjusted_log_e[k] = log(A(exp(log_running_max[k])))
///                                     = adjuster.log_adjust(log_running_max[k])
fn adjust_log_values(log_running_max: &[f64], adjuster: AdjusterType) -> Vec<f64> {
    log_running_max
        .par_iter()
        .map(|&log_m| adjuster.log_adjust(log_m))
        .collect()
}

/// Adjusted e-BH procedure for carefree FDR control.
///
/// Applies an admissible adjuster to running maxima, then runs e-BH.
/// Controls FDR-sup at level K₀α/K (Theorem 1 of Tavyrikov et al. 2025).
///
/// # Arguments
/// * `log_running_max` - Log running maxima of e-processes: log(max_{s<=t} E_s^k)
/// * `alpha` - Target FDR level
/// * `adjuster` - Which admissible adjuster to apply
pub fn adjusted_e_bh(
    log_running_max: &[f64],
    alpha: f64,
    adjuster: AdjusterType,
) -> BhResult {
    let adjusted = adjust_log_values(log_running_max, adjuster);
    bh::e_bh(&adjusted, alpha)
}

/// Adjusted e-Bonferroni procedure for carefree FWER control.
///
/// Applies an admissible adjuster to running maxima, then runs e-Bonferroni.
///
/// # Arguments
/// * `log_running_max` - Log running maxima of e-processes
/// * `alpha` - Target FWER level
/// * `adjuster` - Which admissible adjuster to apply
pub fn adjusted_e_bonferroni(
    log_running_max: &[f64],
    alpha: f64,
    adjuster: AdjusterType,
) -> BonferroniResult {
    let adjusted = adjust_log_values(log_running_max, adjuster);
    bonferroni::e_bonferroni(&adjusted, alpha)
}

/// Adjusted e-Holm procedure for carefree FWER control.
///
/// Applies an admissible adjuster to running maxima, then runs e-Holm.
///
/// # Arguments
/// * `log_running_max` - Log running maxima of e-processes
/// * `alpha` - Target FWER level
/// * `adjuster` - Which admissible adjuster to apply
pub fn adjusted_e_holm(
    log_running_max: &[f64],
    alpha: f64,
    adjuster: AdjusterType,
) -> HolmResult {
    let adjusted = adjust_log_values(log_running_max, adjuster);
    holm::e_holm(&adjusted, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_null_zero_rejections() {
        // Under the null, max_log_m stays at 0 (or NEG_INFINITY) → adjusted = 0 → no rejections
        let log_running_max = vec![0.0; 100];
        let result = adjusted_e_bh(&log_running_max, 0.05, AdjusterType::Lookback);
        assert_eq!(result.n_rejected, 0);

        let result = adjusted_e_bonferroni(&log_running_max, 0.05, AdjusterType::Sqrt);
        assert_eq!(result.n_rejected, 0);

        let result = adjusted_e_holm(&log_running_max, 0.05, AdjusterType::Lookback);
        assert_eq!(result.n_rejected, 0);
    }

    #[test]
    fn test_all_null_neg_inf() {
        // Initial state: max_log_m = NEG_INFINITY → 0 rejections
        let log_running_max = vec![f64::NEG_INFINITY; 50];
        let result = adjusted_e_bh(&log_running_max, 0.05, AdjusterType::Lookback);
        assert_eq!(result.n_rejected, 0);
    }

    #[test]
    fn test_strong_signal_rejected() {
        // Very large running maxima should be rejected
        let mut log_running_max = vec![0.0; 10];
        // Make the first test have a very large e-value
        log_running_max[0] = 100.0; // exp(100) is enormous
        let result = adjusted_e_bh(&log_running_max, 0.05, AdjusterType::Lookback);
        assert!(result.n_rejected >= 1);
        assert!(result.rejected[0]);
    }

    #[test]
    fn test_adjusted_more_conservative() {
        // Adjusted procedures should reject no more than unadjusted
        // (adjusters shrink values since A(E) < E for all E > 1)
        let log_running_max: Vec<f64> = (0..20)
            .map(|i| if i < 5 { 5.0 } else { 0.5 })
            .collect();

        let unadjusted = bh::e_bh(&log_running_max, 0.05);
        let adjusted_lb = adjusted_e_bh(&log_running_max, 0.05, AdjusterType::Lookback);
        let adjusted_sq = adjusted_e_bh(&log_running_max, 0.05, AdjusterType::Sqrt);

        assert!(
            adjusted_lb.n_rejected <= unadjusted.n_rejected,
            "Lookback adjusted ({}) > unadjusted ({})",
            adjusted_lb.n_rejected,
            unadjusted.n_rejected
        );
        assert!(
            adjusted_sq.n_rejected <= unadjusted.n_rejected,
            "Sqrt adjusted ({}) > unadjusted ({})",
            adjusted_sq.n_rejected,
            unadjusted.n_rejected
        );
    }

    #[test]
    fn test_adjusted_bonferroni_strong() {
        let mut log_running_max = vec![0.0; 5];
        log_running_max[0] = 50.0;
        log_running_max[1] = 50.0;

        let result = adjusted_e_bonferroni(&log_running_max, 0.05, AdjusterType::Sqrt);
        assert!(result.n_rejected >= 2);
    }

    #[test]
    fn test_adjusted_holm_strong() {
        let mut log_running_max = vec![0.0; 5];
        log_running_max[0] = 50.0;

        let result = adjusted_e_holm(&log_running_max, 0.05, AdjusterType::Lookback);
        assert!(result.n_rejected >= 1);
    }

    #[test]
    fn test_empty_input() {
        let result = adjusted_e_bh(&[], 0.05, AdjusterType::Lookback);
        assert_eq!(result.n_rejected, 0);
        assert!(result.rejected.is_empty());
    }
}
