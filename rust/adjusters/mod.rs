//! Admissible e-value adjusters for carefree multiple testing.
//!
//! Implements two adjusters from Tavyrikov, Goeman & de Heide (2025),
//! "Carefree multiple testing with e-processes" (arXiv:2501.19360v2), Eq. (5):
//!
//! - **Lookback** (Dawid et al. 2011a):
//!     `A_1(E) = (E - 1 - ln E) / (ln E)^2`
//!
//! - **Sqrt**:
//!     `A_2(E) = sqrt(E) - 1`
//!
//! Both satisfy `∫₁^∞ A(E) / E² dE = 1`, so `(A(sup_{s≤t} E_s))` is a
//! valid e-process when applied to running maxima of e-values.
//!
//! Theorem 1 of the paper: applying adjusted e-BH to running maxima controls
//! FDR-sup at level K₀α/K, yielding **carefree** rejections that are
//! monotonically non-decreasing over time.
//!
//! # Numerical stability
//!
//! The lookback adjuster `A_1(E) = (e^x - 1 - x) / x²` where `x = ln E`
//! has a removable 0/0 singularity at x = 0. We use a 4-term Taylor
//! expansion for |x| < 1e-4:
//!
//!     `A_1 ≈ 1/2 + x/6 + x²/24 + x³/120`
//!
//! For the direct branch, `expm1(x)` avoids catastrophic cancellation.
//!
//! References:
//! - Tavyrikov, Goeman & de Heide (2025). Carefree multiple testing with
//!   e-processes. arXiv:2501.19360v2.
//! - Dawid, Ryter, Vovk, de Heide (2011a). Prequential probability.

/// Which adjuster to apply to running maxima before multiple testing.
///
/// Reference: Tavyrikov, Goeman & de Heide (2025), Eq. (5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdjusterType {
    /// A_1(E) = (E - 1 - ln E) / (ln E)^2. Dawid et al. (2011a).
    Lookback,
    /// A_2(E) = sqrt(E) - 1.
    Sqrt,
}

impl AdjusterType {
    /// Compute the adjuster on the natural scale: A(E).
    ///
    /// Returns 0.0 for E <= 1 (no evidence against the null).
    #[inline]
    pub fn adjust(self, e: f64) -> f64 {
        match self {
            AdjusterType::Lookback => adjust_lookback(e),
            AdjusterType::Sqrt => adjust_sqrt(e),
        }
    }

    /// Compute the adjuster in log space: ln(A(exp(log_e))).
    ///
    /// Returns `NEG_INFINITY` for log_e <= 0.
    #[inline]
    pub fn log_adjust(self, log_e: f64) -> f64 {
        match self {
            AdjusterType::Lookback => log_adjust_lookback(log_e),
            AdjusterType::Sqrt => log_adjust_sqrt(log_e),
        }
    }
}

// ── Lookback: A_1(E) = (E - 1 - ln E) / (ln E)^2 ──────────────────────

/// Taylor expansion threshold. For |x| < 1e-4, x^8/40320 < 1e-32,
/// so a 4-term Taylor is accurate to well beyond f64 precision.
const TAYLOR_THRESHOLD_SQ: f64 = 1e-8;

/// Lookback adjuster on the natural scale.
///
/// A_1(E) = (E - 1 - ln E) / (ln E)^2
///
/// For E <= 1, returns 0.0 (no evidence).
/// Near E = 1, uses Taylor expansion to avoid 0/0.
#[inline]
fn adjust_lookback(e: f64) -> f64 {
    if e <= 1.0 {
        return 0.0;
    }
    let x = e.ln();
    lookback_from_log(x)
}

/// Lookback adjuster in log space: ln(A_1(exp(log_e))).
///
/// For log_e <= 0, returns NEG_INFINITY.
#[inline]
fn log_adjust_lookback(log_e: f64) -> f64 {
    if log_e <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let val = lookback_from_log(log_e);
    if val <= 0.0 {
        f64::NEG_INFINITY
    } else {
        val.ln()
    }
}

/// Core lookback computation given x = ln(E) > 0.
///
/// A_1 = (e^x - 1 - x) / x^2
///
/// Taylor for small x: 1/2 + x/6 + x^2/24 + x^3/120
/// Direct for larger x: expm1(x) / x^2 - 1/x
#[inline]
fn lookback_from_log(x: f64) -> f64 {
    let x_sq = x * x;
    if x_sq < TAYLOR_THRESHOLD_SQ {
        // Taylor: A_1 = 1/2 + x/6 + x^2/24 + x^3/120
        0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0)))
    } else {
        // Direct: (expm1(x) - x) / x^2 = expm1(x)/x^2 - 1/x
        // Using expm1 avoids cancellation in e^x - 1.
        (x.exp_m1() - x) / x_sq
    }
}

// ── Sqrt: A_2(E) = sqrt(E) - 1 ─────────────────────────────────────────

/// Sqrt adjuster on the natural scale.
///
/// A_2(E) = sqrt(E) - 1
///
/// For E <= 1, returns 0.0.
#[inline]
fn adjust_sqrt(e: f64) -> f64 {
    if e <= 1.0 {
        return 0.0;
    }
    e.sqrt() - 1.0
}

/// Sqrt adjuster in log space: ln(sqrt(exp(log_e)) - 1) = ln(expm1(log_e / 2)).
///
/// For log_e <= 0, returns NEG_INFINITY.
/// Uses expm1 for stability when log_e is small.
#[inline]
fn log_adjust_sqrt(log_e: f64) -> f64 {
    if log_e <= 0.0 {
        return f64::NEG_INFINITY;
    }
    // sqrt(E) - 1 = exp(log_e/2) - 1 = expm1(log_e/2)
    let val = (log_e / 2.0).exp_m1();
    if val <= 0.0 {
        f64::NEG_INFINITY
    } else {
        val.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Boundary tests ──────────────────────────────────────────────

    #[test]
    fn test_lookback_at_one() {
        assert_eq!(AdjusterType::Lookback.adjust(1.0), 0.0);
    }

    #[test]
    fn test_lookback_below_one() {
        assert_eq!(AdjusterType::Lookback.adjust(0.5), 0.0);
        assert_eq!(AdjusterType::Lookback.adjust(0.0), 0.0);
    }

    #[test]
    fn test_sqrt_at_one() {
        assert_eq!(AdjusterType::Sqrt.adjust(1.0), 0.0);
    }

    #[test]
    fn test_sqrt_below_one() {
        assert_eq!(AdjusterType::Sqrt.adjust(0.5), 0.0);
    }

    #[test]
    fn test_log_lookback_at_zero() {
        assert_eq!(AdjusterType::Lookback.log_adjust(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_lookback_negative() {
        assert_eq!(AdjusterType::Lookback.log_adjust(-1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn test_log_sqrt_at_zero() {
        assert_eq!(AdjusterType::Sqrt.log_adjust(0.0), f64::NEG_INFINITY);
    }

    // ── Known values ────────────────────────────────────────────────

    #[test]
    fn test_lookback_at_e() {
        // E = e, x = 1: A_1 = (e - 1 - 1) / 1 = e - 2 ≈ 0.71828
        let result = AdjusterType::Lookback.adjust(std::f64::consts::E);
        let expected = std::f64::consts::E - 2.0;
        assert!(
            (result - expected).abs() < 1e-14,
            "Lookback(e) = {}, expected {}",
            result,
            expected
        );
    }

    #[test]
    fn test_sqrt_at_4() {
        // A_2(4) = sqrt(4) - 1 = 1.0
        let result = AdjusterType::Sqrt.adjust(4.0);
        assert!((result - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_sqrt_at_9() {
        // A_2(9) = sqrt(9) - 1 = 2.0
        let result = AdjusterType::Sqrt.adjust(9.0);
        assert!((result - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_log_sqrt_known() {
        // log(A_2(exp(2))) = log(exp(1) - 1) = log(e - 1)
        let result = AdjusterType::Sqrt.log_adjust(2.0);
        let expected = (std::f64::consts::E - 1.0).ln();
        assert!(
            (result - expected).abs() < 1e-14,
            "log_sqrt(2) = {}, expected {}",
            result,
            expected
        );
    }

    // ── Lookback limit at E→1⁺ ──────────────────────────────────────

    #[test]
    fn test_lookback_limit_near_one() {
        // As E → 1+, A_1(E) → 1/2 (Taylor leading term)
        let e = 1.0 + 1e-10;
        let result = AdjusterType::Lookback.adjust(e);
        assert!(
            (result - 0.5).abs() < 1e-6,
            "Lookback(1+eps) = {}, expected ~0.5",
            result
        );
    }

    // ── Taylor vs direct accuracy ───────────────────────────────────

    #[test]
    fn test_taylor_vs_direct_accuracy() {
        // At x = 1e-5 (within Taylor range), compare against a
        // high-precision reference: 0.5 + x/6 + x^2/24 + x^3/120
        let x = 1e-5_f64;
        let e = x.exp();
        let result = AdjusterType::Lookback.adjust(e);
        let reference = 0.5 + x / 6.0 + x * x / 24.0 + x * x * x / 120.0;
        assert!(
            (result - reference).abs() < 1e-13,
            "Taylor result {} vs reference {}",
            result,
            reference
        );
    }

    #[test]
    fn test_taylor_direct_continuity() {
        // At the boundary x ≈ 1e-4, both branches should agree closely.
        let x_below = 0.99e-4;
        let x_above = 1.01e-4;
        let result_below = lookback_from_log(x_below);
        let result_above = lookback_from_log(x_above);
        assert!(
            (result_below - result_above).abs() < 1e-13,
            "Discontinuity at Taylor threshold: {} vs {}",
            result_below,
            result_above
        );
    }

    // ── Monotonicity ────────────────────────────────────────────────

    #[test]
    fn test_lookback_monotonicity() {
        let mut prev = 0.0_f64;
        for i in 1..2000 {
            let log_e = i as f64 * 0.01;
            let val = AdjusterType::Lookback.adjust(log_e.exp());
            assert!(
                val >= prev,
                "Lookback not monotone at log_e = {}: {} < {}",
                log_e,
                val,
                prev
            );
            prev = val;
        }
    }

    #[test]
    fn test_sqrt_monotonicity() {
        let mut prev = 0.0_f64;
        for i in 1..2000 {
            let log_e = i as f64 * 0.01;
            let val = AdjusterType::Sqrt.adjust(log_e.exp());
            assert!(
                val >= prev,
                "Sqrt not monotone at log_e = {}: {} < {}",
                log_e,
                val,
                prev
            );
            prev = val;
        }
    }

    // ── Calibration integral ────────────────────────────────────────

    #[test]
    fn test_lookback_calibration_integral() {
        // ∫₁^∞ A(E)/E² dE should equal 1.
        // Numerical quadrature with substitution u = ln(E):
        // ∫₀^∞ A(exp(u)) * exp(-u) du = ∫₀^∞ (expm1(u) - u)/u² * exp(-u) du
        let n = 1_000_000;
        let u_max = 30.0; // exp(-30) ≈ 1e-13, negligible tail
        let du = u_max / n as f64;
        let mut integral = 0.0;
        for i in 1..n {
            let u = i as f64 * du;
            let a = lookback_from_log(u);
            integral += a * (-u).exp() * du;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "Lookback calibration integral = {}, expected 1.0",
            integral
        );
    }

    #[test]
    fn test_sqrt_calibration_integral() {
        // ∫₁^∞ A_2(E)/E² dE = ∫₀^∞ (exp(u/2) - 1) * exp(-u) du
        //                    = ∫₀^∞ exp(-u/2) du - ∫₀^∞ exp(-u) du = 2 - 1 = 1
        let n = 1_000_000;
        let u_max = 30.0;
        let du = u_max / n as f64;
        let mut integral = 0.0;
        for i in 1..n {
            let u = i as f64 * du;
            let a = (u / 2.0).exp() - 1.0;
            integral += a * (-u).exp() * du;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "Sqrt calibration integral = {}, expected 1.0",
            integral
        );
    }

    // ── Log-space consistency ───────────────────────────────────────

    #[test]
    fn test_log_adjust_consistency_lookback() {
        for &log_e in &[0.01_f64, 0.1, 1.0, 2.0, 5.0, 10.0] {
            let direct = AdjusterType::Lookback.adjust(log_e.exp()).ln();
            let log_fn = AdjusterType::Lookback.log_adjust(log_e);
            assert!(
                (direct - log_fn).abs() < 1e-13,
                "Lookback log inconsistency at log_e={}: direct={}, log_fn={}",
                log_e,
                direct,
                log_fn
            );
        }
    }

    #[test]
    fn test_log_adjust_consistency_sqrt() {
        for &log_e in &[0.01_f64, 0.1, 1.0, 2.0, 5.0, 10.0] {
            let direct = AdjusterType::Sqrt.adjust(log_e.exp()).ln();
            let log_fn = AdjusterType::Sqrt.log_adjust(log_e);
            assert!(
                (direct - log_fn).abs() < 1e-13,
                "Sqrt log inconsistency at log_e={}: direct={}, log_fn={}",
                log_e,
                direct,
                log_fn
            );
        }
    }
}
