//! Complementary error function and normal CDF from first principles.
//!
//! Implements erfc using the Sun/fdlibm rational approximation, which is
//! the standard used by glibc, musl, and most C standard libraries.
//! Achieves relative error < 2e-15 across the entire real line.
//!
//! Reference:
//!   Sun Microsystems, fdlibm (Freely Distributable LIBM), s_erf.c
//!   W.J. Cody, "Rational Chebyshev approximations for the error function,"
//!   Math. Comp. 23(107), 631-637, 1969.

use std::f64::consts::FRAC_1_SQRT_2;

// ── Coefficients from Sun fdlibm s_erf.c ──────────────────────────────

const ERX: f64 = 8.450_629_115_104_675_3e-01;

// ── Region 1: |x| < 0.84375 ───────────────────────────────────────────
// erf(x) = x + x*R(x²) where R = P/Q

const PP0: f64 =  1.283_791_670_955_125_59e-01;
const PP1: f64 = -3.250_421_072_470_015_0e-01;
const PP2: f64 = -2.848_174_957_559_851_0e-02;
const PP3: f64 = -5.770_270_296_489_441_6e-03;
const PP4: f64 = -2.376_301_665_665_016_3e-05;
const QQ1: f64 =  3.979_172_239_591_553_5e-01;
const QQ2: f64 =  6.502_224_998_876_729_4e-02;
const QQ3: f64 =  5.081_306_281_875_766e-03;
const QQ4: f64 =  1.324_947_380_043_216_4e-04;
const QQ5: f64 = -3.960_228_278_775_368_1e-06;

// ── Region 2: 0.84375 <= |x| < 1.25 ───────────────────────────────────
const PA0: f64 = -2.362_118_560_752_659_4e-03;
const PA1: f64 =  4.148_561_186_837_483_3e-01;
const PA2: f64 = -3.722_078_760_357_013_2e-01;
const PA3: f64 =  3.183_466_199_011_617_5e-01;
const PA4: f64 = -1.108_946_942_823_966_8e-01;
const PA5: f64 =  3.547_830_431_952_018_8e-02;
const PA6: f64 = -2.166_375_599_832_541_0e-03;
const QA1: f64 =  1.064_208_804_008_442_3e-01;
const QA2: f64 =  5.403_979_177_021_710_5e-01;
const QA3: f64 =  7.182_865_441_419_625_4e-02;
const QA4: f64 =  1.261_712_198_087_616_4e-01;
const QA5: f64 =  1.363_708_391_202_905_1e-02;
const QA6: f64 =  1.198_449_984_679_910_7e-02;

// ── Region 3: 1.25 <= |x| < 1/0.35 ≈ 2.857 ───────────────────────────
const RA0: f64 = -9.864_944_034_847_148_2e-03;
const RA1: f64 = -6.938_585_727_071_817_6e-01;
const RA2: f64 = -1.055_862_622_532_329_1e+01;
const RA3: f64 = -6.237_533_245_032_600_6e+01;
const RA4: f64 = -1.623_966_694_625_730_7e+02;
const RA5: f64 = -1.846_050_929_067_110_4e+02;
const RA6: f64 = -8.128_743_550_630_659_3e+01;
const RA7: f64 = -9.814_329_344_169_145_5e+00;
const SA1: f64 =  1.965_127_166_743_925_7e+01;
const SA2: f64 =  1.376_577_541_435_197_0e+02;
const SA3: f64 =  4.345_658_774_752_292_3e+02;
const SA4: f64 =  6.453_872_717_332_679e+02;
const SA5: f64 =  4.290_081_400_275_678_3e+02;
const SA6: f64 =  1.086_350_055_417_794_4e+02;
const SA7: f64 =  6.570_249_770_319_282e+00;
const SA8: f64 = -6.042_441_521_485_810e-02;

// ── Region 4: 1/0.35 <= |x| < 28 ──────────────────────────────────────
const RB0: f64 = -9.864_942_924_700_099_3e-03;
const RB1: f64 = -7.992_832_376_805_230_1e-01;
const RB2: f64 = -1.775_795_491_775_475_2e+01;
const RB3: f64 = -1.606_363_848_555_579_4e+02;
const RB4: f64 = -6.375_664_433_683_890_9e+02;
const RB5: f64 = -1.025_095_131_611_077_2e+03;
const RB6: f64 = -4.835_191_916_086_514e+02;
const SB1: f64 =  3.033_806_078_756_257_8e+01;
const SB2: f64 =  3.257_925_129_965_739_2e+02;
const SB3: f64 =  1.536_729_586_084_437_0e+03;
const SB4: f64 =  3.199_858_219_508_595_5e+03;
const SB5: f64 =  2.553_050_406_433_164_4e+03;
const SB6: f64 =  4.745_285_412_069_553_7e+02;
const SB7: f64 = -2.244_095_244_658_582e+01;

/// Complementary error function: erfc(x) = 1 - erf(x).
///
/// Four-region rational approximation (Sun fdlibm / Cody 1969).
/// Relative error < 2e-15 for all finite x.
///
/// # Special cases
/// - `erfc(NaN) = NaN`
/// - `erfc(+inf) = 0`
/// - `erfc(-inf) = 2`
#[inline]
pub fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }

    let ax = x.abs();

    if ax < 0.84375 {
        // Region 1: |x| < 0.84375
        // erf(x) = x + x * P(x²)/Q(x²)
        if ax < 3.725_290_298_461_914_1e-09 {
            // |x| < 2^{-28}: erf(x) ≈ (2/√π)*x
            return 1.0 - 2.0 * x / std::f64::consts::FRAC_2_SQRT_PI;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        if ax < 0.25 {
            1.0 - (x + x * y)
        } else {
            let r_val = x * y + (x - 0.5);
            0.5 - r_val
        }
    } else if ax < 1.25 {
        // Region 2: 0.84375 <= |x| < 1.25
        let s = ax - 1.0;
        let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        if x >= 0.0 {
            1.0 - ERX - p / q
        } else {
            1.0 + ERX + p / q
        }
    } else if ax < 28.0 {
        // Regions 3 & 4: 1.25 <= |x| < 28
        let s = 1.0 / (ax * ax);
        let r_over_s = if ax < 1.0 / 0.35 {
            // Region 3: [1.25, ~2.857)
            let r = RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
            let sv = 1.0 + s * (SA1 + s * (SA2 + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
            r / sv
        } else {
            // Region 4: [~2.857, 28)
            let r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
            let sv = 1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
            r / sv
        };

        // Split exp(-x²) for accuracy:
        // z = ax with low 32 bits zeroed
        // exp(-z²-0.5625) * exp(z²-x² + R/S)  = exp(-x²-0.5625+R/S)
        // Dividing by ax gives erfc(|x|)
        let z = f64::from_bits(ax.to_bits() & 0xFFFF_FFFF_0000_0000);
        let result = (-z * z - 0.5625).exp() * ((z - ax) * (z + ax) + r_over_s).exp() / ax;

        if x >= 0.0 {
            result
        } else {
            2.0 - result
        }
    } else {
        // |x| >= 28: erfc(x) ≈ 0, erfc(-x) ≈ 2
        if x >= 0.0 { 0.0 } else { 2.0 }
    }
}

/// Standard normal CDF: Φ(x) = P(Z <= x) for Z ~ N(0,1).
///
/// Computed as: Φ(x) = 0.5 * erfc(-x / √2).
///
/// # Special cases
/// - `ndtr(NaN) = NaN`
/// - `ndtr(+inf) = 1`
/// - `ndtr(-inf) = 0`
#[inline]
pub fn ndtr(x: f64) -> f64 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// Log of the standard normal CDF: ln Φ(x).
///
/// Uses a numerically stable computation for large negative x where
/// Φ(x) would underflow to zero.
///
/// For x < -20, uses the asymptotic expansion:
///   ln Φ(x) ≈ -x²/2 - ln(-x) - 0.5*ln(2π) + ln(1 - 1/x² + 3/x⁴ - ...)
///
/// For x >= -20, uses: ln(0.5 * erfc(-x/√2))
#[inline]
pub fn log_ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }

    if x < -20.0 {
        // Asymptotic expansion for the upper tail
        // Φ(x) ≈ φ(x)/(-x) * (1 - 1/x² + 3/x⁴ - 15/x⁶ + ...)
        // ln Φ(x) ≈ -x²/2 - ln(2π)/2 - ln(-x) + ln(1 - 1/x² + 3/x⁴ - 15/x⁶)
        let xsq = x * x;
        let inv_xsq = 1.0 / xsq;
        let series = 1.0 + inv_xsq * (-1.0 + inv_xsq * (3.0 + inv_xsq * (-15.0 + inv_xsq * 105.0)));
        -0.5 * xsq - (-x).ln() - 0.5 * (2.0 * std::f64::consts::PI).ln() + series.ln()
    } else {
        let val = 0.5 * erfc(-x * FRAC_1_SQRT_2);
        val.ln()
    }
}

// Unit tests are in rust/tests/test_erfc.rs
