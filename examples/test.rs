use thiserror::Error;
use std::time::Instant;
use std::hint::black_box;

#[derive(Error, Debug)]
pub enum MartingaleError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Numerical computation failed: {0}")]
    NumericalError(String),

    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize)
}

pub type Result<T> = std::result::Result<T, MartingaleError>;


pub trait MixtureSuperMartingale {
    fn log_super_mg(&self, s: f64, v: f64) -> f64;
    fn s_upper_bound(&self, v: f64) -> f64;
    fn bound(&self, v: f64, log_threshold: f64) -> f64;
}

pub struct TwoSidedNormalMixture {
    rho: f64
}

impl TwoSidedNormalMixture {
    pub fn new(v_opt: f64, alpha_opt: f64) -> Result<Self> {
        if v_opt <= 0.0 {
            return Err(MartingaleError::InvalidParameter(
                "v_opt must be positive".to_string()
            ));
        }

        let rho = Self::best_rho(v_opt, alpha_opt)?;
        Ok(Self { rho })
    }

    pub fn best_rho(v: f64, alpha: f64) -> Result<f64> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(MartingaleError::InvalidParameter(
                "alpha must be between 0 and 1".to_string()
            ));
        }

        let log_inv_alpha = (1.0 / alpha).ln();
        Ok(v / (2.0 * log_inv_alpha + (1.0 + 2.0 * log_inv_alpha).ln()))
    }

    pub fn rho(&self) -> f64 {
        self.rho
    }
}

impl MixtureSuperMartingale for TwoSidedNormalMixture {
    fn log_super_mg(&self, s: f64, v: f64) -> f64 {
        let v_plus_rho = v + self.rho;
        0.5 * (self.rho / v_plus_rho).ln() + (s*s) / (2.0 * v_plus_rho)
    }

    fn s_upper_bound(&self, _v: f64) -> f64 {
        f64::INFINITY
    }

    fn bound(&self, v: f64, log_threshold: f64) -> f64 {
        let v_plus_rho = v + self.rho;
        (v_plus_rho * ((1.0 + v / self.rho).ln() + 2.0 * log_threshold)).sqrt()
    }
}

// fn main() {
//       let mixture = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
//       println!("rho: {}", mixture.rho());
//       println!("log_super_mg(0.5, 1.0): {}", mixture.log_super_mg(0.5, 1.0));
//       println!("bound(1.0, -3.0): {}", mixture.bound(1.0, -3.0));
// }

fn main() {
    let mixture = TwoSidedNormalMixture::new(1.0, 0.05).unwrap();
    let iterations = 10_000_000;

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(mixture.log_super_mg(black_box(0.5), black_box(1.0)));
    }
    let elapsed = start.elapsed();

    println!("Rust: {:.2} ns/call", elapsed.as_nanos() / iterations);
    println!("Total: {:.4} seconds for {} iterations", elapsed.as_secs_f64(), iterations);
}