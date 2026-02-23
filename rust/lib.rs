//! Rust acceleration layer for the expectation library.
//!
//! Provides high-performance implementations of:
//! - Mixture supermartingales (two-sided normal, one-sided normal)
//! - ParallelSequentialTest engine for 300K+ simultaneous hypothesis tests
//! - Multiple testing procedures (e-Bonferroni, e-BH, e-Holm)
//!
//! # Architecture
//!
//! The hot path is fully monomorphized: `ParallelSequentialTest<M>` is generic
//! over `MixtureSuperMartingale`, so the compiler inlines `log_super_mg` into
//! the rayon parallel loop with zero vtable overhead. Enum dispatch happens
//! once at the PyO3 boundary (in `py.rs`), not per test.
//!
//! # References
//!
//! - Howard, Ramdas, McAuliffe, Sekhon (2022). Time-uniform confidence sequences.
//! - Ramdas, Wang (2025). Hypothesis testing with e-values, Ch. 4 & 7.

pub mod error;
pub mod math;
pub mod martingale;
pub mod merge;
pub mod multiple_testing;
pub mod py;
pub mod par_seqtest;

#[cfg(test)]
mod tests;

use pyo3::prelude::*;

/// Python module: `expectation._rust`
///
/// Exposes the Rust acceleration layer to Python.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)?;
    Ok(())
}
