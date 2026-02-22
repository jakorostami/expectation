//! Mathematical primitives for the expectation engine.
//!
//! Provides first-principles implementations of special functions
//! to avoid external dependencies.

pub mod erfc;

pub use erfc::{erfc, log_ndtr, ndtr};
