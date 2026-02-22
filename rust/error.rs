//! Error types for the voxel field engine.
//!
//! Provides a unified error enum with automatic conversion to PyO3 exceptions.

use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VoxelError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Numerical computation failed: {0}")]
    NumericalError(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, VoxelError>;

impl From<VoxelError> for PyErr {
    fn from(err: VoxelError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
