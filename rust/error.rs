// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Error types for the parallel sequential testing engine.
//!
//! Provides a unified error enum with automatic conversion to PyO3 exceptions.

use pyo3::exceptions::PyValueError;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Numerical computation failed: {0}")]
    NumericalError(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

pub type Result<T> = std::result::Result<T, EngineError>;

impl From<EngineError> for PyErr {
    fn from(err: EngineError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}
