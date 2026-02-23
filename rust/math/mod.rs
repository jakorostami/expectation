// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Mathematical primitives for the expectation engine.
//!
//! Provides first-principles implementations of special functions
//! to avoid external dependencies.

pub mod erfc;

pub use erfc::{erfc, log_ndtr, ndtr};
