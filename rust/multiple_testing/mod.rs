// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Multiple testing procedures for cross-test error control.
//!
//! Implements e-value analogues of classical multiple testing corrections:
//! - e-Bonferroni (FWER control)
//! - e-BH (FDR control)
//! - e-Holm (step-down FWER control)
//!
//! Reference: Ramdas & Wang (2025), Hypothesis testing with e-values, Ch. 4.

pub mod adjusted;
pub mod bh;
pub mod bonferroni;
pub mod holm;
