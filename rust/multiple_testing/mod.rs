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
