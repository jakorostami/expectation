// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! Rust unit tests for the expectation engine.

mod test_two_sided_normal;
mod test_one_sided_normal;
mod test_erfc;
mod test_par_test_state;
mod test_par_seqtest_update;
mod test_par_seqtest;
mod test_bonferroni;
mod test_bh;
mod test_holm;
mod test_alternative_direction;
mod test_conservative_combiner;
mod test_ons_combiner;
mod test_adjusters;
mod test_adjusted_multiple_testing;
