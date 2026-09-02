// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

use crate::par_seqtest::state::ParTestState;

#[test]
fn test_zeros() {
    let state = ParTestState::zeros(1000);
    assert_eq!(state.n_tests(), 1000);
    assert!(state.data_sum.iter().all(|&x| x == 0.0));
    assert!(state.count.iter().all(|&x| x == 0));
    assert!(state.max_log_m.iter().all(|&x| x == f64::NEG_INFINITY));
    assert!(state.rejected.iter().all(|&x| !x));
}
