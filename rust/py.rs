// SPDX-License-Identifier: GPL-3.0-only AND LicenseRef-AI-Training-Prohibited
// Copyright (c) Jako Rostami 2024-present
// Project: expectation
//
// Licensed under GPL-3.0 with additional restrictions per Section 7(b).
// Use of this code for AI/ML model training is strictly prohibited.
// See LICENSE for full terms.

//! PyO3 bindings for the ParallelSequentialTest engine.
//!
//! Exposes `PyParallelSequentialTest` to Python with numpy array I/O.
//! Uses enum dispatch at the Python boundary (one match per call, not per test)
//! while the inner `ParallelSequentialTest<M>` remains fully monomorphized.

// This module is a thin PyO3 FFI boundary. The #[pymethods] proc-macro (PyO3
// 0.22) expands every fallible method into code that converts EngineError ->
// PyErr; clippy attributes that macro-generated code to the method signature
// spans as `useless_conversion` / `redundant_closure`. They are false positives
// on generated code, not hand-written conversions. `too_many_arguments` is also
// allowed here: the constructor exposes 22 configuration parameters by design.
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::adjusters::AdjusterType;
use crate::error::EngineError;
use crate::martingale::{OneSidedNormalMixture, TwoSidedNormalMixture};
use crate::merge::{MergeCombinerType, MergeConfig, MergeFunction};
use crate::multiple_testing::{adjusted, bh, bonferroni, holm};
use crate::par_seqtest::update::{AlternativeDirection, CombinerType, VarianceConfig};
use crate::par_seqtest::ParallelSequentialTest;

/// Enum dispatch: one match per Python call, zero overhead in the hot loop.
enum MartingaleKind {
    TwoSidedNormal(ParallelSequentialTest<TwoSidedNormalMixture>),
    OneSidedNormal(ParallelSequentialTest<OneSidedNormalMixture>),
}

/// High-performance parallel engine for massively concurrent sequential tests.
///
/// Processes 300K+ tests simultaneously using rayon parallelism and
/// Structure-of-Arrays memory layout. Each test runs an independent
/// sequential hypothesis test with anytime-valid guarantees.
///
/// When `global_merge` is specified, the engine additionally merges all K
/// per-step e-values into a single merged e-value (Vovk & Wang 2024) and
/// accumulates it temporally into an e-process for the intersection
/// hypothesis (Ramdas & Wang 2025, Definition 7.21).
#[pyclass(name = "PyParallelSequentialTest")]
pub struct PyParallelSequentialTest {
    inner: MartingaleKind,
}

#[pymethods]
impl PyParallelSequentialTest {
    /// Create a new ParallelSequentialTest.
    ///
    /// Args:
    ///     n_tests: Number of simultaneous hypothesis tests.
    ///     null_values: Per-test null values (numpy array or scalar broadcast).
    ///     alpha: Significance level for per-test Ville rejection.
    ///     martingale_type: "two_sided_normal" or "one_sided_normal".
    ///     v_opt: Optimal intrinsic time for the mixture.
    ///     alpha_opt: Optimal alpha for mixing parameter.
    ///     variance: Known variance (scalar or array). None for empirical.
    ///     min_samples: Min samples before empirical variance (default 30).
    ///     alternative: "two_sided", "greater", or "less" (default "two_sided").
    ///     combiner: "all_in", "conservative", or "empirically_adaptive" (default "all_in").
    ///     conservative_lambda: Lambda for conservative combiner (default 0.5).
    ///     gamma: Cap for adaptive combiner lambda (default 0.5).
    ///     epsilon: Regularization for adaptive combiner (default 1e-6).
    ///     global_merge: Merging function name or None (default None).
    ///     merge_u_order: U-statistic order n (default 1).
    ///     merge_lambda_param: Lambda for lambda_product merge (default 0.5).
    ///     merge_segments: Segment boundaries for segment_product merge (default None).
    ///     merge_combiner: Temporal combiner for merged stream (default "all_in").
    ///     merge_conservative_lambda: Lambda for conservative merge combiner (default 0.5).
    ///     merge_gamma: Cap for adaptive merge combiner (default 0.5).
    ///     merge_epsilon: Regularization for adaptive merge combiner (default 1e-6).
    ///     merge_include_rejected: Include rejected tests in merge (default true).
    #[new]
    #[pyo3(signature = (
        n_tests, null_values, alpha, martingale_type, v_opt, alpha_opt,
        variance=None, min_samples=30, alternative="two_sided",
        combiner="all_in", conservative_lambda=0.5, gamma=0.5, epsilon=1e-6,
        global_merge=None, merge_u_order=1, merge_lambda_param=0.5,
        merge_segments=None, merge_combiner="all_in",
        merge_conservative_lambda=0.5, merge_gamma=0.5, merge_epsilon=1e-6,
        merge_include_rejected=true
    ))]
    fn new(
        py: Python<'_>,
        n_tests: usize,
        null_values: &Bound<'_, PyAny>,
        alpha: f64,
        martingale_type: &str,
        v_opt: f64,
        alpha_opt: f64,
        variance: Option<&Bound<'_, PyAny>>,
        min_samples: u32,
        alternative: &str,
        combiner: &str,
        conservative_lambda: f64,
        gamma: f64,
        epsilon: f64,
        global_merge: Option<&str>,
        merge_u_order: usize,
        merge_lambda_param: f64,
        merge_segments: Option<Vec<usize>>,
        merge_combiner: &str,
        merge_conservative_lambda: f64,
        merge_gamma: f64,
        merge_epsilon: f64,
        merge_include_rejected: bool,
    ) -> PyResult<Self> {
        let null_vec = parse_float_input(py, null_values, n_tests, "null_values")?;

        let variance_config = match variance {
            Some(var_obj) => {
                if let Ok(scalar) = var_obj.extract::<f64>() {
                    VarianceConfig::KnownHomogeneous(scalar)
                } else {
                    let arr: Vec<f64> = var_obj.extract()?;
                    if arr.len() != n_tests {
                        return Err(EngineError::DimensionMismatch {
                            expected: n_tests,
                            got: arr.len(),
                        }
                        .into());
                    }
                    VarianceConfig::KnownHeterogeneous(arr)
                }
            }
            None => VarianceConfig::Empirical { min_samples },
        };

        let alt = match alternative {
            "two_sided" => AlternativeDirection::TwoSided,
            "greater" => AlternativeDirection::Greater,
            "less" => AlternativeDirection::Less,
            other => {
                return Err(EngineError::InvalidParameter(format!(
                    "Unknown alternative: '{}'. Supported: 'two_sided', 'greater', 'less'",
                    other
                ))
                .into())
            }
        };

        let comb = match combiner {
            "all_in" => CombinerType::AllIn,
            "conservative" => CombinerType::Conservative {
                lambda: conservative_lambda,
            },
            "empirically_adaptive" => CombinerType::EmpiricallyAdaptive { gamma, epsilon },
            other => {
                return Err(EngineError::InvalidParameter(format!(
                    "Unknown combiner: '{}'. Supported: 'all_in', 'conservative', 'empirically_adaptive'",
                    other
                ))
                .into())
            }
        };

        // Parse merge configuration
        let merge_config = match global_merge {
            Some(merge_fn) => {
                let function = match merge_fn {
                    "arithmetic_mean" => MergeFunction::ArithmeticMean,
                    "u_statistic" => MergeFunction::UStatistic { n: merge_u_order },
                    "lambda_product" => MergeFunction::LambdaProduct {
                        lambda: merge_lambda_param,
                    },
                    "segment_product" => {
                        let segs = merge_segments.ok_or_else(|| {
                            EngineError::InvalidParameter(
                                "merge_segments required for segment_product merge".into(),
                            )
                        })?;
                        MergeFunction::SegmentProduct { segments: segs }
                    }
                    "product" => MergeFunction::Product,
                    other => {
                        return Err(EngineError::InvalidParameter(format!(
                            "Unknown global_merge: '{}'. Supported: 'arithmetic_mean', \
                             'u_statistic', 'lambda_product', 'segment_product', 'product'",
                            other
                        ))
                        .into())
                    }
                };

                let merge_comb = match merge_combiner {
                    "all_in" => MergeCombinerType::AllIn,
                    "conservative" => MergeCombinerType::Conservative {
                        lambda: merge_conservative_lambda,
                    },
                    "empirically_adaptive" => MergeCombinerType::EmpiricallyAdaptive {
                        gamma: merge_gamma,
                        epsilon: merge_epsilon,
                    },
                    other => {
                        return Err(EngineError::InvalidParameter(format!(
                            "Unknown merge_combiner: '{}'. Supported: 'all_in', \
                             'conservative', 'empirically_adaptive'",
                            other
                        ))
                        .into())
                    }
                };

                Some(MergeConfig {
                    function,
                    combiner: merge_comb,
                    include_rejected: merge_include_rejected,
                })
            }
            None => None,
        };

        // Auto-select martingale based on type and alternative
        match martingale_type {
            "two_sided_normal" => {
                let m = TwoSidedNormalMixture::new(v_opt, alpha_opt)
                    .map_err(|e| Into::<PyErr>::into(e))?;
                let pst = ParallelSequentialTest::new(
                    n_tests, null_vec, alpha, variance_config, comb, alt, m, merge_config,
                )
                .map_err(|e| Into::<PyErr>::into(e))?;
                Ok(Self {
                    inner: MartingaleKind::TwoSidedNormal(pst),
                })
            }
            "one_sided_normal" => {
                let m = OneSidedNormalMixture::new(v_opt, alpha_opt)
                    .map_err(|e| Into::<PyErr>::into(e))?;
                let pst = ParallelSequentialTest::new(
                    n_tests, null_vec, alpha, variance_config, comb, alt, m, merge_config,
                )
                .map_err(|e| Into::<PyErr>::into(e))?;
                Ok(Self {
                    inner: MartingaleKind::OneSidedNormal(pst),
                })
            }
            other => Err(EngineError::InvalidParameter(format!(
                "Unknown martingale_type: '{}'. Supported: 'two_sided_normal', 'one_sided_normal'",
                other
            ))
            .into()),
        }
    }

    /// Process one observation per test for this time step.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        observations: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let obs: Vec<f64> = observations.extract()?;
        let result = match &mut self.inner {
            MartingaleKind::TwoSidedNormal(pst) => {
                pst.step(&obs).map_err(|e| Into::<PyErr>::into(e))?
            }
            MartingaleKind::OneSidedNormal(pst) => {
                pst.step(&obs).map_err(|e| Into::<PyErr>::into(e))?
            }
        };

        let dict = PyDict::new_bound(py);
        dict.set_item("time_step", result.time_step)?;
        dict.set_item("n_rejected", result.n_rejected)?;
        dict.set_item("n_tests", result.n_tests)?;
        dict.set_item("n_newly_rejected", result.n_newly_rejected)?;

        // Merged fields (only set when merge is configured)
        if let Some(v) = result.merged_e_value {
            dict.set_item("merged_e_value", v)?;
        }
        if let Some(v) = result.log_merged_e_value {
            dict.set_item("log_merged_e_value", v)?;
        }
        if let Some(v) = result.merged_e_process {
            dict.set_item("merged_e_process", v)?;
        }
        if let Some(v) = result.log_merged_e_process {
            dict.set_item("log_merged_e_process", v)?;
        }
        if let Some(v) = result.merged_rejected {
            dict.set_item("merged_rejected", v)?;
        }
        if let Some(v) = result.merged_p_value {
            dict.set_item("merged_p_value", v)?;
        }
        if let Some(v) = result.merged_lambda {
            dict.set_item("merged_lambda", v)?;
        }

        Ok(dict)
    }

    /// Get current log e-process values as numpy array.
    fn log_e_processes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.log_e_processes(),
            MartingaleKind::OneSidedNormal(pst) => pst.log_e_processes(),
        };
        Ok(PyArray1::from_slice_bound(py, values))
    }

    /// Get per-test rejection flags (Ville's inequality, no correction).
    fn rejected<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let values = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.rejected(),
            MartingaleKind::OneSidedNormal(pst) => pst.rejected(),
        };
        Ok(PyArray1::from_slice_bound(py, values))
    }

    /// Get per-step sequential log e-values.
    fn log_e_sequential<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.log_e_sequential(),
            MartingaleKind::OneSidedNormal(pst) => pst.log_e_sequential(),
        };
        Ok(PyArray1::from_slice_bound(py, values))
    }

    /// Get per-test p-values.
    fn p_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.p_values(),
            MartingaleKind::OneSidedNormal(pst) => pst.p_values(),
        };
        Ok(PyArray1::from_slice_bound(py, values))
    }

    /// Get per-test stopping times (0 = not stopped).
    fn stopping_times<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let values = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.stopping_times(),
            MartingaleKind::OneSidedNormal(pst) => pst.stopping_times(),
        };
        Ok(PyArray1::from_slice_bound(py, values))
    }

    /// Get per-test current lambda (betting fraction).
    fn lambdas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.lambdas(),
            MartingaleKind::OneSidedNormal(pst) => pst.lambdas(),
        };
        Ok(PyArray1::from_slice_bound(py, values))
    }

    // ── Merge accessors ───────────────────────────────────────────────

    /// Get current merged e-value (None if merge not configured).
    fn merged_e_value(&self) -> PyResult<Option<f64>> {
        Ok(match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.merged_e_value(),
            MartingaleKind::OneSidedNormal(pst) => pst.merged_e_value(),
        })
    }

    /// Get current log merged e-process (None if merge not configured).
    fn log_merged_e_process(&self) -> PyResult<Option<f64>> {
        Ok(match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.log_merged_e_process(),
            MartingaleKind::OneSidedNormal(pst) => pst.log_merged_e_process(),
        })
    }

    /// Get whether intersection null has been rejected (None if merge not configured).
    fn merged_rejected(&self) -> PyResult<Option<bool>> {
        Ok(match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.merged_rejected(),
            MartingaleKind::OneSidedNormal(pst) => pst.merged_rejected(),
        })
    }

    /// Get merged p-value (None if merge not configured).
    fn merged_p_value(&self) -> PyResult<Option<f64>> {
        Ok(match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.merged_p_value(),
            MartingaleKind::OneSidedNormal(pst) => pst.merged_p_value(),
        })
    }

    /// Get merged stopping time (None if merge not configured).
    fn merged_stopping_time(&self) -> PyResult<Option<u64>> {
        Ok(match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.merged_stopping_time(),
            MartingaleKind::OneSidedNormal(pst) => pst.merged_stopping_time(),
        })
    }

    /// Get current merged temporal lambda (None if merge not configured).
    fn merged_lambda(&self) -> PyResult<Option<f64>> {
        Ok(match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.merged_lambda(),
            MartingaleKind::OneSidedNormal(pst) => pst.merged_lambda(),
        })
    }

    // ── Multiple testing corrections ──────────────────────────────────

    /// Apply e-Bonferroni correction for FWER control.
    ///
    /// WARNING: Not carefree. Rejections can disappear with more data.
    /// For FWER-sup control with monotone rejections, use `adjusted_e_bonferroni()`.
    /// Reference: Tavyrikov, Goeman & de Heide (2025), arXiv:2501.19360v2.
    #[pyo3(signature = (alpha=None))]
    fn e_bonferroni<'py>(
        &self,
        py: Python<'py>,
        alpha: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (log_e, default_alpha) = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => (pst.log_e_processes(), pst.alpha()),
            MartingaleKind::OneSidedNormal(pst) => (pst.log_e_processes(), pst.alpha()),
        };
        let alpha = alpha.unwrap_or(default_alpha);
        let result = bonferroni::e_bonferroni(log_e, alpha);

        let dict = PyDict::new_bound(py);
        dict.set_item("rejected", PyArray1::from_slice_bound(py, &result.rejected))?;
        dict.set_item("n_rejected", result.n_rejected)?;
        Ok(dict)
    }

    /// Apply e-BH procedure for FDR control.
    ///
    /// WARNING: Not carefree. Rejections can disappear with more data.
    /// For FDR-sup control with monotone rejections, use `adjusted_e_bh()`.
    /// Reference: Tavyrikov, Goeman & de Heide (2025), arXiv:2501.19360v2.
    #[pyo3(signature = (alpha=None))]
    fn e_bh<'py>(
        &self,
        py: Python<'py>,
        alpha: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (log_e, default_alpha) = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => (pst.log_e_processes(), pst.alpha()),
            MartingaleKind::OneSidedNormal(pst) => (pst.log_e_processes(), pst.alpha()),
        };
        let alpha = alpha.unwrap_or(default_alpha);
        let result = bh::e_bh(log_e, alpha);

        let dict = PyDict::new_bound(py);
        dict.set_item("rejected", PyArray1::from_slice_bound(py, &result.rejected))?;
        dict.set_item("n_rejected", result.n_rejected)?;
        Ok(dict)
    }

    /// Apply e-Holm step-down procedure for FWER control.
    ///
    /// WARNING: Not carefree. Rejections can disappear with more data.
    /// For FWER-sup control with monotone rejections, use `adjusted_e_holm()`.
    /// Reference: Tavyrikov, Goeman & de Heide (2025), arXiv:2501.19360v2.
    #[pyo3(signature = (alpha=None))]
    fn e_holm<'py>(
        &self,
        py: Python<'py>,
        alpha: Option<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (log_e, default_alpha) = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => (pst.log_e_processes(), pst.alpha()),
            MartingaleKind::OneSidedNormal(pst) => (pst.log_e_processes(), pst.alpha()),
        };
        let alpha = alpha.unwrap_or(default_alpha);
        let result = holm::e_holm(log_e, alpha);

        let dict = PyDict::new_bound(py);
        dict.set_item("rejected", PyArray1::from_slice_bound(py, &result.rejected))?;
        dict.set_item("n_rejected", result.n_rejected)?;
        Ok(dict)
    }

    // ── Running maxima accessor ──────────────────────────────────────

    /// Get per-test running maxima: log(max_{s<=t} M_s) as numpy array.
    ///
    /// Used by adjusted multiple testing procedures for carefree error control.
    /// Reference: Tavyrikov, Goeman & de Heide (2025), Section 2.
    fn max_log_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.max_log_m(),
            MartingaleKind::OneSidedNormal(pst) => pst.max_log_m(),
        };
        Ok(PyArray1::from_slice_bound(py, values))
    }

    // ── Adjusted (carefree) multiple testing corrections ─────────────

    /// Apply adjusted e-BH procedure for carefree FDR control.
    ///
    /// Applies an admissible adjuster to running maxima of e-processes,
    /// then runs e-BH. Controls FDR-sup at level K₀α/K, yielding
    /// monotonically non-decreasing rejections over time.
    ///
    /// Args:
    ///     alpha: Target FDR level (default: engine alpha).
    ///     adjuster: "lookback" or "sqrt" (default: "lookback").
    ///
    /// Reference: Tavyrikov, Goeman & de Heide (2025), Theorem 1.
    #[pyo3(signature = (alpha=None, adjuster="lookback"))]
    fn adjusted_e_bh<'py>(
        &self,
        py: Python<'py>,
        alpha: Option<f64>,
        adjuster: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let adj = parse_adjuster(adjuster)?;
        let (max_log, default_alpha) = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => (pst.max_log_m(), pst.alpha()),
            MartingaleKind::OneSidedNormal(pst) => (pst.max_log_m(), pst.alpha()),
        };
        let alpha = alpha.unwrap_or(default_alpha);
        let result = adjusted::adjusted_e_bh(max_log, alpha, adj);

        let dict = PyDict::new_bound(py);
        dict.set_item("rejected", PyArray1::from_slice_bound(py, &result.rejected))?;
        dict.set_item("n_rejected", result.n_rejected)?;
        Ok(dict)
    }

    /// Apply adjusted e-Bonferroni procedure for carefree FWER control.
    ///
    /// Applies an admissible adjuster to running maxima of e-processes,
    /// then runs e-Bonferroni.
    ///
    /// Args:
    ///     alpha: Target FWER level (default: engine alpha).
    ///     adjuster: "lookback" or "sqrt" (default: "lookback").
    ///
    /// Reference: Tavyrikov, Goeman & de Heide (2025), Theorem 1.
    #[pyo3(signature = (alpha=None, adjuster="lookback"))]
    fn adjusted_e_bonferroni<'py>(
        &self,
        py: Python<'py>,
        alpha: Option<f64>,
        adjuster: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let adj = parse_adjuster(adjuster)?;
        let (max_log, default_alpha) = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => (pst.max_log_m(), pst.alpha()),
            MartingaleKind::OneSidedNormal(pst) => (pst.max_log_m(), pst.alpha()),
        };
        let alpha = alpha.unwrap_or(default_alpha);
        let result = adjusted::adjusted_e_bonferroni(max_log, alpha, adj);

        let dict = PyDict::new_bound(py);
        dict.set_item("rejected", PyArray1::from_slice_bound(py, &result.rejected))?;
        dict.set_item("n_rejected", result.n_rejected)?;
        Ok(dict)
    }

    /// Apply adjusted e-Holm procedure for carefree FWER control.
    ///
    /// Applies an admissible adjuster to running maxima of e-processes,
    /// then runs e-Holm.
    ///
    /// Args:
    ///     alpha: Target FWER level (default: engine alpha).
    ///     adjuster: "lookback" or "sqrt" (default: "lookback").
    ///
    /// Reference: Tavyrikov, Goeman & de Heide (2025), Theorem 1.
    #[pyo3(signature = (alpha=None, adjuster="lookback"))]
    fn adjusted_e_holm<'py>(
        &self,
        py: Python<'py>,
        alpha: Option<f64>,
        adjuster: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let adj = parse_adjuster(adjuster)?;
        let (max_log, default_alpha) = match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => (pst.max_log_m(), pst.alpha()),
            MartingaleKind::OneSidedNormal(pst) => (pst.max_log_m(), pst.alpha()),
        };
        let alpha = alpha.unwrap_or(default_alpha);
        let result = adjusted::adjusted_e_holm(max_log, alpha, adj);

        let dict = PyDict::new_bound(py);
        dict.set_item("rejected", PyArray1::from_slice_bound(py, &result.rejected))?;
        dict.set_item("n_rejected", result.n_rejected)?;
        Ok(dict)
    }

    /// Number of tests.
    #[getter]
    fn n_tests(&self) -> usize {
        match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.n_tests(),
            MartingaleKind::OneSidedNormal(pst) => pst.n_tests(),
        }
    }

    /// Current time step.
    #[getter]
    fn time_step(&self) -> u64 {
        match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.time_step(),
            MartingaleKind::OneSidedNormal(pst) => pst.time_step(),
        }
    }

    /// Significance level.
    #[getter]
    fn alpha(&self) -> f64 {
        match &self.inner {
            MartingaleKind::TwoSidedNormal(pst) => pst.alpha(),
            MartingaleKind::OneSidedNormal(pst) => pst.alpha(),
        }
    }
}

/// Parse a Python object as either a scalar (broadcast to n) or a numpy/list array.
fn parse_float_input(
    _py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    expected_len: usize,
    _name: &str,
) -> PyResult<Vec<f64>> {
    if let Ok(scalar) = obj.extract::<f64>() {
        return Ok(vec![scalar; expected_len]);
    }

    let arr: Vec<f64> = obj.extract()?;
    if arr.len() != expected_len {
        return Err(EngineError::DimensionMismatch {
            expected: expected_len,
            got: arr.len(),
        }
        .into());
    }
    Ok(arr)
}

/// Parse adjuster type from Python string.
fn parse_adjuster(name: &str) -> PyResult<AdjusterType> {
    match name {
        "lookback" => Ok(AdjusterType::Lookback),
        "sqrt" => Ok(AdjusterType::Sqrt),
        other => Err(EngineError::InvalidParameter(format!(
            "Unknown adjuster: '{}'. Supported: 'lookback', 'sqrt'",
            other
        ))
        .into()),
    }
}

/// Register the ParallelSequentialTest class and related items into the _rust module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyParallelSequentialTest>()?;
    Ok(())
}
