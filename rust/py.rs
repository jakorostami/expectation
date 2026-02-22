//! PyO3 bindings for the ParallelSequentialTest engine.
//!
//! Exposes `PyParallelSequentialTest` to Python with numpy array I/O.
//! Uses enum dispatch at the Python boundary (one match per call, not per test)
//! while the inner `ParallelSequentialTest<M>` remains fully monomorphized.

use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::EngineError;
use crate::martingale::{OneSidedNormalMixture, TwoSidedNormalMixture};
use crate::multiple_testing::{bh, bonferroni, holm};
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
    #[new]
    #[pyo3(signature = (
        n_tests, null_values, alpha, martingale_type, v_opt, alpha_opt,
        variance=None, min_samples=30, alternative="two_sided",
        combiner="all_in", conservative_lambda=0.5, gamma=0.5, epsilon=1e-6
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

        // Auto-select martingale based on type and alternative
        match martingale_type {
            "two_sided_normal" => {
                let m = TwoSidedNormalMixture::new(v_opt, alpha_opt)
                    .map_err(|e| Into::<PyErr>::into(e))?;
                let pst = ParallelSequentialTest::new(
                    n_tests, null_vec, alpha, variance_config, comb, alt, m,
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
                    n_tests, null_vec, alpha, variance_config, comb, alt, m,
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

    /// Apply e-Bonferroni correction for FWER control.
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

/// Register the ParallelSequentialTest class and related items into the _rust module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyParallelSequentialTest>()?;
    Ok(())
}
