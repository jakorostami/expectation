# AGENTS.md

## Current Stabilization State (2026-09-07)

The full-source statistical audit is complete at revision `6f1ef40a1a0661ded83c30ffa7b7f53b00a93804` (version 0.6.1); **implementation repairs have not started**. Current work is documentation synchronization, not a claim that the library is statistically stabilized.

- **`AUDIT_REPAIR_PRIORITIES.md`** governs adjudication, unresolved decisions and the ranked repair order. Start with **P03 (owned segment partitions), LIFE-01 (complete reset), then SEQ-03 (signed/reflected Bernoulli evidence)**.
- **`POSSIBLE_BUGS.md`** is the corrected evidence catalog; **`PROPOSED_SOLUTIONS.md`** is a candidate repair catalog, not blanket design approval.
- **`docs/superpowers/plans/2026-09-01-expectation-executive-stabilization.md`** is a historical program catalog. Its old task numbering, registry-first ordering and proposed abstractions do not override the ranked audit.
- Review all sides of a mathematical contract: null class, filtration, clock, dependence, numerical domain and downstream use. A confirmed bug does not automatically validate its proposed repair. Full PDF source text, including proofs and appendices, must be read for this stabilization's paper audits; selected passages alone previously missed important context.
- Preserve existing Python-first/OOP ownership and the Rust performance architecture. Candidates, revision tokens, linked histories and a new contract registry are optional designs, not prerequisites. P07's numerical enclosure remains unverified; SEQ-02 sampling and shared genuine-zero/infinity semantics still require decisions.
- The paths and flows below describe the current implementation, not uniform validity or production-readiness guarantees. `tasks/todo.md` (plan + review of the 2026-09-11 protocol/upperexp/merging/stein additions) and `tasks/lessons.md` exist; the audit repairs above remain unstarted.

## Build and Test Commands

### Python
```bash
pip install -e ".[dev]"               # editable install + dev deps (pytest, maturin, black, isort, mypy)
                                       # runtime deps (numpy, scipy, pydantic, pandas, plotly, matplotlib) come from pyproject.toml
pytest tests/                         # all tests
pytest tests/modules/test_martingales.py  # single file
pytest tests/ -k "test_name"          # single test by name
pytest -m "not performance"           # exclude timing-sensitive tests (as CI does)
```

### Rust (PyO3 bindings)
```bash
maturin develop                       # dev build, installs wheel into active venv
maturin develop --release             # optimized build
cargo check                           # fast compile check (no linking)
```
`cargo test` and `cargo clippy` link successfully because `extension-module` is intentionally NOT a Cargo.toml default feature (maturin injects it via `[tool.maturin]`). The CI `rust` job runs `cargo fmt --all -- --check` (advisory), `cargo clippy --locked --all-targets -- -D warnings`, and `cargo test --locked`. Use `maturin develop --release --locked` (inside a venv) for the actual Python extension build. Rust integration tests also run via `pytest tests/rust/`.

### Linting
Config lives in `pyproject.toml`: `[tool.black]` (line-length 100), `[tool.isort]` (profile black, line 100), `[tool.mypy]`. CI runs `black --check` and `isort --check-only` repo-wide as **blocking** gates, `mypy` as **advisory** (non-blocking — the numpy-heavy code has ~100 pre-existing type notes), and `cargo fmt --check` as advisory. No pre-commit config exists yet.

### CI
- `.github/workflows/ci.yml` -- runs on PRs (and pushes to `develop`/`main`), skipping docs-only changes via `paths-ignore` (`**.md`, `**.ipynb`, `docs/**`, `assets/**`, `LICENSE`). Jobs: version-check (`scripts/check_version.sh`), lint, rust (fmt/clippy/test), test matrix (ubuntu/macos/windows x Python 3.12/3.13), min-deps (floor versions). Builds create a `.venv` then `maturin develop --release --locked`.
- `.github/workflows/bump.yml` -- dispatch-only version bump via `scripts/bump_version.sh` (patch/minor/major or explicit X.Y.Z); a release-PR variant is planned for protected `main`/`develop`.
- `.github/workflows/release.yml` -- tag/dispatch-driven: test gate, build abi3 wheels (linux x86_64, macOS universal2, windows x64) + sdist, verify artifacts, publish to PyPI via trusted publishing, create GitHub Release.
- Legacy `run-tests.yml` and the original `release.yml` are archived under `.github/workflows-archive/`.

## Architecture

Python library for sequential hypothesis testing using e-values and e-processes (game-theoretic statistics), with a Rust parallel engine via PyO3/maturin for massively parallel testing. Existing benchmark figures are historical; the statistical audit used an unoptimized isolated native build and did not establish performance.

### Three entry points

1. **`SequentialTesting`** (`expectation/seqtest/sequential_e_testing.py`) -- Single-stream sequential testing. Orchestrates martingales, e-process updater, boundaries, calibrators, and e-power. Test types: MEAN, QUANTILE, VARIANCE, PROPORTION. Each test type creates an `e_calculator` closure (not a subclass) via `_setup_*_test()` methods that captures and mutates `self` state.

2. **`ParallelSequentialTest`** (`expectation/par_seqtest.py`) -- Python orchestrator with frozen Pydantic configuration/results around the Rust engine (`expectation._rust.PyParallelSequentialTest`). Defines its own enums (`MartingaleType`, `AlternativeDirection`, `CombinerStrategy`, `VarianceMode`, `MultipleTestingMethod`, `MergingMethod`) and uses string dispatch at the boundary. Exposes snapshot and adjusted running-maximum multiple testing, plus separate global intersection merging. The method named `e_holm` currently implements reciprocal p-Holm, not literature mean-closure e-Holm. `variance_mode` is not authoritative: the supplied `variance` value currently determines the native branch.

3. **`KSampleSequentialTest`** (`expectation/ksample/ksample_test.py`) -- Multi-group (k >= 2) sequential homogeneity test for Bernoulli data. Uses RIPr (Reverse Information Projection) e-variables from Turner, Ly & Grunwald (2022). Three alternative types: unrestricted (Proposition 2, closed-form Beta posterior), effect-size restricted (Appendix S1, grid-based discretized posterior, k=2 only), and simple (Eq. 3.2, fixed theta). Reuses `EProcessUpdater` and `EProcess` for temporal combination. Does **not** use the `MixtureSupermartingale(s, v)` interface -- the RIPr construction computes per-step Bernoulli log-likelihood ratios directly.

### Python package (`expectation/`)

Package `__init__.py` files have no public re-exports (some contain license headers). Users must import from full module paths (e.g., `from expectation.modules.martingales import TwoSidedNormalMixture`).

Core building blocks in `modules/`:

- **martingales.py** -- ABC `MixtureSupermartingale` with subclasses: `TwoSidedNormalMixture`, `OneSidedNormalMixture`, `GammaExponentialMixture`, `GammaPoissonMixture`, `BetaBinomialMixture`. Also ABC `SequentialEValueCombiner` with subclasses: `AllInCombiner`, `ConservativeCombiner`, `EmpiricallyAdaptiveCombiner`, `LogOptimalCombiner`.
- **hypothesistesting.py** -- Pydantic models (`Hypothesis`, `EValueConfig`, `EProcessConfig`, `EProcess`), the `EValue` ABC with `LikelihoodRatioEValue` and `UniversalEValue`, and `SymmetryETest`. Config models here are frozen; `EProcess` is a mutable streaming accumulator. This does not mean every existing wrapper config/result elsewhere is already immutable.
- **eprocessupdater.py** -- `EProcessUpdater` applies combiner strategy to update running e-process. Factory method `_create_combiner()` selects strategy from `BettingStrategy` enum.
- **merging.py** -- Five merging implementations from Vovk & Wang (2024): `ArithmeticMeanMerger`, `UStatisticMerger` (ESP recurrence), `LambdaProductMerger`, `SegmentProductMerger`, `ProductMerger`, under `EValueMerger`. Each implements `merge()` and `gambling_system()` (Eq. 4 V&W 2024), with `create_merger()` and convenience functions. Arithmetic mean permits arbitrary dependence; product-like methods require their independent/sequential contracts. Some existing book theorem labels are wrong; use the audited source references rather than copying them.
- **adjusters.py** -- Lookback and square-root adjusters, including log-valued interfaces, for running-maximum adjustment. Rust mirrors these in `rust/adjusters/`. Current overflow and endpoint qualifications are covered by P05.
- **calibrators.py** -- `EToPCalibrator` and `PToECalibrator` (5 calibration types).
- **boundaries.py** -- Functional API: `normal_mixture_bound()`, `beta_binomial_mixture_bound()`, `poly_stitching_bound()`, `bernoulli_confidence_interval()`, etc.
- **epower.py** -- `EPowerCalculator`, `EPowerConfig`, `EPowerResult`.
- **orderstatistics.py** -- `OrderStatisticInterface` ABC with `StaticOrderStatistics`.
- **quantiletest.py** -- `QuantileABTest` for quantile-based sequential testing.
- **protocol.py** -- Forecaster-Skeptic-Reality layer (Shafer & Vovk 2019 Ch. 1, 8; Shafer 2021). `SkepticStrategy` ABC with a two-phase contract: `bet(x, announcement)` is pure and F_{t-1}-measurable, `observe(x, announcement)` commits state. `BettingProtocol` validates the emitted e-value (finite, >= 0) before touching the `EProcess`, delegates temporal betting to `EProcessUpdater` (the existing combiners are the betting rules), applies Ville, computes the anytime p-value via `EToPCalibrator`, keeps history. Mirrors `KSampleSequentialTest` once; new strategies implement this ABC instead of copying the orchestration.
- **merging.py** also holds `SparseMixtureMerger`: the anytime-valid sparse-anomaly mixture of Pérez-Ortiz, Castro & Stoepker (2025, arXiv:2506.22588; Eq. (4), (6), grid (9), Thms 2.1-2.7) as an `EValueMerger`. `merge`/`merge_log` take the *current* values of K independent per-stream test supermartingales and return a global e-process value (do not accumulate temporally again); single-eps output equals `LambdaProductMerger` exactly, and `gambling_system` is the posterior-mean sparsity, so the Vovk-Wang Eq. (4) reconstruction reproduces the merge (tested). `diagnostics` gives posterior over sparsity, participation ratio and top streams; `detection_boundary` and `sparsity_grid` are the Donoho-Jin function and the PCS grid. **Contract:** independent streams (Wang, Dandapanthula & Ramdas 2025); otherwise use `ArithmeticMeanMerger`.

Other subpackages:
- `ksample/` -- Bernoulli k-sample sequential e-testing (Turner, Ly & Grunwald 2022). `config.py`: enums (`KSampleAlternativeType`, `DivergenceType`), frozen Pydantic models (`KSampleConfig`, `KSampleStepResult`). `bernoulli.py`: unrestricted Beta-posterior, effect-size restricted grid (k=2) and fixed-theta paths. Scoring/grid weights use logs, but `ksample_test.py` exponentiates before shared temporal accumulation. The restricted calculator mutates grid weights after scoring but before the updater succeeds, which causes RIPr-02's retry defect. `update(group_data)`, buffered `update_single()`, history/summary/reset are provided. Default prior gamma=.18 is an experimental recommendation for the source's particular setting, not a universal optimum; it is currently also passed as the betting cap.
- `confseq/` -- Config in `confidenceconfig.py` declares 5 boundary types and 4 estimands, but the base update always computes a gamma-exponential mean interval. `ConfidenceSequence` and `EmpiricalBernsteinConfidenceSequence` have the variance-process/units defects in CS-02/CS-03; enum labels do not establish implemented constructions.
- `conformal/` -- **Experimental** (emits `UserWarning`). `ConformalEValue`, `ConformalEPseudomartingale`, `TruncatedEPseudomartingale` (Vovk et al. 2024), `ConformalCUSUM` with Shiryaev-Roberts variant, `EfficiencyAnalyzer` (Monte Carlo), `AdaptiveThresholdHandler`.
- `parametric/` -- Wang & Ramdas t-test kernels (Thms. 4.9/4.7) and incomplete factories/CS. `TestType.TTEST` is missing; `TtestConfidenceSequence` also fails positional Pydantic initialization. Kernels/radii have independent mathematical defects and reduced-filtration restrictions. Do not enable the factories before repairing the underlying paths.
- `upperexp/` -- Upper-expectation (convex-set) nulls on finite outcome spaces. `nulls.py`: `UpperExpectationNull` ABC (`support_function` exact, `certify` = divide by `max(1, sigma)`, `worst_case_member`, `optimal_e_variable`); `ContaminationNull` (closed-form support; Huber-Strassen clipped LR via a 1-D KKT search, verified against a generic convex programme), `WassersteinNull` (transport LPs for support and distance, dual LP for the Kantorovich potential, witness `1 + lam[(h - <p0,h>) - rho - viol]` with the LP's Lipschitz violation paid as tax, so validity is exact; attains the GRO value to solver precision on small spaces), `PolytopeNull` (vertex maximum; RIPr by Csiszar-Tusnady EM; `q/p*` certified), `CHSHLocalRealismNull(PolytopeNull)` (16 local-deterministic vertices, `singlet_distribution`, `chsh_value`). `skeptic.py`: `UpperExpectationConfig`, `UpperExpectationSkeptic` (KT-smoothed predictable belief or fixed alternative; per-round announcements allow time-varying sets such as bid-ask spreads; valid under any adapted choice of a set member each round), `WassersteinDistanceSequence` (grid of radii sharing one potential, one `EProcess` per radius driven by `EProcessUpdater`; smallest non-rejected radius is an anytime-valid lower bound on `W1(Reality, P0)` up to one grid step). Sources: Larsson, Ramdas & Ruf (2025) numeraire Def. 2.1 / Thm 2.6 / Thm 3.6 and constraint-generated hypotheses; Grünwald, de Heide & Koolen (2024) Thm 1; Huber & Strassen (1973); Villani (2009) Ch. 6; Zhang, Glancy & Knill (2011); van Dam, Gill & Grünwald (2005). Cost per round: contamination O(K) root-finds, Wasserstein one LP with K^2 variables, polytope one EM; intended for small K.
- `stein/` -- Sequential kernelized Stein discrepancy (Martinez-Taboada & Ramdas, AISTATS 2025; Eq. (1), (5), Thm 1, Thm 2, Sec. 5.1) as a `SkepticStrategy`. `kernels.py`: `SteinKernel` ABC for radial kernels (`stein_kernel`, `lower_bound`, abstract `psi`, `dpsi_over_u`, `d2psi`, `bound_constants`); `IMQSteinKernel` (global Lipschitz constant of the score needed), `WendlandSteinKernel` (compact support: Stein identity holds for every C^1 positive density with no moment condition; only a local Lipschitz constant on the support ball is needed; constants derived in the module docstring and checked against grid maxima). `skeptic.py`: `ScoreTarget` ABC (`_score`, `lipschitz_constants`), `CallableScoreTarget`, `GaussianTarget`, `BoltzmannTarget`, and `SteinSkeptic` emitting `E_t = 1 + sum_i h_p(X_i, x)/sum_i M_p(X_i) >= 0` (SKSD Eq. 5) so the library's combiners bet `1 + lambda g_t`; `EMPIRICALLY_ADAPTIVE` with `gamma=1` is the GRAPA-type rule. Null is the conditional law given the past; autocorrelated MCMC output violates it. Compact support forgoes IMQ's tail sensitivity (Gorham & Mackey 2017).
- `utils/helper_functions.py` -- Plotly visualization (not a proper subpackage, missing `__init__.py`).

### Rust layer (`rust/`)

Single crate, source rooted at `rust/lib.rs`. PyO3 extension module exposed as `expectation._rust`. The binding uses ABI3 for Python 3.10+, while package metadata requires Python 3.12+. Dependencies include pyo3, numpy, ndarray, rayon and thiserror. Release profile: opt-level 3, thin LTO, codegen-units 1, strip true.

**Module structure:**
- `lib.rs` -- `#[pymodule]` entry point
- `py.rs` -- `PyParallelSequentialTest` `#[pyclass]` with `MartingaleKind` enum dispatch at the Python boundary. ~22 constructor params including merge configuration. Returns `PyDict` from `step()` (Python wrapper constructs frozen Pydantic `StepResult`).
- `error.rs` -- `EngineError` with `thiserror` derive, converts to `PyValueError`
- `math/erfc.rs` -- `erfc`, `ndtr`, `log_ndtr` from first principles (fdlibm/Cody 1969), zero external math deps
- `martingale/` -- `MixtureSuperMartingale` trait (`Send + Sync`) with `TwoSidedNormalMixture` and `OneSidedNormalMixture`. Both are `Copy + Clone + Send + Sync` value types. All methods `#[inline(always)]` for inlining into rayon loops.
- `par_seqtest/` -- `ParallelSequentialTest<M>` generic engine. `state.rs`: SoA layout (13 parallel `Vec`s, ~28 MB at 300K tests, fits L3 cache). `update.rs`: 3 combiner functions x 3 variance branches = 9 specializations, hoisted outside the rayon hot loop. Two `zip_state!` macros zip SoA into parallel iterators.
- `multiple_testing/` -- `e_bonferroni` (FWER), `e_bh` (FDR step-up with `par_sort_unstable`, does NOT early-break), and the currently named `e_holm` (reciprocal p-Holm step-down, early breaks). `adjusted.rs` applies corresponding procedures to adjusted running maxima. All depend on valid input evidence and the stated joint filtration; naming does not confer validity.
- `adjusters/` -- Lookback/square-root adjustment kernels used by adjusted multiple testing.
- `merge/` -- Spatial merging of K per-step e-values into a single merged e-value, then temporal accumulation into an e-process for the **intersection hypothesis** ("at least one null is false"). `MergeFunction` enum (5 variants matching Python), `MergeCombinerType` enum (AllIn, Conservative, EmpiricallyAdaptive), `MergeConfig`, `MergeState` (9 scalar fields, allocated only when merge is configured). `apply_merge()` called in `step()` after `step_parallel()` completes. Ref: Ramdas & Wang (2025) Ch. 8.
- `tests/` -- Native unit-test modules, including merge and adjusted multiple testing; additional inline Rust tests also exist.

**Key design: monomorphization over vtables.** `ParallelSequentialTest<M>` is generic over `M: MixtureSuperMartingale`. The compiler monomorphizes separately for each martingale type, so `log_super_mg` calls inside the rayon hot loop are fully inlined (~2.9 ns/call in release). The `MartingaleKind` enum in `py.rs` dispatches once per Python call, not once per test.

### Data flow

**Single-stream** (`SequentialTesting.update(observations)`):
```
observations -> e_calculator closure (accumulates running sums, computes centered sum s and intrinsic time v)
  -> mixture.log_superMG(s, v) -> sequential e-value (exp(log_e_cum - prev_log_e_cum))
  -> EProcessUpdater.update() (combiner strategy -> increment -> running product)
  -> boundary check (Ville's inequality) -> SequentialTestResult
```

**Parallel** (`ParallelSequentialTest.step(observations)`):
```
observations[n_tests] -> PyO3 boundary -> rayon par_iter_mut over SoA state
  -> per-test: centered sum -> log_super_mg -> combiner -> e-process update
  -> (if merge configured) apply_merge: spatial merge of K e-values -> temporal accumulation
  -> StepResult (frozen Pydantic, includes optional merged fields)
Then: .e_bh() / .e_bonferroni() / .e_holm() for cross-test error control
```

**Merge pipeline** (intersection hypothesis):
```
per-test log_e_sequential[K] -> spatial merge (arithmetic_mean | u_statistic | lambda_product | segment_product | product)
  -> merged_e_value -> temporal combiner (all_in | conservative | adaptive)
  -> merged_e_process -> Ville rejection check
```

**K-sample** (`KSampleSequentialTest.update(group_data)`):
```
group_data {g: binary_array} -> validate binary, all k groups present
  -> BernoulliRIPrCalculator.compute_log_e_value() (posterior means F_{j-1}-measurable)
     -> unrestricted: Beta(gamma,gamma) posterior per group -> theta_g_hat, theta_0_hat (RIPr)
     -> restricted:   grid posterior softmax -> theta_a_hat, theta_b_hat, theta_0_hat
     -> simple:       fixed theta -> theta_0_hat
  -> Eq. 3.4: sum_g [s_g*log(theta_g/theta_0) + (n_g-s_g)*log((1-theta_g)/(1-theta_0))]
     (restricted path mutates grid weights inside compute_log_e_value after scoring)
  -> exp() -> e_value -> EProcessUpdater.update(e_process, e_value)
  -> update cumulative successes/counts (AFTER updater)
  -> Ville's inequality: max(M_1,...,M_j) >= 1/alpha -> KSampleStepResult
```

### Test structure

Tests mirror source layout: `tests/modules/`, `tests/confseq/`, `tests/parametric/`, `tests/seqtest/`, `tests/ksample/`, `tests/rust/`, `tests/upperexp/`, `tests/stein/`, plus merge integration tests. The `upperexp`, `stein`, sparse-mixture (in `test_merging.py`) and `protocol` suites are exact where the object is finite (support functions vs brute force, Huber-Strassen structure, RIPr vs SLSQP, Tsirelson's bound, Stein identity by quadrature, kernel constants vs grid maxima, single-eps equality with `LambdaProductMerger`) and Monte Carlo with binomial tolerances otherwise (adaptive-adversary supermartingale checks, time-uniform type-I, PCS detection moment). Golden fixtures exercise Python-Rust numerical equivalence. Existing tests do not cover every audited witness: some conformal assertions encode the incorrect capital floor/CUSUM recurrence, parametric CS tests are skipped, and fixed-time CS coverage is not simultaneous coverage. Use source-derived regressions, not existing green results alone, to establish a repair.

## Key Conventions

- Pydantic v2 models with `frozen=True` for all config/hypothesis objects. Use Pydantic, not dataclass.
- `str, Enum` pattern for all enums (e.g., `BettingStrategy`, `TestType`, `AlternativeType`)
- NumPy-style docstrings
- Type hints are mandatory
- Main branch is `develop`
- Numerical precision: `log1p`-based stable computations can differ from naive `np.log` at the ~15th digit; use tolerance 1e-13 for golden tests, not tighter
- Golden fixture JSON files use `null` (not `NaN`) -- serde_json rejects `NaN` literals. Use `Option<f64>` in Rust, convert to `f64::NAN` at test time.
- Dependencies: NumPy, SciPy, Pydantic for core. Pandas supports history/visualization, including module-level imports. Plotly is isolated in `utils/`; the production Rust import is in `par_seqtest.py`. Code supports numpy 1.26+ **and** numpy 2.x (use `np.float64`, not the removed `np.float_`).
- `pyproject.toml` is the single source of truth for dependencies (runtime + `[project.optional-dependencies].dev`); `requirements.txt` has been removed. Lower bounds (numpy>=1.26, scipy>=1.12, pydantic>=2.5, pandas>=2.2, plotly>=5.18, matplotlib>=3.8) are verified by the `min-deps` CI job.
- Version consistency across `pyproject.toml`, `Cargo.toml`, `Cargo.lock`, `CITATION.cff`, and `README.md` is enforced by `scripts/check_version.sh`; bump all of them via `scripts/bump_version.sh`. `Cargo.lock` IS committed.

## Python is the language, Rust for specifics

The codebase is based on Python. It uses Rust for specifics and not for generals. Python implementations come first; Rust mirrors them for performance in the parallel engine. Cross-validate Rust against Python at 1e-13 tolerance.

## Principles to follow

- Architectural design and pattern: follow existing codebase patterns
- Robust software engineering with test driven development
- Code patterns: Pydantic (not dataclass), OOP with ABCs, modular and portable code
- Minimal dependencies: Python native, NumPy and SciPy oriented. Develop from first principles instead of adding libraries.
- Always reference the paper and which section in the paper when developing
- Mathematical constructions follow cutting-edge research in game-theoretic probability, e-values, e-statistics, e-processes. Do not make things up -- read papers and design accordingly.

## Key researchers

The library is built on the work of: Aaditya Ramdas, Vladimir Vovk, Peter Grunwald, Glenn Shafer, Alexander Ly, Ruodu Wang, Hongjian Wang, Ian Waudby-Smith, Wouter M. Koolen, Ben Chugg, Rosanne Turner.

## Roadmap

Products to be built on this library:
- Competitor site to Polymarket and Kalshi
- A/B testing platform (e-value framework) to compete with StatSig and Optimizely
- Quant Trading software
- Physics software
- Clinical experiments software

## Side notes

When you aren't confident, ask the user instead of assuming. Be explicit in your reasoning, plan ahead. Do not look for papers unless asked by the user or needed for the work; in the latter case ask whether they can provide the source or approve retrieving the cited edition. For statistical audits, read the complete PDF source text, including appendices and qualifications, rather than selected passages. Local `hypotevalues.pdf` has 264 PDF pages and differs from the downloaded arXiv v6 edition; consult the audit's edition record. Use only explicitly authorized principal-prefix reviewers, with no further delegation; the parent owns synthesis.

- Do not look in other branches unless the user explicitly says to
- Do not commit .env files
- Do not use git to commit anything unless asked by the user
- Never use basic exploration agents, always use principal prefix named agents


---

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - dont keep pushing
- Use plan mode for verification steps and not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use principal prefix named subagents to keep main context window clean
- Never use basic exploration agents
- Offload research, exploration, and parallel analysis to principal prefix named agents
- For complex problems, throw more compute at it via principal prefix named agents
- One tack per subagent for focused execution
- When schemas and routes are built by different agents, ALWAYS run a reconciliation pass after

### 3. Self-Improvement Loop
- After ANY correction from the user: update "tasks/lessons.md" with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behaviour between main and your changes when relevant
- Ask yourself: "Would a principal staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - dont over-engineer
- Challenge your own work before presenting it
- Self critique, self evaluate, self review when needed to maintain balance

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Dont ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to "tasks/todo.md" with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High level summary at each step
5. **Document Results**: Add review section to "tasks/todo.md"
6. **Capture Lessons**: Update "tasks/lessons.md" after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Avoid Reactive Changes**: Plan ahead instead of reactively change on the spot. Understand what is at hand.
- **Minimal Impact**: Changes should only touch what's necessary while connecting the dots. Avoid introducing bugs.
