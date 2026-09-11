# Expectation Statistical Stabilization: Historical Program Catalog

> **Status synchronized 2026-09-07:** this September 1 plan is historical context, not the current executable plan. The completed full-source audit in [AUDIT_REPAIR_PRIORITIES.md](../../../AUDIT_REPAIR_PRIORITIES.md) governs adjudication, unresolved decisions and repair order. Statistical implementation repairs have not started. Historical task numbers, checkboxes, file-creation proposals and commit examples below are not current completion status or authorization to execute them.

The corrected evidence and candidate repairs are in [POSSIBLE_BUGS.md](../../../POSSIBLE_BUGS.md) and [PROPOSED_SOLUTIONS.md](../../../PROPOSED_SOLUTIONS.md). Confirm a finding separately from approving its proposed design. New registries, public candidate/revision protocols, linked histories and persistence are not automatic prerequisites.

**Goal:** Convert `expectation` from a broad research prototype into a statistically explicit, failure-safe, installable library whose public e-value, e-process, confidence-sequence, merging, and multiple-testing claims are supported by their implementations and tests.

**Architecture:** Preserve Python as the statistical reference, existing OOP ownership/calculator closures, and Rust as a performance mirror. Define the null class, filtration, clock, dependence and numerical domain for the selected repair. Implement complete bounded corrections in the ranked audit's order rather than requiring a repository-wide contract framework first.

**Tech Stack:** Python 3.12+, NumPy, SciPy, Pydantic v2, Pandas, Plotly, Rust 2021, PyO3, Rayon, maturin, pytest, GitHub Actions. The binding's ABI3 target starts at Python 3.10; package metadata still requires 3.12+.

## Global Constraints

- Mathematical validity takes priority over backward compatibility and feature breadth.
- Python implementations are the statistical reference; Rust mirrors selected kernels for performance.
- Every public sequential method must document its null class, alternative, filtration, predictable inputs, clock, dependence assumptions, stopping guarantee, and source theorem.
- Pydantic models, not dataclasses, remain the configuration and result contract.
- Configuration models remain frozen; mutable state must be isolated from immutable result snapshots.
- Type hints and NumPy-style docstrings remain mandatory.
- No new third-party statistical dependency is introduced without a demonstrated need.
- Use log-space evidence internally wherever natural-scale overflow can affect inference.
- Cross-language numerical parity uses tolerance `1e-13` unless a documented numerical kernel requires another tolerance.
- The main integration branch is `develop`; do not switch branches or rewrite history during implementation.
- Do not commit `.env` files, generated native binaries, Python bytecode, pytest caches, or Rust `target/` artifacts.
- Do not claim confidence-sequence coverage from fixed-time coverage tests.
- Do not claim arbitrary-dependence validity for product-like merging functions.
- Do not silently replace invalid or non-finite evidence with success-shaped fallback values.
- Scope only the selected complete repair and obtain approval for material public-contract choices. Do not generate a separate planning/documentation framework for every small correction.

---

## 1. Executive Decision

The current repository is a substantial research codebase, not a release-ready statistical product. Its strongest paths are:

1. Known-variance-proxy normal-mixture testing in the Rust parallel engine.
2. The blockwise unrestricted Bernoulli k-sample RIPr construction.
3. Snapshot e-BH, e-Bonferroni, and the currently named `e_holm` (reciprocal p-Holm) when their inputs are valid e-values.
4. Adjusted running-maximum multiple testing when the base processes are valid and share the required filtration.
5. Arithmetic-mean merging under arbitrary dependence.

The following paths must not retain production validity claims in their current form:

1. Default empirical-variance mean testing.
2. Variance testing and variance confidence intervals.
3. The current sequential quantile workflow.
4. Generic and empirical-Bernstein confidence-sequence facades.
5. Parametric t-test factories and t-confidence sequences.
6. Conformal mixture mode, CUSUM, and Shiryaev-Roberts behavior.
7. Product, U-statistic, lambda-product, and segment-product global merging without an explicit conditional-independence or sequential-order contract.

## 2. Current Execution Order

The original registry-first sequence is superseded by [the ranked repair order](../../../AUDIT_REPAIR_PRIORITIES.md#3-ranked-repair-order). The first units are:

| Rank | Repair | Boundary of completion |
|---|---|---|
| 1 | P03: owned segment partitions | Validate at Python, PyO3 and direct Rust entry points before mutation; caller changes cannot invalidate the partition. |
| 2 | LIFE-01: complete reset | A reset run matches a fresh configured experiment, including evidence normalization, tuning and strategy state. |
| 3 | SEQ-03: signed/reflected Bernoulli evidence | Repair two-sided signs and LESS support reflection, with asymmetric-null and public-update regressions. |

Continue with the complete units and dependencies in that report, not the task numbering below. Do not keep finding cosmetic/easy slices instead of addressing the core mean, state and numerical failures. Execution order is not severity order: unresolved invalid defaults remain barriers to production-validity claims.

## 3. Repository Snapshot

**Current snapshot, 2026-09-07:**

- Branch: `fix/polishing`; revision `6f1ef40a1a0661ded83c30ffa7b7f53b00a93804`.
- Version: `0.6.1`; package Python floor 3.12.
- Full-source audit: all 43 retained IDs adjudicated; no statistical implementation repairs yet.
- Dependencies, version consistency, Python/OS matrices and Rust test/linking configuration are present; see section 10.3. This is not certification of a live release.
- The current changes synchronize documentation. Inspect actual worktree status before implementation; do not restore or recreate files from the historical list below.

**Original assessment baseline (historical, not current):**

- Branch: `feature/k-sample`
- Version: `0.6.0` (was 0.5.2 during this .md creation)
- Python package: `expectation/`
- Rust crate root: `rust/lib.rs`
- Build backend: maturin
- CI: Ubuntu, Python 3.11, `maturin develop --release`, `pytest tests/`

Pre-existing working-tree changes at that earlier assessment:

```text
 M expectation/seqtest/sequential_e_testing.py
 M expectation/utils/helper_functions.py
?? FIX_MULTIPLETESTING.md
?? examples/ksample-bernoulli.ipynb
?? expectation/ksample/ksample_test.py
?? tests/ksample/
```

The earlier review did not modify those files. This list is retained only as history; it is not the current working-tree inventory or an instruction to manipulate those paths.

## 4. Current Product Architecture

| Layer | Responsibility | Main files | Current assessment |
|---|---|---|---|
| Statistical contracts | Hypotheses, configs, e-value results, process state | `expectation/modules/hypothesistesting.py` | Useful foundation; result and state semantics diverge elsewhere |
| Statistical kernels | Mixture supermartingales, boundaries, combiners | `expectation/modules/martingales.py`, `expectation/modules/boundaries.py` | Strong formulas; assumptions are not encoded |
| Temporal composition | Predictable betting and e-process updates | `expectation/modules/eprocessupdater.py` | Mathematical product requires conditional e-values; current admission, state and numerical defects remain |
| Single-stream orchestration | Mean, proportion, variance, quantile, intervals, history | `expectation/seqtest/sequential_e_testing.py` | Overloaded and not production-ready |
| Parallel acceleration | Python facade, PyO3 dispatch, Rayon SoA engine | `expectation/par_seqtest.py`, `rust/py.rs`, `rust/par_seqtest/` | Strong systems design; weak domain enforcement |
| Spatial evidence | Merging and intersection testing | `expectation/modules/merging.py`, `rust/merge/` | Retain valid source formulas under their dependence contracts; partition and numerical defects remain |
| Multiple testing | Snapshot and carefree procedures | `rust/multiple_testing/`, `rust/adjusters/` | Structurally strong conditional on valid base processes |
| K-sample | Bernoulli RIPr calculation and orchestration | `expectation/ksample/` | Most promising Python-native subsystem |
| Confidence sequences | Mean/estimand facade | `expectation/confseq/` | Exposed configuration does not control implementation |
| Conformal | Experimental e-values and change detection | `expectation/conformal/` | Prototype mechanics, not established inference |
| Parametric | Universal and scale-invariant t methods | `expectation/parametric/` | Broken high-level entry points |
| Presentation | Pandas history and Plotly dashboards | `expectation/utils/helper_functions.py` | Coupled to one result schema and can disagree with decisions |
| Delivery | Packaging, CI, examples, metadata | `pyproject.toml`, `Cargo.toml`, `.github/` | Earlier metadata/CI gaps largely addressed in configuration; statistical readiness is not established |

## 5. Current Execution Flows

### 5.1 Single-stream flow

```text
observations in one update call
  -> test-specific closure mutates sufficient statistics
  -> evaluate cumulative log mixture M_t(S_t, V_t)
  -> derive log sequential e-value:
       log E_t = log M_t - log M_previous
  -> EProcessUpdater chooses predictable lambda_t from prior e-values
  -> update product of (1 - lambda_t + lambda_t E_t)
  -> Ville decision from running maximum
  -> report current p-value, separately constructed interval, e-power, history
```

The observation count and process clock differ: `sample_size` counts observations, while stopping time and `EProcess.total_samples` count calls to `update()`.

### 5.2 Parallel flow

```text
one observation per test
  -> Pydantic config and NumPy conversion
  -> PyO3 string/enum dispatch
  -> Rayon loop over Structure-of-Arrays state
  -> per-test centered sum and intrinsic time
  -> mixture log M_t and sequential E_t
  -> per-test temporal combiner and Ville state
  -> optional spatial merge of current sequential e-values
  -> temporal merged process and intersection decision
  -> snapshot or adjusted multiple-testing query
```

The Rust hot path is architecturally strong:

- `ParallelSequentialTest<M>` is generic over the martingale type.
- Martingale calls are inlined into the Rayon loop.
- Martingale dispatch occurs once at the Python boundary.
- Variance and combiner branches are hoisted outside the per-test loop.
- Thirteen parallel vectors provide a cache-friendly state layout.

### 5.3 K-sample flow

```text
predictably designed block for every group
  -> validate group keys and binary observations
  -> compute posterior means from blocks 1 through j-1
  -> compute RIPr null projection weighted by current block sizes
  -> evaluate block Bernoulli log likelihood ratio
  -> update temporal e-process
  -> update posterior/cumulative state after evidence calculation
  -> report running-max p-process and Ville decision
```

This order correctly preserves the previous-block measurability requirement. The current API does not guarantee that block sizes were chosen before observing the block.

## 6. Mathematical Standard

For every null distribution `P`, an e-process `M_t` adapted to filtration `F_t` must satisfy

```text
E_P[M_tau] <= 1
```

for every permitted stopping time `tau`.

For sequential e-values, every increment must satisfy

```text
E_P[E_t | F_(t-1)] <= 1.
```

For a betting process

```text
M_t = product over s <= t of (1 - lambda_s + lambda_s E_s),
```

each `lambda_s` must be `F_(s-1)`-measurable and lie in `[0, 1]`.

A confidence sequence `C_t` must satisfy

```text
inf over P of P(theta(P) belongs to C_t for every t) >= 1 - alpha.
```

Checking coverage at one deterministic sample size is insufficient.

## 7. Statistical Validity Ledger

| Workflow | Required contract | Current verdict |
|---|---|---|
| `LikelihoodRatioEValue` | Correct normalized fixed densities and joint or i.i.d. factorization | Conventional fixed-batch e-value; domain validation is weak |
| `UniversalEValue` | Independent training/evaluation split; alternative fitted off evaluation data; null likelihood supremum on evaluation data | The implementation matches split universal inference; not automatically an online e-process |
| Mean with known proxy | Conditionally sub-Gaussian increments; known predictable variance proxy; fixed mixture | Theorem-faithful restricted path |
| Mean with empirical variance | Valid self-normalized process and fixed or properly mixed tuning | Invalid as currently composed |
| One-sided proportion | Conditional Bernoulli model and fixed beta-binomial mixture | Plausible restricted path; numerical extremes need hardening |
| Two-sided proportion | Signed beta-binomial process or another proved two-sided construction | Unverified for null probability other than 0.5 |
| Variance | Centered conditional sub-gamma or sub-exponential process | Invalid current construction |
| Sequential quantile | Bernoulli indicator process, correct direction, continuity/tie rule | Recoverable kernel, broken public implementation |
| K-sample unrestricted | Bernoulli groups, block sizes selected before block observations, previous-block posterior | Strongest Python-native path |
| K-sample restricted/simple | Same block contract plus correct prior/restriction | Promising, less comprehensively validated |
| Arithmetic-mean merge | Valid e-values; arbitrary cross-test dependence | Safe default |
| Product-like merge | Conditional independence or a sequential cross-test order | Valid only under unstated assumptions |
| Snapshot multiple testing | Valid e-values at a fixed or valid stopped snapshot; common filtration | Correct conditional procedure, not carefree |
| Adjusted running-max testing | Valid base e-processes, common filtration, admissible adjuster | Structurally strong; cannot repair invalid base processes |
| Generic confidence sequence | Correct boundary/estimand dispatch and simultaneous coverage | Not established |
| Empirical Bernstein CS | Bounded observations with correct normalization and EB process | Not correctly implemented |
| Conformal calibration | Exact exchangeability/sequential theorem for normalized ratios | Experimental and unresolved |
| Conformal CUSUM/SR | Correct restart/SR recurrence and false-alarm theorem | Broken prototype |
| Parametric t-test | Correct sample-indexed scale-invariant or universal martingale | Broken and unreachable |
| E-to-p conversion | Current `1/M_t` for snapshot; running-max inverse for anytime p-process | Semantics differ across products |
| P-to-e conversion | Input in `[0,1]`; endpoint-stable calibrator | Mixture endpoint and negative-input defects |
| E-power | Expected log evidence under an explicit alternative distribution | Current output is primarily a realized diagnostic |

## 8. Proof-Breaking Findings

This section retains the earlier assessment narrative. Use the corrected evidence catalog and ranked audit for final classifications and exact source references; a historical simulation or hypothesis here is not an additional independently confirmed finding.

### 8.1 Default mean testing

`expectation/seqtest/sequential_e_testing.py:300-328` uses current observations to:

1. estimate variance;
2. set the current intrinsic time;
3. replace `v_opt`;
4. instantiate a new mixture;
5. subtract a previous log value produced by the old mixture.

The resulting quantity is

```text
log E_t = log M_t^(rho_t) - log M_(t-1)^(rho_(t-1)),
```

not a ratio from one nonnegative supermartingale. The default `use_empirical_variance=True` therefore invalidates the main advertised mean workflow.

A read-only null diagnostic reported 291 crossings among 1,000 independent `N(0,1)` paths by observation 200 at `alpha=0.05`.

### 8.2 Variance testing

`expectation/seqtest/sequential_e_testing.py:378-406` uses

```text
s = (n - 1) * sample_variance / null_variance
v = n - 1
```

as input to a gamma-exponential mixture. Under a Gaussian boundary null, `E[s] = n - 1`; this is not a centered partial-sum process. The source itself marks both the test and interval as unvalidated.

A read-only null diagnostic reported 200 rejections among 200 `N(0,1)` samples of size 100 with null variance 1.

### 8.3 Two-sided beta-binomial use

The two-sided proportion and quantile paths replace a signed statistic with `abs(s)`.

`BetaBinomialMixture(..., is_one_sided=False)` is not even in `s` unless `g == h`. A direct numerical check with `g=0.2`, `h=0.8` gave:

```text
log M( 3, 16) = -0.9677734724
log M(-3, 16) = -0.8481680465
```

Therefore:

- two-sided proportion is unverified when `p0 != 0.5`;
- two-sided quantile is unverified when `tau != 0.5`;
- the confidence-inversion path, which uses signed `s`, is inconsistent with the test path.

### 8.4 Quantile direction and state

For a continuous distribution,

```text
Q_tau > q0  if and only if  F(q0) < tau.
```

The current `GREATER` branch uses `(prop_below - tau) * n`, so it accumulates positive evidence for the opposite quantile direction. `LESS` is correspondingly reversed.

The evaluator appends observations to `all_data` but never increments `data_count`. Results report `sample_size=0`, and quantile confidence bounds remain unreachable.

### 8.5 Confidence sequences

`expectation/confseq/confidencesequence.py:42-95` exposes several estimands and boundaries but always executes the same gamma-exponential mean interval. The selected boundary is returned as metadata without controlling the formula.

The empirical-Bernstein subclass applies a range multiplier to endpoints already in original units. The current proposal keeps observations, residuals and intervals in original units, with only a final range intersection; normalizing and inverse-transforming everything would be a different consistent design.

Existing coverage tests evaluate one fixed endpoint at sample size 10. They do not establish simultaneous coverage over time.

### 8.6 Stronger spatial mergers

The local merging results distinguish:

- e-merging under arbitrary dependence;
- independent e-merging;
- sequential e-merging where each spatial input is conditionally valid given prior spatial inputs.

Arithmetic averaging is safe under arbitrary dependence. Product, U-statistic, lambda-product, and segment-product require their independent/sequential contracts. The parallel class describes independent streams; correlated-input counterexamples outside that premise do not refute the product formula. An explicit contract field is one design option, not a runtime proof of independence.

The `merge_include_rejected=False` behavior is conservative rather than proof-breaking:

- previously rejected streams are predictable;
- a newly rejected stream necessarily has a current increment greater than one;
- replacing that current e-value by one only decreases each coordinatewise-increasing merger.

### 8.7 Conformal and parametric paths

The conformal mixture mode lacks the normalization/equivariance required by the conformal source. The standard CUSUM starts and resets at zero, so evidence is multiplied from the truncation floor rather than using the source's restart recurrence. The cited SR procedure is **reverse** Shiryaev-Roberts; substituting ordinary forward SR would not repair the translation. Unfloored conformal pseudomartingales intentionally have a fixed-horizon rather than general anytime guarantee.

The t-test martingales infer sample size using `len(s)` and default to one for scalar sufficient statistics. Public callers pass scalar sums, so the sample-dependent exponents cannot represent the actual sample size. Factories also reference a nonexistent `TestType.TTEST`.

The full audit corrected T-02's numerical witness (actual infinity guard, not `.248843376`), identified the unconstructible t-CS, and added precision/biased-variance/filtration prerequisites. Do not enable factories ahead of the mathematical paths. For SEQ-02, compare H's full-sample recursive-residual variance construction with disjoint pairs before changing the experiment.

## 9. State, API, and Numerical Findings

This is the historical inventory, not a second current defect ledger. The 43-ID audit distinguishes confirmed paper-to-code findings from excluded or separately scoped engineering observations; its current source locations and qualifications take precedence over this table.

| Finding | Location | Consequence |
|---|---|---|
| Reset leaves previous cumulative log evidence | `sequential_e_testing.py:815-833` | First post-reset e-value is compared with pre-reset evidence |
| Reset omits intrinsic-time and log-optimal state | Same | Reset is not equivalent to a new object |
| Results contain live mutable `EProcess` | `sequential_e_testing.py:515-545` | Earlier result objects change after later updates |
| `e_power_result` is not a result field | `:111-128,483-536` | Pydantic silently discards computed output |
| Rejection uses max, p-value uses current value | Single, parallel, merged paths | A permanently rejected test can later report `p > alpha` |
| Plotting recomputes current rejection | `utils/helper_functions.py` dashboard path | Visual state can contradict Ville state |
| Python/Rust adaptive algorithms differ | `martingales.py` vs `rust/par_seqtest/update.rs` | Same name denotes different optimization behavior |
| `variance_mode` is unused | `par_seqtest.py:121-160` | `variance` argument silently determines actual mode |
| Failed Rust step increments clock | `rust/par_seqtest/mod.rs:173-188` | Stopping-time labels desynchronize from observations |
| Invalid segments reach Rust slices | `rust/merge/mod.rs:177-204` | Out-of-range input raises `PanicException`; unsorted input can be silently skipped |
| Non-finite mixture values are clamped | `rust/par_seqtest/update.rs` | Evidence corruption is hidden rather than rejected |
| Merge exponentiates before log computations | `rust/merge/mod.rs:255-274` | Product-like mergers can overflow before taking logs |
| Tiny-argument `erfc` slope is incorrect | `rust/math/erfc.rs:111` | Small but real numerical-kernel defect |
| K-sample `gamma` has two meanings | `ksample/config.py`, `ksample_test.py` | Beta prior shape also controls adaptive betting cap |
| K-sample float observation is truncated | `ksample_test.py:206-239` | `0.5` silently becomes binary zero |
| K-sample simple keys are not checked | `ksample/config.py` | Length-correct but wrongly keyed maps fail later |
| Calibrator accepts negative p-values | `modules/calibrators.py` | Invalid p-values can produce enormous evidence |
| Mixture calibrator is `NaN` at p=1 | Same | Endpoint contract is broken |

The lookback adjuster returns zero at `E=1` although the right limit is `1/2`. This is not the cited right-continuous admissible version, but it is pointwise no larger than that version and is therefore conservative for rejection.

## 10. Test and Delivery Assessment

### 10.1 Strong evidence already present

- Python/Rust normal-mixture golden comparisons at tolerance `1e-13`.
- Formula and gambling-system consistency tests for mergers.
- Numerical calibration tests for adjusters.
- K-sample unrestricted Type-I, power, reset, p-process, and long-run stability tests.
- Explicit documentation that unadjusted multiple-testing procedures are snapshot procedures rather than carefree procedures.
- Performance-oriented tests for the parallel engine.

### 10.2 Remaining coverage limitations

- Rust `#[cfg(test)]` modules are now included in CI through `cargo test --locked`; the old omission is no longer a configuration gap.
- The adaptive Rust fixture manually reimplements the Rust rule instead of calling Python's public adaptive combiner.
- Merger simulations use independent inputs and do not stress correlated null streams.
- Sequential quantile is omitted from active parameterized update tests.
- Confidence-sequence tests check marginal rather than simultaneous coverage.
- Variance has no valid Type-I test and fails direct null diagnostics.
- Some conformal tests encode the erroneous floor/CUSUM behavior; a correct repair must replace those expectations.
- Parametric confidence-sequence tests are skipped.
- The audit demonstrated failed-call, malformed-input and numerical cases missed by existing tests. Release workflow artifact checks are now configured, but their presence is not proof of statistical correctness or current release success.

### 10.3 Current delivery configuration

- `pyproject.toml` declares runtime and development dependencies; `requirements.txt` is no longer the dependency source.
- Python metadata identifies `GPL-3.0-only AND LicenseRef-AI-Training-Prohibited`; Cargo uses the repository license file. The old MIT metadata claim is obsolete.
- CI defines Ubuntu/macOS/Windows with Python 3.12/3.13, minimum dependencies, version consistency, and Rust fmt/clippy/test jobs. PyO3 extension-module linking is configured through maturin rather than default Cargo features.
- Release workflow definitions build and check wheels/sdist. No live GitHub run or release certification is implied by this documentation update.
- README now links the unresolved statistical audit, uses the full single-stream import path and supplies known variance in its illustrative mean example. Public package re-exports have not been added.
- Normal package construction still uses maturin and bundles the native extension. The archived workflows and removed `RUST_SETUP.md` are not current implementation targets.

## 11. Candidate Architecture (Not Approved)

The following sketches are retained as design options. The current audit requires statistical/state invariants, not these exact classes or a wholesale rewrite of existing calculator closures. Select the smallest coherent implementation for the chosen repair.

### 11.1 Statistical contract

An optional structured representation of a method's documented contract is:

```python
class StatisticalContract(BaseModel):
    method: str
    null_hypothesis: str
    alternative_hypothesis: str
    observation_assumptions: tuple[str, ...]
    filtration: str
    predictable_inputs: tuple[str, ...]
    clock: Literal["observation", "block", "engine_step"]
    dependence: Literal[
        "arbitrary",
        "conditionally_independent",
        "spatially_sequential",
    ]
    guarantee: Literal[
        "fixed_time_e_value",
        "anytime_e_process",
        "confidence_sequence",
        "snapshot_fdr",
        "snapshot_fwer",
        "fdr_sup",
        "fwer_sup",
    ]
    citation: str

    model_config = ConfigDict(frozen=True)
```

The relevant assumptions must be explicit, but a new `StatisticalContract` class/registry is not mandatory. An adequate existing config/docstring/result contract can represent them without introducing this framework.

### 11.2 Stateless kernel plus explicit state transition

One possible pattern is:

```python
class SequentialKernel(ABC):
    @abstractmethod
    def step(
        self,
        state: SequentialState,
        observations: NDArray[np.float64],
    ) -> tuple[SequentialState, StepEvidence]:
        ...
```

This sketch separates calculations from publication. Existing test-specific closures with private staged state can achieve the same invariant; replacing them with a new ABC is not required. Persistence is a separate capability, not an automatic stabilization prerequisite.

### 11.3 Common result semantics

Results must distinguish the applicable counts/evidence/decision semantics. This proposed common model is optional:

```python
class SequentialResult(BaseModel):
    step_count: int
    observation_count: int
    current_e_value: float
    current_e_process: float
    max_e_process: float
    current_p_value: float
    anytime_p_value: float
    rejected: bool
    stopping_step: int | None
    method: str
    contract: StatisticalContract

    model_config = ConfigDict(frozen=True)
```

Returned state must be a snapshot, not a reference to mutable engine state.

### 11.4 Python reference and Rust mirror

For each accelerated method:

1. Python defines the canonical transition and validation.
2. Golden fixtures are generated by calling the canonical Python implementation.
3. Rust implements the same transition.
4. Cross-language tests compare every output field and exceptional domain.
5. Performance-specific divergence requires a separately named algorithm.

## 12. Historical Implementation Programs

Task numbers and checkboxes below are the old program catalog, not today's execution queue or progress record. Reconcile any selected work with the ranked audit and candidate-proposal status first. Some originally proposed files do not exist or are no longer needed. Commit examples are historical and never authorize a commit or GitHub operation.

### Task 1: Establish the Statistical Contract Registry

**Current disposition:** optional architecture; not a prerequisite to P03, LIFE-01, SEQ-03 or other bounded repairs. Do not start registry creation merely because it was Task 1 in the old plan.

**Files:**
- Create: `expectation/contracts.py`
- Create: `docs/statistical-contracts.md`
- Modify: `README.md`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces: immutable `StatisticalContract`
- Produces: `get_statistical_contract(method: str) -> StatisticalContract`
- Consumes: no implementation state

- [ ] **Step 1: Write contract serialization tests**

Test exact round-trip serialization, frozen behavior, enum values, and rejection of missing inferential fields.

Run:

```bash
pytest tests/test_contracts.py -v
```

Expected before implementation: import failure for `expectation.contracts`.

- [ ] **Step 2: Implement the contract model and registry**

Register only methods whose current contract can be stated without qualification:

```python
KNOWN_PROXY_MEAN = StatisticalContract(
    method="known_proxy_mean",
    null_hypothesis="Conditional mean equals the configured null value",
    alternative_hypothesis="Configured one-sided or two-sided mean alternative",
    observation_assumptions=(
        "Conditionally sub-Gaussian centered increments",
        "Known predictable variance proxy",
    ),
    filtration="Common data filtration through the previous observation",
    predictable_inputs=("variance proxy", "betting fraction"),
    clock="observation",
    dependence="arbitrary",
    guarantee="anytime_e_process",
    citation="Howard et al. (2022), normal-mixture construction",
)
```

Experimental methods must receive `readiness="experimental"` in the final model or remain unregistered.

- [ ] **Step 3: Document the registry**

For every registered method, explain inputs, assumptions, clock, stopping guarantee, and invalid use cases in `docs/statistical-contracts.md`.

- [ ] **Step 4: Run focused tests**

```bash
pytest tests/test_contracts.py -v
```

Expected: all contract tests pass.

- [ ] **Step 5: Commit when explicitly requested**

```bash
git add expectation/contracts.py docs/statistical-contracts.md tests/test_contracts.py README.md
git commit -m "docs: define statistical method contracts"
```

### Task 2: Fail Closed on Unsupported Public Workflows

**Files:**
- Modify: `expectation/seqtest/sequential_e_testing.py`
- Modify: `expectation/confseq/confidencesequence.py`
- Modify: `expectation/parametric/ttest_universal.py`
- Modify: `expectation/conformal/conformal.py`
- Modify: `expectation/conformal/cusum.py`
- Modify: `README.md`
- Test: `tests/seqtest/test_sequential_e_testing.py`
- Test: `tests/confseq/test_confidencesequence.py`
- Test: `tests/parametric/test_ttest_universal.py`
- Test: `tests/modules/conformal/`

**Interfaces:**
- Consumes: `StatisticalContract`
- Produces: explicit errors or experimental readiness labels instead of plausible invalid results

- [ ] **Step 1: Write failing quarantine tests**

Tests must assert that variance, empirical-variance mean, generic confidence-sequence estimands, t-test factories, and conformal detectors cannot be constructed as production-ready methods.

- [ ] **Step 2: Add explicit readiness gates**

Use a typed enum rather than warnings alone:

```python
class MethodReadiness(str, Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    UNAVAILABLE = "unavailable"
```

Unproved methods must require explicit experimental opt-in or raise `NotImplementedError` with their blocking mathematical requirement.

- [ ] **Step 3: Narrow README claims**

Document the stable restricted core separately from experimental research modules.

- [ ] **Step 4: Run focused tests**

```bash
pytest tests/seqtest/test_sequential_e_testing.py \
  tests/confseq/test_confidencesequence.py \
  tests/parametric/test_ttest_universal.py \
  tests/modules/conformal/ -v
```

Expected: unsupported workflows fail explicitly and stable workflows remain usable.

- [ ] **Step 5: Commit when explicitly requested**

```bash
git add expectation/ README.md tests/
git commit -m "fix: fail closed on unvalidated statistical methods"
```

### Task 3: Repair Sequential State and Result Semantics

**Files:**
- Modify: `expectation/modules/hypothesistesting.py`
- Modify: `expectation/modules/eprocessupdater.py`
- Modify: `expectation/seqtest/sequential_e_testing.py`
- Modify: `expectation/utils/helper_functions.py`
- Test: `tests/seqtest/test_sequential_e_testing.py`
- Test: `tests/seqtest/test_eprocess_updater_integration.py`

**Interfaces:**
- Produces: immutable result snapshots
- Produces: explicit `step_count`, `observation_count`, `current_p_value`, and `anytime_p_value`
- Preserves: mutable internal engine state

- [ ] **Step 1: Write regression tests**

Cover:

- update -> reset -> update equals a fresh instance;
- prior results do not change after later updates;
- `current_p_value` can change while `anytime_p_value` is non-increasing;
- dashboard rejection equals stored Ville rejection;
- log-optimal reset restores its expectation callback.

- [ ] **Step 2: Reset all state**

Reset `previous_log_e_cumulative`, intrinsic time, empirical variance, tuning state, histories, callbacks, and counters from a single initialization helper.

- [ ] **Step 3: Snapshot process output**

Do not attach the live `EProcess` to a returned result. Copy the required scalar and tuple fields into a frozen result model.

- [ ] **Step 4: Separate p-value semantics**

Use:

```python
current_p_value = min(1.0, 1.0 / current_e_process)
anytime_p_value = min(1.0, 1.0 / max_e_process)
```

- [ ] **Step 5: Run focused tests**

```bash
pytest tests/seqtest/ -v
```

Expected: state equivalence, immutable snapshots, and p-process monotonicity pass.

- [ ] **Step 6: Commit when explicitly requested**

```bash
git add expectation/modules/hypothesistesting.py \
  expectation/modules/eprocessupdater.py \
  expectation/seqtest/sequential_e_testing.py \
  expectation/utils/helper_functions.py \
  tests/seqtest/
git commit -m "fix: make sequential state and results reproducible"
```

### Task 4: Stabilize Known-Proxy Mean Testing

**Files:**
- Modify: `expectation/seqtest/sequential_e_testing.py`
- Modify: `expectation/modules/martingales.py`
- Modify: `expectation/par_seqtest.py`
- Modify: `rust/py.rs`
- Modify: `rust/par_seqtest/update.rs`
- Test: `tests/seqtest/test_sequential_e_testing.py`
- Test: `tests/rust/test_par_seqtest.py`

**Interfaces:**
- Consumes: positive finite predictable variance proxy
- Produces: fixed-mixture sequential e-values
- Excludes: ordinary-variance-only and current-data plug-in claims

- [ ] **Step 1: Write assumption and validation tests**

Reject zero, negative, NaN, and infinite proxies before state mutation. Require the stable mean workflow to receive a known proxy explicitly.

- [ ] **Step 2: Remove retrospective mixture retuning**

Construct `rho` once from fixed configuration. Do not replace the cumulative mixture using current data.

- [ ] **Step 3: Align Python and Rust validation**

Both boundaries must reject the same invalid `v_opt`, `alpha_opt`, variance, observation, and dimension domains.

- [ ] **Step 4: Add optional-stopping tests**

For multiple null parameter values and horizons, estimate:

```text
P(max over t <= T of M_t >= 1 / alpha)
```

under Gaussian and bounded sub-Gaussian nulls. Use deterministic seeds and statistically justified tolerance bands.

- [ ] **Step 5: Run focused tests**

```bash
maturin develop --release
pytest tests/seqtest/test_sequential_e_testing.py \
  tests/rust/test_par_seqtest.py -v
```

Expected: Python and Rust known-proxy paths satisfy parity and null crossing budgets.

- [ ] **Step 6: Commit when explicitly requested**

```bash
git add expectation/ rust/ tests/seqtest/ tests/rust/
git commit -m "fix: stabilize known-proxy mean e-processes"
```

### Task 5: Repair Bernoulli Proportion and Quantile Testing

**Files:**
- Modify: `expectation/seqtest/sequential_e_testing.py`
- Modify: `expectation/modules/boundaries.py`
- Modify: `expectation/modules/martingales.py`
- Modify: `expectation/modules/quantiletest.py`
- Test: `tests/seqtest/test_sequential_e_testing.py`
- Test: `tests/modules/test_boundaries.py`
- Test: `tests/modules/test_quantiletest.py`

**Interfaces:**
- Produces: signed beta-binomial evidence
- Produces: quantile alternatives defined in quantile rather than CDF space
- Requires: explicit tie/continuity convention

- [ ] **Step 1: Write asymmetry and direction tests**

Test `p0` and `tau` values `0.1`, `0.3`, `0.5`, `0.7`, `0.9`. Verify that signed deviations remain distinct when `g != h`, and that a distribution with `Q_tau > q0` favors `GREATER`.

- [ ] **Step 2: Remove unproved absolute-value transforms**

Use a theorem-backed two-sided signed construction. If that construction requires a two-component mixture, implement and name it explicitly rather than applying `abs`.

- [ ] **Step 3: Repair quantile state**

Increment observation count, return correct sample size, and make confidence bounds reachable.

- [ ] **Step 4: Define ties**

Document and test continuous-data behavior and the selected convention for atoms at `q0`.

- [ ] **Step 5: Harden Bernoulli inversion**

Validate root brackets before bisection and ensure interval failures do not occur after test state has mutated.

- [ ] **Step 6: Add theorem-level simulations**

Test optional-stopping Type-I error across null probabilities, directions, batch sizes, and horizons.

- [ ] **Step 7: Run focused tests**

```bash
pytest tests/seqtest/test_sequential_e_testing.py \
  tests/modules/test_boundaries.py \
  tests/modules/test_quantiletest.py -v
```

Expected: signed evidence, direction, ties, intervals, and optional-stopping tests pass.

- [ ] **Step 8: Commit when explicitly requested**

```bash
git add expectation/seqtest/ expectation/modules/ tests/seqtest/ tests/modules/
git commit -m "fix: restore Bernoulli and quantile test validity"
```

### Task 6: Harden K-Sample RIPr

**Files:**
- Modify: `expectation/ksample/config.py`
- Modify: `expectation/ksample/bernoulli.py`
- Modify: `expectation/ksample/ksample_test.py`
- Test: `tests/ksample/test_bernoulli_ksample.py`
- Modify: `examples/ksample-bernoulli.ipynb`

**Interfaces:**
- Produces: an explicit product-sampling/completed-block contract, including permitted exogenous arrivals
- Produces: separate prior `gamma` and betting cap; names/default migration require approval
- Requires: outcome-independent completion within the current block, not predeclared sizes for every future block

- [ ] **Step 1: Write invalid-configuration tests**

Cover wrong simple-theta keys, additive effect size outside `(0,1)`, empty grids, non-finite theta values, unsupported log-optimal betting, and lossy scalar observations.

- [ ] **Step 2: Separate gamma parameters**

Use:

```python
gamma: float = Field(default=0.18, gt=0)
betting_gamma: float = Field(default=0.5, gt=0, le=1)  # proposed default, not approved
```

- [ ] **Step 3: Replace or constrain `update_single()`**

Preserve the source's product-sampling and current-block completion restrictions. Predeclared blocks and the permitted exogenous-arrival scheme are both possibilities; buffered labels `0,0,1` can legitimately emit sizes `(2,1)`. Do not infer independence from numeric validation or reject all arrival-driven buffering. The current proposal also requires coherent failed-trigger buffer/retry semantics.

- [ ] **Step 4: Preserve log evidence**

Avoid `exp(log_e_value)` until output presentation. Add a log-space e-process update path so extreme blocks cannot overflow before accumulation.

- [ ] **Step 5: Expand validity simulations**

Cover:

- null probabilities near 0 and 1;
- unequal but predictable block sizes;
- k values 2, 3, and 10;
- all supported betting strategies;
- unrestricted, simple, additive, and log-odds alternatives;
- grid precision and effect-size extremes.

- [ ] **Step 6: Run focused tests**

```bash
pytest tests/ksample/test_bernoulli_ksample.py -v
```

Expected: configuration, filtration, log-space, Type-I, power, and reset contracts pass.

- [ ] **Step 7: Commit when explicitly requested**

```bash
git add expectation/ksample/ tests/ksample/ examples/ksample-bernoulli.ipynb
git commit -m "fix: harden blockwise k-sample RIPr testing"
```

### Task 7: Encode Merging Dependence Contracts

**Files:**
- Modify: `expectation/modules/merging.py`
- Modify: `expectation/par_seqtest.py`
- Modify: `rust/merge/mod.rs`
- Modify: `rust/py.rs`
- Test: `tests/modules/test_merging.py`
- Test: `tests/test_merged_par_seqtest.py`

**Interfaces:**
- Produces: `DependenceAssumption`
- Defaults: arithmetic mean with arbitrary dependence
- Gates: stronger mergers behind conditional independence or spatial sequentiality

- [ ] **Step 1: Write dependence-gate tests**

Constructing product-like global merging without an explicit compatible dependence assumption must fail before Rust construction.

- [ ] **Step 2: Add the dependence enum**

```python
class DependenceAssumption(str, Enum):
    ARBITRARY = "arbitrary"
    CONDITIONALLY_INDEPENDENT = "conditionally_independent"
    SPATIALLY_SEQUENTIAL = "spatially_sequential"
```

- [ ] **Step 3: Enforce method compatibility**

Permit arithmetic mean for all three assumptions. Permit product, U-statistic, lambda-product, and segment-product only for the latter two.

- [ ] **Step 4: Validate all segment boundaries**

Require a strictly increasing, unique sequence with every boundary in `(0, n_tests)`. Duplicate validation in the PyO3 boundary so direct native construction cannot panic.

- [ ] **Step 5: Keep merge calculations in log space**

Use log-sum-exp for means and stable recurrences for product-like methods. Do not exponentiate every sequential log e-value before choosing the merger.

- [ ] **Step 6: Add dependence simulations**

Test arithmetic merging under correlated null streams and stronger mergers under their declared conditional-independence model.

- [ ] **Step 7: Run focused tests**

```bash
maturin develop --release
pytest tests/modules/test_merging.py tests/test_merged_par_seqtest.py -v
```

Expected: dependence gates, segment validation, overflow handling, and Python/Rust parity pass.

- [ ] **Step 8: Commit when explicitly requested**

```bash
git add expectation/modules/merging.py expectation/par_seqtest.py \
  rust/merge/ rust/py.rs tests/modules/test_merging.py \
  tests/test_merged_par_seqtest.py
git commit -m "fix: encode dependence assumptions for e-value merging"
```

### Task 8: Harden the Parallel Engine Boundary

**Files:**
- Modify: `expectation/par_seqtest.py`
- Modify: `rust/error.rs`
- Modify: `rust/py.rs`
- Modify: `rust/par_seqtest/mod.rs`
- Modify: `rust/par_seqtest/update.rs`
- Modify: `rust/math/erfc.rs`
- Test: `tests/rust/test_par_seqtest.py`
- Test: `rust/tests/test_erfc.rs`

**Interfaces:**
- Produces: failure-atomic `step()`
- Produces: explicit variance-mode behavior
- Rejects: non-finite observations and invalid numerical domains

- [ ] **Step 1: Write failure-atomicity tests**

After a dimension, NaN, infinite, or variance-domain failure, assert that time, counts, sums, evidence, rejection flags, and merge state are unchanged.

- [ ] **Step 2: Make `variance_mode` authoritative**

Cross-validate `variance` against `variance_mode` and reject contradictory combinations.

- [ ] **Step 3: Validate before incrementing time**

Move dimension and finite-input validation ahead of `self.time_step += 1`.

- [ ] **Step 4: Remove evidence clamping**

Return a typed numerical error containing the test index and invalid intermediate rather than replacing NaN or infinity with fixed evidence.

- [ ] **Step 5: Correct the `erfc` small-x term**

The first-order branch must use:

```rust
1.0 - FRAC_2_SQRT_PI * x
```

- [ ] **Step 6: Add small-x numerical tests**

Compare at zero, `2^-30`, `2^-29`, and both sides of the branch threshold against a trusted reference with absolute tolerance `2e-15`.

- [ ] **Step 7: Run focused checks**

```bash
cargo check
maturin develop --release
pytest tests/rust/test_par_seqtest.py -v
```

Expected: clean failures, stable clocks, and numerical parity.

- [ ] **Step 8: Commit when explicitly requested**

```bash
git add expectation/par_seqtest.py rust/ tests/rust/test_par_seqtest.py
git commit -m "fix: make the parallel engine failure atomic"
```

### Task 9: Unify Adaptive Betting Semantics

**Files:**
- Modify: `expectation/modules/martingales.py`
- Modify: `expectation/modules/eprocessupdater.py`
- Modify: `expectation/par_seqtest.py`
- Modify: `rust/par_seqtest/update.rs`
- Modify: `rust/merge/mod.rs`
- Modify: `tests/rust/fixtures/generate_golden_fixtures.py`
- Test: `tests/seqtest/test_eprocess_updater_integration.py`
- Test: `tests/rust/test_par_seqtest.py`

**Interfaces:**
- Produces: one canonical named algorithm per enum value
- Produces: Python-generated fixtures using the public reference implementation

- [ ] **Step 1: Choose the theorem-backed algorithm**

Select exact empirical log-growth optimization, true ONS, or the current ratio approximation. Give distinct algorithms distinct enum values.

- [ ] **Step 2: Write trajectory parity tests**

Compare every lambda, sequential e-value, process increment, and cumulative log process over deterministic trajectories.

- [ ] **Step 3: Implement the selected semantics in Python**

The public Python combiner is the reference.

- [ ] **Step 4: Mirror in Rust**

Preserve predictability by computing lambda from prior e-values or sufficient statistics only.

- [ ] **Step 5: Regenerate fixtures through the public Python implementation**

Delete hand-written duplication of the Rust recurrence from the fixture generator.

- [ ] **Step 6: Run focused tests**

```bash
maturin develop --release
pytest tests/seqtest/test_eprocess_updater_integration.py \
  tests/rust/test_par_seqtest.py -v
```

Expected: exact algorithm naming and trajectory parity at `1e-13`.

- [ ] **Step 7: Commit when explicitly requested**

```bash
git add expectation/modules/ expectation/par_seqtest.py rust/ \
  tests/seqtest/ tests/rust/
git commit -m "refactor: unify adaptive e-process betting"
```

### Task 10: Rebuild Confidence Sequences One Estimand at a Time

**Current disposition:** CS-02/CS-03 form one bounded empirical-Bernstein repair. Use predictable prediction residuals and the selected original-unit design; the old affine-normalization suggestion below is not the current recommendation.

**Files:**
- Modify: `expectation/confseq/confidenceconfig.py`
- Modify: `expectation/confseq/confidencesequence.py`
- Modify: `expectation/modules/boundaries.py`
- Test: `tests/confseq/test_confidencesequence.py`

**Interfaces:**
- Produces: separate classes for each proved estimand/boundary combination
- Removes: enum values that merely relabel one implementation

- [ ] **Step 1: Limit the first rebuild to one theorem-backed mean CS**

Require its exact boundedness, sub-Gaussian, or sub-exponential assumptions in configuration.

- [ ] **Step 2: Write simultaneous coverage tests**

Each simulation counts coverage only when the true estimand remains in every interval through horizon `T`.

- [ ] **Step 3: Add optional-stopping tests**

Stop at narrowest width, first exclusion, and data-dependent horizons; verify the declared coverage budget.

- [ ] **Step 4: Rebuild bounded empirical Bernstein in consistent original units**

The current candidate follows H Theorem 4 with:

```text
V_t = sum((X_i - prediction_i)^2), prediction_i fixed before X_i
c = upper_bound - lower_bound
interval = original-data mean +/- original-unit radius
```

Intersect the completed interval with the declared bounds. A fully normalized implementation with a correct inverse affine transformation is another valid design, but multiplying already original-unit endpoints is not.

- [ ] **Step 5: Add each additional estimand only with an independent derivation**

Quantile, variance, and proportion implementations receive separate classes and tests rather than an estimand label on a mean calculation.

- [ ] **Step 6: Run focused tests**

```bash
pytest tests/confseq/test_confidencesequence.py -v
```

Expected: every exposed class has simultaneous-coverage evidence under its exact assumptions.

- [ ] **Step 7: Commit when explicitly requested**

```bash
git add expectation/confseq/ expectation/modules/boundaries.py tests/confseq/
git commit -m "refactor: rebuild theorem-backed confidence sequences"
```

### Task 11: Rebuild or Remove Parametric and Conformal Prototypes

**Files:**
- Modify: `expectation/parametric/ttest_universal.py`
- Modify: `expectation/conformal/conformal.py`
- Modify: `expectation/conformal/cusum.py`
- Modify: `expectation/conformal/adaptivethreshold.py`
- Test: `tests/parametric/test_ttest_universal.py`
- Test: `tests/modules/conformal/`

**Interfaces:**
- Produces: explicit sample-indexed t-state if retained
- Produces: paper-matched conformal recurrence if retained
- Otherwise: removes public factories and keeps research code explicitly internal

- [ ] **Step 1: Complete an equation-to-state specification before coding**

For t-tests, specify cumulative `n`, `S_n`, `V_n`, nuisance estimation, alternative direction, and confidence inversion. For conformal methods, specify exchangeability, normalization, restart capital, SR recurrence, and false-alarm claim.

- [ ] **Step 2: Replace scalar-length inference**

Pass sample size explicitly to every t-kernel.

- [ ] **Step 3: Remove monkey-patched factory behavior**

Use typed Pydantic configuration and a normal constructor path.

- [ ] **Step 4: Correct CUSUM and SR state**

Start and restart from the theorem's unit-capital state, and implement the cited recurrence exactly.

- [ ] **Step 5: Add theorem-level null simulations**

Test optional-stopping Type-I or false-alarm guarantees, not only positive output and finite arithmetic.

- [ ] **Step 6: Remove skipped tests**

Every public path must have active tests. If the derivation is not ready, remove the public path rather than retaining skipped confidence tests.

- [ ] **Step 7: Run focused tests**

```bash
pytest tests/parametric/test_ttest_universal.py \
  tests/modules/conformal/ -v
```

Expected: retained methods match their specifications; removed methods fail explicitly.

- [ ] **Step 8: Commit when explicitly requested**

```bash
git add expectation/parametric/ expectation/conformal/ \
  tests/parametric/ tests/modules/conformal/
git commit -m "refactor: reconcile parametric and conformal methods"
```

### Task 12: Create a Stable Public Package Surface

**Files:**
- Modify: `expectation/__init__.py`
- Modify: `expectation/seqtest/__init__.py`
- Modify: `expectation/ksample/__init__.py`
- Modify: `expectation/confseq/__init__.py`
- Modify: `expectation/conformal/__init__.py`
- Modify: `expectation/parametric/__init__.py`
- Modify: `README.md`
- Test: `tests/test_public_api.py`

**Interfaces:**
- Produces: documented stable imports
- Separates: stable and experimental namespaces

- [ ] **Step 1: Write public import tests**

Test every README import in a fresh interpreter.

- [ ] **Step 2: Export stable symbols explicitly**

Use explicit imports and `__all__`; do not wildcard-export experimental implementations.

- [ ] **Step 3: Move experimental APIs behind clear namespaces**

For example:

```python
from expectation.experimental.conformal import ConformalEValue
```

- [ ] **Step 4: Exercise README examples**

Convert minimal examples into smoke tests so documentation cannot drift silently.

- [ ] **Step 5: Run focused tests**

```bash
pytest tests/test_public_api.py -v
```

Expected: documented imports and minimal workflows succeed.

- [ ] **Step 6: Commit when explicitly requested**

```bash
git add expectation/**/__init__.py expectation/__init__.py README.md \
  tests/test_public_api.py
git commit -m "feat: define the stable expectation API"
```

### Task 13: Repair Packaging, Licensing, and CI

**Current disposition:** much of this historical program is already present in configuration, independently of the still-unimplemented statistical repairs. Do not recreate removed files or split the crate to solve a linking issue already addressed by the current PyO3 feature setup.

**Existing surfaces:** `pyproject.toml`, `Cargo.toml`, `Cargo.lock`, `CITATION.cff`, `LICENSE`, `README.md`, `scripts/check_version.sh`, and `.github/workflows/{ci,bump,release}.yml`. Runtime/dev dependencies, current license metadata, Python/OS matrices, minimum-dependency checks and link-safe Rust tests are configured.

**Remaining obligation when delivery work is actually requested:** verify the exact shipped artifacts, documented imports and supported environments, rather than equating workflow definitions with a successful release. Keep version changes synchronized through the existing scripts. Do not add `requirements.txt`, restore `RUST_SETUP.md`, or target the archived `run-tests.yml`.

Representative existing commands, not a request to execute them during documentation synchronization:

```bash
bash scripts/check_version.sh
cargo test --locked
maturin build --release --locked
```

The existing release workflow describes isolated wheel/sdist installation checks. All GitHub operations and commits remain separately authorized.

### Task 14: Add Persistence, Auditability, and Reproducible Performance

**Current disposition:** optional future capability, not an automatic prerequisite to the 43-ID repair program. Only promise save/resume or benchmark behavior after that API is selected and implemented.

**Files:**
- Create: `expectation/state.py`
- Modify: `expectation/seqtest/sequential_e_testing.py`
- Modify: `expectation/par_seqtest.py`
- Modify: `expectation/ksample/ksample_test.py`
- Modify: `benchmark.py`
- Test: `tests/test_state_roundtrip.py`
- Test: `tests/rust/test_par_seqtest.py`

**Interfaces:**
- Produces: versioned state snapshots
- Produces: save/resume equivalence
- Produces: result metadata containing method, assumptions, version, clock, and configuration

- [ ] **Step 1: Define a schema-versioned state envelope**

```python
class StateEnvelope(BaseModel):
    schema_version: str
    library_version: str
    method: str
    contract: StatisticalContract
    state: dict[str, object]

    model_config = ConfigDict(frozen=True)
```

- [ ] **Step 2: Write uninterrupted-versus-resumed tests**

For single-stream, parallel, and k-sample engines, assert that save/resume produces the same evidence, decisions, p-process, and stopping times as uninterrupted execution.

- [ ] **Step 3: Add audit metadata**

Every result must identify the implementation version, statistical contract, active assumptions, algorithm variant, and state schema.

- [ ] **Step 4: Separate benchmark tests from correctness tests**

Record hardware, build profile, sample sizes, repetitions, median, and tail latency. Align README claims with measured distributions rather than a single number.

- [ ] **Step 5: Run focused tests**

```bash
maturin develop --release
pytest tests/test_state_roundtrip.py tests/rust/test_par_seqtest.py -v
```

Expected: deterministic save/resume and reproducible performance reporting.

- [ ] **Step 6: Commit when explicitly requested**

```bash
git add expectation/state.py expectation/seqtest/ expectation/par_seqtest.py \
  expectation/ksample/ benchmark.py tests/
git commit -m "feat: add versioned state and audit metadata"
```

## 13. Program Validation Gates

### Gate A: Mathematical contract

- Every stable method has an explicit statistical contract; a new registry/model is optional.
- Every tuning quantity used at time `t` is fixed or `F_(t-1)`-measurable.
- Every public dependence claim is documented and reflected in the selected API where appropriate; numeric validation does not prove independence.
- Each intrinsic-time process satisfies its source condition. Adapted variance is not invalid merely because it is not predictable.
- Every stopping-time and p-value output states whether it is snapshot or anytime-valid.

### Gate B: State correctness

- Reset equals a fresh object.
- Returned results never mutate.
- Failed inputs leave all state unchanged.
- Observation, block, and engine-step clocks are explicit.
- If persistence is supported, it round-trips exactly; adding persistence is a separate scope decision.

### Gate C: Statistical evidence

- E-process tests use running-maximum rejection probabilities.
- Confidence-sequence tests require coverage at every time through the horizon.
- Multiple-testing tests cover correlated streams and common filtrations.
- Simulations span nuisance parameters, tails, block designs, and edge domains.
- Numerical parity tests use an independent reference, not duplicated implementation logic.

### Gate D: Delivery

- Fresh wheel installation succeeds.
- README imports execute.
- Python and Rust metadata agree.
- Runtime dependencies install automatically.
- Rust unit tests and Python tests run in CI.
- Supported Python/OS combinations are exercised.

## 14. Recommended Product Order

After the stabilization gates, product development should proceed in this order:

1. **A/B testing:** one-sided Bernoulli proportion and unrestricted blockwise k-sample RIPr.
2. **Massively parallel monitoring:** known conditional sub-Gaussian variance-proxy mean tests with snapshot and adjusted multiple testing.
3. **Global evidence:** arithmetic-mean intersection testing under arbitrary dependence; stronger merging only with explicit assumptions.
4. **Unknown-variance means and confidence sequences:** only after a theorem-backed predictable/self-normalized construction is implemented.
5. **Quantile and variance products:** after direction, ties, state, and coverage are independently established.
6. **Parametric t and conformal change detection:** after equation-to-state specifications and active theorem-level tests.
7. **Domain applications:** prediction markets, trading, physics, and clinical experimentation only after the statistical contracts required by each application are represented and auditable.

## 15. Definition of Done

The repository is ready to call a production statistical library only when:

- stable public methods are a deliberately narrow subset of implemented research code;
- every stable method's proof assumptions are visible at construction and in results;
- default configurations preserve the claimed inferential guarantee;
- no invalid input or numerical failure silently changes evidence;
- Python and Rust methods with the same name implement the same algorithm;
- optional-stopping and simultaneous-coverage tests exercise the actual guarantees;
- state can be reset and its decisions audited reproducibly; any advertised serialization/resume API preserves those semantics;
- installation, imports, dependencies, licensing, versions, documentation, wheels, and CI agree.
