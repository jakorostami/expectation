# Independent audit and repair priorities

**Audit date:** 2026-09-06  
**Documentation synchronized:** 2026-09-07; findings/proposal catalogs, repository guidance and the historical plan now reflect this adjudication.  
**Source revision:** `6f1ef40a1a0661ded83c30ffa7b7f53b00a93804` (`fix/polishing`, version 0.6.1)  
**Scope:** all 43 retained IDs in `POSSIBLE_BUGS.md`, their corresponding proposals in `PROPOSED_SOLUTIONS.md`, and tightly coupled omissions that affect those repairs.  
**Status:** audit complete; no statistical implementation changes made. This report does not approve every proposed design.

## 1. Decision

**Start with owned segment partitions (P03), then complete reset semantics (LIFE-01), then the signed/reflected Bernoulli construction (SEQ-03).** These are substantive, bounded repairs with decisive counterexamples. They improve existing workflows without first introducing a new framework.

The original audit contains real defects, but it is not a list of 43 independent violations of statistical error guarantees. It also contains parameter-contract gaps, numerical failures, naming differences, overlapping findings, and unreachable prototype branches. The proposed solutions must not be accepted as one package.

The most important changes from the pre-review versions of the findings/proposals are recorded below. Their later documentation synchronization does not mean implementation repairs have landed:

1. **SEQ-01 is supported, including for genuinely Gaussian observations.** A normal mixture is not restricted to normally distributed observations. Howard et al. explicitly use it on Rademacher data. The default implementation nevertheless fails even under iid standard normal observations.
2. **T-02's reported numerical consequence is wrong.** At its stated `n=10, c2=1, alpha=.05` example, the actual exponent parsing produces a negative denominator and the code returns infinity, not `0.248843376`. The equation defect remains real; its error direction is parameter-dependent.
3. **SEQ-02's disjoint-pair proposal is not the only source-backed repair.** The complete Howard paper already gives a full-sample Gaussian variance construction. Pairing is defensible under its declared sampling/filtration contract, but changes information use and should not be adopted without comparing the source's construction.
4. **The shared zero/infinity policy needs correction.** The book explicitly uses context-dependent conventions, including `0 * infinity = 0`. Treating every mixed zero/infinity product as inherently undefined and requiring rejection is not dictated by the cited sources.
5. **P07's numerical proposal is not ready to implement as an established solution.** The cancellation example is real. A description involving floating-point expansions and outward error bounds is not yet a specified, verified numerical algorithm.
6. **The proposed publication/history architecture is optional.** Correct scoring order, coherent failure semantics, and faithful arithmetic are necessary. Public candidates, revision tokens, linked histories, and a generic transaction-like interface are not conclusions of the papers.
7. **Several t-test prerequisites are missing from the proposals.** These include prior-precision semantics, the correct biased-variance multiplier, and restrictions on downstream use of reduced-filtration statistics. Simply fixing the factories would activate wrong calculations.

**Execution order is not severity order.** In particular, the default empirical-variance mean path and the current variance path remain barriers to claiming production-valid inference, even while smaller repairs are implemented first. Finishing the first few rows below does not make the library statistically ready for release.

## 2. Source coverage and verification standard

The parent and the two authorized reviewers read the complete source-text corpus, including introductions, assumptions, proofs, appendices, qualifications, and bibliographies. Initial passage-based reviewer judgments were superseded by their full-document reconciliations.

| Source | Edition actually examined | Complete PDF coverage | Reader |
|---|---|---:|---|
| H | Howard et al., [arXiv:1810.08240v9](https://arxiv.org/pdf/1810.08240v9), title dated August 9, 2022 | 48/48 pages | Parent |
| Q | Howard and Ramdas, [arXiv:1906.09712v5](https://arxiv.org/pdf/1906.09712v5), July 8, 2022 | 35/35 | Parent |
| T | Wang and Ramdas, [arXiv:2310.03722v5](https://arxiv.org/pdf/2310.03722v5), title dated November 8, 2024 | 51/51 | Opus 5, xhigh |
| C | [Conformal e-testing, Working Paper 29](https://www.alrw.net/articles/29.pdf); downloaded PDF build matches November 2, 2024 | 22/22 | Opus |
| CP | Vovk, [arXiv:2001.05989v5](https://arxiv.org/pdf/2001.05989v5), May 18, 2025 | 30/30 | Opus |
| S | Vovk and Wang, [NEJSDS 2, 261-270](https://nejsds.nestat.org/journal/NEJSDS/article/69/file/pdf), DOI `10.51387/24-NEJSDS60` | 10/10 | Opus |
| R | Local `genericevar.pdf`, arXiv:2106.02693v3, title dated June 23, 2022 | 39/39 | GPT-6 Astra, high |
| M | Local `merging.pdf`, arXiv:2007.06382v3, title dated February 23, 2024 | 21/21 | Astra |
| A | Local `carefree.pdf`, arXiv:2501.19360v2, title dated July 18, 2025 | 10/10 | Astra |
| B | Local `hypotevalues.pdf` and exact downloaded [arXiv:2410.23614v6](https://arxiv.org/pdf/2410.23614v6) | 264/264 each | Astra |
| L | Hartog and Lei, [arXiv:2501.09015v1](https://arxiv.org/pdf/2501.09015v1), title dated January 16, 2025 | 24/24 | Astra |
| W | [Published qkad009 article](https://doi.org/10.1093/jrsssb/qkad009), including bundled discussion/reply, and publisher-linked `qkad009_supplementary_data.pdf` | 61/61 main; 51/51 supplement | Astra |

The two B PDFs are **not identical**: the local title is dated September 10, while the downloaded v6 title is dated September 11, 2025. Every page was compared. The 196 pages with identical non-whitespace extracted text were read once for both editions; all 68 differing pages were read completely in both editions. This was full-page deduplication, not reading only changed fragments.

This is complete **source-text reading**, not an assertion that every theorem in these papers was independently proved or every plot visually inspected. W's ONS-m page was additionally rendered to resolve mathematical glyphs lost in extraction. The independent mathematical/code audit is limited to the stated findings and proposals.

Evidence includes exact finite-outcome enumeration, algebraic derivation, numerical integration, public Python executions, and an isolated current-source Rust extension. Native witnesses did **not** use the installed extension, which could be stale. The audit build was unoptimized and supports no performance claim.

Page conventions used below: H/Q/T/R/M/A/L use printed = PDF; C uses PDF = printed + 2; B uses PDF = printed + 13; S printed 261 = PDF 1; W supplement printed 45/46/48 = PDF 11/12/14.

## 3. Ranked repair order

Each row is a complete repair unit, not necessarily one commit. Keep commits grouped by concern; do not combine this table into a repository-wide rewrite. "Low blast radius" refers to localized implementation and stable statistical meaning, not an assumption that public APIs have no external callers.

The first seven rows are a **bounded starting sequence**, not permission to keep searching for easier work. Move into the main statistical/state repairs rather than polishing auxiliary code indefinitely. If a contract decision blocks a row, take the next independent ready unit; do not silently choose a new scientific model.

| Rank | Complete repair unit | IDs | Why here / blast radius | Required boundary of completion |
|---|---|---|---|---|
| **1** | **Validate and own segment partitions at every entry point** | P03 | Low; restores a mathematical partition and prevents accepted malformed designs from reusing coordinates. Existing independent-input witness has mean evidence 1.25. | Python merger, wrapper, PyO3 and direct Rust agree on integer/interior/ordered/unique boundaries and lengths. Caller mutation cannot invalidate a partition. Validation precedes stream updates. Decide empty-partition behavior explicitly. |
| **2** | **Make reset equivalent to a fresh configured experiment** | LIFE-01 | Low; directly fixes false rejection after reset in the known-proxy path. | Reset previous cumulative evidence, sufficient statistics, clocks, histories and owned strategy state; preserve fixed tuning and callback/factory semantics. Exercise update-reset-update, not just zeroed fields. No new public candidate protocol is needed for this repair. |
| **3** | **Restore signed and reflected Bernoulli evidence** | SEQ-03 | Low; corrects the existing proportion calculator rather than replacing its kernel. | Two-sided uses signed counts; LESS negates the statistic and swaps `g,h`. Cover asymmetric nulls, all directions, label reflection and public update behavior. Bounds/result construction must not introduce a failed-call inconsistency. |
| **4** | **Repair the complete calibrator/adjuster numerical maps** | CAL-01, P05 | Low-medium; pure numerical operations with exact source formulas. For positive `p<1`, the mixture calibrator and lookback adjuster share the same analytic function under `L=-log(p)`. | Fix endpoints, near-zero cancellation, positive-small-p flooring, large log inputs, scalar/array behavior, and adjusted decisions in Python/Rust. Preserve the deliberate difference between calibrator `f(1)=1/2` and the current conservative adjuster-at-one convention. A new result-model hierarchy is optional. |
| **5** | **Repair legitimate zero-time root bracketing** | NUM-02 | Low; localized solver correction and a prerequisite for bounded CS paths with zero residual variance. | Start from a positive bracket, establish progress and range/error handling, and solve at `v=0` without inventing positive intrinsic time. Do not change the separately qualified beta physical cap. |
| **6** | **Keep all rejection thresholds logarithmic** | P10 | Low-medium; a stable algebraic identity, but several Python/native/merged/adjusted entry points must agree. | Use log subtraction rather than materializing reciprocals or underflowed alpha allocations. Validate overrides; preserve non-strict crossing at the threshold. Native Ville, Bonferroni, reciprocal Holm and BH must agree in the one-test witness. |
| **7** | **Repair the standalone quantile A/B minimization domain** | QUANT-03 | Low-medium; confined to the existing order-statistic-based algorithm. | Search the gap between the two minimizer plateaus, including the correct breakpoints. Cover swapped arms, ties, overlap and unequal sample sizes against exhaustive small-case oracles. Inner minimization must also provide trustworthy lower values; merely changing an endpoint is insufficient. |
| **8** | **Restore a declared, valid mean construction and matching boundary semantics** | SEQ-01, P01, CS-01 | Medium; main advertised workflow and highest-priority statistical decision after the bounded starter work. | Choose the supported null class, use its justified variance process, fix mixture tuning for that construction, synchronize evidence/bounds/reset, and return directional half-lines where appropriate. Do not silently turn an unknown-variance interface into a known-variance promise. Python and Rust must expose the same chosen contract. |
| **9** | **Separate RIPr prior strength from betting configuration** | RIPr-03; sampling clarification RIPr-01 accompanies RIPr work | Low, but requires a compatibility decision. | Separate parameters in construction and reset; allow the paper's positive prior shapes independently of the betting cap. Decide the cap default/migration explicitly. Document product sampling and permitted exogenous arrivals without building an unnecessary sampling framework. |
| **10** | **Repair restricted-grid construction under an approved convention** | RIPr-04 | Low-medium; restricted alternative only. | Both additive and log-odds modes require a nonempty representable grid and finite normalized weights. Reject invalid configurations before an experiment. Keep main-text marginal means unless separately deciding otherwise; do not claim the appendix ambiguity has been resolved. |
| **11** | **Repair the complete failed-call/retry boundary** | RIPr-02, RIPr-06, EP-01 admission/state portion | Medium; shared updater and multiple owners, so a scalar-validator-only patch is not the unit. | Validate original observations/evidence before accepting them; score from committed previous-block state; preserve prior buffers while rejecting a failed triggering observation. Posterior, capital, counts, history and results must agree on what was accepted. Use the simplest ownership/staging mechanism that provides those semantics. |
| **12** | **Make log evidence authoritative through the supported evidence pipeline** | NUM-01, P04, RIPr-05, EP-02, EP-01 representation/source portion | High/cross-cutting, but unavoidable; do not replace it with log-valued display fields. | Stable producer kernels, temporal/spatial composition, log fractions, current/max capital, decisions, projections and raw-factor log history must agree. Complete the normal/beta tail repairs as well as composition. Decide genuine zero/infinity and source/filtration contracts first. Remove repeated history copying/scanning where a constant-time strategy is claimed; do not assume linked histories alone solve complexity. |
| **13** | **Replace unstable U-statistic normalization and its gambling reconstruction** | P06 | Medium; depends on row 12's log interfaces/domain decisions. | Normalized log ESP, descending updates, exact neutral behavior, all orders and structural endpoints; gambling coefficients use the other `K-1` coordinates. Verify small subset sums and extreme histories, not only the all-ones example. |
| **14** | **Repair bounded empirical-Bernstein confidence sequences in original units** | CS-02, CS-03 | Medium; one coherent construction, not an endpoint-rescaling patch. | Use predictable prediction residuals, declared finite bounds, `c=b-a`, fixed tuning and correct tail allocation. Preserve units, partition-invariant sufficient state and failure-safe results. Establish simultaneous-over-time coverage, not just fixed-time coverage. |
| **15** | **Choose and implement the Gaussian variance construction** | SEQ-02 | Medium-high mathematical contract work. | Compare full-sample recursive residuals with disjoint pairs, including their filtrations and information use. Correct centering, intrinsic time, scale, alternatives and interval inversion together. A lower endpoint error at one sample size is not a reason to change the sampling design automatically. |
| **16** | **Repair the complete single-stream quantile workflow** | QUANT-01, QUANT-02 | Medium-high; count direction, atoms, clocks, evidence kind and intervals interact. | State the quantile-set null, use strict/inclusive counts correctly, and preserve the true observation count. If using the source's cumulative lower envelope, do not fractionally reinvest unproved ratios. Certified minimization and source-correct interval inversion are part of the unit. |
| **17** | **Repair symmetry evaluation and its admitted null/data contract** | SYM-01, SYM-02 | Medium public-contract impact despite localized code. | Decide continuous-model restriction versus the declared conditional-fair-sign extension. Fix all three numerical formulas plus public results/decisions/e-power consumers. Handle zero observations and sign-independent ranks consistently. Do not prioritize this solely because there are few internal callers. |
| **18** | **Restore the conformal producer, capital and detector workflow** | CONF-01, CONF-02, CONF-03, DET-01, DET-02, DET-03 | Medium, explicitly experimental; several coherent subcommits, not a new platform. | Approve scoring/observation-clock semantics; normalize common equivariant scores; separate display floors from capital; implement the actual CUSUM and reverse-SR recurrences; report each alarm once. Disclose reverse-SR segment cost. Preserve fixed-horizon versus anytime distinctions throughout. |
| **19** | **Rebuild t-specific moments, radii and source-aware integration before enabling factories** | T-01, T-02, T-03, T-04, T-05, coupled omissions in section 6 | High semantic impact despite a small module; currently disconnected high-level APIs. | Correct `n`, precision semantics, stable moments, biased-variance factors, calibrated Lai radius, observation-indexed prediction and universal inversion. Explicitly handle reduced-filtration downstream restrictions. Only then enable constructors/factories, or keep unsupported paths explicitly unavailable. Do not alter every ordinary mixture kernel's ABC to accommodate t-state. |
| **20** | **Specify and verify the chosen adaptive strategy's numerical implementation** | P07 | Research/numerical uncertainty; not a ready-made fix. | Distinguish exact empirical optimization, ONS and the existing quadratic rule. Specify domains, epsilon, cutoffs, remainder/rounding bounds, sign/fraction certification, ambiguity behavior and actual cost. The proposed floating expansions remain unapproved until demonstrated. |
| **21** | **Finish compatible method naming and citation corrections** | P08, P09; related merger references | Low runtime impact, but not a substitute for correctness work. | Preserve reciprocal p-Holm unless a different algorithm is explicitly requested. Describe the supported rho approximation honestly. Correct citations when touching the implementation; do not run a naming campaign and count it as statistical stabilization. |

Rows 11-13 are the main engineering dependency chain: **coherent acceptance/failure semantics -> canonical evidence composition -> normalized ESP**. Row 20 also needs the canonical representation. EP-01 is deliberately not marked fully repaired after its admission checks alone.

The ranking does not justify attaching all desired future capabilities to those repairs. In particular, generic direct-level support is useful only where an accepted source needs it; a broad source registry, persistence system or public transaction API is not a prerequisite to fixing negative evidence, resets or malformed partitions.

## 4. SEQ-01: resolving the distribution objection

For a fixed positive `rho`, the two-sided normal mixture has the form

```text
M_n = sqrt(rho / (V_n + rho)) * exp(S_n^2 / (2 * (V_n + rho))).
```

"Normal" describes the mixing construction. For iid fair signs `X_i in {-1,+1}`,

```text
E[exp(lambda * X_i)] = cosh(lambda) <= exp(lambda^2 / 2).
```

Thus `S_n=sum(X_i)` with `V_n=n` satisfies the relevant sub-Gaussian condition. A beta-binomial construction is another valid choice, not a prerequisite for testing this mean. **H Figure 1, p.2, explicitly uses the two-sided normal mixture on Rademacher observations.** H Definition 1, p.5, and Appendix J, p.47, make the broader assumptions clear.

The code instead substitutes current sample variance and, in Python, chooses new mixture tuning from current observations (`sequential_e_testing.py:279-296`). On the four equiprobable two-observation sign paths:

| Paths | Default Python evidence | Reject at alpha=.05? | Fixed proxy-one evidence |
|---|---:|---|---:|
| `(-1,-1)` and `(1,1)` | `1.2511067969687328e81` | Yes | `0.6236876513637661` |
| `(-1,1)` and `(1,-1)` | `0.24345165829835655` | No | `0.24345165829835655` |

That is an exact 50% rejection probability for the default mean test under this null. The isolated current-source Rust empirical branch likewise rejects two of these four paths; it keeps `rho` fixed but still uses the unjustified sample-variance substitution.

There is also a strictly Gaussian counterexample, so this conclusion does not depend on admitting non-Gaussian observations. Let `X1,X2` be iid `N(0,1)`, `S=X1+X2`, and `D=X1-X2`. Then `S,D` are independent `N(0,2)`. On the event

```text
abs(D) <= .3 and 1 <= abs(S) <= 2.5,
```

the default Python path has `V=max(D^2,.01)` and retunes `rho` proportionally to `V`. Its log capital is between `3.8134480601254865` and `292.5656912627136`, always above `log(20)=2.995732273553991` and below overflow. The event probability is

```text
(2*Phi(.3/sqrt(2))-1) * 2*(Phi(2.5/sqrt(2))-Phi(1/sqrt(2)))
    = 0.06760162097585523.
```

Therefore the default test's Gaussian null rejection probability at `n=2` is **at least 6.7602%**, not necessarily exactly that value.

The proposed fixed-proxy repair is sound for its stated model. It is not the uniquely necessary replacement. H also supports properly justified self-normalized processes: for conditionally symmetric null-centered increments, for example, Table 3 allows `V_n=sum((X_i-mu0)^2)`. That is a different declared null contract, not permission to substitute ordinary centered sample variance. **Choose the intended supported model before deciding that all unknown-variance use must be removed.**

## 5. Verdict on every retained ID

"Confirmed" below means the stated, qualified discrepancy is supported. It does not mean every example violates a paper's assumptions and conclusion simultaneously, or that every proposal is implementation-ready. Numeric witnesses are finite-case evidence; paper assumptions and proofs supply the statistical argument.

### Single-stream, boundaries and quantiles

| ID | Finding verdict and decisive evidence | Proposal verdict / constraint |
|---|---|---|
| SEQ-01 | **Confirmed.** Exact sign-path and Gaussian witnesses above; H Definition 1/Lemma 2/Prop.5. Code: `sequential_e_testing.py:279-311`. | Fixed-proxy construction is sound; supported model and compatibility are decisions, not consequences of the example alone. |
| P01 | **Confirmed.** Native empirical variance has the same assumption failure without Python's retuning. Current-source four-path run rejects 2/4; `rust/par_seqtest/update.rs:390-406`. | Checked fixed proxies are sound. Preserve homogeneous/heterogeneous dispatch; do not confuse ordinary variance with a sub-Gaussian proxy. |
| SEQ-02 | **Confirmed.** Uncentered statistic, wrong scale/time pairing. Current cutoff `16.718790931214897` gives `chi2_19` tail `0.608912532291435`; public calls immediately below/above the cutoff reproduce the decision. Code: `sequential_e_testing.py:361-392`. | Pairing is a valid proposed design, not the only repair. Reconsider using H Example 1 and Appendix H; see section 6. |
| SEQ-03 | **Confirmed.** At `p0=.8,n=20`, exact expected implemented evidence is `14.758453867036758`. Correct signed/reflected versions average to one across `.1,.2,.5,.8,.9` nulls. LESS at `.2` with 11 successes returns NaN. Code: `sequential_e_testing.py:317-357`. | Signed counts and swapped reflected support parameters match H Eqs.(57)-(62), p.29. |
| QUANT-01 | **Confirmed direction error.** GREATER against zero gives `6.330771948299871e22` for 100 values of -1, but `.029847005440418756` for +1. This witness demonstrates reversed direction, not equality-null type-I inflation. Code: `sequential_e_testing.py:427-435`. | Q Eqs.(126)-(127): deficit of inclusive counts for GREATER, excess of strict counts for LESS; state the quantile-set convention. |
| QUANT-02 | **Confirmed atom failure.** A point mass at its null median zero gives `3.563442048939098e23` and rejection. Feasible-tie minimization instead gives log evidence `-1.1512204321549042`. Same calculator. | Q Eq.(125)/Appendix C.5 support a cumulative lower envelope. Its successive ratios are not automatically conditional e-values; certified lower evaluation is required. |
| QUANT-03 | **Confirmed search error.** `arange(20)` versus `arange(20)+5`: alleged log lower bound `.02823384494958603` exceeds feasible/exhaustive `-.5051735494398599`. Code: `quantiletest.py:87-104`. | The cross-plateau search in Q Appendix E, p.33, reproduces the exhaustive answer. Accurate inner minima and endpoint/tie handling remain part of the repair. |
| CS-01 | **Confirmed explicit-None path.** With known variance one and explicit `BoundaryConfig()`, `radius*sqrt(n)=3.03520899037625` at 10,100,1000. The Gaussian LIL contradicts its claimed simultaneous coverage. Code: `sequential_e_testing.py:565-573`. | Fix tuning once and use appropriate directional intervals. The internally generated fixed-`v_opt` default is a different path. |
| CS-02 | **Confirmed theorem mistranslation.** Two constant observations give code intrinsic time zero; H's midpoint-prediction residual time is `.25`. For `[0,1]`, code time is `1`, source time `1.25`. Code: `confidencesequence.py:66-71`. | H Theorem 4/A.8: predictable residuals and finite bounds, with `c=b-a`. This does not imply every sample-variance confidence method is invalid. |
| CS-03 | **Confirmed dimensional error.** Alternating 14/16 within declared `[10,20]` returns `(145.66900367369317,20)`. Code: `confidencesequence.py:132-143`. | Original-unit calculation and final intersection with the range are sound. Removing the extra scaling alone does not repair CS-02. |
| LIFE-01 | **Confirmed lifecycle normalization error.** Known-proxy balanced 1,000-observation prefix, reset, then `[1]`: evidence `46.46270295448182`, versus fresh `.5215213008595565`. Exact second-run null rejection probability `.8052336715382342`. Code: `sequential_e_testing.py:807-825`. | Complete reset is necessary. Revision epochs/tokens are only needed if the chosen architecture introduces retained candidates. |
| NUM-01 | **Confirmed numerical loss.** Normal `(-50,1)` returns `-inf`, reference `-5.060837056835`; 2,000-failure beta witness returns `-inf`, independently scaled integral `-6.496740225305102`. Code: `martingales.py:130-132,315-318`. | Log-tail/`erfcx` evaluation and a reliable logarithmic beta integral are appropriate. Continued-fraction domain, convergence and accuracy still require implementation work. |
| NUM-02 | **Confirmed solver failure.** One-sided normal bound at `v=0,log_threshold=log(20)` raises; a positive bracket finds `.8559380908578476`. Code: `martingales.py:80-86`. | Positive bracketing with progress checks is sound. H Appendix D explicitly discusses the physical beta cap with closed confidence sets; keep that separate. |
| P09 | **Confirmed exactness/naming qualification, not invalid evidence.** At alpha `.05`, exact rho `.12177348869865692`, code `.1260056097925634`. Q Eq.(21), p.10, explicitly gives the code's approximation. Code: `rust/martingale/two_sided_normal.rs:64-73`. | Document approximation and actual parameter domain. An API rename or Lambert-W implementation is not required for validity. |
| P10 | **Confirmed numerical threshold failure.** Fresh native log capital 720 at alpha `1e-310`: Ville/Bonferroni/reciprocal Holm reject none; BH rejects one. Correct threshold `713.8013788281542`. Code: `rust/par_seqtest/mod.rs:134`, `bonferroni.rs:40`, `holm.rs:48`. | Stable log subtraction matches B Fact 7.6. Apply consistently, including overrides and merged/adjusted paths. |

### RIPr, shared evidence, calibration and parallel composition

| ID | Finding verdict and decisive evidence | Proposal verdict / constraint |
|---|---|---|
| RIPr-01 | **Confirmed assumption-exposure gap.** Correlated equal-marginal pairs average evidence 1.36 but violate R's product model; independent fair pairs average one. `ksample_test.py:45-48,224-229`. | State the actual sampling contract. Exogenous buffering is permitted; do not demand that every future block size be fixed at construction. |
| RIPr-02 | **Confirmed retry contamination.** Failed/retried restricted block averages capital `1.0114603015943562`, clean block one, with the same predictable fraction across outcomes. `bernoulli.py:259-262`, `ksample_test.py:136-145`, `eprocessupdater.py:88-94`. | Coherent staged acceptance is sound; public prepare/commit, linked history and revision-token designs are optional. |
| RIPr-03 | **Confirmed parameter conflation.** Prior gamma two passes `KSampleConfig` but even an all-in test fails the unrelated betting-cap constraint. `ksample_test.py:88-93,278-283`. | Separate prior and betting parameters. Changing the adaptive cap default is an explicit compatibility decision. |
| RIPr-04 | **Confirmed empty-grid acceptance.** Additive `delta=.9,K=.5` and log-odds `delta=1,K=.6` produce zero fitted means, neutral evidence and advancing counts. `bernoulli.py:88-136,224-241`. | A disclosed interior-grid convention is sound. Main-text versus appendix transformed-mean ambiguity remains unresolved, not a proven validity defect. |
| RIPr-05 | **Confirmed representation failure.** The three-block witness has reference final log capital `123.13766581780555`; code remains at zero capital and does not reject. `ksample_test.py:140`, `eprocessupdater.py:107-121`. | Sound finite-positive log pipeline; genuine infinity policy must follow the explicitly chosen source contract. |
| RIPr-06 | **Confirmed ingestion mismatch.** Scalar `1.9` becomes one; `.9` and `-.9` become zero; block ingestion rejects them. `ksample_test.py:218-222,298-303`. | Validate before coercion, with failed-trigger buffer semantics. Additional group-ID and dimensionality rules are API policies. |
| EP-01 | **Confirmed invalid-input admission.** Two -10 updates produce capital `[1,-10,100]`, inconsistent logs and rejection; NaN is also committed. `eprocessupdater.py:85-121`. | **Needs correction/qualification:** finite-domain validation is sound; extended-domain rejection is not universally mandated. Source-kind distinctions are necessary when different source kinds are supported, but adding a generic direct-level API is a separate feature. |
| EP-02 | **Confirmed quantity mismatch.** `[0,2]` yields `log(2)` instead of empirical expected log evidence `-inf`. `eprocessupdater.py:164-172`. | Preserve all raw log-factor mass, including genuine zeros. Mixed positive/negative infinite log contributions need a defined diagnostic policy; adaptive factors are not automatically iid samples from one e-power distribution. |
| CAL-01 | **Confirmed endpoint and tail errors.** At one: NaN rather than `.5`; next float below one: approximately one rather than `.5`; at `1e-310`, current log value `677.6998980584054`, reference `700.6601693426999`. `calibrators.py:156-168`. | Sound series/log map; remove the positive-p floor and avoid eagerly evaluated invalid branches. A frozen shape/projection result wrapper is optional. Calibration alone gives marginal, not automatically conditional, e-validity. |
| P03 | **Confirmed invalid/aliased partition.** Current-source Rust with independent `{0,2}` inputs and accepted `[2,1]` cuts averages 1.25. Mutating Python's caller-owned cuts changes valid output to zero/log NaN with `is_valid=True`. `merging.py:429`, `rust/merge/mod.rs:184-200`. | Validate and own the partition at every public/native boundary before mutation. The out-of-range native slice panic was source-inspected, not executed. |
| P04 | **Confirmed finite-log composition failure.** Native observation 50: all-in log `1109.0237505725038`, conservative infinity, adaptive NaN at zero stake. Singleton arithmetic merge on `50,-50` becomes NaN while base log capital remains finite. `rust/par_seqtest/update.rs:177-178,245-246`, `rust/merge/mod.rs:276-345`. | Finite-log fractional/spatial formulas are sound. **Extended-product specification needs correction**; see section 6. This overlaps RIPr-05's root cause. |
| P05 | **Confirmed numerical map and decision failure.** Log input 710: lookback infinity versus `696.8694700599293`; 1500: sqrt infinity versus 750. Native adjusted rejections on `5,100` change 1 to 0. `adjusters.py:159-162,201-204`, `rust/adjusters/mod.rs:108-165`. | Sound full-domain log evaluation. Zero at one is conservative, but not the cited right-continuous admissible version. |
| P06 | **Confirmed neutral-input identity failure.** Native `K=1024,order=512`, all unit inputs gives zero; mathematical answer one. `rust/merge/mod.rs:146-148,234-238`. | Normalized finite-input log ESP and gambling coefficients check out. **Infinity-admitting semantics need specification**, including absent/zero coefficients. |
| P07 | **Confirmed identity difference, not inherently invalid bounded betting.** Native second lambda on `3,0,0` is `.05814240057744242`; Python optimizer gives `.4999999865167874`. `rust/par_seqtest/update.rs:241-251`. | Honest quadratic-rule naming is sound. **Proposed numerical enclosure mechanism remains unverified**; a correct acceptance witness does not establish that mechanism. |
| P08 | **Confirmed name/citation/power difference, not FWER invalidity.** `(30,30)` at `.05`: code rejects neither; L's mean-closure e-Holm rejects both. `rust/multiple_testing/holm.rs:57-64`. | Preserve reciprocal p-Holm and describe it accurately. Mean-closure e-Holm is a different feature, not a mandatory bug fix. |

**Supported baseline retained:** RIPr-00 is not a defect. R Proposition 2/Eqs.(3.2)-(3.4) agree with the regular unrestricted calculation. An unequal-block three-group witness with nontrivial previous history gave code log evidence `.19934770632494314` versus independent high-precision `.19934770632494378`. Surrounding retry/numerical defects do not justify replacing that algebra.

### Parametric, conformal and symmetry

| ID | Finding verdict and decisive evidence | Proposal verdict / constraint |
|---|---|---|
| T-01 | **Confirmed missing sufficient state.** `[1,2]` and `[0,1,2]` share `(S,V)=(3,5)` but have different source log evidence; code gives `.804718956` for both by forcing `n=1`. `ttest_universal.py:65-68,125-127`. | Stable t-specific moments are sound. Include prior-precision and filtration restrictions; do not add an unused `n` parameter to every ordinary mixture kernel. |
| T-02 | **Confirmed equation error; original numerical consequence incorrect.** Actual exponent parsing at the stated n=10 example hits the infinity guard, not `.248843376`. Source radius `1.289928913123774`. `ttest_universal.py:100-111`. | The proposed logarithmic `q` formula matches T Eq.(36). Correct the variance multiplier and t-state integration as well; do not claim a live n=10 high-level CS currently exists. |
| T-03 | **Confirmed latent substitutions.** Correct Lai calibration at `m=2,alpha=.05`: `a=25.438593438097698`, `b=210031.086844926`; radius at n=10, biased variance one is `1.8132123877715423`. Code substitutes three different operations. `ttest_universal.py:160-164`. | Exact calibration, fixed start m and whole-line output before m are sound. Flat extended-martingale infinities are intentional source features. |
| T-04 | **Confirmed unreachable prefix-state mismatch.** One predictor per batch is indexed as one per observation; length-three replay indexes out of range. Factory first fails on missing `TestType.TTEST`. `ttest_universal.py:188-214`. | Observation-indexed predictive loss and the source's null-only denominator shift are sound. Wiring must follow complete reconstruction, not precede it. |
| T-05 | **Confirmed latent inversion/construction defects.** Missing `1/e`, centered instead of raw second moment, and an unconstructible positional Pydantic superclass call. `ttest_universal.py:302,324-338`. | Log-radius inversion is sound; the proposed extreme reference radius `1.7026434277235864e218` is finite. This is not observed coverage failure from a working high-level class. |
| CONF-01 | **Confirmed conformal contract mismatch.** Singleton `[10]` yields `4.1718589556781686e18`; chronological scoring also violates equivariance. `conformal.py:142-154`. | A common fixed raw score plus source normalization is sound, but changes the public model/API. The witness does not refute Howard's kernel under its own null. |
| CONF-02 | **Confirmed batch-clock mismatch.** Eight fair binary triples with unequal 2+1 batching give expected score `1.1958685154241875`; singleton updates average one at each index. `conformal.py:183,191-196`. | Observation-indexed normalization is sound. An explicitly exchangeable fixed-block contract is another choice; arbitrary unequal blocks cannot be inferred to satisfy it. |
| CONF-03 | **Confirmed altered fixed-horizon capital.** Source terminal values `(1,2,0,1)` average one; floor `.1` raises the mean to 1.025 and can revive zero capital. `conformal.py:274,315-321`. | Display-only floor is sound. Default-floor distortion is tiny; the structural issue, not an exaggerated default false-positive rate, justifies repair. Unfloored pseudomartingale optional-stopping failure is not a mistranslation. |
| DET-01 | **Confirmed missed-alarm recurrence.** `(1,1.9,19/13)` should cross two at step three; current zero/floor-started recurrence does not. `cusum.py:60-80`. | `C_n=e_n*max(1,C_previous)` and pre-reset alarm reporting match C Eq.(6). |
| DET-02 | **Confirmed reverse-SR mismatch.** All ones cross three at step three; `(2,.1,3)` gives reverse statistics `(2,2.2,3)`, not forward SR's final 3.9. Same `update` method. | Proposed per-start products/prefix sums implement C Eq.(9). Disclose growing-segment work/state; do not substitute forward SR under the old name. |
| DET-03 | **Confirmed repeated counting.** The documented all-one-score run has one alarm `[4]`, but reports `false_alarm_rate=2.0`. `cusum.py:165-171`. | Event cursor and distinct count/probability/exposure metrics are sound. C's asymptotic false-alarm theorem is not a finite post-change simulation guarantee. |
| SYM-01 | **Confirmed admitted-input/null-contract mismatch.** At lambda `-.5`, 20 zeros produce sign evidence `79.9501989411765` and Wilcoxon `280357.20028473384`. `hypothesistesting.py:229-233,273-298`. | Conditional fair signs with zeros removed from sign/rank counts is a sound declared extension; the original continuous model is another policy. Negative lambda itself and sign-independent nonzero ties are not defects. |
| SYM-02 | **Confirmed numerical failures.** Fisher `[2000]` produces NaN rather than two; sign `1..1500` infinity rather than finite log `328.6052945697579`; Wilcoxon `1..60` NaN rather than log about `40.2696370236952`. `hypothesistesting.py:258-261,294-298`. | Stable per-factor log normalization is sound. Include public result/decision paths, which currently raise on these projections. Last-digit differences within the stated tolerance are not additional bugs. |

## 6. Decisions and corrections that must precede implementation

### 6.1 Variance: compare the source construction before changing the experiment

H Example 1, pp.6-7, and Appendix H, p.45, use the full-sample statistic

```text
S_(n-1) = M2_n / theta - (n-1)
V_(n-1) = 2*(n-1), c=2,
M2_n = sum((X_i - mean(X_1,...,X_n))^2).
```

The proof writes its increments using orthogonal recursive residuals:

```text
Y_t = sqrt(t/(t+1)) * (X_(t+1) - mean(X_1,...,X_t)) / sqrt(theta).
```

At the Gaussian equality null, these residuals are iid standard normal and the increments are `Y_t^2-1`. The current uncentered statistic is wrong, but the paper already explains how to use the running centered sum of squares correctly.

The disjoint-pair proposal instead gets 10 independent squared differences from 20 observations, rather than 19 recursive-residual degrees of freedom. Its reported cutoff/tail, `28.333322995330832` and `.001596136247244685`, were reproduced. A corresponding full-residual source calculation gives cutoff `41.43945681500426` on `chi2_19`, with tail `.0021083308565535714`. These numbers confirm both constructions' arithmetic; they are **not** a complete power comparison.

The important qualification is filtration. The recursive residuals are independent of previous residuals, not generally independent of the entire raw-data past. A running-boundary guarantee must not be silently upgraded to conditional increments or stopped e-values under a larger information set. Disjoint pairs provide a different, explicit completed-pair construction. Choose between these based on the intended testing, stopping and downstream-composition contract, not merely on which is easier to implement.

### 6.2 Extended zero/infinity arithmetic is a declared convention

B Convention 12, printed p.xi/PDF12, generally uses `0*x=0`, including infinite x. Chapter 8, printed p.113/PDF126, defines merger formulas on finite inputs and explains a safe infinity extension on a null-impossible event. The distinct cross-merging construction on printed p.176/PDF189 explicitly chooses `0*infinity=infinity`.

Therefore the EP-01/P04/P06 proposals need an explicit, context-appropriate admitted-domain and endpoint policy. Product-zero absorption, a justified extension on a null-impossible event, and rejecting inputs outside a declared API domain are different choices. The papers do not mandate one global exception policy.

This does not excuse floating-point damage to finite quantities. Finite log evidence 1000 is not genuine infinity merely because its natural projection overflows; finite log evidence -1000 is not genuine zero. The finite-log counterexamples remain confirmed.

Nor does a product convention define an empirical log expectation containing both positive and negative infinity. EP-02 still needs a separate diagnostic-domain decision.

### 6.3 The actual T-02 parse and additional t-state obligations

Python exponentiation is right-associative. At `n=10,c2=1,alpha=.05`, the current denominator uses

```text
alpha ** ((2*c2/(n+c2)) ** (1/n)) = .07996276628865169,
```

while its numerator uses

```text
(alpha ** (2*c2/(n+c2))) ** (1/n) = .9469889450487462.
```

The denominator is `-.12040957082483139`, so the guard returns infinity. The old audit used the numerator's grouping in both places to obtain its incorrect narrow-radius witness. The corrected source radius is finite, `1.289928913123774`. Other parameter choices can yield overly narrow values; the direction is not uniform. The current scalar interface also forces n to one, so these n-specific expression comparisons must remain distinct from claims about an operational t-CS.

Three coupled issues belong in the complete t repair, not separate opportunistic tickets:

1. `prior_precision` is documented as `c2` but is squared again (`ttest_universal.py:55-63`). Supplying `.1` stores `.01`.
2. The CS caller passes `n*M2/(n-1)` where T Eq.(36) requires biased variance `M2/n` (`ttest_universal.py:344-346`). Their ratio is `n^2/(n-1)`, independent of the exponent defect.
3. T Section 4.6.5 explicitly distinguishes crossing validity under arbitrary data-dependent observation from stopped-e-value expectation under a larger filtration. A reduced-filtration statistic cannot silently enter e-BH or spatial merging just because an adapter accepts a cumulative level.

Fixing `TestType.TTEST` or positional Pydantic initialization first would expose wrong formulas. These entry-point changes belong last, or must retain explicit unavailability until the mathematical paths are complete.

### 6.4 P07 needs an algorithm, not an implementation slogan

For past log factors `+1e-20,-1e-20`, high-precision evaluation gives

```text
sum(E_i-1) approximately 1e-40
log(lambda), with epsilon=1e-6, approximately -78.28789316179756
next log factor, when log(E)=80, approximately 1.878032326750534.
```

Separately rounded `expm1` values cancel to zero, so compensation after that loss cannot restore the nonlinear term. This confirms the difficulty, not the proposed solution.

Before approving the expansion/enclosure design, specify its small-log cutoff, truncation bound, range reduction, rounding enclosures, accumulation error, first-moment sign certification, fraction precision, parameter/horizon domain, and behavior when ambiguity cannot be resolved. Measure its cost only after correctness exists. The report does not establish that a bounded-state implementation succeeds on every finite history.

Also retain the source's optimality qualifications: B Theorem 7.22 assumes iid evidence under the alternative, finite expected log evidence, and gamma one. W's authors' reply notes that maximizing generic capital growth need not minimize confidence-set width. Neither exact empirical optimization nor ONS is automatically the mandatory replacement.

### 6.5 Preserve valid constructions and avoid unnecessary infrastructure

The following are not reasons for wholesale replacement:

1. Arithmetic merging remains valid under arbitrary dependence; product-like constructions retain their stated independent/sequential domains.
2. R's regular unrestricted RIPr algebra is supported. The nonlinear main-text/S1 posterior-mean difference is still not a proven validity bug.
3. Reciprocal p-Holm is valid FWER control on valid e-inputs, although its current name is misleading.
4. The fixed rho approximation is explicitly supported as an approximation by Q Eq.(21); it need not be replaced to preserve validity.
5. The untruncated conformal pseudomartingale's fixed-horizon rather than anytime guarantee, and the flat t-process's extended infinities, are intentional.
6. Returning the beta physical cap with a closed confidence set is explicitly discussed in H Appendix D. The withdrawn generic cap-as-type-I-failure claim must stay withdrawn.

Failure safety is still substantive. A retry must not score the same data with a posterior learned from those data. But there are several engineering ways to enforce this: private staged state, validation before mutation, or an explicitly declared fail-stop object after an unrecoverable failure. Choose coherent semantics first; do not introduce public revision-token machinery merely to justify its own use.

Likewise, immutable linked histories do not alone prove constant-time updates. The current updater copies history at `eprocessupdater.py:97` and scans maxima at `:127`. Remove the actual repeated work and demonstrate complexity before making a million-update claim.

## 7. Evidence, limits and preservation

The numerical examples in this report were reproduced against the audited source or independently derived source formulas. Parent diagnostics cover the mean/variance/proportion/quantile/CS/reset/tail/root/threshold cases; reviewers supplied independent integration, sign-orbit and finite-outcome enumerations, normalized-ESP oracles, and current-source native executions. Parent synthesis independently checked the material T-02 correction, the book's endpoint conventions, and native P01/P10 behavior.

The isolated extension was built from the stated revision using `maturin build --locked --offline`, with a fresh session-local target directory. Its SHA-256 was:

```text
9c5e4e902a91e6d31c4fcd069fc59ad8eef94eecc79098e4d328670b9e0a47ec
```

The installed extension was not replaced. This was a debug build, not performance evidence.

Existing focused checks also ran: 137 k-sample/merging/adjuster tests, 14 calibrator tests, and two parametric tests passed; two parametric CS tests remain skipped. Those results do not contradict the findings. In particular, existing conformal tests pin the erroneous tiny CUSUM recurrence and replenished capital floor; correct repairs must replace those expectations rather than preserve them for a green run.

Reproduction scripts, exact downloaded editions, full-reading notes, reviewer reports and native build provenance are retained under this session's `files/` directory:

```text
audit-parent/reproduce.py
audit-parent/reproduction-output.txt
audit-parent/native_reproduce.py
audit-parent/native-output.txt
audit-opus/witness_t.py
audit-opus/witness_t02_sweep.py
audit-opus/witness_conformal.py
audit-opus/witness_sym.py
audit-opus/witness_fullread.py
audit-opus/report.json
audit-astra/diagnostics.py
audit-astra/diagnostic.rs
audit-astra/cal01_diagnostics.py
audit-astra/full-reading-reconciliation.json
audit-astra/full-reading-notes.md
audit-astra/native-provenance.md
```

These are audit artifacts, not a new committed testing framework. Some diagnostics intentionally expose exceptions from the current implementation; their purpose is to record the failure, not to claim the affected API succeeds.

**Limits:** no full-suite or release-performance claim; no independently established global FDR/FWER inflation rates beyond the explicitly derived examples; no executed out-of-range native panic witness; no resolved authorial intent for R Appendix S1; no verified implementation of P07's proposed enclosure or million-update history design. A sound proposed formula is not evidence that an implementation already exists or satisfies its entire domain.

The 2026-09-06 audit preserved `POSSIBLE_BUGS.md`, `PROPOSED_SOLUTIONS.md`, source code and the pre-existing untracked files. On 2026-09-07, the user requested documentation synchronization: the evidence/proposal catalogs, repository guidance, README and historical plan were updated to reflect this report. Implementation remains unchanged; no unresolved public-contract choice is approved by that documentation update.
