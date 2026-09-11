# Proposed solutions

**Reviewed proposal catalog; no implementation repairs yet.** Synchronized on 2026-09-07 with the full-source audit in [AUDIT_REPAIR_PRIORITIES.md](AUDIT_REPAIR_PRIORITIES.md), which governs current verdicts and repair order. All 43 retained IDs are covered, but this is not a blanket approval of the proposed formulas, APIs or architecture.

This document maps the retained findings in [POSSIBLE_BUGS.md](POSSIBLE_BUGS.md) to proposed mathematical corrections and the existing OOP architecture. Source keys and pinned editions are defined in that report's [source table](POSSIBLE_BUGS.md#sources-and-page-conventions). Page references below identify the actual paper operation being restored.

The paper determines the statistical construction. Existing classes, immutable configuration/results, owned streaming state, and Python/Rust boundaries must preserve it. No proposal authorizes implementation before the public-contract changes are approved.

## Current disposition

Start with **P03 (owned partitions), LIFE-01 (complete reset), then SEQ-03 (signed/reflected Bernoulli evidence)**. These are complete repair units, not isolated easy patches. The ranked audit contains the remaining order and dependencies; this catalog's section order is not execution order.

- Most finite-input mathematical corrections are supported under their stated assumptions; they are not implemented.
- SEQ-02's pairing design must be compared with H's existing full-sample variance construction.
- EP-01/P04/P06 need explicit context-appropriate genuine-zero/infinity contracts.
- P07's floating-expansion/enclosure mechanism remains **unverified and not implementation-ready**.
- Named classes, methods, candidates, revision tokens and linked histories below are proposed designs, not existing APIs or paper-mandated infrastructure. Prefer the simplest existing-architecture solution that preserves the required invariant.
- Mean-model support, RIPr cap defaults/grid policy, conformal update shape, symmetry null and t-source integration remain explicit public-contract decisions.

## Candidate corrections

### SEQ-01 — restore a declared fixed-proxy mean construction

**Paper:** H Definition 1, Lemma 2 and Proposition 5, pp.5,9,28/PDF same.

**Proposal:** use the explicit `MeanModelConfig` shared with P01: conditional sub-Gaussian proxy, explicitly Gaussian known variance, or bounded observations with the Hoeffding proxy. Resolve the proxy and rho once, then use `V_n=n*proxy`. Empirical variance remains descriptive; it cannot retune or replace the inferential construction.

**Review qualification:** this is a sound restricted model family, not the only possible repair. H Figure 1 uses normal mixtures on Rademacher data; Appendix J also permits justified self-normalized processes, including null-centered squared observations under conditional symmetry. Choose the supported null class before removing all unknown-variance use. The default failure was also demonstrated for iid Gaussian observations, not only fair signs.

**OOP mapping:** `SequentialTesting` retains test-specific calculator closures, but they return evidence and candidate sufficient state rather than mutate the live object. The complete wrapper publication protocol below includes bounds, diagnostics and result construction.

**Acceptance:** the four fair-sign paths at n=2 with proxy one do not reject at alpha=.05; same-sign capital is .6236876513637661. Unsupported or missing model declarations fail before observations are accepted. No undocumented variance-one fallback remains.

### SEQ-02 — choose the Gaussian variance construction before changing sampling

**Paper:** H Example 1, pp.6-7, Appendix H, p.45, and the sub-exponential/normal-mixture constructions.

**Decision required:** compare the paper's full-sample recursive-residual construction with the disjoint-pair alternative. Both assume iid Gaussian observations with a common unknown mean and tested variance theta>0. Pairing is not prescribed by H and is not approved merely because the current uncentered statistic is wrong.

**Full-sample option:** H uses `S_(n-1)=M2_n/theta-(n-1)`, `V_(n-1)=2(n-1)`, `c=2`. Its increments are squared orthogonal recursive residuals minus one. The residuals are independent of previous residuals, not generally the complete raw-data past; downstream fractions/stopping must respect the chosen filtration.

**Paired option:** preassign disjoint pairs, `Y_j=(X_(2j)-X_(2j-1))/sqrt(2)` and `Z_j=Y_j^2/theta`. At equality, Z is chi-square(1). For `S_m=sum(Z_j-1)`, use gamma-exponential scale c=2 and V=2m: the log MGF is `m*(-log(1-2*lambda)-2*lambda)/2`. It decreases when variance is below theta. For the lower alternative, `-S_m` has normal proxy 2m because `lambda-log(1+2*lambda)/2 <= lambda^2`. A fixed half-mixture supplies a two-sided equality test.

**OOP mapping:** retain the existing variance calculator ownership. A residual design owns cumulative centered moments; a paired design additionally owns pending observations and fractions chosen before either pair member. Preserve whole-call failure semantics without requiring a public token/candidate API. Correct centering, scale, time, alternatives and interval inversion together.

**Reference comparisons:** at twenty observations, the paired upper cutoff is chi-square(10)>=28.333322995330832, with null tail .001596136247244685; the corresponding full-residual cutoff is chi-square(19)>=41.43945681500426, with tail .0021083308565535714. These validate arithmetic, not comparative power. The paired option uses 10 rather than 19 degrees of freedom. Translation invariance, the selected filtration and complete reset/failure semantics belong in acceptance for the chosen design.

### SEQ-03 — preserve signed counts and exchange reflected support parameters

**Paper:** H Eqs. (57)-(62), p.29/PDF 29.

**Proposal:** use the signed count `S=successes-n*p0` for the two-sided mixture. GREATER uses `(g,h)=(p0,1-p0)`; LESS uses `-S` and exchanges those parameters. Do not apply absolute value to an asymmetric likelihood-mixture statistic.

**OOP mapping:** configure the correct `BetaBinomialMixture` once in the existing proportion evaluator. Prepare binary counts and cumulative evidence without publishing them; bounds and the public result share the final wrapper barrier.

**Acceptance:** exact binomial expectations at asymmetric null probabilities satisfy the intended e-bound; the p0=.8,n=20 witness no longer has mean evidence 14.758. Reflection agrees with swapping success/failure labels and p0 with 1-p0.

### QUANT-01 — translate quantile direction into the correct count direction

**Paper:** Q quantile-set definitions and Eqs. (125)-(127), p.26/PDF 26; the H beta kernel used by the wrapper.

**Proposal:** GREATER uses the deficit in inclusive counts, `n*p-count(X<=q0)`, with reflected beta parameters. LESS uses the excess in strict counts, `count(X<q0)-n*p`. State the quantile-set convention explicitly; do not infer that a larger quantile means more observations below q0.

**OOP mapping:** keep the quantile closure and order-statistic abstraction, adding correct observation counts and candidate snapshots. Use the complete wrapper publication protocol, not a kernel-only mutation barrier.

**Acceptance:** evidence favors the correct shifted alternative. Strict/inclusive counts handle atoms consistently; reflection swaps the data direction and p with 1-p.

### QUANT-02 — use the feasible CDF interval without inventing conditional ratios

**Paper:** Q Eqs. (37)-(38), p.14/PDF 14, and Eq. (125), p.26/PDF 26.

**Proposal:** at a candidate quantile minimize the two-sided log mixture over `[Fhat^-(q0),Fhat(q0)]`, representing every feasible allocation of ties. Under the null, the population's fixed tie allocation lies inside that interval, so the resulting level is dominated by the corresponding source process. Numerical minimization must supply a conservative lower certificate, not an unchecked upper approximation.

**OOP mapping:** the quantile calculator returns a cumulative e-process level with its order-statistic candidate. The shared updater observes that level directly; it does not fractionally multiply unproved ratios of successive minima. A fixed cash/evidence mixture is a separately declared direct-level transformation, not adaptive reinvestment.

**Acceptance:** an all-zero population tested at median zero is not rejected through discarded ties. Candidate evidence, counts and result remain unchanged if minimization or subsequent result preparation fails.

### QUANT-03 — search the reference A/B minimization domain

**Paper:** Q Appendix E, p.33/PDF 33, and Theorem 6, p.27/PDF 27.

**Proposal:** order the two minimizer plateaus by their upper endpoints. Search from the first plateau's upper endpoint to the second plateau's lower endpoint, including both endpoints and all distinct second-arm observations between them. If the plateaus overlap, evaluate the overlap endpoint. Uncertain minimizer brackets must enlarge, not shrink, the candidate search.

**OOP mapping:** repair `QuantileABTest.find_log_superMG_lower_bound` while retaining `get_G_fn` and `OrderStatisticInterface`. Evaluate stable snapshots of both arms; no new order-statistics subsystem is required.

**Review qualification:** enlarging the search interval is not sufficient if the inner `G` evaluations overstate their minima. Validate the inner lower values and discrete endpoint/tie handling as part of this local repair.

**Acceptance:** `arange(20)` versus `arange(20)+5` gives `-0.5051735494398599`, not `0.02823384494958603`. Arm swapping, ties and overlapping plateaus agree with an exhaustive breakpoint oracle.

### CS-01 — fix normal-boundary tuning for the whole experiment

**Paper:** H Proposition 5/Eq. (51), p.28/PDF 28; Proposition 3, p.12/PDF 12.

**Proposal:** resolve `v_opt=None` once at construction, never from current intrinsic time. Use one fixed rho for evidence, intervals and reset. Two-sided radius is `sqrt((V+rho)*(log1p(V/rho)-2*log(alpha)))/n`; directional tests return the corresponding half-line, not a two-sided interval using a one-sided error budget.

**OOP mapping:** frozen `BoundaryConfig` selects fixed tuning or explicit `rho`; the wrapper owns the resolved boundary object. Keep functional boundary APIs as adapters. Explicit `rho` and explicit `v_opt` are mutually exclusive.

**Acceptance:** rho does not change between n=10,100,1000 or after reset; explicit `BoundaryConfig()` no longer produces constant `radius*sqrt(n)`. Tuning alpha remains separate from decision alpha.

### CS-02 — implement the actual empirical-Bernstein variance process

**Paper:** H Theorem 4/Eq. (23), p.14/PDF 14; proof Eqs. (89)-(90), p.35/PDF 35.

**Proposal:** require finite declared bounds `[a,b]`. Start prediction at `(a+b)/2`; for each observation add `(x_i-prediction_i)^2` before learning from it. Use the previous running mean for the next prediction. Keep descriptive centered `M2` separate from this intrinsic time. Use gamma scale `c=b-a`, fixed tuning and logarithmic tail allocation.

**OOP mapping:** `ConfidenceSequenceState` owns the residual sum and prediction state. Small next-state/radius hooks select the bounded construction; construct the complete immutable result before publishing state. Unconstrained callers must instead declare a supported known-proxy model.

**Acceptance:** on `[0,1]`, constant prefixes `[0,0]` and `[1,1]` give residual sum .25; `[0,1]` gives 1.25. Batch partitioning preserves the final state.

### CS-03 — keep confidence intervals in one system of units

**Paper:** H Theorem 4 and its original-data mean, p.14/PDF 14.

**Proposal:** retain observations, mean, residual squares and boundary calculation in original units. The subclass intersects the finished interval with `[a,b]`; it does not multiply its endpoints by the range width. A fully normalized implementation would require both input normalization and inverse affine transformation, and is not the selected approach.

**OOP mapping:** `EmpiricalBernsteinConfidenceSequence` selects the bounded configuration and clips the prepared result; it does not rescale an already original-unit parent calculation.

**Acceptance:** alternating 14 and 16 under bounds [10,20] gives an ordered interval containing 15. Affine transformation `x -> 10+10*x`, with squared-unit tuning transformed consistently, transforms the un-clipped endpoints accordingly.

### T-02 — restore the Gaussian t-radius equation

**Paper:** T Theorem 4.9/Eq. (36), p.14/PDF 14.

**Proposal:** compute `logq=(2*log(alpha)+log(c2)-log(n+c2))/n`; the factor is `(alpha^2*c2/(n+c2))^(1/n)`, not an altered power of alpha. Evaluate `1-q` with `-expm1(logq)`. A nonpositive reference denominator gives an infinite radius before multiplication by the biased empirical variance `M2/n`, including when that variance is zero.

**OOP mapping:** use a t-specific `confidence_radius(biased_variance, n, alpha)` operation fed by the t-state owner. Do not overload the ordinary mixture boundary's intrinsic-time argument or pass the current caller's `n*M2/(n-1)` in place of `M2/n`.

**Acceptance:** n=10, c2=1, alpha=.05 and biased variance one give radius `1.289928913123774`; substituting finite endpoints into Eq. (35) reaches the stated threshold. The original audit's `0.248843376` current-code witness is withdrawn: the actual exponent parsing makes the denominator negative and returns infinity. Repair the mathematical path before enabling the currently broken factories.

### T-03 — implement Lai's calibration rather than its asymptotic order

**Paper:** T Theorem 4.1/Eqs. (16)-(18), p.10/PDF 10.

**Proposal:** choose fixed starting `m>=2` and solve `2*(t.sf(a,m-1)+a*t.pdf(a,m-1))=alpha` before observing data. Store `logb=-log(m)+m*log1p(a*a/(m-1))`. For n>=m use `sqrt((M2/n)*expm1((logb+log(n))/n))`; before m return the whole line.

**OOP mapping:** a fixed Lai boundary owned by `TtestConfidenceSequence` stores the calibration. Keep the flat extended martingale separate from ordinary unit-capital e-process accumulation.

**Acceptance:** m=2, alpha=.05 gives a approximately 25.4385934381 and b approximately 210031.086845; at n=10 and variance one, radius is approximately 1.81321238777. Calibration survives reset unchanged; unsuccessful numerical calibration raises explicitly.

### T-04 — make universal t inference observation-indexed

**Paper:** T Theorems 3.2/3.4, Eqs. (9)-(11), p.8/PDF 8.

**Proposal:** accumulate the predictive log-loss using the predictor available before each observation, then update that predictor. Maintain cumulative moments and `A_n=sum(log(sigma_i^2)+standardized_error_i^2)`. Shift only the null-denominator moment for a nonzero null value, not the observations scored by the alternative predictor.

**OOP mapping:** a shared `UniversalTAccumulator` owns the observation count, moment and predictor state. Return cumulative log evidence through the direct-level interface; its ratios are not asserted to be conditional e-values.

**Acceptance:** `[1,2,3]` works as one batch or three observations with identical final A, n and log evidence. Fixed predictors mean zero/scale one give A=14 and raw second moment 14/3. A source level may recover from zero in direct mode.

### CONF-03 — separate display flooring from statistical capital

**Paper:** C Section 3 and Proposition 3, pp.4-5/PDF 6-7.

**Proposal:** statistical capital remains the exact unfloored product. `min_capital` becomes a display-only floor with a distinctly named accessor; it cannot replenish wealth or drive an inferential threshold. Use canonical logs to distinguish a tiny positive product from genuine zero.

**OOP mapping:** `ConformalEPseudomartingale` owns the product and descriptive maximum; `TruncatedEPseudomartingale` supplies only the display view. Preserve the intentionally fixed-horizon, non-anytime guarantee.

**Acceptance:** terminal values `(1,2,0,1)` retain mean one with a .1 display floor. Genuine product zero remains zero after a positive factor. Non-unit initial capital is explicitly normalized when reporting e-evidence.

### DET-01 — restore the CUSUM starting contribution

**Paper:** C Eq. (6), p.8/PDF 10.

**Proposal:** use `C_n=e_n*max(1,C_previous)`, or `logC=logE+max(0,logC_previous)`. Record the triggering statistic and alarm before resetting the active segment. Numerical floors affect display only.

**OOP mapping:** retain `ConformalCUSUM` as the owner of the transition. Return a frozen result carrying the pre-reset statistic and whether a new alarm occurred.

**Acceptance:** `(1,1.9,19/13)` crosses threshold two at step three. `(0,3)` can start a new product and alarm at step two. An all-one predictor never alarms for threshold greater than one.

### DET-02 — implement the actual reverse Shiryaev-Roberts procedure

**Paper:** C Eq. (9) and Example 7, p.9/PDF 11.

**Proposal:** for each active start i retain its current product `P_i` and accumulated prefix sum `R_i`. On new e, update `P_i*=e`, `R_i+=P_i`, append the new start `(e,e)`, and take `max_i R_i`. Use logs/log-addition and clear only the active segment after an alarm.

**OOP mapping:** two segment-local arrays in the existing `use_sr` branch replace summing historical aggregate statistics. This is explicitly reverse SR, not the simpler forward-SR recurrence.

**Acceptance:** all ones alarm every third observation at threshold three. `(2,.1,3)` gives reverse statistics `(2,2.2,3)`, not forward SR's 3.9. Segment-linear work is disclosed rather than hidden as constant-space behavior.

### DET-03 — count newly occurring alarms once

**Paper:** C Eq. (7), p.8/PDF 10.

**Proposal:** maintain a per-trial `seen_alarm_count` cursor and inspect only newly appended alarm times. Separate mean prechange alarms per trial, probability of any prechange alarm, and alarms per unit exposure. Handle no-detection and zero-exposure cases explicitly.

**OOP mapping:** the detector produces immutable event records; `EfficiencyAnalyzer` owns trial counters and censoring. Retain the old ambiguous metric only as a documented deprecated alias.

**Acceptance:** a history containing one alarm at time four contributes one event, not one event on every later update. Two real alarms give count two but any-alarm probability one.

### LIFE-01 — reset every component of the configured experiment

**Paper:** H Definition 1 and mixture initialization, pp.5,9/PDF 5,9.

**Proposal:** constructor and reset use one run-state initializer. Reset sufficient statistics, prior cumulative evidence, pending observations, histories and owned strategy state while preserving immutable model/tuning configuration. A stateful external predictor or callback must have an explicit reset/factory contract.

**OOP mapping:** `SequentialTesting` creates fresh run state from its existing configuration and initializer. Do not change v_opt to one or preserve learned state accidentally. If a later design introduces retained candidates or fraction tokens, reset must invalidate them; introducing such machinery is not a prerequisite to this repair.

**Acceptance:** after 1,000 balanced Rademacher observations and reset, `[1]` gives `0.5215213008595565`, identical to a fresh known-proxy test, not `46.46270295448182`.

### NUM-01 — evaluate the source's positive tail integrals logarithmically

**Paper:** H Eqs. (52),(59), pp.28-29/PDF 28-29.

**Proposal:** use `log_ndtr` for normal tails; in the cancelling negative-tail regime evaluate the entire mixture through `erfcx`. Implement one shared logarithmic incomplete-beta routine using a checked continued fraction and complement/log-difference identities, rather than logging an underflowed CDF.

**OOP mapping:** numerical evaluation belongs to the existing mixture kernels and one shared beta helper; callers do not add clipping or alternate accumulators. Convergence/range failures occur before state publication.

**Acceptance:** the normal witness returns approximately -5.060837056835; the 2,000-failure beta witness approximately -6.4967402253051. Endpoint and branch-transition identities use independent reference calculations.

### NUM-02 — bracket legitimate zero-time roots

**Paper:** H Definition 1 and Eqs. (52),(64)-(65), pp.5,28,30/PDF same.

**Proposal:** start an unbounded root search at a positive value such as `max(1,sqrt(v))`, or a source-derived analytic upper bound. Check representable progress and bracket validity; never double zero repeatedly. Keep the separate finite-beta support question out of this repair.

**OOP mapping:** repair `find_s_upper_bound` and `find_mixture_bound`; no caller invents positive variance merely to make the solver run.

**Acceptance:** the one-sided normal root at v=0 and threshold 20 is approximately .8559380908579381. Positive v approaching zero approaches that root.

### P01 — expose supported mean models, not a plug-in variance promise

**Paper:** H Definition 1, fixed mixture, Propositions 5-6 and bounded-range proxy.

**Proposal:** use one shared frozen `MeanModelConfig`: `sub_gaussian` with explicit `variance_proxy`; `gaussian_known_variance` with explicit `variance`; or `bounded` with finite bounds resolving to `(b-a)^2/4`. All use fixed rho and `V_n=n*proxy`. Reject missing assumptions and the empirical inferential mode rather than silently substitute variance one.

**Review qualification:** these are proposed supported modes, not the uniquely required API. A supplied ordinary variance is not automatically a sub-Gaussian proxy. Decide the model/compatibility contract alongside SEQ-01; justified self-normalized alternatives remain possible under their own assumptions.

**OOP mapping:** Python resolves the declared model; the native generic engine consumes the checked scalar or fixed per-stream proxies through its existing homogeneous/heterogeneous branches. Preserve monomorphization, SoA and hoisted dispatch.

**Acceptance:** same-sign Rademacher pairs with proxy one give capital .6236876513637661, not rejection. Vector/mode/domain mismatches fail before mutation. Unknown-variance Gaussian and bounded variance-adaptive methods remain separately named constructions.

### P03 — own and validate the mathematical partition

**Paper:** M segment-product definition, p.10/PDF 10.

**Proposal:** validate integer K, ordered unique internal boundaries and exact vector lengths; store an immutable owned partition and precompute native ranges. An empty boundary sequence consistently denotes one segment/arithmetic mean. Caller mutation cannot change the validated partition.

**OOP mapping:** frozen partition configuration, `SegmentProductMerger`, and `MergeConfig::validate(n_tests)` enforce the same contract at Python, PyO3 and direct Rust boundaries. Remove skipped reversed windows and late unchecked slices.

**Acceptance:** reject `[2,1]`, `[0]`, `[K]`, duplicates and out-of-range boundaries before updates. K=3, boundary `[2]`, values `[2,4,6]` gives 18; an empty partition gives 4.

### P05 — keep adjuster evaluation in the log domain

**Paper:** A Section 4/Eq. (5), p.6/PDF 6.

**Proposal:** for large x=log E, use `log A1=x-2*log(x)+log1p(-(1+x)*exp(-x))` in a stable evaluation; near zero use the positive Taylor series. Use a stable log-expm1 expression/series for A2, including positive subnormal x. Genuine infinity maps to infinity before indeterminate arithmetic.

**OOP mapping:** retain the adjuster ABC/classes and Rust enum, replacing only numerical kernels. Preserve the existing conservative zero-at-one convention with accurate admissibility wording.

**Acceptance:** A1 log input 710 gives 696.8694700599293; A2 log input 1500 gives 750. Increasing maxima cannot lose a rejection through NaN adjustment.

### P08 — preserve and honestly name reciprocal p-Holm

**Paper:** L Section 2.3 versus Theorem 2.1/Eq. (2.2), pp.5-6/PDF same.

**Proposal:** expose `reciprocal_p_holm` and its adjusted counterpart; retain the existing valid rank-threshold algorithm. Deprecate the misleading old names with warnings and truthful method metadata. Do not silently redirect existing calls to the more powerful mean-closure algorithm.

**OOP mapping:** update wrapper methods, enums, PyO3 and native naming consistently. No new state layout or sorter is required.

**Acceptance:** `(30,30)` at alpha=.05 still rejects neither under reciprocal p-Holm, while documentation explicitly distinguishes literature e-Holm's two rejections. Implementing the latter would be a separately approved feature.

### P09 — identify approximate tuning accurately

**Paper:** H Proposition 3/Eq. (21), p.12/PDF 12, gives the exact optimum; Q Eq. (21), p.10/PDF 10, explicitly gives the logarithmic approximation used in the code.

**Proposal:** retain and accurately document the fixed logarithmic approximation, with `L=-log(alpha_opt)` and stable `log1p`. An `approximate_rho` alias/deprecation of `best_rho`, and a Python fixed-`rho` constructor matching Rust's existing `from_rho`, are optional compatibility decisions. No rename or Lambert-W implementation is required for statistical validity.

**OOP mapping:** Python and Rust kernel constructors and provenance use the same name/rule. One-sided doubled-alpha tuning remains an explicitly supported approximation, with its actual domain enforced.

**Acceptance:** preserve .1260056097925634 at v_opt=1, alpha_opt=.05 while distinguishing the exact optimizer .12177348869865692. No online retuning or new Lambert-W dependency is introduced.

### P10 — compute thresholds without reciprocals

**Paper:** B Fact 7.6/Eq. (7.4a), p.98/PDF 111.

**Proposal:** use `-log(alpha)` for Ville, `log(K)-log(alpha)` for Bonferroni, and the corresponding log-rank term for reciprocal p-Holm. Validate overrides at every public/native boundary. Do not materialize alpha allocations or reciprocals that underflow/overflow first.

**OOP mapping:** all scalar, merged, adjusted and native decisions consume the same checked log-threshold semantics; natural presentation never controls a decision.

**Acceptance:** alpha=1e-310 and log evidence 720 rejects consistently; the threshold is 713.8013788281542. Values immediately below, at and above a threshold respect non-strict comparison.

### RIPr-01 — make block/sampling assumptions part of the public contract

**Paper:** R Eq. (2.1), pp.11-13/PDF same.

**Proposal:** distinguish predeclared blocks from explicitly declared exogenous-arrival buffering. Preserve unequal group sizes and permitted adaptation based on completed blocks. Do not claim that numerical validation proves independence or that current-block outcomes may determine completion.

**OOP mapping:** make the sampling and ingestion obligations explicit while retaining the weighted null projection. A configuration field may record the design if needed; a new sampling framework is not required to document the contract. Pending observations do not become evidence.

**Acceptance:** exogenous labels `0,0,1` emit one block with sizes `(2,1)`. Correlated equal-marginal examples are documented as outside the product model, not as theorem failures.

### RIPr-03 — separate the prior from the betting cap

**Paper:** R Section 5, p.19/PDF 19; B Definition 7.21(iii), p.107/PDF 120.

**Proposal:** retain `gamma` as the positive Beta-prior parameter and add a separate betting cap in (0,1], mapped explicitly to `EProcessConfig.gamma` in construction and reset. `betting_gamma=.5` is a proposed default, not an approved migration; decide compatibility before changing existing adaptive users' fractions.

**OOP mapping:** the calculator consumes the prior only; the updater consumes the betting configuration only. Document .18 as the source's experimental recommendation in its particular setting.

**Acceptance:** prior gamma=2 constructs successfully; changing betting_gamma does not change fixed-history RIPr predictions. Adaptive callers requiring the former .18 cap specify it explicitly.

### RIPr-05 — carry RIPr evidence to decisions without exponentiating it

**Paper:** R Eq. (3.4), p.16/PDF 16; B Eq. (7.10), p.107/PDF 120.

**Proposal:** pass the calculator's log factor directly to the shared temporal updater. Current capital, maximum, p-values and rejection are derived from canonical logs. Natural-scale values are labelled projections; genuine zero is not confused with underflow.

**OOP mapping:** the calculator supplies evidence, `EProcessUpdater` supplies fractional accumulation, and `EProcess` owns canonical state. The orchestrator does not maintain a competing capped product.

**Acceptance:** the three-block witness ends at log capital 123.13766581780555 and rejects. Finite logs -1000 and +1000 recover to zero log capital; genuine zero in a product does not recover through a finite factor.

### RIPr-06 — validate the observation before coercion

**Paper:** R Bernoulli counts, Eq. (3.1) and Proposition 2, pp.15-16/PDF same.

**Proposal:** share one binary validator between scalar and block ingestion. Accept only explicitly supported finite numeric values exactly equal to zero or one before conversion. Validate integral nonboolean group IDs and one-dimensional, nonempty group arrays.

**OOP mapping:** `KSampleSequentialTest` owns ingestion and buffering; the calculator receives validated integer successes/sizes. A failed completing call retains old buffers but does not accept its triggering observation.

**Acceptance:** 1.9, .9, -.9, NaN and infinity fail without changing counts/buffers. Valid scalar ingestion and the corresponding direct block produce the same sufficient statistics.

### EP-02 — preserve zero mass in empirical e-power

**Paper:** B Definition 3.11, p.40/PDF 53.

**Proposal:** average every committed raw conditional log factor, including genuine negative infinity from zero evidence. Empty history returns no estimate; opposite infinite contributions report an undefined expectation. Realized capital growth is a separately named diagnostic. An adaptive sequence of factors is not automatically an iid sample estimating one population e-power.

**OOP mapping:** the updater reads the canonical log-history view; it does not filter natural projections. Direct cumulative-level mode does not invent a conditional-increment e-power statistic.

**Acceptance:** `[0,2]` gives negative infinity; canonical logs `[-1000,log(2)]` give their finite mean. Conservative betting on `[2,2]` distinguishes raw log(2) e-power from log(1.5) capital growth.

### SYM-02 — evaluate the original normalized factors stably

**Paper:** S Eqs. (4.4),(7.3),(8.1)-(8.3), printed pp.264,267,268/PDF 4,7,8.

**Proposal:** add canonical `compute_log`. Evaluate Fisher, sign and signed-rank factors through per-factor softplus/log-normalizer identities, not separately overflowing aggregate numerator/denominator exponentials. Derive natural values and decisions from the log result.

**OOP mapping:** `SymmetryETest` remains the mathematical calculator; frozen results distinguish finite log evidence from natural projection overflow. The zero/tie policy is an explicit model choice, not an accidental numeric shortcut.

**Acceptance:** Fisher `[2000]` at lambda=.5 is approximately 2; sign `1..1500` has log evidence 328.6052945697579; Wilcoxon `1..60` has log evidence 40.269637023695275.

## Reconciled temporal-state proposals

### Complete `SequentialTesting` publication boundary

The required invariant is coherent whole-call acceptance for SEQ-01, SEQ-02, SEQ-03, QUANT-01 and QUANT-02. The staging approach below is a candidate implementation, not a required public protocol.

Retain the existing calculator closures and ownership. They can prepare next sufficient state and evidence privately, then prepare the temporal update, bounds, calibration, diagnostics, history and result before replacing live state. Fractions must be chosen at the declared evidence clock. Public `prepare/commit` APIs, revision tokens and linked-history types are not necessary consequences of the audit.

If the paired variance design is selected, a multi-pair call must preserve the selected whole-call failure semantics and fractions fixed before each pair opens. Private staged-prefix calculations are one option. Do not introduce pair state into the full-sample residual design merely because the earlier proposal assumed pairing. Raw-observation count, evidence-step count and API-call count remain distinct.

Under the proposed retryable, failure-atomic contract, any failure in input validation, callbacks, kernels, minimization, bounds, diagnostics, history or result preparation accepts none of the current call; pre-call pending data survives. An explicitly fail-stop object is another possible contract, but silently continuing with partially learned state is not. History representation and complexity must be demonstrated rather than inferred from immutability.

**Acceptance:** for the selected contract, inject failures at the actual calculation, temporal-update and result-publication boundaries. Retryable failures leave every owner and previous snapshot unchanged, and retry matches a clean call. Add pair-opening cases only if pairing is selected. This is synchronous exception safety, not crash durability or rollback of external callback side effects.

### T-01 — supply stable sufficient moments to the t-specific kernel

**Paper:** T Eqs. (32),(35), pp.13-14/PDF same.

**Proposal:** the kernel receives actual n, centered M2 and null-centered V through a t-specific moment view. Evaluate Eq. (35)'s denominator as `n*M2+c2*V`, never as `(n+c2)*V-S^2` or a ratio rounded to one. With log moments use positive-sum arithmetic for both denominator and prefactor. Return Gaussian log G1=0 exactly on its nonzero domain; for M2=0,V>0, use `(n-1)/2*(log(n+c2)-log(c2))`.

**OOP mapping:** `TtestSufficientState` owns anchored, compensated centered moments and a common scaling convention; `TtestMomentView` exposes the required n/log moments. Do not add unused n arguments to every ordinary mixture kernel. Flat H's genuine extended infinities remain separate.

**Integration prerequisites:** the state/view classes are one design, not mandatory abstractions. Resolve `prior_precision`: the current constructor documents c2 but squares it again. T Section 4.6.5 restricts downstream use of the reduced-filtration Gaussian-mixture statistic; raw-data stopping need not produce an e-value for e-BH or merging. Direct-level handling does not upgrade its filtration guarantee. Correct the t-specific arithmetic and supported consumption contract before enabling factories.

**Acceptance:** c2=1e-20 gives log G1=0 for `[1]` and 23.37242452022043 for `[1,1]`, not spurious infinity. Constant, near-constant and scaled data agree with the source equation. Unresolved moment positivity/range is an explicit numerical failure, not a fabricated zero or infinity.

### T-05 — invert universal t evidence through a log radius

**Paper:** T Eqs. (9)-(11), p.8/PDF 8.

**Proposal:** restore the missing `1/e` and use the correct centered-variance inversion. Set `w=(A_n-2*log(alpha))/n-1` and `d=log(M2)-log(n)`. For w>d, compute `log_radius=.5*(w+log1mexp(d-w))`; if M2=0, use w/2. Do not exponentiate W first. Resolve empty, singleton and interval cases from a checked difference; uncertain numerical equality is not an exact singleton.

**OOP mapping:** `TtestConfidenceSequence` consumes `UniversalTAccumulator`, not the generic gamma-CS updater. Its frozen result records set kind, center, canonical log radius and projection status. A finite interval whose projected endpoints overflow is not mathematically unbounded.

**Current state:** `TtestConfidenceSequence` cannot currently be constructed because it calls Pydantic's initializer positionally. This and the missing `TestType.TTEST` are final wiring prerequisites, not low-risk fixes to enable ahead of the formulas and clocks.

**Acceptance:** x=sqrt(1000), n=1, fixed predictor mean zero/scale one and alpha=.05 give log W=1004.991464547108 and finite radius 1.7026434277235864e218. Genuine empty sets remain empty; overflowing W alone does not produce the whole line.

### CAL-01 — expose the finite logarithm of the corrected calibrator

**Paper:** B Eq. (2.2), p.23/PDF 36.

**Proposal:** set f(1)=.5 exactly and remove the current `1e-300` floor for positive p. For L=-log(p) near zero use the positive series `.5+L/6+L^2/24+L^3/120+L^4/720`, with justified cutoffs/remainder bounds. Evaluate canonical log f through stable intermediate/large-L formulas; only p=0 is genuinely infinite. Handle p>1 as zero per the calibrator definition. Mask branches so unused 0/0 expressions are never evaluated.

**OOP mapping:** a scalar/array `PToECalibrator.calibrate_log` can preserve shape and expose canonical logs; a frozen projection-status result is optional. Existing natural calls remain projections. A finite projected infinity must not be reclassified as a genuine infinite input, and calibration of a marginal p-variable does not automatically produce a conditional sequential e-value.

**Acceptance:** p=1 gives .5; p=1e-310 has log f=700.6601693426999. The smallest positive binary64 p has finite log f=731.214807212407 although its natural projection overflows. p=0 is explicitly different.

### CONF-01 — normalize a common fixed raw score into a conformal e-measure

**Paper:** C Eq. (1), p.2/PDF 4, and CP's normalization construction.

**Proposal:** retain normal-mixture scoring only as a raw nonnegative score. Freeze its reference location, scale and tuning, apply the same scoring function to every observation, then normalize the score vector. Remove chronological mean/variance standardization from this fixed-score mode.

**OOP mapping:** a small `ConformalScore` abstraction supplies normal-mixture and likelihood-ratio scores; `ConformalEValue` owns normalization. Canonical log output feeds pseudomartingale/detector log interfaces; natural values are compatibility projections.

**Numerical clarification:** compute normalized log scores from centered log-softmax values, or a running maximum plus scaled-sum pair. Do not subtract a collapsed huge log normalizer. For log scores `[0,1e300]`, the last log e-value is log(2), not zero; equal huge log scores normalize to one each. Handle all-zero raw scores separately using the paper's all-ones convention.

**Acceptance:** a singleton, including `[10]`, returns one. Permuting observations permutes the normalized score vector, whose mean is one. No Gaussian null is implied merely by choosing a normal-mixture raw score.

### CONF-02 — make the normalization clock observation-based

**Paper:** C Eqs. (10),(12), p.10/PDF 12; exchangeability in Proposition 1.

**Proposal:** `update` processes one observation; `update_many` returns the ordered observation-level results. Normalize each common raw score against the prefix through that observation. Do not replace observations by products over arbitrarily sized batches.

**OOP mapping:** reuse the common scorer and centered normalization state from CONF-01. Provide log-valued update paths so downstream products do not reconstruct logs from projected zeros. Reset the observation count and normalization state together; stage a batch before publication.

**Acceptance:** any partition of the same ordered stream produces the same e-value sequence. The eight fair binary triples with likelihood scores .1/1.9 give mean E3=1, rather than 1.1958685154241875. A fixed-block alternative would require its own declared equal-size/exchangeability contract and is not silently inferred.

### EP-01 — preserve the source kind, actual initialization and numerical domain

**Paper:** B Definition 7.3, p.97/PDF 110; Definitions 7.19/7.21, pp.106-107/PDF 119-120.

**Proposal:** bind the process to a declared source kind and filtration. Conditional increments use fractional/product accumulation. Generic cumulative e-processes use direct-level observation, never unproved consecutive ratios. Direct initialization requires the source's actual step-zero log value; only a genuinely unit-starting source is initialized at one.

**OOP mapping:** preserve `EProcess` ownership of canonical state and `EProcessUpdater` ownership of temporal composition. Private staging can provide coherent validation/publication. Source descriptors and public `prepare/commit` methods are optional. Supporting generic direct levels is a separate feature, needed only for sources selected for integration; it is not required merely to reject negative/NaN increments.

**Initialization acceptance:** let A be F0-measurable with probability .5, source E0=.5, and all later levels 1.5 on A and .5 otherwise. Its maximal stopped expectation is one; padding E0 to one would permit 1.25. This requires the stated nontrivial F0. Direct E0=0 may later become positive; product-zero absorption must not be applied to direct levels.

**Domain acceptance:** negative inputs, NaN and invalid fractions fail before mutation. Finite log evidence 1000 is distinct from genuine infinite evidence. Preserve the actual evidence maximum; clamp only the derived p-value to at most one.

**Unresolved extended-domain contract:** see the shared log-fraction section below. The finite-domain repair is supported, but the earlier universal requirement to reject mixed zero/infinity products was not justified by B's context-dependent conventions. Choose and document admission and product semantics before implementing infinity support.

### RIPr-02 — prepare the complete block before publishing any owner

**Paper:** R Eqs. (2.6),(3.4), pp.14,16/PDF same; B Definition 7.19.

**Proposal:** make scoring pure. Prepare next posterior/count state from the committed previous-block state, apply the already chosen predictable fraction to a staged temporal update, and prepare result/history/buffer state before accepting the block. The failure/retry contract must cover all owners, not only the calculator.

**OOP mapping:** retain calculator, updater and orchestrator ownership. Private next-state staging is sufficient if it provides the chosen failure semantics. A revision/token protocol may be useful if retained candidates are introduced, but is not required and must not become a transaction service.

**History design option:** immutable links and a read-only prefix view are one option, not an approved requirement. Remove actual repeated work before claiming constant-time fixed-strategy updates: the current updater copies history for fractions and scans maxima. Exact empirical optimization may intentionally scan history. Measure complexity and allocation for the selected representation.

**Acceptance:** the failed/retried restricted block has the same four factors `(.96,1.44,.64,.96)` and mean one as a clean call. A failed triggering scalar observation is not accepted, while earlier pending observations remain. An O(T) history-allocation or million-update performance claim remains unverified until the selected implementation demonstrates it.

### Shared log-fraction and publication contract

Use canonical `eta=log(lambda)`, not a rounded natural fraction:

```text
B(-infinity, ell) = 0
B(0, ell)         = ell
B(eta, ell)       = logaddexp(log(-expm1(eta)), eta+ell), for finite eta<0
```

Only exact `eta=-infinity` is no stake. `eta=-1000, ell=1000` gives log(2), even though displayed lambda is zero. A near-zero negative eta preserves a positive cash component even if displayed lambda rounds to one.

The finite-log formula and endpoint branches above are supported. For admitted genuine infinities, choose a context-appropriate convention rather than treating IEEE NaN as the mathematical definition. B Convention 12 (printed p.xi/PDF 12) generally uses `0*infinity=0`; Chapter 8 (p.113/PDF 126) permits an infinity extension on a null-impossible event; cross-merging (p.176/PDF 189) uses `0*infinity=infinity`. Product-zero absorption, a justified extension and a finite-domain API that rejects infinities are different explicit choices.

Check admission first, then use the selected fraction and product endpoint rules before indeterminate arithmetic. Under the ordinary no-stake convention the factor is one, even when admitted raw evidence is infinite. Finite-log underflow/overflow is never genuine zero/infinity. Mixed positive and negative infinite log contributions also require a separate e-power diagnostic policy.

Coherent publication remains required by the selected failure semantics. Immutable candidates, revision checks and a particular single-writer mechanism are implementation options, not established guarantees of an unimplemented design. Neither crash durability nor arbitrary concurrent access is supplied by this proposal.

## Reconciled parallel numerical proposals

### P04 — carry canonical log evidence and fractions through every composition

**Paper:** B Eq. (7.10), p.107/PDF 120; M's merging constructions, pp.9-10/PDF same.

**Proposal:** use the shared log-fraction operation for temporal updates. Spatial arithmetic mean uses log-sum-exp minus log K; product sums logs; lambda-product sums log-fraction factors; segment-product sums segment log means. No intermediate exponentiation feeds another statistical calculation.

**Review status:** supported for the stated finite-input operations; genuine-infinity/mixed-zero batch semantics require the explicit shared-domain decision above. Do not blindly sum `+infinity` and `-infinity` logs and call the resulting NaN a source requirement.

**OOP mapping:** existing merger subclasses receive log-input operations; Rust mirrors them in its existing enum/SoA architecture. Reusable candidate-log workspace and a commit pass keep fallible calculations ahead of mutation without cloning the entire engine. The sole Python temporal evaluator remains owned by `EProcessUpdater`.

**Spatial interface:** add `log_gambling_system(past_log_e_values,k) -> eta`. A natural `gambling_system` adapter must raise a representation-loss error if an interior fraction projects to zero or one; it must not feed that rounded endpoint back into reconstruction.

**Acceptance:** log evidence 1109.0237505725038 with no stake produces log factor zero, not NaN. A one-element arithmetic merge equals its base all-in process on `50,-50`. Exact zero/infinity policies operate on the evaluated betting factor, not raw evidence.

### P06 — normalize the ESP recurrence instead of dividing overflowing quantities

**Paper:** M Eq. (13), p.10/PDF 10.

**Proposal:** maintain normalized elementary symmetric sums:

```text
u[k,j] = (1-j/k)*u[k-1,j] + (j/k)*E_k*u[k-1,j-1].
```

Store logs, update j in descending order, and omit absent endpoint terms. Keep order-zero/mean/product fast paths. Equal log-mixture arguments return that argument exactly, preserving neutral inputs.

**OOP mapping:** replace numerical internals of `UStatisticMerger` and the native U-statistic enum branch; reuse O(order) workspace. The existing class/merger family is unchanged.

For gambling reconstruction, compute coefficients `A=(1-order/K)*U_order` and `B=(order/K)*U_(order-1)` on the other `K-1` coordinates: past values plus the remaining unit placeholders, excluding the current coordinate. Preserve `eta=-softplus(log A-log B)` instead of returning `exp(eta)` as the authoritative fraction. Omit absent U-statistics and exact zero coefficients before evaluation or ratios.

**Review status:** the finite-input recurrence and extreme gambling witness are supported. Infinite gambling histories may be excluded by an explicit API contract; infinity-admitting batch semantics still need the shared-domain decision. Do not infer a mandatory mixed-zero/infinity rejection policy from the finite-input proof.

**Acceptance:** K=1024, order=512 and unit inputs give log U=0. For K=3, order=2, two past logs of 1000 require eta approximately -999.3068528194401; a third log of 1000 contributes log(3) and reconstructs final log U=2000. Small cases agree with independent subset enumeration.

### P07 — give the quadratic strategy an honest identity and cancellation-aware state

**Paper:** W supplement GRAPA/aGRAPA/ONS-m distinction, printed pp.45-48/PDF 11-14; B Definition 7.21, p.107/PDF 120.

**Review status: numerical mechanism unverified; not implementation-ready.** Honest quadratic-rule naming is supported, but the expansion/enclosure sketch below is not yet a specified, demonstrated algorithm.

**Proposal:** keep the moment rule under the distinct name `QUADRATIC_ADAPTIVE`, not ONS or exact empirical optimization. Preserve the specified ratio `sum(E-1)/(epsilon+sum((E-1)^2))` with a bounded cancellation-aware representation, rather than a scaling-only accumulator.

**Candidate numerical design:** maintain scaled first/second moments as two-component floating expansions plus outward error bounds and a symbolic power-of-two scale. For small log evidence L, add L and the nonlinear remainder `sum_(j=2..18) L^j/j!` separately, with an explicit remainder bound. Otherwise use a range-reduced exponential enclosure. Ordinary compensation after rounding `expm1(L)` is insufficient.

Select cash only when the first-moment enclosure certifies nonpositivity. For a positive moment, resolve the log fraction within the declared precision; otherwise raise a numerical-ambiguity error rather than invent an exact zero stake. A numerical failure cannot be silently treated as an observation to discard and resample.

**OOP mapping:** the optional bounded moment cache belongs to the process/revision/reset epoch, not a freely shared combiner. Prepare and publish its candidate with capital. Keep one named Python reference subclass and mirror it in Rust; the exact Python empirical optimizer remains a different strategy.

**Acceptance:** past logs `(+1e-20,-1e-20)` with epsilon=1e-6 yield a positive first moment near 1e-40 and eta approximately -78.28789316179756. Next log evidence 80 gives log factor approximately 1.878032326750534, not zero. Reset, stale-candidate and independent high-precision checks cover the cache. Numerical enclosure implementation and cost must be established before claiming production performance; this proposal does not claim every finite history is resolvable.

**Before approval:** specify parameter/log-input domains, small-log cutoffs, truncation and rounding bounds, range reduction, accumulation error, first-moment sign and fraction certification, ambiguity behavior, horizon limits and cost. B Theorem 7.22's exact-optimizer guarantee requires iid alternative evidence, finite expected log evidence and gamma=1; it does not transfer automatically to a smaller cap or this quadratic rule. W's authors' reply also distinguishes capital growth from confidence-set width.

## Public-contract choices requiring approval before implementation

### RIPr-04 — use a disclosed interior-grid convention

**Paper:** R Appendix S1, p.35/PDF 35, and main-text Eqs. (3.3)-(3.4).

**Recommendation:** validate the effect-dependent grid domain, construct a nonempty interior grid by integer indices, and normalize a finite discrete prior/posterior. Reject empty or numerically unrepresentable configurations before an experiment starts. Keep both marginal posterior means from the main text.

**OOP mapping:** `KSampleConfig` owns cross-field domains; `BernoulliRIPrCalculator` owns the explicit grid and normalization. The proposal does not silently resolve S1's contradictory endpoint or nonlinear transformed-mean instruction.

**Acceptance:** additive delta=.2, K=.2, gamma=1 gives theta_a `(.2,.4,.6)`, theta_b `(.4,.6,.8)` with equal weights. Delta=.9, K=.5 is rejected. The interior convention is a proposed public policy, not claimed authorial intent.

### SYM-01 — explicitly support conditional fair signs

**Paper:** S's conditional sign-orbit argument, printed pp.263,267-268/PDF 3,7-8.

**Recommendation:** name the supported extension `conditional_fair_signs`: conditional on magnitudes, nonzero signs are independent fair coins; zero coordinates contribute unit factors. Sign/Wilcoxon calculations remove zeros before counting/ranking; ranks use a sign-independent deterministic tie rule. The original continuous model is a possible stricter alternative.

**OOP mapping:** `SymmetryETest` owns the declared null, preprocessing and fixed finite lambda; results expose original and effective sample sizes. This is not automatically a sequential-increment source.

**Acceptance:** all-zero nonempty samples give evidence one; averaging over the eight sign assignments for magnitudes `(0,1,1,2)` gives one. Adopting this named extension instead of restricting to the original continuous model requires approval.
