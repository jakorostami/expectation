# POSSIBLE_BUGS: paper-to-code evidence

Audit date: 2026-09-06. Documentation synchronized: 2026-09-07. Scope: `expectation/` and `rust/` at revision `6f1ef40a1a0661ded83c30ffa7b7f53b00a93804` (version 0.6.1).

**Status: the full-source audit is complete; implementation repairs have not started.** The current adjudication, source-reading coverage and ranked repair order are in [AUDIT_REPAIR_PRIORITIES.md](AUDIT_REPAIR_PRIORITIES.md). Start with **P03, LIFE-01, then SEQ-03**. Severity labels here are not execution order.

The 43 retained IDs include overlapping findings, assumption-exposure gaps, numerical failures, naming differences and latent branches; they are not 43 independent violations of error guarantees. This evidence catalog has been synchronized with the full-document review, including the corrected T-02 consequence. Proposal approval remains separate from confirming a finding.

The paper dictates the statistical construction. OOP and production engineering dictate its structure, ownership, composition, and lifecycle. Both must hold together. A correct formula does not excuse a state transition that changes the experiment; a well-structured class does not excuse a different formula.

Every retained entry below identifies the citation chain, source location, reference operation, current code, translation diff, and limited consequence established. The complete source-text corpus was read across the parent and two reviewers, including proofs and appendices; the parent adjudicated the reports rather than accepting reviewer conclusions automatically. Reading coverage does not imply independent verification of every theorem or plot in those papers.

**Reading the entries:** `REFERENCE` versus `IMPLEMENTED` blocks compare mathematical operations or state semantics; they are not proposed patches or purported quotations of paper Python code. Code fences labelled with repository locations are actual excerpts. A contract gap, numerical failure, optimization difference, and violated error guarantee are different findings. Latent branches are explicitly distinguished from working public entry points.

Source and implementation files were not changed by the audit. Native reproductions used a freshly compiled isolated current-source build, not the stale installed extension. No fixes are included.

## Sources and page conventions

Each entry gives both the printed page and physical PDF page. A header citation, a dependency-linked construction, and a bibliography-traced source are explicitly distinguished; no uncited module is silently assigned a paper.

| Key | Exact source examined | Page convention |
|---|---|---|
| H | Howard, Ramdas, McAuliffe and Sekhon, *Time-uniform, nonparametric, nonasymptotic confidence sequences*, [arXiv:1810.08240v9](https://arxiv.org/pdf/1810.08240v9), August 2022 | Printed = PDF |
| Q | Howard and Ramdas, *Sequential estimation of quantiles with applications to A/B testing and best-arm identification*, [arXiv:1906.09712v5](https://arxiv.org/pdf/1906.09712v5), July 2022 | Printed = PDF |
| T | Hongjian Wang and Aaditya Ramdas, *Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance*, [arXiv:2310.03722v5](https://arxiv.org/pdf/2310.03722v5), November 2024 | Printed = PDF |
| R | Turner, Ly and Grunwald, *Generic E-Variables for Exact Sequential k-Sample Tests that allow for Optional Stopping*, [arXiv:2106.02693v3](https://arxiv.org/pdf/2106.02693v3), June 2022; repository `genericevar.pdf` | Printed = PDF |
| B | Ramdas and Wang, *Hypothesis Testing with E-values*: local `hypotevalues.pdf` has title date September 10, 2025; downloaded [arXiv:2410.23614v6](https://arxiv.org/pdf/2410.23614v6) has title date September 11. Both editions were read and compared; they are not identical. | PDF = printed + 13 |
| C | Vovk, Nouretdinov and Gammerman, *Conformal e-testing*, [Working Paper 29](https://www.alrw.net/articles/29.pdf), revision November 2, 2024 | PDF = printed + 2 |
| CP | Vovk, *Conformal e-prediction*, [arXiv:2001.05989v5](https://arxiv.org/pdf/2001.05989v5), May 2025; not the later version now served by the unversioned author URL | Printed = PDF |
| S | Vovk and Wang, *Nonparametric E-tests of Symmetry*, NEJSDS 2 (2024), 261-270, [doi:10.51387/24-NEJSDS60](https://doi.org/10.51387/24-NEJSDS60), [publisher PDF](https://nejsds.nestat.org/journal/NEJSDS/article/69/file/pdf) | Printed 261 = PDF 1 |
| M | Vovk and Wang, *Merging sequential e-values via martingales*, [arXiv:2007.06382v3](https://arxiv.org/pdf/2007.06382v3), February 2024; repository `merging.pdf` | Printed = PDF |
| A | Tavyrikov, Goeman and de Heide, *Carefree multiple testing with e-processes*, [arXiv:2501.19360v2](https://arxiv.org/pdf/2501.19360v2), July 2025; repository `carefree.pdf` | Printed = PDF |
| W | Waudby-Smith and Ramdas, *Estimating means of bounded random variables by betting*, JRSSB 86(1), 1-27 (2024), [doi:10.1093/jrsssb/qkad009](https://doi.org/10.1093/jrsssb/qkad009), including the published supplement | Supplement printed 45/46/48 = PDF 11/12/14 |
| L | Hartog and Lei, *Family-wise Error Rate Control with E-values*, [arXiv:2501.09015v1](https://arxiv.org/pdf/2501.09015v1), January 2025 | Printed = PDF |

Some code headers say Ramdas-Wang 2024 while citing proposition numbers from 2025. The 2024 v1/v2 editions were also examined: their product/betting results are Proposition 6.10/Definition 6.11, rather than 7.20/7.21. Entries identify the edition actually supplying the numbered result.

The full-document review also found incorrect merger citation labels in code headers: B Theorem 8.4 concerns weighted-mean merging, not a product theorem. Product/U-statistic identities appear in Eqs. (8.3)-(8.4); the retained numerical findings use M's verified formulas. See the ranked audit for exact edition and complete-reading provenance.

## Single-stream and confidence-sequence evidence

### SEQ-01 / P01 — empirical variance and retrospective mixture tuning

**Type:** construction/assumption mismatch; **critical**.

**Citation chain:** the single-stream wrapper is uncited, but constructs the normal mixtures from `expectation/modules/martingales.py:12`, which cites H. The native kernel directly cites H at `rust/martingale/two_sided_normal.rs:11-14`.

**Paper:** H Definition 1, Eq. (4), p.5/PDF 5, requires `exp(lambda*S_t - psi(lambda)*V_t) <= L_t(lambda)` for an appropriate supermartingale. Lemma 2, Eq. (13), p.9/PDF 9, uses a fixed mixing distribution; Proposition 5, Eq. (51), p.28/PDF 28, fixes `rho > 0`. Adapted variance processes are allowed when that domination holds.

**Code:** [Python evaluator, lines 289-296](expectation/seqtest/sequential_e_testing.py#L289-L296), excerpts:

```python
v = max(self.data_count * var_estimate, 0.01)
```

```python
self.v_opt = v / self.data_count  # Update optimal intrinsic time
if self.alternative == AlternativeType.TWO_SIDED:
    self.mixture = TwoSidedNormalMixture(self.v_opt, self.alpha_opt)
```

The [Rust empirical branch, lines 396-403](rust/par_seqtest/update.rs#L396-L403) similarly substitutes:

```rust
let var_est = (new_sum_sq / cnt_f - mean * mean) * (cnt_f / (cnt_f - 1.0));
(cnt_f * var_est).max(0.01)
```

**Translation diff:**

```text
REFERENCE:   m(S_t, V_t; fixed rho), with the sub-psi condition established.
PYTHON:      m(S_t, max(t*sample_variance_t,.01); rho chosen from current data).
RUST:        current sample-variance substitution; unit variance before activation.
```

**Consequence:** on iid fair +/-1 observations, the default Python test rejects the true mean-zero null for `[1,1]` and `[-1,-1]`, but neither opposite-sign pair: exact rejection probability **0.5 at n=2, alpha=.05**. Same-sign capital is `1.2511067969687328e81`. Rust with `variance=None, min_samples=1` also rejects exactly half the paths. Keeping rho fixed does not repair the substituted variance: `TwoSidedNormalMixture(1,.05).log_superMG(2,.01)` exponentiates to `2343347.287942827`.

This disproves the instantiated construction, not H's theorem. H's p.18/PDF 18 discussion of "Naive SN" also explicitly warns against treating estimated variance as sufficient for the normal-mixture guarantee.

**Distribution clarification:** "normal mixture" describes the mixing construction, not a requirement that observations be Gaussian. H Figure 1, p.2, explicitly uses it on Rademacher observations. Fair signs satisfy `E[exp(lambda*X)]=cosh(lambda)<=exp(lambda^2/2)`, so fixed `V_n=n` is justified. A beta-binomial mixture is another valid choice, not a prerequisite for this mean test.

**Strictly Gaussian witness:** for iid `X1,X2~N(0,1)`, let `S=X1+X2` and `D=X1-X2`. On `abs(D)<=.3` and `1<=abs(S)<=2.5`, the default Python log capital lies in `[3.8134480601254865,292.5656912627136]`, above `log(20)` without overflow. Independence of S and D gives this event probability `0.06760162097585523`. The Gaussian null rejection probability is therefore **at least 6.7602%**, not necessarily exactly that value. This does not invalidate other justified adapted variance processes; H Appendix J includes self-normalized constructions.

### SEQ-02 — uncentered variance statistic in a centered exponential process

**Type:** kernel-assumption mismatch; **critical**.

**Citation chain:** `SequentialTesting._setup_variance_test()` calls `GammaExponentialMixture`, whose header cites H.

**Paper:** H Definition 1, p.5/PDF 5, and Proposition 9, Eqs. (64)-(67), p.30/PDF 30:
`m(s,v) = integral exp(lambda*s - psi_E,c(lambda)*v) dF(lambda)`,
where `psi_E,c(lambda) = [-log(1-c*lambda)-c*lambda]/c^2`, so `psi_E,c'(0)=0`.

**Code:** [sequential_e_testing.py:381-387](expectation/seqtest/sequential_e_testing.py#L381-L387):

```python
chi_squared_stat = (self.data_count - 1) * sample_var / self.null_value

s = chi_squared_stat
v = self.data_count - 1
```

**Translation diff:**

```text
REFERENCE:   an exponential-process condition with zero first-order null drift.
IMPLEMENTED: S=Z, V=nu, where Z~chi-square(nu) under the Gaussian variance null.
```

**Consequence:** `log E[exp(lambda*Z - nu*psi_E,c(lambda))] = nu*lambda + O(lambda^2) > 0` for small positive lambda, contradicting the required expectation bound. With `GammaExponentialMixture(100,.05,sqrt(2))`, the threshold at `v=19` is `16.718790931214897`; `scipy.stats.chi2.sf(threshold,19)` is **0.608912532291435**. Thus the n=20 Gaussian variance-one null is rejected 60.89% of the time, not 5%. Centering alone is not asserted to repair the complete sequential construction.

**Repair context:** H Example 1, pp.6-7, and Appendix H, p.45, already give a full-sample construction with `S=M2_n/theta-(n-1)`, `V=2(n-1)`, and `c=2`, using orthogonal recursive residuals. The proposed disjoint-pair alternative is not mandatory: compare information use and filtration/downstream-stopping contracts before changing the sampling design. Previous-residual independence does not establish independence from the full raw-data past.

### SEQ-03 — asymmetric Bernoulli statistics are reflected incorrectly

**Type:** transformation mismatch; **high**.

**Citation chain:** the wrapper calls `BetaBinomialMixture`, directly citing H in `modules/martingales.py:12`.

**Paper:** H Propositions 7-8, Eqs. (57),(59), and their proof Eqs. (60)-(62), p.29/PDF 29. For `g=p, h=1-p`, the likelihood powers use signed `s=K-n*p`, `v=n*p*(1-p)`. Reflecting S exchanges g and h.

**Code:** [sequential_e_testing.py:342-350](expectation/seqtest/sequential_e_testing.py#L342-L350), excerpts; the constructor retains `(g,h)=(p,1-p)`:

```python
if self.alternative == AlternativeType.LESS:
    s = self.data_count * self.null_value - self.data_sum
```

```python
else:  ## two-sided
    s = abs(self.data_sum - self.data_count * self.null_value)
```

**Translation diff:**

```text
REFERENCE two-sided: m_(p,1-p)(K-np,v).
IMPLEMENTED:         m_(p,1-p)(abs(K-np),v).
REFERENCE reflected: m_(1-p,p)(np-K,v).
IMPLEMENTED:         m_(p,1-p)(np-K,v).
```

**Consequence:** summing the implemented two-sided evidence against `Binomial(20,.8)` probabilities gives **14.758453867**, rather than at most one. For `less`, null p=.2 and a block with 11 successes/9 failures produces NaN. These are binary inputs; this finding does not depend on the separate beta-boundary cap question.

### QUANT-01 / QUANT-02 — quantile direction and atoms are not translated into the kernel's count process

**Type:** direction and kernel-assumption mismatches; **high**.

**Citation chain:** the wrapper calls the H-cited beta kernel. Q is explicitly cited by the separate `modules/quantiletest.py:9-14`; its quantile construction is corroborating primary evidence, not a claim that the wrapper calls `QuantileABTest`.

**Paper:** H p.29/PDF 29 requires the signed, centered sub-Bernoulli process. Q Eqs. (37)-(38), p.14/PDF 14, construct
`S_t = sum(1{X_i<Q(p)} + pi(p)*1{X_i=Q(p)} - p)`,
with atom allocation chosen to center the increments. Q Eqs. (125)-(127), p.26/PDF 26, minimize over the feasible CDF interval and exchange p with 1-p for reflection.

**Code:** [sequential_e_testing.py:427-435](expectation/seqtest/sequential_e_testing.py#L427-L435):

```python
count_below = self.order_stats.count_less(self.null_value)
prop_below = count_below / n

if self.alternative == AlternativeType.GREATER:
    s = (prop_below - self.quantile) * n
elif self.alternative == AlternativeType.LESS:
    s = (self.quantile - prop_below) * n
else:  # TWO_SIDED
    s = abs(prop_below - self.quantile) * n
```

**Translation diff:**

```text
REFERENCE direction: a larger quantile corresponds to fewer observations below q0.
IMPLEMENTED GREATER: rewards an excess below q0.
REFERENCE atoms:     allocated ties, or minimization over [Fhat^-,Fhat].
IMPLEMENTED:         strict below-count alone, followed by absolute deviation.
```

**Consequence:** at median level .5, `greater` against zero gives evidence `6.330771948299871e22` for 100 observations equal to -1, but `0.029847005440418756` for 100 equal to +1: reversed direction, not a false-positive claim under an equality-only null. Separately, the iid null population `X=0` has median zero; the tie allocation is .5 and the reference centered statistic is zero. The implemented two-sided test returns `3.563442048939098e23` and rejects. Its strict-count assumption is not justified by that quantile null.

### QUANT-03 — A/B minimization searches the wrong interval

**Type:** algorithm mistranslation; **high**.

**Citation chain:** Q is directly cited at `modules/quantiletest.py:9-14`.

**Paper:** Q Theorem 6, Eqs. (128),(131), p.27/PDF 27, uses `inf_x [G1(x)+G2(x)]`. Appendix E, p.33/PDF 33, requires endpoints `x^- = first-arm upper minimizer endpoint`, `x^+ = second-arm lower minimizer endpoint`, and second-arm observations between them.

**Code:** [quantiletest.py:88-99](expectation/modules/quantiletest.py#L88-L99), excerpts:

```python
G1, _, _ = first_arm_G
G2, x2_lower, x2_upper = second_arm_G
```

```python
min_value = min(objective(x2_lower), objective(x2_upper))
start_index = self.order_stats(second_arm).count_less_or_equal(x2_lower)
end_index = self.order_stats(second_arm).count_less_or_equal(x2_upper)
```

**Translation diff:**

```text
REFERENCE:   search the gap between the two arms' minimizer plateaus.
IMPLEMENTED: discard the first arm's endpoints; search the second plateau only.
```

**Consequence:** construct `QuantileABTest(.5,100,.05,StaticOrderStatistics(arange(20)),StaticOrderStatistics(arange(20)+5))`. Its alleged log lower bound is **0.02823384494958603**, while its own `G1(12)+G2(12)` equals **-0.5051735494398599**. A lower bound cannot exceed this feasible value. This directly breaks the minimization required by Appendix E, without needing a simulated error-rate claim.

### CS-01 — current-time tuning is substituted for one fixed normal boundary

**Type:** fixed-construction mismatch; **high**.

**Citation chain:** wrapper -> `boundaries.normal_mixture_bound()` -> H-cited normal mixture.

**Paper:** H Proposition 5, Eq. (51), p.28/PDF 28:
`u_rho(v)=sqrt((v+rho)*log((v+rho)/(alpha^2*rho)))`, with one fixed rho. Proposition 3, Eq. (21), p.12/PDF 12, selects a tuning horizon.

**Code:** [sequential_e_testing.py:567-572](expectation/seqtest/sequential_e_testing.py#L567-L572):

```python
radius = boundaries.normal_mixture_bound(
    v=np.array([self.intrinsic_time]),
    alpha=alpha,
    v_opt=self.boundary_config.v_opt or self.intrinsic_time,
    alpha_opt=self.boundary_config.alpha_opt,
    is_one_sided=is_one_sided,
```

The [callee, boundaries.py:61-65](expectation/modules/boundaries.py#L61-L65), constructs a new mixture from that `v_opt`.

**Translation diff:**

```text
REFERENCE:                  u_rho with fixed rho for every prefix.
IMPLEMENTED when v_opt=None: recreate rho_t proportional to current V_t.
```

**Consequence:** `SequentialTesting("mean",0,known_variance=1,boundary_config=BoundaryConfig())` returns radius `3.03520899037625/sqrt(n)` at n=10,100,1000. The law of the iterated logarithm implies eventual crossing with probability one for iid Gaussian observations. The zero-data probes identify the deterministic radius; they are not the probabilistic counterexample. The internally generated fixed-`v_opt` default is a different path.

### CS-02 — centered sample variance replaces predictable prediction residuals

**Type:** theorem-construction mismatch; **high**.

**Citation chain:** H is directly cited at `confseq/confidencesequence.py:25`.

**Paper:** H Theorem 4, Eq. (23), p.14/PDF 14, and proof Eqs. (89)-(90), p.35/PDF 35, require bounded observations, predictable bounded predictions, scale `c=b-a`, and `V_t=sum_i (X_i-Xhat_i)^2`.

**Code:** [confidencesequence.py:66-71](expectation/confseq/confidencesequence.py#L66-L71):

```python
if new_n_samples > 1:
    new_sum_squares = self.state.sum_squares + np.sum(
        (data - new_running_mean) * (data - self.state.running_mean)
    )
    variance_estimate = new_sum_squares / (new_n_samples - 1)
    intrinsic_time = new_n_samples * variance_estimate
```

**Translation diff:**

```text
REFERENCE:   sum_i (X_i - prediction fixed before observing X_i)^2.
IMPLEMENTED: t/(t-1) * sum_i (X_i-current_sample_mean)^2.
```

**Consequence:** both all-zero and all-one two-observation prefixes give the implemented value zero. No single predictable first prediction can equal both possible first observations, so this cannot generally be the theorem's residual process. This establishes nontranslation of Theorem 4, not a claim that every conceivable sample-variance confidence method is invalid.

### CS-03 — the subclass rescales endpoints that are already in original units

**Type:** confidence-inversion mismatch across inheritance; **high**.

**Citation chain:** direct H citation in `confidencesequence.py`; subclass calls its parent.

**Paper:** H Theorem 4, Eq. (23), p.14/PDF 14, centers the interval at the original-data mean. The proof's normalization to [0,1], p.35/PDF 35, does not authorize multiplying an already-original-scale center.

**Code:** the parent uses `new_sum = self.state.sum + np.sum(data)` and `new_running_mean = new_sum / new_n_samples`. The [subclass, lines 132-137](expectation/confseq/confidencesequence.py#L132-L137), then does:

```python
result = super().update(data)
range_width = self.config.upper_bound - self.config.lower_bound

return ConfidenceSequenceResult(
    lower=max(self.config.lower_bound, result.lower * range_width),
    upper=min(self.config.upper_bound, result.upper * range_width),
```

**Translation diff:**

```text
REFERENCE:   original-data mean +/- original-unit radius.
IMPLEMENTED: width * (original-data mean +/- parent radius), then clipping.
```

**Consequence:** with declared bounds [10,20] and 100 observations alternating 14 and 16, the result is **(145.66900367369317,20.0)**. All observations satisfy the input bounds. The reversed interval is a direct dimensional contradiction, independent of CS-02.

### LIFE-01 — reset retains the previous experiment's normalization

**Type:** initialization/prefix mismatch across object lifecycle; **high**.

**Citation chain:** mean wrapper -> H-cited normal mixture.

**Paper:** H Definition 1, p.5/PDF 5, starts `S_0=V_0=0`; Lemma 2, p.9/PDF 9, gives the mixture with `m(0,0)=1`.

**Code:** [sequential_e_testing.py:310-311](expectation/seqtest/sequential_e_testing.py#L310-L311):

```python
log_e_sequential = log_e_cumulative - self.previous_log_e_cumulative
self.previous_log_e_cumulative = log_e_cumulative
```

[Reset, lines 807-825](expectation/seqtest/sequential_e_testing.py#L807-L825), clears sums, count, history and `EProcess`, but never resets `previous_log_e_cumulative`; initialization sets that field to zero at line 199.

**Translation diff:**

```text
REFERENCE fresh run:     E_1 = m(S_1,V_1)/m(0,0).
IMPLEMENTED after reset: E_1 = m(S_1,V_1)/m(S_old,V_old).
```

**Consequence:** use known variance one, update `[-1.,1.]*500`, reset, then update `[1.]`. Capital is **46.46270295448182**, versus **0.5215213008595565** for a fresh object. Under actual iid variance-one Rademacher data, exact binomial averaging over the first 1,000 observations gives second-run rejection probability **0.8052336715382342**. This uses an in-model null rather than relying on the earlier constant-zero example.

### NUM-01 — a floating-point CDF zero replaces a positive source quantity

**Type:** numerical evaluation mismatch; **medium**.

**Citation chain:** direct H citation in `modules/martingales.py`.

**Paper:** H Proposition 6, Eq. (52), p.28/PDF 28, contains `Phi(s/sqrt(v+rho))`; Proposition 8, Eq. (59), p.29/PDF 29, contains a positive incomplete-beta integral on the stated domain.

**Code:** [martingales.py:130-132](expectation/modules/martingales.py#L130-L132), excerpt:

```python
+ np.log(stats.norm.cdf(s / np.sqrt(v + self.rho)))
```

[Lines 315-318](expectation/modules/martingales.py#L315-L318):

```python
def log_incomplete_beta(a: float, b: float, x: float) -> float:
    if x == 1:
        return log_beta(a, b)
    return np.log(special.betainc(a, b, x)) + log_beta(a, b)
```

**Translation diff:** logarithm of a positive mathematical CDF/integral becomes logarithm of a floating-point value rounded to zero.

**Consequence:** `OneSidedNormalMixture(1,.05).log_superMG(-50,1)` returns `-inf`; the same source expression evaluated with `scipy.special.log_ndtr` is **-5.060837056835**. `BetaBinomialMixture(25,.05,.5,.5,True).log_superMG(-1000,500)` also returns `-inf` although its defining integral is positive and finite. No substantial false-positive probability is inferred from these tail cases.

### NUM-02 — zero intrinsic time prevents the root search from advancing

**Type:** numerical algorithm mismatch; **medium**.

**Citation chain:** H-cited `martingales.py`, used by one-sided normal and gamma-exponential bounds.

**Paper:** H Definition 1 permits `V_t=0`; Propositions 6 and 9, Eqs. (52),(64)-(65), pp.28,30/PDF 28,30, define finite positive threshold roots at `v=0`, `alpha in (0,1)`.

**Code:** [martingales.py:80-86](expectation/modules/martingales.py#L80-L86):

```python
trial_upper_bound = float(v)
for _ in range(50):
    if mixture.log_superMG(trial_upper_bound, v) > log_threshold:
        return trial_upper_bound
    trial_upper_bound *= 2
```

**Translation diff:** the reference requires finding a positive root; at `v=0`, the implementation explores only `0,0,0,...`.

**Consequence:** `OneSidedNormalMixture(1,.05).bound(0,np.log(20))` raises `RuntimeError`. So does the generic confidence-sequence update on `[0.,0.]`. The low-level zero-time domain is legitimate even independently of the caller's variance-process defect.

### CAL-01 — the mixture calibrator loses its finite endpoint and positive tail

**Type:** numerical evaluation mismatch; **medium**.

**Citation chain:** B Section 2.3 is directly cited at `modules/calibrators.py:9-14`; the method cites Eq. (2.2).

**Paper:** B Eq. (2.2), p.23/PDF 36:
`f(p)=integral_0^1 kappa*p^(kappa-1) dkappa = (1-p+p*log(p))/(p*(-log(p))^2)`.
The integral gives `f(1)=1/2`; p=1 is in the stated domain.

**Code:** [calibrators.py:160-167](expectation/modules/calibrators.py#L160-L167), excerpts:

```python
numerator = 1 - p + p * log_p
denominator = p_safe * (-log_p) ** 2
```

```python
result = np.where(
    p <= 1, numerator / denominator, 0.0
)
```

**Translation diff:** the integral's endpoint `1/2` is replaced by the indeterminate quotient `0/0`.

**Consequence:** the mixture calibrator returns `nan` at 1 and approximately **1.0000000000000002** at `np.nextafter(1.,0.)`, rather than a value close to .5. The constant p-variable equal to one is valid.

The full review also confirmed that the `1e-300` positive-input floor at lines 156-158 changes the tail: at `p=1e-310`, current log output is `677.6998980584054`, versus reference `700.6601693426999`; the smallest positive binary64 p has finite reference log output `731.214807212407`. Repairing only the endpoint or adding a log accessor while retaining the floor is incomplete.

## Parametric t-process evidence

These are source-level mathematical/state mismatches. Entries marked latent do not establish a working public workflow or measured coverage errors from a high-level t-CS API. `create_ttest()` fails on the missing `TestType.TTEST`; `TtestConfidenceSequence` itself fails its positional Pydantic superclass initialization. Correcting these entry points before the mathematical paths would activate broken formulas.

### T-01 — the abstraction omits the observation count

**Type:** sufficient-state mismatch; **high, latent integration**.

**Citation chain:** direct T Theorems 4.7/4.9 citations in `parametric/ttest_universal.py:12-17,52-53,121-122`.

**Paper:** T Eq. (35), p.14/PDF 14:
`G_n = sqrt(c^2/(n+c^2))*((n+c^2)*V_n/((n+c^2)*V_n-S_n^2))^(n/2)`,
with `S_n=sum X_i`, `V_n=sum X_i^2`. Eq. (32), p.13/PDF 13, also requires n.

**Code:** [ttest_universal.py:65-68](expectation/parametric/ttest_universal.py#L65-L68):

```python
def log_superMG(self, s: float, v: float) -> float:
    n = 1
    if hasattr(s, "__len__"):
        n = len(s)
```

The [caller, lines 262-268](expectation/parametric/ttest_universal.py#L262-L268), passes scalar sums from `np.sum(data_centered)` and `np.sum(data_centered**2)`.

**Translation diff:** reference state `(S_n,V_n,n,c)` becomes `(S_n,V_n,c)` with scalar S forcing n=1.

**Consequence:** `[1,2]` and `[0,1,2]` both give `(S,V)=(3,5)`, but the paper's log values are **0.366984588** and **0.203608321**. The code returns **0.804718956** for both. Passing or owning n is a design choice; omitting it cannot implement the equation.

**Coupled obligations:** the constructor documents `prior_precision` as `c^2` but squares it again at `ttest_universal.py:63`; supplying .1 stores .01. T Section 4.6.5 also restricts downstream use of the reduced-filtration Gaussian-mixture statistic: raw-data stopping preserves its crossing guarantee but need not preserve stopped expectation at most one. An adapter must not silently treat such values as ordinary e-inputs to e-BH or merging.

### T-02 — the Gaussian t-radius changes a product into an exponent

**Type:** equation mistranslation; **high, latent**.

**Citation chain:** direct T Theorem 4.9 citation in `ttest_universal.py:86-90`.

**Paper:** T Eq. (36), p.14/PDF 14, uses `q=(alpha^2*c^2/(n+c^2))^(1/n)` in
`radius^2=(n+c^2)*(1-q)/(((n+c^2)*q-c^2) vee 0) * empirical_variance`.

**Code:** [ttest_universal.py:100-102](expectation/parametric/ttest_universal.py#L100-L102):

```python
denominator = (alpha ** (2 * self.c_squared / (n + self.c_squared)) ** (1 / n)) * (
    n + self.c_squared
) - self.c_squared
```

The numerator at lines 107-111 has different grouping:

```python
* (1 - (alpha ** (2 * self.c_squared / (n + self.c_squared))) ** (1 / n))
```

Python exponentiation is right-associative; these are not the same altered expression.

**Translation diff:**

```text
REFERENCE q:             (alpha^2 * c^2/(n+c^2))^(1/n).
IMPLEMENTED denominator: alpha^((2*c^2/(n+c^2))^(1/n)).
IMPLEMENTED numerator:   (alpha^(2*c^2/(n+c^2)))^(1/n).
```

**Corrected consequence:** at n=10, c^2=1, alpha=.05 and biased variance one, the reference radius is **1.289928913123774**. The code's denominator uses `q=.07996276628865169`, while the numerator uses `.9469889450487462`; the denominator is `-.12040957082483139`, so the guard returns **infinity**, not the previously reported `0.248843376`. The equation remains wrong, but its error direction is parameter-dependent. This expression comparison is separate from the scalar interface's missing-n defect and is not a successful public t-CS run. Infinite radius when the *reference* denominator is nonpositive is intentional.

**Coupled variance-scale defect:** the CS caller at `ttest_universal.py:344-346` supplies `n*M2/(n-1)` where Eq. (36) requires biased variance `M2/n`, a ratio of `n^2/(n-1)`. This belongs in the complete t-state/radius repair, not a separate claim that a working public CS has measured undercoverage.

### T-03 — Lai's radius replaces three specified operations

**Type:** equation/algorithm mistranslation; **high, latent**.

**Citation chain:** direct T Theorem 4.1 citation in `ttest_universal.py:144-159`.

**Paper:** T Eqs. (16)-(18), p.10/PDF 10, with fixed starting m>=2: solve `2*(1-F_(m-1)(a)+a*f_(m-1)(a))=alpha`; then `b=(1/m)*(1+a^2/(m-1))^m`, `radius=sqrt(v*((b*n)^(1/n)-1))`. Eq. (21), p.11/PDF 11, is only asymptotic.

**Code:** [ttest_universal.py:160-164](expectation/parametric/ttest_universal.py#L160-L164), excerpts:

```python
a = alpha ** (-1 / (m - 1))
```

```python
b = (1 / m) * (1 + a / 2 * (m - 1) / m)
radius = np.sqrt(v) * ((b * n) ** (1 / n) - 1)
```

**Translation diff:**

```text
REFERENCE a: solve the stated Student-t equation.
IMPLEMENTED: use an asymptotic-order expression as equality.
REFERENCE b: m^-1*(1+a^2/(m-1))^m.
IMPLEMENTED: m^-1*(1+(a/2)*(m-1)/m).
REFERENCE radius: sqrt(v*((bn)^(1/n)-1)).
IMPLEMENTED: sqrt(v)*((bn)^(1/n)-1).
```

**Consequence:** the expressions are algebraically different before rounding or execution. No public coverage-rate claim is made while the factory/clock remains broken.

### T-04 / T-05 — universal-inference predictor state and inversion differ from the paper

**Type:** prefix-state and equation mismatches; **high, latent**.

**Citation chain:** direct T Theorems 3.2/3.4 references in `ttest_universal.py:198-204,223,323-328`.

**Paper:** T Eqs. (9)-(10), p.8/PDF 8, require one pre-observation predictor `(mu_tilde_(i-1),sigma_tilde_(i-1))` per observation, the cumulative predictive likelihood, raw `mean(X_i^2)`, and
`W_n=alpha^(-2/n)*exp(sum_terms/n)/e`.

**Code:** [ttest_universal.py:188-196](expectation/parametric/ttest_universal.py#L188-L196), excerpts:

```python
test._mu_estimators.append(mean)
test._sigma_estimators.append(std)
```

```python
s = np.sum(data_centered)
v = np.sum(data_centered**2)
n = len(data)
```

Yet [lines 213-214](expectation/parametric/ttest_universal.py#L213-L214) index `test._mu_estimators[-n - 1 + i]`. The [CS branch, lines 324-338](expectation/parametric/ttest_universal.py#L324-L338), uses:

```python
x_squared_bar = self.state.sum_squares / n
```

```python
W = (1 / self.config.alpha ** (2 / n)) * np.exp(log_sum_terms / n)
radius = np.sqrt(max(0, mu_bar**2 - x_squared_bar + W))
```

Its superclass stores centered, not raw, sum of squares at `confidencesequence.py:67-69`.

**Translation diff:**

```text
REFERENCE:   one predictor per observation; n and likelihood cover the full prefix.
IMPLEMENTED: one predictor per batch, indexed as per-observation; n is batch length.
REFERENCE W: alpha^(-2/n)*exp(sum_terms/n)/e.
IMPLEMENTED: the 1/e factor is absent.
REFERENCE second moment: sum(X_i^2)/n.
IMPLEMENTED: centered sum((X_i-Xbar)^2)/n.
```

**Consequence:** a first batch of length three leaves two predictor entries but requests index -4. Independently, W is multiplied by e and the second-moment substitution adds an extra squared mean to the radicand. These are static, inspectable contradictions in latent branches, not an asserted live public error rate.

The high-level CS is currently unconstructible: `super().__init__(config)` at line 302 passes a positional argument to Pydantic `BaseModel`. This and the missing test enum must not be repaired ahead of the formulas, clocks and filtration contract.

## Conformal evidence

### CONF-01 — default mixture scoring is not the cited conformal e-measure

**Type:** nonconformity-score assumption mismatch; **high**.

**Citation chain:** direct C/CP references at `conformal/conformal.py:19-24`; the class claims proper nonconformity e-measures.

**Paper:** C Eq. (1), p.2/PDF 4, and CP Eq. (1), p.4/PDF 4, require an equivariant nonnegative score vector with `sum(alpha_i)/m <= 1`. At m=1 its only score is at most one.

**Code:** [conformal.py:142-154](expectation/conformal/conformal.py#L142-L154), excerpts:

```python
if self._n_samples == 0:
    # First batch - compare to null
    s = np.sqrt(batch_size) * np.mean(data)
    v = self.v_opt
```

```python
log_e_score = self.mixture.log_superMG(s, v)
e_score = np.exp(log_e_score)
```

**Translation diff:** a conformal singleton score bounded by one becomes an unnormalized Gaussian-mixture score.

**Consequence:** `ConformalEValue().update([10.])` returns **4.1718589556781686e18**. The constant population X=10 is exchangeable and therefore in the cited conformal model. This is not a counterexample under Howard's different mean-zero null.

The full-source review also identified the chronological running-mean/variance standardization at lines 145-149 as order-dependent, conflicting with CP Eq. (1)'s equivariance requirement. A valid common raw score must be normalized under the conformal model; naming its kernel a mixture does not supply that property.

### CONF-02 — unequal batches are substituted for exchangeable scoring units

**Type:** exchangeability-contract mismatch; **high**.

**Citation chain:** direct C Eq. (12) citation at `conformal.py:173,193-194`.

**Paper:** C Eqs. (10),(12), p.10/PDF 12, define `L_n=f1(z_n)/f0(z_n)` and `E_n=L_n/mean(L_1,...,L_n)`. Proposition 1, p.3/PDF 5, requires exchangeability.

**Code:** [conformal.py:183,191-196](expectation/conformal/conformal.py#L183-L196), excerpts:

```python
likelihood_ratio = float(np.prod(alt_probs / null_probs))
```

```python
self._likelihood_ratios.append(likelihood_ratio)
mean_lr = np.mean(self._likelihood_ratios)
conformal_e_value = likelihood_ratio / max(mean_lr, 1e-10)
```

**Translation diff:**

```text
REFERENCE index n:   one exchangeable observation/common score.
IMPLEMENTED index n: a product over an arbitrarily sized update batch.
```

**Consequence:** enumerate eight iid fair binary triples, with null masses .5 and alternative masses .05/.95. Update the first two observations as one batch and the third separately. The second output's exact expectation is **1.1958685154241875**. The original observations are exchangeable but the two differently sized batch products are not. Fixed-size exchangeable blocks remain a valid restriction; the paper's theorem is not disproved.

### CONF-03 — an upward capital floor changes the fixed-horizon product

**Type:** product modification; **high**.

**Citation chain:** `TruncatedEPseudomartingale` inherits the implementation citing C Section 3.

**Paper:** C Section 3, p.4/PDF 6, defines `S_n=product(E_1,...,E_n)`, `S_0=1`; Proposition 3, p.5/PDF 7, gives fixed-horizon e-validity. CP Eq. (4), p.4/PDF 4, supplies the all-zero-score convention; its Example 3, p.8, supplies the binary normalization example below.

**Code:** parent [conformal.py:274](expectation/conformal/conformal.py#L274) executes `self._capital *= e_value`. The [subclass, lines 315-321](expectation/conformal/conformal.py#L315-L321), adds:

```python
capital, max_cap = super().update(e_value)

# Apply truncation
if capital < self.min_capital:
    self._capital = self.min_capital
    self._capital_history[-1] = self.min_capital
```

**Translation diff:** `S_n=S_(n-1)*E_n` becomes `S_n=max(epsilon,S_(n-1)*E_n)`.

**Consequence:** normalize raw score A(z)=z for iid fair binary data. Then E1=1 and E2 takes `(1,2,0,1)` equiprobably. With `min_capital=.1`, the terminal-capital mean is **1.025**, instead of one. This breaks fixed-horizon e-validity; no alpha=.05 rejection-rate inflation is claimed. C's truncated waiting times are not a capital-floor construction.

At the default floor `1e-10`, the mean excess in this example is only about `2.5e-11`. The durable defect is altered statistical capital and revival after genuine zero, not a claim of large default false-positive inflation. Existing floor/CUSUM test expectations must be corrected with the implementation rather than preserved as statistical evidence.

### DET-01 / DET-02 — the detector recurrences differ from both cited procedures

**Type:** algorithm mistranslations; **high**.

**Citation chain:** direct C citation at `conformal/cusum.py:13-14`.

**Paper:** C Eq. (6), p.8/PDF 10, uses `max_i product_(j=i..n) e_j`, equivalent between alarms to `C_n=e_n*max(1,C_(n-1))`. Eq. (9), p.9/PDF 11, defines reverse SR as `max_i sum_(j=i..n) product_(k=i..j) e_k`. It is not ordinary forward SR.

**Code:** [cusum.py:60-80](expectation/conformal/cusum.py#L60-L80), excerpts:

```python
self._cusum_stat = 0.0  # Current CUSUM statistic
```

```python
if self.use_sr:
    # Shiryaev-Roberts modification
    sum_stat = sum(e_value * stat for stat in self._stats_history[self._last_alarm :])
    self._cusum_stat = max(sum_stat, self.min_value if self.truncate else 0)
else:
    # Standard CUSUM
    self._cusum_stat = max(
        self._cusum_stat * e_value, self.min_value if self.truncate else 0
    )
```

**Translation diff:**

```text
REFERENCE CUSUM: e_n*max(1,previous CUSUM).
IMPLEMENTED:    max(previous CUSUM*e_n, numerical floor).
REFERENCE reverse SR: maximum of sums of prefix products indexed by start time.
IMPLEMENTED:         current e_n times a sum of earlier aggregate statistics.
```

**Consequence:** with truncation disabled, zero is absorbing. Proper normalized scores `(1,1.9,19/13)` cross the paper's CUSUM threshold 2 at step three, while the code stays zero. For reverse SR with threshold 3, the paper's own all-one predictor example alarms at step three; the code does not. These are missed-alarm/algorithm failures, not claims of excessive false-alarm probability.

### DET-03 — one alarm is counted on multiple later updates

**Type:** paper-defined event-count mismatch; **medium**.

**Citation chain:** C is directly cited by `cusum.py`; `EfficiencyAnalyzer` cites its Section 6.

**Paper:** C Eq. (7), p.8/PDF 10, defines `A_n=max{k:tau_k<=n}`, the number of distinct alarm events. Eq. (8) concerns asymptotic `A_n/n`, not a finite-horizon probability bound.

**Code:** [cusum.py:165-171](expectation/conformal/cusum.py#L165-L171):

```python
if not detected and result.n_alarms > 0:
    last_alarm = result.alarms[-1]
    if last_alarm > n_pre:
        detection_delays.append(last_alarm - n_pre)
        detected = True
    else:
        false_alarms.append(last_alarm)
```

The reported metric is `len(false_alarms) / n_trials`.

**Translation diff:** count each new `tau_k` once versus append the same historical `tau_k` on every subsequent update before postchange detection.

**Consequence:** use `pre_dist` and `post_dist` returning constant .25, producing e-values identically one; set `ConformalCUSUM(threshold=3,use_sr=True,truncate=True,min_value=1)`, `n_pre=4,n_post=1,n_trials=1`. Actual alarm list is **[4]**, but `false_alarm_rate` is **2.0**, not the one event per trial. This valid constant-score example supersedes the earlier insufficiently specified 4.0 example. No violation of the asymptotic theorem is inferred from this metric alone.

## Shared betting and RIPr evidence

### RIPr-00 — reference check: the regular unrestricted calculation matches

**Status:** supported translation, not a defect.

**Citation chain:** R is directly cited at `ksample/bernoulli.py:12-20`.

**Paper:** R Proposition 2 and Eqs. (3.2)-(3.4), pp.15-16/PDF 15-16, give posterior means `(U_g+gamma)/(N_g+2*gamma)` and the block-size-weighted null projection.

**Code:** [bernoulli.py:188-200](expectation/ksample/bernoulli.py#L188-L200), excerpts:

```python
theta_estimates[g] = (U_g + self.gamma) / (N_g + 2.0 * self.gamma)
```

```python
theta_null = sum(
    (block_sizes[g] / total_block_size) * theta_estimates[g] for g in range(self.k)
)
```

**Comparison:** these are the paper's symmetric-Beta specialization and weighted projection. The likelihood terms at lines 331-337 are their Bernoulli log likelihood ratio. This establishes regular-domain algebraic fidelity, not safety of every surrounding state/error path.

### RIPr-01 — the public sampling contract omits the paper's restrictions

**Type:** assumption-exposure gap, not a theorem counterexample.

**Citation chain:** direct R citation in `ksample/ksample_test.py`.

**Paper:** R Eq. (2.1), p.11/PDF 11, factorizes sampling across observations and groups. The variable-block rule on p.13/PDF 13 prohibits outcome-dependent completion within the current block, while permitting the stated exogenous group-arrival framework.

**Code:** [ksample_test.py:45-48](expectation/ksample/ksample_test.py#L45-L48):

```python
"""Sequential k-sample homogeneity test for Bernoulli data.

Tests H0: theta_1 = theta_2 = ... = theta_k (all groups have the same
success probability) against alternatives specified by ``config``.
```

[Lines 224-226](expectation/ksample/ksample_test.py#L224-L226) flush all buffers when every group has data.

**Translation diff:**

```text
REFERENCE:   product sampling plus the specified block-design restrictions.
PUBLIC TEXT: marginal homogeneity, without those qualifications.
```

**Consequence/limit:** pairs `(0,1)` and `(1,0)` equiprobably have equal Bernoulli(.5) marginals. With simple alternative `{0:.2,1:.8}`, evidence is `2.56` or `.16`, mean **1.36**; four blocks reject on 1/16 paths. These pairs violate the paper's independence assumption. The example establishes why the omitted qualification matters, not incorrect RIPr algebra under the paper's model. Runtime code cannot determine independence from an array; exogenous buffered arrivals are not inherently invalid.

### RIPr-02 — failure/retry scores a block using a posterior trained on that block

**Type:** statistical state-transition mismatch; **high**.

**Citation chain:** direct R citations in `bernoulli.py` and `ksample_test.py`; the updater directly cites B.

**Paper:** R Eq. (2.6), p.14/PDF 14, and Eq. (3.4), p.16/PDF 16, score block j with `W_1 | Y^(j-1)`. B Definition 7.19, p.106/PDF 119, requires conditional expected increment at most one.

**Code:** [bernoulli.py:259-264](expectation/ksample/bernoulli.py#L259-L264) changes the posterior inside the calculation:

```python
self.log_weights = self.log_weights + log_lik
```

```python
self.log_weights -= logsumexp(self.log_weights)

return log_e, theta_estimates, theta_null
```

The [orchestrator, lines 136-145](expectation/ksample/ksample_test.py#L136-L145), calls that calculation before the updater:

```python
log_e_value, theta_estimates, theta_null = self._calculator.compute_log_e_value(
    block_successes, block_sizes
)
```

```python
self.e_process_updater.update(self.e_process, e_value)

self._calculator.update_state(block_successes, block_sizes)
self.block_count += 1
```

The [updater, lines 88-94](expectation/modules/eprocessupdater.py#L88-L94), can then fail:

```python
process.values.append(e_value)
process.total_samples += 1

if self.combiner is None:
    raise ValueError(
        "Combiner not initialized. Call set_log_optimal_expectation if using LOG_OPTIMAL strategy."
    )
```

**Translation diff:**

```text
REFERENCE scoring state:        S_j = s(Y_j; W_1 | Y^(j-1)).
AFTER FAILED CALL AND RETRY:    S_j = s(Y_j; W_1 | Y^j).
```

**Consequence:** configure k=2, additive restricted delta=.2, `betting_strategy="log_optimal"`. For each of the four independent Bernoulli(.5) group outcomes, call `update()` before initializing the combiner; after the exception set `lambda fraction,past,step: np.log1p(.44*fraction)` and retry the same physical block. The callback chooses the same predictable fraction across all cases.

Clean factors are `.96,1.44,.64,.96`; retry factors are `.9829207398160559,1.44,.64,.982920739816056`. Exact expected capital changes from **1** to **1.0114603015943562**. The clocks become `block_count=1`, `total_samples=2`, one capital update. This is a proved expectation failure for that recovery protocol, not a quantified alpha=.05 rejection-rate claim. The required posterior-at-scoring invariant is mathematical; preserving it across failure is the engineering responsibility.

### RIPr-03 — prior gamma is also used as a betting cap

**Type:** parameter-meaning mismatch across configuration composition.

**Citation chain:** R Section 5 and B Chapter 7 are directly referenced by the k-sample configuration.

**Paper:** R p.19/PDF 19 permits any Beta prior `gamma>0`. B Definition 7.21(iii), p.107/PDF 120, separately uses `gamma in (0,1]` to constrain betting fractions.

**Code:** [ksample_test.py:88-93](expectation/ksample/ksample_test.py#L88-L93):

```python
e_process_config = EProcessConfig(
    significance_level=config.significance_level,
    betting_strategy=config.betting_strategy,
    gamma=config.gamma,
    conservative_lambda=config.conservative_lambda,
)
```

`EProcessConfig.gamma` is constrained with `gt=0, le=1` at `hypothesistesting.py:87-89`.

**Translation diff:** two parameters governing different mathematical objects become one shared configuration value.

**Consequence:** `KSampleConfig(k=2,gamma=2)` accepts a paper-permitted prior, but constructing the test fails the unrelated betting-cap validation, even for all-in betting. Changing prior strength also changes the adaptive investment cap. R's p.19 additionally describes .18 as experimentally selected in a restricted prior family for large block count and `n_a=n_b=1`, not a universal optimality theorem.

### RIPr-04 — an empty grid is accepted as a restricted posterior

**Type:** construction-domain mismatch.

**Citation chain:** direct R Appendix S1 references in `bernoulli.py:78-79` and `config.py`.

**Paper:** R Appendix S1, p.35/PDF 35, requires `K in (0,1-zeta)` and normalizes a nonempty grid distribution; for additive effects `zeta=delta`.

**Code:** [config.py:111-116](expectation/ksample/config.py#L111-L116) checks only `0<K<1`. The [calculator, lines 88-92](expectation/ksample/bernoulli.py#L88-L92), then uses:

```python
zeta = delta
self.grid_theta_a = np.arange(K, 1.0 - zeta, K)
self.grid_theta_b = self.grid_theta_a + delta
```

[Lines 228-229](expectation/ksample/bernoulli.py#L228-L229):

```python
theta_a_hat = float(np.dot(posterior, self.grid_theta_a))
theta_b_hat = float(np.dot(posterior, self.grid_theta_b))
```

**Translation diff:** `K<1-delta` and a normalized probability distribution become `K<1` and potentially empty dot products.

**Consequence:** additive delta=.9, K=.5 yields an empty grid, both estimates zero, and neutral evidence one while counts advance. This is not demonstrated type-I inflation. It silently replaces the configured restricted-posterior procedure with an uninformative computation outside S1's grid domain.

### RIPr-05 / P04 — intermediate exponentiation changes the capital recurrence

**Type:** numerical representation/composition mismatch; **high**.

**Citation chain:** R Eq. (3.4) is cited by the k-sample orchestrator; B Definition 7.21 is cited by both Python and Rust updaters.

**Paper:** R p.16/PDF 16 specifies the product of block factors. B Eq. (7.10), p.107/PDF 120, specifies `M_t=product(1-lambda_s+lambda_s*E_s)` for predictable fractions. For finite E, lambda=0 gives factor exactly one.

**Code:** [ksample_test.py:140-142](expectation/ksample/ksample_test.py#L140-L142):

```python
e_value = np.exp(log_e_value)
self.e_process_updater.update(self.e_process, e_value)
```

[eprocessupdater.py:107-111](expectation/modules/eprocessupdater.py#L107-L111), excerpts:

```python
new_value = current_value * increment
```

```python
if np.isinf(new_value) or new_value > 1e308:
    new_value = 1e308  # Near float64 max but safe
```

[Rust update.rs:245-246](rust/par_seqtest/update.rs#L245-L246):

```rust
let e_t = log_e_t.exp();
*lep += ((1.0 - lambda_t) + lambda_t * e_t).ln();
```

**Translation diff:**

```text
REFERENCE:   positive finite factors and their exact product/log product.
PYTHON:      exp(log factor) can become zero; capped linear capital differs from logs.
REFERENCE at lambda=0: log(1)=0 for finite mathematical E.
RUST:        0*exp(large finite logE) becomes 0*infinity=NaN in binary64.
```

**Consequences:** for simple RIPr probabilities `{0:.1,1:.9}`, use blocks `(size,group0 value,group1 value)` of `(400,1,0)`, `(600,0,1)`, `(600,0,1)`. The reference log product is `800*log(.2)+2400*log(1.8)=123.13766581780555`; the implementation returns zero capital and no rejection.

For one native known-variance-one test, observation 50 gives finite all-in log capital **1109.0237505725038**, conservative log capital infinity, and adaptive first-step log capital NaN with lambda=0. A one-element arithmetic merge also loses the identity with its base process on observations `50,-50`. These are numerical/decision departures, not measured type-I inflation.

### EP-01 — negative and undefined values are admitted as sequential e-values

**Type:** unenforced mathematical input contract.

**Citation chain:** direct B citation in `eprocessupdater.py:9-11,79-83`.

**Paper:** B Definition 7.3, p.97/PDF 110, and Definition 7.19/Proposition 7.20, p.106/PDF 119, concern nonnegative e-variables with the conditional expectation bound.

**Code:** [eprocessupdater.py:85-89](expectation/modules/eprocessupdater.py#L85-L89):

```python
if not self.config.allow_infinite and np.isinf(e_value):
    raise ValueError(f"Infinite e-value {e_value} not allowed by config")

process.values.append(e_value)
process.total_samples += 1
```

**Translation diff:** the mathematical nonnegative, defined domain is not enforced; only the configured infinity policy is checked before mutation.

**Consequence:** two updates of -10 produce capital history `[1,-10,100]`, log history `[0,-inf,-inf]`, and significance at alpha=.05. NaN is also committed. These inputs are outside the reference domain: this is an admission/state-contract failure, not an incorrect multiplication formula or a theorem counterexample on valid inputs. Conditional validity itself cannot be established by a numerical validator.

### EP-02 — empirical e-power discards zero evidence

**Type:** quantity mistranslation.

**Citation chain:** direct B Definition 3.11 reference at `eprocessupdater.py:164-166`.

**Paper:** B Definition 3.11, p.40/PDF 53, defines `E_Q[log E]`, explicitly allowing negative infinity. The empirical distribution includes all observations.

**Code:** [eprocessupdater.py:171-172](expectation/modules/eprocessupdater.py#L171-L172):

```python
log_terms = [np.log(e) for e in process.values if e > 0]
return np.mean(log_terms) if log_terms else -np.inf
```

**Translation diff:** average logarithms under the empirical distribution versus condition on positive evidence and renormalize the surviving sample.

**Consequence:** values `[0,2]` give **0.6931471805599453**, while the empirical expected logarithm is `.5*log(0)+.5*log(2)=-infinity`. This changes the explicitly cited quantity, not merely its display format.

### RIPr-06 — scalar ingestion validates the recoded value instead of the observation

**Type:** mathematical-input translation mismatch.

**Citation chain:** direct R citation in `ksample_test.py`.

**Paper:** R Eq. (3.1), p.15/PDF 15, and Proposition 2, p.16/PDF 16, define Bernoulli observations and their success counts.

**Code:** [ksample_test.py:218-222](expectation/ksample/ksample_test.py#L218-L222):

```python
obs = int(observation)
if obs not in (0, 1):
    raise ValueError(f"Observation must be 0 or 1, got {observation}")

self._buffer[group_id].append(obs)
```

**Translation diff:** supplied binary data or domain rejection versus truncation toward zero followed by validation of the transformed value.

**Consequence:** `update_single(0,1.9)` followed by `update_single(1,0)` records group 0's mean as one; `.9` and `-.9` become zero. Direct block input rejects those fractional values. This changes the sufficient statistics across ingestion interfaces; it is not a false-positive claim for correctly supplied binary observations.

### SYM-01 — the stated symmetry null omits the paper's sign assumptions

**Type:** assumption-exposure gap.

**Citation chain:** direct S DOI citation at `hypothesistesting.py:213-219`, with Eqs. (7.3),(8.3) cited by the methods.

**Paper:** S starts with continuous iid observations at p.263/PDF 3. Section 7, p.267/PDF 7, uses independent fair signs; Lemma 3, p.268/PDF 8, uses uniform positive-rank subsets.

**Code:** [hypothesistesting.py:229-233](expectation/modules/hypothesistesting.py#L229-L233), excerpt:

```python
description="Distribution is symmetric around 0",
```

[Lines 273-276](expectation/modules/hypothesistesting.py#L273-L276):

```python
k = np.sum(data > 0)
n = len(data)

e_value = np.exp(lambda_val * k) * (2 / (1 + np.exp(lambda_val))) ** n
```

**Translation diff:** fair nonzero signs under the reference conditions versus zero observations counted in n but never as positive signs, under an unqualified symmetry description.

**Consequence:** for 20 zeros and lambda=-.5, sign evidence is **79.9501989411765** and Wilcoxon evidence **280357.20028473384**. The point mass at zero is symmetric but outside the paper's continuous model. This exposes the omitted input contract, not an error in the published formulas. Negative lambda alone and nonzero ties ranked without looking at signs are not retained as defects.

### SYM-02 — uncancelled exponentials lose finite symmetry evidence

**Type:** numerical evaluation mismatch.

**Citation chain:** direct S Eqs. (4.4),(7.3),(8.3) citations in the corresponding methods.

**Paper:** S p.264/PDF 4, p.267/PDF 7, and p.268/PDF 8:
`E_F=product(exp(lambda*z_i)/cosh(lambda*z_i))`,
`E_S=exp(lambda*k)*(2/(1+exp(lambda)))^n`,
and Eqs. (8.1)-(8.2) give `log E_W=lambda*V_n-sum_i log((1+exp(lambda*i))/2)`.

**Code:** [hypothesistesting.py:258-261](expectation/modules/hypothesistesting.py#L258-L261):

```python
numerator = np.exp(lambda_val * data)
denominator = 0.5 * (np.exp(lambda_val * data) + np.exp(-lambda_val * data))

e_value = np.prod(numerator / denominator)
```

[Lines 294-298](expectation/modules/hypothesistesting.py#L294-L298):

```python
numerator = np.exp(lambda_val * V_n)
denominator_factors = np.array([1 + np.exp(lambda_val * i) for i in range(1, n + 1)])
denominator = np.prod(2 / denominator_factors)

e_value = numerator * denominator
```

**Translation diff:** the symbolic formulas agree, but separately exponentiating uncancelled terms creates `inf/inf`, `inf*0`, or infinity before their finite combination is evaluated.

**Consequence at lambda=.5:** Fisher `[2000.]` gives NaN instead of approximately 2; sign observations `1..1500` give infinity instead of **5.1459570349e142**; Wilcoxon `1..60` gives NaN instead of **3.0823442968e17**. Reference values follow the same equations in logarithmic form. No type-I inflation is inferred from these evaluation failures.

The public `test()` path raises on these invalid projections. A complete numerical repair must carry valid log evidence through results and decisions, not only change the internal formula while leaving public calls failing.

## Parallel merging, adjustment, and algorithm identity

### P03 — segment validation and ownership do not preserve the reference partition

**Type:** mathematical parameter-contract mismatch; **high**.

**Citation chain:** direct M citation at `rust/merge/mod.rs:175` and `modules/merging.py:403-405`.

**Paper:** M p.10/PDF 10 requires `1 <= K1 < ... < Km < K` and multiplies the means of the resulting disjoint segments.

**Code:** [par_seqtest.py:204-206](expectation/par_seqtest.py#L204-L206):

```python
if self.global_merge == MergingMethod.SEGMENT_PRODUCT:
    if self.merge_segments is None:
        raise ValueError("merge_segments is required when global_merge is SEGMENT_PRODUCT")
```

[rust/merge/mod.rs:190-198](rust/merge/mod.rs#L190-L198), excerpt:

```rust
let start = window[0];
let end = window[1];
if end <= start {
    continue;
}
let seg_len = (end - start) as f64;
let seg_sum: f64 = e_values[start..end].iter().sum();
```

The standalone Python constructor validates ordering but retains `self.segments = segments` at line 429.

**Translation diff:** a fixed ordered partition versus an existence-only check and skipped reversed windows; externally aliased lists can also invalidate an initially validated partition.

**Consequence:** K=3 with boundaries `[2,1]` computes `(E1+E2)*(E2+E3)/4`. For independent precise e-values taking 0 or 2 equiprobably, its expectation is `(1+1+2+1)/4 = **1.25**`. Unlike a correlated-product counterexample, independence is satisfied here. Boundaries `[3]` for K=2 also cause a native slice panic after individual updates. Mutating a caller-owned Python boundary list can similarly destroy the constructor's partition prerequisite.

### P05 — log adjusters evaluate a different numerical function at large inputs

**Type:** numerical function/decision mismatch; **high**.

**Citation chain:** direct A Eq. (5)/Theorem 1 references in Python and Rust adjuster modules.

**Paper:** A Section 4, p.6/PDF 6:
`A1(E)=(E-1-log E)/(log E)^2`, `A2(E)=sqrt(E)-1`.
The admissible adjuster is increasing, right-continuous, and tends to infinity. Its application assumes valid underlying processes.

**Code:** [rust/adjusters/mod.rs:108-131](rust/adjusters/mod.rs#L108-L131), excerpts:

```rust
let val = lookback_from_log(log_e);
```

```rust
(x.exp_m1() - x) / x_sq
```

The result is subsequently logged. The sqrt log-adjuster similarly computes `(log_e / 2.0).exp_m1()` at line 160 before taking its logarithm.

**Translation diff:**

```text
REFERENCE log A1(exp(x)) = x - 2*log(x) + log(1-(1+x)*exp(-x)).
REFERENCE log A2(exp(x)) = x/2 + log(1-exp(-x/2)).
IMPLEMENTED: materialize exp(x) or exp(x/2) before cancellation/logarithm.
```

**Consequence:** x=710 returns infinity for lookback instead of **696.8694700599293**; x=1500 returns infinity for sqrt instead of approximately **750**. The former causes an incorrect adjusted-BH decision at alpha=`1e-305`, whose reference log threshold is **702.288453363184**. At ordinary alpha=.05, conservative observations `5,100` make adjusted lookback rejections change **1 to 0** when infinite stored evidence becomes NaN adjustment. This is a decision/monotonicity failure, not a measured global FDR inflation rate. The separate zero-at-E=1 choice is conservative.

### P06 — U-statistic normalization destroys exactly neutral evidence

**Type:** numerical identity mismatch.

**Citation chain:** direct M Eq. (13) references in Python and Rust merging implementations.

**Paper:** M Eq. (13), p.10/PDF 10:
`U_n = sum_(|A|=n) product_(k in A) e_k / binomial(K,n)`.
Consequently `U_n(1,...,1)=1`; no dependence assumption is needed for this deterministic identity.

**Code:** [rust/merge/mod.rs:146-148](rust/merge/mod.rs#L146-L148):

```rust
let binom = binomial_coeff(k, n);
p[n] / binom
```

[Lines 234-238](rust/merge/mod.rs#L234-L238):

```rust
let mut result = 1.0_f64;
for i in 0..k {
    result *= (n - i) as f64;
    result /= (i + 1) as f64;
}
```

**Translation diff:** division by the finite coefficient `binomial(1024,512)` becomes division by infinity because multiplication overflows before the compensating division.

**Consequence:** for K=1024, order=512 and exactly unit per-test multipliers, native merged evidence is **0**, versus mathematical **1** and Python **1.0000000000000002**. The all-in merged process loses all evidence. This is not a claim that downward rounding inflates type-I error.

### P07 — the native adaptive rule is not either named reference algorithm

**Type:** algorithm-identity/optimization mismatch.

**Citation chain:** `rust/par_seqtest/update.rs:15-18,65-71` cites W and B and labels the strategy ONS-based.

**Paper:** W's published supplement distinguishes GRAPA Eq. (43), p.45/PDF 11; aGRAPA, p.46/PDF 12; and ONS-m Algorithm 1, p.48/PDF 14. B Definition 7.21(iii), p.107/PDF 120, defines the exact argmax of `sum_(s<t) log(1+lambda*(E_s-1))` on `[0,gamma]`. ONS-m instead updates a previous bet using normalized gradients and their squared accumulator.

**Code:** [update.rs:241,249-251](rust/par_seqtest/update.rs#L241-L251), excerpts:

```rust
let lambda_t = (*s1 / (*s2 + epsilon)).clamp(0.0, gamma);
```

```rust
let e_minus_1 = e_t - 1.0;
*s1 += e_minus_1;
*s2 += e_minus_1 * e_minus_1;
```

The [fixture generator, lines 228-230](tests/rust/fixtures/generate_golden_fixtures.py#L228-L230), duplicates that recurrence rather than invoking Python's optimizer.

**Translation diff:**

```text
REFERENCE ONS-m: previous bet + normalized-gradient/accumulator update.
REFERENCE B:    exact empirical log-growth argmax.
IMPLEMENTED:    clipped, regularized sum(E-1)/sum((E-1)^2), an aGRAPA-like rule.
```

**Consequence:** observations `3,0,0` give second-step lambda **.05814240057744242** natively versus **.4999999865167874** from Python's optimizer. For constant alternative multipliers four and gamma=1, the ratio tends to 1/3 with growth log(2); the empirical optimum is one with growth log(4). Predictable bounded betting remains valid on valid inputs; the exact optimizer's guarantee and ONS identity do not transfer.

B Theorem 7.22's exact-optimizer guarantee assumes iid evidence under the alternative, finite expected log evidence and gamma=1. W's authors' reply also distinguishes capital growth from confidence-set width. The proposed cancellation-aware expansion/enclosure mechanism remains unverified; reproducing its tiny-log acceptance example does not establish the mechanism or its performance.

### P08 — the function named e-Holm is reciprocal p-Holm

**Type:** method-identity difference; not a FWER-invalidity claim.

**Citation chain:** `rust/multiple_testing/holm.rs:14` cites B Chapter 4, which does not contain this algorithm. The book's relevant mean-based closure is Algorithm 8.18, p.123/PDF 136; printed p.148/PDF 161 explicitly calls mean-e-collection closure "the e-Holm procedure." Its bibliography leads to L. L is bibliography-traced, not falsely labelled a header citation.

**Paper:** L Theorem 2.1, Eq. (2.2), p.5/PDF 5, rejects i when
`e_i >= 1/alpha + sum_j max(1/alpha-e_j,0)`.
Section 2.3, p.6/PDF 6, explicitly distinguishes p-Holm on reciprocal e-values.

**Code:** [holm.rs:57-64](rust/multiple_testing/holm.rs#L57-L64), excerpt:

```rust
let k = rank_0 + 1;
let remaining = (m - k + 1) as f64;
let log_threshold_k = remaining.ln() + log_inv_alpha;
if log_e >= log_threshold_k {
    k_star = k;
} else {
    break; // Step-down: stop at first non-rejection
```

**Translation diff:** reference e-Holm mean-based closure versus successive rank thresholds `(K-k+1)/alpha`, explicitly called reciprocal p-Holm by L.

**Consequence:** for `(30,30)`, alpha=.05, reference e-Holm rejects both because its threshold is 20; the implementation rejects neither because its first threshold is 40. The naming/source identity and power differ. Reciprocal p-Holm is still FWER-valid on valid inputs; changing the method rather than correcting its name/reference requires an explicit implementation decision.

### P09 — the purported optimal rho is an unlabelled approximation

**Type:** optimization-fidelity difference; not invalid evidence.

**Citation chain:** H is directly cited by `rust/martingale/two_sided_normal.rs`, which calls `best_rho` optimal.

**Paper:** H Proposition 3, Eq. (21), p.12/PDF 12, for scalar `l0=1`:
`rho_* = v_opt / (-W_-1(-alpha_opt^2/e)-1)`.

Q Eq. (21), p.10/PDF 10, explicitly gives the logarithmic approximation used by the code. It is a source-supported approximation, not invented tuning.

**Code:** [two_sided_normal.rs:72-73](rust/martingale/two_sided_normal.rs#L72-L73):

```rust
let log_inv_alpha = (1.0 / alpha).ln();
Ok(v / (2.0 * log_inv_alpha + (1.0 + 2.0 * log_inv_alpha).ln()))
```

**Translation diff:** the Lambert-W optimizer is replaced by a logarithmic approximation while still labelled optimal.

**Consequence:** at `v_opt=1, alpha_opt=.9`, exact rho is **1.2552746833068595**, implemented rho **2.4879514290590095**; the target-time boundary is approximately 3.20% wider. At alpha=.05, values are .12177348869865692 and .1260056097925634. Any fixed positive rho remains permitted. H expressly supports doubled-alpha one-sided tuning as an approximation; that technique is not a defect.

### P10 — reciprocal overflow changes a finite log rejection threshold

**Type:** numerical threshold mismatch.

**Citation chain:** the parallel engine cites B for Ville's inequality; the correct location is Fact 7.6, Eq. (7.4a), p.98/PDF 111.

**Paper:** `M_t >= 1/alpha` is equivalent to `log M_t >= -log(alpha)` for positive alpha. For one hypothesis the Bonferroni threshold is identical.

**Code:** [rust/par_seqtest/mod.rs:134](rust/par_seqtest/mod.rs#L134):

```rust
log_threshold: (1.0 / alpha).ln(),
```

[bonferroni.rs:39-40](rust/multiple_testing/bonferroni.rs#L39-L40):

```rust
let m = log_e_values.len() as f64;
let log_threshold = (m / alpha).ln();
```

**Translation diff:** a finite logarithmic threshold for accepted positive alpha becomes infinity when the reciprocal is materialized first.

**Consequence:** at alpha=`1e-310`, the correct one-test log threshold is **713.8013788281542**. A current-source engine with log evidence 720 does not reject via Ville/Bonferroni/reciprocal-Holm, while e-BH does. The latter uses stable log subtraction. This is a false-negative/algorithm difference, not type-I inflation.

## Withdrawn, unresolved, and excluded claims

These are not accepted paper-translation defects:

- **Untruncated conformal pseudomartingale optional stopping:** C Proposition 2, pp.4-5/PDF 6-7, explicitly establishes possible severe failure of Ville's bound; Proposition 3 establishes fixed-horizon validity. The code's ordinary product matches that construction. The earlier stopped-capital expectation 1.093956044 is not a mistranslation.
- **Product merging on correlated inputs:** M p.3/PDF 3 requires sequential or independent inputs for product merging, and the parallel class explicitly describes independent streams. The product formula matches the source. Identical-stream counterexamples violate its premise and do not establish an in-contract algorithm defect. Numerical nonnegativity is not proof of independence.
- **RIPr Appendix S1:** R's main text Eqs. (3.3)-(3.4), p.16/PDF 16, uses both marginal posterior means; S1, p.35/PDF 35, instead transforms one mean to obtain the second. The code computes both marginals. For nonlinear log-odds mapping, `E[f(theta)] != f(E[theta])` in general; observed second means .5814498600 and .7010422900 differ. The main-text formula matches the code, both predictable fitted pairs can be valid under the stated null, and authorial intent is unresolved. This is not a proven statistical bug.
- **Beta-boundary physical cap:** H Eqs. (56),(58), p.29/PDF 29, search up to `(r+v)/g`; the code caps at `v/g`. H Definition 2 uses non-strict crossing, while a physical cap can be conservative for strict crossing/closed intervals. Existing rejection paths use `log_superMG`, not this cap. The earlier 50% public-error interpretation is withdrawn; the API inequality contract remains unresolved.
- **Physical-cap context from the full paper:** H Appendix D, p.40, explicitly recommends the physical cap with a closed confidence set when the threshold root is outside the attainable range. Do not reintroduce a blanket cap-as-type-I-failure claim.
- **Lookback at E=1:** returning zero instead of the right limit 1/2 differs from the stated right-continuous admissible version, but is downward-conservative. It is separate from P05's overflow failure.
- **Adapted variance:** H Definition 1 allows it when the domination condition holds. Lack of predictability of V alone is not a defect. Theorem 4 separately requires predictable predictions.
- **FDLIBM coefficients:** the original Netlib software and current Rust were compared, but that is not reading Cody's original paper. Its coefficient/accuracy observations are excluded from the paper-validated list under the requested standard.
- **Bibliography:** the literal adjuster citation "Dawid, Ryter, Vovk, de Heide (2011a). Prequential probability" was not identifiable as written. A's bibliography points to a different lookback work. Bibliographic errors alone do not prove an inferential defect; P05 rests on the directly cited A Eq. (5).
- **Engineering-only observations:** broken imports/factories, CI artifact selection, MSRV/license metadata, callback preservation, serialization, mutable result records, ordinary rejected-call clock ticks, and dashboard wording are not promoted to paper-derived bugs here. The papers do not prescribe Python rollback, object immutability, import layout, or serialization. The retained lifecycle/ownership findings instead show a specific source normalization, posterior, partition, or event-count invariant being changed.
- **Uncited constructions:** `modules/epower.py`'s all-or-nothing transform, adaptive-threshold control, and the decay heuristic lack a sufficient direct citation chain for the previously claimed paper mistranslations. They remain unverified under this audit standard.
- **Intended t-process behavior:** the flat extended martingale's infinite initial value and reduced-filtration qualifications are deliberate features of T, not defects merely because they differ from ordinary unit-capital martingales.
- **Genuine zero/infinity products:** B Convention 12, printed p.xi/PDF 12, generally uses `0*infinity=0`; Chapter 8, p.113/PDF 126, permits an infinity extension on a null-impossible event; cross-merging on p.176/PDF 189 chooses `0*infinity=infinity`. Rejection is an API-domain choice, not a universal source requirement. This does not excuse underflow/overflow of finite-log evidence.

## Scope of the conclusion

The retained entries establish the specific differences and consequences shown, not that every exposed method is invalid or that every numerical discrepancy inflates type-I error. Source assumptions and OOP/engineering invariants are evaluated together. Correct regular-domain RIPr algebra, fixed normal-mixture kernels under their actual assumptions, and declared independent/sequential merging must not be replaced merely because surrounding implementation paths fail.

This evidence catalog retains the corrected source/code comparisons. [AUDIT_REPAIR_PRIORITIES.md](AUDIT_REPAIR_PRIORITIES.md) governs current adjudication and execution order; [PROPOSED_SOLUTIONS.md](PROPOSED_SOLUTIONS.md) records candidate repairs and unresolved choices. The full audit does not verify P07's proposed enclosure implementation, resolve R Appendix S1's authorial intent, or require a particular transaction/history architecture. No implementation repairs are claimed complete.
