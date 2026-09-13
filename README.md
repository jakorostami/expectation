<p>
    <a target="_blank">
      <img width="100%" src="https://github.com/jakorostami/expectation/blob/develop/assets/images/expectation.png" alt="expectation banner"></a>
  </p>

# expectation

Sequential hypothesis testing with e-values and e-processes, with a bundled Rust engine for massively parallel testing.

> v0.6.1 — the public API may still evolve before 1.0.

> **Statistical stabilization in progress (2026-09-07).** A completed full-source audit found unresolved construction, state and numerical defects. Implementation repairs have not started; do not rely on the current defaults or every exposed method as production-valid inference. See [the findings and ranked repair order](AUDIT_REPAIR_PRIORITIES.md), starting with owned segment partitions, complete reset and signed/reflected Bernoulli evidence.

## What this does

E-values and e-processes can support continuous monitoring and optional stopping under their stated mathematical assumptions. Those guarantees depend on the construction, filtration, sampling design and numerical implementation; an API or method name alone does not establish them.

The implemented research surface includes:

- **Sequential testing** — mean, proportion, quantile and variance workflows; their current validity limitations are documented in the audit
- **Parallel engine** — Rust + rayon backend for 300K+ simultaneous tests (brain voxels, genomics, A/B tests at scale)
- **Multiple testing** — e-Bonferroni, e-BH and the API currently named `e_holm` (reciprocal p-Holm), plus adjusted running-maximum procedures; their guarantees require valid input evidence and the appropriate joint filtration
- **Confidence sequences** — implementations under repair; current boundary labels and fixed-time coverage checks do not establish simultaneous coverage
- **Calibration** — convert between p-values and e-values
- **Testing protocol** (`modules/protocol.py`) — Forecaster–Skeptic–Reality layer: any `SkepticStrategy` emitting sequential e-values plugs into the existing temporal betting (`EProcessUpdater` combiners), Ville's inequality and the anytime p-value
- **Upper-expectation nulls** (`upperexp/`) — certified e-variables for convex-set nulls on finite outcome spaces: Huber contamination (least-favourable clipped likelihood ratio), Wasserstein-1 balls (Kantorovich-potential witness with a robustness tax, plus an anytime-valid lower bound on the distance) and finite-vertex polytopes (reverse information projection by EM), with CHSH/Bell local realism and bid–ask spreads as instances. Validity is enforced by an exactly computed support function
- **Sparse-mixture merging** (`modules/merging.py`, `SparseMixtureMerger`) — the anytime-valid sparse-anomaly test of Pérez-Ortiz, Castro & Stoepker (2025) as a martingale merging function over per-stream e-processes; evidence compounds multiplicatively across several individually strong streams, where the arithmetic-mean merge only grows logarithmically. In replicated simulations the two reach the rejection threshold at comparable times (the mean is often marginally earlier); the sparse mixture's gain is in evidence strength and localisation, not alarm latency. Requires independent streams
- **Stein Skeptic** (`stein/`) — the sequential kernelized Stein discrepancy test of Martinez-Taboada & Ramdas (2025) for score-only targets (Boltzmann distributions, energy-based models), with closed-form certified payoff bounds from a Lipschitz constant of the score for the IMQ kernel and for a compactly supported Wendland kernel needing only a local constant

Worked examples: `examples/commercial-playbook.ipynb` (a 22-section handbook: foundations, defensive forecasting, ten commercial settings each with the incumbent method run and shown failing, sensitivity tables and runbooks), `examples/toolkit-tour.ipynb` (every tool, what it does and what is not finished), `examples/commercial-case-studies.ipynb` (three companies, three decisions, with the traps measured), `examples/imprecise-forecasters-upper-expectations.ipynb`, `examples/sparse-anomaly-global-null.ipynb`, `examples/stein-score-only-goodness-of-fit.ipynb`.

![](https://github.com/jakorostami/expectation/blob/main/assets/images/seqplot.png)

## Install

```bash
pip install expectation
```

Prebuilt wheels bundle the Rust parallel engine, so no Rust toolchain is required.

### From source (development)

```bash
git clone https://github.com/jakorostami/expectation.git
cd expectation
pip install -e ".[dev]"
maturin develop --release
```

## Usage

### Single sequential test

This example supplies known variance one, as appropriate under an iid Gaussian model with that variance. It avoids the invalid empirical-variance default; it does not remove the other lifecycle/numerical limitations documented in the audit.

```python
from expectation.seqtest.sequential_e_testing import SequentialTesting

test = SequentialTesting(
    test_type="mean", null_value=0, alternative="greater", known_variance=1.0
)

result = test.update([0.5, 1.2, 0.8])
print(f"e-value: {result.e_value:.2f}, reject: {result.reject_null}")

result = test.update([1.5, 1.1])
print(f"cumulative e-process: {result.e_process.cumulative_value:.2f}")
```

### Massively parallel testing (Rust)

The simulated streams below use variance one. `ALL_IN` avoids presenting the native quadratic adaptive rule as equivalent to Python's empirical optimizer; see P07 in the audit. Multiple-testing calls here are snapshot queries, not repeated-look guarantees for arbitrary inputs.

```python
import numpy as np
from expectation.par_seqtest import ParallelSequentialTest, ParallelTestConfig

config = ParallelTestConfig(
    n_tests=300_000,
    alpha=0.05,
    alternative="greater",
    combiner="all_in",
)
engine = ParallelSequentialTest(config=config, null_values=0.0, variance=1.0)

for t in range(100):
    obs = np.random.randn(300_000)
    obs[:1000] += 0.5  # signal in first 1000
    result = engine.step(obs)

# Cross-test error control
bh = engine.e_bh()          # FDR control
bonf = engine.e_bonferroni() # FWER control

# Per-test state
log_ep = engine.log_e_processes()   # log e-process values
pvals = engine.p_values()           # calibrated p-values
stops = engine.stopping_times()     # when each test rejected
```

### Testing a Forecaster who announces a convex set (bid–ask spread)

```python
import numpy as np
from expectation.modules.hypothesistesting import EProcessConfig
from expectation.modules.protocol import BettingProtocol
from expectation.upperexp.config import UpperExpectationConfig
from expectation.upperexp.nulls import PolytopeNull
from expectation.upperexp.skeptic import UpperExpectationSkeptic

# Skeptic believes P(event) = 0.7; the market quotes bid 0.55 / ask 0.60 each round.
skeptic = UpperExpectationSkeptic(config=UpperExpectationConfig(alternative=[0.3, 0.7]))
protocol = BettingProtocol(skeptic, EProcessConfig(significance_level=0.05))
spread = PolytopeNull(np.array([[0.45, 0.55], [0.40, 0.60]]))  # hull of two Bernoullis
for outcome in np.random.default_rng(0).choice(2, p=[0.3, 0.7], size=200):
    result = protocol.update(outcome, announcement=spread)  # a fresh spread may be passed each round
print(result.reject_null, result.log_e_process)
```

## References

- Ramdas, Wang (2025). *Hypothesis testing with e-values*
- Howard, Ramdas, McAuliffe, Sekhon (2022). *Time-uniform, nonparametric, nonasymptotic confidence sequences*
- Waudby-Smith, Ramdas (2024). *Estimating means of bounded random variables by betting*
- Vovk, Wang (2021). *E-values: calibration, combination, and applications*
- Shafer, Vovk (2019). *Game-Theoretic Foundations for Probability and Finance*; Shafer (2021). *Testing by betting*
- Larsson, Ramdas, Ruf (2025). *Testing hypotheses generated by constraints*; Larsson, Ramdas, Ruf (2025). *The numeraire e-variable and reverse information projection*
- Grünwald, de Heide, Koolen (2024). *Safe testing*; Huber, Strassen (1973). *Minimax tests and the Neyman–Pearson lemma for capacities*
- Zhang, Glancy, Knill (2011). *Asymptotically optimal data analysis for rejecting local realism*; van Dam, Gill, Grünwald (2005). *The statistical strength of nonlocality proofs*
- Pérez-Ortiz, Castro, Stoepker (2025). *Anytime-valid tests for sparse anomalies*; Wang, Dandapanthula, Ramdas (2025). *Anytime-valid FDR control with the stopped e-BH procedure*
- Martinez-Taboada, Ramdas (2025). *Sequential kernelized Stein discrepancy*; Liu, Lee, Jordan (2016); Chwialkowski, Strathmann, Gretton (2016); Gorham, Mackey (2017); Wendland (1995)

## Citation

### BibTeX
```bibtex
@software{rostami2024expectation,
  author = {Rostami, Jako},
  title = {expectation: Sequential testing with e-values and e-processes},
  year = {2024},
  url = {https://github.com/jakorostami/expectation},
  version = {0.6.1}
}
```

### APA
```
Rostami, J. (2024). expectation: Python library for sequential testing and e-processes (Version 0.6.1) [Computer software]. https://github.com/jakorostami/expectation
```
