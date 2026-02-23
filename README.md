<p>
    <a target="_blank">
      <img width="100%" src="https://github.com/jakorostami/expectation/blob/develop/assets/images/expectation.png" alt="expectation banner"></a>
  </p>

# expectation

Sequential hypothesis testing with e-values and e-processes. Optional Rust acceleration for massively parallel testing (300K+ simultaneous tests at ~24 ns/test).

> Pre-release (v0.5.2). API may change before 1.0.

## What this does

E-values replace p-values with a framework where you can monitor data continuously, stop whenever you want, and still have valid inference. No sample size calculations, no correction for peeking.

The library covers:

- **Sequential testing** — mean, proportion, quantile, variance tests with anytime-valid guarantees
- **Parallel engine** — Rust + rayon backend for 300K+ simultaneous tests (brain voxels, genomics, A/B tests at scale)
- **Multiple testing** — e-Bonferroni (FWER), e-BH (FDR), e-Holm (FWER) for cross-test error control
- **Confidence sequences** — time-uniform confidence intervals that are valid at every sample size
- **Calibration** — convert between p-values and e-values

![](https://github.com/jakorostami/expectation/blob/main/assets/images/seqplot.png)

## Install

```bash
git clone https://github.com/jakorostami/expectation.git
cd expectation
pip install -e .
```

For the Rust parallel engine:
```bash
pip install maturin
maturin develop --release
```

## Usage

### Single sequential test

```python
from expectation.seqtest import SequentialTesting

test = SequentialTesting(test_type="mean", null_value=0, alternative="greater")

result = test.update([0.5, 1.2, 0.8])
print(f"e-value: {result.e_value:.2f}, reject: {result.reject_null}")

result = test.update([1.5, 1.1])
print(f"cumulative e-process: {result.e_process.cumulative_value:.2f}")
```

### Massively parallel testing (Rust)

```python
import numpy as np
from expectation.par_seqtest import ParallelSequentialTest, ParallelTestConfig

config = ParallelTestConfig(
    n_tests=300_000,
    alpha=0.05,
    alternative="greater",
    combiner="empirically_adaptive",
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

## References

- Ramdas, Wang (2025). *Hypothesis testing with e-values*
- Howard, Ramdas, McAuliffe, Sekhon (2022). *Time-uniform, nonparametric, nonasymptotic confidence sequences*
- Waudby-Smith, Ramdas (2024). *Estimating means of bounded random variables by betting*
- Vovk, Wang (2021). *E-values: calibration, combination, and applications*

## Citation

### BibTeX
```bibtex
@software{rostami2024expectation,
  author = {Rostami, Jako},
  title = {expectation: Sequential testing with e-values and e-processes},
  year = {2024},
  url = {https://github.com/jakorostami/expectation},
  version = {0.5.2}
}
```

### APA
```
Rostami, J. (2024). expectation: Python library for sequential testing and e-processes (Version 0.5.2) [Computer software]. https://github.com/jakorostami/expectation
```
