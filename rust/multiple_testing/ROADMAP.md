# Multiple Testing Module — Missing Pieces

## What We Have

| Procedure | Type | What it does |
|-----------|------|-------------|
| e-Bonferroni | FWER control | Reject test i if E_i >= m/alpha |
| e-BH | FDR control | Step-up: reject top k* where E_(k) >= m/(k*alpha) |
| e-Holm | FWER control | Step-down: reject while E_(k) >= (m-k+1)/alpha |

All three are **selection procedures** — they take per-test e-values and return which individual tests to reject. They never produce a combined quantity.

## What We Need

### 1. Vovk Admissible Merger (Global Null Test)

**Reference:** Vovk & Wang (2021), "E-values: calibration, combination, and applications", Section 4.

Given e-values E_1, ..., E_m from m tests, the arithmetic mean:

    E_bar = (1/m) * sum(E_i)

is itself an e-value for the intersection (global) null H_0^cap = "all nulls hold simultaneously". Reject the global null if E_bar >= 1/alpha.

Vovk proved this is the **only admissible** merging function for independent e-values.

**Why it matters:** The selection procedures tell you *which* voxels are active. The global merger tells you *whether any voxel is active at all* — a fundamentally different question. For brain imaging, this is the difference between "voxel 47,231 is active" and "there is brain activity somewhere in this region."

**Numerical note:** Must use log-sum-exp trick to avoid overflow when computing log(mean(exp(log_e))).
