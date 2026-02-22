# Rust Setup Guide for Expectation

This document explains how to work with the Rust components of the `expectation` library.

## Prerequisites

### 1. Install Rust

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installation, restart your terminal or run:
```bash
source $HOME/.cargo/env
```

Verify installation:
```bash
rustc --version
cargo --version
```

### 2. Install Maturin

Maturin is the build tool for PyO3-based Python extensions:

```bash
pip install maturin
```

## Project Structure

```
expectation/
├── rust/              # Rust source code
│   └── lib.rs         # Main Rust library entry point
├── expectation/       # Python source code
│   ├── __init__.py
│   ├── conformal/
│   ├── confseq/
│   └── ...
├── Cargo.toml         # Rust package configuration
└── pyproject.toml     # Python package configuration (with maturin backend)
```

## Development Workflow

### Building the Extension

**Development build** (faster, includes debug symbols):
```bash
maturin develop
```

This compiles the Rust code and installs the extension into your current Python environment.

**Release build** (optimized):
```bash
maturin develop --release
```

### Testing the Rust Extension

After building, you can test the Rust integration from Python:

```python
from expectation._rust import hello_rust

print(hello_rust())  # Should print: "Hello from Rust! PyO3 integration is working."
```

### Running Rust Tests

```bash
cargo test
```

### Running Rust Benchmarks

```bash
cargo bench
```

### Building Python Wheels

To create distributable wheels:

```bash
maturin build --release
```

Wheels will be created in `target/wheels/`.

## Adding Rust Functionality

### 1. Create a New Module

Add a new file in the `rust/` directory, e.g., `rust/statistics.rs`:

```rust
use pyo3::prelude::*;

#[pyfunction]
fn fast_mean(values: Vec<f64>) -> PyResult<f64> {
    let sum: f64 = values.iter().sum();
    Ok(sum / values.len() as f64)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_mean, m)?)?;
    Ok(())
}
```

### 2. Register the Module

Update `rust/lib.rs`:

```rust
// Add module declaration
pub mod statistics;

// Inside the expectation_rust function:
statistics::register(m)?;
```

### 3. Rebuild and Test

```bash
maturin develop
python -c "from expectation._rust import fast_mean; print(fast_mean([1.0, 2.0, 3.0]))"
```

## Common Dependencies for Statistical Work

Uncomment these in `Cargo.toml` as needed:

```toml
[dependencies]
# Linear algebra and arrays
ndarray = "0.16"
numpy = "0.22"  # NumPy integration

# Statistical functions
statrs = "0.17"

# Numerical traits
num-traits = "0.2"

# Parallel processing
rayon = "1.10"
```

## Performance Tips

1. **Use `--release` for benchmarking**: Development builds are ~10-100x slower
2. **Profile before optimizing**: Use `cargo bench` to measure
3. **Leverage parallelism**: Use `rayon` for data-parallel operations
4. **Minimize Python/Rust boundary crossings**: Process data in bulk rather than element-by-element

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Test Rust

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - uses: dtolnay/rust-toolchain@stable
      - name: Install maturin
        run: pip install maturin
      - name: Build
        run: maturin develop
      - name: Test Rust
        run: cargo test
      - name: Test Python
        run: pytest
```

## Troubleshooting

### "cargo: command not found"

Make sure Rust is installed and in your PATH:
```bash
source $HOME/.cargo/env
```

### "maturin: command not found"

Install maturin:
```bash
pip install maturin
```

### Build errors with PyO3

Ensure you're using Python 3.10 or later (as specified in `Cargo.toml`).

### Slow compilation

First build is always slow. Subsequent builds use incremental compilation and are much faster.

## Resources

- [PyO3 Documentation](https://pyo3.rs/)
- [Maturin Documentation](https://www.maturin.rs/)
- [Rust Book](https://doc.rust-lang.org/book/)
- [ndarray Documentation](https://docs.rs/ndarray/)
- [statrs Documentation](https://docs.rs/statrs/)
