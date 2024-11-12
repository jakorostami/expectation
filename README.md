# Expectation

A Python library for sequential testing and monitoring using e-values and e-processes. Based on modern developments in game-theoretic statistics, `expectation` provides valid inference at any stopping time, making it ideal for continuous monitoring and sequential analysis.

⚠️ **Pre-release Software Notice**: This library is currently in pre-release (v0.1.0). The repo may undergo significant changes before the 1.0.0 release. While the statistical implementations are sound, we recommend testing thoroughly before using in production environments.

## Why use Expectation?

🔄 **Truly Sequential**: Unlike traditional methods that require fixed sample sizes, `expectation` lets you analyze your data as it arrives, without penalty for multiple looks.

📊 **Always Valid**: Through the use of e-values and e-processes, your inference remains valid regardless of when you stop. Look at your data whenever you want!

💪 **Statistically Rigorous**: Built on solid theoretical foundations from game-theoretic probability and martingale theory, providing strong guarantees for error control.

🎯 **Interpretable**: E-values have a natural interpretation as betting outcomes or likelihood ratios, making them more intuitive than p-values for measuring evidence.

🛠️ **Flexible**: Supports various types of tests (means, proportions, quantiles, variances) and can be extended to custom scenarios.

![](https://github.com/jakorostami/expectation/blob/main/assets/images/eprocess.png)

## Who is it for?

### Data Scientists & Analysts
- Monitor A/B tests in real-time without worrying about peeking problems
- Analyze streaming data with valid statistical inference
- Get early signals about treatment effects while maintaining error control

### Researchers
- Conduct sequential analyses with proper error control
- Implement flexible stopping rules in experiments
- Use modern statistical methods based on game-theoretic foundations

### Engineers
- Build monitoring systems with statistical guarantees
- Implement automated decision rules based on sequential data
- Create robust testing pipelines

## Installing expectation 🎲

Getting started with `expectation` is easy! Here's how to set up the library for your statistical adventures.

### From Source 📦

If you want the latest development version:
```bash
git clone https://github.com/jakorostami/expectation.git
cd expectation
pip install -e .
```

## Simple Demo

Here's a quick example of how to use `expectation` for a sequential mean test:

```python
from expectation import SequentialTest

# Initialize a test for H0: μ = 0 vs H1: μ > 0
test = SequentialTest(
    test_type="mean",
    null_value=0,
    alternative="greater"
)

# First batch of data
result1 = test.update([0.5, 1.2, 0.8])
print(f"After 3 observations:")
print(f"E-value: {result1.e_value:.2f}")
print(f"Reject null: {result1.reject_null}")

# More data arrives
result2 = test.update([1.5, 1.1])
print(f"\nAfter 5 observations:")
print(f"E-value: {result2.e_value:.2f}")
print(f"Cumulative e-value: {result2.e_process.cumulative_value:.2f}")
print(f"Reject null: {result2.reject_null}")
```

Key features demonstrated:
1. Simple, intuitive interface
2. Sequential updates as new data arrives
3. Cumulative evidence tracking via e-process
4. Automatic handling of optional stopping
5. Clear rejection decisions

The test controls Type I error at level α (default 0.05) at ANY stopping time. No need to specify sample sizes in advance or adjust for multiple looks at the data!

## Contributing 🤝

We love contributions! Whether you're fixing bugs, adding features, or improving documentation, your help makes `expectation` better for everyone.

Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and join our friendly community. No contribution is too small, and all contributors are valued!

Want to help but not sure how? See our [Issues](https://github.com/jakorostami/expectation/issues) or start a [Discussion](https://github.com/jakorostami/expectation/discussions). We're happy to guide you! 🎲✨
