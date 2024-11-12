# Contributing to expectation

First off, thank you for considering contributing to expectation! It's people like you that help make expectation a great tool for statistical analysis and sequential testing.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct (see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)).

## License

By contributing to expectation, you agree that your contributions will be licensed under the GPL-3.0 License. See [LICENSE](LICENSE) for full details.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots or animated GIFs if possible
* Include your environment details (OS, Python version, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* A clear and descriptive title
* A detailed description of the proposed enhancement
* Examples of how the enhancement would be used
* If applicable, mathematical foundations or references

### Pull Requests

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Run the test suite and ensure all tests pass
5. Update documentation if needed
6. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
7. Push to the branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request


### Development Setup

1. Clone your fork of the repository:
```bash
git clone https://github.com/YOUR_USERNAME/expectation.git
cd expectation
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the package in editable mode with required dependencies:
```bash
pip install -e .
```

4. Install development tools:
```bash
pip install pytest pytest-cov black isort mypy pre-commit
```

5. Set up pre-commit hooks:
```bash
pre-commit install
```

## Styleguides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

### Python Styleguide

* Follow PEP 8 guidelines
* Use type hints for function arguments and return values
* Document functions and classes using NumPy docstring format
* Include doctest examples where appropriate
* Use descriptive variable names that reflect statistical/mathematical concepts

### Documentation Styleguide

* Use Markdown for documentation
* Include mathematical notation using LaTeX when necessary
* Provide examples for new features
* Reference academic papers or resources when introducing statistical concepts

### Testing

* Write tests for new features using pytest
* Ensure all tests pass before submitting a pull request
* Include tests for edge cases and error conditions
* For statistical functions, include tests with known distributions and outcomes

## Mathematical Contributions

When contributing new statistical methods or algorithms:

* Include references to relevant papers or textbooks
* Provide mathematical proofs or justifications where appropriate
* Include simulation studies demonstrating correctness
* Document assumptions and limitations
* Add appropriate test cases with known theoretical results

## Additional Notes

### Issue and Pull Request Labels

* `bug`: Something isn't working
* `enhancement`: New feature or request
* `documentation`: Documentation only changes
* `good first issue`: Good for newcomers
* `help wanted`: Extra attention is needed
* `math`: Involves mathematical theory or proofs
* `optimization`: Performance improvements
* `tests`: Adding or modifying tests

## Questions?

Feel free to open an issue with any questions about contributing. We're here to help!

Thank you for contributing to expectation!
