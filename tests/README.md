# RNA 3D Folding Pipeline Testing Framework

This directory contains the comprehensive testing framework for the RNA 3D folding pipeline. It includes unit tests, integration tests, performance benchmarks, and verification utilities for ensuring the correctness, robustness, and performance of all pipeline components.

## Test Organization

The test suite is organized by component, mirroring the structure of the `src/` directory:

- `test_data_loading.py`: Tests for RNA dataset, feature loading, and collation
- `test_embeddings.py`: Tests for sequence and positional embedding components
- `test_transformer_block.py`: Tests for self-attention and transformer mechanisms
- `test_ipa_module.py`: Tests for the IPA (Invariant Point Attention) module
- `test_losses.py`: Tests for loss functions (FAPE, confidence, angle losses)
- `test_model.py`: Tests for the end-to-end RNA folding model
- `test_padding.py`: Tests for padding utilities
- `test_integration.py`: Full pipeline integration tests

## Test Fixtures

Common test fixtures and utilities are defined in `conftest.py`, including:

- Mock data generators for RNA sequences, coordinates, features
- Device management (CPU/CUDA)
- Tensor validation helpers
- Memory profiling utilities
- Gradient flow verification
- Reproducibility helpers

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest tests/

# Run tests for a specific component
python -m pytest tests/test_data_loading.py

# Run a specific test class
python -m pytest tests/test_data_loading.py::TestRNADataset

# Run a specific test
python -m pytest tests/test_data_loading.py::TestRNADataset::test_initialization
```

### Test Options

```bash
# Run with verbose output
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src

# Run tests that match a pattern
python -m pytest tests/ -k "embedding"

# Run tests with print output
python -m pytest tests/ -s

# Run tests with xvs flag for detailed tensor assertions
python -m pytest tests/ -xvs
```

## Performance Benchmarking

A comprehensive benchmarking framework is provided to measure and track performance:

```bash
# Run benchmarks for all components
python tests/run_benchmarks.py

# Run benchmarks for specific components
python tests/run_benchmarks.py --components embeddings transformer ipa model

# Specify batch sizes and sequence lengths
python tests/run_benchmarks.py --batch_sizes 1 2 4 8 --seq_lengths 50 100 200

# Run quick benchmark
python tests/run_benchmarks.py --quick

# Run on CPU only
python tests/run_benchmarks.py --cpu_only
```

Benchmark results are saved to the `benchmark_results/` directory in CSV, JSON, and PNG formats.

## Verification Framework

The testing instance includes a formal verification framework for component validation:

- Verification plans define the approach, test cases, and acceptance criteria
- Verification reports document the results of verification activities
- Issue reports track identified problems and their resolution

Verification artifacts are stored in the `docs/claude/03_code-instances/instance_04_testing/verification_reports/` directory.

## Test Coverage

The test suite aims for >90% code coverage across all components, with 100% coverage for critical components like loss functions and the IPA module. Current coverage can be checked with:

```bash
python -m pytest tests/ --cov=src --cov-report=html
```

This will generate an HTML report in the `htmlcov/` directory.

## Test Development Guidelines

When adding new tests:

1. Use fixtures from `conftest.py` where possible
2. Follow the existing test organization pattern
3. Include both positive and negative test cases
4. Test edge cases and error conditions
5. Verify tensor shapes, types, and devices
6. Check gradient flow for components that should support backpropagation
7. Include memory and performance tests for complex operations
8. Add type annotations to all test functions
9. Document non-obvious test cases with docstrings

## Integration with CI/CD

The test suite is designed to integrate with CI/CD pipelines:

- All tests should be runnable in automation
- Tests should be deterministic and not depend on external resources
- Critical tests are tagged for inclusion in fast test runs
- Performance benchmarks can be run separately from functional tests

## Contributing New Tests

When contributing new tests:

1. Review the existing tests for similar components
2. Use the appropriate fixtures and utilities
3. Follow the naming convention: `test_[component]_[functionality]_[scenario]`
4. Run the tests locally before submitting
5. Include coverage for all edge cases and error conditions
6. Add verification plans for new components

## License

The testing framework is part of the RNA 3D Folding Pipeline and is subject to the same license as the main project.