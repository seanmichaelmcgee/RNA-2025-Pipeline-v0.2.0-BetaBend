# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Install environment: `conda env create -f environment.yml`
- Run all tests: `python -m pytest tests/`
- Run single test: `python -m pytest tests/test_data_loading.py::TestRNADataset::test_initialization`
- Lint: `black src/ tests/` and `isort src/ tests/`
- Type checking: `mypy src/`

## Code Style
- Import order: standard lib, third-party packages, local modules
- Type hints required for all function signatures (PEP 484)
- DocStrings: Follow Google style with Args/Returns sections
- Naming: snake_case for functions/variables, CamelCase for classes
- Error handling: Use explicit error messages, handle expected exceptions
- Comments: Add comments for complex algorithms or non-obvious code
- Use f-strings for string formatting
- Always validate inputs, especially for data loading functions
- Use dict/list comprehensions when appropriate for readability