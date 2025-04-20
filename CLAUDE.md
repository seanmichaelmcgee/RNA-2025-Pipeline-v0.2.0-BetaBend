# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Install environment: `mamba env create -f environment.yml` (or `conda env create -f environment.yml`)
- Activate environment: `mamba activate rna-3d-folding` (or `conda activate rna-3d-folding`)
- Run all tests: `python -m pytest tests/`
- Run single test: `python -m pytest tests/test_data_loading.py::TestRNADataset::test_initialization`
- Run tests with verbosity: `python -m pytest tests/ -v`
- Run tests with coverage: `python -m pytest tests/ --cov=src`
- Lint: `black src/ tests/` and `isort src/ tests/`
- Type checking: `mypy src/`
- View logs: `tensorboard --logdir=logs/` or using wandb

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
- Use PyTorch tensors for all model operations
- Ensure numerical stability with epsilon values in mathematical operations
- Mark optional parameters with default values in function signatures

## Project Dependencies
- Core ML: PyTorch 2.1+, CUDA 12.1, numpy, pandas, scipy
- Tensor operations: einops for transformer implementations
- Visualization: matplotlib, seaborn, plotly, py3Dmol
- Experiment tracking: tensorboard, wandb
- Testing: pytest with coverage
- Code quality: black, isort, mypy

## Core Components
1. Data Loading: `src/data_loading.py` - Dataset implementation for feature handling
2. Embeddings: `src/models/embeddings.py` - Sequence embeddings and positional encoding
3. Transformer: `src/models/transformer_block.py` - Self-attention mechanisms 
4. IPA Module: `src/models/ipa_module.py` - Coordinate prediction
5. Loss Functions: `src/losses.py` - FAPE loss and auxiliaries
6. Main Model: `src/models/rna_folding_model.py` - End-to-end architecture

## Development Philosophy
- Use multi-instance architecture with focused components
- Maintain path parameterization for Kaggle compatibility
- Keep modular design with well-defined interfaces
- Ensure comprehensive test coverage
- Target both CPU and CUDA environments