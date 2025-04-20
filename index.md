# RNA-2025-Pipeline Index

Date: 2025-04-20
Version: v0.2.0-BetaBend

## Project Structure

```
├── CLAUDE.md                  # Claude Code configuration and instructions
├── Dockerfile                 # Docker configuration for project environment
├── LICENSE                    # Project license
├── README.md                  # Project overview and documentation
├── data/                      # Directory for RNA data files
│   └── raw/                   # Raw data files
│       ├── MSA/               # Multiple Sequence Alignment data (500+ FASTA files)
│       ├── snippet/           # Data snippets for testing
│       │   ├── train_labels.csv
│       │   └── train_sequences.csv
│       ├── create_snippets.py # Script to create data snippets
│       ├── README.md          # Data documentation
│       ├── sample_submission.csv
│       ├── test_sequences.csv
│       ├── train_labels.csv
│       ├── train_sequences.csv
│       ├── validation_labels.csv
│       └── validation_sequences.csv
├── docs/                      # Project documentation
│   ├── 1_Context_and_Setup.md
│   ├── 2_Feature_Specification.md
│   ├── 3_Architecture_Specification.md
│   ├── 4_Product_Requirements_V1.md
│   ├── 5_Roadmap_V1.md
│   ├── 6_Tactical_Plan_V1.md
│   ├── 7_AI_Agent_Rules.md
│   ├── Kaggle_References/     # Kaggle competition reference materials
│   │   ├── Kaggle_Data.md
│   │   ├── Kaggle_Overview.md
│   │   └── Kaggle_Rules.md
│   ├── claude-code-prompting-strategy-Apr9.md
│   ├── claude/                # Claude Code implementation documentation
│   │   ├── 00-master_guide.md
│   │   ├── 01_implementation_principles.md
│   │   ├── 02_components/     # Component-specific documentation
│   │   │   ├── 10_data_loading/
│   │   │   │   ├── 11_data_loading_guide.md
│   │   │   │   ├── 12_data_loading_examples.md
│   │   │   │   ├── 13_data_loading_testing.md
│   │   │   │   └── 14_data_loading_temporal_cutoff_implementation.md
│   │   │   ├── 20_embeddings/
│   │   │   │   ├── 21_embeddings_guide.md
│   │   │   │   ├── 22_embeddings_examples.md
│   │   │   │   └── 23_embeddings_testing.md
│   │   │   ├── 30_transformer_block/
│   │   │   │   ├── 31_transformer_guide.md
│   │   │   │   ├── 32_transformer_examples.md
│   │   │   │   └── 33_transformer_testing.md
│   │   │   ├── 40_ipa_module/
│   │   │   │   ├── 41_ipa_guide.md
│   │   │   │   ├── 42_ipa_examples.md
│   │   │   │   └── 43_ipa-testing-guide.md
│   │   │   ├── 50_losses/
│   │   │   │   ├── 51_losses_guide.md
│   │   │   │   ├── 52_losses_examples.md
│   │   │   │   └── 53_losses_tests.md
│   │   │   ├── 60_visualizations/
│   │   │   │   └── 61_visualizations_guide.md
│   │   │   └── component-guide-template.md
│   │   ├── 03_code-instances/ # Multi-instance architecture documentation
│   │   │   ├── 01_data_pipeline_kickoff.md  # Data instance kickoff
│   │   │   ├── 02_model_kickoff.md          # Model instance kickoff
│   │   │   ├── 03_integration_kickoff.md    # Integration instance kickoff
│   │   │   ├── 04_testing_kickoff.md        # Testing instance kickoff
│   │   │   ├── README.md                    # Overview of multi-instance approach
│   │   │   ├── coordination/                # Coordination mechanisms
│   │   │   ├── instance_01_data/            # Data instance workspace
│   │   │   ├── instance_02_model/           # Model instance workspace
│   │   │   ├── instance_03_integration/     # Integration instance workspace
│   │   │   ├── instance_04_testing/         # Testing instance workspace (fixed typo)
│   │   │   └── shared/                      # Shared documentation for code instances
│   │   │       ├── 04_implementation_journal_template.md
│   │   │       ├── 05_interface_contract_template.md
│   │   │       ├── 06_component_handoff_protocol.md
│   │   │       ├── 07_component_status_tracker.md
│   │   │       ├── 08_component_handoff_template.md
│   │   │       └── 09_advanced_reference.md
│   │   ├── 04_reference/      # Reference materials
│   │   │   ├── configuration.md
│   │   │   ├── feature_formats.md
│   │   │   └── pytorch_patterns.md
│   │   └── 05_workflows/      # Workflow documentation
│   │       ├── 60_model_integration.md
│   │       ├── 70_pipeline_testing.md
│   │       ├── 80_debugging.md
│   │       ├── 90_kaggle_submission.md
│   │       └── 100_advanced_training_techniques.md
│   └── data_examples/         # Example data files
│       ├── 1A51_A_dihedral_features.npz.txt
│       ├── 1A51_A_features.npz.txt
│       ├── 1A51_A_thermo_features.npz.txt
│       └── train_features_example.md
├── environment.yml            # Conda/mamba environment configuration
├── index.md                   # This file - codebase index
├── scripts/                   # Utility scripts
│   ├── run_data_loading.py    # Script to run the data pipeline
│   └── test_data_loading.py   # Script to test data loading
├── setup_project.sh           # Project setup script
├── src/                       # Source code
│   ├── data_loading.py        # Data loading functionality
│   ├── ipa-module-tests.py    # IPA module test scripts
│   ├── losses.py              # Loss functions
│   ├── models/                # Model components
│   │   ├── embeddings.py      # Feature embedding module
│   │   ├── ipa_module.py      # Invariant Point Attention module
│   │   ├── rna_folding_model.py # Main RNA folding model
│   │   └── transformer_block.py # Transformer architecture
│   └── utils/                 # Utility functions
│       └── padding.py         # Padding utilities for variable-length sequences
└── tests/                     # Test suite
    ├── test_data_loading.py   # Tests for data loading
    ├── test_losses.py         # Tests for loss functions
    └── test_padding.py        # Tests for padding utilities
```

## Key Components

### 1. Data Processing
- **Data Loading** (`src/data_loading.py`): Handles loading and processing RNA sequence and feature data.
  - PyTorch Dataset implementation for RNA sequences
  - Feature preprocessing and normalization
  - Batch collation with padding for variable-length sequences
  - Mask generation for attention mechanisms
  - Partial data handling with feature availability detection
- **Padding Utilities** (`src/utils/padding.py`): Provides efficient padding for tensors.
  - Memory-efficient padding for 1D, 2D, and N-D tensors
  - Supports different padding values and strategies
  - Optimized for variable-length RNA sequences
- **Raw Data** (`data/raw/`): Contains MSA files, training sequences, labels, and validation data.

### 2. Model Architecture
- **RNA Folding Model** (`src/models/rna_folding_model.py`): Main model implementation for RNA folding prediction.
  - End-to-end model integrating all components
  - Configuration-driven architecture
  - Multiple prediction heads for coordinates and confidence

- **Embeddings** (`src/models/embeddings.py`): Handles feature embedding functionality.
  - Sequence token embeddings
  - Positional encoding (both absolute and relative)
  - Feature projection layers

- **Transformer Block** (`src/models/transformer_block.py`): Implements transformer architecture.
  - Multi-head self-attention mechanism
  - Pair representation updates
  - Pre-normalization and residual connections

- **IPA Module** (`src/models/ipa_module.py`): Implements geometric reasoning for 3D coordinates.
  - Invariant Point Attention mechanism
  - Frame-based representation
  - Coordinate prediction

### 3. Training Components
- **Loss Functions** (`src/losses.py`): Implements various loss functions.
  - FAPE loss for coordinate prediction
  - Confidence loss for structure quality estimation
  - Auxiliary angle prediction loss
  - Multi-component weighting mechanism

### 4. Testing Infrastructure
- **Test Suite** (`tests/`): Comprehensive testing for all components.
  - `test_data_loading.py`: Validates data loading functionality
  - `test_losses.py`: Verifies correctness of loss function implementations
  - `test_padding.py`: Tests padding utilities for variable-length sequences
  - More test files to be added for each component
  - Coverage reporting and integration tests

### 5. Documentation
- **Project Documentation** (`docs/`): Comprehensive documentation including:
  - Project context and requirements
  - Architecture specifications
  - Implementation guides for each component
  - Workflow instructions and testing procedures
  - Kaggle competition references
  - Multi-instance coordination protocols

## Development Environment

### Build and Test Commands

- **Environment Setup**: 
  ```bash
  mamba env create -f environment.yml  # Faster than conda
  mamba activate rna-3d-folding
  ```

- **Running Tests**:
  ```bash
  python -m pytest tests/                                                     # All tests
  python -m pytest tests/test_data_loading.py::TestRNADataset::test_initialization  # Specific test
  python -m pytest tests/ -v                                                  # Verbose output
  python -m pytest tests/ --cov=src                                           # With coverage
  ```

- **Code Quality**:
  ```bash
  black src/ tests/     # Format code
  isort src/ tests/     # Sort imports
  mypy src/             # Type checking
  ```

- **Visualization & Experiment Tracking**:
  ```bash
  tensorboard --logdir=logs/  # View training logs
  # or use wandb for more robust experiment tracking
  ```

## Multi-Instance Development Architecture

This project uses a specialized multi-instance Claude Code architecture with four dedicated instances:

1. **Data Pipeline Instance (01)**: Specializes in data loading, feature processing, batch handling
2. **Model Components Instance (02)**: Focuses on transformer blocks, IPA module, embeddings
3. **Integration Instance (03)**: Manages end-to-end model, loss functions, configuration
4. **Testing Instance (04)**: Dedicated to comprehensive testing across components

Each instance maintains deep context knowledge of specific components while sharing interfaces through structured handoff protocols and documentation.