# RNA-2025-Pipeline Index

Date: 2025-04-19
Version: v0.2.0

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
│   │   ├── 03_code-instances/ # Code instance documentation
│   │   │   ├── 01_data_pipeline_kickoff.md
│   │   │   ├── 02_model_kickoff.md
│   │   │   ├── 03_integration_kickoff.md
│   │   │   ├── 04_testing_kickoff.md
│   │   │   ├── README.md
│   │   │   ├── coordination/  # Empty coordination directory
│   │   │   ├── instance_01_data/  # Empty instance directory
│   │   │   ├── instance_02_model/ # Empty instance directory
│   │   │   ├── instance_03_integration/ # Empty instance directory
│   │   │   ├── instsance_04_testing/ # Empty instance directory (typo in name)
│   │   │   └── shared/        # Shared documentation for code instances
│   │   │       ├── 01_Code_instances_plan.md
│   │   │       ├── 02_Code_instances_implementation_and_GUIDE.md
│   │   │       ├── 03_Advanced_implementation_strategy.md
│   │   │       ├── 04_0implementation_jorunal_template.md
│   │   │       ├── 05_interface_contract_template.md
│   │   │       ├── 06_component_handoff_protocol
│   │   │       ├── 06_component_handoff_protocol.D
│   │   │       ├── 06_component_handoff_protocol.md
│   │   │       └── 07_component_status_tracker.md
│   │   ├── 04_reference/     # Reference materials
│   │   │   ├── configuration.md
│   │   │   ├── feature_formats.md
│   │   │   └── pytorch_patterns.md
│   │   └── 05_workflows/     # Workflow documentation
│   │       ├── 100_advanced_training_techniques.md
│   │       ├── 60_model_integration.md
│   │       ├── 70-pipeline-testing.md
│   │       ├── 80_debugging.md
│   │       └── 90_kaggle_submission.md
│   └── data_examples/        # Example data files
│       ├── 1A51_A_dihedral_features.npz.txt
│       ├── 1A51_A_features.npz.txt
│       ├── 1A51_A_thermo_features.npz.txt
│       └── train_features_example.md
├── environment.yml           # Conda environment configuration
├── index.md                  # This file - codebase index
├── scripts/                  # Utility scripts
│   └── test_data_loading.py  # Script to test data loading
├── setup_project.sh          # Project setup script
├── src/                      # Source code
│   ├── data_loading.py       # Data loading functionality
│   ├── ipa-module-tests.py   # IPA module test scripts
│   ├── losses.py             # Loss functions
│   └── models/               # Model components
│       ├── embeddings.py     # Feature embedding module
│       ├── ipa_module.py     # Invariant Point Attention module
│       ├── rna_folding_model.py # Main RNA folding model
│       └── transformer_block.py # Transformer architecture
└── tests/                    # Test suite
    ├── test_data_loading.py  # Tests for data loading
    └── test_losses.py        # Tests for loss functions
```

## Key Components

### Data Processing
- **Data Loading** (`src/data_loading.py`): Handles loading and processing RNA sequence and feature data.
- **Raw Data** (`data/raw/`): Contains MSA files, training sequences, labels, and validation data.

### Model Architecture
- **RNA Folding Model** (`src/models/rna_folding_model.py`): Main model implementation for RNA folding prediction.
- **Embeddings** (`src/models/embeddings.py`): Handles feature embedding functionality to convert raw features into model-compatible representations.
- **IPA Module** (`src/models/ipa_module.py`): Implements the Invariant Point Attention mechanism for geometric reasoning.
- **Transformer Block** (`src/models/transformer_block.py`): Contains transformer architecture components for sequence modeling.

### Training Components
- **Loss Functions** (`src/losses.py`): Implements various loss functions for training the RNA folding model.

### Testing 
- **Test Suite** (`tests/`): Contains unit tests for different components of the pipeline.
  - `test_data_loading.py`: Validates data loading functionality.
  - `test_losses.py`: Verifies correctness of loss function implementations.

### Documentation
- **Project Documentation** (`docs/`): Comprehensive documentation covering:
  - Project context and requirements
  - Architecture specifications
  - Implementation guides for each component
  - Workflows and testing procedures
  - Kaggle competition references

## Build and Test Commands

- **Environment Setup**: `conda env create -f environment.yml`
- **Running Tests**: `python -m pytest tests/`
- **Running Specific Tests**: `python -m pytest tests/test_data_loading.py::TestRNADataset::test_initialization`
- **Test with Verbosity**: `python -m pytest tests/ -v`
- **Test with Coverage**: `python -m pytest tests/ --cov=src`
- **Linting**: `black src/ tests/` and `isort src/ tests/`
- **Type Checking**: `mypy src/`