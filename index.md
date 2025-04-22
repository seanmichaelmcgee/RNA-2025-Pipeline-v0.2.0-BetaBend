# RNA-2025-Pipeline Index

Date: 2025-04-21
Version: v0.2.0-BetaBend
Last Updated: 2025-04-21

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
│   ├── 8_Prompting_Strategy_and_Best_practices.md
│   ├── 09_prompt_aliases_cheatsheet.md
│   ├── 10_Validation_and_iteration_stragety
│   ├── Kaggle_References/     # Kaggle competition reference materials
│   │   ├── Kaggle_Data.md
│   │   ├── Kaggle_Overview.md
│   │   └── Kaggle_Rules.md
│   ├── claude/                # Claude Code implementation documentation
│   │   ├── 00-master_guide.md
│   │   ├── 01_implementation_principles.md
│   │   ├── 02_components/     # Component-specific documentation
│   │   │   ├── 10_data_loading/
│   │   │   │   ├── 11_data_loading_guide.md
│   │   │   │   ├── 12_data_loading_examples.md
│   │   │   │   ├── 13_data_loading_testing.md
│   │   │   │   ├── 14_data_loading_temporal_cutoff_implementation.md
│   │   │   │   └── 15_data_loading_v2_readiness_risks.md
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
│   │   │   │   ├── CLAUDE.md
│   │   │   │   ├── completed_components.md
│   │   │   │   ├── handoff/
│   │   │   │   │   └── nn_components_handoff.md
│   │   │   │   ├── implementation_journal.md
│   │   │   │   ├── implementation_plan.md
│   │   │   │   ├── implementation_timeline.md
│   │   │   │   ├── interface_exports.md
│   │   │   │   └── partial_data_handling.md
│   │   │   ├── instance_02_model/           # Model instance workspace
│   │   │   │   ├── CLAUDE.md
│   │   │   │   ├── completed_components.md
│   │   │   │   ├── handoff/
│   │   │   │   │   └── nn_components_handoff.md
│   │   │   │   ├── implementation_journal.md
│   │   │   │   └── interface_exports.md
│   │   │   ├── instance_03_integration/     # Integration instance workspace
│   │   │   │   ├── CLAUDE.md
│   │   │   │   ├── completed_components.md
│   │   │   │   ├── handoffs/
│   │   │   │   │   └── provided/
│   │   │   │   │       └── testing/
│   │   │   │   │           ├── losses_handoff.md
│   │   │   │   │           ├── rna_folding_model_handoff.md
│   │   │   │   │           ├── structure_metrics_handoff.md
│   │   │   │   │           └── validation_framework_handoff.md
│   │   │   │   ├── implementation_journal.md
│   │   │   │   ├── loss_function_analysis.md
│   │   │   │   └── loss_function_enhancement_plan.md
│   │   │   ├── instance_04_testing/         # Testing instance workspace
│   │   │   │   ├── CLAUDE.md
│   │   │   │   ├── completed_components.md
│   │   │   │   ├── implementation_journal.md
│   │   │   │   ├── issue_classification_system.md
│   │   │   │   ├── v1_implementation_plan.md
│   │   │   │   ├── verification_reports/     # Component verification reports
│   │   │   │   │   ├── component_verification_plan_template.md
│   │   │   │   │   ├── data_loading_verification_plan.md
│   │   │   │   │   ├── embeddings_verification_plan.md
│   │   │   │   │   ├── ipa_module_verification_plan.md
│   │   │   │   │   ├── issue_LOSS-001_kabsch_rotation.md
│   │   │   │   │   ├── issue_LOSS-002_collinear_points.md
│   │   │   │   │   ├── issue_LOSS-003_robust_distance.md
│   │   │   │   │   ├── issue_report_template.md
│   │   │   │   │   ├── loss_functions_verification_plan.md
│   │   │   │   │   ├── rna_folding_model_verification_plan.md
│   │   │   │   │   ├── transformer_block_verification_plan.md
│   │   │   │   │   └── verification_report_template.md
│   │   │   │   └── verification_status.md
│   │   │   └── shared/                      # Shared documentation for code instances
│   │   │       ├── 04_implementation_jorunal_template.md
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
│   │       ├── 70-pipeline-testing.md
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
│   └── validate_model.py      # Script to validate model performance
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
│       ├── padding.py         # Padding utilities for variable-length sequences
│       └── structure_metrics.py # Structure quality evaluation metrics
├── tests/                     # Test suite
│   ├── README.md              # Testing documentation
│   ├── benchmark_utils.py     # Benchmark utilities
│   ├── conftest.py            # Pytest configuration
│   ├── run_benchmarks.py      # Benchmark runner
│   ├── test_data_loading.py   # Tests for data loading
│   ├── test_embeddings.py     # Tests for embedding components
│   ├── test_integration.py    # Integration tests
│   ├── test_integration_fixed.py # Fixed integration tests
│   ├── test_ipa_module.py     # Tests for IPA module
│   ├── test_losses.py         # Tests for loss functions
│   ├── test_model.py          # Tests for main model
│   ├── test_padding.py        # Tests for padding utilities
│   ├── test_structure_metrics.py # Tests for structure metrics
│   └── test_transformer_block.py # Tests for transformer block
└── validation/                # Validation framework
    ├── README.md              # Validation framework overview
    ├── csv_coordinate_loader.py # Coordinate loader for validation
    ├── csv_data_loader.py     # Data loader for CSV files
    ├── dual_mode_implementation_summary.md # Dual-mode implementation summary
    ├── dual_mode_validation_README.md # Dual-mode validation guide
    ├── export_notebook_results.py # Export notebook results to JSON/CSV
    ├── metrics/               # Scientific validation metrics modules
    │   ├── feature_importance/ # Feature importance analysis
    │   │   ├── feature_importance.py # Feature ablation and importance quantification
    │   │   └── __init__.py    # Module initialization
    │   ├── __init__.py        # Metrics module initialization
    │   ├── rna_family/        # RNA family classification and analysis
    │   │   ├── __init__.py    # Module initialization
    │   │   └── rna_family.py  # RNA family classification and evaluation
    │   └── secondary_structure/ # Secondary structure analysis
    │       ├── __init__.py    # Module initialization
    │       └── secondary_structure.py # Base pair and motif evaluation
    ├── npz_feature_analysis.md # NPZ feature format analysis
    ├── npz_feature_loader.py  # Feature loader for NPZ files
    ├── run_dual_mode_validation.sh # Script to run validation in both modes
    ├── tier1_technical/       # Fast technical validation
    │   ├── README.md          # Technical validation documentation
    │   ├── debug_feature_loading.py # Debug feature loading issues
    │   ├── debug_rmsd_calculation.py # Debug RMSD calculation issues
    │   ├── debug_validation_runner.py # Debug validation runner issues
    │   ├── export_results.sh  # Export technical validation results
    │   ├── results/           # Technical validation results
    │   │   ├── per_residue_rmsd.png # Per-residue RMSD visualization
    │   │   ├── validation_results.json # Technical validation metrics
    │   │   └── validation_technical_results.md # Technical validation report
    │   ├── run_dual_mode_validation.py # Run validation in both modes
    │   ├── run_validation.py  # Run validation in single mode
    │   ├── tiered_dataset.py  # Dataset implementation for tiered validation
    │   └── validation_technical.ipynb # Technical validation notebook
    ├── tier2_scientific/      # Scientific validation for RNA-specific metrics
    │   ├── README.md          # Scientific validation documentation
    │   ├── results/           # Scientific validation results
    │   └── run_scientific_validation.py # Run scientific validation
    ├── tier3_comprehensive/   # Comprehensive validation for full benchmarking
    │   ├── README.md          # Comprehensive validation documentation
    │   └── results/           # Comprehensive validation results
    ├── validation_dataset.py  # Validation dataset implementation
    ├── validation_implementation_summary.md # Validation implementation summary
    ├── validation_runner.py   # Dual-mode validation runner
    ├── validation_status_update.md # Validation status update
    └── validation_status_update_v2.md # Updated validation status
```

## Key Components

### 1. Data Processing
- **Data Loading** (`src/data_loading.py`): ✅ Handles loading and processing RNA sequence and feature data.
  - PyTorch Dataset implementation for RNA sequences
  - Feature preprocessing and normalization
  - Batch collation with padding for variable-length sequences
  - Mask generation for attention mechanisms
  - Partial data handling with feature availability detection
  - Temporal cutoff for preventing data leakage
  - MI matrix optimization for memory efficiency
  - *Status: Implemented and tested*
- **Padding Utilities** (`src/utils/padding.py`): ✅ Provides efficient padding for tensors.
  - Memory-efficient padding for 1D, 2D, and N-D tensors
  - Supports different padding values and strategies
  - Optimized for variable-length RNA sequences
  - *Status: Implemented and tested*
- **Raw Data** (`data/raw/`): Contains MSA files, training sequences, labels, and validation data.

### 2. Model Architecture
- **RNA Folding Model** (`src/models/rna_folding_model.py`): Main model implementation for RNA folding prediction.
  - End-to-end model integrating all components
  - Configuration-driven architecture
  - Multiple prediction heads for coordinates and confidence
  - *Status: Implemented with basic functionality*

- **Embeddings** (`src/models/embeddings.py`): ✅ Handles feature embedding functionality.
  - Sequence token embeddings with support for padding
  - Sinusoidal positional encoding for absolute positions
  - Relative positional encoding between nucleotides
  - Feature projection with optional conservation features
  - Residue and pair representation initialization
  - *Status: Implemented and tested*

- **Transformer Block** (`src/models/transformer_block.py`): ✅ Implements transformer architecture.
  - Multi-head self-attention mechanism
  - Pair representation updates via outer product operations
  - Pre-normalization architecture for better training stability
  - Comprehensive mask handling for padded positions
  - *Status: Implemented and tested*

- **IPA Module** (`src/models/ipa_module.py`): ✅ Implements coordinate prediction (V1 placeholder).
  - Simplified MLP projection from residue representations to 3D coordinates
  - Interface-compatible with future full IPA implementation
  - Proper mask handling for padded positions
  - *Status: V1 placeholder implemented and tested*

### 3. Training Components
- **Loss Functions** (`src/losses.py`): Implements various loss functions.
  - FAPE loss for coordinate prediction
  - Confidence loss for structure quality estimation
  - Auxiliary angle prediction loss
  - Multi-component weighting mechanism
  - Numerical stability safeguards with robust distance calculation
  - *Status: Implemented, enhanced with specialized loss components*

### 4. Validation Framework
- **Validation Framework** (`validation/`): ✅ Infrastructure for comprehensive model validation.
  - Tiered validation system (Technical, Scientific, Comprehensive)
  - Dual-mode validation for test vs. train feature differences
  - Structure quality metrics (RMSD, TM-score)
  - Automatic error detection and reporting
  - Scientific validation metrics:
    - RNA family analysis (`validation/metrics/rna_family/`)
    - Secondary structure analysis (`validation/metrics/secondary_structure/`)
    - Feature importance analysis (`validation/metrics/feature_importance/`)
  - Technical validation tier (`validation/tier1_technical/`)
  - Scientific validation tier (`validation/tier2_scientific/`)
  - Comprehensive validation tier (`validation/tier3_comprehensive/`)
  - *Status: Technical validation complete, scientific validation in progress*

- **Structure Metrics** (`src/utils/structure_metrics.py`): ✅ Implements structure quality evaluation metrics.
  - RMSD calculation with Kabsch alignment
  - TM-score implementation
  - Robust distance calculation
  - Coordinate validation and sanity checks
  - *Status: Implemented and tested*

### 5. Testing Infrastructure
- **Test Suite** (`tests/`): Comprehensive testing for all components.
  - `test_data_loading.py`: Validates data loading functionality
  - `test_embeddings.py`: Tests embedding components and interfaces
  - `test_ipa_module.py`: Verifies IPA module coordinate prediction
  - `test_losses.py`: Verifies correctness of loss function implementations
  - `test_padding.py`: Tests padding utilities for variable-length sequences
  - `test_transformer_block.py`: Validates transformer block functionality
  - `test_structure_metrics.py`: Tests structure metrics implementations
  - `test_integration.py`: Integration tests across components
  - Coverage reporting and integration tests
  - *Status: Actively expanding with each component implementation*

### 6. Documentation
- **Project Documentation** (`docs/`): Comprehensive documentation including:
  - Project context and requirements
  - Architecture specifications
  - Implementation guides for each component
  - Workflow instructions and testing procedures
  - Kaggle competition references
  - Multi-instance coordination protocols
  - **Instance-Specific Documentation**:
    - Data pipeline documentation with interface contracts
    - Neural network components documentation with handoff protocols
    - Testing instance documentation with verification reports
    - *Status: Actively maintained and updated with implementations*

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

- **Validation**:
  ```bash
  cd validation
  ./run_dual_mode_validation.sh  # Run validation in both test and train modes
  ./run_dual_mode_validation.sh --subset scientific  # Run scientific validation
  ./run_dual_mode_validation.sh --rna-ids R1107 R1108  # Validate specific RNA IDs
  ```

- **Visualization & Experiment Tracking**:
  ```bash
  tensorboard --logdir=logs/  # View training logs
  # or use wandb for more robust experiment tracking
  ```

## Multi-Instance Development Architecture

This project uses a specialized multi-instance Claude Code architecture with four dedicated instances:

1. **Data Pipeline Instance (01)**: ✅ Specializes in data loading, feature processing, batch handling
   - Status: Completed with full test coverage and documentation

2. **Model Components Instance (02)**: ✅ Focuses on transformer blocks, IPA module, embeddings
   - Status: Core components implemented with tests and documentation

3. **Integration Instance (03)**: Manages end-to-end model, loss functions, configuration
   - Status: Basic implementation completed, validation framework in development

4. **Testing Instance (04)**: Dedicated to comprehensive testing across components
   - Status: Supporting component development with verification reports and integration tests

Each instance maintains deep context knowledge of specific components while sharing interfaces through structured handoff protocols and documentation. The project actively uses handoff documentation to enable smooth transitions between implementation phases.

## Current Implementation Progress

- ✅ Data pipeline functionality (100%)
- ✅ Core neural network components (100%)
- 🔄 End-to-end model integration (70%) 
- 🔄 Validation framework development (75%)
  - ✅ Technical validation tier (100%)
  - 🔄 Scientific validation tier (60%)
  - ⏳ Comprehensive validation tier (10%)
- ⬜ Training and evaluation pipeline (0%)
- ⬜ Kaggle submission preparation (0%)

## Recent Updates (2025-04-21)

- Added multi-tier validation framework with dual-mode architecture
- Created metrics module structure for scientific validation
- Fixed numerical stability issues in RMSD calculation
- Added RNA family, secondary structure, and feature importance analysis
- Enhanced reporting with problematic sample tracking
- Improved documentation and visualization components