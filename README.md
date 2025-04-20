# RNA 2025 Pipeline: 3D Structure Prediction

A PyTorch-based machine learning pipeline for predicting RNA 3D structures from sequence data, targeting the Stanford RNA 3D Folding Kaggle competition.

## Project Overview

This project implements a neural network architecture that integrates RNA thermodynamic features, evolutionary coupling information, and specialized transformer-based components to predict RNA tertiary structures. The pipeline uses a modular design for high reproducibility, maintainability, and performance.

Key features:
- Reproducible containerized environment with comprehensive dependencies
- Modular, well-tested components with full test coverage
- Strong separation between core logic and orchestration
- Flexible configuration system for experiment tracking
- Multiple candidate structure prediction with confidence scores
- Specialized multi-instance development architecture

## Data Format

The pipeline expects the following data structure:

```
data/
├── raw/                      # Raw input data
│   ├── train_sequences.csv   # RNA target ID and sequences
│   └── train_labels.csv      # Ground truth 3D coordinates
└── processed/                # Precomputed features
    ├── dihedral_features/    # Backbone geometry features
    │   └── {target_id}_dihedral_features.npz
    ├── thermo_features/      # Thermodynamic folding features
    │   └── {target_id}_thermo_features.npz
    └── mi_features/          # Mutual information/evolutionary features
        └── {target_id}_mi_features.npz
```

### Feature File Details

1. **Dihedral Features** (.npz):
   - `features`: Sin/cos encodings of η and θ pseudo-dihedral angles (shape `(N, 4)`)
   - Used for auxiliary supervision during training

2. **Thermodynamic Features** (.npz):
   - `pairing_probs`: Base-pairing probability matrix (shape `(N, N)`)
   - `positional_entropy`: Shannon entropy for each position (shape `(N,)`)
   - `accessibility`: Unpaired probability per nucleotide (shape `(N,)`)
   - Various scalar features (MFE, ensemble energy, etc.)

3. **MI Features** (.npz):
   - `coupling_matrix`: Mutual information scores between residue pairs (shape `(N, N)`)
   - Used to capture evolutionary co-variation signal

## Core Components

### 1. Data Loading (`src/data_loading.py`)

- `RNADataset`: PyTorch Dataset for loading RNA sequences and precomputed features
- `collate_fn`: Handles variable-length sequences with proper padding and masking
- Feature normalization and default handling for missing features

```python
# Example usage
dataset = RNADataset(
    sequences_csv_path=config.sequences_path,
    labels_csv_path=config.labels_path,
    features_dir=config.features_dir
)
```

### 2. Embeddings (`src/models/embeddings.py`)

- `SequenceEmbedding`: Maps nucleotide tokens to learned representations
- `PositionalEncoding`: Provides position information using sinusoidal patterns
- `RelativePositionalEncoding`: Captures relative distances between nucleotide pairs
- Feature projection layers for both residue and pair representations

### 3. Transformer Block (`src/models/transformer_block.py`)

- Multi-head self-attention mechanism for processing residue representations
- Pair representation update using outer products and MLPs
- Pre-normalization architecture for training stability
- Residual connections and dropouts for robust learning

### 4. IPA Module (`src/models/ipa_module.py`)

- V1: Simplified placeholder that projects residue features to 3D coordinates
- Future V2+: Full Invariant Point Attention with frame-based representation and iterative refinement
- Designed for predicting C1' atom coordinates of RNA nucleotides

### 5. Loss Functions (`src/losses.py`)

- Simplified FAPE loss proxy for coordinate prediction
- Confidence loss for structure quality prediction
- Angle prediction loss for auxiliary supervision
- Multi-component loss weighting mechanism

### 6. Main Model (`src/models/rna_folding_model.py`)

- Integration of all components into a unified architecture
- Multiple transformer block stacking
- End-to-end prediction from features to 3D coordinates
- Configuration-driven parameter management

## Multi-Instance Development Architecture

This project is being developed using a specialized multi-instance Claude Code architecture that divides work across four specialized AI instances, each focused on specific components of the system.

### 1. Overview of Multi-Instance Approach

Each Claude Code instance specializes in a specific part of the codebase, enabling:
- Deep expertise in particular component groups
- Efficient context utilization through focused development
- Clear responsibility boundaries
- Parallel development across the pipeline
- Structured knowledge transfer through documentation

### 2. The Four Specialized Instances

1. **Data Pipeline Instance (01_data_pipeline)**
   - Dataset implementation for RNA sequences and features
   - Feature loading and processing utilities
   - Batch collation for variable-length sequences
   - Masking and padding mechanisms

2. **Model Components Instance (02_model_components)**
   - Embedding layers for sequence and positional representations
   - Transformer blocks with attention mechanisms
   - IPA module implementation for coordinate prediction
   - Component-level testing and validation

3. **Integration Instance (03_integration)**
   - Main model architecture assembly
   - Loss function implementation
   - End-to-end data flow coordination
   - Configuration handling and hyperparameter management

4. **Testing Instance (04_testing)**
   - Comprehensive test suite development
   - Edge case testing and validation
   - Performance benchmarking
   - Integration testing across components

### 3. Coordination and Knowledge Sharing

This architecture relies on structured communication and formalized documentation:

- **Interface Contracts**: Formal specifications detailing tensor shapes, types, and behavior
- **Implementation Journals**: Records of decisions, progress, and challenges
- **Handoff Protocol**: Standardized process for transitioning components between instances
- **Component Status Tracking**: Global visibility into implementation progress
- **Shared Documentation**: Common reference materials and design patterns

### 4. Communication Flow

```
┌───────────────────┐      ┌───────────────────┐
│                   │      │                   │
│  01_data_pipeline ├─────►│  03_integration   │
│  (Data Processing)│      │  (Model Assembly) │
│                   │      │                   │
└───────────────────┘      └─────────┬─────────┘
                                     │
┌───────────────────┐                │
│                   │                │
│ 02_model_components ◄───────┐      │
│  (Neural Network)  │        │      │
│                   │        │      │
└─────────┬─────────┘        │      │
          │                  │      │
          ▼                  │      ▼
┌───────────────────┐        │      │
│                   │        │      │
│    04_testing     ◄────────┴──────┘
│  (Verification)   │
│                   │
└───────────────────┘
```

## Workflows

### Development Workflow

1. **Local Development** (`Dev`):
   - Work within Docker container with mounted local directories
   - Access to full GPU resources for training
   - Regular unit testing during development

2. **Validation/Testing** (`Test`):
   - Run comprehensive tests with validation datasets
   - Evaluate model performance metrics
   - Verify Kaggle compatibility

3. **Kaggle Submission** (`Prod`):
   - Package code for Kaggle notebook execution
   - Ensure submission format matches competition requirements
   - Optimize for Kaggle runtime constraints

### Training and Inference

```python
# Training workflow
model = RNAFoldingModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

for epoch in range(config.num_epochs):
    for batch in train_dataloader:
        outputs = model(batch)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Inference workflow
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        outputs = model(batch)
        coordinates = outputs['pred_coords']  # Shape: (B, L, 3)
        confidence = outputs['confidence']    # Shape: (B, L)
```

## Key Implementation Principles

1. **Path Parameterization (Critical)**
   - No hardcoded paths in source modules
   - All file/directory paths passed as arguments
   - Use `os.path.join()` for path construction
   - Enables compatibility with both local dev and Kaggle environments

2. **Modularity and Testing**
   - Clear interfaces between components
   - Independent testability of each module
   - Comprehensive unit tests alongside implementation
   - Memory and performance optimization

3. **Device Compatibility**
   - All components work on both CPU and CUDA
   - Device-agnostic tensor operations
   - Proper handling of tensor device placement

4. **Error Handling and Robustness**
   - Informative error messages
   - Graceful handling of missing features or edge cases
   - Numerical stability in all operations
   - Proper mask propagation throughout pipeline

## Next Development Steps

1. **Complete Data Pipeline Implementation**
   - Finalize feature loading with robust error handling
   - Implement efficient batch collation for variable sequences
   - Add comprehensive test coverage

2. **Develop Core Model Components**
   - Implement embedding and transformer modules
   - Create IPA module placeholder with coordinate prediction
   - Test shape transformations and mask propagation

3. **Integrate Full Model Architecture**
   - Assemble components into end-to-end model
   - Implement loss functions with proper weighting
   - Create training loop and evaluation metrics

4. **Comprehensive Testing and Optimization**
   - Validate with synthetic data and real examples
   - Optimize memory usage and performance
   - Prepare Kaggle-compatible submission pipeline

## References

- **Project Documentation**: See `docs/` directory for detailed specifications
- **Feature Format Reference**: `docs/claude/reference/feature_formats.md`
- **PyTorch Patterns**: `docs/claude/reference/pytorch_patterns.md`
- **Component Guides**: `docs/claude/components/` directory
- **Claude Code Instances**: `docs/claude/code-instances/` directory

## Development Environment

### Setup and Testing
```bash
# Create and activate environment (use mamba for faster installation)
mamba env create -f environment.yml  # or use conda: conda env create -f environment.yml
mamba activate rna-3d-folding        # or use conda: conda activate rna-3d-folding

# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_data_loading.py::TestRNADataset::test_initialization

# Run tests with coverage
python -m pytest tests/ --cov=src
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Visualization and Experiment Tracking
The environment includes PyTorch with CUDA support, plotting libraries (Matplotlib, Seaborn, Plotly), 
molecular visualization (py3Dmol), and experiment tracking tools (TensorBoard, Weights & Biases).

## Contributors

- RNA 2025 Team
- Claude AI Assistant (Anthropic)