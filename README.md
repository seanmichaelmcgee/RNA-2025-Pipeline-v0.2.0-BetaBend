# RNA 2025 Pipeline: 3D Structure Prediction

A PyTorch-based machine learning pipeline for predicting RNA 3D structures from sequence data, targeting the Stanford RNA 3D Folding Kaggle competition.

## Project Overview

This project implements a neural network architecture that integrates RNA thermodynamic features, evolutionary coupling information, and specialized transformer-based components to predict RNA tertiary structures. The pipeline uses a modular design for high reproducibility, maintainability, and performance.

Key features:
- Reproducible containerized environment
- Modular, well-tested components
- Strong separation between core logic and orchestration
- Flexible configuration system
- Multiple candidate structure prediction with confidence scores

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

## Claude Code Multi-Instance Strategy

This project employs an innovative approach using multiple Claude Code instances working in parallel on different components. Each instance specializes in a specific part of the codebase:

### 1. Instance Structure

- **Data Pipeline Instance (01_data_pipeline)**:
  - Implements data loading components
  - Handles feature preprocessing
  - Creates dataset and dataloader classes

- **Model Components Instance (02_model_components)**:
  - Implements embedding layers
  - Builds transformer blocks
  - Creates IPA module placeholder

- **Integration Instance (03_integration)**:
  - Assembles the main model
  - Implements loss functions
  - Handles end-to-end integration

- **Testing Instance (04_testing)**:
  - Develops comprehensive test suites
  - Tests edge cases and error conditions
  - Verifies performance and memory usage

### 2. Coordination Approach

- **Interface-First Development**:
  - Define interfaces before implementation
  - Establish strict contracts between components
  - Document tensor shapes and types thoroughly

- **Cross-Instance Communication**:
  - Standardized handoff procedures
  - Detailed interface documentation
  - Common reference materials

- **Unified Code Standards**:
  - Consistent naming conventions
  - Standard docstring format
  - Clear component boundaries

### 3. Implementation Workflow

- **Initial Sequential Phase**:
  - Begin with Data Pipeline instance
  - Followed by Model Components
  - Then Integration instance
  - Testing in parallel throughout

- **Transition to Parallel Development**:
  - Multiple instances work simultaneously
  - Coordinate through interface documentation
  - Regular synchronization points

## Next Major Steps

1. **Data Pipeline Implementation**:
   - Complete `RNADataset` class implementation
   - Add comprehensive feature loading utilities
   - Implement collate function for batching
   - Write unit tests for data components

2. **Model Component Development**:
   - Implement embedding layers following specifications
   - Create transformer block with residue/pair updates
   - Develop IPA module placeholder (V1)
   - Test each component independently

3. **Integration Phase**:
   - Assemble full RNA folding model
   - Implement and test loss functions
   - Create training loop and evaluation metrics
   - Verify end-to-end data flow

4. **Testing and Validation**:
   - Build comprehensive test suite
   - Verify with validation data
   - Conduct performance analysis
   - Ensure Kaggle compatibility

5. **Documentation and Refinement**:
   - Update component documentation
   - Refine configuration system
   - Optimize memory usage and performance
   - Prepare submission template

## References

- **Project Documentation**: See `docs/` directory for detailed specifications
- **Feature Format Reference**: `docs/claude/reference/feature_formats.md`
- **PyTorch Patterns**: `docs/claude/reference/pytorch_patterns.md`
- **Component Guides**: `docs/claude/components/` directory
- **Claude Code Instances**: `docs/claude/code-instances/` directory

## Development Environment

- Install dependencies: `conda env create -f environment.yml`
- Run tests: `python -m pytest tests/`
- Lint code: `black src/ tests/` and `isort src/ tests/`
- Type checking: `mypy src/`

## Contributors

- RNA 2025 Team
- Claude AI Assistant (Anthropic)