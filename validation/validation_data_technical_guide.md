# Technical Guide: Scientific Features vs. Coordinates in RNA Structure Validation

## Introduction

This technical guide explains the scientific principles behind our approach to RNA structure validation, particularly focusing on why we need both scientific features (from NPZ files) and 3D coordinates (from CSV files).

## The Dual Nature of RNA Structure Prediction

RNA structure prediction is fundamentally a translation problem: converting sequence and derived features into 3D structural coordinates. This duality defines our validation approach.

## Scientific Features in Model Training and Validation

### Why We Need Preprocessed Scientific Features

RNA structure prediction relies on extracting complex patterns from sequences and their derived features:

1. **Thermodynamic Features**: Capture energy landscape information critical for understanding folding propensity
   - Base-pairing energies
   - Stacking interactions
   - Entropy considerations

2. **Pseudo-Dihedral Angles**: Represent backbone conformations that constrain possible 3D structures
   - Alpha, beta, gamma, delta, epsilon, zeta angles
   - Sugar pucker conformations
   - Chi angles for base orientation

3. **Mutual Information Matrices**: Capture evolutionary covariance and potential long-range interactions
   - Co-evolving nucleotide positions
   - Conservation patterns
   - Structural contact information

**Key Insight**: These features represent the *input* dimension of the model's translation task. They provide the rich information needed for the model to make accurate structural predictions.

### Feature Processing Consistency

A critical machine learning principle is that validation must use the same feature processing pipeline as training:

```
Training:   Raw Data → Feature Processing → Model → Predicted Coordinates
Validation: Raw Data → Feature Processing → Model → Predicted Coordinates → Compare with Truth
```

Using different feature processing between training and validation would create a domain shift, invalidating our assessment.

## Coordinate Data in Validation

### The Ground Truth Role

3D coordinates serve as the "ground truth" in our validation framework:

1. **Prediction Target**: What the model is ultimately trying to predict
2. **Evaluation Reference**: The benchmark against which we measure model performance

### Structure Comparison Metrics

The key metrics for assessing RNA structure prediction quality require actual 3D coordinates:

1. **RMSD (Root Mean Square Deviation)**:
   - Measures the average distance between predicted and actual atom positions
   - Lower values indicate better prediction (typically < 5Å for usable predictions)
   - Formula: √(1/n ∑ᵢ ||xᵢ - yᵢ||²) where x and y are coordinates

2. **TM-Score (Template Modeling Score)**:
   - Length-normalized measure of structural similarity
   - Ranges from 0 to 1, with 1 indicating perfect match
   - Less sensitive to local errors than RMSD
   - Values > 0.5 typically indicate correct overall fold

3. **Per-residue RMSD**:
   - Localized RMSD calculations for each residue
   - Identifies regions of high/low prediction accuracy
   - Useful for targeted model improvements

These metrics require actual atom coordinates for both prediction and ground truth, which explains why we need the CSV coordinate data.

## The Validation Process: Bridging Features and Coordinates

### Conceptual Workflow

```
                    ┌─────────────────┐
                    │   NPZ Features  │
                    │ (Thermodynamic, │
                    │ Dihedral, etc.) │
                    └────────┬────────┘
                             │
                             ▼
┌──────────────┐     ┌─────────────────┐     ┌───────────────┐
│  RNA Model   │     │    Predicted    │     │ CSV Reference │
│  (Forward    │────>│   Coordinates   │<────│  Coordinates  │
│   Pass)      │     │    (x, y, z)    │     │   (Ground     │
└──────────────┘     └────────┬────────┘     │    Truth)     │
                             │                └───────────────┘
                             ▼
                    ┌─────────────────┐
                    │   Evaluation    │
                    │ Metrics (RMSD,  │
                    │   TM-Score)     │
                    └─────────────────┘
```

### Scientific Rationale

1. **Feature Consistency**: Using our pre-calculated NPZ features ensures the model receives inputs consistent with its training regime.

2. **Predictive Validation**: Unlike simple classification tasks, structure prediction requires evaluating the model's ability to translate features to coordinates.

3. **Direct Comparison**: By comparing predicted coordinates against reference coordinates, we assess the model's ability to perform its core function.

4. **Interpretation**: The dual approach allows us to identify whether errors stem from feature extraction or the model's structure prediction capability.

## Validation Tiers and Data Requirements

Our tiered validation approach builds on this understanding:

1. **Tier 1 (Technical)**: Validates basic model functionality
   - Requires both features and coordinates
   - Focuses on gradient flow and basic structure prediction

2. **Tier 2 (Scientific)**: Validates scientific accuracy
   - Requires complete feature sets and high-quality reference coordinates
   - Evaluates prediction quality across multiple RNA types

3. **Tier 3 (Comprehensive)**: Validates generalization
   - Requires diverse test sets with both features and coordinates
   - Evaluates performance across RNA families and structural motifs

## Model Architecture

### Complete Architecture Overview

Our RNA 3D folding model integrates multiple specialized components in a hierarchical architecture designed to progressively refine RNA structure predictions. The model successfully demonstrates forward pass capability and gradient flow, confirming trainability.

### Architecture Components

#### 1. Input Processing and Embedding

- **Feature Processing**:
  - Sequence encoding (one-hot of AUCG)
  - Thermodynamic features (base pairing energies, stacking interactions)
  - Pseudo-dihedral angles (backbone conformations)
  - Mutual information matrices (evolutionary co-variance)

- **Embedding Layer**:
  - Projects raw features to higher-dimensional embeddings
  - Incorporates positional encoding (both absolute and relative)
  - Creates initial residue representations (R) and pair representations (P)

#### 2. Representation Update Cycles

- **Transformer Blocks**:
  - Self-attention over sequence positions
  - Pair representation updates via outer product operations
  - Multi-head attention mechanisms
  - Layer normalization and residual connections
  - Iterative refinement of internal representations

#### 3. Coordinate Prediction

- **IPA Module** (Invariant Point Attention):
  - Projects residue representations to 3D space
  - Maintains invariance to rotations and translations
  - Predicts atomic coordinates with proper equivariance
  - Current implementation: Simplified MLP projection (placeholder)

#### 4. Loss Calculation

- **FAPE Loss** (Frame-Aligned Point Error):
  - Measures coordinate prediction quality
  - Aligns predicted and ground truth coordinate frames
  - Calculates position errors in aligned reference frames

- **Auxiliary Losses**:
  - Secondary structure prediction loss
  - Confidence estimation loss
  - Multi-component weighted loss combination

### Detailed Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                INPUT LAYER                               │
│                                                                         │
│  ┌───────────────┐    ┌─────────────────┐    ┌─────────────────────┐   │
│  │ RNA Sequence  │    │  Thermodynamic  │    │ Pseudo-Dihedrals &  │   │
│  │ (AUCG tokens) │    │    Features     │    │  MI Matrices        │   │
│  └───────┬───────┘    └────────┬────────┘    └──────────┬──────────┘   │
│          │                     │                        │              │
└──────────┼─────────────────────┼────────────────────────┼──────────────┘
           │                     │                        │               
           ▼                     ▼                        ▼               
┌──────────────────────────────────────────────────────────────────────┐  
│                        EMBEDDING LAYER                                │  
│                                                                       │  
│   ┌───────────────┐   ┌────────────────┐   ┌─────────────────────┐   │  
│   │ Token Embeds  │   │  Positional    │   │ Feature Projection  │   │  
│   │ dim=(L, D)    │   │   Encoding     │   │ dim=(L, F) → (L, D) │   │  
│   └───────┬───────┘   └────────┬───────┘   └──────────┬──────────┘   │  
│           │                    │                      │               │  
│           └───────────┬────────┘                      │               │  
│                       ▼                               │               │  
│           ┌────────────────────┐                      │               │  
│           │ Residue Embedding  │◄─────────────────────┘               │  
│           │  dim=(L, D)        │                                      │  
│           └──────────┬─────────┘                                      │  
│                      │                  ┌─────────────────────────┐   │  
│                      │                  │ Outer Product + Linear  │   │  
│                      └─────────────────►│ dim=(L, D) → (L, L, P)  │   │  
│                                         └────────────┬────────────┘   │  
│                                                      │                │  
│                                          ┌───────────▼────────────┐   │  
│                                          │  Pair Representation   │   │  
│                                          │  dim=(L, L, P)         │   │  
│                                          └───────────┬────────────┘   │  
└──────────────────────────────────────────────────────┼────────────────┘  
                                                       │                   
┌──────────────────────────────────────────────────────▼────────────────┐  
│                         TRANSFORMER BLOCKS                             │  
│  ┌─────────────────────────────────────────────────────────────────┐  │  
│  │                 Transformer Block (x N layers)                   │  │  
│  │                                                                  │  │  
│  │  ┌──────────────────┐    ┌────────────────────────────────────┐ │  │  
│  │  │  Sequence → Seq  │    │       Sequence → Pair Update       │ │  │  
│  │  │  Self-Attention  │    │                                    │ │  │  
│  │  │                  │    │  ┌───────────┐    ┌─────────────┐  │ │  │  
│  │  │ ┌─────────────┐ │    │  │ Attention │    │  Pairwise   │  │ │  │  
│  │  │ │  Residue R  │ │    │  │ Queries+  │    │  Features   │  │ │  │  
│  │  │ │ dim=(L, D)  │◄┼────┼──┤  Keys     │◄───┤ dim=(L,L,P) │  │ │  │  
│  │  │ └──────┬──────┘ │    │  │           │    │             │  │ │  │  
│  │  │        │        │    │  └─────┬─────┘    └─────────────┘  │ │  │  
│  │  │        ▼        │    │        │                           │ │  │  
│  │  │ ┌─────────────┐ │    │        ▼                           │ │  │  
│  │  │ │  Updated R  │ │    │  ┌───────────┐                     │ │  │  
│  │  │ │ dim=(L, D)  │ │    │  │ Attention │                     │ │  │  
│  │  │ └──────┬──────┘ │    │  │  Output   │                     │ │  │  
│  │  └────────┼────────┘    │  └─────┬─────┘                     │ │  │  
│  │           │             │        │                           │ │  │  
│  │           │             │        ▼                           │ │  │  
│  │           │             │  ┌───────────┐    ┌─────────────┐  │ │  │  
│  │           │             │  │  Updated  │    │   Current   │  │ │  │  
│  │           │             │  │ Pairwise  │◄───┤  Pairwise   │  │ │  │  
│  │           │             │  │ Features  │    │  Features   │  │ │  │  
│  │           │             │  └─────┬─────┘    └─────────────┘  │ │  │  
│  │           │             └────────┼────────────────────────────┘ │  │  
│  │           │                      │                               │  │  
│  │           └──────────────────────┼───────────────────────────────┘  │  
│  │                                  │                                  │  
│  └──────────────────────────────────┼──────────────────────────────────┘  
│                                     │                                     
│                                     ▼                                     
│  ┌─────────────────────────────────────────────────────────────────┐     
│  │                 Final Representations                            │     
│  │  ┌─────────────────────────┐     ┌─────────────────────────┐    │     
│  │  │      Final Residue      │     │       Final Pair        │    │     
│  │  │     Representations     │     │     Representations     │    │     
│  │  │       dim=(L, D)        │     │      dim=(L, L, P)      │    │     
│  │  └────────────┬────────────┘     └─────────────────────────┘    │     
│  └───────────────┼───────────────────────────────────────────────────┘  
└────────────────────┼───────────────────────────────────────────────────┘  
                     │                                                      
                     ▼                                                      
┌────────────────────────────────────────────────────────────────────────┐ 
│                          IPA MODULE                                     │ 
│  ┌─────────────────────────────────────────────────────────────────┐   │ 
│  │                    Coordinate Generation                         │   │ 
│  │                                                                  │   │ 
│  │  ┌─────────────────┐     ┌─────────────────┐      ┌───────────┐ │   │ 
│  │  │ Initial Frames  │     │  Frame Updates  │      │  Final    │ │   │ 
│  │  │    Creation     │────►│  via Attention  │─────►│  Atomic   │ │   │ 
│  │  │                 │     │                 │      │ Positions │ │   │ 
│  │  └─────────────────┘     └─────────────────┘      └─────┬─────┘ │   │ 
│  │                                                          │       │   │ 
│  └──────────────────────────────────────────────────────────┼───────┘   │ 
│                                                             │           │ 
└─────────────────────────────────────────────────────────────┼───────────┘ 
                                                              │             
                                                              ▼             
┌─────────────────────────────────────────────────────────────────────────┐
│                              LOSS CALCULATION                            │
│                                                                         │
│   ┌───────────────────┐  ┌───────────────────┐  ┌────────────────────┐ │
│   │  Predicted 3D     │  │  Ground Truth     │  │     Secondary      │ │
│   │   Coordinates     │  │   Coordinates     │  │     Structure      │ │
│   │  dim=(L, 3)       │  │  dim=(L, 3)       │  │     Features       │ │
│   └─────────┬─────────┘  └──────────┬────────┘  └──────────┬─────────┘ │
│             │                       │                      │           │
│             └───────────┬───────────┘                      │           │
│                         │                                  │           │
│                         ▼                                  │           │
│             ┌─────────────────────┐                        │           │
│             │     FAPE Loss       │                        │           │
│             │                     │                        │           │
│             └───────────┬─────────┘                        │           │
│                         │                                  │           │
│                         └────────────────┬─────────────────┘           │
│                                          │                             │
│                                          ▼                             │
│                              ┌─────────────────────┐                   │
│                              │  Total Loss + Backprop  │                   │
│                              └─────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Architecture Implementation Status

We have successfully implemented and verified:

1. **Forward Pass**: Complete data flow from input features through all architectural components to predicted 3D coordinates.

2. **Gradient Flow**: Confirmed proper gradient propagation through the entire model during backpropagation.

3. **Loss Calculation**: Implemented and validated structure-aware loss functions for coordinate prediction.

Current limitations:

1. The IPA module is currently a simplified placeholder that will be enhanced with full invariant point attention mechanisms in future iterations.

2. The training loop is not yet fully implemented, though gradient flow verification confirms trainability.

### Computational Profile

- **Model Size**: ~10-20M parameters
- **Memory Requirements**: ~2-4GB GPU memory for batch size 16 (sequence length dependent)
- **Input Dimensions**:
  - Sequence: (batch_size, seq_length)
  - Features: (batch_size, seq_length, feature_dim)
  - MI Matrices: (batch_size, seq_length, seq_length)
- **Output Dimensions**:
  - Coordinates: (batch_size, seq_length, 3) for backbone atoms

## Conclusion

The dual data requirements in our validation framework reflect the fundamental nature of RNA structure prediction as a translation from sequence features to 3D coordinates. By loading both scientific features from NPZ files and reference coordinates from CSV files, we create a validation environment that accurately assesses the model's true capabilities while maintaining consistency with the training regime.

Our model architecture successfully demonstrates forward pass capability and gradient flow, confirming that it is properly designed for the challenging task of RNA 3D structure prediction. The hierarchical processing from feature embedding through transformer-based representation refinement to coordinate prediction provides a solid foundation for our structure prediction goals.