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

## Conclusion

The dual data requirements in our validation framework reflect the fundamental nature of RNA structure prediction as a translation from sequence features to 3D coordinates. By loading both scientific features from NPZ files and reference coordinates from CSV files, we create a validation environment that accurately assesses the model's true capabilities while maintaining consistency with the training regime.