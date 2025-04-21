# RNA 3D Folding Model Validation Framework

This directory contains the tiered validation framework for the RNA 3D structure prediction model. The validation strategy uses a three-tier approach to assess model quality at different levels of detail and computational cost.

## Validation Tiers

### Tier 1: Technical Validation

**Location:** `tier1_technical/`

**Purpose:** Fast verification of model functionality and basic technical performance.

**Characteristics:**
- Quick to run (<5 minutes)
- Uses small subset of data (3-5 sequences)
- Focuses on shape checks, gradient flow, and loss values
- Checks for basic plausibility of predicted structures
- Suitable for frequent testing during development

**Implementation Status:** ✅ Complete

### Tier 2: Scientific Validation

**Location:** `tier2_scientific/`

**Purpose:** Evaluation of model scientific accuracy on challenging cases.

**Characteristics:**
- Moderate runtime (15-30 minutes)
- Uses diverse set of sequences (10-15) balanced by length and complexity
- Evaluates RMSD, TM-score, and per-residue accuracy
- Includes detailed visualization of predicted structures
- Analyzes performance correlation with sequence length
- Does not allow mock data (requires complete feature sets)
- Suitable for validating model improvements

**Implementation Status:** ✅ Complete

### Tier 3: Comprehensive Validation

**Location:** `tier3_comprehensive/`

**Purpose:** Full evaluation of model performance on entire validation set.

**Characteristics:**
- Longer runtime (1+ hours)
- Uses all available validation data (50+ sequences)
- Provides complete statistical analysis of performance by RNA family
- Evaluates base-pair accuracy and structural motif prediction
- Requires all feature types (dihedral, thermo, and MI features)
- Benchmarks against baseline approaches
- Suitable for final model evaluation before competition submission

**Implementation Status:** 🔄 Planned implementation

## Using the Validation Framework

### Running Tier 1 Validation

You can run Tier 1 validation using either the Jupyter notebook or the command-line script:

#### Using the Run Script
```bash
# Navigate to the tier1_technical directory
cd validation/tier1_technical

# Run with default settings
./run_validation.py

# Run with a specific model checkpoint
./run_validation.py --checkpoint /path/to/checkpoint.pt

# Get help on available options
./run_validation.py --help
```

#### Using the Notebook Directly
1. Activate the RNA folding environment
2. Launch Jupyter Notebook
3. Open `validation/tier1_technical/validation_technical.ipynb`
4. Update the checkpoint path if needed
5. Run all cells

### Interpreting Results

Validation results are saved to the tier-specific results directory with the following structure:

```
tier1_technical/results/
├── validation_results.json    # Complete validation metrics and results
├── validation_results.html    # HTML report of the executed notebook
├── per_residue_rmsd.png       # Visualization of per-residue RMSD
└── ...
```

## Implementation Status

| Tier | Status | Components |
|------|--------|------------|
| Tier 1: Technical | ✅ Complete | Notebook, run script, robust feature loading, results reporting |
| Tier 2: Scientific | ✅ Complete | Notebook, enhanced metrics, visualizations, sequence length analysis |
| Tier 3: Comprehensive | 🔄 Planned | Directory structure created, implementation pending |

## V1-V2 Transition Criteria

The model will be considered ready for transition from V1 to V2 when it meets these criteria:

1. Mean TM-score > 0.4 on scientific validation set
2. Per-residue confidence correlation > 0.5
3. All validation tiers complete without errors
4. Memory usage within acceptable limits for target hardware

If performance plateaus with no improvement for 3+ iterations, the transition may also be triggered.

## Validation Components

- **Shape and Configuration Check:** Verifies input/output tensor shapes and model configuration
- **Gradient Flow Verification:** Checks that gradients flow through critical model components
- **Structure Metrics:** RMSD, TM-score, per-residue RMSD
- **Performance Benchmarking:** Inference time and memory usage
- **Confidence Evaluation:** Correlation between predicted confidence and true accuracy
- **Visualization:** Per-residue error plots, 3D structure comparison
- **Resource Utilization:** Memory usage, CUDA memory tracking

## Future Enhancements

1. **Tier 3 Implementation:**
   - Comprehensive validation notebook
   - RNA family classification and analysis
   - Detailed statistical analysis by RNA type
   - Comparison with state-of-the-art models
   - Kaggle-specific metrics and validation

2. **General Improvements:**
   - Dashboard for cross-tier result visualization and comparison
   - Synthetic data generation for controlled testing
   - Enhanced 3D visualization of predicted structures with py3Dmol
   - Version tracking system for model iterations
   - Resource management for large-scale validation
   - Automatic report generation for model improvement tracking

## Documentation

For detailed information about each validation tier, see the README files in the respective directories:

- [Tier 1: Technical Validation](./tier1_technical/README.md)
- [Tier 2: Scientific Validation](./tier2_scientific/README.md)
- [Tier 3: Comprehensive Validation](./tier3_comprehensive/README.md)

For the formal handoff documentation, see:
- [Validation Framework Handoff](../docs/claude/03_code-instances/instance_03_integration/handoffs/provided/testing/validation_framework_handoff.md)