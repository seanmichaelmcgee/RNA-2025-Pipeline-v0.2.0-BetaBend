# RNA 3D Folding Model Validation Framework

This directory contains the tiered validation framework for the RNA 3D structure prediction model. The validation strategy uses a three-tier approach to assess model quality at different levels of detail and computational cost, with a dual-mode architecture to address the feature availability differences between training and testing environments.

## Data Sources and Structure

- **Sequences**: Target IDs are loaded from CSV files with following priority:
  1. `data/raw/validation_sequences.csv` (for validation)
  2. `data/raw/train_sequences.csv` (fallback)
  3. `data/raw/test_sequences.csv` (second fallback)
  
- **Features**: Three types of feature files in `data/processed/`:
  - `{target_id}_thermo_features.npz`: Thermodynamic features (available in both modes)
  - `{target_id}_mi_features.npz`: Mutual information features (available in both modes)
  - `{target_id}_dihedral_features.npz`: Dihedral angles (only available in train mode)

### Coordinate Data Format
- **Source**: `data/raw/validation_labels.csv`
- **ID Format**: `{target_id}_{residue_position}` (e.g., "R1107_1", "R1107_2")
- **Multiple Coordinate Sets**: Each residue has up to 40 sets of (x,y,z) coordinates
  - Columns: x_1,y_1,z_1, x_2,y_2,z_2, ..., x_40,y_40,z_40
  - Missing coordinate values use -1e+18 as placeholder
  - The validation framework automatically selects the most complete coordinate set

### Dual-Mode Validation
The framework implements a comprehensive dual-mode validation approach:
- **Test-Equivalent Mode**: Uses only features available at test time (thermo + MI)
- **Training-Equivalent Mode**: Uses all available features (thermo + MI + dihedral)

This sophisticated approach:
1. Quantifies the performance gap caused by feature availability differences
2. Provides scientific insights to guide model architecture decisions
3. Offers accurate estimates of expected Kaggle leaderboard performance
4. Identifies which RNA structures and families are most impacted by missing features
5. Guides the development of feature prediction strategies to bridge the gap

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

### Running Validation

You can run validation in both modes (test and train) using the dual-mode validation script:

#### Using the Dual-Mode Validation Script
```bash
# Navigate to the validation directory
cd validation

# Run with default settings (technical tier)
./run_dual_mode_validation.sh

# Validate specific RNA IDs
./run_dual_mode_validation.sh --rna-ids R1107 R1108

# Run scientific validation with a specific checkpoint
./run_dual_mode_validation.sh --subset scientific --checkpoint /path/to/checkpoint.pt

# Force CPU usage
./run_dual_mode_validation.sh --cpu

# Get help on available options
./run_dual_mode_validation.sh --help
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

| Component | Status | Description |
|-----------|--------|------------|
| Core Framework | ✅ Complete | ValidationRunner, ValidationDataset, NPZFeatureLoader, CSVCoordinateLoader |
| Structure Metrics | ✅ Complete | RMSD with Kabsch alignment, TM-score, per-residue RMSD |
| Tier 1: Technical | ✅ Complete | Technical validation notebook, run scripts, visualization, diagnostics |
| Tier 2: Scientific | ⏳ In Progress | Scientific validation notebook, RNA family analysis, feature importance analysis |
| RNA Family Analysis | ⏳ In Progress | Classification, grouping, and analysis of different RNA families |
| Secondary Structure | ⏳ In Progress | Base-pairing prediction accuracy, secondary structure metrics |
| Feature Importance | ⏳ In Progress | Feature ablation study, importance quantification, visualization |
| Tier 3: Comprehensive | 🔄 Planned | Directory structure created, implementation pending |

### Recent Updates (2025-04-21)
- Fixed extreme RMSD values caused by numerical instability
- Added proper coordinate validation and error handling
- Enhanced reporting with problematic sample tracking
- Added scientific metrics modules for RNA family, secondary structure, and feature importance
- Updated documentation and visualization components

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

### Core Components

- **NPZFeatureLoader:** Loads features from NPZ files with test-mode filtering
- **CSVCoordinateLoader:** Handles residue-level coordinate loading with multiple coordinate sets
- **ValidationDataset:** Combines features and coordinates for dual-mode validation
- **ValidationRunner:** Executes validation in both modes and analyzes differences

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