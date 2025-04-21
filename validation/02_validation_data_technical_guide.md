# Dual-Mode Validation Framework Technical Guide

## Introduction

This technical guide documents the implementation approach for the dual-mode validation framework that addresses the critical challenge of feature availability mismatch between training and testing environments. It provides detailed information on the implementation design, code structures, and integration plans.

## Feature Availability Challenge

In the RNA 3D folding competition, there's a significant difference between features available during training versus testing:

| Feature Type | Training Availability | Testing Availability |
|--------------|----------------------|---------------------|
| Thermodynamic features | ✅ Available | ✅ Available |
| Mutual information matrices | ✅ Available | ✅ Available |
| Pseudo-dihedral angles | ✅ Available | ❌ Not available |

This discrepancy creates a key challenge: models trained with all features may perform worse in the competition where dihedrals are not available. Our dual-mode validation framework addresses this by validating in both modes to quantify the impact.

## Dual-Mode Validation Approach

The dual-mode validation framework uses a two-pronged approach:

1. **Test-Equivalent Mode**: Uses only features available at competition test time (thermodynamic features and mutual information matrices)
2. **Training-Equivalent Mode**: Uses all features available during training (including pseudo-dihedral angles)

By comparing performance across these modes, we can:
- Quantify the performance gap due to feature availability differences
- Identify which RNA structures are most affected
- Guide model design to minimize the impact of missing features

## Core Components

### 1. NPZ Feature Loader

The `NPZFeatureLoader` handles loading scientific features from NPZ files with test-mode filtering.

**Key Features:**
- Robust path resolution using multiple strategies
- Support for feature subdirectories (`thermo_features`, `mi_features`, `dihedral_features`)
- Test/train mode switching to control feature availability
- Graceful handling of missing features with fallbacks
- Support for sequence priority selection

**Implementation:**
```python
def get_features(self, target_id):
    """
    Get features for a target, with test mode filtering.
    
    Args:
        target_id: Target ID string
        
    Returns:
        Dictionary with loaded features
    """
    features = {}
    
    # Always load sequence info and thermodynamic features
    sequence_info = self._load_sequence_info(target_id)
    if sequence_info:
        features.update(sequence_info)
    
    thermo_features = self._load_thermo_features(target_id)
    if thermo_features:
        features.update(thermo_features)
    
    # Always load MI features
    mi_features = self._load_mi_features(target_id)
    if mi_features:
        features.update(mi_features)
    
    # Only load dihedral features in train mode
    if not self.test_mode:
        dihedral_features = self._load_dihedral_features(target_id)
        if dihedral_features:
            features.update(dihedral_features)
    
    return features
```

### 2. CSV Coordinate Loader

The `CSVCoordinateLoader` provides 3D coordinate loading from CSV files.

**Key Features:**
- Extracts atom positions and masks
- Handles multiple coordinate sets
- Provides sequence information
- Path agnostic using robust path resolution

### 3. Validation Dataset

The `ValidationDataset` is a PyTorch Dataset that combines features and coordinates with dual-mode support.

**Key Features:**
- Handles both test and train modes
- Works with different validation subsets (technical, scientific, comprehensive)
- Provides custom batch collation with padding
- Ensures consistent key naming (e.g., `dihedral_angles` → `dihedral_features`)
- Provides fallbacks for missing features

**Implementation:**
```python
def __getitem__(self, idx):
    """
    Get a dataset sample with test/train mode feature filtering.
    """
    target_id = self.target_ids[idx]
    
    # Get features (filtered by test_mode flag)
    features = self.feature_loader.get_features(target_id)
    
    # Get coordinates
    coordinates = self.coord_loader.get_coordinates(target_id)
    
    # Create sample with required fields for the model
    sample = {
        "target_id": target_id,
        "sequence": features["sequence"],
        "sequence_int": features["sequence_int"],
        "length": features["length"],
        "mask": features["mask"],
        "atom_positions": coordinates["atom_positions"],
        "atom_mask": coordinates["atom_mask"],
        
        # Add pairing and entropy features (always available)
        "pairing_probs": features["pairing_probs"],
        "positional_entropy": features["positional_entropy"],
        "coupling_matrix": features["coupling_matrix"],
    }
    
    # Handle dihedral features (rename to match model's expected key name)
    if "dihedral_angles" in features:
        # Rename dihedral_angles to dihedral_features to match model expectations
        sample["dihedral_features"] = features["dihedral_angles"]
        
    # Add accessibility if available or create placeholder
    if "accessibility" in features:
        sample["accessibility"] = features["accessibility"]
    else:
        # Create zero tensor for accessibility
        seq_len = len(features["sequence"])
        sample["accessibility"] = torch.zeros(seq_len, dtype=torch.float32)
    
    return sample
```

### 4. Validation Runner

The `ValidationRunner` executes validation in dual mode and analyzes the differences.

**Key Features:**
- Runs validation in test-equivalent and training-equivalent modes
- Computes structure metrics (RMSD, TM-score, per-residue RMSD)
- Analyzes performance differences between modes
- Generates visualizations comparing modes
- Saves results and reports

**Implementation:**
```python
def run_validation(self, subset_name="technical", run_both_modes=True):
    """
    Run validation in one or both modes.
    """
    results = {}
    
    # Always run test-equivalent mode
    logger.info("Running TEST-EQUIVALENT mode validation")
    test_results = self.run_test_equivalent_mode(subset_name)
    results["test_mode"] = test_results
    
    # Optionally run training-equivalent mode
    if run_both_modes:
        logger.info("Running TRAINING-EQUIVALENT mode validation")
        train_results = self.run_training_equivalent_mode(subset_name)
        results["train_mode"] = train_results
        
        # Compare modes and analyze difference
        logger.info("Analyzing performance differences")
        analysis = self.analyze_mode_differences(test_results, train_results)
        results["analysis"] = analysis
    
    return results
```

## Tiered Validation Approach

Our validation framework is organized in three tiers of increasing complexity:

### Tier 1: Technical Validation
- Fast, technical verification (running time: <5 minutes)
- Small subset of data (3-5 sequences)
- Focuses on shape checks, gradient flow, basic metrics
- Current notebook needs updating to use ValidationRunner

### Tier 2: Scientific Validation
- Mid-sized subset (10-15 sequences)
- Focuses on RNA-specific structural quality
- Detailed analysis of prediction errors
- In planning stage

### Tier 3: Comprehensive Validation
- Large subset with diverse RNA types
- Focuses on generalization and edge cases
- Tests Kaggle submission performance
- In planning stage

## Integration Plan for Tier 1 Notebook

The current Tier 1 validation notebook needs to be updated to use our new dual-mode framework. Here's the integration plan:

### 1. Import New Components

Add imports for ValidationRunner and related components:

```python
# Import dual-mode validation components
from validation.validation_runner import ValidationRunner
from validation.validation_dataset import ValidationDataset
from validation.npz_feature_loader import NPZFeatureLoader
from validation.csv_coordinate_loader import CSVCoordinateLoader
```

### 2. Replace Direct Data Loading

Replace the current CSV data loading with ValidationRunner:

```python
# Create ValidationRunner with the model
runner = ValidationRunner(
    model=model,
    data_dir=os.path.join(project_root, "data"),
    config={
        "batch_size": CONFIG["batch_size"],
        "results_dir": CONFIG["results_dir"],
        "seed": CONFIG["random_seed"]
    }
)

# Run validation in both modes
results = runner.run_validation("technical", run_both_modes=True)
```

### 3. Add Mode Comparison Section

Add a new section to compare test and train modes:

```python
# Display mode comparison
def display_mode_comparison(results):
    """Display comparison between test and train modes."""
    if "test_mode" in results and "train_mode" in results:
        test_rmsd = results["test_mode"].get("mean_rmsd")
        train_rmsd = results["train_mode"].get("mean_rmsd")
        
        print("\nFeature Availability Impact:")
        print(f"  Test-Equivalent RMSD: {test_rmsd:.4f} Å")
        print(f"  Training-Equivalent RMSD: {train_rmsd:.4f} Å")
        
        if train_rmsd > 0:
            diff = test_rmsd - train_rmsd
            diff_pct = (diff / train_rmsd) * 100
            print(f"  Performance Gap: {diff:.4f} Å ({diff_pct:.1f}%)")
        
        # Show analysis conclusion
        if "analysis" in results and "conclusion" in results["analysis"]:
            conclusion = results["analysis"]["conclusion"]
            print(f"\nImpact Analysis:")
            print(f"  Overall Impact: {conclusion['overall_impact']}")
            print(f"  Severity: {conclusion['severity']}")
            print(f"  Recommendation: {conclusion['recommendation']}")
```

### 4. Modify Visualization Code

Update visualization code to show comparisons between modes:

```python
# Show mode comparison visualization
if "avg_per_residue_error" in results["test_mode"] and "avg_per_residue_error" in results["train_mode"]:
    plt.figure(figsize=(12, 6))
    
    test_errors = np.array(results["test_mode"]["avg_per_residue_error"])
    train_errors = np.array(results["train_mode"]["avg_per_residue_error"])
    
    plt.plot(test_errors[:, 0], test_errors[:, 1], 'r-', label='Test-Equivalent Mode')
    plt.plot(train_errors[:, 0], train_errors[:, 1], 'b-', label='Training-Equivalent Mode')
    
    plt.xlabel('Normalized Position')
    plt.ylabel('RMSD (Å)')
    plt.title('Impact of Feature Availability on Structure Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
```

## Proposed Changes and Enhancements

### 1. Key Name Standardization

Currently, there's a mismatch between key names used in the loader (`dihedral_angles`) and what the model expects (`dihedral_features`). We've addressed this in the ValidationDataset:

```python
# Handle dihedral features (rename to match model's expected key name)
if "dihedral_angles" in features:
    # Rename dihedral_angles to dihedral_features to match model expectations
    sample["dihedral_features"] = features["dihedral_angles"]
```

### 2. Path Resolution Enhancement

We've enhanced the path resolution to handle different directory structures:

```python
def _find_data_dir(self):
    """Find the data directory using multiple strategies with path objects."""
    # Get script directory and convert to Path for better manipulation
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Try common locations relative to script path and working directory
    possible_dirs = [
        project_root / "data",             # Most likely location
        script_dir / "data",               # In validation directory
        Path.cwd() / "data",               # Current working directory
        Path.cwd().parent / "data",        # Parent of current directory
        Path.cwd().parent.parent / "data", # Grandparent of current directory
    ]
    
    for dir_path in possible_dirs:
        if dir_path.exists():
            return str(dir_path)
```

### 3. Feature Fallbacks

We've implemented fallbacks for missing features to make the validation more robust:

```python
# Add accessibility if available or create placeholder
if "accessibility" in features:
    sample["accessibility"] = features["accessibility"]
else:
    # Create zero tensor for accessibility
    seq_len = len(features["sequence"])
    sample["accessibility"] = torch.zeros(seq_len, dtype=torch.float32)
```

## Conclusion

The dual-mode validation framework provides a comprehensive solution to the feature availability mismatch problem. By validating in both test-equivalent and training-equivalent modes, we can quantify the impact of missing features and guide model development to minimize this impact.

Most core components are now implemented and functional, with the immediate next step being integration into the Tier 1 notebook. Following that, implementation of Tier 2 and 3 notebooks will provide deeper insights into model performance across different RNA types and structures.

This framework will ultimately help develop a model architecture that performs well despite the feature availability constraints of the competition testing environment.