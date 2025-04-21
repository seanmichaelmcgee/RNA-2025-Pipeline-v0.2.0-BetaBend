# Dual-Mode Validation Framework Implementation Plan

## Overview

This document outlines a concrete plan for implementing the Dual-Mode Validation Framework in our RNA 3D folding model. This approach addresses the critical challenge of feature availability mismatch between training and testing environments.

## Core Implementation Components

### 1. Feature Loaders (2 days)

#### NPZFeatureLoader Implementation

```python
# validation/feature_loaders.py

class NPZFeatureLoader:
    """
    Loads scientific features from NPZ files with test/train mode filtering.
    
    In test mode, only loads features available at competition test time:
    - Thermodynamic features
    - Mutual information matrices
    
    In train mode, loads all available features:
    - Thermodynamic features
    - Mutual information matrices
    - Pseudo-dihedral angles
    """
    
    def __init__(self, data_dir, test_mode=True):
        """
        Initialize feature loader.
        
        Args:
            data_dir: Base directory for data
            test_mode: If True, excludes pseudo-dihedral features to match test conditions
        """
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Verify directories exist
        if not os.path.exists(self.processed_dir):
            print(f"Warning: Processed directory not found at {self.processed_dir}")
            
    def get_features(self, target_id):
        """Get features for a target, with test mode filtering."""
        # Initialize features dictionary
        features = {}
        
        # Load sequence (always needed)
        sequence_data = self._load_sequence(target_id)
        if sequence_data:
            features.update(sequence_data)
        
        # Load thermodynamic features (always needed)
        thermo_features = self._load_thermo_features(target_id)
        if thermo_features:
            features.update(thermo_features)
        else:
            print(f"Warning: No thermodynamic features found for {target_id}")
        
        # Load MI features (always needed)
        mi_features = self._load_mi_features(target_id)
        if mi_features:
            features.update(mi_features)
        else:
            print(f"Warning: No MI features found for {target_id}")
        
        # Load dihedral features (only in train mode)
        if not self.test_mode:
            dihedral_features = self._load_dihedral_features(target_id)
            if dihedral_features:
                features.update(dihedral_features)
            else:
                print(f"Warning: No dihedral features found for {target_id}")
        
        return features
    
    def _load_sequence(self, target_id):
        """Load sequence data."""
        # Implementation detail: extract from NPZ or other source
        pass
    
    def _load_thermo_features(self, target_id):
        """Load thermodynamic features from NPZ file."""
        thermo_path = os.path.join(self.processed_dir, f"{target_id}_thermo_features.npz")
        if not os.path.exists(thermo_path):
            return None
            
        try:
            data = np.load(thermo_path, allow_pickle=True)
            
            return {
                "pairing_probs": torch.tensor(data["pairing_probs"], dtype=torch.float32),
                "positional_entropy": torch.tensor(data["positional_entropy"], dtype=torch.float32),
                # Add other thermodynamic features
            }
        except Exception as e:
            print(f"Error loading thermo features for {target_id}: {e}")
            return None
    
    def _load_mi_features(self, target_id):
        """Load mutual information features from NPZ file."""
        mi_path = os.path.join(self.processed_dir, f"{target_id}_mi_features.npz")
        if not os.path.exists(mi_path):
            return None
            
        try:
            data = np.load(mi_path, allow_pickle=True)
            
            return {
                "coupling_matrix": torch.tensor(data["coupling_matrix"], dtype=torch.float32),
                # Add other MI features
            }
        except Exception as e:
            print(f"Error loading MI features for {target_id}: {e}")
            return None
    
    def _load_dihedral_features(self, target_id):
        """Load pseudo-dihedral features from NPZ file."""
        dihedral_path = os.path.join(self.processed_dir, f"{target_id}_dihedral_features.npz")
        if not os.path.exists(dihedral_path):
            return None
            
        try:
            data = np.load(dihedral_path, allow_pickle=True)
            
            return {
                "dihedral_features": torch.tensor(data["dihedral_features"], dtype=torch.float32),
                "dihedral_angles": torch.tensor(data["dihedral_angles"], dtype=torch.float32),
                # Add other dihedral features
            }
        except Exception as e:
            print(f"Error loading dihedral features for {target_id}: {e}")
            return None
```

#### CSVCoordinateLoader Refactoring

```python
# validation/coordinate_loaders.py

class CSVCoordinateLoader:
    """Loader specifically for 3D coordinates from CSV files."""
    
    def __init__(self, data_dir=None, split="validation"):
        """
        Initialize coordinate loader.
        
        Args:
            data_dir: Path to the data directory
            split: Data split to use ("train", "validation", or "test")
        """
        # Setup logic similar to current CSVDataLoader
        # but focused only on coordinate loading
        pass
        
    def get_coordinates(self, target_id):
        """
        Get coordinates for a specific target ID.
        
        Returns:
            Dictionary with atom positions and masks
        """
        # Simplified from current CSVDataLoader.get_coordinates
        # Focus only on extracting and returning coordinates
        pass
```

### 2. Dual-Mode Validation Dataset (1 day)

```python
# validation/datasets.py

class ValidationDataset(Dataset):
    """
    Validation dataset with dual-mode support (test vs. train mode).
    
    Combines features from NPZ files with coordinates from CSV files,
    with feature selection based on mode.
    """
    
    def __init__(self, data_dir, subset_name="technical", test_mode=True, seed=42):
        """
        Initialize validation dataset.
        
        Args:
            data_dir: Base directory for data
            subset_name: Validation subset to use ("technical", "scientific", "comprehensive")
            test_mode: If True, excludes features not available during testing
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.subset_name = subset_name
        self.test_mode = test_mode
        
        # Initialize loaders
        self.coord_loader = CSVCoordinateLoader(data_dir)
        self.feature_loader = NPZFeatureLoader(data_dir, test_mode=test_mode)
        
        # Load target IDs for validation subset
        self.target_ids = self._load_validation_subset(subset_name, seed)
        
        # Report initialization details
        mode_str = "TEST-EQUIVALENT" if test_mode else "TRAINING-EQUIVALENT"
        print(f"Initialized {mode_str} validation dataset with {len(self.target_ids)} targets")
        print(f"Subset: {subset_name}")
        
    def __len__(self):
        return len(self.target_ids)
    
    def __getitem__(self, idx):
        target_id = self.target_ids[idx]
        
        # Get coordinates and features
        coordinates = self.coord_loader.get_coordinates(target_id)
        features = self.feature_loader.get_features(target_id)
        
        # Combine into a single sample
        sample = {
            "target_id": target_id,
            "sequence": features.get("sequence", ""),
            "atom_positions": coordinates["atom_positions"],
            "mask": coordinates["mask"],
        }
        
        # Add all available features
        for key, value in features.items():
            if key not in sample:
                sample[key] = value
        
        return sample
    
    def _load_validation_subset(self, subset_name, seed):
        """Load validation target IDs based on subset name."""
        # Implementation details for selecting appropriate targets
        # based on subset name (technical, scientific, comprehensive)
        pass
    
    def collate_fn(self, batch):
        """Custom collation function for batching."""
        # Implementation details for proper batching with padding
        pass
```

### 3. Validation Runner (2 days)

```python
# validation/runner.py

class ValidationRunner:
    """
    Runs validation in dual mode: test-equivalent and training-equivalent.
    
    Provides comprehensive analysis of model performance under both conditions.
    """
    
    def __init__(self, model, data_dir, config):
        """
        Initialize validation runner.
        
        Args:
            model: Model to validate
            data_dir: Path to data directory
            config: Configuration dictionary
        """
        self.model = model
        self.data_dir = data_dir
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
    def run_validation(self, subset_name="technical", run_both_modes=True):
        """
        Run validation in one or both modes.
        
        Args:
            subset_name: Validation subset ("technical", "scientific", "comprehensive")
            run_both_modes: Whether to run both test and train modes
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        # Always run test-equivalent mode (for Kaggle performance estimation)
        print(f"\n{'='*50}\nRunning TEST-EQUIVALENT mode validation\n{'='*50}\n")
        test_results = self.run_test_equivalent_mode(subset_name)
        results["test_mode"] = test_results
        
        # Optionally run training-equivalent mode (with all features)
        if run_both_modes:
            print(f"\n{'='*50}\nRunning TRAINING-EQUIVALENT mode validation\n{'='*50}\n")
            train_results = self.run_training_equivalent_mode(subset_name)
            results["train_mode"] = train_results
            
            # Compare modes and analyze difference
            print(f"\n{'='*50}\nAnalyzing performance differences\n{'='*50}\n")
            analysis = self.analyze_mode_differences(test_results, train_results)
            results["analysis"] = analysis
        
        return results
    
    def run_test_equivalent_mode(self, subset_name):
        """
        Run validation using only test-available features.
        
        Args:
            subset_name: Validation subset name
            
        Returns:
            Dictionary with test-mode validation results
        """
        # Create test-mode dataset (no pseudo-dihedrals)
        dataset = ValidationDataset(
            data_dir=self.data_dir,
            subset_name=subset_name,
            test_mode=True
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self._calculate_batch_size(dataset),
            collate_fn=dataset.collate_fn,
            shuffle=False
        )
        
        # Run evaluation
        return self._evaluate_model(dataloader, "test_equivalent")
    
    def run_training_equivalent_mode(self, subset_name):
        """
        Run validation using all training features (including pseudo-dihedrals).
        
        Args:
            subset_name: Validation subset name
            
        Returns:
            Dictionary with train-mode validation results
        """
        # Create train-mode dataset (with pseudo-dihedrals)
        dataset = ValidationDataset(
            data_dir=self.data_dir,
            subset_name=subset_name,
            test_mode=False
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self._calculate_batch_size(dataset),
            collate_fn=dataset.collate_fn,
            shuffle=False
        )
        
        # Run evaluation
        return self._evaluate_model(dataloader, "training_equivalent")
    
    def _calculate_batch_size(self, dataset):
        """Calculate appropriate batch size based on sequence lengths."""
        # Implementation details for memory-efficient batch sizing
        pass
    
    def _evaluate_model(self, dataloader, mode_name):
        """
        Evaluate model on provided dataloader.
        
        Args:
            dataloader: DataLoader with validation samples
            mode_name: Mode name for logging ("test_equivalent" or "training_equivalent")
        
        Returns:
            Dictionary with evaluation metrics
        """
        # Implementation details for model evaluation
        pass
    
    def analyze_mode_differences(self, test_results, train_results):
        """
        Analyze differences between test and training modes.
        
        Calculates absolute and relative improvements, significance of differences,
        and provides scientific insights about feature importance.
        
        Args:
            test_results: Results from test-equivalent mode
            train_results: Results from training-equivalent mode
            
        Returns:
            Dictionary with analysis metrics
        """
        # Implementation details for performance comparison
        pass
```

### 4. Validation Notebooks (3 days)

#### Tier 1: Technical Validation

```python
# validation/tier1_technical/validation_technical.ipynb

# Implementation plan for the notebook:

# 1. Setup and Configuration
# - Import dependencies
# - Set up paths and configuration
# - Load model

# 2. Dual-Mode Validation
# - Create ValidationRunner
# - Run validation in both modes
# - Display and visualize results

# 3. Technical Analysis
# - Gradient flow verification
# - Model architecture verification
# - Memory usage analysis
# - Performance benchmarking

# 4. Feature Impact Analysis
# - Analyze impact of pseudo-dihedral features
# - Visualize performance differences
# - Identify model weaknesses

# 5. Reporting
# - Generate summary tables
# - Save results for tracking
# - Create visualizations
```

#### Tier 2: Scientific Validation (Coming Soon)

```python
# validation/tier2_scientific/validation_scientific.ipynb

# Implementation plan to include:
# - More comprehensive scientific evaluation
# - Larger validation set with diverse RNA types
# - Detailed structure quality assessment
# - Feature ablation studies
# - Structure visualization and comparison
```

#### Tier 3: Comprehensive Validation (Coming Soon)

```python
# validation/tier3_comprehensive/validation_comprehensive.ipynb

# Implementation plan to include:
# - Full validation across all RNA families
# - Generalization assessment
# - Edge case handling
# - Performance under varying conditions
# - Computational resource scaling
```

## Validation Subset Selection Approach

```python
# validation/subset_selection.py

def create_validation_subsets(data_dir, output_dir, temporal_cutoff="2022-05-27"):
    """
    Create validation subsets for different validation tiers.
    
    Creates three subsets:
    1. Technical (5 sequences): For quick technical validation
    2. Scientific (15 sequences): For scientific method validation
    3. Comprehensive (30 sequences): For thorough model evaluation
    
    Args:
        data_dir: Directory containing sequence and label data
        output_dir: Directory to save subset files
        temporal_cutoff: Only use sequences before this date
    """
    # Implementation details as in framework document
    pass
```

## Implementation Timeline

### Week 1: Core Implementation

| Day | Tasks | Expected Outcomes |
|-----|-------|-------------------|
| 1 | Implement NPZFeatureLoader | Working feature loader with test/train mode |
| 2 | Refactor CSVCoordinateLoader | Coordinate-focused loader with clean interface |
| 3 | Implement ValidationDataset | Dual-mode dataset with proper feature handling |
| 4 | Implement ValidationRunner (Part 1) | Base validation logic and mode handling |
| 5 | Implement ValidationRunner (Part 2) | Analysis and comparison functionality |

### Week 2: Notebook and Integration

| Day | Tasks | Expected Outcomes |
|-----|-------|-------------------|
| 1 | Update Tier 1 Notebook | Dual-mode technical validation notebook |
| 2 | Implement Subset Selection | Validation subset creation utilities |
| 3 | Create Tier 2 Notebook | Scientific validation notebook structure |
| 4 | Implement Visualizations | Performance comparison visualizations |
| 5 | Documentation and Testing | Complete documentation and test coverage |

## Critical Success Metrics

The implementation will be considered successful when:

1. **Technical Validation**
   - Model runs in both test and train modes without errors
   - Gradient flow verification succeeds in both modes
   - Memory usage is within acceptable limits

2. **Performance Measurement**
   - TM-score and RMSD are calculated for both modes
   - Performance differences are quantified (absolute and relative)
   - Visual comparison of structure quality is available

3. **Scientific Insights**
   - Impact of pseudo-dihedral features is clearly quantified
   - Areas for model improvement are identified
   - Feature importance is analyzed

4. **User Experience**
   - Notebooks run without errors
   - Results are clearly presented and visualized
   - Documentation is comprehensive and clear

## Conclusion

This implementation plan provides a clear roadmap for integrating the Dual-Mode Validation Framework into our existing validation workflow. By addressing the feature availability mismatch between training and testing, we'll gain valuable scientific insights and create a more robust model evaluation approach.

The implementation prioritizes:
1. Clean separation of test and train mode features
2. Comprehensive performance comparison
3. Clear scientific insights about feature importance
4. Streamlined user experience in validation notebooks

This approach will maximize our chances of success in the RNA 3D structure prediction competition while maintaining scientific rigor.