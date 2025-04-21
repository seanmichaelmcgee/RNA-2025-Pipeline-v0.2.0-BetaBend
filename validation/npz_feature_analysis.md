# NPZ Feature Analysis for Dual-Mode Validation

## Overview

This analysis examines the structure of our NPZ feature files and provides a comprehensive strategy for implementing the NPZFeatureLoader in our dual-mode validation framework. Understanding the NPZ file structure is critical for properly implementing test-mode vs. train-mode feature filtering.

## NPZ File Structure Analysis

Our data is organized into three distinct NPZ files per RNA target:

### 1. Mutual Information Features (`[target_id]_mi_features.npz`)

```
- coupling_matrix.npy         # (L, L) matrix of mutual information between positions
- method.npy                  # Method used to calculate MI ("mutual_information")
- top_pairs.npy               # List of top pairs with high MI scores
- score_distance_correlation.npy # Correlation between MI scores and distances
```

Key observations:
- The `coupling_matrix.npy` contains a square matrix (L×L) of pairwise mutual information values
- This matrix is important for capturing long-range interactions in RNA sequences
- **Available in both test and train modes**

### 2. Dihedral Features (`[target_id]_dihedral_features.npz`)

```
- features.npy                # (L, 4) matrix of sin/cos encoded dihedral angles
- eta.npy                     # Raw eta angles in degrees
- theta.npy                   # Raw theta angles in degrees
- feature_names.npy           # Names of features ("eta_sin", "eta_cos", "theta_sin", "theta_cos")
- metadata.npy                # Metadata about feature extraction
```

Key observations:
- The `features.npy` contains sine/cosine encoded pseudo-dihedral angles
- These are critical for capturing backbone geometry
- Feature dimension is (L, 4) with columns representing [eta_sin, eta_cos, theta_sin, theta_cos]
- **Only available in train mode (NOT in test mode)**

### 3. Thermodynamic Features (`[target_id]_thermo_features.npz`)

```
- mfe.npy                     # Minimum free energy
- ensemble_energy.npy         # Ensemble energy
- energy_gap.npy              # Energy gap between MFE and ensemble
- gc_content.npy              # GC content of sequence
- positional_entropy.npy      # (L,) array of per-position entropy values
- pairing_probs.npy           # (L, L) matrix of base pairing probabilities
- structure.npy               # Dot-bracket notation of secondary structure
- sequence.npy                # RNA sequence (AUCG)
- target_id.npy               # Target ID string
```

Key observations:
- Contains both global features (mfe, gc_content) and position-specific features (positional_entropy)
- The `pairing_probs.npy` matrix is particularly important for secondary structure prediction
- The `sequence.npy` provides the actual RNA sequence, eliminating the need to load it separately
- **Available in both test and train modes**

## NPZ File Availability Assessment

| File Type | Available in Test Mode | Available in Train Mode | Critical for Model |
|-----------|------------------------|-------------------------|-------------------|
| MI Features | ✅ Yes | ✅ Yes | ✅ High |
| Thermo Features | ✅ Yes | ✅ Yes | ✅ High |
| Dihedral Features | ❌ No | ✅ Yes | ✅ High |

## Edge Cases and Multiple Coordinate Sets

An important consideration is that "multiple coordinate sets are sometimes available for validation." This implies:

1. Some RNA structures may have multiple conformations or models
2. We need to handle this properly when loading coordinates from CSV files
3. This affects how we match coordinates with features in our dual-mode validation

## NPZFeatureLoader Implementation Strategy

Based on this analysis, here's a comprehensive strategy for implementing the NPZFeatureLoader:

### 1. File Organization Design

```python
class NPZFeatureLoader:
    def __init__(self, data_dir, test_mode=True):
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.processed_dir = os.path.join(data_dir, "processed")
        
    def get_features(self, target_id):
        """Get features for a target, with test mode filtering."""
        features = {}
        
        # Always load sequence (from thermo features)
        sequence_info = self._load_sequence_info(target_id)
        if sequence_info:
            features.update(sequence_info)
        else:
            raise ValueError(f"No sequence information found for {target_id}")
            
        # Always load thermodynamic features (available in both modes)
        thermo_features = self._load_thermo_features(target_id)
        if thermo_features:
            features.update(thermo_features)
        else:
            raise ValueError(f"Thermodynamic features missing for {target_id}")
            
        # Always load MI features (available in both modes)
        mi_features = self._load_mi_features(target_id)
        if mi_features:
            features.update(mi_features)
        else:
            print(f"Warning: MI features missing for {target_id}, using zeros")
            # Create zero-filled MI matrix as fallback
            sequence_length = len(features["sequence"])
            features["coupling_matrix"] = torch.zeros((sequence_length, sequence_length))
            
        # Only load dihedral features in train mode
        if not self.test_mode:
            dihedral_features = self._load_dihedral_features(target_id)
            if dihedral_features:
                features.update(dihedral_features)
            else:
                print(f"Warning: Dihedral features missing for {target_id} in train mode")
                # No fallback for dihedral features in train mode - model should handle missing
        
        return features
```

### 2. Feature Loading Functions

```python
def _load_sequence_info(self, target_id):
    """Load sequence from thermo features (available in both modes)."""
    thermo_path = os.path.join(self.processed_dir, f"{target_id}_thermo_features.npz")
    if not os.path.exists(thermo_path):
        return None
        
    try:
        with np.load(thermo_path, allow_pickle=True) as data:
            sequence = str(data["sequence"])
            
            # Create sequence integer mapping (AUCG -> 0123)
            seq_map = {'A': 0, 'U': 1, 'C': 2, 'G': 3, 'N': 4}
            sequence_int = torch.tensor([seq_map.get(base, 4) for base in sequence], dtype=torch.long)
            
            return {
                "target_id": str(data["target_id"]),
                "sequence": sequence,
                "sequence_int": sequence_int,
                "length": len(sequence),
                "mask": torch.ones(len(sequence), dtype=torch.bool)
            }
    except Exception as e:
        print(f"Error loading sequence for {target_id}: {e}")
        return None
        
def _load_thermo_features(self, target_id):
    """Load thermodynamic features (available in both modes)."""
    thermo_path = os.path.join(self.processed_dir, f"{target_id}_thermo_features.npz")
    if not os.path.exists(thermo_path):
        return None
        
    try:
        with np.load(thermo_path, allow_pickle=True) as data:
            sequence_length = int(data["length"])
            
            # Convert all features to PyTorch tensors
            features = {
                # Global features
                "mfe": torch.tensor(float(data["mfe"]), dtype=torch.float32),
                "gc_content": torch.tensor(float(data["gc_content"]), dtype=torch.float32),
                
                # Per-position features
                "positional_entropy": torch.tensor(data["positional_entropy"], dtype=torch.float32),
                
                # Matrix features
                "pairing_probs": torch.tensor(data["pairing_probs"], dtype=torch.float32),
            }
            
            # Add structure information if needed
            if "structure" in data:
                features["structure"] = str(data["structure"])
                
            return features
    except Exception as e:
        print(f"Error loading thermo features for {target_id}: {e}")
        return None
        
def _load_mi_features(self, target_id):
    """Load mutual information features (available in both modes)."""
    mi_path = os.path.join(self.processed_dir, f"{target_id}_mi_features.npz")
    if not os.path.exists(mi_path):
        return None
        
    try:
        with np.load(mi_path, allow_pickle=True) as data:
            features = {
                "coupling_matrix": torch.tensor(data["coupling_matrix"], dtype=torch.float32),
            }
            
            # Add method information if needed for debugging
            if "method" in data:
                features["mi_method"] = str(data["method"])
                
            return features
    except Exception as e:
        print(f"Error loading MI features for {target_id}: {e}")
        return None
        
def _load_dihedral_features(self, target_id):
    """Load dihedral features (only available in train mode)."""
    # Skip completely if in test mode
    if self.test_mode:
        return None
        
    dihedral_path = os.path.join(self.processed_dir, f"{target_id}_dihedral_features.npz")
    if not os.path.exists(dihedral_path):
        return None
        
    try:
        with np.load(dihedral_path, allow_pickle=True) as data:
            # Extract 4D dihedral features (sin/cos encoded)
            dihedral_angles = torch.tensor(data["features"], dtype=torch.float32)
            
            # We need to expand these to the 39D feature space expected by the model
            # This must match whatever was done during training preprocessing
            sequence_length = dihedral_angles.shape[0]
            dihedral_features = torch.zeros((sequence_length, 39), dtype=torch.float32)
            
            # Place the 4 dihedral features at the beginning
            dihedral_features[:, :4] = dihedral_angles
            
            # Return both the raw angles and the expanded feature vector
            return {
                "dihedral_angles": dihedral_angles,  # Original (L, 4) format
                "dihedral_features": dihedral_features  # Expanded (L, 39) format
            }
    except Exception as e:
        print(f"Error loading dihedral features for {target_id}: {e}")
        return None
```

### 3. Multiple Coordinate Set Handling

For the edge case of multiple coordinate sets in validation, we need to enhance our `CSVCoordinateLoader`:

```python
def get_coordinates(self, target_id):
    """
    Get coordinates for a specific target ID.
    
    Handles multiple coordinate sets by returning all available conformations.
    """
    if self.labels_df is None:
        return None
        
    # Filter all entries for this target
    target_prefix = f"{target_id}_"
    coords_df = self.labels_df[self.labels_df['ID'].str.startswith(target_prefix)]
    
    if coords_df.empty:
        print(f"No coordinates found for {target_id}")
        return None
    
    # Check for multiple conformations (different model/chains)
    conformations = {}
    
    # Group by model/chain identifiers if present
    if 'model' in coords_df.columns:
        model_groups = coords_df.groupby('model')
        for model_id, model_df in model_groups:
            conformation_id = f"{target_id}_model{model_id}"
            conformations[conformation_id] = self._extract_atom_positions(model_df)
    else:
        # Single conformation case
        conformations[target_id] = self._extract_atom_positions(coords_df)
    
    # Return primary conformation and any alternates
    result = {
        'target_id': target_id,
        'atom_positions': conformations[target_id]['atom_positions'],
        'atom_mask': conformations[target_id]['atom_mask'],
    }
    
    # Add alternate conformations if present
    if len(conformations) > 1:
        result['alternate_conformations'] = {
            conf_id: conf_data 
            for conf_id, conf_data in conformations.items() 
            if conf_id != target_id
        }
    
    return result
```

### 4. Sequence-Coordinate Validation

To ensure proper matching between features and coordinates:

```python
def validate_features_coordinates(features, coordinates):
    """
    Validate that features and coordinates match in sequence length.
    
    Args:
        features: Dictionary of features from NPZFeatureLoader
        coordinates: Dictionary of coordinates from CSVCoordinateLoader
        
    Returns:
        Bool: True if valid, raises ValueError otherwise
    """
    seq_length = features['length']
    coord_length = coordinates['atom_positions'].shape[0]
    
    # Check exact match first
    if seq_length == coord_length:
        return True
        
    # Check if coordinates are for a fragment
    if coord_length < seq_length:
        if 'residue_ids' in coordinates:
            # Verify residue IDs are within sequence range
            min_id = min(coordinates['residue_ids'])
            max_id = max(coordinates['residue_ids'])
            if min_id >= 1 and max_id <= seq_length:
                # Looks like a valid fragment
                return True
                
    raise ValueError(
        f"Sequence-coordinate mismatch: sequence length={seq_length}, "
        f"coordinate length={coord_length}"
    )
```

## Implementation Impact on Dual-Mode Validation

### Test-Equivalent Mode

In test-equivalent mode (test_mode=True):
- Load sequence, thermodynamic features, and MI features
- Exclude dihedral features completely
- Model receives input without pseudo-dihedral information, simulating Kaggle test conditions

### Training-Equivalent Mode

In training-equivalent mode (test_mode=False):
- Load sequence, thermodynamic features, and MI features
- Also load dihedral features 
- Model receives complete feature set available during training

### ValidationDataset Integration

```python
class ValidationDataset(Dataset):
    def __init__(self, data_dir, subset_name="technical", test_mode=True):
        """
        Initialize validation dataset with dual-mode support.
        
        Args:
            data_dir: Base directory for data
            subset_name: Which validation subset to use
            test_mode: If True, excludes features not available at test time
        """
        self.test_mode = test_mode
        
        # Initialize loaders
        self.feature_loader = NPZFeatureLoader(data_dir, test_mode=test_mode)
        self.coord_loader = CSVCoordinateLoader(data_dir)
        
        # Load target IDs for validation subset
        self.target_ids = self._load_validation_subset(subset_name)
        
        # Report initialization
        mode_str = "TEST-EQUIVALENT" if test_mode else "TRAINING-EQUIVALENT"
        print(f"Initialized {mode_str} validation dataset with {len(self.target_ids)} targets")
        
    def __getitem__(self, idx):
        """Get sample with test/train mode feature filtering."""
        target_id = self.target_ids[idx]
        
        # Get features (filtered by test_mode flag)
        features = self.feature_loader.get_features(target_id)
        
        # Get coordinates (always loaded the same way)
        coordinates = self.coord_loader.get_coordinates(target_id)
        
        # Validate feature-coordinate compatibility
        validate_features_coordinates(features, coordinates)
        
        # Combine into a single sample with all required fields
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
        
        # Only include dihedral features in train mode
        if not self.test_mode and "dihedral_features" in features:
            sample["dihedral_features"] = features["dihedral_features"]
            sample["dihedral_angles"] = features["dihedral_angles"]
        elif not self.test_mode:
            # Provide zeros if missing in train mode
            sequence_length = features["length"]
            sample["dihedral_features"] = torch.zeros((sequence_length, 39), dtype=torch.float32)
            sample["dihedral_angles"] = torch.zeros((sequence_length, 4), dtype=torch.float32)
        
        return sample
```

## Updated Implementation Timeline

Based on this detailed analysis, we should adjust our implementation timeline:

| Day | Task | Updated Details |
|-----|------|----------------|
| 1 | NPZFeatureLoader Implementation | Focus on proper file structure handling and test-mode filtering; carefully implement the feature merging with compatible shapes |
| 2 | CSVCoordinateLoader Refactoring | Add support for multiple coordinate sets and conformations; implement sequence-coordinate validation |
| 3-5 | Continue with original timeline | ValidationDataset, ValidationRunner, and Mode Analysis |

## Conclusion

This analysis provides a comprehensive understanding of our NPZ feature files and a detailed implementation strategy for the NPZFeatureLoader component of our dual-mode validation framework. The key findings are:

1. We have three types of NPZ files per target, with dihedral features only available in train mode
2. The sequence information is available in the thermo features, eliminating the need for separate sequence loading
3. Multiple coordinate sets in validation require special handling in the CSVCoordinateLoader
4. Feature dimensions need careful handling, especially expanding dihedral_angles (4D) to dihedral_features (39D)

By implementing this strategy, our dual-mode validation framework will correctly handle the feature availability mismatch between training and testing environments, providing valuable insights into the impact of pseudo-dihedral angles on model performance.