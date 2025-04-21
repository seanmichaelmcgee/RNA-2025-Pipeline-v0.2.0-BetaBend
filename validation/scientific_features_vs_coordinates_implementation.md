# Implementation Plan: Scientific Features vs. Coordinates

## Overview

This document outlines the implementation plan for integrating scientific feature data (from NPZ files) with coordinate data (from CSV files) in our RNA structure validation framework.

## Problem Statement

We currently have two separate data sources that need to be integrated for effective validation:

1. **Scientific Features (NPZ Files)**: Pre-calculated features including:
   - Thermodynamic features
   - Pseudo-dihedral angles
   - Mutual information matrices

2. **3D Coordinates (CSV Files)**: Raw spatial coordinates (x, y, z) for RNA structures

We need to load both during validation - the scientific features to feed into the model and the coordinates to compare against model predictions.

## Implementation Steps

### 1. Feature Loading Architecture

Create a unified data loading pipeline that combines both data sources:

```
ValidationDataLoader
├── NPZFeatureLoader - Load scientific features from processed NPZ files
└── CSVCoordinateLoader - Load 3D coordinates from raw CSV files
```

### 2. Modify CSVDataLoader

Refocus the CSVDataLoader to handle coordinates only:

```python
class CSVCoordinateLoader:
    """Loader specifically for 3D coordinates from CSV files"""
    
    def __init__(self, data_dir=None, split="validation"):
        # Setup logic similar to current implementation
        pass
        
    def get_coordinates(self, target_id):
        # Keep only coordinate loading logic
        # Return atom positions in format needed for TM-score/RMSD calculation
        pass
```

### 3. Implement NPZFeatureLoader

Create a dedicated class for loading the pre-processed scientific features:

```python
class NPZFeatureLoader:
    """Loader for scientific features from NPZ files"""
    
    def __init__(self, data_dir=None, split="validation"):
        # Setup logic to find processed features
        pass
        
    def get_features(self, target_id):
        # Load:
        # - Dihedral features 
        # - Thermodynamic features
        # - Mutual information matrices
        pass
```

### 4. Create Unified ValidationDataset

Implement a PyTorch Dataset that combines both data sources:

```python
class ValidationDataset(Dataset):
    def __init__(self, data_dir=None, split="validation", subset_size=5, seed=42):
        self.coord_loader = CSVCoordinateLoader(data_dir, split)
        self.feature_loader = NPZFeatureLoader(data_dir, split)
        # Setup target IDs and filtering logic
        
    def __getitem__(self, idx):
        target_id = self.target_ids[idx]
        
        # Get coordinates from CSV files
        coordinates = self.coord_loader.get_coordinates(target_id)
        
        # Get scientific features from NPZ files
        features = self.feature_loader.get_features(target_id)
        
        # Combine into a single sample dict with all required fields
        sample = {
            "target_id": target_id,
            "atom_positions": coordinates["atom_positions"],
            "dihedral_features": features["dihedral_features"],
            # Add all other required fields
        }
        
        return sample
```

### 5. Update Validation Framework

Modify the validation notebook to use the new unified dataset:

```python
# Create dataset with both feature types
validation_dataset = ValidationDataset(data_dir=data_dir, split="validation")

# Create dataloader
dataloader = DataLoader(
    validation_dataset, 
    batch_size=batch_size,
    shuffle=False,
    collate_fn=validation_dataset.collate_fn
)

# Run validation with access to both features and coordinates
for batch in dataloader:
    # Model receives scientific features
    outputs = model(batch)
    
    # Evaluation metrics compare predicted coordinates to ground truth
    metrics = evaluate_structure(outputs["positions"], batch["atom_positions"])
```

### 6. Edge Case Handling

Implement robust error handling for:
- Missing NPZ features for some target IDs
- Missing CSV coordinates for some target IDs
- Feature dimension mismatches between expected and actual

## Implementation Timeline

1. **Phase 1: Data Loader Refactoring (2 days)**
   - Refactor CSVDataLoader to focus purely on coordinates
   - Implement NPZFeatureLoader

2. **Phase 2: Integration (2 days)**
   - Create ValidationDataset class that combines both loaders
   - Update validation notebook to use the new dataset

3. **Phase 3: Testing & Validation (1 day)**
   - Test with subset of validation data
   - Verify all features and coordinates are loaded correctly
   - Validate structure metrics calculation

## Success Metrics

- Successful loading of both scientific features and 3D coordinates
- Correctly calculated RMSD and TM-score metrics
- No dimension mismatch warnings in model forward pass
- Memory efficient data loading (no unnecessary duplication)