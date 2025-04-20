# Partial Data Handling Strategy

This document outlines the detailed strategy for handling partial data in the RNA 3D folding pipeline. The requirement to support sequences with varying feature availability is a central design consideration throughout the data loading implementation.

## Overview

The data pipeline must support a scenario where:

1. Only a subset of RNA sequences have computed features
2. Each sequence either has all feature types or none
3. The dataset can be dynamically updated as more features become available

This capability is essential for iterative model development, where feature generation may be ongoing while model training proceeds with the available data.

## Key Components

### 1. Feature Availability Detection

The foundation of partial data handling is a robust feature availability detection system:

```python
def check_features_availability(target_id: str, features_dir: str) -> Dict[str, bool]:
    """Check which features are available for a given target.
    
    Args:
        target_id: The ID of the target RNA sequence
        features_dir: Directory containing feature subdirectories
        
    Returns:
        Dictionary mapping feature types to availability (True/False)
    """
    availability = {
        "dihedral": False,
        "thermo": False,
        "mi": False
    }
    
    # Check each feature type
    for feature_type in availability.keys():
        feature_path = os.path.join(
            features_dir, 
            f"{feature_type}_features", 
            f"{target_id}_{feature_type}_features.npz"
        )
        availability[feature_type] = os.path.exists(feature_path)
        
    return availability
```

Key aspects:
- Returns a dictionary mapping feature types to boolean availability
- Performs filesystem checks for each feature type
- Path construction follows the required pattern: `{features_dir}/{feature_type}_features/{target_id}_{feature_type}_features.npz`

### 2. Caching for Performance

To avoid repeated filesystem operations, a caching mechanism is implemented:

```python
class RNADataset(Dataset):
    def __init__(self, ...):
        # Initialize availability cache
        self._availability_cache = {}
        
        # Populate cache during initialization
        self._scan_available_features()
    
    def _scan_available_features(self):
        """Scan features directory and populate availability cache."""
        for target_id in self.target_ids:
            self._availability_cache[target_id] = check_features_availability(
                target_id, self.features_dir)
```

Key aspects:
- Cache is populated during initialization
- Stores results of availability checks for each target ID
- Cache is invalidated and rebuilt when update mechanism is triggered

### 3. Sequence Filtering

The dataset provides a filtering mechanism to include only sequences with required features:

```python
def _update_filtered_sequences(self):
    """Update the list of filtered sequences based on feature requirements."""
    if not self.require_features:
        # Include all sequences if features not required
        self.filtered_sequences = self.sequences_df['target_id'].tolist()
        return
        
    # Filter based on feature availability
    self.filtered_sequences = []
    for target_id in self.sequences_df['target_id']:
        if target_id not in self._availability_cache:
            continue
            
        # Check if all required features are available
        avail = self._availability_cache[target_id]
        if all(avail.values()):
            self.filtered_sequences.append(target_id)
```

Key aspects:
- Maintains a `filtered_sequences` list containing only targets with required features
- Responds to `require_features` parameter to control filtering behavior
- Can be updated dynamically when new features become available

### 4. Dynamic Update Mechanism

The dataset provides a method to incorporate newly available features:

```python
def update_available_features(self):
    """Update the list of available features.
    
    Rescans the features directory to identify newly available feature files
    and updates the filtered sequence list accordingly.
    
    Returns:
        Number of sequences with complete feature sets
    """
    # Reset availability cache
    self._availability_cache = {}
    
    # Scan for available features
    self._scan_available_features()
    
    # Update filtered sequences
    self._update_filtered_sequences()
    
    # Return number of valid sequences
    return len(self.filtered_sequences)
```

Key aspects:
- Invalidates and rebuilds the availability cache
- Updates the filtered sequence list based on new availability
- Returns the number of sequences with complete feature sets for monitoring

### 5. Metadata Flag Generation

The data pipeline generates metadata flags indicating feature presence:

```python
def __getitem__(self, idx):
    """Get features and labels for a single RNA sequence."""
    target_id = self.filtered_sequences[idx]
    
    # ... load features and other data ...
    
    # Generate metadata flags
    meta = {
        "has_dihedrals": torch.tensor(self._availability_cache[target_id]["dihedral"]),
        "has_thermo": torch.tensor(self._availability_cache[target_id]["thermo"]),
        "has_msa": torch.tensor(self._availability_cache[target_id]["mi"]),
    }
    
    return {
        "target_id": target_id,
        "sequence_int": sequence_int,
        # ... other features ...
        "meta": meta
    }
```

Key aspects:
- Each feature type has a corresponding boolean flag
- Flags are derived from the availability cache
- Flags are included in the metadata dictionary

### 6. Batch Aggregation

The collation function aggregates metadata flags across the batch:

```python
def collate_fn(batch):
    """Collate a list of samples into a batch."""
    # ... collate other features ...
    
    # Collate metadata flags
    meta = {}
    for key in batch[0]["meta"].keys():
        meta[key] = torch.stack([sample["meta"][key] for sample in batch])
    
    return {
        # ... other batched features ...
        "meta": meta
    }
```

Key aspects:
- Metadata flags are stacked into tensors of shape (batch_size,)
- Each flag maintains its boolean dtype
- Flags are included in the batch dictionary under the "meta" key

## Usage Examples

### 1. Creating a Dataset with Feature Filtering

```python
# Only use sequences with available features
dataset = RNADataset(
    sequences_csv_path="data/train_sequences.csv",
    labels_csv_path="data/train_labels.csv",
    features_dir="data/processed",
    require_features=True  # Only include sequences with features
)

print(f"Dataset contains {len(dataset)} sequences with features")
```

### 2. Updating with New Features

```python
# Initial dataset
dataset = RNADataset(
    sequences_csv_path="data/train_sequences.csv",
    labels_csv_path="data/train_labels.csv",
    features_dir="data/processed",
    require_features=True
)

initial_count = len(dataset)
print(f"Initial dataset contains {initial_count} sequences")

# After generating new features
count = dataset.update_available_features()
print(f"Updated dataset contains {count} sequences (+{count - initial_count} new)")
```

### 3. Using Metadata Flags in the Model

```python
# Create data loader
loader = create_data_loader(
    sequences_csv_path="data/train_sequences.csv",
    labels_csv_path="data/train_labels.csv",
    features_dir="data/processed",
    batch_size=32,
    require_features=True
)

# Training loop
for batch in loader:
    # Extract metadata flags
    has_dihedrals = batch["meta"]["has_dihedrals"]  # shape: (batch_size,)
    has_msa = batch["meta"]["has_msa"]              # shape: (batch_size,)
    
    # Use flags to control model behavior
    if has_dihedrals.all():
        # Use dihedral features for all sequences in batch
        dihedral_output = model.dihedral_module(batch["dihedral_features"])
    
    # ... rest of training loop ...
```

## Testing Strategy

Testing for partial data handling is comprehensive:

1. **Feature Availability Tests**
   - Test detection of available features for different targets
   - Test caching mechanism for performance
   - Test with invalid or malformed feature files

2. **Sequence Filtering Tests**
   - Test filtering with `require_features=True`
   - Test inclusion of all sequences with `require_features=False`
   - Test with mix of available and unavailable features

3. **Update Mechanism Tests**
   - Test updating when new features become available
   - Test updating when features are removed
   - Test with empty feature directories

4. **Metadata Flag Tests**
   - Test flag generation for all feature types
   - Test batch aggregation of flags
   - Test consistency between flags and feature availability

5. **End-to-End Tests**
   - Test full pipeline with partial feature availability
   - Test integration with model components using metadata flags
   - Test performance with varying levels of feature availability

This comprehensive testing ensures the partial data handling system is robust and reliable.

## Conclusion

The partial data handling strategy enables progressive feature generation and incorporation, supporting an iterative development approach. By properly detecting feature availability, filtering sequences, and providing update mechanisms, the data pipeline maintains flexibility while ensuring deterministic behavior.