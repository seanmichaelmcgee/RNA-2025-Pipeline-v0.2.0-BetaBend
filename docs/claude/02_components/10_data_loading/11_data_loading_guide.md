# Data Loading Implementation Guide

## Component Overview

The data loading component is responsible for loading, preprocessing, and batching precomputed RNA features and labels for training and inference. This component forms the foundation of our pipeline, providing properly formatted tensor inputs to the model.

## Core Requirements

| Requirement ID | Description |
|---------------|-------------|
| DL-01 | Implement `RNADataset` class inheriting from `torch.utils.data.Dataset` |
| DL-02 | Constructor accepts paths for sequences, labels, and features as arguments (no hardcoded paths) |
| DL-03 | Support temporal cutoff filtering based on provided date |
| DL-04 | Support validation set loading with appropriate filtering |
| DL-05 to DL-06 | Implement standard Dataset methods (__len__, __getitem__) |
| DL-07 | Implement `collate_fn` for batching variable-length sequences with proper padding and masking |
| DL-08 | Ensure compatibility with DistributedSampler for future multi-GPU training |

## Implementation Strategy

### 1. Feature Loading Functions

First, implement helper functions to load specific feature types from the precomputed files:

```python
def load_coordinates(labels_df: pd.DataFrame, target_id: str) -> Tuple[np.ndarray, List[str]]:
    """
    Extract C1' coordinates for a given target ID from labels DataFrame.
    
    Args:
        labels_df: DataFrame containing coordinates data
        target_id: Target ID to extract coordinates for
        
    Returns:
        Tuple of (coordinates array of shape (N, 3), list of residue names)
    """
    # Filter rows for the target_id
    target_rows = labels_df[labels_df['ID'].str.startswith(f"{target_id}_")]
    
    if len(target_rows) == 0:
        raise ValueError(f"No coordinates found for target {target_id}")
    
    # Extract coordinates and sort by residue ID
    target_rows = target_rows.sort_values(by='resid')
    
    # Get coordinates (x_1, y_1, z_1)
    coords = target_rows[['x_1', 'y_1', 'z_1']].values
    
    # Get residue names
    resnames = target_rows['resname'].tolist()
    
    return coords, resnames
```

```python
def load_precomputed_features(target_id: str, features_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all precomputed features for a target from .npz files.
    
    Args:
        target_id: RNA sequence identifier
        features_dir: Directory containing feature files
        
    Returns:
        Dictionary of feature dictionaries with structure:
        {
            'dihedral': {'features': array(...)},
            'thermo': {'pairing_probs': array(...), 'mfe': value, ...},
            'evolutionary': {'coupling_matrix': array(...)}
        }
    """
    features = {}
    
    # 1. Load dihedral features
    dihedral_path = os.path.join(features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
    if os.path.exists(dihedral_path):
        with np.load(dihedral_path) as data:
            features['dihedral'] = {
                'features': data['features'].astype(np.float32)
            }
            # Handle NaN values if present
            if np.isnan(features['dihedral']['features']).any():
                features['dihedral']['features'] = np.nan_to_num(
                    features['dihedral']['features'], nan=0.0
                )
    else:
        # For test data or if file is missing, default will be created later based on sequence length
        features['dihedral'] = None
    
    # 2. Load thermodynamic features (required)
    thermo_path = os.path.join(features_dir, "thermo_features", f"{target_id}_thermo_features.npz")
    if not os.path.exists(thermo_path):
        raise ValueError(f"Thermodynamic features not found for {target_id}. Required for prediction.")
    
    with np.load(thermo_path) as data:
        # Extract key arrays and scalar values
        thermo_features = {}
        
        # Get pairing probabilities matrix (critical)
        key = 'pairing_probs' if 'pairing_probs' in data else 'base_pair_probs'
        thermo_features['pairing_probs'] = data[key].astype(np.float32)
        
        # Get sequence length for defaults
        seq_len = thermo_features['pairing_probs'].shape[0]
        
        # Get positional entropy
        key = 'positional_entropy' if 'positional_entropy' in data else 'position_entropy'
        if key in data:
            thermo_features['positional_entropy'] = data[key].astype(np.float32)
        
        # Get accessibility
        if 'accessibility' in data:
            thermo_features['accessibility'] = data['accessibility'].astype(np.float32)
            
        # Get sequence if available
        if 'sequence' in data:
            thermo_features['sequence'] = str(data['sequence'])
            
        # Get scalar features
        scalar_features = ['mfe', 'ensemble_energy', 'mfe_probability', 'gc_content', 'paired_fraction']
        for key in scalar_features:
            if key in data:
                thermo_features[key] = float(data[key])
                
        features['thermo'] = thermo_features
    
    # 3. Load evolutionary coupling features (optional)
    mi_path = os.path.join(features_dir, "mi_features", f"{target_id}_mi_features.npz")
    if os.path.exists(mi_path):
        with np.load(mi_path) as data:
            evolutionary_features = {}
            
            # Get coupling matrix
            if 'coupling_matrix' in data:
                coupling_matrix = data['coupling_matrix'].astype(np.float32)
                
                # Check if matrix is uniform (single sequence MSA case)
                if is_uniform_mi_matrix(coupling_matrix):
                    warnings.warn(f"Uniform MI matrix detected for {target_id}. Treating as missing.")
                    evolutionary_features['coupling_matrix'] = np.zeros_like(coupling_matrix)
                    evolutionary_features['has_valid_mi'] = False
                else:
                    evolutionary_features['coupling_matrix'] = coupling_matrix
                    evolutionary_features['has_valid_mi'] = True
            else:
                raise ValueError(f"No coupling matrix found for {target_id}")
            
            # Get conservation if available
            if 'conservation' in data:
                evolutionary_features['conservation'] = data['conservation'].astype(np.float32)
            
            features['evolutionary'] = evolutionary_features
    else:
        # Create empty evolutionary features based on sequence length
        if 'thermo' in features and 'pairing_probs' in features['thermo']:
            seq_len = features['thermo']['pairing_probs'].shape[0]
            features['evolutionary'] = {
                'coupling_matrix': np.zeros((seq_len, seq_len), dtype=np.float32),
                'has_valid_mi': False
            }
        else:
            features['evolutionary'] = None
    
    return features

def is_uniform_mi_matrix(matrix: np.ndarray, epsilon: float = 1e-6) -> bool:
    """
    Check if an MI matrix contains uniform values, indicating a single sequence MSA.
    
    Args:
        matrix: Mutual information matrix
        epsilon: Threshold for standard deviation to consider uniform
        
    Returns:
        True if matrix appears to have uniform off-diagonal values
    """
    # Get values excluding diagonal 
    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    
    # Empty matrix case
    if len(off_diag) == 0:
        return False
        
    # Check if standard deviation is near zero
    return np.std(off_diag) < epsilon
```

### 2. RNADataset Class

Next, implement the main dataset class to load sequences, labels, and features:

```python
class RNADataset(torch.utils.data.Dataset):
    """Dataset class for RNA 3D structure prediction."""
    
    def __init__(
        self, 
        sequences_csv_path: str,
        labels_csv_path: str,
        features_dir: str,
        temporal_cutoff: Optional[str] = None,
        use_validation_set: bool = False,
        require_features: bool = True
    ):
        """
        Initialize RNA dataset.
        
        Args:
            sequences_csv_path: Path to sequences CSV file
            labels_csv_path: Path to labels CSV file
            features_dir: Directory containing feature files
            temporal_cutoff: Optional date string (YYYY-MM-DD) for filtering training data
            use_validation_set: Whether to use validation data
            require_features: Whether to require all features (filter out targets missing features)
        """
        # Store paths (NO hardcoded paths)
        self.features_dir = features_dir
        
        # Load sequences
        self.sequences_df = pd.read_csv(sequences_csv_path)
        
        # Filter by temporal cutoff if specified
        if temporal_cutoff is not None and not use_validation_set:
            self.sequences_df = self.sequences_df[
                pd.to_datetime(self.sequences_df['temporal_cutoff']) <= pd.to_datetime(temporal_cutoff)
            ]
        
        # Extract target IDs and sequences
        self.target_ids = self.sequences_df['target_id'].tolist()
        self.sequences = self.sequences_df['sequence'].tolist()
        
        # Filter by feature availability if required
        if require_features:
            self.available_features = self.get_available_features()
            self.target_ids = [
                target_id for target_id in self.target_ids
                if target_id in self.available_features
            ]
            self.sequences = [
                seq for i, seq in enumerate(self.sequences)
                if self.sequences_df.iloc[i]['target_id'] in self.available_features
            ]
        
        # Load labels if available
        self.labels_df = None
        self.coordinates = None
        if labels_csv_path is not None and os.path.exists(labels_csv_path):
            self.labels_df = pd.read_csv(labels_csv_path)
            
            # Pre-load all coordinates if not too memory intensive
            # For large datasets, you might want to load coordinates on-demand in __getitem__
            self.coordinates = {}
            for target_id in self.target_ids:
                try:
                    coords, _ = load_coordinates(self.labels_df, target_id)
                    self.coordinates[target_id] = coords
                except Exception as e:
                    print(f"Warning: Could not load coordinates for {target_id}: {e}")
        
        # Nucleotide to integer mapping
        self.nuc_to_int = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}
    
    def get_available_features(self) -> Set[str]:
        """
        Scan features directory to find targets with required features.
        
        Returns:
            Set of target IDs with available features
        """
        available = set()
        
        # For each target, check if it has thermodynamic features (minimum requirement)
        for target_id in self.target_ids:
            thermo_path = os.path.join(
                self.features_dir, "thermo_features", f"{target_id}_thermo_features.npz"
            )
            if os.path.exists(thermo_path):
                available.add(target_id)
        
        return available
    
    def update_available_features(self):
        """
        Rescan features directory to update available features.
        Call this after adding new feature files to incorporate them.
        """
        new_available = self.get_available_features()
        # Find targets that are now available but weren't before
        new_targets = new_available - set(self.target_ids)
        
        # Add new targets to our dataset
        if new_targets:
            new_indices = [
                i for i, target_id in enumerate(self.sequences_df['target_id']) 
                if target_id in new_targets
            ]
            new_rows = self.sequences_df.iloc[new_indices]
            
            # Add targets and sequences
            self.target_ids.extend(new_rows['target_id'].tolist())
            self.sequences.extend(new_rows['sequence'].tolist())
            
            # Load coordinates for new targets if labels available
            if self.labels_df is not None:
                for target_id in new_targets:
                    try:
                        coords, _ = load_coordinates(self.labels_df, target_id)
                        self.coordinates[target_id] = coords
                    except Exception as e:
                        print(f"Warning: Could not load coordinates for {target_id}: {e}")
            
            print(f"Added {len(new_targets)} new targets with features")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.target_ids)
    
    def sequence_to_int(self, sequence: str) -> List[int]:
        """Convert nucleotide sequence to integer indices."""
        return [self.nuc_to_int.get(nuc, 4) for nuc in sequence]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary of tensors and metadata
        """
        target_id = self.target_ids[idx]
        sequence = self.sequences[idx]
        sequence_length = len(sequence)
        
        # Load precomputed features
        try:
            features = load_precomputed_features(target_id, self.features_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading features for {target_id}: {e}")
        
        # Convert sequence to integers
        sequence_int = self.sequence_to_int(sequence)
        
        # Create sample dictionary
        sample = {
            'target_id': target_id,
            'sequence_int': torch.tensor(sequence_int, dtype=torch.long),
            'length': sequence_length
        }
        
        # Add dihedral features if available
        if features['dihedral'] is not None:
            sample['dihedral_features'] = torch.tensor(
                features['dihedral']['features'], dtype=torch.float32)
        else:
            # Create default zero tensor
            sample['dihedral_features'] = torch.zeros((sequence_length, 4), dtype=torch.float32)
        
        # Add thermodynamic features
        sample['pairing_probs'] = torch.tensor(
            features['thermo']['pairing_probs'], dtype=torch.float32)
        
        if 'positional_entropy' in features['thermo']:
            sample['positional_entropy'] = torch.tensor(
                features['thermo']['positional_entropy'], dtype=torch.float32)
        else:
            sample['positional_entropy'] = torch.zeros(sequence_length, dtype=torch.float32)
            
        if 'accessibility' in features['thermo']:
            sample['accessibility'] = torch.tensor(
                features['thermo']['accessibility'], dtype=torch.float32)
        else:
            sample['accessibility'] = torch.zeros(sequence_length, dtype=torch.float32)
        
        # Add scalar thermodynamic features (broadcast to per-residue)
        scalar_features = ['mfe', 'ensemble_energy', 'mfe_probability', 'gc_content']
        for key in scalar_features:
            if key in features['thermo']:
                # Store as scalar
                sample[key] = features['thermo'][key]
        
        # Add evolutionary features if available
        if features['evolutionary'] is not None:
            sample['coupling_matrix'] = torch.tensor(
                features['evolutionary']['coupling_matrix'], dtype=torch.float32)
            
            if 'conservation' in features['evolutionary']:
                sample['conservation'] = torch.tensor(
                    features['evolutionary']['conservation'], dtype=torch.float32)
            else:
                sample['conservation'] = torch.zeros(sequence_length, dtype=torch.float32)
        else:
            # Create default zero tensors
            sample['coupling_matrix'] = torch.zeros(
                (sequence_length, sequence_length), dtype=torch.float32)
            sample['conservation'] = torch.zeros(sequence_length, dtype=torch.float32)
        
        # Add coordinates if available (for training)
        if self.coordinates is not None and target_id in self.coordinates:
            sample['coordinates'] = torch.tensor(
                self.coordinates[target_id], dtype=torch.float32)
        
        # Add metadata flags for feature presence
        sample['meta'] = {
            'has_dihedrals': features['dihedral'] is not None,
            'has_msa': features['evolutionary'] is not None and features['evolutionary'].get('has_valid_mi', False),
        }
        
        # Verify shapes
        expected_length = len(sequence)
        for key, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                if key in ['dihedral_features', 'positional_entropy', 'accessibility', 'conservation']:
                    assert tensor.shape[0] == expected_length, \
                        f"Feature {key} length mismatch: {tensor.shape[0]} vs {expected_length}"
                elif key in ['pairing_probs', 'coupling_matrix']:
                    assert tensor.shape[0] == expected_length and tensor.shape[1] == expected_length, \
                        f"Feature {key} shape mismatch: {tensor.shape} vs ({expected_length}, {expected_length})"
        
        return sample
```

### 3. Collate Function

Implement the collate function to handle variable-length sequences:

```python
def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch: List of samples from RNADataset.__getitem__
        
    Returns:
        Dictionary of batched tensors with padding
    """
    # Get batch size and maximum sequence length
    batch_size = len(batch)
    max_len = max(sample['length'] for sample in batch)
    
    # Extract target IDs
    target_ids = [sample['target_id'] for sample in batch]
    
    # Initialize output dictionary
    output = {
        'target_ids': target_ids,
        'lengths': torch.tensor([sample['length'] for sample in batch], dtype=torch.long)
    }
    
    # Process each tensor in the batch
    for key in batch[0].keys():
        if key in ['target_id', 'length', 'meta']:
            continue  # Already processed or handled separately
            
        if isinstance(batch[0][key], torch.Tensor):
            # Process tensor based on its shape
            sample_shape = batch[0][key].shape
            
            if len(sample_shape) == 1:
                # 1D tensor (sequence, per-residue features)
                padded = torch.zeros((batch_size, max_len), dtype=batch[0][key].dtype)
                for i, sample in enumerate(batch):
                    length = sample['length']
                    padded[i, :length] = sample[key]
                output[key] = padded
                
            elif len(sample_shape) == 2 and key in ['dihedral_features']:
                # 2D tensor with feature dimension
                feature_dim = sample_shape[1]
                padded = torch.zeros((batch_size, max_len, feature_dim), dtype=batch[0][key].dtype)
                for i, sample in enumerate(batch):
                    length = sample['length']
                    padded[i, :length, :] = sample[key]
                output[key] = padded
                
            elif len(sample_shape) == 2 and key in ['pairing_probs', 'coupling_matrix']:
                # 2D matrix (pair features)
                padded = torch.zeros((batch_size, max_len, max_len), dtype=batch[0][key].dtype)
                for i, sample in enumerate(batch):
                    length = sample['length']
                    padded[i, :length, :length] = sample[key]
                output[key] = padded
                
            elif len(sample_shape) == 2 and key == 'coordinates':
                # Coordinates (N, 3)
                padded = torch.zeros((batch_size, max_len, 3), dtype=batch[0][key].dtype)
                for i, sample in enumerate(batch):
                    if key in sample:  # Check if coordinates are available
                        length = sample['length']
                        padded[i, :length, :] = sample[key]
                output[key] = padded
            
            else:
                # Handle other tensor types if needed
                continue
        
        elif isinstance(batch[0][key], (int, float)):
            # Handle scalar values
            output[key] = torch.tensor([sample[key] for sample in batch if key in sample])
    
    # Create attention mask (1 for valid positions, 0 for padding)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, sample in enumerate(batch):
        mask[i, :sample['length']] = True
    output['mask'] = mask
    
    # Collect metadata flags
    meta = {}
    for key in batch[0]['meta'].keys():
        meta[key] = torch.tensor([sample['meta'][key] for sample in batch], dtype=torch.bool)
    output['meta'] = meta
    
    return output
```

## Shape and Type Specifications

Here are the expected tensor shapes in the batched output:

| Tensor Name | Shape | Type | Description |
|-------------|-------|------|-------------|
| `sequence_int` | `(B, L)` | `torch.long` | Integer-encoded RNA sequence |
| `dihedral_features` | `(B, L, 4)` | `torch.float32` | Sin/cos of dihedral angles |
| `pairing_probs` | `(B, L, L)` | `torch.float32` | Base-pairing probability matrix |
| `positional_entropy` | `(B, L)` | `torch.float32` | Shannon entropy at each position |
| `accessibility` | `(B, L)` | `torch.float32` | Unpaired probability per nucleotide |
| `coupling_matrix` | `(B, L, L)` | `torch.float32` | Evolutionary coupling score matrix |
| `conservation` | `(B, L)` | `torch.float32` | Conservation score per position |
| `coordinates` | `(B, L, 3)` | `torch.float32` | C1' coordinates (x, y, z) |
| `mask` | `(B, L)` | `torch.bool` | Attention mask (True for valid positions) |
| `lengths` | `(B)` | `torch.long` | Sequence lengths |
| `meta` | Dict | Various | Metadata flags including feature presence |

Where:
- `B` is the batch size
- `L` is the maximum sequence length in the batch

## Data Loader Creation

Use the Dataset and collate function to create a DataLoader:

```python
def create_data_loader(
    sequences_csv_path: str,
    labels_csv_path: str,
    features_dir: str,
    batch_size: int,
    temporal_cutoff: Optional[str] = None,
    use_validation_set: bool = False,
    require_features: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False
):
    """
    Create a DataLoader for RNA 3D structure prediction.
    
    Args:
        sequences_csv_path: Path to sequences CSV
        labels_csv_path: Path to labels CSV
        features_dir: Path to directory with feature files
        batch_size: Batch size
        temporal_cutoff: Optional cutoff date for filtering
        use_validation_set: Whether to use validation set
        require_features: Whether to require all features
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        distributed: Whether to use DistributedSampler
        
    Returns:
        PyTorch DataLoader
    """
    # Create dataset
    dataset = RNADataset(
        sequences_csv_path=sequences_csv_path,
        labels_csv_path=labels_csv_path,
        features_dir=features_dir,
        temporal_cutoff=temporal_cutoff,
        use_validation_set=use_validation_set,
        require_features=require_features
    )
    
    # Create sampler for distributed training
    sampler = None
    if distributed and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=shuffle
        )
        shuffle = False  # Sampling is handled by the DistributedSampler
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    return data_loader
```

## Testing Strategy

To ensure the data loading component works correctly, implement these tests:

1. **Dataset Initialization**
   - Test with different temporal cutoffs
   - Test with validation set mode
   - Verify filtering logic works correctly
   - Test feature availability filtering

2. **Feature Loading**
   - Test loading different feature types
   - Verify handling of missing feature files
   - Check shape consistency across features
   - **Test uniform MI matrix detection**

3. **Batch Collation**
   - Test with variable sequence lengths
   - Verify padding and masking
   - Check all tensor shapes and types
   - Verify metadata flags are properly collected

4. **End-to-End Data Flow**
   - Create data loader and iterate through batches
   - Verify batch contents match expectations
   - Test compatibility with device transfer
   - Test feature update mechanism

## Common Pitfalls

1. **Path Handling**: Ensure all paths are passed as arguments, not hardcoded in the implementation.
2. **Missing Features**: Some sequences may not have all feature types (especially evolutionary). Handle gracefully with default tensors.
3. **Padding Consistency**: Ensure all tensors are properly padded to the same sequence length within a batch.
4. **Shape Verification**: Validate that all feature shapes are consistent with the sequence length.
5. **NaN Handling**: Check for and handle NaN values in dihedral features (common at sequence boundaries).
6. **Memory Efficiency**: For large datasets, consider lazy loading of features rather than pre-loading all coordinates.
7. **Device Transfer**: The `collate_fn` should return CPU tensors; device transfer happens later in the training loop.
8. **Uniform MI Matrices**: Watch for MI matrices with identical values throughout (excluding diagonal), typically resulting from single-sequence MSAs with no actual evolutionary information. These should be treated as missing features.

## Integration with Config

The data loading component should read paths and parameters from the configuration:

```python
# Example usage in scripts/train.py
def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Extract data config
    data_config = config['data']
    
    # Create data loader
    data_loader = create_data_loader(
        sequences_csv_path=data_config['sequences_csv_path'],
        labels_csv_path=data_config['labels_csv_path'],
        features_dir=data_config['features_dir'],
        batch_size=data_config['batch_size'],
        temporal_cutoff=data_config.get('temporal_cutoff'),
        require_features=data_config.get('require_features', True),
        num_workers=data_config.get('num_workers', 4)
    )
    
    # Use data_loader for training
    # ...
```

## Next Steps

After implementing the data loading component:

1. Validate with example feature files to ensure correct loading
2. Implement unit tests for the dataset and collate function
3. Test the uniform MI matrix detection with examples from the training data
4. Verify the metadata flags are properly propagated through batch collation
5. Proceed to implementing the embedding layers, which will consume the outputs from this component
