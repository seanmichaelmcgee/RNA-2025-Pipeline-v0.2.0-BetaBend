# Interface Exports

This document defines the formalized interfaces that the Data Pipeline instance exports to other instances, particularly for integration and testing.

## Batch Structure Interface

```python
# Batch dictionary structure returned by DataLoader
batch = {
    # Core sequence information
    "target_ids": List[str],                 # List of target IDs in batch
    "sequence_int": torch.Tensor,            # (batch_size, max_seq_len), dtype: int64
    "lengths": torch.Tensor,                 # (batch_size), dtype: int64
    "mask": torch.Tensor,                    # (batch_size, max_seq_len), dtype: bool, True for valid positions
    
    # Feature tensors
    "dihedral_features": torch.Tensor,       # (batch_size, max_seq_len, 4), dtype: float32
    "pairing_probs": torch.Tensor,           # (batch_size, max_seq_len, max_seq_len), dtype: float32
    "positional_entropy": torch.Tensor,      # (batch_size, max_seq_len), dtype: float32
    "coupling_matrix": torch.Tensor,         # (batch_size, max_seq_len, max_seq_len), dtype: float32
    
    # Target values
    "coordinates": torch.Tensor,             # (batch_size, max_seq_len, 3), dtype: float32
    
    # Metadata
    "meta": {
        "has_dihedrals": torch.Tensor,       # (batch_size), dtype: bool
        "has_thermo": torch.Tensor,          # (batch_size), dtype: bool
        "has_msa": torch.Tensor,             # (batch_size), dtype: bool - TRUE only if MI exists AND is valid
        "before_cutoff": torch.Tensor,       # (batch_size), dtype: bool - TRUE if sequence is before temporal cutoff
        "is_train": torch.Tensor,            # (batch_size), dtype: bool - Indicates train vs validation
    }
}
```

## DataLoader Interface

```python
def create_data_loader(
    sequences_csv_path: str,
    labels_csv_path: Optional[str],
    features_dir: str,
    batch_size: int,
    split_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    temporal_cutoff: Optional[str] = None,
    use_validation_set: bool = False,
    require_features: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False
) -> torch.utils.data.DataLoader:
    """Create data loader for RNA structure prediction with flexible splitting and feature filtering.
    
    Args:
        sequences_csv_path: Path to CSV file containing RNA sequences
        labels_csv_path: Path to CSV file containing 3D coordinates (optional for inference)
        features_dir: Directory containing feature subdirectories
        batch_size: Number of sequences per batch
        split_fn: Optional function to apply custom splitting logic
        temporal_cutoff: Optional date string for temporal validation split
        use_validation_set: If True, use validation set, otherwise training set
        require_features: If True, only use sequences with available features
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker threads for data loading
        distributed: Whether to use DistributedSampler
        
    Returns:
        DataLoader object that yields batches in the format specified above
        with methods:
            * set_temporal_cutoff(new_cutoff) - Change temporal cutoff dynamically
            * update_available_features() - Rescan for new available features
    """
    pass
```

## RNADataset Interface

```python
class RNADataset(torch.utils.data.Dataset):
    """Dataset for RNA 3D structure prediction with feature availability filtering."""
    
    def __init__(
        self,
        sequences_csv_path: str,
        labels_csv_path: Optional[str],
        features_dir: str,
        split_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        temporal_cutoff: Optional[str] = None,
        use_validation_set: bool = False,
        require_features: bool = True
    ):
        """Initialize RNA dataset with pluggable split logic and feature filtering.
        
        Args:
            sequences_csv_path: Path to CSV file containing RNA sequences
            labels_csv_path: Path to CSV file containing 3D coordinates (optional for inference)
            features_dir: Directory containing feature subdirectories
            split_fn: Optional function to apply custom splitting logic
            temporal_cutoff: Optional date string for temporal validation split
            use_validation_set: If True, use validation set, otherwise training set
            require_features: If True, only use sequences with available features
        """
        pass
        
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        pass
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get features and labels for a single RNA sequence.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Dictionary containing features and labels for the sequence
        """
        pass
        
    def update_available_features(self) -> int:
        """Update the list of available features.
        
        Rescans the features directory to identify newly available feature files
        and updates the filtered sequence list accordingly, maintaining temporal boundaries.
        
        Returns:
            Number of sequences with complete feature sets
        """
        pass
        
    def set_temporal_cutoff(self, new_cutoff: Optional[str] = None) -> None:
        """Update the temporal cutoff and refilter sequences.
        
        Args:
            new_cutoff: New temporal cutoff date or None to remove cutoff
        """
        pass
```

## Feature Loading Interfaces

```python
def is_uniform_top_pairs(top_pairs: np.ndarray, epsilon: float = 1e-6) -> bool:
    """Check if top pairs from MI features have uniform scores, indicating a single sequence MSA.
    
    Args:
        top_pairs: Array of shape (P, 3) with format [pos_i, pos_j, score]
        epsilon: Threshold for standard deviation to consider uniform
        
    Returns:
        True if all scores are effectively identical
    """
    pass

def is_uniform_mi_matrix(matrix: np.ndarray, epsilon: float = 1e-6) -> bool:
    """Check if an MI matrix contains uniform values, indicating a single sequence MSA.
    
    Args:
        matrix: Mutual information matrix
        epsilon: Threshold for standard deviation to consider uniform
        
    Returns:
        True if matrix appears to have uniform off-diagonal values
    """
    pass

def check_features_availability(
    target_id: str, 
    features_dir: str
) -> Dict[str, bool]:
    """Check which features are available for a given target.
    
    Args:
        target_id: The ID of the target RNA sequence
        features_dir: Directory containing feature subdirectories
        
    Returns:
        Dictionary mapping feature types to availability (True/False)
    """
    pass
    
def load_precomputed_features(
    target_id: str, 
    features_dir: str,
    temporal_cutoff: Optional[str] = None
) -> Dict[str, Union[Dict[str, np.ndarray], None]]:
    """Load all available precomputed features for a given target.
    
    Args:
        target_id: The ID of the target RNA sequence
        features_dir: Directory containing feature subdirectories
        temporal_cutoff: Optional date string for temporal filtering
        
    Returns:
        Dictionary of feature dictionaries with structure:
        {
            'dihedral': {'features': array(...)},
            'thermo': {'pairing_probs': array(...), 'mfe': value, ...},
            'evolutionary': {
                'coupling_matrix': array(...),
                'has_valid_mi': bool  # True if MI data is meaningful (not uniform)
            }
        }
    """
    pass
    
def load_coordinates(
    labels_df: pd.DataFrame, 
    target_id: str
) -> Tuple[np.ndarray, List[str]]:
    """Load C1' coordinates for a given target from labels DataFrame.
    
    Args:
        labels_df: DataFrame containing 3D coordinates
        target_id: The ID of the target RNA sequence
        
    Returns:
        Tuple of (coordinates array of shape (N, 3), list of residue names)
    """
    pass
    
def get_dihedral_tensors(
    target_id: str, 
    features_dir: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get both input and target dihedral tensors.
    
    Args:
        target_id: The ID of the target RNA sequence
        features_dir: Directory containing feature subdirectories
        
    Returns:
        Tuple of (input_tensor, target_tensor), each of shape (seq_len, 4)
    """
    pass
```

## Padding Utility Interfaces

```python
def pad_1d(
    x: torch.Tensor, 
    max_len: int, 
    pad_value: float = 0
) -> torch.Tensor:
    """Pad a 1D tensor to a specified length.
    
    Args:
        x: Input tensor of shape (L,)
        max_len: Target length for padding
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor of shape (max_len,)
    """
    pass
    
def pad_2d(
    x: torch.Tensor, 
    max_len: int, 
    pad_value: float = 0
) -> torch.Tensor:
    """Pad a 2D tensor to a specified length in both dimensions.
    
    Args:
        x: Input tensor of shape (L, L)
        max_len: Target length for padding
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor of shape (max_len, max_len)
    """
    pass
    
def pad_tensor(
    x: torch.Tensor, 
    target_shape: Tuple[int,...], 
    pad_value: float = 0
) -> torch.Tensor:
    """Pad a tensor to a specified shape.
    
    Args:
        x: Input tensor
        target_shape: Target shape for padding
        pad_value: Value to use for padding
        
    Returns:
        Padded tensor of shape target_shape
    """
    pass
```

## Batch Collation Interface

```python
def collate_fn(
    batch: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Collate a list of samples into a batch.
    
    Args:
        batch: List of dictionaries, each containing features for a single sequence
        
    Returns:
        Dictionary containing batched tensors
    """
    pass
```

These interfaces define the public API that other instances can rely on when interacting with the data pipeline components. Any changes to these interfaces will require communication and coordination with the affected instances.