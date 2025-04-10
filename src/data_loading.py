import os
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


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
    coords = target_rows[['x_1', 'y_1', 'z_1']].values.astype(np.float32)
    
    # Get residue names
    resnames = target_rows['resname'].tolist()
    
    return coords, resnames


def load_precomputed_features(target_id: str, features_dir: str) -> Dict[str, Union[Dict[str, np.ndarray], None]]:
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
        warnings.warn(f"Dihedral features not found for {target_id}. Using zeros.")
    
    # 2. Load thermodynamic features (required)
    thermo_path = os.path.join(features_dir, "thermo_features", f"{target_id}_thermo_features.npz")
    if not os.path.exists(thermo_path):
        raise ValueError(f"Thermodynamic features not found for {target_id}. Required for prediction.")
    
    with np.load(thermo_path) as data:
        # Extract key arrays and scalar values
        thermo_features = {}
        
        # Get pairing probabilities matrix (critical)
        if 'pairing_probs' in data:
            thermo_features['pairing_probs'] = data['pairing_probs'].astype(np.float32)
        elif 'base_pair_probs' in data:
            thermo_features['pairing_probs'] = data['base_pair_probs'].astype(np.float32)
        else:
            raise ValueError(f"No pairing probability matrix found for {target_id}")
        
        # Get sequence length for defaults
        seq_len = thermo_features['pairing_probs'].shape[0]
        
        # Get positional entropy
        if 'positional_entropy' in data:
            thermo_features['positional_entropy'] = data['positional_entropy'].astype(np.float32)
        elif 'position_entropy' in data:
            thermo_features['positional_entropy'] = data['position_entropy'].astype(np.float32)
        else:
            thermo_features['positional_entropy'] = np.zeros(seq_len, dtype=np.float32)
            warnings.warn(f"No positional entropy found for {target_id}. Using zeros.")
        
        # Get accessibility
        if 'accessibility' in data:
            thermo_features['accessibility'] = data['accessibility'].astype(np.float32)
        else:
            thermo_features['accessibility'] = np.zeros(seq_len, dtype=np.float32)
            warnings.warn(f"No accessibility found for {target_id}. Using zeros.")
            
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
    mi_path = os.path.join(features_dir, "mi_features", f"{target_id}_features.npz")
    if os.path.exists(mi_path):
        with np.load(mi_path) as data:
            features['evolutionary'] = {
                'coupling_matrix': data['coupling_matrix'].astype(np.float32)
            }
            if 'conservation' in data:
                features['evolutionary']['conservation'] = data['conservation'].astype(np.float32)
    else:
        # Create empty evolutionary features based on sequence length
        if 'thermo' in features and 'pairing_probs' in features['thermo']:
            seq_len = features['thermo']['pairing_probs'].shape[0]
            features['evolutionary'] = {
                'coupling_matrix': np.zeros((seq_len, seq_len), dtype=np.float32)
            }
        else:
            features['evolutionary'] = None
        warnings.warn(f"Evolutionary features not found for {target_id}. Using zeros.")
    
    return features


class RNADataset(Dataset):
    """Dataset class for RNA 3D structure prediction."""
    
    def __init__(
        self, 
        sequences_csv_path: str,
        labels_csv_path: str,
        features_dir: str,
        temporal_cutoff: Optional[str] = None,
        use_validation_set: bool = False
    ):
        """
        Initialize RNA dataset.
        
        Args:
            sequences_csv_path: Path to sequences CSV file
            labels_csv_path: Path to labels CSV file
            features_dir: Directory containing feature files
            temporal_cutoff: Optional date string (YYYY-MM-DD) for filtering training data
            use_validation_set: Whether to use validation data
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
                    warnings.warn(f"Could not load coordinates for {target_id}: {e}")
        
        # Nucleotide to integer mapping
        self.nuc_to_int = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}
    
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
        
        # Add scalar thermodynamic features (as individual scalars)
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


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        if key in ['target_id', 'length']:
            continue  # Already processed
            
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
                
            elif len(sample_shape) == 2 and key in ['dihedral_features', 'coordinates']:
                # 2D tensor with feature dimension
                feature_dim = sample_shape[1]
                padded = torch.zeros((batch_size, max_len, feature_dim), dtype=batch[0][key].dtype)
                for i, sample in enumerate(batch):
                    if key in sample:  # Check if feature is available
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
            
            else:
                # Handle other tensor types
                continue
        
        elif isinstance(batch[0][key], (int, float)):
            # Handle scalar values
            output[key] = torch.tensor([sample[key] for sample in batch if key in sample])
    
    # Create attention mask (True for valid positions, False for padding)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, sample in enumerate(batch):
        mask[i, :sample['length']] = True
    output['mask'] = mask
    
    return output


def create_data_loader(
    sequences_csv_path: str,
    labels_csv_path: str,
    features_dir: str,
    batch_size: int,
    temporal_cutoff: Optional[str] = None,
    use_validation_set: bool = False,
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
        use_validation_set=use_validation_set
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
