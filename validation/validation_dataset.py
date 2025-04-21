"""
Validation Dataset for RNA 3D folding model.

This module implements a PyTorch Dataset that combines features from NPZ files
and coordinates from CSV files, with dual-mode support (test vs. train).
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import logging
from typing import List, Dict, Optional, Tuple, Any, Union, Callable

# Import our custom loaders
from validation.npz_feature_loader import NPZFeatureLoader
from validation.csv_coordinate_loader import CSVCoordinateLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ValidationDataset")

class ValidationDataset(Dataset):
    """
    Dataset for RNA structure validation with dual-mode support.
    
    Combines features from NPZ files with coordinates from CSV files,
    with separate test and train modes to simulate different feature availability.
    """
    
    def __init__(self, 
                 data_dir: Optional[str] = None,
                 subset_name: str = "technical",
                 test_mode: bool = True,
                 max_targets: Optional[int] = None,
                 seed: int = 42,
                 target_ids: Optional[List[str]] = None):
        """
        Initialize validation dataset.
        
        Args:
            data_dir: Base directory for data. If None, will try to find it.
            subset_name: Validation subset to use ("technical", "scientific", "comprehensive")
            test_mode: If True, excludes features not available at test time
            max_targets: Maximum number of targets to include. If None, uses tier-specific default.
            seed: Random seed for reproducibility
            target_ids: Optional list of specific RNA IDs to validate. If provided, overrides subset selection.
        """
        self.data_dir = data_dir
        self.subset_name = subset_name
        self.test_mode = test_mode
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get max targets based on subset if not specified
        if max_targets is None:
            self.max_targets = self._get_default_size(subset_name)
        else:
            self.max_targets = max_targets
            
        # Initialize loaders
        self.feature_loader = NPZFeatureLoader(data_dir, test_mode=test_mode)
        self.coord_loader = CSVCoordinateLoader(data_dir, split="validation")
        
        # Store explicit target IDs if provided
        self.explicit_target_ids = target_ids
        
        # Find targets with both features and coordinates
        self.target_ids = self._find_valid_targets()
        
        # If specific target IDs were provided, filter to those
        if self.explicit_target_ids:
            self._filter_explicit_targets()
        else:
            # Otherwise select subset of targets based on validation tier
            self.target_ids = self._select_validation_subset()
        
        # Log initialization details
        mode_str = "TEST-EQUIVALENT" if test_mode else "TRAINING-EQUIVALENT"
        logger.info(f"Initialized {mode_str} validation dataset ({subset_name})")
        if self.explicit_target_ids:
            logger.info(f"Using explicitly specified targets: {', '.join(self.target_ids)}")
        else:
            logger.info(f"Selected {len(self.target_ids)} targets from {self.max_targets} max")
        
    def _get_default_size(self, subset_name: str) -> int:
        """Get default subset size based on validation tier."""
        subset_sizes = {
            "technical": 5,      # Small, fast subset for basic validation
            "scientific": 15,    # Medium-sized subset for scientific validation
            "comprehensive": 30  # Large subset for comprehensive validation
        }
        return subset_sizes.get(subset_name, 5)
        
    def _find_valid_targets(self) -> List[str]:
        """Find targets with both features and coordinates."""
        # Get all available targets from both loaders
        feature_targets = set(self.feature_loader.list_available_targets())
        coord_targets = set(self.coord_loader.list_available_targets())
        
        # Intersect to find targets with both features and coordinates
        valid_targets = list(feature_targets.intersection(coord_targets))
        
        if not valid_targets:
            logger.warning("No targets found with both features and coordinates!")
            
        return valid_targets
        
    def _filter_explicit_targets(self) -> None:
        """Filter target IDs to only include explicitly specified IDs."""
        if not self.explicit_target_ids:
            return
        
        # Find the intersection between available targets and requested targets
        valid_targets = []
        missing_targets = []
        
        for target_id in self.explicit_target_ids:
            if target_id in self.target_ids:
                valid_targets.append(target_id)
            else:
                missing_targets.append(target_id)
        
        # Log warnings for missing targets
        if missing_targets:
            logger.warning(f"The following requested RNA IDs were not found: {', '.join(missing_targets)}")
        
        if not valid_targets:
            logger.warning("None of the requested RNA IDs were found in the available validation targets!")
            # Keep the original list if no valid targets were found
            return
        
        # Update the target IDs list with only the valid explicit targets
        self.target_ids = valid_targets
    
    def _select_validation_subset(self) -> List[str]:
        """Select a diverse subset of targets for validation."""
        if not self.target_ids:
            logger.warning("No valid targets found for validation")
            return []
            
        # If we have fewer targets than requested, use all available
        if len(self.target_ids) <= self.max_targets:
            logger.info(f"Using all {len(self.target_ids)} available targets (fewer than requested {self.max_targets})")
            return self.target_ids
            
        # Get sequence information for length-based diversity
        targets_with_length = []
        for target_id in self.target_ids:
            seq_info = self.coord_loader.get_sequence(target_id)
            if seq_info:
                targets_with_length.append((target_id, len(seq_info['sequence'])))
            
        # Group targets by length category
        short = [(t, l) for t, l in targets_with_length if l < 50]
        medium = [(t, l) for t, l in targets_with_length if 50 <= l < 150]
        long = [(t, l) for t, l in targets_with_length if l >= 150]
        
        # Select approximately equal numbers from each length category
        num_each = max(1, self.max_targets // 3)
        
        subset = []
        subset.extend(random.sample(short, min(num_each, len(short))))
        subset.extend(random.sample(medium, min(num_each, len(medium))))
        subset.extend(random.sample(long, min(num_each, len(long))))
        
        # If we need more to reach desired size, sample from all
        if len(subset) < self.max_targets:
            remaining = [(t, l) for t, l in targets_with_length if (t, l) not in subset]
            subset.extend(random.sample(remaining, min(self.max_targets - len(subset), len(remaining))))
            
        # Extract just the target IDs
        return [t for t, _ in subset[:self.max_targets]]
        
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.target_ids)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a dataset sample with test/train mode feature filtering.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary with features and coordinates
            
        Raises:
            ValueError: If target is missing required data
        """
        target_id = self.target_ids[idx]
        
        # Get features (filtered by test_mode flag)
        features = self.feature_loader.get_features(target_id)
        
        # Get coordinates
        coordinates = self.coord_loader.get_coordinates(target_id)
        
        if coordinates is None:
            raise ValueError(f"No coordinates found for {target_id}")
            
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
        
    def get_average_sequence_length(self) -> float:
        """Get average sequence length in this dataset."""
        total_length = 0
        count = 0
        
        for target_id in self.target_ids:
            seq_info = self.coord_loader.get_sequence(target_id)
            if seq_info:
                total_length += len(seq_info['sequence'])
                count += 1
                
        return total_length / max(1, count)
        
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for batching samples.
        
        Args:
            batch: List of samples to collate
            
        Returns:
            Batched samples with proper padding
        """
        # Get batch size and maximum sequence length
        batch_size = len(batch)
        max_len = max(sample["length"] for sample in batch)
        
        # Extract IDs
        ids = [sample["target_id"] for sample in batch]
        
        # Initialize output dictionary with required fields
        output = {
            "ids": ids,
            "target_id": ids,  # Duplicate for compatibility
            "lengths": torch.tensor([sample["length"] for sample in batch], dtype=torch.long),
        }
        
        # Required keys for the model - ensure all these are present
        required_keys = [
            "sequence_int",
            "pairing_probs",
            "positional_entropy",
            "coupling_matrix",
            "accessibility",
            "mask",
            "atom_positions",
            "atom_mask"
        ]
        
        # Add dihedral_features to required keys (it's already conditionally included in __getitem__)
        if "dihedral_features" in batch[0]:
            required_keys.append("dihedral_features")
        
        # Process each tensor in the batch
        for key in batch[0].keys():
            if key in ["target_id", "ids", "length", "sequence"]:
                continue  # Already processed or non-tensor
                
            if isinstance(batch[0][key], torch.Tensor):
                # Process tensor based on its shape
                sample_shape = batch[0][key].shape
                
                if len(sample_shape) == 1:
                    # 1D tensor (sequence, per-residue features)
                    padded = []
                    for sample in batch:
                        tensor = sample[key]
                        padded_tensor = torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device)
                        padded_tensor[:len(tensor)] = tensor
                        padded.append(padded_tensor)
                    output[key] = torch.stack(padded)
                    
                elif len(sample_shape) == 2 and sample_shape[0] == sample_shape[1]:
                    # 2D square matrix (L, L) like pairing_probs or coupling_matrix
                    padded = []
                    for sample in batch:
                        tensor = sample[key]
                        padded_tensor = torch.zeros(
                            (max_len, max_len), dtype=tensor.dtype, device=tensor.device
                        )
                        padded_tensor[:len(tensor), :len(tensor)] = tensor
                        padded.append(padded_tensor)
                    output[key] = torch.stack(padded)
                    
                elif len(sample_shape) == 2:
                    # 2D tensor with feature dimension (L, D) like dihedral_angles
                    feature_dim = sample_shape[1]
                    padded = []
                    for sample in batch:
                        tensor = sample[key]
                        padded_tensor = torch.zeros(
                            (max_len, feature_dim), dtype=tensor.dtype, device=tensor.device
                        )
                        padded_tensor[:len(tensor)] = tensor
                        padded.append(padded_tensor)
                    output[key] = torch.stack(padded)
        
        # Ensure all required keys are present
        for key in required_keys:
            if key not in output:
                logger.warning(f"Required key '{key}' not in output batch. Adding dummy tensor.")
                
                # Add appropriate dummy tensor based on key
                if key == "sequence_int":
                    output[key] = torch.zeros((batch_size, max_len), dtype=torch.long)
                elif key == "dihedral_features":
                    output[key] = torch.zeros((batch_size, max_len, 4), dtype=torch.float32)
                elif key in ["pairing_probs", "coupling_matrix"]:
                    output[key] = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
                elif key in ["positional_entropy", "accessibility"]:
                    output[key] = torch.zeros((batch_size, max_len), dtype=torch.float32)
                elif key in ["atom_positions"]:
                    output[key] = torch.zeros((batch_size, max_len, 3), dtype=torch.float32)
                elif key in ["mask", "atom_mask"]:
                    output[key] = torch.zeros((batch_size, max_len), dtype=torch.bool)
        
        return output


# Example usage
if __name__ == "__main__":
    # Try to find the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    
    # Test in both modes
    for test_mode in [True, False]:
        print(f"\n{'-'*20} {'TEST' if test_mode else 'TRAIN'} MODE {'-'*20}")
        
        # Try all validation tiers
        for subset_name in ["technical", "scientific", "comprehensive"]:
            print(f"\nTesting {subset_name} validation:")
            try:
                dataset = ValidationDataset(
                    data_dir=data_dir,
                    subset_name=subset_name,
                    test_mode=test_mode,
                    max_targets=3  # Small number for testing
                )
                
                print(f"Created dataset with {len(dataset)} samples")
                print(f"Average sequence length: {dataset.get_average_sequence_length():.1f}")
                
                if len(dataset) > 0:
                    # Try getting a sample
                    sample = dataset[0]
                    print("\nSample keys:")
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: {type(value)} with shape {value.shape}")
                        else:
                            print(f"  {key}: {type(value)}")
                    
                    # Try collating a batch
                    from torch.utils.data import DataLoader
                    dataloader = DataLoader(
                        dataset,
                        batch_size=2,
                        shuffle=False,
                        collate_fn=dataset.collate_fn
                    )
                    
                    if len(dataset) >= 2:
                        batch = next(iter(dataloader))
                        print("\nBatch shapes:")
                        for key, value in batch.items():
                            if isinstance(value, torch.Tensor):
                                print(f"  {key}: {value.shape}")
                            else:
                                print(f"  {key}: {type(value)}")
            except Exception as e:
                print(f"Error creating {subset_name} dataset: {e}")