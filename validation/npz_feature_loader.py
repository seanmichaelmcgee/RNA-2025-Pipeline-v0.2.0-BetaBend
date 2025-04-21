"""
NPZ Feature Loader for RNA 3D folding model validation.

This module implements the feature loading from NPZ files with test-mode filtering
capability for the dual-mode validation framework.
"""

import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NPZFeatureLoader")

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
    
    def __init__(self, data_dir=None, test_mode=True):
        """
        Initialize feature loader.
        
        Args:
            data_dir: Base directory for data. If None, will try to find it.
            test_mode: If True, excludes pseudo-dihedral features to match test conditions
        """
        # Find data directory if not provided
        if data_dir is None:
            data_dir = self._find_data_dir()
            
        self.data_dir = data_dir
        self.test_mode = test_mode
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Verify directories exist
        if not os.path.exists(self.processed_dir):
            logger.warning(f"Processed directory not found at {self.processed_dir}")
            # Try to create it if it doesn't exist
            try:
                os.makedirs(self.processed_dir, exist_ok=True)
                logger.info(f"Created processed directory at {self.processed_dir}")
            except Exception as e:
                logger.error(f"Failed to create processed directory: {e}")

        # Load valid target IDs from train_sequences.csv
        self.valid_target_ids = self._load_target_ids_from_csv()
        
        # Log initialization
        mode_str = "TEST-EQUIVALENT" if test_mode else "TRAINING-EQUIVALENT"
        logger.info(f"Initialized NPZFeatureLoader in {mode_str} mode with data_dir={data_dir}")
        
    def _load_target_ids_from_csv(self):
        """
        Load target IDs from appropriate sequences CSV file.
        
        Order of priority:
        1. validation_sequences.csv (for validation mode)
        2. train_sequences.csv (for training mode)
        3. test_sequences.csv (for submission mode)
        """
        # Try validation_sequences.csv first (for validation mode)
        csv_path = os.path.join(self.data_dir, "raw", "validation_sequences.csv")
        
        # If validation sequences don't exist, try train_sequences.csv
        if not os.path.exists(csv_path):
            csv_path = os.path.join(self.data_dir, "raw", "train_sequences.csv")
            
            # If train sequences don't exist, try test_sequences.csv
            if not os.path.exists(csv_path):
                csv_path = os.path.join(self.data_dir, "raw", "test_sequences.csv")
                
                # If none of the sequence files exist
                if not os.path.exists(csv_path):
                    logger.warning("No sequence files found: validation_sequences.csv, train_sequences.csv, or test_sequences.csv")
                    return []
        
        try:
            df = pd.read_csv(csv_path)
            if 'target_id' not in df.columns:
                logger.warning(f"No 'target_id' column found in {csv_path}")
                return []
                
            # Extract unique target IDs
            target_ids = df['target_id'].unique().tolist()
            logger.info(f"Loaded {len(target_ids)} target IDs from {os.path.basename(csv_path)}")
            return target_ids
        except Exception as e:
            logger.error(f"Error loading target IDs from {csv_path}: {e}")
            return []
    
    def _find_data_dir(self):
        """Find the data directory using multiple strategies with path objects for better portability."""
        # Get script directory and convert to Path for better manipulation
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        
        # Try common locations relative to script path and working directory
        possible_dirs = [
            project_root / "data",             # Most likely location (project_root/data)
            script_dir / "data",               # In validation directory
            Path.cwd() / "data",               # Current working directory
            Path.cwd().parent / "data",        # Parent of current directory
            Path.cwd().parent.parent / "data", # Grandparent of current directory
        ]
        
        for dir_path in possible_dirs:
            if dir_path.exists():
                logger.info(f"Found data directory: {dir_path}")
                return str(dir_path)
        
        # Use the project root's data directory as a last resort
        fallback_dir = project_root / "data"
        logger.warning(f"No data directory found, using fallback: {fallback_dir}")
        return str(fallback_dir)
            
    def get_features(self, target_id):
        """
        Get features for a target, with test mode filtering.
        
        Args:
            target_id: Target ID string
            
        Returns:
            Dictionary with loaded features
            
        Raises:
            ValueError: If required features cannot be loaded
        """
        # First check if the target ID is valid or needs expansion
        if self.valid_target_ids and target_id not in self.valid_target_ids:
            # Try to find an expanded version of this ID (e.g., 1A51 -> 1A51_A)
            expanded_ids = [tid for tid in self.valid_target_ids if tid.startswith(target_id + "_")]
            if expanded_ids:
                logger.info(f"Using expanded ID {expanded_ids[0]} for requested ID {target_id}")
                target_id = expanded_ids[0]
            else:
                logger.warning(f"Target ID {target_id} not found in valid target list")
        
        features = {}
        
        # Always load sequence info (from thermo features)
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
            logger.warning(f"Thermodynamic features missing for {target_id}")
            # Since thermo features are critical, we'll create minimal placeholders
            if "sequence" in features:
                sequence_length = len(features["sequence"])
                features["positional_entropy"] = torch.zeros(sequence_length, dtype=torch.float32)
                features["pairing_probs"] = torch.zeros((sequence_length, sequence_length), dtype=torch.float32)
            
        # Always load MI features (available in both modes)
        mi_features = self._load_mi_features(target_id)
        if mi_features:
            features.update(mi_features)
        else:
            logger.warning(f"MI features missing for {target_id}, using zeros")
            # Create zero-filled MI matrix as fallback
            if "sequence" in features:
                sequence_length = len(features["sequence"])
                features["coupling_matrix"] = torch.zeros((sequence_length, sequence_length), dtype=torch.float32)
            
        # Only load dihedral features in train mode
        if not self.test_mode:
            dihedral_features = self._load_dihedral_features(target_id)
            if dihedral_features:
                features.update(dihedral_features)
            else:
                logger.warning(f"Dihedral features missing for {target_id} in train mode")
                # Create placeholder dihedral angles in train mode
                if "sequence" in features:
                    sequence_length = len(features["sequence"])
                    features["dihedral_angles"] = torch.zeros((sequence_length, 4), dtype=torch.float32)
        
        return features

    def _load_sequence_info(self, target_id):
        """
        Load sequence from thermo features (available in both modes).
        
        Args:
            target_id: Target ID string
            
        Returns:
            Dictionary with sequence information or None if not found
        """
        thermo_path = os.path.join(self.processed_dir, "thermo_features", f"{target_id}_thermo_features.npz")
        if not os.path.exists(thermo_path):
            logger.warning(f"Thermo features file not found: {thermo_path}")
            return None
            
        try:
            with np.load(thermo_path, allow_pickle=True) as data:
                if "sequence" not in data:
                    logger.warning(f"No sequence found in {thermo_path}")
                    return None
                    
                sequence = str(data["sequence"])
                
                # Create sequence integer mapping (AUCG -> 0123)
                seq_map = {'A': 0, 'U': 1, 'C': 2, 'G': 3, 'N': 4}
                sequence_int = torch.tensor([seq_map.get(base, 4) for base in sequence], dtype=torch.long)
                
                return {
                    "target_id": str(data["target_id"]) if "target_id" in data else target_id,
                    "sequence": sequence,
                    "sequence_int": sequence_int,
                    "length": len(sequence),
                    "mask": torch.ones(len(sequence), dtype=torch.bool),
                    "angle_mask": torch.ones(len(sequence), dtype=torch.bool),
                }
        except Exception as e:
            logger.error(f"Error loading sequence for {target_id}: {e}")
            return None
            
    def _load_thermo_features(self, target_id):
        """
        Load thermodynamic features (available in both modes).
        
        Args:
            target_id: Target ID string
            
        Returns:
            Dictionary with thermodynamic features or None if not found
        """
        thermo_path = os.path.join(self.processed_dir, "thermo_features", f"{target_id}_thermo_features.npz")
        if not os.path.exists(thermo_path):
            logger.warning(f"Thermo features file not found: {thermo_path}")
            return None
            
        try:
            with np.load(thermo_path, allow_pickle=True) as data:
                # Check if we have the required thermodynamic features
                required_fields = ["positional_entropy", "pairing_probs"]
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    logger.warning(f"Missing required thermo fields in {thermo_path}: {missing_fields}")
                    return None
                
                # Get sequence length either from length field or positional_entropy shape
                if "length" in data:
                    sequence_length = int(data["length"])
                else:
                    sequence_length = data["positional_entropy"].shape[0]
                
                # Convert all features to PyTorch tensors
                features = {}
                
                # Per-position features
                if "positional_entropy" in data:
                    features["positional_entropy"] = torch.tensor(data["positional_entropy"], dtype=torch.float32)
                
                # Matrix features
                if "pairing_probs" in data:
                    features["pairing_probs"] = torch.tensor(data["pairing_probs"], dtype=torch.float32)
                
                # Global features - add as needed
                global_features = ["mfe", "gc_content", "ensemble_energy", "energy_gap"]
                for feat in global_features:
                    if feat in data:
                        features[feat] = torch.tensor(float(data[feat]), dtype=torch.float32)
                
                # Add structure information if needed
                if "structure" in data:
                    features["structure"] = str(data["structure"])
                    
                return features
        except Exception as e:
            logger.error(f"Error loading thermo features for {target_id}: {e}")
            return None
            
    def _load_mi_features(self, target_id):
        """
        Load mutual information features (available in both modes).
        
        Args:
            target_id: Target ID string
            
        Returns:
            Dictionary with MI features or None if not found
        """
        mi_path = os.path.join(self.processed_dir, "mi_features", f"{target_id}_mi_features.npz")
        if not os.path.exists(mi_path):
            logger.warning(f"MI features file not found: {mi_path}")
            return None
            
        try:
            with np.load(mi_path, allow_pickle=True) as data:
                # Check if we have the coupling matrix
                if "coupling_matrix" not in data:
                    logger.warning(f"No coupling_matrix found in {mi_path}")
                    return None
                
                features = {
                    "coupling_matrix": torch.tensor(data["coupling_matrix"], dtype=torch.float32),
                }
                
                # Add method information if needed for debugging
                if "method" in data:
                    features["mi_method"] = str(data["method"])
                    
                return features
        except Exception as e:
            logger.error(f"Error loading MI features for {target_id}: {e}")
            return None
            
    def _load_dihedral_features(self, target_id):
        """
        Load dihedral features (only available in train mode).
        
        Args:
            target_id: Target ID string
            
        Returns:
            Dictionary with dihedral features or None if not found or in test mode
        """
        # Skip completely if in test mode
        if self.test_mode:
            return None
            
        dihedral_path = os.path.join(self.processed_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
        if not os.path.exists(dihedral_path):
            logger.warning(f"Dihedral features file not found: {dihedral_path}")
            return None
            
        try:
            with np.load(dihedral_path, allow_pickle=True) as data:
                # Get a list of all keys
                keys = list(data.keys())
                
                # First try the simple case - direct 'features' key
                if "features" in data:
                    dihedral_angles = torch.tensor(data["features"], dtype=torch.float32)
                else:
                    # Look for structure-specific features (struct_*_features)
                    feature_keys = [k for k in keys if k.startswith("struct_") and k.endswith("_features")]
                    
                    if not feature_keys:
                        logger.warning(f"No feature arrays found in {dihedral_path}")
                        return None
                    
                    # Use the first structure by default (usually struct_1_features)
                    # Could be enhanced to select the best structure based on completeness
                    struct_key = feature_keys[0]
                    logger.info(f"Using dihedral features from {struct_key} for {target_id}")
                    
                    # Extract dihedral features (sin/cos encoded)
                    dihedral_angles = torch.tensor(data[struct_key], dtype=torch.float32)
                
                # Verify shape - should be (L, 4)
                if dihedral_angles.dim() != 2 or dihedral_angles.size(1) != 4:
                    logger.warning(f"Unexpected dihedral angles shape: {dihedral_angles.shape}")
                
                # Initialize features dictionary
                features = {
                    "dihedral_angles": dihedral_angles,  # (L, 4) with sin/cos encoded angles
                }
                
                # Add raw angles if available (using direct keys or struct-specific keys)
                if "eta" in data and "theta" in data:
                    features["eta"] = torch.tensor(data["eta"], dtype=torch.float32)
                    features["theta"] = torch.tensor(data["theta"], dtype=torch.float32)
                else:
                    # Try finding structure-specific angle keys
                    eta_key = struct_key.replace("_features", "_eta") if "struct_" in struct_key else None
                    theta_key = struct_key.replace("_features", "_theta") if "struct_" in struct_key else None
                    
                    if eta_key in data and theta_key in data:
                        eta_values = data[eta_key]
                        theta_values = data[theta_key]
                        
                        # Convert to tensor, handling NaN values
                        eta_tensor = torch.tensor(np.nan_to_num(eta_values, nan=0.0), dtype=torch.float32)
                        theta_tensor = torch.tensor(np.nan_to_num(theta_values, nan=0.0), dtype=torch.float32)
                        
                        features["eta"] = eta_tensor
                        features["theta"] = theta_tensor
                
                return features
        except Exception as e:
            logger.error(f"Error loading dihedral features for {target_id}: {e}")
            return None

    def list_available_targets(self):
        """
        List all available RNA targets with necessary features.
        
        Returns:
            List of target IDs that have the required features
        """
        # If we have already loaded target IDs from CSV, prioritize those
        if self.valid_target_ids:
            return self._filter_ids_with_features(self.valid_target_ids)
        
        # Fallback to scanning directory if no CSV data available
        thermo_dir = os.path.join(self.processed_dir, "thermo_features")
        if not os.path.exists(thermo_dir):
            logger.warning(f"Thermo features directory not found: {thermo_dir}")
            return []
        
        # Get all NPZ files from thermo_features directory
        npz_files = [f for f in os.listdir(thermo_dir) if f.endswith('.npz')]
        
        # Extract unique target IDs
        targets = set()
        for filename in npz_files:
            # Remove feature type suffix to get target ID
            if '_thermo_features.npz' in filename:
                target_id = filename.replace('_thermo_features.npz', '')
                targets.add(target_id)
        
        return self._filter_ids_with_features(list(targets))
    
    def _filter_ids_with_features(self, target_ids):
        """Filter target IDs to those with required features available."""
        valid_targets = []
        
        for target_id in target_ids:
            # In test mode, we need thermo and MI features
            thermo_path = os.path.join(self.processed_dir, "thermo_features", f"{target_id}_thermo_features.npz")
            mi_path = os.path.join(self.processed_dir, "mi_features", f"{target_id}_mi_features.npz")
            
            if not os.path.exists(thermo_path):
                continue
            
            # In train mode, we also need dihedral features
            if not self.test_mode:
                dihedral_path = os.path.join(self.processed_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
                if not os.path.exists(dihedral_path):
                    continue
            
            valid_targets.append(target_id)
        
        return valid_targets
            
    def get_feature_stats(self, target_ids=None):
        """
        Get statistics about feature availability for target IDs.
        
        Args:
            target_ids: List of target IDs to check. If None, checks all targets.
            
        Returns:
            Dictionary with feature availability statistics
        """
        if target_ids is None:
            target_ids = self.list_available_targets()
            
        total_targets = len(target_ids)
        stats = {
            "total_targets": total_targets,
            "with_thermo": 0,
            "with_mi": 0,
            "with_dihedral": 0,
            "with_all_required": 0,
            "mode": "test" if self.test_mode else "train"
        }
        
        for target_id in target_ids:
            thermo_path = os.path.join(self.processed_dir, "thermo_features", f"{target_id}_thermo_features.npz")
            mi_path = os.path.join(self.processed_dir, "mi_features", f"{target_id}_mi_features.npz")
            dihedral_path = os.path.join(self.processed_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
            
            has_thermo = os.path.exists(thermo_path)
            has_mi = os.path.exists(mi_path)
            has_dihedral = os.path.exists(dihedral_path)
            
            if has_thermo:
                stats["with_thermo"] += 1
                
            if has_mi:
                stats["with_mi"] += 1
                
            if has_dihedral:
                stats["with_dihedral"] += 1
                
            # In test mode, we need thermo and MI features
            if self.test_mode and has_thermo and has_mi:
                stats["with_all_required"] += 1
                
            # In train mode, we need all three feature types
            elif not self.test_mode and has_thermo and has_mi and has_dihedral:
                stats["with_all_required"] += 1
                
        return stats


# Example usage
if __name__ == "__main__":
    # Try to find the processed directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    
    # Test in both modes
    for test_mode in [True, False]:
        print(f"\n{'-'*20} {'TEST' if test_mode else 'TRAIN'} MODE {'-'*20}")
        loader = NPZFeatureLoader(data_dir=data_dir, test_mode=test_mode)
        
        # List available targets
        targets = loader.list_available_targets()
        print(f"Found {len(targets)} valid targets")
        
        # Print feature stats
        stats = loader.get_feature_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
            
        # Try loading features for the first target if available
        if targets:
            first_target = targets[0]
            print(f"\nLoading features for {first_target}:")
            try:
                features = loader.get_features(first_target)
                print("Successfully loaded features:")
                for key, value in features.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {type(value)} with shape {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            except Exception as e:
                print(f"Error loading features: {e}")
        else:
            print("No targets available to test loading")