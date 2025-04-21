"""
CSV Coordinate Loader for RNA 3D folding model validation.

This module provides functionality to load 3D coordinates directly from CSV files,
supporting the dual-mode validation framework.
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CSVCoordinateLoader")

class CSVCoordinateLoader:
    """
    Loader specifically for 3D coordinates from CSV files.
    
    This loader focuses purely on extracting atom coordinates for structure
    evaluation, separating this functionality from feature loading.
    """
    
    def __init__(self, data_dir=None, split="validation"):
        """
        Initialize the coordinate loader.
        
        Args:
            data_dir: Path to the data directory. If None, will try to find it.
            split: Data split to use ("train", "validation", or "test")
        """
        # Find data directory if not provided
        if data_dir is None:
            data_dir = self._find_data_dir()
            
        self.data_dir = data_dir
        self.split = split
        
        # Raw data directory for CSV files
        self.raw_dir = os.path.join(data_dir, "raw")
        
        # Load sequences and labels
        self.sequences_df = self._load_sequences()
        self.labels_df = self._load_labels()
        
        # Create a mapping of target IDs to their indices
        self.target_ids = list(self.sequences_df['target_id'].unique()) if self.sequences_df is not None else []
        
        logger.info(f"Loaded {len(self.target_ids)} sequences for {split} split")
        if self.target_ids:
            logger.info(f"First few IDs: {', '.join(self.target_ids[:5])}")
        
    def _find_data_dir(self):
        """Find the data directory using multiple strategies."""
        # Try common locations
        possible_dirs = [
            os.path.join(os.getcwd(), "data"),
            os.path.join(os.getcwd(), "..", "data"),
            os.path.join(os.getcwd(), "..", "..", "data"),
            os.path.join(os.path.dirname(os.getcwd()), "data"),
        ]
        
        for dir_path in possible_dirs:
            raw_dir = os.path.join(dir_path, "raw")
            if os.path.exists(raw_dir):
                csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
                if csv_files:
                    logger.info(f"Found data directory: {dir_path}")
                    return dir_path
        
        # Use the current working directory's parent as a last resort
        return os.path.join(os.getcwd(), "..", "data")
    
    def _load_sequences(self):
        """
        Load sequence data from CSV.
        
        Returns:
            DataFrame with sequence data or None if file not found
        """
        file_path = os.path.join(self.raw_dir, f"{self.split}_sequences.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Sequence file not found: {file_path}")
            return None
            
        return pd.read_csv(file_path)
    
    def _load_labels(self):
        """
        Load label data (3D coordinates) from CSV.
        
        Returns:
            DataFrame with coordinate data or None if file not found
        """
        file_path = os.path.join(self.raw_dir, f"{self.split}_labels.csv")
        
        if not os.path.exists(file_path):
            if self.split == "test":
                # Test split typically doesn't have labels
                logger.info("No labels file found for test split (expected)")
                return None
            else:
                logger.warning(f"Labels file not found: {file_path}")
                return None
            
        return pd.read_csv(file_path)
    
    def get_sequence(self, target_id):
        """
        Get the sequence for a specific target ID.
        
        Args:
            target_id: Target ID string
            
        Returns:
            Dictionary with sequence information or None if not found
            
        Raises:
            ValueError: If target ID not found
        """
        if self.sequences_df is None:
            logger.warning("No sequence data available")
            return None
            
        if target_id not in self.target_ids:
            raise ValueError(f"Target ID not found: {target_id}")
            
        sequence_row = self.sequences_df[self.sequences_df['target_id'] == target_id].iloc[0]
        return {
            'target_id': target_id,
            'sequence': sequence_row['sequence'],
            'description': sequence_row['description'] if 'description' in sequence_row else None
        }
    
    def get_coordinates(self, target_id):
        """
        Get the 3D coordinates for a specific target ID.
        
        Args:
            target_id: Target ID string
            
        Returns:
            Dictionary with coordinate data or None if not found
            
        Notes:
            Handles multiple coordinate sets by returning all available conformations.
        """
        if self.labels_df is None:
            logger.warning("No coordinate data available")
            return None
            
        # Filter by target ID (which might have a format like TARGET_CHAIN_RESIDUE)
        target_prefix = f"{target_id}_"
        coords_df = self.labels_df[self.labels_df['ID'].str.startswith(target_prefix)]
        
        if coords_df.empty:
            logger.warning(f"No coordinates found for {target_id}")
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
        
        # Add residue information if available
        if 'residue_ids' in conformations[target_id]:
            result['residue_ids'] = conformations[target_id]['residue_ids']
            
        if 'residue_names' in conformations[target_id]:
            result['residue_names'] = conformations[target_id]['residue_names']
        
        # Add alternate conformations if present
        if len(conformations) > 1:
            result['alternate_conformations'] = {
                conf_id: conf_data 
                for conf_id, conf_data in conformations.items() 
                if conf_id != target_id
            }
        
        return result
    
    def _extract_atom_positions(self, coords_df):
        """
        Extract atom positions from coordinate DataFrame, handling multiple coordinate sets.
        
        Args:
            coords_df: DataFrame with coordinate data
            
        Returns:
            Dictionary with atom positions and masks, including alternate conformations
        """
        # Extract target_id (shared part before underscore in ID column)
        id_parts = coords_df['ID'].iloc[0].split('_')
        target_id = id_parts[0] if len(id_parts) > 1 else coords_df['ID'].iloc[0]
        
        # Basic position data
        position_data = {}
        
        # Add residue information if available
        if 'resname' in coords_df.columns:
            position_data['residue_names'] = coords_df['resname'].values
            
        if 'resid' in coords_df.columns:
            position_data['residue_ids'] = coords_df['resid'].values
        
        # Identify all coordinate sets available (x_1/y_1/z_1 through x_40/y_40/z_40)
        coordinate_sets = []
        
        # Determine available coordinate columns
        x_cols = [col for col in coords_df.columns if col.startswith('x_')]
        if not x_cols:
            logger.warning(f"No coordinate columns found for {target_id}")
            return self._create_empty_position_data(len(coords_df))
            
        # Find all atom indices
        atom_indices = []
        for col in x_cols:
            try:
                atom_idx = int(col.split('_')[1])
                atom_indices.append(atom_idx)
            except (IndexError, ValueError):
                continue
                
        atom_indices = sorted(set(atom_indices))
        
        if not atom_indices:
            logger.warning(f"No valid atom indices found for {target_id}")
            return self._create_empty_position_data(len(coords_df))
            
        # Process each coordinate set (atom)
        for atom_idx in atom_indices:
            x_col = f'x_{atom_idx}'
            y_col = f'y_{atom_idx}'
            z_col = f'z_{atom_idx}'
            
            if all(col in coords_df.columns for col in [x_col, y_col, z_col]):
                # Get coordinates for this atom
                positions = np.array([
                    coords_df[x_col].values,
                    coords_df[y_col].values,
                    coords_df[z_col].values
                ]).T
                
                # Filter out invalid coordinates (typically -1e+18 placeholder)
                valid_mask = np.all(np.abs(positions) < 1e17, axis=1)
                valid_count = np.sum(valid_mask)
                
                if valid_count > 0:
                    # Store this coordinate set
                    coordinate_sets.append({
                        'atom_idx': atom_idx,
                        'positions': positions,
                        'valid_mask': valid_mask,
                        'valid_count': valid_count
                    })
        
        # If no valid coordinate sets found
        if not coordinate_sets:
            logger.warning(f"No valid coordinate sets found for {target_id}")
            return self._create_empty_position_data(len(coords_df))
            
        # Sort coordinate sets by number of valid positions (descending)
        coordinate_sets.sort(key=lambda x: x['valid_count'], reverse=True)
        
        # Use the most complete set as primary
        primary_set = coordinate_sets[0]
        logger.info(f"Using coordinate set {primary_set['atom_idx']} as primary for {target_id} "
                   f"({primary_set['valid_count']}/{len(primary_set['valid_mask'])} valid positions)")
        
        # Create result with primary set
        result = {
            'atom_positions': torch.tensor(primary_set['positions'], dtype=torch.float32),
            'atom_mask': torch.tensor(primary_set['valid_mask'], dtype=torch.bool),
            'primary_atom_index': primary_set['atom_idx']
        }
        
        # Add alternate coordinate sets if available
        if len(coordinate_sets) > 1:
            alternate_sets = {}
            for cs in coordinate_sets[1:]:
                alt_key = f"atom_{cs['atom_idx']}"
                alternate_sets[alt_key] = {
                    'positions': torch.tensor(cs['positions'], dtype=torch.float32),
                    'valid_mask': torch.tensor(cs['valid_mask'], dtype=torch.bool),
                    'valid_count': cs['valid_count']
                }
            result['alternate_conformations'] = alternate_sets
            logger.info(f"Added {len(alternate_sets)} alternate coordinate sets for {target_id}")
        
        return result
        
    def _create_empty_position_data(self, length):
        """Create empty position data when no valid coordinates are found."""
        return {
            'atom_positions': torch.zeros((length, 3), dtype=torch.float32),
            'atom_mask': torch.zeros(length, dtype=torch.bool)
        }
    
    def list_available_targets(self):
        """
        List all available RNA targets with coordinate data.
        
        Returns:
            List of target IDs with coordinates
        """
        return self.target_ids
    
    def get_coordinate_stats(self, target_ids=None):
        """
        Get statistics about coordinate availability.
        
        Args:
            target_ids: List of target IDs to check. If None, checks all targets.
            
        Returns:
            Dictionary with coordinate statistics
        """
        if target_ids is None:
            target_ids = self.target_ids
            
        total_targets = len(target_ids)
        stats = {
            "total_targets": total_targets,
            "with_coordinates": 0,
            "with_multiple_conformations": 0,
            "average_atoms_per_target": 0.0,
            "split": self.split
        }
        
        total_atoms = 0
        
        for target_id in target_ids:
            coords = self.get_coordinates(target_id)
            if coords is None:
                continue
                
            stats["with_coordinates"] += 1
            
            if "alternate_conformations" in coords:
                stats["with_multiple_conformations"] += 1
                
            # Count atoms
            if "atom_positions" in coords:
                num_atoms = coords["atom_positions"].shape[0]
                total_atoms += num_atoms
                
        if stats["with_coordinates"] > 0:
            stats["average_atoms_per_target"] = total_atoms / stats["with_coordinates"]
            
        return stats


# Example usage
if __name__ == "__main__":
    # Try to find the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    
    # Test for validation split
    loader = CSVCoordinateLoader(data_dir=data_dir, split="validation")
    
    # List available targets
    targets = loader.list_available_targets()
    print(f"Found {len(targets)} targets with sequence data")
    
    # Print coordinate stats
    stats = loader.get_coordinate_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
        
    # Try loading coordinates for the first target if available
    if targets:
        first_target = targets[0]
        print(f"\nLoading coordinates for {first_target}:")
        try:
            coords = loader.get_coordinates(first_target)
            if coords:
                print("Successfully loaded coordinates:")
                for key, value in coords.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: {type(value)} with shape {value.shape}")
                    else:
                        print(f"  {key}: {type(value)}")
            else:
                print("No coordinates available for this target")
        except Exception as e:
            print(f"Error loading coordinates: {e}")
    else:
        print("No targets available to test loading")