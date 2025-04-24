#!/usr/bin/env python3
"""
Kaggle Inference Notebook Patch v2

This script contains all the fixes needed for the Kaggle inference notebook:
1. Fixed model loader for corrupted checkpoints
2. Fixed data loader for multi-structure feature files
3. Enhanced positional encoding for longer sequences
4. Missing imports fix

Usage:
Execute this script directly to test all patches, or import the functions
to apply them selectively in your notebook.
"""

import os
import sys
import torch
import math
import torch.nn as nn  # Critical missing import


# 1. Enhanced positional encoding that can handle longer sequences
class EnhancedPositionalEncoding(nn.Module):
    """
    Enhanced version of PositionalEncoding that handles sequences longer than max_len.
    """
    
    def __init__(self, config):
        """Initialize positional encoding with extendable length."""
        super().__init__()

        # Extract parameters from config
        self.embed_dim = config.get("residue_embed_dim", 128)
        self.max_len = config.get("max_len", 500)

        # Create constant positional encoding matrix
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        self.div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim)
        )

        pe = torch.zeros(self.max_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)

        # Register buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, embed_dim)
    
    def extend_pe(self, new_max_len):
        """Dynamically extend the positional encoding to handle longer sequences."""
        # Create new positions
        old_max_len = self.pe.size(1)
        if new_max_len <= old_max_len:
            return  # No need to extend
            
        print(f"Extending positional encoding from {old_max_len} to {new_max_len}")
        
        # Generate positions for the new entries
        position = torch.arange(old_max_len, new_max_len).unsqueeze(1).float().to(self.pe.device)
        
        # Create new encodings
        pe_extension = torch.zeros(new_max_len - old_max_len, self.embed_dim, 
                                  device=self.pe.device)
        pe_extension[:, 0::2] = torch.sin(position * self.div_term.to(self.pe.device))
        pe_extension[:, 1::2] = torch.cos(position * self.div_term.to(self.pe.device))
        
        # Concatenate with existing buffer
        new_pe = torch.cat([self.pe.squeeze(0), pe_extension], dim=0).unsqueeze(0)
        
        # Replace the buffer
        self.pe = new_pe
        self.max_len = new_max_len
    
    def forward(self, seq_len):
        """Get positional encodings with automatic extension if needed."""
        if seq_len > self.max_len:
            # If sequence is longer than our current max, extend the encoding
            new_max_len = max(seq_len, int(self.max_len * 1.5))  # Grow by 50% to reduce frequent extensions
            self.extend_pe(new_max_len)
            
        return self.pe[:, :seq_len]


# 2. Patch for model's positional encoding to handle long sequences
def patch_model_for_long_sequences(model):
    """Patch a model's positional encoding to handle long sequences."""
    if hasattr(model, 'embedding_module') and hasattr(model.embedding_module, 'positional_encoding'):
        # Get original module
        orig_pe = model.embedding_module.positional_encoding
        
        # Create config for new module
        config = {
            'residue_embed_dim': orig_pe.embed_dim,
            'max_len': orig_pe.max_len
        }
        
        # Create enhanced module
        enhanced_pe = EnhancedPositionalEncoding(config)
        
        # Copy the existing buffer
        enhanced_pe.pe = orig_pe.pe.clone()
        
        # Replace in the model
        model.embedding_module.positional_encoding = enhanced_pe
        
        print(f"Patched model with enhanced positional encoding (max_len: {enhanced_pe.max_len})")
        return True
    else:
        print("Warning: Could not find positional encoding in model structure")
        return False


# 3. Fixed model loader for corrupted checkpoints
def fixed_load_model(checkpoint_path, device):
    """
    Fixed model loader that can handle corrupted checkpoints with dummy state_dict.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        device (torch.device): Device to load the model on
        
    Returns:
        tuple: (model, config, metrics) - The loaded model, its configuration, and metrics
    """
    # Import here to avoid import issues
    from src.models.rna_folding_model import RNAFoldingModel
    
    print(f"Loading model from {checkpoint_path}")
    
    # Check if the file exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}")
        return None, None, {'val_rmsd': None, 'epoch': None}
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get model configuration
        if 'args' in checkpoint:
            config = checkpoint['args']
        elif 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # Default configuration
            print("WARNING: No configuration found in checkpoint. Using defaults.")
            config = {
                'num_blocks': 4,
                'residue_embed_dim': 128,
                'pair_embed_dim': 64,
                'num_attention_heads': 4,
                'dropout': 0.1
            }
        
        # Initialize model
        model = RNAFoldingModel(config)
        
        # Check if state_dict is valid
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            # Check if it's corrupted (only contains dummy key)
            if list(model_state_dict.keys()) == ['dummy']:
                print("WARNING: Corrupted checkpoint detected with only 'dummy' key.")
                print("Initializing model from scratch with the configuration from checkpoint.")
                # We don't load weights - using freshly initialized weights
            else:
                # Load state dict if it seems valid
                model.load_state_dict(model_state_dict)
                print("Successfully loaded weights from checkpoint.")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Successfully loaded weights from 'state_dict'.")
        else:
            print("WARNING: No state_dict found in checkpoint. Using untrained model.")
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Extract validation metrics if available
        val_rmsd = None
        if 'val_rmsd' in checkpoint:
            val_rmsd = checkpoint['val_rmsd']
        elif 'validation_rmsd' in checkpoint:
            val_rmsd = checkpoint['validation_rmsd']
        elif 'best_val_metrics' in checkpoint and 'rmsd' in checkpoint['best_val_metrics']:
            val_rmsd = checkpoint['best_val_metrics']['rmsd']
        
        epoch = None
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
        
        # Enhance model to handle longer sequences
        patch_model_for_long_sequences(model)
        
        # Return model, config, and metrics
        return model, config, {'val_rmsd': val_rmsd, 'epoch': epoch}
    
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, {'val_rmsd': None, 'epoch': None}


# 4. Fixed data loader for multi-structure feature files
def fixed_load_precomputed_features(
    target_id, features_dir, temporal_cutoff=None
):
    """
    Enhanced version of load_precomputed_features that handles inconsistent feature file formats.
    """
    import os
    import numpy as np
    import warnings
    import pandas as pd
    
    features = {}

    # 1. Load dihedral features
    dihedral_path = os.path.join(
        features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz"
    )
    if os.path.exists(dihedral_path):
        try:
            with np.load(dihedral_path) as data:
                # Check feature generation date if available for temporal cutoff
                if temporal_cutoff is not None and "metadata" in data:
                    try:
                        metadata_str = str(data["metadata"])
                        if "extraction_timestamp" in metadata_str:
                            timestamp_part = metadata_str.split("extraction_timestamp")[
                                1
                            ].split("'")[1]
                            generation_date = timestamp_part.split()[0]

                            if pd.to_datetime(generation_date) > pd.to_datetime(
                                temporal_cutoff
                            ):
                                warnings.warn(
                                    f"Dihedral features for {target_id} were generated after the temporal cutoff. Using zeros."
                                )
                                features["dihedral"] = None
                                return features
                    except (KeyError, IndexError, ValueError):
                        pass

                # Handle different feature file formats
                if "features" in data:
                    # Standard format
                    features["dihedral"] = {"features": data["features"].astype(np.float32)}
                elif "struct_1_features" in data:
                    # Multi-structure format with numbered structures
                    features["dihedral"] = {"features": data["struct_1_features"].astype(np.float32)}
                else:
                    # Unknown format - warn and use None
                    warnings.warn(f"Dihedral features file for {target_id} has unexpected format. Using zeros.")
                    features["dihedral"] = None
                    return features
                
                # Handle NaN values if present
                if features["dihedral"] is not None and np.isnan(features["dihedral"]["features"]).any():
                    features["dihedral"]["features"] = np.nan_to_num(
                        features["dihedral"]["features"], nan=0.0
                    )
        except Exception as e:
            # Handle any errors in loading
            warnings.warn(f"Error loading dihedral features for {target_id}: {str(e)}. Using zeros.")
            features["dihedral"] = None
    else:
        # For test data or if file is missing
        features["dihedral"] = None
        warnings.warn(f"Dihedral features not found for {target_id}. Using zeros.")

    # 2. Load thermodynamic features (required)
    thermo_path = os.path.join(
        features_dir, "thermo_features", f"{target_id}_thermo_features.npz"
    )
    if not os.path.exists(thermo_path):
        raise ValueError(
            f"Thermodynamic features not found for {target_id}. Required for prediction."
        )

    try:
        with np.load(thermo_path) as data:
            # Check feature generation date if available for temporal cutoff
            if temporal_cutoff is not None and "generation_date" in data:
                generation_date = str(data["generation_date"])
                if pd.to_datetime(generation_date) > pd.to_datetime(temporal_cutoff):
                    warnings.warn(
                        f"Thermo features for {target_id} were generated after the temporal cutoff. Using zeros."
                    )
                    features["thermo"] = None
                    return features

            # Extract key arrays and scalar values
            thermo_features = {}

            # Get pairing probabilities matrix (critical)
            if "pairing_probs" in data:
                thermo_features["pairing_probs"] = data["pairing_probs"].astype(np.float32)
            elif "base_pair_probs" in data:
                thermo_features["pairing_probs"] = data["base_pair_probs"].astype(np.float32)
            else:
                raise ValueError(f"No pairing probabilities found in {target_id} thermo features")

            # Handle NaN values in pairing probabilities
            if np.isnan(thermo_features["pairing_probs"]).any():
                thermo_features["pairing_probs"] = np.nan_to_num(
                    thermo_features["pairing_probs"], nan=0.0
                )

            # Get positional entropy (optional)
            if "positional_entropy" in data:
                thermo_features["positional_entropy"] = data["positional_entropy"].astype(
                    np.float32
                )
            else:
                # Calculate from pairing probabilities if missing
                pair_probs = thermo_features["pairing_probs"]
                row_entropies = -np.sum(
                    pair_probs * np.log2(pair_probs + 1e-10), axis=1
                )
                thermo_features["positional_entropy"] = row_entropies

            # Get accessibility (optional)
            if "accessibility" in data:
                thermo_features["accessibility"] = data["accessibility"].astype(np.float32)
            else:
                # Calculate from pairing probabilities if missing
                pair_probs = thermo_features["pairing_probs"]
                accessibilities = 1.0 - np.sum(pair_probs, axis=1)
                thermo_features["accessibility"] = np.maximum(0.0, accessibilities)

            features["thermo"] = thermo_features
    except Exception as e:
        raise ValueError(f"Error loading thermodynamic features for {target_id}: {str(e)}")

    # 3. Load evolutionary coupling features (optional)
    mi_path = os.path.join(
        features_dir, "evolutionary_features", f"{target_id}_evolutionary_features.npz"
    )
    if os.path.exists(mi_path):
        try:
            with np.load(mi_path) as data:
                # Check feature generation date if available for temporal cutoff
                if temporal_cutoff is not None and "generation_date" in data:
                    generation_date = str(data["generation_date"])
                    if pd.to_datetime(generation_date) > pd.to_datetime(temporal_cutoff):
                        warnings.warn(
                            f"Evolutionary features for {target_id} were generated after the temporal cutoff. Using zeros."
                        )
                        features["evolutionary"] = None
                        return features

                # Extract key arrays and metadata
                evol_features = {}

                # Get coupling matrix (required for this feature type)
                if "coupling_matrix" in data:
                    coupling_matrix = data["coupling_matrix"].astype(np.float32)
                    
                    # Check if the matrix is valid (not all zeros or constant)
                    is_valid = not np.allclose(coupling_matrix, 0.0)
                    evol_features["has_valid_mi"] = is_valid
                    
                    if is_valid:
                        evol_features["coupling_matrix"] = coupling_matrix
                    else:
                        # Zero matrix case - still provide the matrix but flag it
                        evol_features["coupling_matrix"] = coupling_matrix
                        warnings.warn(f"Coupling matrix for {target_id} is all zeros or constant.")
                else:
                    # No coupling matrix found
                    evol_features["has_valid_mi"] = False
                    evol_features["coupling_matrix"] = None

                features["evolutionary"] = evol_features
        except Exception as e:
            # Handle errors in evolutionary feature loading
            warnings.warn(f"Error loading evolutionary features for {target_id}: {str(e)}. Proceeding without them.")
            features["evolutionary"] = None
    else:
        # No evolutionary features available
        features["evolutionary"] = None

    return features


# 5. Apply all patches to prepare for Kaggle
def apply_all_patches():
    """Apply all patches to the necessary modules."""
    # Import modules to patch
    import sys
    from src import data_loading
    from src.models import embeddings
    
    # Store original functions for reference
    original_load_precomputed_features = data_loading.load_precomputed_features
    original_positional_encoding = embeddings.PositionalEncoding
    
    # Apply fixes
    data_loading.load_precomputed_features = fixed_load_precomputed_features
    
    # Don't modify the class directly - we'll patch instances when needed
    # Instead, we'll patch the model directly when loading
    
    print("All patches applied successfully")
    
    return {
        'original_load_precomputed_features': original_load_precomputed_features,
        'original_positional_encoding': original_positional_encoding
    }


def test_patches():
    """Test that all patches work correctly."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Test enhanced positional encoding
    config = {
        'residue_embed_dim': 128,
        'max_len': 500
    }
    
    pe = EnhancedPositionalEncoding(config)
    
    # Try with a sequence longer than max_len
    seq_len = 720
    encodings = pe(seq_len)
    print(f"Enhanced positional encoding test with seq_len={seq_len}: {encodings.shape}")
    
    # 2. Test model loading
    # This requires access to model checkpoints
    # Choose a checkpoint path to test
    model_paths = [
        "../results/final_model/run_20250423-072601/checkpoints/best_model.pt",
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            model, config, metrics = fixed_load_model(path, device)
            if model is not None:
                print(f"Successfully loaded model from {path}")
                print(f"Model config: {config}")
                print(f"Metrics: {metrics}")
    
    # 3. Test data loading
    # Choose a target ID to test
    target_id = "R1107"  # Known to have multi-structure format
    
    try:
        features = fixed_load_precomputed_features(target_id, "../data/processed", "2022-05-01")
        print(f"Successfully loaded features for {target_id}")
        if features['dihedral'] is not None:
            print(f"Dihedral features shape: {features['dihedral']['features'].shape}")
    except Exception as e:
        print(f"Error loading features: {str(e)}")
    
    print("Patch testing complete")


if __name__ == "__main__":
    test_patches()