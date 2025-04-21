#!/usr/bin/env python
"""
Debug script for feature loading in the RNA 3D folding validation framework.
This script helps diagnose issues with loading features from NPZ files.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.data_loading import load_precomputed_features

def verify_data_paths(data_dir):
    """Verify that all necessary data directories exist."""
    dirs_to_check = [
        os.path.join(data_dir, "dihedral_features"),
        os.path.join(data_dir, "thermo_features"),
        os.path.join(data_dir, "mi_features")
    ]
    
    all_exist = True
    for dir_path in dirs_to_check:
        if not os.path.exists(dir_path):
            print(f"ERROR: Directory not found: {dir_path}")
            all_exist = False
        else:
            # Count files to verify content
            files = [f for f in os.listdir(dir_path) if f.endswith('.npz')]
            print(f"Found {len(files)} files in {dir_path}")
    
    return all_exist

def debug_feature_path(target_id, feature_type, data_dir):
    """Print and verify the exact path being used to load features."""
    if feature_type == "dihedral":
        path = os.path.join(data_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
    elif feature_type == "thermo":
        path = os.path.join(data_dir, "thermo_features", f"{target_id}_thermo_features.npz")
    elif feature_type == "mi":
        path = os.path.join(data_dir, "mi_features", f"{target_id}_mi_features.npz")
    else:
        path = None
        
    if path:
        exists = os.path.exists(path)
        print(f"Path for {target_id} {feature_type}: {path}")
        print(f"File exists: {exists}")
        
        if exists:
            # Try to open and list contents
            try:
                with np.load(path) as data:
                    print(f"File contains keys: {list(data.keys())}")
                    
                    # Check shapes of key elements
                    if feature_type == "dihedral" and "features" in data:
                        print(f"Dihedral features shape: {data['features'].shape}")
                    elif feature_type == "thermo" and "pairing_probs" in data:
                        print(f"Pairing probs shape: {data['pairing_probs'].shape}")
                    elif feature_type == "mi" and "coupling_matrix" in data:
                        print(f"Coupling matrix shape: {data['coupling_matrix'].shape}")
            except Exception as e:
                print(f"Error opening file: {e}")
    
    return path, exists

def inspect_npz_file(file_path):
    """Inspect an NPZ file and return detailed information about its contents."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
        
    try:
        with np.load(file_path) as data:
            info = {
                "keys": list(data.keys()),
                "shapes": {k: data[k].shape for k in data.keys()},
                "dtypes": {k: str(data[k].dtype) for k in data.keys()},
                "file_size": os.path.getsize(file_path) / 1024,  # KB
            }
            return info
    except Exception as e:
        return f"Error inspecting file: {e}"

def test_feature_loading(target_id, data_dir):
    """Test loading features for a specific target ID."""
    print(f"\n=== Testing feature loading for {target_id} ===")
    
    # Check if all directories exist
    verify_data_paths(data_dir)
    
    # Check paths for each feature type
    for feature_type in ["dihedral", "thermo", "mi"]:
        path, exists = debug_feature_path(target_id, feature_type, data_dir)
        
        if exists:
            print(f"\nDetailed inspection of {feature_type} file:")
            info = inspect_npz_file(path)
            if isinstance(info, dict):
                print(f"  Keys: {info['keys']}")
                print(f"  Shapes: {info['shapes']}")
                print(f"  File size: {info['file_size']:.2f} KB")
            else:
                print(f"  {info}")
    
    # Try using the original function that's failing
    print("\nAttempting to load features with load_precomputed_features:")
    try:
        features = load_precomputed_features(target_id, data_dir)
        print("SUCCESS: Features loaded successfully")
        
        # Check what's in the features
        if "dihedral" in features and features["dihedral"] is not None:
            print(f"Dihedral features available: {list(features['dihedral'].keys())}")
        else:
            print("No dihedral features loaded")
            
        if "thermo" in features and features["thermo"] is not None:
            print(f"Thermo features available: {list(features['thermo'].keys())}")
        else:
            print("No thermo features loaded")
            
        if "evolutionary" in features and features["evolutionary"] is not None:
            print(f"Evolutionary features available: {list(features['evolutionary'].keys())}")
        else:
            print("No evolutionary features loaded")
            
    except Exception as e:
        print(f"ERROR: Failed to load features: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use relative path from project root
    data_dir = os.path.join(project_root, "data", "processed")
    
    # Check if a target ID was provided
    if len(sys.argv) > 1:
        target_id = sys.argv[1]
    else:
        # Default target ID for testing
        target_id = "6C4H_x"  # The one that failed in the notebook
    
    print(f"Using data directory: {data_dir}")
    test_feature_loading(target_id, data_dir)