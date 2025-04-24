#!/usr/bin/env python3
"""
Minimal training script that successfully imports all necessary modules.
"""
import os
import sys
from pathlib import Path

# Get the project root directory
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent

# Add project root to Python path
sys.path.insert(0, str(project_root))

# Add src to Python path 
sys.path.insert(0, str(project_root / "src"))

# Print paths
print(f"Project root: {project_root}")
print(f"Python path:")
for p in sys.path:
    print(f"  {p}")

# Import modules 
print("\nImporting modules...")

# Import model
try:
    from src.models.rna_folding_model import RNAFoldingModel
    print("✓ Successfully imported RNAFoldingModel")
except ImportError as e:
    print(f"✗ Failed to import RNAFoldingModel: {e}")

# Import data loading
try:
    from src.data_loading_fixed import RNADataset, collate_fn, create_data_loader
    print("✓ Successfully imported data loading modules")
except ImportError as e:
    print(f"✗ Failed to import data loading modules: {e}")

# Import losses
try:
    from src.losses import compute_stable_fape_loss, compute_confidence_loss, compute_angle_loss
    print("✓ Successfully imported loss functions")
except ImportError as e:
    print(f"✗ Failed to import loss functions: {e}")

# Try to create a model
print("\nTesting model creation...")
try:
    model_config = {
        'num_blocks': 3,
        'residue_embed_dim': 128,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'ff_dim': 256,
        'dropout': 0.1,
    }
    model = RNAFoldingModel(model_config)
    print(f"✓ Successfully created model: {type(model)}")
except Exception as e:
    print(f"✗ Failed to create model: {e}")

print("\nAll imports completed!")