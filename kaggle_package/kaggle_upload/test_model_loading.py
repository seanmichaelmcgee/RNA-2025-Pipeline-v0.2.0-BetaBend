#!/usr/bin/env python3
"""
Test Model Loading Script

This script tests loading the dummy model checkpoints to verify the 
fixed_load_model function is working properly.
"""

import os
import sys
import torch

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
kaggle_env_dir = current_dir  # This script is at the root of the kaggle_upload directory

# Add paths for Kaggle-like environment
src_path = os.path.join(kaggle_env_dir, "rna-model-src")
if src_path not in sys.path:
    sys.path.append(src_path)
    print(f"Added {src_path} to Python path")

# Try importing required modules
try:
    from src.models.rna_folding_model import RNAFoldingModel
    from src.utils.model_loader import fixed_load_model, test_model_inference
    print("✅ Successfully imported required modules")
except ImportError as e:
    print(f"❌ Error importing modules: {str(e)}")
    print("Check that rna-model-src contains all required source files")
    sys.exit(1)

def test_model(model_path, device):
    """Test loading and running inference with a model."""
    print(f"\nTesting model: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        return False
    
    # Load model
    try:
        model, config, metrics = fixed_load_model(model_path, device)
        if model is None:
            print(f"❌ Failed to load model from {model_path}")
            return False
        
        print(f"✅ Successfully loaded model from {model_path}")
        print(f"   Configuration: {config}")
        print(f"   Validation RMSD: {metrics['val_rmsd']}")
        print(f"   Trained epochs: {metrics['epoch']}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test inference
        print("   Testing inference capability...")
        inference_ok = test_model_inference(model, device)
        
        if inference_ok:
            print(f"✅ Model passed inference test")
            return True
        else:
            print(f"❌ Model failed inference test")
            return False
    except Exception as e:
        import traceback
        print(f"❌ Error during model loading/testing: {str(e)}")
        traceback.print_exc()
        return False

def main():
    print("=== RNA 3D Structure Prediction - Model Test ===")
    print(f"Current directory: {current_dir}")
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Models to test
    models = [
        os.path.join(kaggle_env_dir, "rna-3d-models", "best_model.pt"),
        os.path.join(kaggle_env_dir, "rna-3d-models", "production_model.pt")
    ]
    
    # Test each model
    success_count = 0
    for model_path in models:
        if test_model(model_path, device):
            success_count += 1
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Models tested: {len(models)}")
    print(f"Models passed: {success_count}")
    
    if success_count == len(models):
        print("\n🎉 All models loaded and passed inference tests!")
        return 0
    else:
        print("\n⚠️ Some models failed testing. See details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())