#!/usr/bin/env python3
"""
Validate Kaggle Package

This script validates that the kaggle_package contains all required
components and can successfully load models and run inference.
"""

import os
import sys
import torch
from pathlib import Path

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import project modules
from src.models.rna_folding_model import RNAFoldingModel
from src.utils.model_loader import fixed_load_model, test_model_inference

def check_directory_exists(path, name):
    """Check if a directory exists and print status."""
    exists = os.path.exists(path) and os.path.isdir(path)
    status = "✅" if exists else "❌"
    print(f"{status} {name} directory: {path}")
    return exists

def check_file_exists(path, name):
    """Check if a file exists and print status."""
    exists = os.path.exists(path) and os.path.isfile(path)
    status = "✅" if exists else "❌"
    print(f"{status} {name} file: {path}")
    return exists

def validate_structure():
    """Validate the package directory structure."""
    print("\n=== Validating Directory Structure ===")
    
    # Check main directories
    src_dir = os.path.join(current_dir, "src")
    models_dir = os.path.join(src_dir, "models")
    utils_dir = os.path.join(src_dir, "utils")
    data_dir = os.path.join(current_dir, "data")
    results_dir = os.path.join(current_dir, "results")
    
    main_dirs_valid = all([
        check_directory_exists(src_dir, "Source code"),
        check_directory_exists(models_dir, "Models"),
        check_directory_exists(utils_dir, "Utilities"),
        check_directory_exists(data_dir, "Data"),
        check_directory_exists(results_dir, "Results")
    ])
    
    # Note: We've removed the redundant Kaggle-specific directories since they're prepared as needed
    kaggle_dirs_valid = True
    print("✅ Kaggle upload directories will be created as needed (see USAGE.md)")
    
    # Check notebooks
    notebooks_dir = os.path.join(current_dir, "notebooks")
    notebook_valid = check_directory_exists(notebooks_dir, "Notebooks")
    if notebook_valid:
        notebook_path = os.path.join(notebooks_dir, "kaggle_inference_v2.1.ipynb")
        notebook_valid = check_file_exists(notebook_path, "Inference notebook")
    
    # Check test data
    test_data_valid = check_file_exists(
        os.path.join(current_dir, "data", "raw", "test_sequences.csv"),
        "Test sequences"
    )
    
    # Check model files
    model_files_valid = check_file_exists(
        os.path.join(current_dir, "results", "final_model", "run_20250423-072601", 
                     "checkpoints", "best_model.pt"),
        "Local best model"
    )
    
    # Check for required modules
    required_modules = [
        "src/__init__.py",
        "src/data_loading.py",
        "src/models/__init__.py",
        "src/models/rna_folding_model.py",
        "src/models/embeddings.py", 
        "src/models/transformer_block.py",
        "src/models/ipa_module.py",
        "src/utils/__init__.py",
        "src/utils/padding.py",
        "src/utils/structure_metrics.py",
        "src/utils/model_loader.py"
    ]
    
    print("\n=== Checking Required Module Files ===")
    modules_valid = True
    for module in required_modules:
        module_path = os.path.join(current_dir, module)
        if not check_file_exists(module_path, os.path.basename(module)):
            modules_valid = False
    
    # Overall structure validation
    structure_valid = all([
        main_dirs_valid, kaggle_dirs_valid, notebook_valid, 
        test_data_valid, model_files_valid, modules_valid
    ])
    
    print(f"\nDirectory Structure Validation: {'✅ PASSED' if structure_valid else '❌ FAILED'}")
    return structure_valid

def validate_model_loading():
    """Validate that models can be loaded and used for inference."""
    print("\n=== Validating Model Loading ===")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test loading the models
    model_paths = [
        os.path.join(current_dir, "results", "final_model", "run_20250423-072601", 
                     "checkpoints", "best_model.pt"),
        os.path.join(current_dir, "results", "production_run_1", "run_20250423-072209", 
                     "checkpoints", "best_model.pt")
    ]
    
    all_models_valid = True
    
    for i, model_path in enumerate(model_paths):
        print(f"\nTesting model {i+1}: {os.path.basename(model_path)}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file does not exist: {model_path}")
            all_models_valid = False
            continue
            
        try:
            # Load the model using our fixed loader
            model, config, metrics = fixed_load_model(model_path, device)
            
            if model is None:
                print(f"❌ Failed to load model from {model_path}")
                all_models_valid = False
                continue
                
            print(f"✅ Successfully loaded model from {model_path}")
            print(f"   Model configuration: {len(config)} parameters")
            print(f"   Validation RMSD: {metrics['val_rmsd']}")
            print(f"   Trained epochs: {metrics['epoch']}")
            print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Test basic inference
            print("   Testing inference capability...")
            inference_ok = test_model_inference(model, device)
            
            if not inference_ok:
                print(f"❌ Model failed inference test")
                all_models_valid = False
            else:
                print(f"✅ Model passed inference test")
                
        except Exception as e:
            print(f"❌ Error during model validation: {str(e)}")
            import traceback
            traceback.print_exc()
            all_models_valid = False
    
    print(f"\nModel Loading Validation: {'✅ PASSED' if all_models_valid else '❌ FAILED'}")
    return all_models_valid

def validate_notebook_imports():
    """Validate that notebook imports will work."""
    print("\n=== Validating Notebook Imports ===")
    
    try:
        # Test importing the key components used by the notebook
        from src.models.rna_folding_model import RNAFoldingModel
        from src.data_loading import RNADataset, collate_fn, create_data_loader
        from src.utils.model_loader import fixed_load_model
        
        print("✅ Successfully imported all required modules")
        return True
    except Exception as e:
        print(f"❌ Error importing modules: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the package validation."""
    print("=== RNA 3D Structure Prediction - Kaggle Package Validation ===")
    print(f"Current directory: {current_dir}")
    
    # Run validation tests
    structure_valid = validate_structure()
    imports_valid = validate_notebook_imports()
    model_valid = validate_model_loading()
    
    # Overall validation result
    all_valid = all([structure_valid, imports_valid, model_valid])
    
    print("\n=== Validation Summary ===")
    print(f"Directory Structure: {'✅ PASSED' if structure_valid else '❌ FAILED'}")
    print(f"Module Imports: {'✅ PASSED' if imports_valid else '❌ FAILED'}")
    print(f"Model Loading: {'✅ PASSED' if model_valid else '❌ FAILED'}")
    print(f"\nOverall Validation: {'✅ PASSED' if all_valid else '❌ FAILED'}")
    
    if all_valid:
        print("\n🎉 Package is ready for Kaggle submission! 🎉")
        print("1. Upload rna-model-src, rna-3d-models, and rna-3d-features as separate datasets")
        print("2. Upload the notebook to Kaggle")
        print("3. Add the stanford-rna-3d-folding competition dataset")
        print("4. Run the notebook to generate predictions")
    else:
        print("\n⚠️ Package needs attention before Kaggle submission!")
        print("Please fix the issues noted above and run validation again.")
    
    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())