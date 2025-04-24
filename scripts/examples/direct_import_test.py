#!/usr/bin/env python3
"""
Script to test direct imports of the project modules.
This will help diagnose why the imports are failing.
"""

import os
import sys
from pathlib import Path

def main():
    print("Running import test")
    
    # Get project paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = script_dir
    scripts_dir = script_dir.parent
    project_root = scripts_dir.parent
    
    print(f"Script directory: {script_dir}")
    print(f"Scripts directory: {scripts_dir}")
    print(f"Project root: {project_root}")
    
    # Add project root to path
    sys.path.insert(0, str(project_root))
    print(f"Added to sys.path: {project_root}")
    
    # Print full path
    print("\nFull sys.path:")
    for idx, path in enumerate(sys.path):
        print(f"  {idx}: {path}")
    
    # Check for src directory
    src_dir = project_root / "src"
    if src_dir.exists():
        print(f"\nThe src directory exists: {src_dir}")
    else:
        print(f"\nWARNING: The src directory does not exist: {src_dir}")
    
    # Check for src models directory
    models_dir = src_dir / "models"
    if models_dir.exists():
        print(f"The models directory exists: {models_dir}")
    else:
        print(f"WARNING: The models directory does not exist: {models_dir}")
    
    # Check for model file
    model_file = models_dir / "rna_folding_model.py"
    if model_file.exists():
        print(f"The model file exists: {model_file}")
    else:
        print(f"WARNING: The model file does not exist: {model_file}")
    
    # Print files in src directory
    print("\nFiles in src directory:")
    try:
        for item in os.listdir(src_dir):
            print(f"  - {item}")
    except Exception as e:
        print(f"  Error listing src directory: {e}")
    
    # Print files in models directory
    print("\nFiles in models directory:")
    try:
        for item in os.listdir(models_dir):
            print(f"  - {item}")
    except Exception as e:
        print(f"  Error listing models directory: {e}")
    
    # List __init__.py files
    init_src = src_dir / "__init__.py"
    init_models = models_dir / "__init__.py"
    
    print(f"\nSrc __init__.py exists: {init_src.exists()}")
    print(f"Models __init__.py exists: {init_models.exists()}")
    
    # Try direct imports
    print("\nTrying imports:")
    
    try:
        import src
        print(f"✓ Successfully imported src: {src.__file__}")
        
        try:
            import src.models
            print(f"✓ Successfully imported src.models: {src.models.__file__}")
            
            try:
                from src.models import rna_folding_model
                print(f"✓ Successfully imported src.models.rna_folding_model: {rna_folding_model.__file__}")
                
                try:
                    from src.models.rna_folding_model import RNAFoldingModel
                    print(f"✓ Successfully imported RNAFoldingModel")
                except ImportError as e:
                    print(f"✗ Error importing RNAFoldingModel: {e}")
            except ImportError as e:
                print(f"✗ Error importing src.models.rna_folding_model: {e}")
        except ImportError as e:
            print(f"✗ Error importing src.models: {e}")
    except ImportError as e:
        print(f"✗ Error importing src: {e}")
    
    print("\nImport test complete!")

if __name__ == "__main__":
    main()