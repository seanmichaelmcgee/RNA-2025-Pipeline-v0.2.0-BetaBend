#!/usr/bin/env python3
"""
Creates a fixed version of data_loading.py with absolute imports instead of relative.
"""
import os
import sys
from pathlib import Path

def main():
    # Get the project root
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = script_dir.parent
    src_dir = project_root / "src"
    
    # Path to original data_loading.py
    original_file = src_dir / "data_loading.py"
    
    # Create a fixed version in the same directory
    fixed_file = src_dir / "data_loading_fixed.py"
    
    print(f"Creating fixed data_loading module at: {fixed_file}")
    
    # Read the original file
    with open(original_file, 'r') as f:
        content = f.read()
    
    # Replace the relative import with absolute import
    fixed_content = content.replace(
        "from .utils.padding import pad_1d, pad_2d, pad_tensor",
        "from src.utils.padding import pad_1d, pad_2d, pad_tensor"
    )
    
    # Write the fixed file
    with open(fixed_file, 'w') as f:
        f.write(fixed_content)
    
    # Make it executable
    os.chmod(fixed_file, 0o755)
    
    print("Fixed data_loading.py created. Now update train_enhanced_model_direct.py to use this")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())