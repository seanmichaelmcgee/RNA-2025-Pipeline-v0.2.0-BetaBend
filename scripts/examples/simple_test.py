#!/usr/bin/env python3
import os
import sys

# Get the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

print(f"Project root: {project_root}")

# Add src to Python path
sys.path.insert(0, os.path.join(project_root, "src"))

# Try importing models
try:
    # First try the package-style import
    print("Trying package-style import...")
    import src.models
    print("Success: imported src.models")
except ImportError as e:
    print(f"Failed package-style import: {e}")
    
    # Now try direct import
    try:
        print("Trying direct import...")
        import models
        print("Success: imported models directly")
    except ImportError as e:
        print(f"Failed direct import: {e}")

# Print all modules in sys.modules
print("\nLoaded modules:")
for module in sorted(sys.modules.keys()):
    if 'models' in module:
        print(f"  {module}")

print("\nPython path:")
for path in sys.path:
    print(f"  {path}")