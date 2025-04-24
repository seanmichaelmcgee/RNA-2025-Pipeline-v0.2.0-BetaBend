"""
Import Test Cell

Copy this as one of the first cells in the notebook to test all imports
"""

print("=== Testing Required Imports ===")

# Standard library imports
import_status = {
    "Standard Library": {},
    "Deep Learning": {},
    "Data Science": {},
    "Project Modules": {}
}

# Test standard library imports
for module in ["os", "sys", "json", "pathlib", "datetime", "traceback", "math", "warnings"]:
    try:
        exec(f"import {module}")
        import_status["Standard Library"][module] = "✅ Success"
    except ImportError as e:
        import_status["Standard Library"][module] = f"❌ Failed: {str(e)}"

# Test deep learning imports
for module in ["torch", "torch.nn", "torch.utils.data", "torch.nn.functional"]:
    try:
        exec(f"import {module}")
        import_status["Deep Learning"][module] = "✅ Success"
    except ImportError as e:
        import_status["Deep Learning"][module] = f"❌ Failed: {str(e)}"

# Test data science imports
for module in ["numpy", "pandas", "matplotlib.pyplot", "tqdm.notebook"]:
    try:
        exec(f"import {module}")
        import_status["Data Science"][module] = "✅ Success"
    except ImportError as e:
        import_status["Data Science"][module] = f"❌ Failed: {str(e)}"

# Test project module imports
project_modules = [
    "src.models.rna_folding_model", 
    "src.models.embeddings",
    "src.models.transformer_block",
    "src.models.ipa_module",
    "src.data_loading",
    "src.utils.padding",
    "src.utils.structure_metrics",
    "src.utils.model_loader"
]

for module in project_modules:
    try:
        exec(f"import {module}")
        import_status["Project Modules"][module] = "✅ Success"
    except ImportError as e:
        import_status["Project Modules"][module] = f"❌ Failed: {str(e)}"

# Test specific class imports
specific_imports = [
    "from src.models.rna_folding_model import RNAFoldingModel",
    "from src.data_loading import RNADataset, collate_fn, create_data_loader",
    "from src.utils.model_loader import fixed_load_model, test_model_inference",
    "from src.models.embeddings import SequenceEmbedding, PositionalEncoding",
    "from src.models.transformer_block import TransformerBlock",
    "from src.models.ipa_module import IPAModule"
]

print("\n=== Testing Specific Class Imports ===")
for imp in specific_imports:
    try:
        exec(imp)
        print(f"{imp}: ✅ Success")
    except ImportError as e:
        print(f"{imp}: ❌ Failed: {str(e)}")
    except Exception as e:
        print(f"{imp}: ⚠️ Error: {str(e)}")

# Print import status by category
for category, modules in import_status.items():
    print(f"\n=== {category} Imports ===")
    for module, status in modules.items():
        print(f"{module}: {status}")

# Report GPU availability for PyTorch
print("\n=== PyTorch GPU Status ===")
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ CUDA not available, using CPU")
    
print("\nImport check complete!")