"""
Model Loader Patch for Kaggle Notebook

This patch updates the notebook to use the fixed model loader from src.utils.model_loader
instead of the inline implementation. This improves maintainability and consistency.

To apply this patch:
1. Open the notebook in edit mode
2. Replace the existing load_model function with the import and code below
"""

# Import fixed model loader
from src.utils.model_loader import fixed_load_model

# Example of how to use in the notebook:
"""
# Load model
model, config, metrics = fixed_load_model(model_path, device)
if model is None:
    raise RuntimeError(f"Failed to load model from {model_path}")
    
print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Validation RMSD: {metrics['val_rmsd']}")
"""