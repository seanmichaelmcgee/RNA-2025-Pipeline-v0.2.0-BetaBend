# START OF FILE: docs/claude/workflows/80_debugging.md

# Debugging Workflow: RNA 3D Folding Pipeline

**Version:** 1.3 (Restored original content, integrated tools from Loss Examples)
**Date:** 2024-04-15

## 1. Overview & Debug Philosophy

### 1.1 Introduction

Debugging a complex machine learning pipeline like our RNA 3D folding system requires a systematic, scientific approach. This document provides a comprehensive framework for identifying, diagnosing, and resolving issues throughout the pipeline, from data loading to model prediction and evaluation.

Our debugging philosophy is guided by these core principles:

*   **Isolation**: Isolate components to find the exact source of an issue.
*   **Reproducibility**: Ensure bugs can be consistently reproduced with fixed seeds and environments.
*   **Evidence-Based**: Collect concrete evidence through logging, assertions, and instrumentation.
*   **Incremental**: Fix issues one at a time, verifying each solution before proceeding.
*   **Documentation**: Record findings, root causes, and solutions for future reference.

### 1.2 Debug Environment Setup

Before beginning any debugging session, establish a clean and controlled environment:

```bash
# 1. Start with a fresh clone or clean working directory
# git clean -fdx
# git reset --hard HEAD

# 2. Use the Docker environment for reproducibility
# Ensure the image is built with the latest code
docker build -t rna-3d-folding .
# Run the container, mounting project and data directories
docker run --rm -it --gpus all \
  -v $(pwd):/app \
  -v /path/to/local/data:/app/data \
  rna-3d-folding bash

# 3. Activate the conda environment inside the container
conda activate rna-3d-folding

# 4. Set debugging flags for PyTorch (optional, can slow down execution)
# export PYTORCH_DEBUG=1 # More verbose PyTorch errors
export CUDA_LAUNCH_BLOCKING=1 # For clearer CUDA error stack traces
```

### 1.3 Debug Toolkit and Utilities

Equip yourself with these essential debugging tools:

1.  **PyTorch Debugging Tools**:
    *   `torch.autograd.detect_anomaly()`: Enable autograd anomaly detection during backward pass. Use within a `with` block: `with torch.autograd.detect_anomaly(): loss.backward()`.
    *   `torch.autograd.profiler`: Profile computational performance and memory usage.
2.  **Memory Profiling**:
    *   `torch.cuda.memory_summary()`, `torch.cuda.memory_allocated()`, `torch.cuda.max_memory_allocated()`: GPU memory inspection.
    *   `nvidia-smi`: System-level GPU monitoring (run outside container if needed, or inside if installed).
    *   Python `tracemalloc` module for detailed CPU memory allocation tracking.
3.  **Logging Utilities**:
    *   Python's `logging` module with appropriate levels (DEBUG, INFO, WARNING, ERROR). Configure it early in your scripts.
    *   TensorBoard for visualizing metrics, losses, and model graphs (`torch.utils.tensorboard.SummaryWriter`).
4.  **Debug Helper Functions**: (Place these in a `src/utils/debug_utils.py` or similar file and import them)

    ```python
    # src/utils/debug_utils.py (Example)
    import torch
    import numpy as np
    import logging
    import gc
    from typing import Dict, Any, Optional, List, Tuple

    def inspect_tensor(tensor: Optional[torch.Tensor], name: str = "tensor", full_stats: bool = False):
        """Print comprehensive information about a tensor for debugging."""
        if not isinstance(tensor, torch.Tensor):
            print(f"--- {name}: Not a Tensor (type: {type(tensor)}) ---")
            return
        print(f"--- {name} ---")
        print(f"Shape: {tuple(tensor.shape)}")
        print(f"Dtype: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        print(f"Requires Grad: {tensor.requires_grad}")
        if tensor.numel() == 0:
            print("Tensor is empty.")
            print("-------------------")
            return
        try:
            # Attempt conversion to float for stats, handle non-float types gracefully
            try:
                # Use .detach() in case tensor requires grad, for stats calculation
                float_tensor = tensor.detach().float()
                stats_possible = True
            except RuntimeError:
                stats_possible = False # Cannot convert non-numeric tensor (e.g., bool)

            if stats_possible:
                # Check for NaNs/Infs first, as stats methods might fail on them
                has_nan = torch.isnan(float_tensor).any().item()
                has_inf = torch.isinf(float_tensor).any().item()
                print(f"Has NaN: {has_nan}")
                print(f"Has Inf: {has_inf}")

                if not (has_nan or has_inf): # Compute stats only if values are finite
                    print(f"Min/Max: {float_tensor.min().item():.6f} / {float_tensor.max().item():.6f}")
                    print(f"Mean/Std: {float_tensor.mean().item():.6f} / {float_tensor.std().item():.6f}")
                    if full_stats and tensor.numel() > 0:
                        # Ensure tensor is on CPU for histc if needed, handle potential errors
                        try:
                            hist = torch.histc(float_tensor.cpu(), bins=10).long().tolist()
                            print(f"Histogram (10 bins): {hist}")
                        except Exception as hist_e:
                            print(f"Could not compute histogram: {hist_e}")
                else:
                     print("Skipping Min/Max/Mean/Std/Histogram due to NaN/Inf values.")


            # First few values for inspection
            flat = tensor.flatten()
            print(f"First 5 values: {flat[:min(5, flat.numel())].tolist()}")

        except Exception as e:
            print(f"Error during tensor inspection for '{name}': {e}")
        print("-------------------")

    def profile_memory(label: str = "", verbose: bool = True) -> Dict[str, float]:
        """Print memory usage statistics and return them."""
        stats = {}
        if torch.cuda.is_available():
            gc.collect() # Suggest garbage collection first
            torch.cuda.empty_cache() # Clear unused cache
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            # Note: max_memory_allocated is peak since last reset, not current peak
            # For peak *during* an operation, reset *before* the operation
            max_allocated_since_reset = torch.cuda.max_memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            max_reserved_since_reset = torch.cuda.max_memory_reserved() / (1024 ** 2)
            stats = {
                'allocated_mb': allocated, 'peak_allocated_mb': max_allocated_since_reset,
                'reserved_mb': reserved, 'peak_reserved_mb': max_reserved_since_reset
            }
            if verbose:
                print(f"\n--- Memory usage at {label} ---")
                print(f"Allocated: {allocated:.2f} MB (Peak Since Reset: {max_allocated_since_reset:.2f} MB)")
                print(f"Reserved: {reserved:.2f} MB (Peak Since Reset: {max_reserved_since_reset:.2f} MB)")
        elif verbose:
            print(f"\n--- Memory usage at {label}: CUDA not available ---")
        return stats

    # Add other helpers like analyze_gradients, log_gradient_issues etc. here
    # ... (Include implementations for analyze_gradients, log_gradient_issues etc. from previous response) ...
    def analyze_gradients(model: torch.nn.Module, norm_threshold: float = 10.0) -> Dict:
        """Analyze gradient statistics per parameter."""
        gradient_stats = {}
        total_norm = 0.0
        num_params_with_grad = 0
        num_params_zero_grad = 0
        num_params_nan_inf_grad = 0
        num_params_exploding = 0
        num_params_vanishing = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                # logging.debug(f"No gradient for parameter: {name}") # Too verbose usually
                stats = {'norm': 0.0, 'status': 'NoGrad'}
            else1.1:
                grad = param.grad.detach()
                stats = {'status': 'OK'}
                if not torch.isfinite(grad).all(): # Check for NaN/Inf
                    logging.warning(f"NaN/Inf gradient detected in parameter: {name}")
                    stats = {'norm': float('inf'), 'status': 'NaN/Inf'}
                    num_params_nan_inf_grad += 1
                else:
                    norm = grad.norm().item()
                    stats = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'min': grad.min().item(),
                        'max': grad.max().item(),
                        'norm': norm,
                        'magnitude_mean': torch.abs(grad).mean().item(),
                        'zero_fraction': (grad == 0).float().mean().item(),
                        'status': 'OK'
                    }
                    total_norm += norm**2
                    num_params_with_grad += 1
                    if norm == 0.0:
                        stats['status'] = 'Zero'
                        num_params_zero_grad += 1
                    elif norm > norm_threshold:
                        stats['status'] = 'Exploding'
                        num_params_exploding += 1
                    elif norm < 1e-8: # Adjust threshold as needed
                        stats['status'] = 'Vanishing'
                        num_params_vanishing += 1

            gradient_stats[name] = stats

        total_norm = total_norm**0.5
        gradient_stats['overall'] = {
            'total_norm': total_norm,
            'params_with_grad': num_params_with_grad,
            'params_zero_grad': num_params_zero_grad,
            'params_nan_inf_grad': num_params_nan_inf_grad,
            'params_exploding': num_params_exploding,
            'params_vanishing': num_params_vanishing
            }
        # log_gradient_issues(gradient_stats) # Call logging function separately if desired
        return gradient_stats

    def log_gradient_issues(gradient_stats: Dict) -> bool:
        """Log gradient issues based on analysis."""
        overall = gradient_stats.get('overall', {})
        has_issues = False
        if overall.get('params_nan_inf_grad', 0) > 0:
            logging.warning(f"NaN/Inf gradients detected in {overall['params_nan_inf_grad']} parameter(s).")
            has_issues = True
        if overall.get('params_exploding', 0) > 0:
            logging.warning(f"Exploding gradients detected in {overall['params_exploding']} parameter(s).")
            has_issues = True
        # Decide if vanishing/zero are critical issues to return True for
        if overall.get('params_vanishing', 0) > 0:
             logging.warning(f"Vanishing gradients detected in {overall['params_vanishing']} parameter(s).")
        if overall.get('params_zero_grad', 0) > 0:
             logging.warning(f"Zero gradients detected in {overall['params_zero_grad']} parameter(s).")
        return has_issues
    ```

## 2. Debugging Methodology

### 2.1 Systematic Debugging Framework

Follow this structured approach to efficiently diagnose and fix issues:
    1. Observe: Clearly define the problem and how it manifests 
    2. Reproduce: Create a minimal test case that reliably triggers the issue 
    3. Hypothesize: Formulate testable hypotheses about the cause 
    4. Test: Design experiments to confirm or reject each hypothesis 
    5. Isolate: Narrow down to the exact source of the problem 
    6. Solve: Implement and verify a solution 
    7. Document: Record the issue, root cause, and solution 
    
### 2.2 Issue Categorization

Classify issues to guide your debugging strategy:
Category
Symptoms
Debugging Approach
Data Pipeline
Missing features, shape mismatches, NaN values
Data validation, format checking
Model Architecture
Shape errors, device mismatches, incorrect tensor operations
Component isolation, shape/device tracing
Training Loop
Loss not decreasing, exploding/vanishing gradients
Gradient inspection, loss decomposition
Performance
OOM errors, slow processing, GPU underutilization
Memory profiling, operation optimization
Integration
Module interaction errors, path resolution failures
Interface validation, dependency testing


### 2.3 Diagnostic Hierarchy & `debug_test_failure`

Use this decision tree to systematically narrow down the source of issues:
1. Is the issue:
   ├── Reproducible → Continue debugging
   │   
   └── Intermittent → Check for:
       ├── Race conditions
       ├── Random seed inconsistencies
       ├── Memory leaks
       └── Hardware limitations

2. At which stage does the issue occur?
   ├── Data loading → Debug data pipeline
   ├── Model forward pass → Debug architecture
   ├── Loss calculation → Debug loss functions
   ├── Backpropagation → Debug gradient flow
   └── Post-training → Debug inference pipeline

3. What is the nature of the error?
   ├── Exception/Error message → Parse for clues
   ├── Incorrect outputs → Compare with expected values
   ├── Performance degradation → Profile resource usage
   └── Silent failure → Add instrumentation & logging
```python
import re
import os

def debug_test_failure(test_name, error_message):
    """Guide debugging based on test failure type."""
    print(f"\nDebugging test failure: {test_name}")
    print(f"Error message:\n{'-'*20}\n{error_message}\n{'-'*20}")

    # Categorize the error
    error_type = "unknown"
    # ... (Error categorization logic as provided before) ...
    if "size mismatch" in error_message.lower() or "shape" in error_message.lower() or "dimension" in error_message.lower(): error_type = "shape_mismatch"
    elif "cuda out of memory" in error_message.lower(): error_type = "memory_error"
    elif "nan" in error_message.lower() or "inf" in error_message.lower(): error_type = "numerical_error"
    elif "device" in error_message.lower() and ("cpu" in error_message.lower() or "cuda" in error_message.lower()): error_type = "device_error"
    elif "modulenotfounderror" in error_message.lower() or "importerror" in error_message.lower(): error_type = "import_error"
    elif "keyerror" in error_message.lower(): error_type = "key_error"
    elif "filenotfounderror" in error_message.lower(): error_type = "file_not_found"
    elif "assertionerror" in error_message.lower(): error_type = "assertion_failure"

    print(f"Detected error type: {error_type}")

    # Suggest debugging approaches based on error type
    suggestions = []
    if error_type == "shape_mismatch":
        suggestions.extend([
            "Locate the operation causing the mismatch in the stack trace.",
            "Use inspect_tensor() on input tensors just before the failing operation.",
            "Trace tensor shapes through the model using debug_shape_flow() if needed.",
            "Verify related config parameters (embedding dims, heads, etc.).",
            "Check padding/masking logic in collate_fn and model forward pass.",
        ])
        shapes = re.findall(r'size (\d+)', error_message)
        if len(shapes) >= 2: suggestions.append(f"Mismatched dimensions found: {shapes[0]} vs {shapes[1]}. Check config params.")
    elif error_type == "memory_error":
        suggestions.extend([
            "Run the test with a smaller batch size (e.g., 1).",
            "Use profile_memory() before and after the failing operation.",
            "Check for operations creating large intermediate tensors (e.g., N x N pair matrices).",
            "Consider using gradient checkpointing if it occurs during backward pass.",
            "Delete intermediate tensors explicitly (`del tensor`) and call `gc.collect()`, `torch.cuda.empty_cache()`."
        ])
    elif error_type == "numerical_error":
        suggestions.extend([
            "Enable torch.autograd.detect_anomaly().",
            "Use inspect_tensor() on inputs and outputs of the failing operation/module.",
            "If loss is NaN/Inf, use diagnose_nan_loss() or specific debug_loss_function().",
            "Check for division by zero, log(0), sqrt(negative). Add epsilon where needed.",
            "Check parameter initialization (analyze_parameter_initialization).",
            "Reduce learning rate, check gradient norms (analyze_gradients)."
        ])
    elif error_type == "device_error":
        suggestions.extend([
            "Ensure all input tensors to an operation are on the same device.",
            "Check model parameters are on the correct device (`model.to(device)`).",
            "Ensure batch tensors (including mask, targets) are moved to device.",
            "Verify tensor creation ops (`torch.zeros`, `torch.randn`) specify `device=...`."
        ])
    elif error_type == "key_error":
        key = re.search(r"KeyError: '([^']+)'", error_message)
        suggestions.append("Check dictionary contents (e.g., batch dict, config dict) before access.")
        if key: suggestions.append(f"  - The key '{key.group(1)}' was expected but not found.")
        suggestions.append("  - Verify spelling and case of keys.")
    elif error_type == "file_not_found":
        path = re.search(r"FileNotFoundError:.*? '([^']+)'", error_message)
        suggestions.append("Verify the path exists relative to the execution context (e.g., inside Docker).")
        if path: suggestions.append(f"  - The path '{path.group(1)}' could not be found.")
        suggestions.append("  - Check path construction logic (use os.path.join).")
        suggestions.append("  - Ensure path parameterization principle is followed.")
    elif error_type == "assertion_failure":
         suggestions.extend([
             "Examine the specific test assertion that failed.",
             "Check if the expected behavior matches implementation logic.",
             "Verify calculation logic for expected values in the test.",
             "Compare actual outputs against reference/expected values step-by-step."
         ])
    else:
        suggestions.extend([
            "Carefully read the full error message and stack trace.",
            "Isolate the failing code in a minimal script.",
            "Add print statements or use a debugger (like pdb or IDE debugger)."
        ])

    print("\nDebugging suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")

    # Extract file and line info if available
    # Find the last match in the traceback which is likely the error origin
    all_matches = re.findall(r'File "([^"]+)", line (\d+)', error_message)
    if all_matches:
         file_path, line_num = all_matches[-1]
         print(f"\nError likely originates near: {file_path}:{line_num}")
         # ... (Suggest relevant specific debug functions based on file_path) ...

    print("\nNext steps:")
    print("1. Run the specific failing test: `pytest -v -k test_function_name`")
    print("2. Add print statements or use `inspect_tensor` around the error location.")
    print("3. Create a minimal example script outside pytest if needed.")
```
    
## 3. Data Pipeline Debugging

### 3.1 Feature Loading and Preprocessing Issues
Common issues in the data loading pipeline include:
    1. Missing or Corrupted Feature Files: 
def verify_feature_files(target_id, features_dir):
    """Verify that all required feature files exist and are valid."""
    expected_files = [
        os.path.join(features_dir, "thermo_features", f"{target_id}_thermo_features.npz"),
        os.path.join(features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz"),
        os.path.join(features_dir, "mi_features", f"{target_id}_features.npz")
    ]
    
    for file_path in expected_files:
        if not os.path.exists(file_path):
            print(f"WARNING: Missing file {file_path}")
            continue
            
        try:
            with np.load(file_path) as data:
                keys = list(data.keys())
                print(f"File {os.path.basename(file_path)} contains keys: {keys}")
                
                # Check file size and content validity
                for key in keys:
                    array = data[key]
                    if isinstance(array, np.ndarray):
                        print(f"  - {key}: shape={array.shape}, dtype={array.dtype}")
                        
                        # Check for NaN or Inf values
                        if np.issubdtype(array.dtype, np.number):
                            print(f"    Has NaN: {np.isnan(array).any()}")
                            print(f"    Has Inf: {np.isinf(array).any()}")
        except Exception as e:
            print(f"ERROR: Failed to load {file_path}: {e}")
    2. Inconsistent Feature Shapes: 
def validate_shape_consistency(features):
    """Validate that all features have consistent shapes based on sequence length."""
    # Determine sequence length from thermo features
    if 'thermo' in features and 'pairing_probs' in features['thermo']:
        seq_len = features['thermo']['pairing_probs'].shape[0]
        print(f"Detected sequence length: {seq_len}")
        
        # Check 1D feature shapes
        for feature_type, feature_dict in features.items():
            if feature_dict is None:
                print(f"WARNING: {feature_type} features are None")
                continue
                
            for key, value in feature_dict.items():
                if not isinstance(value, np.ndarray):
                    continue
                    
                if value.ndim == 1 and value.shape[0] != seq_len:
                    print(f"ERROR: {feature_type}.{key} has shape {value.shape}, expected ({seq_len},)")
                elif value.ndim == 2:
                    if key in ['pairing_probs', 'coupling_matrix'] and (value.shape[0] != seq_len or value.shape[1] != seq_len):
                        print(f"ERROR: {feature_type}.{key} has shape {value.shape}, expected ({seq_len}, {seq_len})")
                    elif key == 'features' and value.shape[0] != seq_len:
                        print(f"ERROR: {feature_type}.{key} has shape {value.shape}, expected ({seq_len}, {value.shape[1]})")
    3. Handling NaN and Missing Values: 
def diagnose_nan_values(dataset_path, features_dir):
    """Scan dataset for NaN values and suggest fixes."""
    dataset = RNADataset(
        sequences_csv_path=dataset_path,
        labels_csv_path=None,  # Not needed for this test
        features_dir=features_dir
    )
    
    nan_counts = {}
    
    for idx in range(min(len(dataset), 10)):  # Check first 10 samples
        sample = dataset[idx]
        target_id = sample['target_id']
        nan_counts[target_id] = {}
        
        for key, tensor in sample.items():
            if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                nan_count = torch.isnan(tensor).sum().item()
                if nan_count > 0:
                    nan_counts[target_id][key] = nan_count
                    print(f"Found {nan_count} NaN values in {target_id}.{key}")
    
    return nan_counts
    
### 3.2 Batch Processing and Collation Problems
Issues often arise during batch creation and collation:
    1. Testing Collation Function: 
def test_collate_function(dataset, batch_size=2):
    """Test the collate function with different sequence lengths."""
    from torch.utils.data import DataLoader
    
    # Create samples with different lengths to test padding
    indices = list(range(min(batch_size, len(dataset))))
    samples = [dataset[i] for i in indices]
    
    # Get sequence lengths to verify padding
    lengths = [sample['length'] for sample in samples]
    print(f"Sample lengths: {lengths}")
    
    # Test collate_fn directly
    try:
        batch = collate_fn(samples)
        print("Collate function executed successfully")
        
        # Verify batch structure
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                
        # Verify mask correctness
        if 'mask' in batch:
            mask = batch['mask']
            for i, length in enumerate(lengths):
                assert torch.all(mask[i, :length]), f"Mask is incorrect for sample {i}"
                if length < mask.shape[1]:
                    assert torch.all(~mask[i, length:]), f"Padding mask is incorrect for sample {i}"
            print("Mask verification passed")
            
    except Exception as e:
        print(f"Collate function failed: {e}")
        import traceback
        traceback.print_exc()
    2. Padding Verification: 
def verify_padding(batch):
    """Verify that padding is correctly applied and masked."""
    if 'mask' not in batch or 'lengths' not in batch:
        print("ERROR: Batch missing mask or lengths")
        return False
    
    mask = batch['mask']
    lengths = batch['lengths']
    
    # Check each sample in the batch
    for i, length in enumerate(lengths):
        # Verify that the mask matches the expected length
        if not torch.all(mask[i, :length]):
            print(f"ERROR: Mask has False values in non-padded region for sample {i}")
            return False
        
        if length < mask.shape[1]:
            if not torch.all(~mask[i, length:]):
                print(f"ERROR: Mask has True values in padded region for sample {i}")
                return False
        
        # Check that padded values are zero in key tensors
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() >= 2:
                if key in ['sequence_int', 'dihedral_features', 'positional_entropy', 'coordinates']:
                    padded_region = tensor[i, length:]
                    if padded_region.numel() > 0 and not torch.all(padded_region == 0):
                        print(f"ERROR: Non-zero padding in {key} for sample {i}")
                        return False
    
    print("Padding verification passed")
    return True
### 3.3 Input Validation and Sanitization Strategies
Proactive data validation prevents downstream errors:
def validate_data_pipeline(sequences_csv_path, labels_csv_path, features_dir, batch_size=4):
    """End-to-end validation of the data pipeline."""
    
    # 1. Validate CSV files
    try:
        sequences_df = pd.read_csv(sequences_csv_path)
        print(f"Sequences CSV loaded successfully: {len(sequences_df)} entries")
        
        if labels_csv_path:
            labels_df = pd.read_csv(labels_csv_path)
            print(f"Labels CSV loaded successfully: {len(labels_df)} entries")
    except Exception as e:
        print(f"ERROR: Failed to load CSV files: {e}")
        return False
    
    # 2. Check feature directory structure
    required_dirs = [
        os.path.join(features_dir, "thermo_features"),
        os.path.join(features_dir, "dihedral_features"),
        os.path.join(features_dir, "mi_features")
    ]
    
    for directory in required_dirs:
        if not os.path.isdir(directory):
            print(f"WARNING: Expected directory not found: {directory}")
    
    # 3. Create dataset and sample a few entries
    try:
        dataset = RNADataset(
            sequences_csv_path=sequences_csv_path,
            labels_csv_path=labels_csv_path,
            features_dir=features_dir
        )
        print(f"Dataset created successfully: {len(dataset)} samples")
        
        # Sample a few entries for validation
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i} - target_id: {sample['target_id']}, length: {sample['length']}")
            
            # Validate key tensors
            for key in ['sequence_int', 'dihedral_features', 'pairing_probs']:
                if key in sample:
                    tensor = sample[key]
                    print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
                else:
                    print(f"  WARNING: Missing key '{key}' in sample")
    
    except Exception as e:
        print(f"ERROR: Dataset validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Create dataloader and fetch a batch
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        batch = next(iter(dataloader))
        print(f"Successfully fetched batch of size {batch_size}")
        
        # Validate batch structure
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Verify mask and padding
        verify_padding(batch)
        
    except Exception as e:
        print(f"ERROR: DataLoader validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Data pipeline validation completed successfully")
    return True
    
**Key Debugging Steps:**

1.  **Check Paths:** Ensure paths passed to `RNADataset` and helpers are correct within the Docker environment's mount points.
2.  **Verify Files:** Use `verify_feature_files` for a specific problematic `target_id`. Check file contents manually (`np.load(...)`).
3.  **Inspect Sample:** Add `inspect_tensor` calls within `RNADataset.__getitem__` for feature tensors right after loading and before returning. Check for NaNs using `diagnose_nan_values`.
4.  **Test Collation:** Use `test_collate_function` with a list of samples, including ones causing issues, to check padding and masking. Use `verify_padding` on the output batch.
5.  **End-to-End Check:** Run `validate_data_pipeline` with the relevant CSV paths and feature directory.


## 4. Model Architecture Debugging

### 4.1 Component-Level Debugging
Our modular architecture requires systematic component-level validation:
def debug_component(component, sample_input, expected_output=None):
    """Debug a single model component with sample input."""
    print(f"Debugging component: {component.__class__.__name__}")
    
    # 1. Validate input
    if isinstance(sample_input, dict):
        for key, tensor in sample_input.items():
            if isinstance(tensor, torch.Tensor):
                inspect_tensor(tensor, f"Input[{key}]")
    else:
        inspect_tensor(sample_input, "Input")
    
    # 2. Run forward pass with gradient tracking
    try:
        with torch.autograd.detect_anomaly():
            output = component(**sample_input) if isinstance(sample_input, dict) else component(sample_input)
        
        # 3. Inspect output
        if isinstance(output, tuple):
            for i, out in enumerate(output):
                inspect_tensor(out, f"Output[{i}]")
        elif isinstance(output, dict):
            for key, tensor in output.items():
                if isinstance(tensor, torch.Tensor):
                    inspect_tensor(tensor, f"Output[{key}]")
        else:
            inspect_tensor(output, "Output")
            
        # 4. Compare with expected output if provided
        if expected_output is not None:
            if isinstance(expected_output, dict):
                for key, expected in expected_output.items():
                    if key in output and isinstance(expected, torch.Tensor):
                        diff = torch.abs(output[key] - expected).mean()
                        print(f"Mean difference for {key}: {diff.item():.6f}")
            elif isinstance(expected_output, torch.Tensor):
                diff = torch.abs(output - expected).mean()
                print(f"Mean difference: {diff.item():.6f}")
        
        # 5. Test backward pass with dummy loss
        if isinstance(output, torch.Tensor) and output.requires_grad:
            dummy_loss = output.mean()
            dummy_loss.backward()
            print("Backward pass completed successfully")
            
        return output
        
    except Exception as e:
        print(f"ERROR: Component execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None
For the RNA folding model, test each component separately:
def debug_embedding_layer(config, seq_len=10, batch_size=2):
    """Debug the embedding layers."""
    # Create sequence embedding
    from src.models.embeddings import SequenceEmbedding
    seq_embed = SequenceEmbedding(
        num_embeddings=5,  # A, C, G, U, N/padding
        embedding_dim=config['seq_embed_dim']
    )
    
    # Create sample input
    sequence_int = torch.randint(0, 5, (batch_size, seq_len))
    
    # Debug
    output = debug_component(seq_embed, sequence_int)
    assert output.shape == (batch_size, seq_len, config['seq_embed_dim']), \
        f"Expected shape {(batch_size, seq_len, config['seq_embed_dim'])}, got {output.shape}"

def debug_transformer_block(config, seq_len=10, batch_size=2):
    """Debug a transformer block."""
    from src.models.transformer_block import TransformerBlock
    
    # Create transformer block
    block = TransformerBlock(config)
    
    # Create sample inputs
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    # Add padding to test masking
    mask[0, -2:] = False
    
    # Debug
    inputs = {
        'residue_repr': residue_repr,
        'pair_repr': pair_repr,
        'mask': mask
    }
    
    outputs = debug_component(block, inputs)
    
    # Verify outputs
    if isinstance(outputs, tuple) and len(outputs) == 2:
        res_out, pair_out = outputs
        assert res_out.shape == residue_repr.shape, \
            f"Expected residue shape {residue_repr.shape}, got {res_out.shape}"
        assert pair_out.shape == pair_repr.shape, \
            f"Expected pair shape {pair_repr.shape}, got {pair_out.shape}"
        
        # Verify masking was applied correctly
        assert torch.all(res_out[0, -2:] == 0), "Masking not applied correctly to residue output"
### 4.2 Shape Mismatch and Tensor Dimension Problems
Shape issues are among the most common bugs in deep learning pipelines:
def debug_shape_flow(model, batch):
    """Trace tensor shapes through the model to identify mismatches."""
    
    # Enable hooks to capture intermediate activations
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = {
                'input': [x.shape if isinstance(x, torch.Tensor) else x for x in input],
                'output': output.shape if isinstance(output, torch.Tensor) else 
                           [x.shape if isinstance(x, torch.Tensor) else x for x in output]
            }
        return hook
    
    # Register hooks for key modules
    hooks = []
    for name, module in model.named_modules():
        if any(key in name for key in ['embed', 'projection', 'attention', 'transformer', 'ipa']):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    try:
        with torch.no_grad():
            outputs = model(batch)
        
        # Print shape flow
        print("Tensor shape flow through model:")
        for name, shapes in activations.items():
            print(f"{name}:")
            print(f"  Input: {shapes['input']}")
            print(f"  Output: {shapes['output']}")
            
        return outputs, activations
    
    except Exception as e:
        print(f"ERROR: Shape tracing failed: {e}")
        # Print the last successful module's shapes
        last_module = list(activations.keys())[-1] if activations else None
        if last_module:
            print(f"Last successful module: {last_module}")
            print(f"  Input: {activations[last_module]['input']}")
            print(f"  Output: {activations[last_module]['output']}")
        
        import traceback
        traceback.print_exc()
        return None, activations
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
Shape debugging decision tree:
Shape Error detected → Check:
├── Batch dimension missing? → Add unsqueeze(0)
├── Feature dimension missing? → Add unsqueeze(-1)
├── Wrong dimension order? → Use permute() or transpose()
├── Inconsistent sequence lengths? → Debug collate_fn padding
└── Mask incorrectly applied? → Verify mask broadcasting
### 4.3 Parameter Initialization Issues
Improper initialization can lead to training instability:
def analyze_parameter_initialization(model):
    """Analyze parameter initialization statistics for each module."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Compute statistics
            mean = param.data.mean().item()
            std = param.data.std().item()
            min_val = param.data.min().item()
            max_val = param.data.max().item()
            
            # Check for potential issues
            has_nan = torch.isnan(param.data).any().item()
            has_inf = torch.isinf(param.data).any().item()
            is_zero = (param.data == 0).all().item()
            is_uniform = (max_val - min_val) < 1e-6
            
            # Print analysis
            print(f"{name}:")
            print(f"  Shape: {param.shape}")
            print(f"  Stats: mean={mean:.6f}, std={std:.6f}, min={min_val:.6f}, max={max_val:.6f}")
            
            # Flag potential issues
            issues = []
            if has_nan:
                issues.append("HAS NaN VALUES")
            if has_inf:
                issues.append("HAS INFINITE VALUES")
            if is_zero:
                issues.append("ALL ZEROS")
            if is_uniform and param.numel() > 1:
                issues.append("SUSPICIOUSLY UNIFORM")
            if std < 1e-8 and param.numel() > 1:
                issues.append("NEAR-ZERO VARIANCE")
            if abs(mean) > 1.0:
                issues.append("HIGH MEAN VALUE")
            if std > 1.0:
                issues.append("HIGH VARIANCE")
                
            if issues:
                print(f"  ISSUES DETECTED: {', '.join(issues)}")
            print("")
### 4.4 Forward/Backward Pass Debugging
Diagnose issues in the computational graph:
def debug_forward_backward(model, batch, compute_loss=None):
    """Debug forward and backward passes with gradient flow visualization."""
    # Reset gradients
    model.zero_grad()
    
    # Forward pass with anomaly detection
    with torch.autograd.detect_anomaly():
        try:
            # Forward pass
            outputs = model(batch)
            print("Forward pass completed successfully")
            
            # Print output structure
            print("Output structure:")
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                              f"requires_grad={value.requires_grad}")
            elif isinstance(outputs, torch.Tensor):
                print(f"  Output tensor: shape={outputs.shape}, dtype={outputs.dtype}, "
                      f"requires_grad={outputs.requires_grad}")
            
            # Compute loss
            if compute_loss:
                loss = compute_loss(outputs, batch)
            else:
                # Default simple loss for testing gradient flow
                if isinstance(outputs, dict) and 'pred_coords' in outputs:
                    loss = outputs['pred_coords'].abs().mean()
                elif isinstance(outputs, torch.Tensor):
                    loss = outputs.abs().mean()
                else:
                    raise ValueError("Cannot create default loss from outputs")
            
            print(f"Loss value: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            print("Backward pass completed successfully")
            
            # Check gradient flow
            grad_stats = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_stats[name] = {
                            'mean': param.grad.mean().item(),
                            'std': param.grad.std().item(),
                            'norm': grad_norm,
                            'has_nan': torch.isnan(param.grad).any().item(),
                            'has_inf': torch.isinf(param.grad).any().item()
                        }
                        
                        if grad_stats[name]['has_nan'] or grad_stats[name]['has_inf']:
                            print(f"WARNING: Parameter {name} has NaN or Inf gradients")
                    else:
                        print(f"WARNING: Parameter {name} has no gradient")
            
            # Find potential gradient issues
            print("\nGradient flow analysis:")
            zero_grad_params = [name for name, stats in grad_stats.items() if abs(stats['norm']) < 1e-8]
            high_grad_params = [name for name, stats in grad_stats.items() if abs(stats['norm']) > 10.0]
            
            if zero_grad_params:
                print(f"Parameters with near-zero gradients: {', '.join(zero_grad_params[:5])}"
                      f"{' and more...' if len(zero_grad_params) > 5 else ''}")
            if high_grad_params:
                print(f"Parameters with high gradients: {', '.join(high_grad_params[:5])}"
                      f"{' and more...' if len(high_grad_params) > 5 else ''}")
                
            return outputs, grad_stats
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            


**Key Debugging Steps:**

1.  **Initialization:** Use `analyze_parameter_initialization` after `model = RNAFoldingModel(config)` to check weights.
2.  **Component Isolation:** Use `debug_component` on individual layers (`SequenceEmbedding`, `TransformerBlock`, `IPAModule`) with controlled inputs to verify their internal logic.
3.  **Shape Tracing:** If shape errors occur during the full model forward pass, use `debug_shape_flow(model, batch)` to pinpoint where dimensions change unexpectedly. Use `inspect_tensor` liberally inside the `forward` methods of components.
4.  **Forward/Backward Graph:** Use `debug_forward_backward(model, batch, compute_combined_loss)` to ensure the entire graph connects and gradients can be computed. Check the gradient analysis output.

## 5. Loss Function Debugging
### 5.1 Numerical Stability Issues

Use `detect_and_handle_nan_loss` in the training loop for immediate detection and skipping faulty steps. For deeper diagnosis, use `diagnose_nan_loss`.

```python
# In training loop:
# ... forward pass ...
# total_loss, loss_components_tensors = compute_combined_loss(...)
# skip_step, safe_loss = detect_and_handle_nan_loss(
#     total_loss, loss_components_tensors, model, optimizer, batch
# )
# if skip_step: continue # Skip optimizer step
# safe_loss.backward()
# ...

# For detailed diagnosis outside the loop:
# diagnose_nan_loss(model, compute_combined_loss, specific_batch_causing_nan)
# debug_specific_loss('fape', model, specific_batch_causing_nan) # Focus on FAPE

def detect_and_handle_nan_loss(
    loss: torch.Tensor,
    loss_components: Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, Any]
) -> Tuple[bool, torch.Tensor]:
    """Detect NaN/Inf in loss values and respond, returning whether to skip step."""
    skip_step = False
    safe_loss = loss # Start with original loss

    if not torch.isfinite(loss).all(): # Checks for NaN and Inf
        logging.warning(f"NaN or Inf detected in total loss: {loss.item()}! Target IDs: {batch.get('target_ids', 'N/A')}")
        skip_step = True

        # Identify problematic components
        for name, comp_loss in loss_components.items():
            if not torch.isfinite(comp_loss).all():
                 logging.warning(f"  Component '{name}' has NaN/Inf: {comp_loss.item()}")

        # Check parameters and gradients for NaNs/Infs (if grads exist)
        for name, param in model.named_parameters():
             if param.grad is not None and not torch.isfinite(param.grad).all():
                 logging.warning(f"  NaN/Inf gradient detected in parameter: {name}")
             if not torch.isfinite(param.data).all():
                 logging.warning(f"  NaN/Inf value detected in parameter: {name}")

        # Strategy: Skip the update for this batch
        logging.warning("Skipping optimizer step due to NaN/Inf loss.")
        optimizer.zero_grad(set_to_none=True) # Zero gradients to prevent propagation, set_to_none saves memory
        # Return a zero tensor for the loss to avoid errors downstream if loss is used after backward
        safe_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=False)

    return skip_step, safe_loss

def diagnose_nan_loss(model, loss_fn, batch, epsilon=1e-10):
    """Diagnose NaN loss by checking intermediate values (Detailed check)."""
    print("\n--- Running NaN Loss Diagnosis ---")
    inputs_ok = True
    # Check batch inputs first
    print("Checking Batch Inputs:")
    for key, value in batch.items():
         if isinstance(value, torch.Tensor):
              if not torch.isfinite(value).all():
                   logging.error(f"NaN/Inf detected in input batch tensor: {key}")
                   inspect_tensor(value, f"Input Batch['{key}']")
                   inputs_ok = False

    if not inputs_ok:
        print("NaN/Inf detected in inputs, aborting further diagnosis.")
        return None

    # Check model outputs
    print("\nChecking Model Outputs:")
    try:
        with torch.no_grad(): # Check outputs without affecting subsequent grad calcs
             outputs = model(batch)
        for key, value in outputs.items():
             if isinstance(value, torch.Tensor):
                  if not torch.isfinite(value).all():
                       logging.error(f"NaN/Inf detected in model output: {key}")
                       inspect_tensor(value, f"Model Output['{key}']")
                       inputs_ok = False
    except Exception as e:
        logging.error(f"Error during model forward pass in NaN diagnosis: {e}")
        return None

    if not inputs_ok:
        print("NaN/Inf detected in model outputs, aborting loss calculation.")
        return None

    # Check loss calculation
    print("\nChecking Loss Calculation:")
    try:
         loss = loss_fn(outputs, batch) # Assumes loss_fn computes total loss
         if not torch.isfinite(loss).all():
              logging.error(f"NaN/Inf detected in final loss value: {loss.item()}")
              print("Investigating loss components (if applicable)...")
              # Add calls to specific debug_loss_function here if loss_fn returns components
              # e.g., debug_specific_loss('fape', model, batch)
         else:
              print(f"Loss calculated successfully: {loss.item():.6f}")
         return loss
    except Exception as e:
         logging.error(f"Error during loss calculation in NaN diagnosis: {e}")
         import traceback
         traceback.print_exc()
         return None

def debug_specific_loss(loss_name, model, batch):
    """Run detailed debug routine for a specific loss component."""
    from src.losses import compute_stable_fape_loss, compute_confidence_loss, compute_angle_loss # Ensure import
    outputs = model(batch)
    print(f"\n--- Debugging Loss Component: {loss_name} ---")
    loss_fn_map = {
        'fape': lambda o, b: compute_stable_fape_loss(o['pred_coords'], b['coordinates'], b['mask']),
        'confidence': lambda o, b: compute_confidence_loss(o['pred_confidence'], o['pred_coords'], b['coordinates'], b['mask']),
        'angle': lambda o, b: compute_angle_loss(o['pred_angles'], b['dihedral_features'], b['mask'])
    }
    if loss_name in loss_fn_map:
         loss_fn = loss_fn_map[loss_name]
         # TODO: Implement detailed debuggers for each loss type here if needed
         # For now, just compute and inspect
         try:
              loss = loss_fn(outputs, batch)
              print(f"{loss_name} loss value:")
              inspect_tensor(loss, loss_name)
              if not torch.isfinite(loss).all():
                   print(f"ERROR: NaN/Inf detected in {loss_name} loss.")
                   # Add more detailed checks specific to the loss function here
                   # e.g., check distances in FAPE, targets in confidence loss etc.
         except Exception as e:
              print(f"ERROR computing {loss_name} loss: {e}")
    else:
        print(f"No specific debugger or function found for loss component: {loss_name}")

```
Loss functions are particularly susceptible to numerical instability:
def debug_loss_function(loss_fn, outputs, batch, epsilon=1e-10):
    """Debug loss function for numerical stability issues."""
    print(f"Debugging loss function: {loss_fn.__name__ if hasattr(loss_fn, '__name__') else type(loss_fn).__name__}")
    
    # Turn on anomaly detection
    with torch.autograd.detect_anomaly():
        try:
            # Compute loss
            loss = loss_fn(outputs, batch)
            print(f"Loss value: {loss.item():.6f}")
            
            # Check for numerical issues
            if torch.isnan(loss).any():
                print("ERROR: Loss contains NaN values")
            if torch.isinf(loss).any():
                print("ERROR: Loss contains Infinite values")
                
            # For FAPE loss, check intermediate values
            if 'compute_fape_loss' in str(loss_fn):
                # Extract coordinates
                pred_coords = outputs['pred_coords'] if isinstance(outputs, dict) else outputs
                true_coords = batch['coordinates'] if 'coordinates' in batch else None
                
                if true_coords is None:
                    print("ERROR: No ground truth coordinates found in batch")
                    return loss
                
                # Check coordinate ranges
                print(f"Predicted coordinates range: {pred_coords.min().item():.6f} to {pred_coords.max().item():.6f}")
                print(f"True coordinates range: {true_coords.min().item():.6f} to {true_coords.max().item():.6f}")
                
                # Compute distances manually for inspection
                batch_size = pred_coords.shape[0]
                distances = []
                
                for b in range(batch_size):
                    # Get valid indices from mask
                    if 'mask' in batch:
                        valid_mask = batch['mask'][b]
                        p_valid = pred_coords[b, valid_mask]
                        t_valid = true_coords[b, valid_mask]
                    else:
                        p_valid = pred_coords[b]
                        t_valid = true_coords[b]
                    
                    # Skip if no valid positions
                    if p_valid.shape[0] == 0:
                        continue
                    
                    # Try simple distance calculation
                    simple_dist = torch.norm(p_valid - t_valid, dim=1)
                    distances.append(simple_dist)
                    
                    print(f"Sample {b} distance stats: min={simple_dist.min().item():.6f}, "
                          f"max={simple_dist.max().item():.6f}, mean={simple_dist.mean().item():.6f}")
            
            # For confidence loss, check target generation
            if 'compute_confidence_loss' in str(loss_fn):
                # Extract confidence predictions
                pred_confidence = outputs['pred_confidence'] if isinstance(outputs, dict) else None
                
                if pred_confidence is not None:
                    print(f"Confidence predictions range: {pred_confidence.min().item():.6f} to "
                          f"{pred_confidence.max().item():.6f}")
                    
                    # Check if predictions are raw logits or probabilities
                    is_logits = pred_confidence.min() < 0 or pred_confidence.max() > 1
                    print(f"Predictions appear to be {'logits' if is_logits else 'probabilities'}")
            
            # Compute gradients
            loss.backward()
            print("Gradient computation completed successfully")
            
            return loss
            
        except Exception as e:
            print(f"ERROR: Loss function failed: {e}")
            import traceback
            traceback.print_exc()
            return None
### 5.2 Gradient Flow Problems

Use `analyze_gradients`, `log_gradient_issues`, and `adaptive_gradient_handling`.

```python
# After loss.backward()
# grad_stats = analyze_gradients(model)
# issues_found = log_gradient_issues(grad_stats)
# adaptive_gradient_handling(model, clip_norm=1.0) # Apply clipping
```

Diagnose where gradients are breaking down:
def analyze_loss_gradients(model, loss_fn, batch):
    """Analyze gradient flow from different loss components."""
    # Store original parameters for restoration
    orig_params = {name: param.clone() for name, param in model.named_parameters()}
    
    try:
        # Forward pass
        outputs = model(batch)
        
        # Compute individual loss components
        loss_components = {}
        grad_norms = {}
        
        if isinstance(loss_fn, dict):
            # Multiple loss functions
            for loss_name, loss_function in loss_fn.items():
                model.zero_grad()
                
                # Compute this loss component
                loss = loss_function(outputs, batch)
                loss_components[loss_name] = loss.item()
                
                # Backward pass for this component only
                loss.backward(retain_graph=True)
                
                # Collect gradient norms for each parameter
                grad_norms[loss_name] = {}
                for param_name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norms[loss_name][param_name] = param.grad.norm().item()
                
                # Clear gradients for next component
                model.zero_grad()
                
        elif callable(loss_fn):
            # Single loss function that might return components
            model.zero_grad()
            
            # Try to get loss components
            try:
                total_loss, components = loss_fn(outputs, batch)
                loss_components = {k: v.item() for k, v in components.items()}
            except:
                # Simple loss without components
                total_loss = loss_fn(outputs, batch)
                loss_components = {'total': total_loss.item()}
            
            # Backward pass
            total_loss.backward()
            
            # Collect gradient norms
            grad_norms['total'] = {}
            for param_name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms['total'][param_name] = param.grad.norm().item()
        
        # Print analysis
        print("Loss component values:")
        for name, value in loss_components.items():
            print(f"  {name}: {value:.6f}")
        
        print("\nGradient contribution analysis:")
        for loss_name, param_grads in grad_norms.items():
            print(f"\n{loss_name} loss affects these parameter groups (showing top 5 by gradient magnitude):")
            
            # Sort parameters by gradient norm
            sorted_params = sorted(param_grads.items(), key=lambda x: x[1], reverse=True)
            
            for param_name, grad_norm in sorted_params[:5]:
                if grad_norm > 1e-6:  # Only show significant gradients
                    print(f"  {param_name}: gradient norm = {grad_norm:.6f}")
                    
            # Count parameters with near-zero gradients
            zero_grad_count = sum(1 for norm in param_grads.values() if norm < 1e-6)
            if zero_grad_count > 0:
                print(f"  {zero_grad_count} parameters have near-zero gradients from this loss component")
        
        return loss_components, grad_norms
    
    finally:
        # Restore original parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(orig_params[name])
### 5.3 Loss Scaling and Weighting Issues
Diagnose problems with multi-component loss functions:
def debug_loss_weighting(model, loss_weights, batch, epochs=5):
    """Debug loss weighting by simulating training with different weights."""
    import copy
    
    # Store original parameters
    orig_model = copy.deepcopy(model)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Loss functions
    from src.losses import compute_fape_loss, compute_confidence_loss, compute_angle_loss
    
    # Function to compute weighted loss
    def compute_weighted_loss(outputs, batch, weights):
        fape_loss = compute_fape_loss(outputs['pred_coords'], batch['coordinates'], batch['mask'])
        conf_loss = compute_confidence_loss(outputs['pred_confidence'], outputs['pred_coords'], 
                                          batch['coordinates'], batch['mask'])
        angle_loss = compute_angle_loss(outputs['pred_angles'], batch['dihedral_features'], batch['mask'])
        
        total_loss = (
            weights['fape'] * fape_loss +
            weights['confidence'] * conf_loss +
            weights['angle'] * angle_loss
        )
        
        return total_loss, {
            'fape': fape_loss.item(),
            'confidence': conf_loss.item(),
            'angle': angle_loss.item(),
            'total': total_loss.item()
        }
    
    # Simulate training
    print(f"Simulating {epochs} training steps with weights: {loss_weights}")
    
    loss_history = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(batch)
        
        # Compute weighted loss
        total_loss, loss_components = compute_weighted_loss(outputs, batch, loss_weights)
        loss_history.append(loss_components)
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Print progress
        if epoch % 1 == 0:
            print(f"Epoch {epoch}:")
            for name, value in loss_components.items():
                print(f"  {name}: {value:.6f}")
    
    # Analyze loss trends
    print("\nLoss component trends:")
    for component in ['fape', 'confidence', 'angle']:
        initial = loss_history[0][component]
        final = loss_history[-1][component]
        change = (final - initial) / initial * 100 if abs(initial) > 1e-6 else 0
        
        print(f"  {component}: {initial:.6f} → {final:.6f} ({change:.2f}% change)")
    
    # Restore original parameters
    model.load_state_dict(orig_model.state_dict())
    
    return loss_history
### 5.4 NaN/Infinity Detection
Focused debugging for numerical instability in loss functions:
def diagnose_nan_loss(model, loss_fn, batch, epsilon=1e-10):
    """Diagnose NaN loss by checking intermediate values."""
    # Enable gradient recording for all tensors
    torch.set_grad_enabled(True)
    
    # Define hooks to capture intermediate tensors
    intermediate_tensors = {}
    
    def capture_hook(name):
        def hook(grad):
            intermediate_tensors[name + "_grad"] = grad.clone()
        return hook
    
    # Forward pass with logging
    try:
        # Get outputs with hooks
        outputs_dict = {}
        
        def run_with_hooks(model, batch):
            outputs = model(batch)
            
            # Store intermediate outputs
            if isinstance(outputs, dict):
                for key, tensor in outputs.items():
                    if isinstance(tensor, torch.Tensor):
                        outputs_dict[key] = tensor
                        # Register hook to capture gradients
                        if tensor.requires_grad:
                            tensor.register_hook(capture_hook(key))
            elif isinstance(outputs, torch.Tensor):
                outputs_dict["output"] = outputs
                if outputs.requires_grad:
                    outputs.register_hook(capture_hook("output"))
            
            return outputs
        
        # Execute model with hooks
        outputs = run_with_hooks(model, batch)
        
        # Add small epsilon to prevent exact zeros
        for key, tensor in outputs_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                # Add epsilon to prevent division by zero
                if torch.any(tensor == 0):
                    print(f"WARNING: Output '{key}' contains exact zeros, adding epsilon for stability")
                    outputs_dict[key] = tensor + epsilon * torch.randn_like(tensor)
        
        # Check outputs for NaN/Inf before loss calculation
        for key, tensor in outputs_dict.items():
            if isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any():
                    print(f"ERROR: NaN detected in output '{key}' before loss calculation")
                if torch.isinf(tensor).any():
                    print(f"ERROR: Inf detected in output '{key}' before loss calculation")
        
        # Compute loss with modified outputs
        loss = loss_fn(outputs, batch)
        
        # Check loss for NaN/Inf
        if torch.isnan(loss).any():
            print("NaN detected in loss value")
            
            # Check inputs to loss function
            print("\nAnalyzing loss function inputs:")
            
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        min_val = value.min().item()
                        max_val = value.max().item()
                        has_nan = torch.isnan(value).any().item()
                        has_inf = torch.isinf(value).any().item()
                        
                        print(f"  {key}: min={min_val:.6f}, max={max_val:.6f}, "
                              f"has_nan={has_nan}, has_inf={has_inf}")
                        
                        if has_nan or has_inf:
                            # Find the first NaN/Inf position
                            if has_nan:
                                nan_indices = torch.where(torch.isnan(value))
                                print(f"    First NaN at index: {[idx[0].item() for idx in nan_indices]}")
                            if has_inf:
                                inf_indices = torch.where(torch.isinf(value))
                                print(f"    First Inf at index: {[idx[0].item() for idx in inf_indices]}")
            
            # For FAPE loss specific debugging
            if 'compute_fape_loss' in str(loss_fn) and 'pred_coords' in outputs:
                pred = outputs['pred_coords']
                true = batch['coordinates'] if 'coordinates' in batch else None
                
                if true is not None:
                    # Check distance calculation
                    dist = torch.norm(pred - true, dim=-1)
                    print(f"  Coordinate distances: min={dist.min().item():.6f}, "
                          f"max={dist.max().item():.6f}, mean={dist.mean().item():.6f}")
                    
                    if torch.isnan(dist).any() or torch.isinf(dist).any():
                        print("  NaN/Inf detected in coordinate distances")
                
                # Check Kabsch alignment
                if 'mask' in batch:
                    mask = batch['mask']
                    for b in range(pred.shape[0]):
                        valid = mask[b]
                        if valid.sum() > 0:
                            p_valid = pred[b, valid]
                            t_valid = true[b, valid] if true is not None else None
                            
                            if t_valid is not None:
                                # Check centroids
                                p_mean = p_valid.mean(dim=0)
                                t_mean = t_valid.mean(dim=0)
                                
                                print(f"  Sample {b} centroids:")
                                print(f"    Pred: {p_mean.tolist()}")
                                print(f"    True: {t_mean.tolist()}")
                                
                                # Check for extreme values in centroids
                                if torch.isnan(p_mean).any() or torch.isinf(p_mean).any():
                                    print("    NaN/Inf detected in predicted centroid")
                                if torch.isnan(t_mean).any() or torch.isinf(t_mean).any():
                                    print("    NaN/Inf detected in true centroid")
        else:
            print(f"Loss computed successfully: {loss.item():.6f}")
        
        return loss, intermediate_tensors
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, intermediate_tensors
# 6. Performance Debugging
### 6.1 Memory Profiling and Optimization
Diagnose and resolve GPU memory issues:
def profile_memory_usage(model, dataloader, disable_grad=True):
    """Profile memory usage during model execution."""
    import gc
    import time
    
    # Clear memory before profiling
    torch.cuda.empty_cache()
    gc.collect()
    
    # Baseline memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        print(f"Baseline GPU memory: {baseline_allocated / 1024**2:.2f}MB allocated, "
              f"{baseline_reserved / 1024**2:.2f}MB reserved")
    
    # Disable gradients if specified
    with torch.set_grad_enabled(not disable_grad):
        # Get a batch
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == 0:
                print(f"\nProfiling batch of size {len(batch['target_ids'])}")
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                
                # Profile forward pass
                torch.cuda.reset_peak_memory_stats()
                start_time = time.time()
                
                try:
                    outputs = model(batch)
                    forward_time = time.time() - start_time
                    
                    # Memory after forward pass
                    if torch.cuda.is_available():
                        forward_allocated = torch.cuda.memory_allocated()
                        forward_reserved = torch.cuda.memory_reserved()
                        forward_peak = torch.cuda.max_memory_allocated()
                        
                        print(f"Forward pass completed in {forward_time:.4f}s")
                        print(f"Forward GPU memory: {forward_allocated / 1024**2:.2f}MB allocated, "
                              f"{forward_reserved / 1024**2:.2f}MB reserved")
                        print(f"Peak memory usage: {forward_peak / 1024**2:.2f}MB")
                        print(f"Memory increase: {(forward_allocated - baseline_allocated) / 1024**2:.2f}MB")
                    
                    # If gradient tracking is enabled, profile backward pass
                    if not disable_grad and isinstance(outputs, dict) and 'pred_coords' in outputs:
                        # Compute dummy loss
                        dummy_loss = outputs['pred_coords'].abs().mean()
                        
                        # Reset stats before backward
                        torch.cuda.reset_peak_memory_stats()
                        backward_start = time.time()
                        
                        # Backward pass
                        dummy_loss.backward()
                        backward_time = time.time() - backward_start
                        
                        # Memory after backward pass
                        if torch.cuda.is_available():
                            backward_allocated = torch.cuda.memory_allocated()
                            backward_peak = torch.cuda.max_memory_allocated()
                            
                            print(f"\nBackward pass completed in {backward_time:.4f}s")
                            print(f"Backward GPU memory: {backward_allocated / 1024**2:.2f}MB allocated")
                            print(f"Backward peak memory usage: {backward_peak / 1024**2:.2f}MB")
                            print(f"Memory increase during backward: "
                                  f"{(backward_allocated - forward_allocated) / 1024**2:.2f}MB")
                    
                except Exception as e:
                    print(f"ERROR: Profiling failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Free memory
                del outputs
                torch.cuda.empty_cache()
                
                # Only profile the first batch
                break
Memory optimization decision tree:
Out-of-memory error detected → Try:
├── Basic fixes
│   ├── Reduce batch size
│   ├── Use smaller model dimensions
│   └── Move to 16-bit precision
│   
├── Intermediates optimization
│   ├── Use inplace operations where possible
│   ├── Delete intermediate tensors explicitly
│   └── Call torch.cuda.empty_cache() strategically
│   
├── Advanced techniques
│   ├── Implement gradient checkpointing
│   ├── Use torch.utils.checkpoint.checkpoint
│   └── Break large operations into smaller chunks
│   
└── Last resort options
    ├── Re-implement with manual memory management
    ├── Offload tensors to CPU selectively
    └── Process data in smaller chunks

## 6.2 GPU Utilization and Bottleneck Identification
Diagnose performance bottlenecks:
def profile_execution_time(model, dataloader, loss_fn=None, n_batches=3):
    """Profile execution time for model components."""
    import time
    from collections import defaultdict
    
    # Timing dictionary
    timings = defaultdict(list)
    
    # Turn off gradients for forward-only profiling
    with torch.no_grad():
        # Warm-up pass
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == 0:
                print("Performing warm-up pass...")
                if torch.cuda.is_available():
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                model(batch)
                break
    
    # Capture forward pass timings with hooks
    module_timings = defaultdict(list)
    
    def timing_hook(name):
        def hook(module, inputs, outputs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for CUDA operations to finish
            module_timings[name].append(time.time())
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)) or 'block' in name:
            # Register pre-forward hook
            pre_hook = module.register_forward_pre_hook(
                lambda m, i, name=name: module_timings[name].append(time.time()))
            hooks.append(pre_hook)
            
            # Register post-forward hook
            post_hook = module.register_forward_hook(timing_hook(name))
            hooks.append(post_hook)
    
    try:
        print(f"Profiling {n_batches} batches...")
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= n_batches:
                break
                
            print(f"\nBatch {batch_idx+1}/{n_batches}:")
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                torch.cuda.synchronize()
            
            # Time data preparation
            data_start = time.time()
            # Data preparation is already done by the dataloader
            data_end = time.time()
            timings['data_preparation'].append(data_end - data_start)
            
            # Time model forward pass
            forward_start = time.time()
            outputs = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_end = time.time()
            timings['forward_pass'].append(forward_end - forward_start)
            
            # Time loss computation if provided
            if loss_fn is not None:
                loss_start = time.time()
                loss = loss_fn(outputs, batch)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                loss_end = time.time()
                timings['loss_computation'].append(loss_end - loss_start)
            
            # Compute module execution times
            for name in module_timings:
                if len(module_timings[name]) >= 2:
                    start = module_timings[name][-2]
                    end = module_timings[name][-1]
                    if batch_idx < n_batches:  # Only record for profiling batches
                        timings[f"module_{name}"].append(end - start)
            
            # Clear module timing entries for this batch
            for name in module_timings:
                # Keep the last entry for each module as it might be the end time
                if len(module_timings[name]) > 0:
                    module_timings[name] = [module_timings[name][-1]]
        
        # Compute average timings
        avg_timings = {}
        for key, values in timings.items():
            if values:  # Check if non-empty
                avg_timings[key] = sum(values) / len(values)
        
        # Print results
        print("\nAverage execution times:")
        total_time = sum(avg_timings.get(key, 0) for key in ['data_preparation', 'forward_pass', 'loss_computation'])
        
        for key, avg_time in sorted(avg_timings.items(), key=lambda x: x[1], reverse=True):
            if key.startswith('module_'):
                # For modules, show percentage of forward pass time
                forward_time = avg_timings.get('forward_pass', 0)
                if forward_time > 0:
                    percentage = avg_time / forward_time * 100
                    print(f"  {key[7:]}: {avg_time*1000:.2f}ms ({percentage:.1f}% of forward pass)")
            else:
                # For main phases, show percentage of total time
                if total_time > 0:
                    percentage = avg_time / total_time * 100
                    print(f"{key}: {avg_time*1000:.2f}ms ({percentage:.1f}% of total)")
        
        # Identify bottlenecks
        if avg_timings:
            bottlenecks = [k for k, v in sorted(avg_timings.items(), key=lambda x: x[1], reverse=True)
                         if k.startswith('module_')][:3]
            
            if bottlenecks:
                print("\nPotential bottlenecks:")
                for b in bottlenecks:
                    print(f"  {b[7:]}: {avg_timings[b]*1000:.2f}ms")
                    
                print("\nOptimization suggestions:")
                for b in bottlenecks:
                    module_name = b[7:]
                    if 'attention' in module_name.lower():
                        print(f"  • Optimize attention computation in {module_name}")
                        print("    - Consider using flash attention if available")
                        print("    - Check for unnecessary tensor copies")
                    elif 'linear' in module_name.lower():
                        print(f"  • Optimize linear layer in {module_name}")
                        print("    - Consider matrix dimension optimization")
                        print("    - Evaluate mixed precision for linear operations")
                    elif 'block' in module_name.lower():
                        print(f"  • Profile sub-components within {module_name}")
                        print("    - Break down timing within this block")
                        print("    - Look for redundant computations")
                    elif 'pair' in module_name.lower():
                        print(f"  • Optimize pair representation update in {module_name}")
                        print("    - Pair operations scale quadratically with sequence length")
                        print("    - Consider sparse implementation for pair update")
        
        return avg_timings
            
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
### 6.3 Computational Efficiency Improvements
Optimize critical operations for performance:
def suggest_performance_optimizations(model, batch):
    """Analyze model for potential performance optimizations."""
    from collections import Counter
    import operator
    
    # Count module types
    module_counts = Counter()
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in ['Sequential', 'ModuleList', 'RNAFoldingModel']:
            module_counts[module_type] += 1
    
    print("Module type distribution:")
    for module_type, count in module_counts.most_common():
        print(f"  {module_type}: {count}")
    
    # Check for memory-intensive operations
    memory_intensive_ops = []
    
    # Check for large tensor operations
    tensor_shapes = {}
    
    def shape_hook(name):
        def hook(module, inputs, outputs):
            # Record input shapes
            for i, inp in enumerate(inputs):
                if isinstance(inp, torch.Tensor):
                    tensor_shapes[f"{name}_input_{i}"] = inp.shape
            
            # Record output shapes
            if isinstance(outputs, torch.Tensor):
                tensor_shapes[f"{name}_output"] = outputs.shape
            elif isinstance(outputs, tuple):
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor):
                        tensor_shapes[f"{name}_output_{i}"] = out.shape
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if any(op in name.lower() for op in ['pair', 'attention', 'ipa']):
            hook = module.register_forward_hook(shape_hook(name))
            hooks.append(hook)
    
    try:
        # Forward pass to collect shapes
        with torch.no_grad():
            model(batch)
        
        # Analyze tensor shapes
        print("\nLarge tensor operations:")
        large_tensors = {}
        
        for name, shape in tensor_shapes.items():
            # Calculate tensor size in MB
            if len(shape) > 0:  # Check for empty shapes
                elements = torch.tensor(shape).prod().item()
                size_mb = elements * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)
                
                if size_mb > 10:  # Only show tensors larger than 10MB
                    large_tensors[name] = (shape, size_mb)
        
        # Sort by size
        for name, (shape, size_mb) in sorted(large_tensors.items(), key=lambda x: x[1][1], reverse=True):
            print(f"  {name}: {shape} ({size_mb:.2f}MB)")
            
            # Suggest optimizations
            if 'pair' in name.lower() and len(shape) >= 3:
                seq_dim = max(shape[1], shape[2]) if len(shape) >= 3 else 0
                
                if seq_dim > 100:
                    memory_intensive_ops.append(f"{name} (large pair tensor: {shape})")
                    print("    - Consider sparse implementation for pair representation")
                    print("    - Evaluate if full N×N tensor is necessary")
                    
            elif 'attention' in name.lower():
                if any(d > 1000 for d in shape):
                    memory_intensive_ops.append(f"{name} (large attention tensor: {shape})")
                    print("    - Consider using flash attention")
                    print("    - Evaluate attention with linear complexity")
        
        # Overall suggestions
        print("\nOptimization suggestions:")
        
        if memory_intensive_ops:
            print("\n1. Memory-intensive operations detected:")
            for op in memory_intensive_ops:
                print(f"  - {op}")
                
            print("\n   Suggestions:")
            print("   - Implement gradient checkpointing")
            print("   - Use mixed precision training (torch.cuda.amp)")
            print("   - Consider sequence chunking for very long RNAs")
        
        # Check for potential optimization areas
        print("\n2. General optimizations:")
        print("   - Use torch.compile() for accelerated execution")
        print("   - Implement efficient attention mechanisms (e.g., flash attention)")
        print("   - Optimize data transfer between CPU and GPU")
        print("   - Consider JIT compilation for critical operations")
        print("   - Evaluate Triton kernels for custom operations")
        
        # Data loading optimizations
        print("\n3. Data pipeline optimizations:")
        print("   - Increase num_workers in DataLoader")
        print("   - Use pin_memory=True for faster CPU to GPU transfers")
        print("   - Pre-compute and cache features when possible")
        print("   - Use prefetching to overlap computation and data loading")
        
        return large_tensors
        
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
### 6.4 Batch Size Optimization
Find the optimal batch size for training:
def optimize_batch_size(model, dataset, start_size=1, max_size=32, step=1):
    """Find the optimal batch size for memory usage and performance."""
    import gc
    from torch.utils.data import DataLoader
    
    if not torch.cuda.is_available():
        print("CUDA not available. Batch size optimization requires GPU.")
        return None
    
    print("Optimizing batch size...")
    print("Starting with batch size", start_size)
    
    results = []
    
    # Test increasing batch sizes
    for batch_size in range(start_size, max_size + 1, step):
        print(f"\nTesting batch size: {batch_size}")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        try:
            # Get a batch
            batch = next(iter(dataloader))
            
            # Record initial memory
            torch.cuda.reset_peak_memory_stats()
            initial_mem = torch.cuda.memory_allocated()
            
            # Move to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(batch)
            
            # Record memory usage
            peak_mem = torch.cuda.max_memory_allocated()
            mem_used = peak_mem - initial_mem
            
            # Display results
            print(f"  Memory used: {mem_used / 1024**2:.2f}MB")
            print(f"  Peak memory: {peak_mem / 1024**2:.2f}MB")
            
            # Record results
            results.append({
                'batch_size': batch_size,
                'memory_used': mem_used,
                'peak_memory': peak_mem,
                'success': True
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Failed: Out of memory at batch size {batch_size}")
                results.append({
                    'batch_size': batch_size,
                    'memory_used': None,
                    'peak_memory': None,
                    'success': False
                })
                
                # We've reached the limit, no need to try larger sizes
                break
            else:
                print(f"  Failed: {e}")
                results.append({
                    'batch_size': batch_size,
                    'memory_used': None,
                    'peak_memory': None,
                    'success': False,
                    'error': str(e)
                })
                
                # Continue to the next batch size
                continue
        
        # Clear memory after test
        del batch, outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    # Find the largest successful batch size
    successful_results = [r for r in results if r['success']]
    if successful_results:
        max_batch_size = max(successful_results, key=lambda x: x['batch_size'])
        print(f"\nRecommended maximum batch size: {max_batch_size['batch_size']}")
        
        # Calculate memory efficiency
        if len(successful_results) > 1:
            # Get memory usage per sample for largest successful batch
            mem_per_sample = max_batch_size['memory_used'] / max_batch_size['batch_size']
            print(f"Memory usage per sample: {mem_per_sample / 1024**2:.2f}MB")
            
            # Check for anomalies in scaling
            print("\nMemory scaling analysis:")
            for r in successful_results:
                mem_per_sample_r = r['memory_used'] / r['batch_size']
                print(f"  Batch size {r['batch_size']}: {mem_per_sample_r / 1024**2:.2f}MB per sample")
        
        # Suggest batch size with safety margin
        safe_batch_size = int(max_batch_size['batch_size'] * 0.9)
        safe_batch_size = max(1, safe_batch_size)
        print(f"\nRecommended batch size with safety margin: {safe_batch_size}")
    else:
        print("No successful batch size found. Try reducing model size or optimization.")
    
    return results
# 7. Debugging Case Studies
### 7.1 Case Study: Shape Mismatch in the Transformer Block
"""
Problem: RuntimeError: The size of tensor a (128) must match the size of tensor b (64) at non-singleton dimension 2
Location: TransformerBlock._update_pair_repr method
"""

```python
# Diagnostic approach for Shape Mismatch:
# ... (isolate transformer block) ...
# shapes = debug_shape_flow(model, batch) # Use shape tracer
# ... (analyze shapes dict) ...
# print("\nInspecting inputs to the failing operation using inspect_tensor:")
# inspect_tensor(tensor_a, "Tensor A")
# inspect_tensor(tensor_b, "Tensor B")
# ... (suggest fixes) ...
```

# Diagnostic approach:
def debug_transformer_shape_mismatch(model, batch):
    # 1. Isolate the transformer block
    for name, module in model.named_modules():
        if isinstance(module, TransformerBlock):
            transformer_block = module
            break
    
    # 2. Set up shape tracing hooks
    shapes = {}
    
    def shape_hook(name):
        def hook(module, inputs, outputs):
            # Record input shapes
            shapes[f"{name}_inputs"] = [x.shape if isinstance(x, torch.Tensor) else x for x in inputs]
            
            # Record output shape
            if isinstance(outputs, torch.Tensor):
                shapes[f"{name}_output"] = outputs.shape
            elif isinstance(outputs, tuple):
                shapes[f"{name}_outputs"] = [x.shape if isinstance(x, torch.Tensor) else x for x in outputs]
        return hook
    
    # 3. Add hooks to critical methods
    hooks = []
    target_methods = ['_update_residue_repr', '_update_pair_repr']
    
    for method_name in target_methods:
        if hasattr(transformer_block, method_name):
            method = getattr(transformer_block, method_name)
            hook = method.register_forward_hook(shape_hook(method_name))
            hooks.append(hook)
    
    # 4. Trace the forward pass
    try:
        with torch.no_grad():
            # Get first transformer block outputs
            residue_repr = batch['residue_repr'] if 'residue_repr' in batch else None
            pair_repr = batch['pair_repr'] if 'pair_repr' in batch else None
            mask = batch['mask'] if 'mask' in batch else None
            
            if residue_repr is None or pair_repr is None:
                # Extract from embeddings
                # (This would depend on your actual model structure)
                pass
            
            outputs = transformer_block(residue_repr, pair_repr, mask)
    
    except Exception as e:
        print(f"Error during transformer execution: {e}")
    
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    # 5. Analyze the shapes to find mismatch
    print("Shape analysis:")
    for name, shape in shapes.items():
        print(f"  {name}: {shape}")
    
    # 6. Look for the specific mismatch
    if '_update_pair_repr_inputs' in shapes:
        inputs = shapes['_update_pair_repr_inputs']
        
        # Check for shape inconsistencies
        if len(inputs) >= 3:
            residue_shape = inputs[0].shape if isinstance(inputs[0], torch.Tensor) else None
            pair_shape = inputs[1].shape if isinstance(inputs[1], torch.Tensor) else None
            
            if residue_shape and pair_shape:
                residue_dim = residue_shape[-1] if len(residue_shape) >= 2 else None
                pair_dim = pair_shape[-1] if len(pair_shape) >= 3 else None
                
                print(f"\nPotential shape mismatch:")
                print(f"  Residue embedding dimension: {residue_dim}")
                print(f"  Pair embedding dimension: {pair_dim}")
                
                # Check against model config
                print("\nComparing with model configuration:")
                if hasattr(transformer_block, 'residue_dim') and hasattr(transformer_block, 'pair_dim'):
                    print(f"  Expected residue dimension: {transformer_block.residue_dim}")
                    print(f"  Expected pair dimension: {transformer_block.pair_dim}")
                    
                    if residue_dim != transformer_block.residue_dim:
                        print(f"  MISMATCH: Residue dimensions don't match!")
                    if pair_dim != transformer_block.pair_dim:
                        print(f"  MISMATCH: Pair dimensions don't match!")
    
    # 7. Check the model configuration
    print("\nInspecting transformer block configuration:")
    for attr_name in ['residue_dim', 'pair_dim', 'num_heads', 'dropout_rate', 'ffn_dim']:
        if hasattr(transformer_block, attr_name):
            print(f"  {attr_name}: {getattr(transformer_block, attr_name)}")
    
    # 8. Suggest a fix
    print("\nPossible fix for shape mismatch:")
    print("  1. Ensure the 'residue_embed_dim' and 'pair_embed_dim' in the config match")
    print("  2. Check the linear projections in the _init_pair_update_components method")
    print("  3. Verify the pair input calculation: 2 * self.residue_dim + self.pair_dim")
    
    return shapes
    

### 7.2 Case Study: NaN Loss in FAPE Calculation
"""
Problem: Loss becomes NaN after a few iterations of training
Location: compute_fape_loss function in src/losses.py
"""
```python
# Diagnostic approach for NaN Loss:
# diagnose_nan_loss(model, compute_combined_loss, batch_causing_nan) # Overall check
# debug_specific_loss('fape', model, batch_causing_nan) # Detailed FAPE check
# Enable detect_anomaly:
# with torch.autograd.detect_anomaly():
#    loss = compute_combined_loss(...)
#    loss.backward()
# ... (suggest fixes) ...
```

# Diagnostic approach:
def debug_nan_fape_loss(batch, detailed=True):
    from src.losses import compute_fape_loss
    
    # 1. Extract coordinates
    pred_coords = batch['pred_coords'] if 'pred_coords' in batch else None
    true_coords = batch['coordinates'] if 'coordinates' in batch else None
    mask = batch['mask'] if 'mask' in batch else None
    
    if pred_coords is None or true_coords is None:
        print("ERROR: Missing coordinate tensors in batch")
        return
    
    # 2. Examine coordinate statistics
    print("Predicted coordinates:")
    inspect_tensor(pred_coords, "pred_coords", full_stats=True)
    
    print("\nTrue coordinates:")
    inspect_tensor(true_coords, "true_coords", full_stats=True)
    
    # 3. Check for exact overlaps or extreme values
    if detailed:
        # Look for identical coordinates (potential Kabsch problem)
        for b in range(pred_coords.shape[0]):
            if mask is not None:
                valid = mask[b]
                if not torch.any(valid):
                    print(f"WARNING: Sample {b} has no valid positions")
                    continue
                
                p_valid = pred_coords[b, valid]
                t_valid = true_coords[b, valid]
            else:
                p_valid = pred_coords[b]
                t_valid = true_coords[b]
            
            # Check for identical points
            if p_valid.shape[0] <= 1:
                print(f"WARNING: Sample {b} has only {p_valid.shape[0]} valid positions")
                continue
            
            # Check for zero-distance points in predicted coordinates
            pdist = torch.cdist(p_valid, p_valid)
            min_dist = torch.min(pdist + torch.eye(p_valid.shape[0], device=p_valid.device) * 1e6)
            if min_dist < 1e-6:
                print(f"WARNING: Sample {b} has identical predicted points (min dist: {min_dist.item():.8f})")
            
            # Check for zero-distance points in true coordinates
            tdist = torch.cdist(t_valid, t_valid)
            min_dist = torch.min(tdist + torch.eye(t_valid.shape[0], device=t_valid.device) * 1e6)
            if min_dist < 1e-6:
                print(f"WARNING: Sample {b} has identical true points (min dist: {min_dist.item():.8f})")
            
            # Check for all-zero coordinates
            if torch.all(p_valid == 0):
                print(f"WARNING: Sample {b} has all-zero predicted coordinates")
            if torch.all(t_valid == 0):
                print(f"WARNING: Sample {b} has all-zero true coordinates")
    
    # 4. Trace through FAPE loss computation with extra instrumentation
    print("\nTracing FAPE loss computation with debugging...")
    
    try:
        # Create cloned tensors to avoid modifying inputs
        p_coords = pred_coords.clone()
        t_coords = true_coords.clone()
        
        # Add small jitter to prevent degenerate cases
        p_coords = p_coords + torch.randn_like(p_coords) * 1e-6
        
        # Try different clamp values
        clamp_values = [1.0, 5.0, 10.0, 20.0]
        
        for clamp_value in clamp_values:
            print(f"\nTrying with clamp_value={clamp_value}:")
            
            try:
                loss = compute_fape_loss(p_coords, t_coords, mask, clamp_value=clamp_value)
                print(f"  Loss computed successfully: {loss.item():.6f}")
            except Exception as e:
                print(f"  ERROR: {e}")
                
                # Try to pinpoint the issue
                if 'kabsch_align' in str(e):
                    print("\nDebugging Kabsch alignment:")
                    
                    for b in range(p_coords.shape[0]):
                        if mask is not None:
                            valid = mask[b]
                            if not torch.any(valid):
                                print(f"  Sample {b}: No valid positions")
                                continue
                                
                            p_valid = p_coords[b, valid]
                            t_valid = t_coords[b, valid]
                        else:
                            p_valid = p_coords[b]
                            t_valid = t_coords[b]
                        
                        try:
                            # Try simple centering
                            p_mean = p_valid.mean(dim=0, keepdim=True)
                            t_mean = t_valid.mean(dim=0, keepdim=True)
                            p_centered = p_valid - p_mean
                            t_centered = t_valid - t_mean
                            
                            print(f"  Sample {b}:")
                            print(f"    Points: {p_valid.shape[0]}")
                            print(f"    Pred centroid: {p_mean.squeeze().tolist()}")
                            print(f"    True centroid: {t_mean.squeeze().tolist()}")
                            
                            if torch.isnan(p_mean).any() or torch.isnan(t_mean).any():
                                print("    WARNING: NaN detected in centroids")
                            
                            # Try computing covariance matrix
                            C = torch.matmul(p_centered.transpose(-2, -1), t_centered)
                            print(f"    Covariance matrix shape: {C.shape}")
                            print(f"    Has NaN: {torch.isnan(C).any().item()}")
                            print(f"    Has zero: {torch.all(C == 0).item()}")
                            
                            if not torch.isnan(C).any() and not torch.all(C == 0):
                                # Try SVD
                                try:
                                    U, S, Vt = torch.linalg.svd(C)
                                    print(f"    SVD computed successfully")
                                    print(f"    Singular values: {S.tolist()}")
                                    
                                    # Check for zero singular values
                                    if torch.any(S < 1e-10):
                                        print("    WARNING: Near-zero singular values detected")
                                        
                                    # Check for reflection case
                                    V = Vt.transpose(-2, -1)
                                    det = torch.det(torch.matmul(V, U.transpose(-2, -1)))
                                    print(f"    Determinant: {det.item():.6f}")
                                    if det < 0:
                                        print("    Reflection case detected")
                                        
                                except Exception as svd_error:
                                    print(f"    SVD computation failed: {svd_error}")
                        
                        except Exception as sample_error:
                            print(f"    Error processing sample {b}: {sample_error}")
        
        # 5. Provide fix recommendations
        print("\nPotential fixes for NaN FAPE loss:")
        print("  1. Add a small epsilon to prevent degenerate Kabsch alignment")
        print("  2. Use a larger clamp value to avoid extreme gradients")
        print("  3. Ensure mask is properly applied to exclude invalid positions")
        print("  4. Add a small regularization to prevent coordinate collapses")
        print("  5. Check for zero-distance atoms in the structure")
        print("  6. Implement more robust SVD handling in the Kabsch algorithm")
                
    except Exception as e:
        print(f"Error during loss tracing: {e}")
        import traceback
        traceback.print_exc()
        
# 8. Integration with Testing Workflow
## 8.1 Using Test Failures to Guide Debugging
Connect testing and debugging workflows:
def debug_test_failure(test_name, error_message):
    """Guide debugging based on test failure type."""
    print(f"Debugging test failure: {test_name}")
    print(f"Error message: {error_message}")
    
    # Categorize the error
    if "size mismatch" in error_message or "shape" in error_message:
        error_type = "shape_mismatch"
    elif "CUDA out of memory" in error_message:
        error_type = "memory_error"
    elif "NaN" in error_message or "nan" in error_message:
        error_type = "numerical_error"
    elif "device" in error_message and ("CPU" in error_message or "cuda" in error_message):
        error_type = "device_error"
    elif "ModuleNotFoundError" in error_message or "ImportError" in error_message:
        error_type = "import_error"
    elif "AssertionError" in error_message:
        error_type = "assertion_failure"
    else:
        error_type = "unknown"
    
    print(f"Detected error type: {error_type}")
    
    # Suggest debugging approaches based on error type
    if error_type == "shape_mismatch":
        print("\nDebugging suggestions for shape mismatch:")
        print("1. Locate the specific tensors with mismatched shapes")
        print("2. Trace tensor shapes through the model using debug_shape_flow()")
        print("3. Check configuration parameter consistency")
        print("4. Verify padding and mask handling")
        print("5. Inspect tensor transformations (especially reshaping, permute, transpose)")
        
        # Extract tensor shapes from error message if available
        import re
        shapes = re.findall(r'size (\d+)', error_message)
        if len(shapes) >= 2:
            print(f"\nMismatched dimensions: {shapes[0]} vs {shapes[1]}")
            print("Check for parameter mismatches in config:")
            print("  - residue_embed_dim")
            print("  - pair_embed_dim")
            print("  - ffn_dim")
        
    elif error_type == "memory_error":
        print("\nDebugging suggestions for memory error:")
        print("1. Reduce batch size")
        print("2. Profile memory usage with profile_memory_usage()")
        print("3. Check for memory leaks in loops")
        print("4. Examine large tensor operations")
        print("5. Consider gradient checkpointing")
        print("6. Use torch.cuda.empty_cache() strategically")
        
    elif error_type == "numerical_error":
        print("\nDebugging suggestions for numerical error (NaN/Inf):")
        print("1. Identify where NaN values first appear using diagnose_nan_loss()")
        print("2. Check for division by zero or log of zero")
        print("3. Examine loss function stability")
        print("4. Verify initialization of model parameters")
        print("5. Inspect gradient scaling and normalization")
        print("6. Add small epsilon terms to prevent numerical instability")
        
    elif error_type == "device_error":
        print("\nDebugging suggestions for device error:")
        print("1. Ensure all tensors in an operation are on the same device")
        print("2. Check for tensors not moved to GPU in the forward pass")
        print("3. Verify mask and target tensors are on the same device as inputs")
        print("4. Examine module parameter placement")
        print("5. Review tensor creation operations that may default to CPU")
        
    elif error_type == "import_error":
        print("\nDebugging suggestions for import error:")
        print("1. Verify the conda environment is activated")
        print("2. Check package dependencies in environment.yml")
        print("3. Ensure correct module paths and names")
        print("4. Validate import statements")
        
    elif error_type == "assertion_failure":
        print("\nDebugging suggestions for assertion failure:")
        print("1. Examine the specific test assertion that failed")
        print("2. Check if the expected behavior matches implementation")
        print("3. Verify calculation logic for expected values")
        print("4. Compare outputs against reference values")
        
    else:
        print("\nGeneral debugging suggestions:")
        print("1. Break the problem down by isolating components")
        print("2. Add print statements for intermediate values")
        print("3. Create minimal test case that reproduces the error")
        print("4. Examine stack trace for error location")
        
    print("\nNext steps:")
    print("1. Run the test in isolation with pytest -v tests/test_file.py::test_function")
    print("2. Add print statements or debugging code to narrow down the issue")
    print("3. Create a minimal example to reproduce the error")
    print("4. Apply fixes and verify with the test case")
    
    # Extract file and line info if available
    import re
    file_line_match = re.search(r'File "([^"]+)", line (\d+),', error_message)
    if file_line_match:
        file_path, line_num = file_line_match.groups()
        print(f"\nError location: {file_path}:{line_num}")
        
        # Suggest relevant function to debug
        if "transformer_block" in file_path.lower():
            print("Consider using debug_transformer_shape_mismatch() for detailed inspection")
        elif "loss" in file_path.lower():
            print("Consider using debug_nan_loss() or debug_loss_function() for detailed inspection")
        elif "data_loading" in file_path.lower():
            print("Consider using validate_data_pipeline() for detailed inspection")
        elif "ipa_module" in file_path.lower():
            print("Consider using debug_component() with the IPAModule for detailed inspection")
## 8.2 Regression Testing After Fixes
Verify that fixes don't introduce new issues:
def run_regression_tests(module_name, component=None):
    """Run regression tests to ensure fixes don't break other functionality."""
    import sys
    import subprocess
    
    print(f"Running regression tests for {module_name}")
    
    # Determine test file path based on module name
    test_file = f"tests/test_{module_name}.py"
    
    # Specific component to test
    component_flag = f"-k {component}" if component else ""
    
    # Run pytest with appropriate flags
    command = f"pytest {test_file} {component_flag} -v"
    print(f"Executing: {command}")
    
    # Execute the tests
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Print output
    print("\nTest output:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("\nTests failed! Error output:")
        print(result.stderr)
        
        # Extract new failures
        import re
        failures = re.findall(r'FAILED\s+([^\s]+)', result.stdout)
        if failures:
            print(f"\nFound {len(failures)} test failures:")
            for failure in failures:
                print(f"  - {failure}")
        
        return False
    else:
        print("\nAll tests passed successfully!")
        return True
## 8.3 Creating Test Cases from Discovered Bugs
Institutionalize knowledge from debugging:
def create_test_case_for_bug(bug_details, output_file=None):
    """Generate a test case template based on bug details."""
    
    # Extract bug information
    bug_module = bug_details.get("module", "unknown")
    bug_component = bug_details.get("component", "Unknown")
    bug_description = bug_details.get("description", "Undescribed bug")
    reproduction_steps = bug_details.get("reproduction_steps", ["No steps provided"])
    expected_behavior = bug_details.get("expected_behavior", "Unknown")
    fix_description = bug_details.get("fix", "No fix documented")
    
    # Generate test case code
    test_code = f"""
import pytest
import torch
import numpy as np
from src.{bug_module} import {bug_component}

class Test{bug_component}Regression:
    \"\"\"
    Regression tests for {bug_component} to prevent reoccurrence of bugs.
    \"\"\"
    
    @pytest.fixture
    def setup_test_case(self):
        \"\"\"Set up the test case.\"\"\"
        # TODO: Implement setup code based on reproduction steps
        pass
        
    def test_{bug_module}_{bug_component.lower()}_regression(self, setup_test_case):
        \"\"\"
        Test case to verify bug fix for: {bug_description}
        
        Original bug:
        {bug_description}
        
        Fix applied:
        {fix_description}
        \"\"\"
        # TODO: Implement test based on reproduction steps
        
        # Setup
        # [Insert setup code here]
        
        # Execute
        # [Insert execution code here]
        
        # Assert
        # [Insert assertions here]
        
        # This test should fail if the bug reappears
        assert True  # Replace with actual assertion
"""
    
    # Add reproduction steps as comments
    reproduction_code = "\n        # Reproduction steps:\n"
    for i, step in enumerate(reproduction_steps, 1):
        reproduction_code += f"        # {i}. {step}\n"
    
    test_code = test_code.replace("# TODO: Implement test based on reproduction steps", reproduction_code)
    
    # Add expected behavior
    test_code = test_code.replace("# [Insert assertions here]", 
                                 f"# Expected behavior: {expected_behavior}\n        # [Insert assertions here]")
    
    # Write to file or print
    if output_file:
        with open(output_file, 'w') as f:
            f.write(test_code)
        print(f"Test case template written to {output_file}")
    else:
        print("Generated test case template:")
        print(test_code)
        
    print("\nInstructions:")
    print("1. Complete the test case by filling in the TODOs")
    print("2. Ensure the test fails before applying the fix")
    print("3. Verify the test passes after applying the fix")
    print("4. Add the test to the test suite to prevent regression")
    
    return test_code
## 9.Conclusion
This debugging workflow provides a comprehensive framework for identifying, diagnosing, and resolving issues across the RNA 3D folding pipeline. By following these systematic approaches, developers can efficiently troubleshoot problems in data loading, model architecture, loss functions, and performance.
Remember that debugging is as much art as science, requiring both rigorous methodology and creative problem-solving. Always start with isolation, proceed with evidence-based investigation, and validate your solutions thoroughly.
When encountering new issues, contribute your discoveries and solutions back to this guide, helping build a collective knowledge base that makes the RNA 3D folding pipeline more robust and maintainable over time.

