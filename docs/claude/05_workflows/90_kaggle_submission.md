# Kaggle Submission Workflow for RNA 3D Folding

## 1. Overview & Submission Philosophy

### 1.1 Introduction to Kaggle Submission Process

The Stanford RNA 3D Folding Kaggle competition requires submitting predictions for RNA 3D structures based on sequence data. Unlike traditional Kaggle competitions where you might upload a CSV file, this competition requires submitting a notebook that generates the predictions. This approach brings unique challenges for porting our PyTorch-based model from local development to the Kaggle environment.

This document provides a comprehensive guide for converting our locally developed RNA 3D folding pipeline into a successful Kaggle submission, ensuring:

- Code compatibility with Kaggle's execution environment
- Proper file path handling between environments
- Efficient resource utilization within Kaggle constraints
- Correct output format generation for competition evaluation
- Reproducible results meeting competition requirements

### 1.2 Local Development vs. Kaggle Environment Relationship

Our development approach follows a "develop locally, deploy to Kaggle" philosophy with these key relationships:

| Local Development | Kaggle Submission |
|-------------------|-------------------|
| Full repository with modular `src/` | Notebook imports from uploaded `src/` modules |
| Multiple script files for training/inference | Single notebook with sequential execution |
| Docker container environment | Kaggle's pre-configured environment |
| Multiple GPUs potential | Single GPU with memory constraints |
| File paths via config/arguments | Hardcoded Kaggle paths (`/kaggle/input/`, `/kaggle/working/`) |
| Long-running training processes | Time-limited notebook execution (9 hours max) |

### 1.3 Key Principles for Successful Submissions

1. **Strict Path Parameterization**: Core functionality in `src/` modules should never contain hardcoded paths but accept paths as arguments.

2. **Modularity**: Separation between core logic (`src/`) and orchestration (notebook cells). This allows importing `src/` modules without modification.

3. **Configuration Flexibility**: Configuration parameters should be adaptable from local values to Kaggle-optimized values.

4. **Resource Efficiency**: Kaggle has specific memory and computation constraints that require optimization.

5. **Submission Format Precision**: RNA structure prediction requires exact formatting for the 5-prediction output (residue coordinates).

6. **Reproducibility**: Results should be deterministic with fixed random seeds.

## 2. Environment Differences

### 2.1 Local Development vs. Kaggle Environment

| Aspect | Local Development | Kaggle Environment |
|--------|-------------------|-------------------|
| **Hardware** | Custom (e.g., RTX 4070 Ti 16GB) | P100 16GB GPU (or similar) |
| **Storage** | Unlimited local storage | Limited to 20GB in `/kaggle/working/` |
| **Runtime** | Unlimited | 9 hours maximum notebook runtime |
| **File System** | Repository structure | `/kaggle/input/` (read-only), `/kaggle/working/` (writable) |
| **Environment** | Docker container with specific dependencies | Kaggle base environment + custom package installation |
| **Execution** | Separate scripts for different functions | Single notebook with sequential execution |
| **Parallelism** | Potential multi-GPU training | Single GPU inference |
| **Debugging** | Rich debugging tooling | Limited debugging capabilities |

### 2.2 Hardware and Resource Limitations

Kaggle competitions typically provide:
- Single GPU (P100 with 16GB VRAM)
- Limited CPU RAM (approximately 30GB)
- 2 CPU cores
- 20GB of writable disk space

These limitations significantly impact our modeling approach, requiring:
- Memory-efficient implementations
- Reduced batch sizes
- Potential model parameter reduction
- Efficient checkpoint handling

### 2.3 Runtime Constraints and Time Limits

Kaggle notebooks have a 9-hour maximum runtime, which presents challenges for:
- Training from scratch (not feasible)
- Multiple inference runs for the 5-prediction requirement
- Large-scale data processing

Our submission strategy should focus on efficient inference using pre-trained model weights, with careful attention to execution time for each component.

### 2.4 File System Differences and Access Patterns

Kaggle uses a specific file system structure:
- `/kaggle/input/[competition-name]/` - Read-only directory containing competition data
- `/kaggle/input/[your-dataset]/` - Read-only directory for your uploaded datasets (including code modules)
- `/kaggle/working/` - Writable directory for intermediate files and outputs

This differs significantly from our local repository structure and requires careful path handling.

## 3. Code Preparation & Adaptation

### 3.1 Repository to Notebook Conversion

Converting our modular repository to a Kaggle notebook requires:

1. **Upload Core Modules**: Package `src/` directory as a dataset:

```bash
# Local command to prepare src/ for upload
cd /path/to/project
zip -r src.zip src/
# Upload src.zip as a dataset on Kaggle
```

2. **Structure Notebook Cells**: Organize cells in logical execution order:

```
Cell 1: Environment Setup (imports, configurations)
Cell 2: Data Loading and Preparation
Cell 3: Model Configuration and Instantiation
Cell 4: Inference Pipeline
Cell 5: Output Formatting and Submission Generation
```

3. **Maintain Logic Separation**: Keep the same separation between core logic and orchestration:

```python
# CORRECT - Import from src/ modules
from src.models.rna_folding_model import RNAFoldingModel
from src.data_loading import load_precomputed_features

# INCORRECT - Copy-paste implementation into notebook
class RNAFoldingModel(nn.Module):  # Don't do this!
    def __init__(...):
        ...
```

### 3.2 Module Import Strategy

To import modules from our uploaded `src/` package:

```python
# Add the Kaggle dataset path to Python path
import sys
sys.path.append('/kaggle/input/rna3d-src-package')  # Adjust based on your dataset name

# Now import from src as normal
from src.models.rna_folding_model import RNAFoldingModel
from src.data_loading import RNADataset, collate_fn
from src.losses import compute_fape_loss
```

Verify imports with a simple test:

```python
# Verify module imports
try:
    from src.models.rna_folding_model import RNAFoldingModel
    print("✅ Module imports working correctly")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Check that src/ is uploaded correctly and path is correct")
```

### 3.3 Path Modification for Kaggle's File System

Our strict path parameterization makes this adaptation straightforward:

```python
# Define Kaggle-specific paths
COMPETITION_DATA_PATH = '/kaggle/input/stanford-ribonanza-rna-folding-3d'
MODEL_WEIGHTS_PATH = '/kaggle/input/rna3d-model-weights'
FEATURES_DIR = '/kaggle/input/rna3d-precomputed-features'
OUTPUT_PATH = '/kaggle/working/submission.csv'

# Pass these paths to src/ functions that expect path arguments
dataset = RNADataset(
    sequences_csv_path=f'{COMPETITION_DATA_PATH}/test_sequences.csv',
    labels_csv_path=None,  # No labels for test data
    features_dir=FEATURES_DIR
)

# Load model weights
model = RNAFoldingModel(config)
model.load_state_dict(torch.load(f'{MODEL_WEIGHTS_PATH}/model_weights.pt'))
```

### 3.4 Configuration Parameter Adjustment

Adjust model configuration for Kaggle's constraints:

```python
# Local development config
local_config = {
    'residue_embed_dim': 256,
    'pair_embed_dim': 128,
    'num_transformer_blocks': 8,
    'batch_size': 8,
    # ...other parameters
}

# Kaggle-optimized config
kaggle_config = {
    'residue_embed_dim': 128,  # Reduced for memory efficiency
    'pair_embed_dim': 64,
    'num_transformer_blocks': 6,
    'batch_size': 2,  # Smaller batch size for GPU memory
    # ...other adjusted parameters
}

# Use Kaggle config
config = kaggle_config
```

Monitor memory usage to ensure you stay within limits:

```python
def log_memory_usage():
    """Log GPU and CPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    import psutil
    print(f"CPU RAM used: {psutil.virtual_memory().used / 1024**3:.2f} GB")

# Call this function at key points in your notebook
log_memory_usage()
```

## 4. Submission Format Requirements

### 4.1 Output Format Specifications

The RNA 3D folding competition requires predictions in a specific CSV format:

```
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5
target_1,G,1,10.234,5.678,2.901,10.245,5.682,2.922,10.210,5.691,2.888,10.251,5.671,2.905,10.228,5.679,2.898
target_1,A,2,12.456,8.912,3.567,12.461,8.926,3.541,12.459,8.901,3.582,12.470,8.919,3.561,12.451,8.909,3.570
...
```

Key requirements:
- ID format: `{target_id}_{position}`
- Generate predictions for all 5 model outputs
- Match resname and resid exactly
- Provide coordinates with proper precision

### 4.2 Generating the submission.csv file

Here's a function to generate the required submission format:

```python
def generate_submission_csv(predictions, target_ids, sequences, output_path):
    """
    Generate submission CSV in the required format.
    
    Args:
        predictions: Dictionary mapping target_ids to numpy arrays of shape (5, seq_len, 3)
                    representing the 5 sets of predicted coordinates
        target_ids: List of target IDs
        sequences: Dictionary mapping target_ids to sequence strings
        output_path: Path to save submission CSV
    """
    import pandas as pd
    import numpy as np
    
    # Nucleotide to resname mapping
    nuc_to_resname = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U', 'T': 'U'}
    
    # Collect all rows for submission
    rows = []
    
    for target_id in target_ids:
        # Get sequence and predicted coordinates
        sequence = sequences[target_id]
        coords = predictions[target_id]  # Shape: (5, seq_len, 3)
        
        # For each residue position
        for i in range(len(sequence)):
            # Create row dictionary
            row = {
                'ID': f"{target_id}_{i+1}",  # 1-indexed position
                'resname': nuc_to_resname.get(sequence[i], 'X'),
                'resid': i + 1  # 1-indexed
            }
            
            # Add coordinates for all 5 predictions
            for model_idx in range(5):
                x, y, z = coords[model_idx, i]
                row[f'x_{model_idx+1}'] = x.item() if isinstance(x, (np.ndarray, torch.Tensor)) else float(x)
                row[f'y_{model_idx+1}'] = y.item() if isinstance(y, (np.ndarray, torch.Tensor)) else float(y)
                row[f'z_{model_idx+1}'] = z.item() if isinstance(z, (np.ndarray, torch.Tensor)) else float(z)
            
            rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"Submission saved to {output_path}")
    print(f"Submission contains {len(df)} rows for {len(target_ids)} targets")
    
    return df
```

### 4.3 Format Validation Before Submission

Always validate your submission format before submitting:

```python
def validate_submission_format(submission_path):
    """
    Validate the format of a submission CSV file.
    
    Args:
        submission_path: Path to submission CSV
    
    Returns:
        True if format is valid, raises ValueError otherwise
    """
    import pandas as pd
    
    # Load submission
    try:
        df = pd.read_csv(submission_path)
    except Exception as e:
        raise ValueError(f"Failed to read submission CSV: {e}")
    
    # Check required columns
    required_columns = ['ID', 'resname', 'resid'] + [
        f'{coord}_{model}' for coord in ['x', 'y', 'z'] for model in range(1, 6)
    ]
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check ID format
    id_pattern = r'^.+_\d+$'
    invalid_ids = df[~df['ID'].str.match(id_pattern)]['ID'].unique()
    if len(invalid_ids) > 0:
        raise ValueError(f"Invalid ID format in {len(invalid_ids)} rows. Example: {invalid_ids[:5]}")
    
    # Check resname values
    valid_resnames = {'A', 'C', 'G', 'U'}
    invalid_resnames = df[~df['resname'].isin(valid_resnames)]['resname'].unique()
    if len(invalid_resnames) > 0:
        raise ValueError(f"Invalid resnames: {invalid_resnames}")
    
    # Check resid values
    if not all(df['resid'] > 0):
        raise ValueError("All resid values should be positive integers")
    
    # Check coordinates are numeric
    for col in [f'{coord}_{model}' for coord in ['x', 'y', 'z'] for model in range(1, 6)]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} contains non-numeric values")
    
    # Verify each target has complete residues
    targets = set([id.split('_')[0] for id in df['ID']])
    for target in targets:
        target_rows = df[df['ID'].str.startswith(f"{target}_")]
        resids = target_rows['resid'].values
        if len(resids) != len(set(resids)):
            raise ValueError(f"Duplicate resid values for target {target}")
        
        # Check for gaps in resid numbering
        if set(resids) != set(range(1, max(resids) + 1)):
            raise ValueError(f"Gaps in resid numbering for target {target}")
    
    print(f"✅ Submission format validated successfully")
    print(f"Submission contains {len(df)} rows for {len(targets)} targets")
    return True
```

### 4.4 Common Formatting Errors

| Error | Cause | Solution |
|-------|-------|----------|
| Missing columns | Incorrect column naming | Use exact column names: 'x_1', 'y_1', etc. |
| Invalid ID format | Not following target_id_position pattern | Ensure IDs follow format: "target_1", "target_2" |
| Incorrect resid values | Not using 1-indexed positions | Convert to 1-indexed positions in submission |
| Non-numeric coordinates | String values in coordinate columns | Convert all coordinates to floating-point values |
| Coordinate precision issues | Insufficient decimal places | Use float formatting without rounding |
| Duplicate entries | Multiple rows for same position | Check for duplicates in ID generation |
| Missing entries | Incomplete processing of test set | Verify all test sequences are processed |

## 5. Submission Testing & Validation

### 5.1 Local Validation Before Kaggle Submission

Before submitting to Kaggle, validate your pipeline locally:

```python
def run_local_validation():
    """Run validation tests on a small subset of data locally."""
    import torch
    import numpy as np
    from src.models.rna_folding_model import RNAFoldingModel
    from src.data_loading import RNADataset, collate_fn
    
    # 1. Load a small validation set
    print("Loading validation data...")
    val_dataset = RNADataset(
        sequences_csv_path='data/validation_sequences.csv',
        labels_csv_path='data/validation_labels.csv',
        features_dir='data/processed',
        use_validation_set=True
    )
    
    # Use only a few samples
    subset_indices = list(range(min(5, len(val_dataset))))
    subset_targets = [val_dataset.target_ids[i] for i in subset_indices]
    
    # 2. Load model with the same configuration as Kaggle submission
    print("Loading model...")
    config = {
        'residue_embed_dim': 128,
        'pair_embed_dim': 64,
        # ... other parameters matching Kaggle config
    }
    model = RNAFoldingModel(config)
    model.load_state_dict(torch.load('models/model_weights.pt'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 3. Run inference with the same pipeline as Kaggle
    print("Running inference...")
    predictions = {}
    sequences = {}
    
    with torch.no_grad():
        for idx in subset_indices:
            sample = val_dataset[idx]
            target_id = sample['target_id']
            
            # Store sequence
            sequence = ''.join([['A', 'C', 'G', 'U', 'N'][i] for i in sample['sequence_int'].tolist()])
            sequences[target_id] = sequence
            
            # Process sample
            batch = collate_fn([sample])
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            # Generate 5 predictions with different seeds
            target_predictions = []
            for seed in range(5):
                # Set random seed
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Run model
                outputs = model(batch)
                coords = outputs['pred_coords'].cpu().numpy()
                
                # Store prediction for this seed
                target_predictions.append(coords[0])  # Remove batch dimension
            
            # Combine predictions
            predictions[target_id] = np.stack(target_predictions)
    
    # 4. Generate mock submission
    print("Generating test submission...")
    submission_path = 'test_submission.csv'
    submission_df = generate_submission_csv(predictions, subset_targets, sequences, submission_path)
    
    # 5. Validate submission format
    print("Validating submission format...")
    validate_submission_format(submission_path)
    
    return True
```

### 5.2 Using Kaggle's Public Tests

Kaggle provides a scoring mechanism for public test samples. Use these to validate your submission:

1. Submit a small batch run first to verify format
2. Check public leaderboard score
3. Confirm that your model's performance matches expectations
4. Examine any submission errors before full submission

### 5.3 Verifying Model Determinism and Reproducibility

Ensure reproducible predictions with proper seed setting:

```python
def set_deterministic_mode(seed=42):
    """Set deterministic mode for reproducibility."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # These settings may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✅ Set deterministic mode with seed {seed}")
```

Verify determinism by running the same prediction twice:

```python
# Run the same inference twice with the same seed
set_deterministic_mode(42)
output1 = model(batch)

set_deterministic_mode(42)
output2 = model(batch)

# Verify outputs are identical
is_identical = torch.allclose(output1['pred_coords'], output2['pred_coords'], rtol=1e-5, atol=1e-5)
print(f"Deterministic outputs: {is_identical}")
```

### 5.4 Confirming Adherence to Competition Rules

Competition rules often include:
- No use of external data unless explicitly allowed
- No sharing of code during the competition
- No collaboration with other teams
- Specific submission count limits

Review the competition rules carefully and ensure your submission complies with all requirements.

## 6. Performance Optimization for Kaggle

### 6.1 Memory Optimization Techniques

Kaggle's 16GB GPU memory requires careful optimization:

```python
def optimize_memory_usage():
    """Apply memory optimization techniques."""
    import gc
    import torch
    
    # 1. Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 2. Run garbage collection
    gc.collect()
    
    # 3. Use mixed precision if available
    if torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
        print("✅ Mixed precision available")
    else:
        print("❌ Mixed precision NOT available")
    
    # Log memory usage
    log_memory_usage()
```

Additional memory optimization techniques:

```python
# Use gradient checkpointing (if training on Kaggle)
if hasattr(model, 'transformer_blocks'):
    for block in model.transformer_blocks:
        block.use_checkpoint = True
    print("✅ Gradient checkpointing enabled")

# Use mixed precision for inference
with torch.cuda.amp.autocast(enabled=True):
    outputs = model(batch)

# Delete intermediate tensors
del tensor_variable
torch.cuda.empty_cache()

# Process in smaller batches
batch_size = 1  # Reduced from original
```

### 6.2 Runtime Reduction Strategies

To stay within Kaggle's 9-hour limit:

```python
def optimize_runtime():
    """Apply runtime optimization techniques."""
    import torch
    
    # 1. Use eval mode for inference
    model.eval()
    
    # 2. Use torch.no_grad for inference
    with torch.no_grad():
        outputs = model(batch)
    
    # 3. Use vectorized operations
    # Instead of looping through each example
    outputs = model(batch)  # Process whole batch at once
    
    # 4. Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 5. Optimize data loading
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,  # Kaggle provides 2 CPU cores
        pin_memory=True,
        collate_fn=collate_fn
    )
```

### 6.3 Adapting Batch Sizes and Model Parameters

Balance model capability with Kaggle constraints:

```python
def adapt_model_configuration():
    """Adapt model configuration for Kaggle constraints."""
    # Reduce model dimensions
    kaggle_config = config.copy()
    kaggle_config.update({
        'residue_embed_dim': 128,  # Down from 256
        'pair_embed_dim': 64,      # Down from 128
        'num_transformer_blocks': 6,  # Down from 8
        'ipa_dim': 32              # Down from 64
    })
    
    # Adjust batch size based on available memory
    available_gpu_mb = 14000  # Leave margin for safety
    
    # Estimate memory per sample based on sequence length
    avg_seq_len = 100  # Example value, calculate from your dataset
    estimated_mb_per_sample = 2000 if avg_seq_len > 200 else 1000
    
    max_batch_size = available_gpu_mb // estimated_mb_per_sample
    batch_size = min(4, max_batch_size)  # Cap at reasonable value
    
    print(f"Using batch size: {batch_size}")
    return kaggle_config, batch_size
```

### 6.4 Efficient Use of Kaggle GPU Resources

Maximize GPU utilization while staying within limits:

```python
def monitor_resource_usage(interval=5):
    """Monitor GPU usage in background thread."""
    import threading
    import time
    
    def _monitor():
        while True:
            log_memory_usage()
            time.sleep(interval)
    
    # Start monitoring in background
    monitor_thread = threading.Thread(target=_monitor)
    monitor_thread.daemon = True
    monitor_thread.start()
    print("Resource monitoring started")

# Track time for operations
def time_operation(func, name):
    """Time an operation and log result."""
    import time
    start = time.time()
    result = func()
    duration = time.time() - start
    print(f"{name} took {duration:.2f} seconds")
    return result, duration
```

## 7. Submission Error Troubleshooting

### 7.1 Common Kaggle Submission Failures

Decision tree for diagnosing submission failures:

```
1. Does submission fail immediately?
   ├── Yes → Check for syntax errors in notebook
   │          Check for incorrect file paths
   │          Verify src/ package is correctly uploaded
   │
   └── No → Continue

2. Does submission time out?
   ├── Yes → Check if processing loop is too slow
   │          Optimize data loading and preprocessing
   │          Reduce model complexity or batch size
   │
   └── No → Continue

3. Does submission fail with OOM (Out of Memory)?
   ├── Yes → Reduce batch size
   │          Optimize memory usage (see Section 6.1)
   │          Reduce model parameters
   │
   └── No → Continue

4. Does submission complete but score poorly?
   ├── Yes → Check if test data is processed correctly
   │          Verify model weights are loaded properly
   │          Confirm prediction format matches requirements
   │
   └── No → Success! Submission is working correctly
```

### 7.2 Notebook Timeout Problems and Solutions

Strategies for addressing timeouts:

1. **Progress Tracking**: Add progress indicators:

```python
from tqdm.notebook import tqdm

for target_id in tqdm(test_target_ids, desc="Processing targets"):
    # Process each target
    pass
```

2. **Time Budgeting**: Allocate time for different processing stages:

```python
import time

# Kaggle limit: 9 hours = 32400 seconds
total_budget_seconds = 32400

# Allocate 10% for setup, 80% for inference, 10% for output generation
setup_limit = 0.1 * total_budget_seconds
inference_limit = 0.8 * total_budget_seconds
output_limit = 0.1 * total_budget_seconds

# Track time usage
start_time = time.time()
# ... setup code ...
setup_time = time.time() - start_time
print(f"Setup used {setup_time:.2f}s of {setup_limit:.2f}s budget")

# Adjust inference if setup took longer than expected
remaining_time = total_budget_seconds - setup_time
inference_limit = 0.9 * remaining_time  # 90% of remaining time
```

3. **Early Output Saving**: Save partial results during processing:

```python
# Process in batches and save intermediate results
batch_size = 10
all_predictions = {}

for batch_idx in range(0, len(test_target_ids), batch_size):
    batch_targets = test_target_ids[batch_idx:batch_idx+batch_size]
    
    # Process batch
    batch_predictions = process_batch(batch_targets)
    all_predictions.update(batch_predictions)
    
    # Save intermediate results every batch
    interim_df = generate_submission_csv(
        all_predictions, 
        list(all_predictions.keys()),
        sequences,
        '/kaggle/working/submission_interim.csv'
    )
    
    print(f"Saved interim results for {len(all_predictions)}/{len(test_target_ids)} targets")
```

### 7.3 Memory Error Diagnosis and Resolution

If you encounter memory errors:

```python
def diagnose_memory_usage(model, sample_input):
    """Diagnose memory usage for model inference."""
    import torch
    
    # Clear existing cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Check initial memory
    initial_memory = torch.cuda.memory_allocated()
    
    # Run forward pass
    try:
        with torch.no_grad():
            outputs = model(sample_input)
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = peak_memory - initial_memory
        
        print(f"Memory used for forward pass: {memory_used / 1024**2:.2f} MB")
        print(f"Peak memory usage: {peak_memory / 1024**2:.2f} MB")
        
        # Check if we're close to limit
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory - peak_memory
        
        print(f"Available GPU memory: {available_memory / 1024**2:.2f} MB")
        
        if available_memory < 1024**3:  # Less than 1GB available
            print("⚠️ WARNING: Low memory available, consider reducing batch size or model size")
        
        return True
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"❌ Out of memory error: {e}")
            print("Suggestions:")
            print("1. Reduce batch size")
            print("2. Reduce model dimensions (residue_dim, pair_dim)")
            print("3. Reduce number of transformer blocks")
            return False
        else:
            raise e
```

Memory optimization checklist:

- [ ] Reduce batch size to 1 if necessary
- [ ] Apply mixed precision (fp16) for inference
- [ ] Reduce model parameter dimensions
- [ ] Clear GPU cache between processing batches
- [ ] Process very long sequences separately
- [ ] Consider sequence chunking for extremely long RNAs

### 7.4 Inference Pipeline Debugging on Kaggle

Add structured debug output in your notebook:

```python
def debug_inference_pipeline(target_id, debug_level=1):
    """Debug inference pipeline for a specific target."""
    print(f"=== Debugging target: {target_id} ===")
    
    # 1. Check input data
    print(f"Step 1: Validating input data")
    try:
        sample = dataset.get_item_by_target_id(target_id)
        sequence_length = len(sample['sequence_int'])
        print(f"✅ Found target in dataset, sequence length: {sequence_length}")
        
        if debug_level >= 2:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    except Exception as e:
        print(f"❌ Error loading input data: {e}")
        return False
    
    # 2. Check model loading
    print(f"Step 2: Testing model loading")
    try:
        # Perform a small forward pass
        batch = collate_fn([sample])
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device)
        
        with torch.no_grad():
            outputs = model(batch)
        
        print(f"✅ Model forward pass successful")
        if debug_level >= 2:
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    except Exception as e:
        print(f"❌ Error in model forward pass: {e}")
        torch.cuda.empty_cache()  # Clear GPU memory
        return False
    
    # 3. Check prediction generation
    print(f"Step 3: Testing prediction generation")
    try:
        # Generate 5 predictions
        target_predictions = []
        for seed in range(5):
            set_deterministic_mode(seed)
            with torch.no_grad():
                seed_outputs = model(batch)
            
            coords = seed_outputs['pred_coords'].cpu().numpy()
            target_predictions.append(coords[0])
        
        predictions = {target_id: np.stack(target_predictions)}
        print(f"✅ Generated 5 predictions, shape: {predictions[target_id].shape}")
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return False
    
    # 4. Check submission formatting
    print(f"Step 4: Testing submission formatting")
    try:
        sequences = {target_id: ''.join([['A', 'C', 'G', 'U', 'N'][i] for i in sample['sequence_int'].tolist()])}
        mini_submission = generate_submission_csv(
            predictions, 
            [target_id], 
            sequences, 
            '/kaggle/working/debug_submission.csv'
        )
        print(f"✅ Generated submission with {len(mini_submission)} rows")
    except Exception as e:
        print(f"❌ Error formatting submission: {e}")
        return False
    
    print(f"=== Debug complete for {target_id} ===")
    return True
```

## 8. Final Submission Checklist

### 8.1 Pre-submission Verification Steps

✅ **Technical Verification**:

- [ ] All code cells execute without errors
- [ ] Memory usage stays within Kaggle limits
- [ ] Output format matches competition requirements
- [ ] Notebook produces expected submission.csv
- [ ] Code is resistant to transient errors (has retry logic if needed)

✅ **Scientific Verification**:

- [ ] Model configuration matches expected submission settings
- [ ] Weights are loaded from correct checkpoint
- [ ] 5-prediction ensemble strategy is implemented correctly
- [ ] Performance is consistent with local validation

✅ **Efficiency Verification**:

- [ ] Notebook runs within the 9-hour limit
- [ ] Resource usage is optimized for Kaggle
- [ ] No unnecessary computations are performed

### 8.2 Documentation Requirements

Include these documentation elements in your final notebook:

```python
# === RNA 3D Folding Model Submission ===
# Team: [Your Team Name]
# Date: [Submission Date]
# 
# Model Architecture:
# - Backbone: Transformer-based architecture with [X] transformer blocks
# - Residue dimension: [X]
# - Pair dimension: [X]
# - IPA module: [Implementation details]
# 
# Processing Strategy:
# - Direct coordinate prediction for C1' atoms
# - 5 predictions generated with different random seeds
# - Full test set processing with batch size [X]
# 
# This notebook:
# 1. Loads precomputed features for test sequences
# 2. Processes through our RNA 3D folding model
# 3. Generates 5 predictions per target
# 4. Formats results according to competition requirements
```

Include a simple summary at the end:

```python
# Submission Summary
print(f"Total targets processed: {len(test_target_ids)}")
print(f"Total residues predicted: {sum(len(sequences[target_id]) for target_id in test_target_ids)}")
print(f"Submission file size: {os.path.getsize(submission_path) / 1024**2:.2f} MB")
print(f"Total runtime: {(time.time() - overall_start_time) / 60:.2f} minutes")
```

### 8.3 Version Control and Tracking

Maintain version information in your notebook:

```python
# Submission Version Information
SUBMISSION_VERSION = "1.2.3"  # Major.Minor.Patch
MODEL_VERSION = "v1.4"        # Model checkpoint version
FEATURES_VERSION = "v2.1"     # Precomputed features version

# Include version hash for reproducibility
import hashlib
def get_version_hash():
    """Generate a hash based on model configuration and key parameters."""
    import json
    # Create a string with key configuration
    config_str = json.dumps(config, sort_keys=True)
    version_string = f"{config_str}_{MODEL_VERSION}_{FEATURES_VERSION}"
    # Generate hash
    return hashlib.md5(version_string.encode()).hexdigest()[:8]

VERSION_HASH = get_version_hash()
print(f"Submission {SUBMISSION_VERSION}, Model {MODEL_VERSION}, Features {FEATURES_VERSION}")
print(f"Version Hash: {VERSION_HASH}")
```

### 8.4 Backup and Contingency Plans

Implement safeguards against failures:

```python
# Checkpoint regularly during processing
processed_targets = set()

for target_id in test_target_ids:
    if target_id in processed_targets:
        print(f"Skipping already processed target: {target_id}")
        continue
    
    try:
        # Process target
        predictions[target_id] = process_target(target_id)
        processed_targets.add(target_id)
        
        # Save progress after every 10 targets
        if len(processed_targets) % 10 == 0:
            # Save intermediate predictions
            torch.save(predictions, '/kaggle/working/predictions_checkpoint.pt')
            # Update progress file
            with open('/kaggle/working/progress.txt', 'w') as f:
                f.write(f"Processed {len(processed_targets)}/{len(test_target_ids)} targets")
            
            print(f"Checkpoint saved: {len(processed_targets)}/{len(test_target_ids)} targets")
    except Exception as e:
        print(f"Error processing target {target_id}: {e}")
        # Continue with next target
        continue

# If processing was interrupted, try to load from checkpoint
if len(predictions) < len(test_target_ids) and os.path.exists('/kaggle/working/predictions_checkpoint.pt'):
    try:
        checkpoint_predictions = torch.load('/kaggle/working/predictions_checkpoint.pt')
        predictions.update(checkpoint_predictions)
        print(f"Loaded {len(checkpoint_predictions)} predictions from checkpoint")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
```

## Conclusion

Following this comprehensive guide will help you successfully submit your RNA 3D folding solution to the Kaggle competition. The guide addresses the unique challenges of converting our local development pipeline to the Kaggle environment, with particular attention to path handling, memory optimization, and correct output formatting.

Remember these key principles:
1. Strict path parameterization enables seamless environment transition
2. Resource optimization is critical for Kaggle's constraints
3. Careful attention to competition output format is essential
4. Test thoroughly before final submission
5. Implement robust error handling and progress tracking

By following these guidelines, you'll create a reliable, performant Kaggle submission that accurately represents your model's capabilities on the RNA 3D folding challenge.
