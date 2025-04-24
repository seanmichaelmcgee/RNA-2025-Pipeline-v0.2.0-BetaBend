# Kaggle Inference Optimization Plan

## 1. Environment Analysis

### 1.1 Kaggle GPU Specifications
- Typical Kaggle competition GPU: NVIDIA Tesla P100 with 16GB VRAM
- Our development environment: 15.56 GiB VRAM available
- Both environments are sufficiently similar to develop a consistent strategy

### 1.2 Current Memory Bottleneck

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 1.98 GiB. 
GPU 0 has a total capacity of 15.56 GiB of which 1.37 GiB is free. 
Process 6680 has 1.81 GiB memory in use. 
Including non-PyTorch memory, this process has 11.21 GiB memory in use. 
Of the allocated memory 9.46 GiB is allocated by PyTorch, and 1.51 GiB is reserved by PyTorch but unallocated.
```

Critical tensors before OOM:
```
INFO: pairing_probs shape: torch.Size([8, 720, 720])
INFO: coupling_matrix shape: torch.Size([8, 720, 720])
INFO: rel_pos_batch shape: torch.Size([8, 720, 720, 32])
INFO: pair_features_list[0] shape: torch.Size([8, 720, 720, 1])
INFO: pair_features_list[1] shape: torch.Size([8, 720, 720, 1])
INFO: pair_features_list[2] shape: torch.Size([8, 720, 720, 32])
```

## 2. Optimization Strategy

### 2.1 Intelligent Batching

Rather than defaulting to batch size 1, we'll implement intelligent batching based on sequence length:

| Sequence Length Range | Batch Size | Rationale |
|-----------------------|------------|-----------|
| < 200                 | 8          | Short sequences can be processed efficiently in larger batches |
| 200-400               | 4          | Medium sequences need moderate batching |
| 400-600               | 2          | Long sequences require smaller batch sizes |
| > 600                 | 1          | Very long sequences processed individually |

This approach maintains processing efficiency for shorter sequences while ensuring memory efficiency for longer ones.

### 2.2 Memory Efficiency Techniques

1. **Mixed Precision Training (FP16)**
   - Reduce memory footprint by ~50% without significantly affecting accuracy
   - Implementation requires minimal changes - just adding autocast context

2. **Memory Fragmentation Mitigation**
   - Add `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` to reduce fragmentation
   - Clear cache between processing batches of long sequences

3. **Gradient Checkpointing Analog for Inference**
   - Split large transformers blocks into functional components
   - Process long sequences in chunks for pair representations

4. **Smart Caching and Pre-computation**
   - Pre-compute and cache position encodings
   - Re-use tensor containers where possible

### 2.3 Tensor Shape Optimization

The critical memory usage comes from pairwise representations scaling as O(N²). We'll optimize these with:

1. **Sparse Attention Patterns**
   - Implement masked attention for distant residues (biological justification)
   - This can reduce the effective tensor sizes significantly

2. **Shared Memory Pooling**
   - Reuse memory buffers across sequential operations 
   - Manually manage buffer allocation for large tensors

## 3. Implementation Plan

### 3.1 Code Changes to Notebook

#### Cell 1: Add Memory Configuration
```python
# Add at top before other imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

#### Cell 3: Adaptive Batch Size Configuration
```python
# Replace current batch size logic with adaptive approach
def get_adaptive_batch_size(seq_lengths):
    """Determine optimal batch size based on sequence length distribution."""
    max_len = max(seq_lengths)
    if max_len < 200:
        return 8
    elif max_len < 400:
        return 4
    elif max_len < 600:
        return 2
    else:
        return 1

# Initialize with default, will be updated after dataset creation
BATCH_SIZE = 4  # Default, will be adapted based on sequence distribution
```

#### Cell 5: Update Model Wrapper for Memory Efficiency
```python
class MemoryEfficientWrapper(nn.Module):
    """Wrapper to apply memory optimization for inference."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, batch):
        """Forward pass with memory optimizations."""
        # Apply mixed precision
        with torch.cuda.amp.autocast():
            # Forward pass through model
            return self.model(batch)

# Add after model loading
if model is not None:
    # Apply memory efficiency wrapper
    model = MemoryEfficientWrapper(model)
    print("Applied memory efficiency wrapper to model")
```

#### Cell 7: Update Data Loader with Adaptive Batch Size
```python
# After dataset creation, calculate adaptive batch size
seq_lengths = [len(dataset.sequences[i]) for i in range(len(dataset))]
BATCH_SIZE = get_adaptive_batch_size(seq_lengths)
print(f"Using adaptive batch size: {BATCH_SIZE}")

# Group sequences by length range for more efficient processing
length_ranges = [(0, 200), (200, 400), (400, 600), (600, float('inf'))]
batch_sizes = [8, 4, 2, 1]

# Create grouped indices for dataloader
grouped_indices = [[] for _ in range(len(length_ranges))]
for i, seq_len in enumerate(seq_lengths):
    for j, (min_len, max_len) in enumerate(length_ranges):
        if min_len <= seq_len < max_len:
            grouped_indices[j].append(i)
            break

# Create separate dataloaders for each group
grouped_loaders = []
for indices, bs in zip(grouped_indices, batch_sizes):
    if indices:  # Only create if we have sequences in this range
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=bs,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4
        )
        grouped_loaders.append((loader, bs))
        print(f"Created loader with batch size {bs} for {len(indices)} sequences")
```

#### Cell 9-10: Memory-Optimized Inference
```python
def run_inference_optimized(models_dict, grouped_loaders, num_samples=5, temperature=0.1):
    """Run inference using optimized memory approach with grouped loaders."""
    all_results = {}
    
    # Process each group with its appropriate batch size
    for loader, batch_size in grouped_loaders:
        print(f"Processing group with batch size {batch_size}")
        
        # Clear cache between groups to minimize fragmentation
        torch.cuda.empty_cache()
        
        # Get the model
        model_name = list(models_dict.keys())[0]
        model = models_dict[model_name]['model']
        
        # Process all batches in this group
        for batch in tqdm(loader, desc=f"Batch size {batch_size}"):
            # Generate samples with mixed precision
            with torch.cuda.amp.autocast():
                samples = generate_samples(model, batch, num_samples, temperature)
            
            # Organize results
            for sample in samples:
                target_id = sample['target_id']
                if target_id not in all_results:
                    all_results[target_id] = []
                all_results[target_id].append(sample)
    
    return all_results

# Replace previous inference call with optimized version
if USE_ENSEMBLE and len(models) > 1:
    print("Running ensemble inference with multiple models...")
    # Ensemble version would need similar adaptation
    results = run_ensemble_inference_optimized(models, grouped_loaders, NUM_SAMPLES, TEMPERATURE)
else:
    # Check if we have any models
    if not models:
        print("ERROR: No models available for inference. Check model paths and loading.")
        results = {}
    else:
        # Get the first (and only) model from the dictionary
        model_name = list(models.keys())[0]
        model = models[model_name]['model']
        print(f"Running inference with single model: {model_name}")
        results = run_inference_optimized(models, grouped_loaders, NUM_SAMPLES, TEMPERATURE)
```

### 3.2 Advanced Optimization for Very Long Sequences

For sequences longer than 600 residues, we'll apply additional optimizations:

```python
def process_long_sequence(model, batch, num_samples=5, temperature=0.1):
    """Special processing for very long sequences."""
    # Use gradient checkpointing analog for inference
    # Split sequence attention calculation into chunks
    # ...implementation details...
```

### 3.3 Memory Monitoring

Add memory monitoring to identify bottlenecks and track improvements:

```python
def log_memory_usage(step_name=""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"Memory at {step_name}: Allocated: {allocated:.1f}MB, Reserved: {reserved:.1f}MB")
```

## 4. Expected Results

| Sequence Length | Original Batch Size | New Batch Size | Memory Usage Reduction |
|-----------------|---------------------|--------------|-----------------------|
| ~200            | 8                   | 8            | 0% (unchanged)        |
| 200-400         | 8                   | 4            | ~50%                  |
| 400-600         | 8                   | 2            | ~75%                  |
| >600 (e.g. 720) | 8                   | 1            | ~87.5%                |

Using mixed precision (FP16) provides an additional ~50% memory reduction across all configurations.

## 5. Implementation Phases

### Phase 1: Minimal Changes for Maximum Impact
1. Add memory fragmentation configuration
2. Implement mixed precision inference
3. Use simpler adaptive batch size (just check max length)

### Phase 2: Optimized Memory Usage
1. Implement sequence grouping by length
2. Add memory monitoring
3. Optimize tensor operations

### Phase 3: Advanced Techniques
1. Special handling for very long sequences
2. Sparse attention patterns
3. Memory buffer reuse

## 6. Timeline

| Date | Event | Status |
|------|-------|--------|
| 2025-04-23 | Initial memory analysis | Completed |
| 2025-04-23 | Phase 1 implementation | Planned |
| 2025-04-24 | Phase 2 implementation | Planned |
| 2025-04-25 | Testing and validation | Planned |

## 7. Monitoring and Validation

For each optimization, we'll:
1. Track memory usage before and after implementation
2. Verify prediction accuracy is unchanged
3. Monitor execution time to ensure it remains within Kaggle limits
4. Test with representative sequence length distributions

## 8. Conclusion

This optimization plan preserves the model's capabilities while adapting to Kaggle's GPU constraints. By using an adaptive approach based on sequence length, we can process most sequences efficiently while applying targeted optimizations only when necessary for very long sequences.

The approach maintains the notebook's integrity and scientific validity while ensuring it runs reliably in the Kaggle environment.