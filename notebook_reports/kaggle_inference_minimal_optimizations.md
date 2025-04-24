# Kaggle Inference: Minimal Memory Optimizations

## 1. Analysis of Optimization Complexity

For each potential optimization, I've analyzed the implementation complexity and expected impact:

| Optimization | Code Complexity | Memory Impact | Implementation Time |
|--------------|-----------------|---------------|---------------------|
| Mixed Precision (FP16) | Very Low | High (~50% reduction) | <15 minutes |
| Intelligent Batching | Low | High (70-80% for long seqs) | ~30 minutes |
| Memory Fragmentation Config | Minimal | Medium | <5 minutes |
| Sequence Grouping | High | Medium-High | 1-2 hours |
| Advanced Techniques | Very High | High | Multiple days |

Based on this analysis, we should focus on the first three optimizations which offer the best balance of impact vs. implementation effort.

## 2. Mixed Precision Implementation

### 2.1 How Mixed Precision Works

Mixed precision uses float16 (16-bit) instead of float32 (32-bit) for many operations, which:
- Reduces memory usage by ~50%
- Often increases computational speed on modern GPUs
- Maintains numerical accuracy for most operations

PyTorch's `torch.cuda.amp.autocast()` context manager makes this extremely simple to implement - it automatically converts operations to FP16 where appropriate while keeping critical operations in FP32 for stability.

### 2.2 Implementation Approach

The implementation is very code-light, requiring just a few lines of changes:

```python
# In generate_samples function
def generate_samples(model, batch, num_samples=5, temperature=0.1):
    results = []
    batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
    
    for i in range(num_samples):
        with torch.no_grad(), torch.cuda.amp.autocast():  # <-- Add this context manager
            model.train()  # Enable dropout for diverse sampling
            outputs = model(batch_device)
            model.eval()  # Restore eval mode
            
            # Rest of function remains unchanged
            # ...
```

This simple change enables mixed precision throughout the forward pass of the model without requiring any architectural modifications.

## 3. Intelligent Batching Implementation

### 3.1 Concept

Instead of using a fixed batch size for all sequences, we'll adapt the batch size based on sequence length:

- Short sequences (length < 250): Batch size 8
- Medium sequences (250-500): Batch size 4
- Long sequences (>500): Batch size 1-2

### 3.2 Simple Implementation

This can be implemented with minimal code changes in cell 7 where we create the data loader:

```python
# In cell 7, replace the current loader creation with:
def create_adaptive_test_loader(sequences_path, features_dir, temporal_cutoff):
    """Create test data loader with adaptive batch size based on sequence length."""
    # Create dataset as before
    dataset = RNADataset(
        sequences_csv_path=sequences_path,
        features_dir=features_dir,
        temporal_cutoff=temporal_cutoff,
        use_validation_set=True,
        require_features=False,
    )
    
    # Get sequence lengths
    seq_lengths = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        seq_lengths.append(len(sample['sequence']))
    
    # Determine adaptive batch size
    max_length = max(seq_lengths)
    if max_length < 250:
        batch_size = 8
    elif max_length < 500:
        batch_size = 4
    else:
        batch_size = 2  # Even more conservative: use 1 for sequences >600
    
    print(f"Using adaptive batch size {batch_size} for max sequence length {max_length}")
    
    # Create data loader with adapted batch size
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    return loader, batch_size

# Replace test_loader creation
test_loader, BATCH_SIZE = create_adaptive_test_loader(
    TEST_SEQUENCES_PATH, FEATURES_DIR, TEMPORAL_CUTOFF
)
```

This approach requires minimal changes to the existing code structure while providing significant memory benefits for longer sequences.

## 4. Memory Fragmentation Configuration

### 4.1 Configuration Change

Add this single line at the top of the notebook (cell 1) before other imports:

```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

This tells PyTorch's CUDA allocator to use expandable memory segments, which helps reduce fragmentation by allowing the allocator to request more memory from CUDA when needed.

## 5. Combined Implementation Approach

The three optimizations above can be implemented in less than 30 minutes total and should dramatically reduce memory usage, particularly for longer sequences.

### Implementation Order:

1. Add memory configuration (1 minute)
2. Implement mixed precision (5 minutes)
3. Implement intelligent batching (20 minutes)

The combined impact should be sufficient to handle the OOM issues without requiring the more complex sequence grouping or advanced techniques.

## 6. Measuring Success

After implementing these optimizations, we'll measure:

1. Memory usage during inference
2. Whether the OOM error is resolved
3. Impact on inference speed
4. Impact on prediction quality

## 7. Next Steps

If these minimal optimizations don't fully resolve the OOM issues, we can then consider more complex approaches like:

1. Sequence grouping by length ranges
2. Special handling for very long sequences
3. More aggressive tensor optimizations

But given the significant memory reduction from just these simple changes (~75% reduction for long sequences), it's highly likely these will be sufficient.

## 8. Timeline

| Date | Event | Status |
|------|-------|--------|
| 2025-04-23 | Identify and analyze OOM issues | Completed |
| 2025-04-23 | Implement minimal memory optimizations | Planned |
| 2025-04-24 | Test optimization effectiveness | Planned |