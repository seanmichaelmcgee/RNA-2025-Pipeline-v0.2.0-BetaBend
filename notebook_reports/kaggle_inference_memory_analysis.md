# Kaggle Inference Notebook Memory Analysis

## 1. Overview

This document analyzes the CUDA out-of-memory (OOM) error encountered during the execution of our Kaggle inference notebook for RNA 3D structure prediction. We'll examine the causes, provide a comprehensive cell-by-cell analysis, and outline potential solutions.

**Date:** 2025-04-23

## 2. OOM Error Analysis

### 2.1 Error Details

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 1.98 GiB. 
GPU 0 has a total capacity of 15.56 GiB of which 1.37 GiB is free. 
Process 6680 has 1.81 GiB memory in use. 
Including non-PyTorch memory, this process has 11.21 GiB memory in use. 
Of the allocated memory 9.46 GiB is allocated by PyTorch, and 1.51 GiB is reserved by PyTorch but unallocated.
```

### 2.2 Tensor Sizes Before OOM

```
INFO: pairing_probs shape: torch.Size([8, 720, 720])
INFO: coupling_matrix shape: torch.Size([8, 720, 720])
INFO: rel_pos_batch shape: torch.Size([8, 720, 720, 32])
INFO: pair_features_list[0] shape: torch.Size([8, 720, 720, 1])
INFO: pair_features_list[1] shape: torch.Size([8, 720, 720, 1])
INFO: pair_features_list[2] shape: torch.Size([8, 720, 720, 32])
```

### 2.3 Error Location

The error occurs during:
1. `run_inference` calling `generate_samples`
2. In `generate_samples`, when calling `model(batch_device)`
3. Inside the transformer block's `_update_pair_repr` method
4. Specifically at the dropout step after applying the MLP to update pair representations

## 3. Memory Usage Analysis

### 3.1 Key Memory Consumers

1. **Quadratic Complexity Tensors:**
   - Pairwise tensors with shape `[batch_size, seq_len, seq_len, features]` scale as O(N²) with sequence length
   - Example: `rel_pos_batch` tensor of shape `[8, 720, 720, 32]` requires:
     - 8 × 720 × 720 × 32 × 4 bytes = 530,841,600 bytes ≈ 506 MB (for float32)

2. **Batch Size Impact:**
   - Current batch size of 8 multiplies all tensor sizes
   - Reducing batch size will linearly reduce memory usage

3. **Precision Impact:**
   - Using float32 (4 bytes per element)
   - float16/half precision would reduce memory by 50%

4. **Memory Fragmentation:**
   - 1.51 GiB is "reserved but unallocated" by PyTorch
   - Suggests memory fragmentation is occurring

### 3.2 Theoretical Memory Requirements

For a batch of 8 sequences, each with length 720:

| Tensor | Shape | Memory (float32) | Memory (float16) |
|--------|-------|------------------|------------------|
| pairing_probs | [8, 720, 720] | 16.6 MB | 8.3 MB |
| coupling_matrix | [8, 720, 720] | 16.6 MB | 8.3 MB |
| rel_pos_batch | [8, 720, 720, 32] | 530.8 MB | 265.4 MB |
| pair_repr | [8, 720, 720, 64] | 1,061.7 MB | 530.8 MB |
| residue_repr | [8, 720, 128] | 3.0 MB | 1.5 MB |

Memory for intermediate calculations in transformer blocks and IPA module must also be considered, approximately doubling the above requirements.

## 4. Notebook Cell Analysis

### Cell 1: Imports and Data Loading Patch
- Imports necessary libraries
- Defines fixed data loading function
- Minimal memory impact

### Cell 2-4: Configuration
- Sets paths and parameters
- Defines model paths
- No significant memory impact

### Cell 5: Model Loading
- Defines functions for model loading and patching
- Loads models (multiple if using ensemble)
- Memory impact:
  - Model weights and parameters (~100-300MB depending on model size)
  - Enhanced positional encoding buffer (minimal)

### Cell 6-7: Test Data Loading
- Creates data loader for test sequences
- Memory impact:
  - Dataset caching of features (~100-500MB depending on test set size)
  - DataLoader workers and prefetching (~100MB per worker)

### Cell 8-10: Inference Functions and Execution
- Defines and runs inference pipeline
- **Critical memory impact:**
  - Creates large tensors for pair representations
  - Batch processing of sequences
  - Multiple samples per sequence
  - This is where OOM occurs

### Cell 11-12: Formatting and Submission
- Formats results for Kaggle submission
- Lower memory impact as it processes results sequentially

## 5. Memory Bottleneck Analysis

### 5.1 Primary Bottleneck: Pair Representation Tensors

The transformer architecture processes RNA structures using both:
1. **Residue representations**: O(N) complexity with sequence length
2. **Pair representations**: O(N²) complexity with sequence length

For long sequences (N=720), the pair representations dominate memory usage:
- Each transformer block maintains and updates large pair representation tensors
- These tensors grow quadratically with sequence length
- Multiple transformer blocks compound the issue

### 5.2 Memory Fragmentation

The PyTorch CUDA allocator reserves memory in chunks, leading to fragmentation when tensors of different sizes are allocated and freed during processing. This explains the "reserved but unallocated" memory of 1.51 GiB.

### 5.3 Batch Size Issue

Processing 8 sequences simultaneously with a sequence length of 720 creates very large tensors. While this improves throughput, it's unsustainable for the memory constraints.

## 6. Potential Solutions

### 6.1 Immediate Fixes

1. **Reduce Batch Size:**
   - Change from 8 to 1 or 2 sequences per batch
   - Implementation: Modify `BATCH_SIZE` parameter in cell 3

2. **Use Mixed Precision:**
   - Use `torch.cuda.amp.autocast` to automatically use float16 where appropriate
   - Implementation: Wrap model forward pass in autocast context

3. **Address Memory Fragmentation:**
   - Add `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` before PyTorch import
   - Implementation: Add at the start of cell 1

4. **Sequence Length Handling:**
   - Process extremely long sequences (>500) separately with batch size of 1
   - Implementation: Create separate data loader for long sequences

### 6.2 Medium-Term Improvements

1. **Efficient Attention Implementation:**
   - Implement memory-efficient attention mechanisms
   - Consider linear attention variants

2. **Model Architecture Optimization:**
   - Reduce model size (fewer blocks, smaller dimensions)
   - Optimize for inference vs. training

3. **Chunking for Long Sequences:**
   - Process very long sequences in chunks and merge results

4. **Checkpoint Deduplication:**
   - Ensure saved checkpoints don't contain duplicate weights

## 7. Implementation Plan

### 7.1 Immediate Actions

1. Modify cell 3 to reduce batch size to 1 for sequences > 500, 2 for others
2. Add memory tracking metrics to cell 8/9
3. Implement mixed precision inference in generate_samples function
4. Add PYTORCH_CUDA_ALLOC_CONF configuration to cell 1

### 7.2 Code Modifications

```python
# Cell 1: Add at top before imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Cell 3: Modify batch size logic
BATCH_SIZE = 1  # Reduced from 8 for memory efficiency

# Cell 9: Add in generate_samples function
def generate_samples(model, batch, num_samples=5, temperature=0.1):
    results = []
    batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
    
    for i in range(num_samples):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):  # Enable mixed precision
            model.train()
            outputs = model(batch_device)
            model.eval()
            
            # Rest of the function is unchanged
            # ...
```

## 8. Memory Optimization Checklist

- [ ] Reduce batch size to 1
- [ ] Enable mixed precision inference  
- [ ] Configure PyTorch CUDA allocator for expandable segments
- [ ] Add memory monitoring during inference
- [ ] Separate processing for long sequences
- [ ] Clear CUDA cache between batches

## 9. Timeline

| Date | Event | Notes |
|------|-------|-------|
| 2025-04-23 | Initial memory analysis | Identified OOM error with batch size 8 and sequence length 720 |
| 2025-04-23 | Applied initial fixes | Reduced batch size, enabled mixed precision |
| | | |
| | | |

## 10. Additional Insights

### 10.1 Memory Management in PyTorch

The error message indicates significant reserved but unallocated memory (1.51 GiB), pointing to memory fragmentation as a contributor to the OOM issue. This occurs when the allocator can't find contiguous memory blocks large enough for new tensors, even though the total free memory might be sufficient.

The suggestion to set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` would help by allowing PyTorch to request more memory from CUDA when fragmentation occurs, instead of trying to fit within existing blocks.

### 10.2 Quadratic Complexity Challenge

The core challenge is the O(N²) memory complexity in transformer architectures when dealing with pairwise interactions for RNA structure prediction. With sequence lengths of 720, this creates tensors with over 518,000 elements per batch item per feature.

### 10.3 Kaggle Environment Constraints

Kaggle typically provides GPUs with 16GB VRAM, closely matching our development environment. However, Kaggle notebooks have a 9-hour runtime limit, so optimization must balance:
1. Memory efficiency (to avoid OOM)
2. Computational efficiency (to complete within time limits)

This makes proper memory management crucial for successful deployment.