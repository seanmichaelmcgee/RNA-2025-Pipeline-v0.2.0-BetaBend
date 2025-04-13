# RNA 3D Folding Pipeline: Testing Workflow

## 1. Overview & Prerequisites

### 1.1 Introduction

This document outlines the comprehensive testing workflow for the RNA 3D Folding pipeline. Testing is a critical component of our development process, ensuring that predictions are scientifically valid, the model is computationally efficient, and the pipeline meets all requirements for both local execution and Kaggle compatibility.

The testing workflow is designed to validate:
- **Correctness**: Proper implementation of all algorithms and components
- **Integration**: Seamless interaction between pipeline components
- **Performance**: Appropriate resource utilization and prediction accuracy
- **Compatibility**: Adherence to Kaggle submission requirements
- **Robustness**: Consistent behavior across varying inputs

### 1.2 Testing Environment Setup

Before beginning the testing procedure, ensure your environment meets these requirements:

```bash
# 1. Activate the conda environment
conda activate rna-3d-folding

# 2. Verify PyTorch installation with GPU support
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU device count: {torch.cuda.device_count()}')"

# 3. Verify all required packages are installed
pip list | grep -E "torch|numpy|pandas|scipy|matplotlib|pyyaml"

# 4. Check GPU memory
nvidia-smi
```

### 1.3 Required Files and Components

Testing requires these components to be properly implemented and available:

| Component | Path | Purpose |
|-----------|------|---------|
| Data Loading | `src/data_loading.py` | Load and preprocess RNA sequence and feature data |
| Model Architecture | `src/models/*.py` | Model components including transformer blocks and IPA module |
| Loss Functions | `src/losses.py` | Loss calculation for training and evaluation |
| Configuration | `config/default_config.yaml` | Parameter configuration |
| Test Data | `data/test/` | Small dataset for testing (created in preprocessing) |

### 1.4 Directory Structure for Testing

Create a specific testing directory structure to maintain test data separate from production data:

```
project_root/
├── data/
│   ├── test/                      # Test dataset (subset of training data)
│   │   ├── sequences_test.csv     # Test sequence data
│   │   ├── labels_test.csv        # Test coordinate labels 
│   │   └── processed/             # Processed feature files for test data
│   ├── processed/                 # Full processed data (for reference)
│   └── raw/                       # Raw data sources
├── tests/                         # Unit tests
├── scripts/
│   └── test_pipeline.py           # End-to-end testing script
└── logs/
    └── test_results/              # Test outputs and metrics
        ├── unit_tests/            # Unit test reports
        ├── integration_tests/     # Integration test outputs
        └── pipeline_tests/        # Full pipeline test results
```

## 2. Testing Methodology & Hierarchy

### 2.1 Testing Hierarchy

Our testing approach follows a hierarchical structure with increasing levels of integration:

1. **Unit Tests**: Validate individual functions and classes in isolation
2. **Integration Tests**: Verify interactions between related components
3. **End-to-End Tests**: Assess the complete pipeline functionality
4. **Performance Tests**: Evaluate efficiency, memory usage, and scaling
5. **Regression Tests**: Detect any degradation from previous results

### 2.2 Test Data Preparation

Proper test data is critical for meaningful validation. Create a test dataset that:

1. Is small enough for quick iteration (10-20 RNA sequences)
2. Covers a variety of sequence lengths (short, medium, long)
3. Includes diverse structural motifs (hairpins, junctions, etc.)
4. Has ground truth structures verified against PDB data

```python
# Sample script to create test dataset from training data
import pandas as pd
import os
import shutil
import numpy as np

def create_test_dataset(
    sequences_csv_path: str,
    labels_csv_path: str, 
    features_dir: str,
    test_data_dir: str,
    n_samples: int = 15, 
    seed: int = 42
):
    """
    Create a test dataset by sampling from training data.
    
    Args:
        sequences_csv_path: Path to sequences CSV file
        labels_csv_path: Path to labels CSV file
        features_dir: Path to directory with feature files
        test_data_dir: Output directory for test data
        n_samples: Number of RNA sequences to include in test set
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create test directory structure
    os.makedirs(os.path.join(test_data_dir, "processed/dihedral_features"), exist_ok=True)
    os.makedirs(os.path.join(test_data_dir, "processed/thermo_features"), exist_ok=True)
    os.makedirs(os.path.join(test_data_dir, "processed/mi_features"), exist_ok=True)
    
    # Load sequences and select a sample
    sequences_df = pd.read_csv(sequences_csv_path)
    labels_df = pd.read_csv(labels_csv_path)
    
    # Filter for sequences that have labels
    target_ids = labels_df['ID'].str.split('_').str[0].unique()
    valid_sequences = sequences_df[sequences_df['target_id'].isin(target_ids)]
    
    # Sample sequences of varying lengths
    sequences_by_length = {}
    for _, row in valid_sequences.iterrows():
        length = len(row['sequence'])
        length_category = "short" if length < 30 else "medium" if length < 100 else "long"
        if length_category not in sequences_by_length:
            sequences_by_length[length_category] = []
        sequences_by_length[length_category].append(row['target_id'])
    
    # Stratified sampling by length
    selected_targets = []
    for category, targets in sequences_by_length.items():
        category_count = max(1, int(n_samples * len(targets) / len(valid_sequences)))
        sampled = np.random.choice(targets, min(category_count, len(targets)), replace=False)
        selected_targets.extend(sampled)
    
    # Ensure we have enough samples
    if len(selected_targets) < n_samples:
        remaining = list(set(valid_sequences['target_id']) - set(selected_targets))
        additional = np.random.choice(remaining, min(n_samples - len(selected_targets), len(remaining)), replace=False)
        selected_targets.extend(additional)
    
    # Create test sequences CSV
    test_sequences = sequences_df[sequences_df['target_id'].isin(selected_targets)]
    test_sequences.to_csv(os.path.join(test_data_dir, "sequences_test.csv"), index=False)
    
    # Create test labels CSV
    test_labels = labels_df[labels_df['ID'].str.startswith(tuple([f"{t}_" for t in selected_targets]))]
    test_labels.to_csv(os.path.join(test_data_dir, "labels_test.csv"), index=False)
    
    # Copy feature files
    for target_id in selected_targets:
        # Copy dihedral features
        src_path = os.path.join(features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(test_data_dir, "processed/dihedral_features"))
        
        # Copy thermo features
        src_path = os.path.join(features_dir, "thermo_features", f"{target_id}_thermo_features.npz")
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(test_data_dir, "processed/thermo_features"))
        
        # Copy MI features
        src_path = os.path.join(features_dir, "mi_features", f"{target_id}_features.npz")
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(test_data_dir, "processed/mi_features"))
    
    print(f"Created test dataset with {len(selected_targets)} RNA sequences")
    print(f"Test data saved to {test_data_dir}")
```

### 2.3 Test Coverage Strategy

Our testing coverage aims to validate these key aspects:

| Testing Target | Coverage Goal | Validation Method |
|----------------|---------------|-------------------|
| Data Loading | 100% | Verify all feature types and edge cases (missing features, variable lengths) |
| Model Components | 100% | Test each model component with different input shapes and configurations |
| Loss Functions | 100% | Validate numerical stability, masking functionality, and gradient flow |
| End-to-End | 80%+ | Test pipeline with varying batch sizes, sequence lengths, and hyperparameters |
| Performance | N/A | Profile memory usage, inference time, and scaling with sequence length |

## 3. Step-by-Step Testing Procedures

### 3.1 Unit Testing

Unit tests validate individual components in isolation. Run the full test suite with:

```bash
# Run all unit tests
cd /path/to/project_root
python -m pytest tests/

# Run tests for specific components
python -m pytest tests/test_data_loading.py
python -m pytest tests/test_transformer_block.py
python -m pytest tests/test_ipa_module.py
python -m pytest tests/test_losses.py
```

#### 3.1.1 Data Loading Tests

Validate the data loading component's ability to:
- Load and process different feature types
- Handle variable sequence lengths
- Create proper masks for attention
- Batch and collate sequences properly

```python
# Example test for data loading with variable sequence lengths
def test_collate_fn_variable_length():
    """Test collate_fn with variable-length sequences."""
    # Create samples of different lengths
    samples = [
        {
            'target_id': 'seq1',
            'sequence_int': torch.tensor([0, 1, 2, 3, 4]),
            'dihedral_features': torch.rand(5, 4),
            'pairing_probs': torch.rand(5, 5),
            'positional_entropy': torch.rand(5),
            'coupling_matrix': torch.rand(5, 5),
            'coordinates': torch.rand(5, 3),
            'length': 5
        },
        {
            'target_id': 'seq2',
            'sequence_int': torch.tensor([4, 3, 2]),
            'dihedral_features': torch.rand(3, 4),
            'pairing_probs': torch.rand(3, 3),
            'positional_entropy': torch.rand(3),
            'coupling_matrix': torch.rand(3, 3),
            'coordinates': torch.rand(3, 3),
            'length': 3
        }
    ]
    
    # Collate samples
    batch = collate_fn(samples)
    
    # Verify padding and masking
    assert batch['sequence_int'].shape == (2, 5)  # Padded to max_len=5
    assert batch['mask'].shape == (2, 5)
    assert torch.all(batch['mask'][0])  # All positions valid for seq1
    assert torch.all(batch['mask'][1, :3])  # First 3 positions valid for seq2
    assert torch.all(~batch['mask'][1, 3:])  # Last 2 positions are padding
```

#### 3.1.2 Model Component Tests

Test each model component independently:

- **Embeddings**: Validate shape transformations and positional encoding
- **Transformer Block**: Test attention mechanisms and pair updates
- **IPA Module**: Verify coordinate prediction functionality
- **Full Model**: Test the complete model forward pass

```python
# Example test for transformer block
def test_transformer_block_shapes():
    """Test transformer block output shapes."""
    # Create configuration
    config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'dropout': 0.1
    }
    
    # Create transformer block
    transformer = TransformerBlock(config)
    
    # Create dummy inputs
    batch_size, seq_len = 2, 10
    residue_repr = torch.rand(batch_size, seq_len, config['residue_embed_dim'])
    pair_repr = torch.rand(batch_size, seq_len, seq_len, config['pair_embed_dim'])
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -2:] = False  # Mask last two positions in first sequence
    
    # Forward pass
    updated_residue, updated_pair = transformer(residue_repr, pair_repr, mask)
    
    # Check output shapes
    assert updated_residue.shape == residue_repr.shape
    assert updated_pair.shape == pair_repr.shape
    
    # Verify masking is maintained
    masked_sum = updated_residue[0, -2:].abs().sum()
    assert masked_sum == 0, f"Masked positions should be zero, got sum: {masked_sum}"
```

#### 3.1.3 Loss Function Tests

Validate that loss functions:
- Handle proper masking
- Produce expected gradients
- Are numerically stable
- Scale appropriately with batch size

```python
# Example test for FAPE loss with masking
def test_fape_loss_masking():
    """Test that FAPE loss properly respects masks."""
    # Create predictions and targets
    batch_size, seq_len = 2, 10
    pred_coords = torch.rand(batch_size, seq_len, 3, requires_grad=True)
    true_coords = torch.rand(batch_size, seq_len, 3)
    
    # Create mask (mask out last 3 positions in first sequence)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, -3:] = False
    
    # Compute loss with mask
    loss_masked = compute_fape_loss(pred_coords, true_coords, mask)
    
    # Set masked positions in coords to zero
    pred_manual = pred_coords.clone()
    true_manual = true_coords.clone()
    pred_manual[0, -3:] = 0
    true_manual[0, -3:] = 0
    
    # Compute loss without mask but with zeroed coordinates
    mask_all = torch.ones(batch_size, seq_len, dtype=torch.bool)
    loss_manual = compute_fape_loss(pred_manual, true_manual, mask_all)
    
    # Losses should be very close
    assert torch.isclose(loss_masked, loss_manual, rtol=1e-4)
    
    # Gradients should only flow through unmasked positions
    loss_masked.backward()
    grad_sum = pred_coords.grad[0, -3:].abs().sum()
    assert grad_sum == 0, f"Gradients should be zero for masked positions, got: {grad_sum}"
```

### 3.2 Integration Testing

Integration tests validate the interaction between related components:

#### 3.2.1 Data Loading to Model Integration

```python
def test_dataloader_to_model():
    """Test integration of DataLoader with model."""
    # Load configuration
    with open("config/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create dataset and dataloader
    dataset = RNADataset(
        sequences_csv_path=config['data']['test_sequences_path'],
        labels_csv_path=config['data']['test_labels_path'],
        features_dir=config['data']['test_features_dir']
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = RNAFoldingModel(config['model'])
    
    # Get a batch from dataloader
    batch = next(iter(dataloader))
    
    # Try to process batch with model
    try:
        outputs = model(batch)
        batch_success = True
    except Exception as e:
        batch_success = False
        print(f"Error processing batch: {e}")
    
    # Verify successful processing
    assert batch_success, "Failed to process batch through model"
    
    # Check output shapes
    batch_size = len(batch['target_ids'])
    seq_len = batch['sequence_int'].shape[1]
    
    assert outputs['pred_coords'].shape == (batch_size, seq_len, 3)
    assert outputs['pred_confidence'].shape == (batch_size, seq_len)
    assert outputs['pred_angles'].shape == (batch_size, seq_len, 4)
```

#### 3.2.2 Model to Loss Function Integration

```python
def test_model_to_loss():
    """Test integration of model outputs with loss functions."""
    # Load configuration
    with open("config/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = RNAFoldingModel(config['model'])
    
    # Create dummy batch
    batch_size, seq_len = 2, 10
    batch = {
        'sequence_int': torch.randint(0, 4, (batch_size, seq_len)),
        'dihedral_features': torch.rand(batch_size, seq_len, 4),
        'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
        'positional_entropy': torch.rand(batch_size, seq_len),
        'accessibility': torch.rand(batch_size, seq_len),
        'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
        'coordinates': torch.rand(batch_size, seq_len, 3),
        'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
    }
    
    # Mask some positions
    batch['mask'][0, -2:] = False
    
    # Forward pass
    outputs = model(batch)
    
    # Compute losses
    fape_loss = compute_fape_loss(
        outputs['pred_coords'],
        batch['coordinates'],
        batch['mask']
    )
    
    confidence_loss = compute_confidence_loss(
        outputs['pred_confidence'],
        outputs['pred_coords'],
        batch['coordinates'],
        batch['mask']
    )
    
    angle_loss = compute_angle_loss(
        outputs['pred_angles'],
        batch['dihedral_features'],
        batch['mask']
    )
    
    # Combine losses
    loss_weights = config['training']['loss_weights']
    total_loss = (
        loss_weights['fape'] * fape_loss +
        loss_weights['confidence'] * confidence_loss +
        loss_weights['angle'] * angle_loss
    )
    
    # Verify loss values
    assert not torch.isnan(total_loss), "Loss contains NaN values"
    assert not torch.isinf(total_loss), "Loss contains infinite values"
    assert total_loss > 0, "Loss should be positive"
    
    # Verify gradients flow through the model
    total_loss.backward()
    
    # Check that some gradients exist
    has_grad = any(p.grad is not None and torch.any(p.grad != 0) 
                  for p in model.parameters())
    assert has_grad, "No gradients are flowing through the model"
```

### 3.3 End-to-End Pipeline Testing

Test the complete pipeline from data loading through prediction:

```python
def test_end_to_end_pipeline(
    config_path: str,
    sequences_csv_path: str,
    labels_csv_path: str,
    features_dir: str,
    device: str = None
):
    """
    Test end-to-end pipeline with a small dataset.
    
    Args:
        config_path: Path to configuration YAML
        sequences_csv_path: Path to sequences CSV
        labels_csv_path: Path to labels CSV
        features_dir: Path to directory with feature files
        device: Device to run on ('cuda' or 'cpu')
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Create dataset and dataloader
    dataset = RNADataset(
        sequences_csv_path=sequences_csv_path,
        labels_csv_path=labels_csv_path,
        features_dir=features_dir
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Initialize model
    model = RNAFoldingModel(config['model']).to(device)
    
    # Save initial memory usage
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated()
    
    # Process all batches
    results = []
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(batch)
        
        # Calculate losses
        fape_loss = compute_fape_loss(
            outputs['pred_coords'],
            batch['coordinates'],
            batch['mask']
        )
        
        confidence_loss = compute_confidence_loss(
            outputs['pred_confidence'],
            outputs['pred_coords'],
            batch['coordinates'],
            batch['mask']
        )
        
        angle_loss = compute_angle_loss(
            outputs['pred_angles'],
            batch['dihedral_features'],
            batch['mask']
        )
        
        # Combine metrics
        batch_metrics = {
            'batch_idx': batch_idx,
            'target_ids': batch['target_ids'],
            'fape_loss': fape_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'angle_loss': angle_loss.item(),
            'sequence_lengths': batch['lengths'].tolist(),
        }
        results.append(batch_metrics)
        
        print(f"Batch {batch_idx}: FAPE={fape_loss.item():.4f}, "
              f"Conf={confidence_loss.item():.4f}, "
              f"Angle={angle_loss.item():.4f}")
    
    # Check memory usage
    if device == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"Peak memory usage: {(peak_mem - initial_mem) / 1024**2:.2f} MB")
    
    # Aggregate results
    avg_fape = sum(r['fape_loss'] for r in results) / len(results)
    avg_conf = sum(r['confidence_loss'] for r in results) / len(results)
    avg_angle = sum(r['angle_loss'] for r in results) / len(results)
    
    print(f"Average FAPE loss: {avg_fape:.4f}")
    print(f"Average Confidence loss: {avg_conf:.4f}")
    print(f"Average Angle loss: {avg_angle:.4f}")
    
    # Return final metrics
    return {
        'avg_fape_loss': avg_fape,
        'avg_confidence_loss': avg_conf,
        'avg_angle_loss': avg_angle,
        'batch_results': results
    }
```

Run the end-to-end test with:

```bash
# Create a script that uses the test function
python scripts/test_pipeline.py \
    --config config/test_config.yaml \
    --sequences data/test/sequences_test.csv \
    --labels data/test/labels_test.csv \
    --features data/test/processed \
    --output logs/test_results/pipeline_tests/
```

### 3.4 Validation Checkpoints

At each stage of testing, validate these checkpoints:

| Checkpoint | Validation Method | Acceptance Criteria |
|------------|------------------|---------------------|
| Data Loading | Tensor shapes, value ranges | No errors, proper padding and masking |
| Model Forward Pass | Output shapes, value ranges | Output tensors have expected shapes |
| Loss Calculation | Loss values, gradients | No NaNs, positive loss values, gradients flow |
| Memory Usage | GPU memory tracking | Peak memory within GPU limits |
| Error Handling | Test with edge cases | Graceful handling of edge cases |

## 4. Performance Metrics & Evaluation

### 4.1 Scientific Metrics

#### 4.1.1 Coordinate Accuracy Metrics

For RNA 3D structure prediction, use these metrics:

1. **RMSD (Root-Mean-Square Deviation)**: Measures average Euclidean distance between predicted and true coordinates after optimal superposition

```python
def calculate_rmsd(pred_coords, true_coords, mask=None):
    """
    Calculate RMSD between predicted and true coordinates.
    
    Args:
        pred_coords: Predicted coordinates (batch_size, seq_len, 3)
        true_coords: True coordinates (batch_size, seq_len, 3)
        mask: Boolean mask (batch_size, seq_len)
        
    Returns:
        RMSD values per sequence
    """
    batch_size = pred_coords.shape[0]
    rmsd_values = []
    
    for i in range(batch_size):
        # Apply mask if provided
        if mask is not None:
            valid_mask = mask[i]
            p_valid = pred_coords[i, valid_mask]
            t_valid = true_coords[i, valid_mask]
        else:
            p_valid = pred_coords[i]
            t_valid = true_coords[i]
        
        # Skip if no valid positions
        if len(p_valid) == 0:
            rmsd_values.append(float('nan'))
            continue
        
        # Perform Kabsch alignment
        p_aligned = kabsch_align(p_valid, t_valid)
        
        # Calculate RMSD
        squared_diff = torch.sum((p_aligned - t_valid)**2, dim=1)
        rmsd = torch.sqrt(torch.mean(squared_diff)).item()
        rmsd_values.append(rmsd)
    
    return rmsd_values
```

2. **TM-score (Template Modeling score)**: Length-normalized structural similarity score (0-1, higher is better)

For TM-score, we recommend using the US-align external tool:

```python
def calculate_tm_score(pred_pdb_path, true_pdb_path):
    """
    Calculate TM-score using US-align external tool.
    
    Args:
        pred_pdb_path: Path to predicted PDB file
        true_pdb_path: Path to true PDB file
        
    Returns:
        TM-score value
    """
    # Run US-align command
    cmd = f"USalign {pred_pdb_path} {true_pdb_path} -mm"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse TM-score from output
    for line in result.stdout.split('\n'):
        if "TM-score=" in line:
            tm_score = float(line.split("TM-score=")[1].split()[0])
            return tm_score
    
    # Return NaN if TM-score not found
    return float('nan')
```

3. **lDDT (local Distance Difference Test)**: Measures preservation of local distances (0-1, higher is better)

```python
def calculate_lddt(pred_coords, true_coords, mask=None, cutoffs=[0.5, 1.0, 2.0, 4.0]):
    """
    Calculate lDDT score between predicted and true coordinates.
    
    Args:
        pred_coords: Predicted coordinates (batch_size, seq_len, 3)
        true_coords: True coordinates (batch_size, seq_len, 3)
        mask: Boolean mask (batch_size, seq_len)
        cutoffs: Distance cutoffs for lDDT calculation
        
    Returns:
        lDDT scores per sequence
    """
    batch_size = pred_coords.shape[0]
    lddt_values = []
    
    for i in range(batch_size):
        # Apply mask if provided
        if mask is not None:
            valid_mask = mask[i]
            p_valid = pred_coords[i, valid_mask]
            t_valid = true_coords[i, valid_mask]
        else:
            p_valid = pred_coords[i]
            t_valid = true_coords[i]
        
        # Skip if too few valid positions
        if len(p_valid) < 2:
            lddt_values.append(float('nan'))
            continue
        
        # Calculate all pairwise distances for true coordinates
        true_dists = torch.cdist(t_valid, t_valid)
        
        # Calculate all pairwise distances for predicted coordinates
        pred_dists = torch.cdist(p_valid, p_valid)
        
        # Calculate lDDT score
        N = len(p_valid)
        total_pairs = 0
        preserved_pairs = 0
        
        # Exclude self-distances
        true_dists_flat = true_dists.flatten()[1:].view(N, N-1)
        pred_dists_flat = pred_dists.flatten()[1:].view(N, N-1)
        
        # Count preserved distances for each cutoff
        for cutoff in cutoffs:
            # Consider only distances within cutoff
            within_cutoff = true_dists_flat <= cutoff
            
            # Count valid pairs within cutoff
            n_valid = within_cutoff.sum().item()
            total_pairs += n_valid
            
            if n_valid > 0:
                # Calculate absolute differences for valid pairs
                diffs = torch.abs(true_dists_flat - pred_dists_flat)
                
                # Count preserved distances (diff <= 0.5Å)
                n_preserved = ((diffs <= 0.5) & within_cutoff).sum().item()
                preserved_pairs += n_preserved
        
        # Calculate lDDT score
        if total_pairs > 0:
            lddt = preserved_pairs / total_pairs
        else:
            lddt = float('nan')
        
        lddt_values.append(lddt)
    
    return lddt_values
```

#### 4.1.2 Angle Prediction Metrics

For RNA pseudo-dihedral angle prediction:

```python
def calculate_angle_error(pred_angles, true_angles, mask=None):
    """
    Calculate angular error between predicted and true angles.
    
    Args:
        pred_angles: Predicted angles in sin/cos form (batch_size, seq_len, 4)
        true_angles: True angles in sin/cos form (batch_size, seq_len, 4)
        mask: Boolean mask (batch_size, seq_len)
        
    Returns:
        Angular error in degrees
    """
    batch_size = pred_angles.shape[0]
    angle_errors = []
    
    for i in range(batch_size):
        # Apply mask if provided
        if mask is not None:
            valid_mask = mask[i]
            p_valid = pred_angles[i, valid_mask]
            t_valid = true_angles[i, valid_mask]
        else:
            p_valid = pred_angles[i]
            t_valid = true_angles[i]
        
        # Skip if no valid positions
        if len(p_valid) == 0:
            angle_errors.append(float('nan'))
            continue
        
        # Convert sin/cos back to angles
        # Each angle is represented by [sin(θ), cos(θ)]
        pred_eta_sin, pred_eta_cos = p_valid[:, 0], p_valid[:, 1]
        pred_theta_sin, pred_theta_cos = p_valid[:, 2], p_valid[:, 3]
        
        true_eta_sin, true_eta_cos = t_valid[:, 0], t_valid[:, 1]
        true_theta_sin, true_theta_cos = t_valid[:, 2], t_valid[:, 3]
        
        # Calculate angles in radians
        pred_eta = torch.atan2(pred_eta_sin, pred_eta_cos)
        pred_theta = torch.atan2(pred_theta_sin, pred_theta_cos)
        
        true_eta = torch.atan2(true_eta_sin, true_eta_cos)
        true_theta = torch.atan2(true_theta_sin, true_theta_cos)
        
        # Calculate angular differences (ensuring -π to π range)
        eta_diff = torch.remainder(pred_eta - true_eta + torch.pi, 2 * torch.pi) - torch.pi
        theta_diff = torch.remainder(pred_theta - true_theta + torch.pi, 2 * torch.pi) - torch.pi
        
        # Convert to degrees
        eta_diff_deg = torch.abs(eta_diff) * 180 / torch.pi
        theta_diff_deg = torch.abs(theta_diff) * 180 / torch.pi
        
        # Calculate mean angular error
        mean_error = (eta_diff_deg.mean() + theta_diff_deg.mean()) / 2
        angle_errors.append(mean_error.item())
    
    return angle_errors
```

### 4.2 Memory Profiling

Profile GPU memory usage during execution:

```python
def profile_memory_usage(model, dataloader, device='cuda', n_batches=5):
    """
    Profile GPU memory usage during model execution.
    
    Args:
        model: The model to profile
        dataloader: DataLoader providing batches
        device: Device to run on ('cuda' or 'cpu')
        n_batches: Number of batches to process
        
    Returns:
        Dictionary of memory usage statistics
    """
    if device != 'cuda' or not torch.cuda.is_available():
        print("Memory profiling only available on CUDA devices")
        return {}
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Initialize memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    memory_stats = {
        'baseline_allocated': torch.cuda.memory_allocated(),
        'baseline_reserved': torch.cuda.memory_reserved(),
        'per_batch': []
    }
    
    # Process batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        
        # Record pre-batch memory
        pre_batch_allocated = torch.cuda.memory_allocated()
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Record post-load memory
        post_load_allocated = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.no_grad():
            outputs = model(batch)
        
        # Record post-forward memory
        post_forward_allocated = torch.cuda.memory_allocated()
        
        # Calculate losses
        fape_loss = compute_fape_loss(
            outputs['pred_coords'],
            batch['coordinates'],
            batch['mask']
        )
        
        confidence_loss = compute_confidence_loss(
            outputs['pred_confidence'],
            outputs['pred_coords'],
            batch['coordinates'],
            batch['mask']
        )
        
        angle_loss = compute_angle_loss(
            outputs['pred_angles'],
            batch['dihedral_features'],
            batch['mask']
        )
        
        # Record post-loss memory
        post_loss_allocated = torch.cuda.memory_allocated()
        
        # Record peak memory
        peak_memory = torch.cuda.max_memory_allocated()
        
        # Reset peak memory stats for next batch
        torch.cuda.reset_peak_memory_stats()
        
        # Record batch stats
        batch_stats = {
            'batch_idx': batch_idx,
            'batch_size': len(batch['target_ids']),
            'max_seq_len': batch['sequence_int'].shape[1],
            'pre_batch_mb': pre_batch_allocated / (1024**2),
            'post_load_mb': post_load_allocated / (1024**2),
            'post_forward_mb': post_forward_allocated / (1024**2),
            'post_loss_mb': post_loss_allocated / (1024**2),
            'peak_mb': peak_memory / (1024**2),
            'batch_increment_mb': (post_load_allocated - pre_batch_allocated) / (1024**2),
            'forward_increment_mb': (post_forward_allocated - post_load_allocated) / (1024**2),
            'loss_increment_mb': (post_loss_allocated - post_forward_allocated) / (1024**2)
        }
        
        memory_stats['per_batch'].append(batch_stats)
        
        # Print batch stats
        print(f"Batch {batch_idx}: Size={batch_stats['batch_size']}, "
              f"Seq_Len={batch_stats['max_seq_len']}")
        print(f"  Memory: Load: +{batch_stats['batch_increment_mb']:.2f}MB, "
              f"Forward: +{batch_stats['forward_increment_mb']:.2f}MB, "
              f"Loss: +{batch_stats['loss_increment_mb']:.2f}MB")
        print(f"  Peak: {batch_stats['peak_mb']:.2f}MB")
    
    # Calculate overall statistics
    memory_stats['max_allocated_mb'] = max(b['post_loss_mb'] for b in memory_stats['per_batch'])
    memory_stats['max_peak_mb'] = max(b['peak_mb'] for b in memory_stats['per_batch'])
    memory_stats['avg_batch_increment_mb'] = sum(b['batch_increment_mb'] for b in memory_stats['per_batch']) / len(memory_stats['per_batch'])
    memory_stats['avg_forward_increment_mb'] = sum(b['forward_increment_mb'] for b in memory_stats['per_batch']) / len(memory_stats['per_batch'])
    
    return memory_stats
```

### 4.3 Computational Efficiency Assessment

Measure inference time and scaling with sequence length:

```python
def benchmark_performance(model, dataloader, device='cuda', n_batches=5, n_repeats=3):
    """
    Benchmark model inference time and scaling.
    
    Args:
        model: The model to benchmark
        dataloader: DataLoader providing batches
        device: Device to run on ('cuda' or 'cpu')
        n_batches: Number of batches to process
        n_repeats: Number of repeats for timing stability
        
    Returns:
        Dictionary of performance statistics
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Initialize timings
    timing_stats = {
        'per_batch': []
    }
    
    # Process batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches:
            break
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Get batch metadata
        batch_size = len(batch['target_ids'])
        seq_len = batch['sequence_int'].shape[1]
        
        # Warm-up run
        with torch.no_grad():
            _ = model(batch)
        
        # Synchronize GPU
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Time forward passes
        forward_times = []
        for _ in range(n_repeats):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(batch)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            forward_times.append(end_time - start_time)
        
        # Time loss calculation
        loss_times = []
        for _ in range(n_repeats):
            with torch.no_grad():
                outputs = model(batch)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                fape_loss = compute_fape_loss(
                    outputs['pred_coords'],
                    batch['coordinates'],
                    batch['mask']
                )
                
                confidence_loss = compute_confidence_loss(
                    outputs['pred_confidence'],
                    outputs['pred_coords'],
                    batch['coordinates'],
                    batch['mask']
                )
                
                angle_loss = compute_angle_loss(
                    outputs['pred_angles'],
                    batch['dihedral_features'],
                    batch['mask']
                )
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            loss_times.append(end_time - start_time)
        
        # Calculate timing statistics
        avg_forward_time = sum(forward_times) / len(forward_times)
        avg_loss_time = sum(loss_times) / len(loss_times)
        
        # Record batch stats
        batch_stats = {
            'batch_idx': batch_idx,
            'batch_size': batch_size,
            'max_seq_len': seq_len,
            'avg_forward_time': avg_forward_time,
            'avg_loss_time': avg_loss_time,
            'total_time': avg_forward_time + avg_loss_time,
            'time_per_seq': (avg_forward_time + avg_loss_time) / batch_size,
            'forward_times': forward_times,
            'loss_times': loss_times
        }
        
        timing_stats['per_batch'].append(batch_stats)
        
        # Print batch stats
        print(f"Batch {batch_idx}: Size={batch_size}, Seq_Len={seq_len}")
        print(f"  Timing: Forward: {avg_forward_time*1000:.2f}ms, "
              f"Loss: {avg_loss_time*1000:.2f}ms, "
              f"Total: {(avg_forward_time + avg_loss_time)*1000:.2f}ms")
        print(f"  Time per sequence: {batch_stats['time_per_seq']*1000:.2f}ms")
    
    # Calculate overall statistics
    timing_stats['avg_forward_time'] = sum(b['avg_forward_time'] for b in timing_stats['per_batch']) / len(timing_stats['per_batch'])
    timing_stats['avg_loss_time'] = sum(b['avg_loss_time'] for b in timing_stats['per_batch']) / len(timing_stats['per_batch'])
    timing_stats['avg_total_time'] = sum(b['total_time'] for b in timing_stats['per_batch']) / len(timing_stats['per_batch'])
    
    # Analyze scaling with sequence length
    seq_lengths = [b['max_seq_len'] for b in timing_stats['per_batch']]
    total_times = [b['total_time'] for b in timing_stats['per_batch']]
    
    # If we have multiple sequence lengths, fit a scaling model
    if len(set(seq_lengths)) > 1:
        try:
            from scipy import stats
            log_lengths = np.log(seq_lengths)
            log_times = np.log(total_times)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_lengths, log_times)
            timing_stats['scaling_model'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'scaling_description': f"Time ~ Sequence_Length^{slope:.2f}"
            }
            print(f"Scaling analysis: Time ~ Sequence_Length^{slope:.2f} (R² = {r_value**2:.3f})")
        except Exception as e:
            print(f"Could not perform scaling analysis: {e}")
    
    return timing_stats
```

### 4.4 Results Logging and Visualization

Create visualizations of test results:

```python
def visualize_test_results(results, output_dir):
    """
    Create visualizations of test results.
    
    Args:
        results: Dictionary of test results
        output_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot RMSD distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['rmsd_values'], bins=20, alpha=0.7)
    plt.axvline(results['avg_rmsd'], color='r', linestyle='--', label=f"Mean: {results['avg_rmsd']:.2f}Å")
    plt.xlabel('RMSD (Å)')
    plt.ylabel('Count')
    plt.title('Distribution of RMSD Values')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'rmsd_distribution.png'))
    plt.close()
    
    # Plot lDDT vs sequence length
    plt.figure(figsize=(10, 6))
    plt.scatter(results['sequence_lengths'], results['lddt_values'], alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('lDDT Score')
    plt.title('lDDT Score vs Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'lddt_vs_length.png'))
    plt.close()
    
    # Plot memory usage
    if 'memory_stats' in results:
        memory_data = results['memory_stats']
        
        # Batch sizes vs memory usage
        plt.figure(figsize=(10, 6))
        batch_sizes = [b['batch_size'] for b in memory_data['per_batch']]
        peak_memory = [b['peak_mb'] for b in memory_data['per_batch']]
        plt.scatter(batch_sizes, peak_memory, alpha=0.7)
        plt.xlabel('Batch Size')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.title('Memory Usage vs Batch Size')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'memory_vs_batch.png'))
        plt.close()
        
        # Sequence length vs memory usage
        plt.figure(figsize=(10, 6))
        seq_lengths = [b['max_seq_len'] for b in memory_data['per_batch']]
        peak_memory = [b['peak_mb'] for b in memory_data['per_batch']]
        plt.scatter(seq_lengths, peak_memory, alpha=0.7)
        plt.xlabel('Sequence Length')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.title('Memory Usage vs Sequence Length')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'memory_vs_length.png'))
        plt.close()
    
    # Plot timing data
    if 'timing_stats' in results:
        timing_data = results['timing_stats']
        
        # Sequence length vs inference time
        plt.figure(figsize=(10, 6))
        seq_lengths = [b['max_seq_len'] for b in timing_data['per_batch']]
        inference_times = [b['total_time'] * 1000 for b in timing_data['per_batch']]  # Convert to ms
        plt.scatter(seq_lengths, inference_times, alpha=0.7)
        plt.xlabel('Sequence Length')
        plt.ylabel('Inference Time (ms)')
        plt.title('Inference Time vs Sequence Length')
        plt.grid(True, alpha=0.3)
        
        # If we have scaling information, add trend line
        if 'scaling_model' in timing_data:
            scaling = timing_data['scaling_model']
            x_range = np.linspace(min(seq_lengths), max(seq_lengths), 100)
            y_range = np.exp(scaling['intercept'] + scaling['slope'] * np.log(x_range)) * 1000  # Convert to ms
            plt.plot(x_range, y_range, 'r--', 
                    label=f"Scaling: L^{scaling['slope']:.2f} (R²={scaling['r_squared']:.3f})")
            plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'time_vs_length.png'))
        plt.close()
    
    # Save summary text file
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("RNA 3D Folding Pipeline Test Results\n")
        f.write("===================================\n\n")
        
        f.write(f"Test Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Structure Accuracy Metrics:\n")
        f.write(f"  Average RMSD: {results['avg_rmsd']:.4f}Å\n")
        f.write(f"  Average lDDT Score: {results['avg_lddt']:.4f}\n")
        f.write(f"  Average TM-score: {results.get('avg_tm_score', 'N/A')}\n\n")
        
        f.write("Loss Values:\n")
        f.write(f"  Average FAPE Loss: {results['avg_fape_loss']:.4f}\n")
        f.write(f"  Average Confidence Loss: {results['avg_confidence_loss']:.4f}\n")
        f.write(f"  Average Angle Loss: {results['avg_angle_loss']:.4f}\n\n")
        
        if 'memory_stats' in results:
            memory_data = results['memory_stats']
            f.write("Memory Usage:\n")
            f.write(f"  Maximum Allocated: {memory_data['max_allocated_mb']:.2f} MB\n")
            f.write(f"  Maximum Peak: {memory_data['max_peak_mb']:.2f} MB\n\n")
        
        if 'timing_stats' in results:
            timing_data = results['timing_stats']
            f.write("Performance Metrics:\n")
            f.write(f"  Average Forward Time: {timing_data['avg_forward_time']*1000:.2f} ms\n")
            f.write(f"  Average Total Time: {timing_data['avg_total_time']*1000:.2f} ms\n")
            if 'scaling_model' in timing_data:
                scaling = timing_data['scaling_model']
                f.write(f"  Scaling with Sequence Length: Time ~ L^{scaling['slope']:.2f} (R²={scaling['r_squared']:.3f})\n")
```

## 5. Regression Testing & Validation

### 5.1 Baseline Performance Tracking

Create a baseline against which to compare future changes:

```python
def create_baseline(test_results, baseline_path):
    """
    Create a baseline from test results.
    
    Args:
        test_results: Dictionary of test results
        baseline_path: Path to save baseline
    """
    # Save full results as baseline
    with open(baseline_path, 'wb') as f:
        pickle.dump(test_results, f)
    
    print(f"Saved baseline to {baseline_path}")
    
    # Also save a readable summary
    summary_path = baseline_path.replace('.pkl', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RNA 3D Folding Pipeline Baseline\n")
        f.write("================================\n\n")
        
        f.write(f"Creation Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Structure Accuracy Metrics:\n")
        f.write(f"  Average RMSD: {test_results['avg_rmsd']:.4f}Å\n")
        f.write(f"  Average lDDT Score: {test_results['avg_lddt']:.4f}\n\n")
        
        f.write("Loss Values:\n")
        f.write(f"  Average FAPE Loss: {test_results['avg_fape_loss']:.4f}\n")
        f.write(f"  Average Confidence Loss: {test_results['avg_confidence_loss']:.4f}\n")
        f.write(f"  Average Angle Loss: {test_results['avg_angle_loss']:.4f}\n\n")
        
        if 'memory_stats' in test_results:
            memory_data = test_results['memory_stats']
            f.write("Memory Usage:\n")
            f.write(f"  Maximum Allocated: {memory_data['max_allocated_mb']:.2f} MB\n")
            f.write(f"  Maximum Peak: {memory_data['max_peak_mb']:.2f} MB\n\n")
        
        if 'timing_stats' in test_results:
            timing_data = test_results['timing_stats']
            f.write("Performance Metrics:\n")
            f.write(f"  Average Forward Time: {timing_data['avg_forward_time']*1000:.2f} ms\n")
            f.write(f"  Average Total Time: {timing_data['avg_total_time']*1000:.2f} ms\n")
    
    print(f"Saved readable summary to {summary_path}")
```

### 5.2 Compare Against Baseline

When implementing changes, compare against the baseline:

```python
def compare_with_baseline(test_results, baseline_path, output_path):
    """
    Compare test results with baseline.
    
    Args:
        test_results: Dictionary of current test results
        baseline_path: Path to baseline pickle file
        output_path: Path to save comparison results
    
    Returns:
        Dictionary with comparison metrics
    """
    # Load baseline
    with open(baseline_path, 'rb') as f:
        baseline = pickle.load(f)
    
    # Calculate differences
    comparison = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': {}
    }
    
    # Structure metrics
    for metric in ['avg_rmsd', 'avg_lddt']:
        if metric in test_results and metric in baseline:
            diff = test_results[metric] - baseline[metric]
            pct_change = diff / baseline[metric] * 100 if baseline[metric] != 0 else float('inf')
            
            # For RMSD, lower is better; for lDDT, higher is better
            if metric == 'avg_rmsd':
                improved = diff < 0
            else:
                improved = diff > 0
            
            comparison['metrics'][metric] = {
                'baseline': baseline[metric],
                'current': test_results[metric],
                'diff': diff,
                'pct_change': pct_change,
                'improved': improved
            }
    
    # Loss metrics
    for metric in ['avg_fape_loss', 'avg_confidence_loss', 'avg_angle_loss']:
        if metric in test_results and metric in baseline:
            diff = test_results[metric] - baseline[metric]
            pct_change = diff / baseline[metric] * 100 if baseline[metric] != 0 else float('inf')
            
            # For losses, lower is better
            improved = diff < 0
            
            comparison['metrics'][metric] = {
                'baseline': baseline[metric],
                'current': test_results[metric],
                'diff': diff,
                'pct_change': pct_change,
                'improved': improved
            }
    
    # Performance metrics
    if 'timing_stats' in test_results and 'timing_stats' in baseline:
        for metric in ['avg_forward_time', 'avg_total_time']:
            if metric in test_results['timing_stats'] and metric in baseline['timing_stats']:
                base_val = baseline['timing_stats'][metric]
                current_val = test_results['timing_stats'][metric]
                diff = current_val - base_val
                pct_change = diff / base_val * 100 if base_val != 0 else float('inf')
                
                # For timing, lower is better
                improved = diff < 0
                
                comparison['metrics'][f"timing_{metric}"] = {
                    'baseline': base_val,
                    'current': current_val,
                    'diff': diff,
                    'pct_change': pct_change,
                    'improved': improved
                }
    
    # Memory metrics
    if 'memory_stats' in test_results and 'memory_stats' in baseline:
        for metric in ['max_allocated_mb', 'max_peak_mb']:
            if metric in test_results['memory_stats'] and metric in baseline['memory_stats']:
                base_val = baseline['memory_stats'][metric]
                current_val = test_results['memory_stats'][metric]
                diff = current_val - base_val
                pct_change = diff / base_val * 100 if base_val != 0 else float('inf')
                
                # For memory, lower is better
                improved = diff < 0
                
                comparison['metrics'][f"memory_{metric}"] = {
                    'baseline': base_val,
                    'current': current_val,
                    'diff': diff,
                    'pct_change': pct_change,
                    'improved': improved
                }
    
    # Count improvements and regressions
    improvements = sum(1 for m in comparison['metrics'].values() if m['improved'])
    regressions = sum(1 for m in comparison['metrics'].values() if not m['improved'])
    total_metrics = len(comparison['metrics'])
    
    comparison['summary'] = {
        'improvements': improvements,
        'regressions': regressions,
        'total_metrics': total_metrics,
        'improvement_ratio': improvements / total_metrics if total_metrics > 0 else 0
    }
    
    # Save comparison report
    with open(output_path, 'w') as f:
        f.write("RNA 3D Folding Pipeline Comparison Report\n")
        f.write("========================================\n\n")
        
        f.write(f"Comparison Date: {comparison['timestamp']}\n")
        f.write(f"Baseline: {os.path.basename(baseline_path)}\n\n")
        
        f.write(f"Summary: {improvements} improvements, {regressions} regressions out of {total_metrics} metrics\n\n")
        
        f.write("Metrics Comparison:\n")
        f.write("------------------\n\n")
        
        # Print metrics by category
        categories = {
            'Structure Metrics': ['avg_rmsd', 'avg_lddt'],
            'Loss Values': ['avg_fape_loss', 'avg_confidence_loss', 'avg_angle_loss'],
            'Timing Performance': [m for m in comparison['metrics'] if m.startswith('timing_')],
            'Memory Usage': [m for m in comparison['metrics'] if m.startswith('memory_')]
        }
        
        for category, metrics in categories.items():
            f.write(f"{category}:\n")
            for metric in metrics:
                if metric in comparison['metrics']:
                    m = comparison['metrics'][metric]
                    sign = "↓" if (metric == 'avg_rmsd' or 'loss' in metric or 'timing_' in metric or 'memory_' in metric) else "↑"
                    change_sign = "-" if m['diff'] < 0 else "+"
                    status = "✓" if m['improved'] else "✗"
                    
                    # Format the values appropriately
                    if 'timing_' in metric:
                        # Convert to milliseconds for readability
                        baseline_val = f"{m['baseline']*1000:.2f} ms"
                        current_val = f"{m['current']*1000:.2f} ms"
                        diff_val = f"{change_sign}{abs(m['diff'])*1000:.2f} ms"
                    elif 'memory_' in metric:
                        baseline_val = f"{m['baseline']:.2f} MB"
                        current_val = f"{m['current']:.2f} MB"
                        diff_val = f"{change_sign}{abs(m['diff']):.2f} MB"
                    else:
                        baseline_val = f"{m['baseline']:.4f}"
                        current_val = f"{m['current']:.4f}"
                        diff_val = f"{change_sign}{abs(m['diff']):.4f}"
                    
                    f.write(f"  {metric.replace('avg_', '').replace('timing_', '').replace('memory_', '')} ({sign}): ")
                    f.write(f"{baseline_val} → {current_val} ({diff_val}, {change_sign}{abs(m['pct_change']):.2f}%) {status}\n")
            f.write("\n")
    
    print(f"Saved comparison report to {output_path}")
    return comparison
```

### 5.3 Edge Case Testing

Specifically test challenging edge cases:

1. **Very Short Sequences** (5-10 nucleotides)
2. **Very Long Sequences** (300+ nucleotides)
3. **Batches with Highly Variable Lengths**
4. **Sequences with Complex Structures** (multiple junctions, pseudoknots)

```python
def test_edge_cases(model, config_path, edge_case_dir, output_dir):
    """
    Test model performance on edge cases.
    
    Args:
        model: The model to test
        config_path: Path to configuration file
        edge_case_dir: Directory with edge case test data
        output_dir: Directory to save results
    """
    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Test cases and their descriptions
    test_cases = [
        {
            'name': 'short_sequences',
            'desc': 'Very short RNA sequences (5-10 nt)',
            'seq_csv': os.path.join(edge_case_dir, 'short_sequences.csv'),
            'label_csv': os.path.join(edge_case_dir, 'short_labels.csv'),
            'feature_dir': os.path.join(edge_case_dir, 'features')
        },
        {
            'name': 'long_sequences',
            'desc': 'Very long RNA sequences (300+ nt)',
            'seq_csv': os.path.join(edge_case_dir, 'long_sequences.csv'),
            'label_csv': os.path.join(edge_case_dir, 'long_labels.csv'),
            'feature_dir': os.path.join(edge_case_dir, 'features')
        },
        {
            'name': 'variable_lengths',
            'desc': 'Batch with highly variable sequence lengths',
            'seq_csv': os.path.join(edge_case_dir, 'variable_sequences.csv'),
            'label_csv': os.path.join(edge_case_dir, 'variable_labels.csv'),
            'feature_dir': os.path.join(edge_case_dir, 'features')
        },
        {
            'name': 'complex_structures',
            'desc': 'Sequences with complex structural motifs',
            'seq_csv': os.path.join(edge_case_dir, 'complex_sequences.csv'),
            'label_csv': os.path.join(edge_case_dir, 'complex_labels.csv'),
            'feature_dir': os.path.join(edge_case_dir, 'features')
        }
    ]
    
    # Results storage
    edge_case_results = {}
    
    # Run each test case
    for case in test_cases:
        print(f"\nTesting edge case: {case['name']} - {case['desc']}")
        
        try:
            # Create dataset
            dataset = RNADataset(
                sequences_csv_path=case['seq_csv'],
                labels_csv_path=case['label_csv'],
                features_dir=case['feature_dir']
            )
            
            # Create dataloader with batch size 1 for challenging cases
            dataloader = DataLoader(
                dataset, 
                batch_size=1,  # Use batch_size=1 for edge cases
                shuffle=False, 
                collate_fn=collate_fn
            )
            
            # Test on this case
            results = test_end_to_end_pipeline(
                model, 
                dataloader, 
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Store results
            edge_case_results[case['name']] = results
            
            # Save individual case report
            case_output_dir = os.path.join(output_dir, case['name'])
            os.makedirs(case_output_dir, exist_ok=True)
            
            with open(os.path.join(case_output_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Visualize case results
            visualize_test_results(results, case_output_dir)
            
            print(f"Successfully tested {case['name']}")
            print(f"  RMSD: {results['avg_rmsd']:.4f}Å")
            print(f"  lDDT: {results['avg_lddt']:.4f}")
            
        except Exception as e:
            print(f"Error testing {case['name']}: {e}")
            edge_case_results[case['name']] = {'error': str(e)}
    
    # Create summary report
    summary_path = os.path.join(output_dir, 'edge_case_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RNA 3D Folding Pipeline Edge Case Testing\n")
        f.write("========================================\n\n")
        
        f.write(f"Test Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for case in test_cases:
            f.write(f"{case['name']} - {case['desc']}:\n")
            
            if 'error' in edge_case_results[case['name']]:
                f.write(f"  ERROR: {edge_case_results[case['name']]['error']}\n")
            else:
                results = edge_case_results[case['name']]
                f.write(f"  RMSD: {results['avg_rmsd']:.4f}Å\n")
                f.write(f"  lDDT: {results['avg_lddt']:.4f}\n")
                f.write(f"  FAPE Loss: {results['avg_fape_loss']:.4f}\n")
            
            f.write("\n")
    
    return edge_case_results
```

## 6. Common Issues & Troubleshooting

### 6.1 Data Pipeline Issues

| Issue | Symptoms | Troubleshooting Steps |
|-------|----------|----------------------|
| Missing feature files | `KeyError` or `FileNotFoundError` | Check feature file paths and preprocessing steps |
| Shape mismatch | `RuntimeError` in model forward pass | Print tensor shapes at each step, verify collate function |
| Memory errors | CUDA out of memory | Reduce batch size, check for tensor leaks, use gradient checkpointing |
| NaN losses | `nan` values in loss output | Check normalization, use `torch.isnan`, verify no division by zero |

### 6.2 Debugging Guide

When encountering issues, follow this systematic approach:

1. **Check Inputs**:
   ```python
   def debug_model_inputs(batch):
       """Print debug information about model inputs."""
       print("Batch keys:", batch.keys())
       for key, value in batch.items():
           if isinstance(value, torch.Tensor):
               print(f"{key}: shape={value.shape}, dtype={value.dtype}, "
                     f"device={value.device}, range=[{value.min():.4f}, {value.max():.4f}]")
               # Check for NaNs or infinities
               if torch.isnan(value).any():
                   print(f"  WARNING: {key} contains NaN values")
               if torch.isinf(value).any():
                   print(f"  WARNING: {key} contains infinite values")
   ```

2. **Trace Model Execution**:
   ```python
   # Add hooks to monitor activations
   activation = {}
   def get_activation(name):
       def hook(model, input, output):
           activation[name] = output
       return hook

   # Register hooks
   model.transformer_blocks[0].register_forward_hook(get_activation('transformer_block_0'))
   model.ipa_module.register_forward_hook(get_activation('ipa_module'))
   ```

3. **Check Gradient Flow**:
   ```python
   def plot_grad_flow(named_parameters):
       """Plot gradients flowing through different layers."""
       ave_grads = []
       max_grads = []
       layers = []
       for n, p in named_parameters:
           if(p.requires_grad) and ("bias" not in n):
               layers.append(n)
               ave_grads.append(p.grad.abs().mean().item())
               max_grads.append(p.grad.abs().max().item())
       plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
       plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
       plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
       plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
       plt.xlim(left=0, right=len(ave_grads))
       plt.ylim(bottom=0, top=0.02)
       plt.xlabel("Layers")
       plt.ylabel("average gradient")
       plt.title("Gradient flow")
       plt.grid(True)
       plt.tight_layout()
   ```

### 6.3 Common GPU Memory Issues

If experiencing GPU memory issues:

1. **Profile Memory Usage**:
   ```python
   # Track tensor memory
   def print_tensor_memory():
       """Print memory usage of all tensors."""
       for obj in gc.get_objects():
           try:
               if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                   print(type(obj), obj.size(), obj.device)
           except:
               pass
   ```

2. **Fix Memory Leaks**:
   ```python
   # Common memory leak fixes
   # 1. Clear cache periodically
   torch.cuda.empty_cache()
   
   # 2. Use context manager for inference
   with torch.no_grad():
       outputs = model(batch)
   
   # 3. Delete intermediate tensors
   del tensor_a, tensor_b
   ```

3. **Optimize Model Size**:
   ```python
   # Reduce model size by changing config
   config['model']['residue_embed_dim'] = 64  # Reduced from 128
   config['model']['pair_embed_dim'] = 32     # Reduced from 64
   config['model']['num_transformer_blocks'] = 4  # Reduced from 8
   ```

## 7. Integration with Development Workflow

### 7.1 Integration into Development Cycle

Integrate testing at key points in the development workflow:

1. **When to Run Tests**:
   - Unit tests should be run after any component changes
   - Integration tests should be run after significant architecture changes
   - Full pipeline tests should be run before version releases
   - Performance benchmarks should be run when optimizing the code

2. **Automation Script**:
   ```bash
   #!/bin/bash
   # run_tests.sh
   
   # Run unit tests
   echo "Running unit tests..."
   python -m pytest tests/
   
   # Run integration test if unit tests pass
   if [ $? -eq 0 ]; then
       echo "Running integration test..."
       python scripts/test_pipeline.py --config config/test_config.yaml --quick
   else
       echo "Unit tests failed. Skipping integration test."
       exit 1
   fi
   
   # Run full pipeline test if integration test passes
   if [ $? -eq 0 ]; then
       echo "Running full pipeline test..."
       python scripts/test_pipeline.py --config config/test_config.yaml --full
   else
       echo "Integration test failed. Skipping full pipeline test."
       exit 1
   fi
   ```

### 7.2 Version Tracking & Reporting

Track results across different model versions:

```python
def track_version_performance(version, test_results, history_file):
    """
    Track performance across versions.
    
    Args:
        version: Version identifier (e.g., 'v1.2')
        test_results: Dictionary of test results
        history_file: Path to history JSON file
    """
    # Load history if exists
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {'versions': []}
    
    # Extract key metrics
    metrics = {
        'version': version,
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'avg_rmsd': test_results['avg_rmsd'],
        'avg_lddt': test_results['avg_lddt'],
        'avg_fape_loss': test_results['avg_fape_loss'],
        'avg_confidence_loss': test_results['avg_confidence_loss'],
        'avg_angle_loss': test_results['avg_angle_loss']
    }
    
    # Add performance metrics if available
    if 'timing_stats' in test_results:
        metrics['avg_inference_time_ms'] = test_results['timing_stats']['avg_total_time'] * 1000
    
    if 'memory_stats' in test_results:
        metrics['peak_memory_mb'] = test_results['memory_stats']['max_peak_mb']
    
    # Add to history
    history['versions'].append(metrics)
    
    # Save updated history
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"Updated version history in {history_file}")
    
    # Generate graph of performance over versions
    if len(history['versions']) > 1:
        versions = [v['version'] for v in history['versions']]
        rmsd_values = [v['avg_rmsd'] for v in history['versions']]
        lddt_values = [v['avg_lddt'] for v in history['versions']]
        
        plt.figure(figsize=(12, 6))
        
        # Plot RMSD (lower is better)
        ax1 = plt.subplot(121)
        ax1.plot(versions, rmsd_values, 'b-o')
        ax1.set_ylabel('RMSD (Å)', color='b')
        ax1.set_xlabel('Version')
        ax1.set_title('RMSD over Versions')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # Plot lDDT (higher is better)
        ax2 = plt.subplot(122)
        ax2.plot(versions, lddt_values, 'g-o')
        ax2.set_ylabel('lDDT Score', color='g')
        ax2.set_xlabel('Version')
        ax2.set_title('lDDT over Versions')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(history_file), 'version_performance.png'))
        plt.close()
```

## 8. Conclusion

This testing workflow provides a comprehensive framework for validating the RNA 3D folding pipeline. By following these procedures, we can ensure that:

1. **All components function correctly** in isolation and when integrated
2. **Scientific predictions are accurate** according to established metrics
3. **Resource usage is optimized** for both local execution and Kaggle
4. **Regressions are detected early** in the development process
5. **Edge cases are handled properly** across various RNA sequence types

The workflow is designed to be:
- **Reproducible**: All tests can be run consistently with the same results
- **Informative**: Clear reporting of test outcomes and performance metrics
- **Extensible**: Easy to add new test cases or validation methods
- **Integrated**: Fits naturally into the development process

By maintaining this rigorous testing approach, we can develop a robust, high-performance RNA 3D structure prediction pipeline that meets all project requirements.
