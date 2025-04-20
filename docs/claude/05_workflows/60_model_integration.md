# Model Integration Workflow

## 1. Overview

The Model Integration Workflow guides the process of unifying our separately implemented components—data loading, embeddings, transformer blocks, IPA module, and loss functions—into a cohesive RNA 3D folding pipeline. This workflow bridges the gap between modular component development and a trainable end-to-end system. Successful integration results in a functional model that properly processes input features through the entire architecture, generates predictions, and calculates losses to enable gradient-based training.

## 2. Prerequisites

Before beginning the integration process, ensure you have:

- Successfully implemented and tested individual components:
  - `src/data_loading.py` - RNADataset and collate_fn
  - `src/models/embeddings.py` - SequenceEmbedding, PositionalEncoding, RelativePositionalEncoding
  - `src/models/transformer_block.py` - TransformerBlock implementation
  - `src/models/ipa_module.py` - IPAModule (V1 placeholder)
  - `src/losses.py` - FAPE proxy loss, confidence loss, angle loss
- Verified each component with passing unit tests
- Created `config/default_config.yaml` with scaled-down parameters for V1
- Established project directory structure as described in `1_Context_and_Setup.md`

## 3. Step-by-Step Integration Process

### 3.1. Component Inventory and Verification

1. **Review implemented components**
   ```python
   # Verify each component is importable and functional
   from src.data_loading import RNADataset, collate_fn, create_data_loader
   from src.models.embeddings import SequenceEmbedding, PositionalEncoding, RelativePositionalEncoding
   from src.models.transformer_block import TransformerBlock
   from src.models.ipa_module import IPAModule
   from src.losses import compute_fape_loss, compute_confidence_loss, compute_angle_loss
   ```

2. **Create a simple integration test script** (`scripts/test_import.py`)
   ```python
   """Minimal test script to verify component imports."""
   import os
   import torch
   import yaml
   
   # Test importing all components - will raise error if any are missing
   from src.data_loading import RNADataset, collate_fn, create_data_loader
   from src.models.embeddings import SequenceEmbedding, PositionalEncoding, RelativePositionalEncoding
   from src.models.transformer_block import TransformerBlock
   from src.models.ipa_module import IPAModule 
   from src.losses import compute_fape_loss, compute_confidence_loss, compute_angle_loss
   
   # Load config
   with open("config/default_config.yaml", "r") as f:
       config = yaml.safe_load(f)
       
   print("All components imported successfully.")
   print(f"Configuration loaded: {list(config.keys())}")
   ```

### 3.2. Implement the Main Model Class

The core of integration is implementing the `RNAFoldingModel` class that assembles all components:

1. **Create the model file** (`src/models/rna_folding_model.py`):

```python
"""RNA 3D structure prediction model integrating all components."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union

from src.models.embeddings import SequenceEmbedding, PositionalEncoding, RelativePositionalEncoding
from src.models.transformer_block import TransformerBlock
from src.models.ipa_module import IPAModule


class RNAFoldingModel(nn.Module):
    """
    Main RNA 3D folding model integrating all components:
    embeddings, transformer backbone, IPA placeholder, and prediction heads.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize RNA folding model.
        
        Args:
            config: Dictionary containing model parameters
        """
        super().__init__()
        
        # Extract key dimensions from config
        self.residue_dim = config['model']['residue_embed_dim']
        self.pair_dim = config['model']['pair_embed_dim']
        self.seq_embed_dim = config['model']['seq_embed_dim']
        self.num_blocks = config['model']['num_transformer_blocks']
        
        # Input embeddings
        self.sequence_embedding = SequenceEmbedding(
            num_embeddings=5,  # A, C, G, U, N/padding
            embedding_dim=self.seq_embed_dim
        )
        
        self.positional_encoding = PositionalEncoding(
            embed_dim=self.residue_dim,
            max_len=config['model'].get('max_seq_len', 500)
        )
        
        self.relative_pos_encoding = RelativePositionalEncoding(
            max_relative_position=config['model'].get('max_relative_position', 32),
            num_units=config['model'].get('rel_pos_dim', 32)
        )
        
        # Calculate input dimensions for projections
        # Residue features: sequence_embed + dihedral + entropy + accessibility + (conservation)
        residue_in_dim = self.seq_embed_dim + 4 + 1 + 1 
        if config['model'].get('use_conservation', True):
            residue_in_dim += 1
            
        # Pair features: pairing_probs + coupling_matrix + relative_pos
        pair_in_dim = 1 + 1 + config['model'].get('rel_pos_dim', 32)
        
        # Projection layers
        self.residue_projection = nn.Linear(residue_in_dim, self.residue_dim)
        self.pair_projection = nn.Linear(pair_in_dim, self.pair_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config['model']) for _ in range(self.num_blocks)
        ])
        
        # Structure module (IPA placeholder for V1)
        self.ipa_module = IPAModule(config['model'])
        
        # Prediction heads
        self.confidence_head = self._build_confidence_head()
        self.angle_prediction_head = self._build_angle_head()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _build_confidence_head(self):
        """Build the confidence prediction head."""
        return nn.Sequential(
            nn.Linear(self.residue_dim, self.residue_dim // 2),
            nn.ReLU(),
            nn.Linear(self.residue_dim // 2, 1)  # Single scalar confidence per residue
        )
    
    def _build_angle_head(self):
        """Build the angle prediction head."""
        return nn.Sequential(
            nn.Linear(self.residue_dim, self.residue_dim // 2),
            nn.ReLU(),
            nn.Linear(self.residue_dim // 2, 4)  # sin/cos of 2 angles (eta, theta)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RNA folding model.
        
        Args:
            batch: Dictionary containing:
                - sequence_int: (B, L) integer sequence representation
                - dihedral_features: (B, L, 4) dihedral angle sin/cos features
                - pairing_probs: (B, L, L) base pair probabilities
                - positional_entropy: (B, L) per-position entropy
                - accessibility: (B, L) accessibility scores
                - coupling_matrix: (B, L, L) evolutionary coupling matrix
                - conservation: (B, L) optional conservation scores
                - mask: (B, L) boolean mask for valid positions
                
        Returns:
            Dictionary containing:
                - pred_coords: (B, L, 3) predicted C1' coordinates
                - pred_confidence: (B, L) confidence scores per residue
                - pred_angles: (B, L, 4) predicted dihedral angles sin/cos
        """
        # 1. Extract inputs and get device
        sequence_int = batch['sequence_int']
        mask = batch['mask']
        device = sequence_int.device
        
        batch_size, seq_len = sequence_int.shape
        
        # 2. Generate embeddings
        # 2.1 Sequence embedding
        seq_embedding = self.sequence_embedding(sequence_int)
        
        # 2.2 Positional encoding
        pos_encoding = self.positional_encoding(seq_len).to(device)
        pos_encoding = pos_encoding.expand(batch_size, -1, -1)
        
        # 2.3 Relative positional encoding for pairs
        rel_pos_encoding = self.relative_pos_encoding(seq_len).to(device)
        
        # 3. Prepare residue features (concatenate all per-residue features)
        residue_features = [
            seq_embedding,                             # (B, L, seq_embed_dim)
            batch['dihedral_features'],                # (B, L, 4)
            batch['positional_entropy'].unsqueeze(-1), # (B, L, 1)
            batch['accessibility'].unsqueeze(-1)       # (B, L, 1)
        ]
        
        # Add conservation if available
        if 'conservation' in batch:
            residue_features.append(batch['conservation'].unsqueeze(-1))
        
        # Concatenate along feature dimension
        residue_features = torch.cat(residue_features, dim=-1)
        
        # 4. Prepare pair features
        pair_features = [
            batch['pairing_probs'].unsqueeze(-1),     # (B, L, L, 1)
            batch['coupling_matrix'].unsqueeze(-1),   # (B, L, L, 1)
            rel_pos_encoding.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, L, L, rel_pos_dim)
        ]
        
        # Concatenate along feature dimension
        pair_features = torch.cat(pair_features, dim=-1)
        
        # 5. Project to embedding dimensions
        residue_repr = self.residue_projection(residue_features)
        pair_repr = self.pair_projection(pair_features)
        
        # 6. Add positional encoding to residue representation
        residue_repr = residue_repr + pos_encoding
        
        # 7. Apply transformer blocks
        for block in self.transformer_blocks:
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
        
        # 8. Generate 3D coordinates using IPA module
        coords = self.ipa_module(residue_repr, pair_repr, mask)
        
        # 9. Generate auxiliary outputs
        confidence = self.confidence_head(residue_repr).squeeze(-1)
        angles = self.angle_prediction_head(residue_repr)
        
        # 10. Apply mask to outputs
        confidence = confidence * mask.float()
        angles = angles * mask.unsqueeze(-1).float()
        
        # 11. Return all outputs
        return {
            'pred_coords': coords,         # (B, L, 3) - 3D coordinates
            'pred_confidence': confidence, # (B, L) - confidence scores
            'pred_angles': angles          # (B, L, 4) - sin/cos of angles
        }
```

### 3.3. Implement Training Loop Integration

Create a basic training script that integrates the model with the data loader and loss functions:

```python
# scripts/train_v1.py
"""
Basic training script for RNA 3D folding model (V1).
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_loading import create_data_loader
from src.models.rna_folding_model import RNAFoldingModel
from src.losses import compute_fape_loss, compute_confidence_loss, compute_angle_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RNA 3D folding model')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    return parser.parse_args()


def compute_combined_loss(outputs, batch, loss_weights):
    """
    Compute combined loss from multiple loss components.
    
    Args:
        outputs: Dictionary of model outputs
        batch: Dictionary of input batch data
        loss_weights: Dictionary of loss weights
        
    Returns:
        total_loss: Combined weighted loss
        loss_dict: Dictionary of individual loss components
    """
    # Extract tensors
    pred_coords = outputs['pred_coords']
    pred_confidence = outputs['pred_confidence']
    pred_angles = outputs['pred_angles']
    
    true_coords = batch['coordinates']
    true_angles = batch['dihedral_features']
    mask = batch['mask']
    
    # Compute individual losses
    fape_loss = compute_fape_loss(pred_coords, true_coords, mask)
    confidence_loss = compute_confidence_loss(pred_confidence, pred_coords, true_coords, mask)
    angle_loss = compute_angle_loss(pred_angles, true_angles, mask)
    
    # Combine using weights
    total_loss = (
        loss_weights['fape'] * fape_loss +
        loss_weights['confidence'] * confidence_loss +
        loss_weights['angle'] * angle_loss
    )
    
    # Return both total loss and components
    loss_dict = {
        'total': total_loss.item(),
        'fape': fape_loss.item(),
        'confidence': confidence_loss.item(),
        'angle': angle_loss.item()
    }
    
    return total_loss, loss_dict


def train_epoch(model, dataloader, optimizer, loss_weights, device):
    """
    Train for one epoch.
    
    Args:
        model: RNA folding model
        dataloader: DataLoader for training data
        optimizer: PyTorch optimizer
        loss_weights: Loss component weights
        device: Device to use
        
    Returns:
        average_loss: Average loss for the epoch
        loss_components: Dictionary of average loss components
    """
    model.train()
    total_loss = 0.0
    loss_components = {'fape': 0.0, 'confidence': 0.0, 'angle': 0.0, 'total': 0.0}
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        loss, batch_losses = compute_combined_loss(outputs, batch, loss_weights)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for k, v in batch_losses.items():
            loss_components[k] += v
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def validate(model, dataloader, loss_weights, device):
    """
    Validate the model.
    
    Args:
        model: RNA folding model
        dataloader: DataLoader for validation data
        loss_weights: Loss component weights
        device: Device to use
        
    Returns:
        average_loss: Average validation loss
        loss_components: Dictionary of average loss components
    """
    model.eval()
    total_loss = 0.0
    loss_components = {'fape': 0.0, 'confidence': 0.0, 'angle': 0.0, 'total': 0.0}
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Compute loss
            loss, batch_losses = compute_combined_loss(outputs, batch, loss_weights)
            
            # Accumulate losses
            total_loss += loss.item()
            for k, v in batch_losses.items():
                loss_components[k] += v
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create data directory paths (NO hardcoded paths)
    sequences_csv_path = os.path.join(args.data_dir, 'train_sequences.csv')
    labels_csv_path = os.path.join(args.data_dir, 'train_labels.csv')
    features_dir = os.path.join(args.data_dir, 'processed')
    val_sequences_csv_path = os.path.join(args.data_dir, 'validation_sequences.csv')
    val_labels_csv_path = os.path.join(args.data_dir, 'validation_labels.csv')
    
    # Create data loaders
    train_loader = create_data_loader(
        sequences_csv_path=sequences_csv_path,
        labels_csv_path=labels_csv_path,
        features_dir=features_dir,
        batch_size=config['training']['batch_size'],
        temporal_cutoff=config['data'].get('temporal_cutoff'),
        num_workers=config['data'].get('num_workers', 4)
    )
    
    val_loader = create_data_loader(
        sequences_csv_path=val_sequences_csv_path,
        labels_csv_path=val_labels_csv_path,
        features_dir=features_dir,
        batch_size=config['training']['batch_size'],
        use_validation_set=True,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # Create model
    model = RNAFoldingModel(config).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training'].get('lr_factor', 0.5),
        patience=config['training'].get('lr_patience', 5),
        verbose=True
    )
    
    # Extract loss weights
    loss_weights = config['training']['loss_weights']
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['max_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['max_epochs']}")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, loss_weights, device
        )
        print(f"Train Loss: {train_loss:.4f} (FAPE: {train_components['fape']:.4f}, "
              f"Conf: {train_components['confidence']:.4f}, Angle: {train_components['angle']:.4f})")
        
        # Validate
        val_loss, val_components = validate(
            model, val_loader, loss_weights, device
        )
        print(f"Val Loss: {val_loss:.4f} (FAPE: {val_components['fape']:.4f}, "
              f"Conf: {val_components['confidence']:.4f}, Angle: {val_components['angle']:.4f})")
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Save latest model
        checkpoint_path = os.path.join(args.output_dir, 'latest_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, checkpoint_path)


if __name__ == '__main__':
    main()
```

### 3.4. Implement Basic Integration Test Script

Create a minimal test script to verify the flow from data → model → loss:

```python
# scripts/test_pipeline.py
"""
Basic integration test for RNA 3D folding pipeline.
Tests data loading → model forward pass → loss computation.
"""

import os
import argparse
import yaml
import torch
import numpy as np

from src.data_loading import create_data_loader
from src.models.rna_folding_model import RNAFoldingModel
from src.losses import compute_fape_loss, compute_confidence_loss, compute_angle_loss


def test_pipeline(config_path, data_dir, device_str='cuda'):
    """
    Test the full pipeline integration.
    
    Args:
        config_path: Path to configuration file
        data_dir: Path to data directory
        device_str: Device to use ('cuda' or 'cpu')
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up device
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create data directory paths
    sequences_csv_path = os.path.join(data_dir, 'train_sequences.csv')
    labels_csv_path = os.path.join(data_dir, 'train_labels.csv')
    features_dir = os.path.join(data_dir, 'processed')
    
    # Create small data loader for testing (batch size = 2)
    data_loader = create_data_loader(
        sequences_csv_path=sequences_csv_path,
        labels_csv_path=labels_csv_path,
        features_dir=features_dir,
        batch_size=2,  # Small batch size for testing
        num_workers=0  # No multiprocessing for testing
    )
    
    # Create model
    model = RNAFoldingModel(config).to(device)
    
    # Get a batch from the data loader
    print("Loading a batch of data...")
    for batch in data_loader:
        break
    
    # Print batch info
    print("\nBatch contents:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")
    
    # Record initial GPU memory (if available)
    if device.type == 'cuda':
        initial_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"\nInitial GPU memory usage: {initial_mem:.2f} MB")
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # Forward pass
    print("\nRunning model forward pass...")
    outputs = model(batch)
    
    # Print output info
    print("\nModel outputs:")
    for k, v in outputs.items():
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        
        # Check for NaN or inf values
        if torch.isnan(v).any():
            print(f"    WARNING: {k} contains NaN values!")
        if torch.isinf(v).any():
            print(f"    WARNING: {k} contains Inf values!")
        
        # Print some stats
        print(f"    Range: [{v.min().item():.4f}, {v.max().item():.4f}], "
              f"Mean: {v.mean().item():.4f}, Std: {v.std().item():.4f}")
    
    # Record peak GPU memory (if available)
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        current_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f"\nPeak GPU memory usage: {peak_mem:.2f} MB")
        print(f"Current GPU memory usage: {current_mem:.2f} MB")
        print(f"Memory increase during forward pass: {current_mem - initial_mem:.2f} MB")
    
    # Compute individual losses
    print("\nComputing losses...")
    
    # Extract loss weights from config
    loss_weights = config['training']['loss_weights']
    
    # Compute FAPE loss
    fape_loss = compute_fape_loss(outputs['pred_coords'], batch['coordinates'], batch['mask'])
    print(f"FAPE loss: {fape_loss.item():.4f}")
    
    # Compute confidence loss
    conf_loss = compute_confidence_loss(
        outputs['pred_confidence'], outputs['pred_coords'], 
        batch['coordinates'], batch['mask']
    )
    print(f"Confidence loss: {conf_loss.item():.4f}")
    
    # Compute angle loss
    angle_loss = compute_angle_loss(outputs['pred_angles'], batch['dihedral_features'], batch['mask'])
    print(f"Angle loss: {angle_loss.item():.4f}")
    
    # Compute total loss
    total_loss = (
        loss_weights['fape'] * fape_loss +
        loss_weights['confidence'] * conf_loss +
        loss_weights['angle'] * angle_loss
    )
    print(f"Total weighted loss: {total_loss.item():.4f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    total_loss.backward()
    
    # Check if gradients exist
    params_with_grad = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None:
            params_with_grad += 1
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"  {name}: grad_norm={grad_norm:.6f}")
            else:
                print(f"  {name}: grad_norm={grad_norm:.6f} (WARNING: zero gradient)")
    
    print(f"\nParameters with gradients: {params_with_grad}/{total_params}")
    
    print("\nPipeline test completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test RNA 3D folding pipeline')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    test_pipeline(args.config, args.data_dir, args.device)
```

## 4. Common Issues & Solutions

### 4.1. Shape Mismatch Issues

**Problem**: Tensor shape mismatches at component boundaries

**Solution**:
1. Add shape debugging code to track tensor dimensions:
```python
def debug_shapes(tensors_dict, stage_name):
    """Print shapes of tensors for debugging."""
    print(f"\n--- Shape debugging at {stage_name} ---")
    for name, tensor in tensors_dict.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name}: {tuple(tensor.shape)}")
```

2. In the model's forward method, add shape validation checks:
```python
# After embedding preparation
residue_shapes = {
    'seq_embedding': seq_embedding.shape,
    'pos_encoding': pos_encoding.shape,
    'residue_features': residue_features.shape,
    'residue_repr': residue_repr.shape
}
debug_shapes(residue_shapes, "residue embedding")

# After pair preparation
pair_shapes = {
    'pairing_probs': batch['pairing_probs'].shape,
    'coupling_matrix': batch['coupling_matrix'].shape,
    'rel_pos_encoding': rel_pos_encoding.shape,
    'pair_features': pair_features.shape,
    'pair_repr': pair_repr.shape
}
debug_shapes(pair_shapes, "pair embedding")
```

### 4.2. Device Inconsistency Issues

**Problem**: Tensors on different devices causing CUDA errors

**Solution**:
1. Implement a utility function for device management:
```python
def ensure_same_device(tensors_dict, target_device=None):
    """
    Ensure all tensors are on the same device.
    
    Args:
        tensors_dict: Dictionary of tensors
        target_device: Target device, if None, use device of first tensor
        
    Returns:
        Dictionary with all tensors on the same device
    """
    if not tensors_dict:
        return tensors_dict
    
    # Find a reference tensor to get the device
    reference_tensor = None
    for v in tensors_dict.values():
        if isinstance(v, torch.Tensor):
            reference_tensor = v
            break
    
    if reference_tensor is None:
        return tensors_dict  # No tensors found
    
    # Use specified device or the device of reference tensor
    device = target_device if target_device is not None else reference_tensor.device
    
    # Move all tensors to the target device
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in tensors_dict.items()
    }
```

2. Apply this function in the training loop:
```python
# In train_epoch function
for batch_idx, batch in enumerate(dataloader):
    # Move batch to device and ensure consistency
    batch = ensure_same_device(batch, device=device)
    
    # Forward pass
    outputs = model(batch)
```

### 4.3. Memory Overflow Issues

**Problem**: OOM errors during model training

**Solution**:
1. Implement gradient checkpointing to reduce memory usage:
```python
# Add to RNAFoldingModel.__init__
self.use_checkpointing = config['training'].get('use_checkpointing', False)

# Modify the transformer block forward loop in RNAFoldingModel.forward
if self.use_checkpointing and self.training:
    for block in self.transformer_blocks:
        def create_custom_forward(blk):
            def custom_forward(*inputs):
                return blk(*inputs)
            return custom_forward
        
        residue_repr, pair_repr = torch.utils.checkpoint.checkpoint(
            create_custom_forward(block),
            residue_repr, pair_repr, mask
        )
else:
    for block in self.transformer_blocks:
        residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
```

2. Add memory profiling to identify hotspots:
```python
def profile_memory(label):
    """Print memory usage statistics."""
    if torch.cuda.is_available():
        print(f"\n--- Memory usage at {label} ---")
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"Allocated: {allocated:.2f} MB")
        print(f"Max allocated: {max_allocated:.2f} MB")
        print(f"Reserved: {reserved:.2f} MB")
```

### 4.4. Mask Propagation Failures

**Problem**: Masks not properly applied throughout pipeline, leading to errors from padding

**Solution**:
1. Add explicit mask validation in the forward method:
```python
def validate_mask_application(tensor, mask, name, dim=-1):
    """Validate that masked positions are zero."""
    # Expand mask to match tensor dimensions
    if dim == -1:  # Feature dimension is last
        expanded_mask = mask.unsqueeze(-1).expand_as(tensor)
    else:
        raise ValueError(f"Unsupported mask dimension: {dim}")
    
    # Check masked positions
    inv_mask = ~expanded_mask
    masked_positions = tensor[inv_mask]
    
    # Check if masked positions are zero
    if not torch.all(masked_positions == 0):
        print(f"WARNING: {name} has non-zero values in masked positions!")
        print(f"  Number of non-zero masked values: {torch.sum(masked_positions != 0).item()}")
        print(f"  Range of masked values: [{masked_positions.min().item()}, {masked_positions.max().item()}]")
```

2. Apply this validation to model outputs:
```python
# After generating outputs
if mask is not None:
    validate_mask_application(outputs['pred_coords'], mask, 'pred_coords')
    validate_mask_application(outputs['pred_confidence'], mask, 'pred_confidence', dim=None)
    validate_mask_application(outputs['pred_angles'], mask, 'pred_angles')
```

### 4.5. Gradient Flow Problems

**Problem**: Some model parameters not receiving gradients

**Solution**:
1. Add gradient checking utility:
```python
def check_gradient_flow(model, threshold=1e-6):
    """
    Check gradient flow through the model.
    
    Args:
        model: PyTorch model
        threshold: Minimum gradient norm to consider "flowing"
        
    Returns:
        Dict mapping parameter names to gradient norms
    """
    gradient_dict = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                gradient_dict[name] = 0.0
                print(f"WARNING: {name} has no gradient")
            else:
                grad_norm = param.grad.norm().item()
                gradient_dict[name] = grad_norm
                
                if grad_norm < threshold:
                    print(f"WARNING: {name} has very small gradient: {grad_norm}")
    
    return gradient_dict
```

2. Apply in the training loop:
```python
# After loss.backward()
gradient_info = check_gradient_flow(model)

# Optionally log gradient norms
if batch_idx % 100 == 0:
    print("Gradient norms:")
    for name, norm in gradient_info.items():
        if norm > 0:  # Only print non-zero gradients
            print(f"  {name}: {norm:.6f}")
```

## 5. Integration Points

### 5.1. Data Loading to Model Integration

The first integration point is between data loading and model:

```python
# 1. Data loader provides properly batched dictionary of tensors
train_loader = create_data_loader(
    sequences_csv_path=sequences_csv_path,
    labels_csv_path=labels_csv_path,
    features_dir=features_dir,
    batch_size=config['training']['batch_size'],
    temporal_cutoff=config['data'].get('temporal_cutoff'),
    num_workers=config['data'].get('num_workers', 4)
)

# 2. Training loop extracts batch and provides to model
for batch in train_loader:
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # 3. Model processes batch through forward method
    outputs = model(batch)
```

**Key Interface Requirements**:
- Batch dictionary must contain all expected keys: `sequence_int`, `dihedral_features`, `pairing_probs`, `positional_entropy`, `accessibility`, `coupling_matrix`, `mask`
- Tensors must have correct shapes for batch_size and sequence_length
- All tensors must be moved to the appropriate device
- Masks must properly indicate valid vs. padding positions

### 5.2. Model to Loss Functions Integration

The second integration point is between model outputs and loss functions:

```python
# 1. Model produces dictionary of outputs
outputs = model(batch)  # Contains 'pred_coords', 'pred_confidence', 'pred_angles'

# 2. Extract ground truth from batch
true_coords = batch['coordinates']
true_angles = batch['dihedral_features']
mask = batch['mask']

# 3. Compute losses
fape_loss = compute_fape_loss(outputs['pred_coords'], true_coords, mask)
conf_loss = compute_confidence_loss(outputs['pred_confidence'], 
                                  outputs['pred_coords'], true_coords, mask)
angle_loss = compute_angle_loss(outputs['pred_angles'], true_angles, mask)

# 4. Combine losses
total_loss = (
    loss_weights['fape'] * fape_loss +
    loss_weights['confidence'] * conf_loss +
    loss_weights['angle'] * angle_loss
)

# 5. Backward pass
total_loss.backward()
```

**Key Interface Requirements**:
- Model outputs must match expected format and shapes for loss functions
- Loss functions must properly handle masks for variable sequence lengths
- Ensure the mask is applied consistently across all loss components
- All tensors must be on the same device

### 5.3. Training Loop to Validation Process Integration

Integration between training and validation:

```python
# 1. Create data loaders for training and validation
train_loader = create_data_loader(...)
val_loader = create_data_loader(...)

# 2. Training loop
for epoch in range(max_epochs):
    # Train
    train_loss, train_components = train_epoch(
        model, train_loader, optimizer, loss_weights, device
    )
    
    # Validate
    val_loss, val_components = validate(
        model, val_loader, loss_weights, device
    )
    
    # Update scheduler
    scheduler.step(val_loss)
    
    # Save checkpoint if best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(...)
```

**Key Requirements**:
- Consistent application of model.train() and model.eval()
- Proper gradient management (enable for training, disable for validation)
- Checkpoint saving with all necessary state

### 5.4. Checkpointing Integration

Integration for saving and loading model checkpoints:

```python
# Saving checkpoint
def save_checkpoint(model, optimizer, epoch, val_loss, config, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }
    torch.save(checkpoint, path)

# Loading checkpoint
def load_checkpoint(path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    
    model = RNAFoldingModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    
    return model, optimizer, epoch, val_loss, config
```

## 6. Validation Checkpoints

### 6.1. Component Functionality Tests

Before full integration, validate each component individually:

```python
def test_embeddings():
    """Test embedding components."""
    # Create dummy inputs
    sequence_int = torch.randint(0, 5, (2, 10))
    
    # Test sequence embedding
    seq_embedding = SequenceEmbedding(num_embeddings=5, embedding_dim=32)
    seq_output = seq_embedding(sequence_int)
    assert seq_output.shape == (2, 10, 32)
    
    # Test positional encoding
    pos_encoding = PositionalEncoding(embed_dim=64, max_len=100)
    pos_output = pos_encoding(10)
    assert pos_output.shape == (1, 10, 64)
    
    # Test relative positional encoding
    rel_pos_encoding = RelativePositionalEncoding(max_relative_position=32, num_units=32)
    rel_pos_output = rel_pos_encoding(10)
    assert rel_pos_output.shape == (10, 10, 32)
    
    print("Embedding tests passed!")

def test_transformer_block():
    """Test transformer block."""
    config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'num_attention_heads': 4,
        'dropout': 0.1
    }
    
    # Create inputs
    residue_repr = torch.rand(2, 10, 64)
    pair_repr = torch.rand(2, 10, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    
    # Test transformer block
    block = TransformerBlock(config)
    res_out, pair_out = block(residue_repr, pair_repr, mask)
    
    assert res_out.shape == (2, 10, 64)
    assert pair_out.shape == (2, 10, 10, 32)
    
    print("Transformer block test passed!")

def test_ipa_module():
    """Test IPA module."""
    config = {
        'residue_embed_dim': 64,
        'pair_embed_dim': 32,
        'ipa_dim': 32
    }
    
    # Create inputs
    residue_repr = torch.rand(2, 10, 64)
    pair_repr = torch.rand(2, 10, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    
    # Test IPA module
    ipa = IPAModule(config)
    coords = ipa(residue_repr, pair_repr, mask)
    
    assert coords.shape == (2, 10, 3)
    
    print("IPA module test passed!")
```

### 6.2. End-to-End Data Flow Verification

Verify data flow through the entire pipeline:

```python
def verify_data_flow(config, data_loader, device):
    """Verify data flow through the entire pipeline."""
    # Create model
    model = RNAFoldingModel(config).to(device)
    
    # Get a batch
    for batch in data_loader:
        break
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # Forward pass
    outputs = model(batch)
    
    # Verify outputs
    expected_keys = ['pred_coords', 'pred_confidence', 'pred_angles']
    for key in expected_keys:
        assert key in outputs, f"Missing output: {key}"
    
    # Check shapes
    batch_size, seq_len = batch['sequence_int'].shape
    assert outputs['pred_coords'].shape == (batch_size, seq_len, 3)
    assert outputs['pred_confidence'].shape == (batch_size, seq_len)
    assert outputs['pred_angles'].shape == (batch_size, seq_len, 4)
    
    # Verify mask application
    mask = batch['mask']
    for key in ['pred_coords', 'pred_angles']:
        # Expected shape differences between outputs and mask
        extra_dims = outputs[key].dim() - mask.dim()
        expanded_mask = mask
        for _ in range(extra_dims):
            expanded_mask = expanded_mask.unsqueeze(-1)
        # Expand to match output shape
        expanded_mask = expanded_mask.expand_as(outputs[key])
        # Check masked positions are zero
        masked_positions = outputs[key][~expanded_mask]
        assert torch.all(masked_positions == 0), f"Non-zero values in masked positions of {key}"
    
    # Simple mask check for confidence (1D output)
    masked_confidence = outputs['pred_confidence'][~mask]
    assert torch.all(masked_confidence == 0), "Non-zero values in masked positions of pred_confidence"
    
    print("Data flow verification passed!")
```

### 6.3. Memory Usage Validation

Monitor memory usage to catch inefficiencies:

```python
def validate_memory_usage(config, data_loader, device):
    """Validate memory usage during model execution."""
    if device.type != 'cuda':
        print("Memory validation requires CUDA device.")
        return
    
    # Create model
    model = RNAFoldingModel(config).to(device)
    
    # Get a batch
    for batch in data_loader:
        break
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Record initial memory
    initial_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # Forward pass
    outputs = model(batch)
    
    # Record forward memory
    forward_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    forward_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # Compute loss
    fape_loss = compute_fape_loss(outputs['pred_coords'], batch['coordinates'], batch['mask'])
    conf_loss = compute_confidence_loss(outputs['pred_confidence'], 
                                       outputs['pred_coords'], batch['coordinates'], batch['mask'])
    angle_loss = compute_angle_loss(outputs['pred_angles'], batch['dihedral_features'], batch['mask'])
    
    # Combine losses
    loss_weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}
    total_loss = (
        loss_weights['fape'] * fape_loss +
        loss_weights['confidence'] * conf_loss +
        loss_weights['angle'] * angle_loss
    )
    
    # Record loss memory
    loss_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    
    # Backward pass
    total_loss.backward()
    
    # Record backward memory
    backward_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    backward_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    # Print memory stats
    print(f"Initial GPU memory: {initial_mem:.2f} MB")
    print(f"After forward pass: {forward_mem:.2f} MB (+{forward_mem - initial_mem:.2f} MB)")
    print(f"Forward pass peak: {forward_peak:.2f} MB")
    print(f"After loss computation: {loss_mem:.2f} MB (+{loss_mem - forward_mem:.2f} MB)")
    print(f"After backward pass: {backward_mem:.2f} MB (+{backward_mem - loss_mem:.2f} MB)")
    print(f"Backward pass peak: {backward_peak:.2f} MB")
    print(f"Total memory usage: {backward_mem - initial_mem:.2f} MB")
    
    # Check if memory usage is reasonable
    batch_size, seq_len = batch['sequence_int'].shape
    print(f"Memory efficiency: {(backward_mem - initial_mem) / (batch_size * seq_len):.4f} MB per residue")
```

### 6.4. Gradient Flow Verification

Verify gradients flow properly through the model:

```python
def verify_gradient_flow(model, batch, device):
    """Verify gradients flow through the entire model."""
    # Move model to device
    model = model.to(device)
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()}
    
    # Forward pass
    outputs = model(batch)
    
    # Compute loss
    loss_weights = {'fape': 1.0, 'confidence': 0.1, 'angle': 0.5}
    fape_loss = compute_fape_loss(outputs['pred_coords'], batch['coordinates'], batch['mask'])
    conf_loss = compute_confidence_loss(outputs['pred_confidence'], 
                                       outputs['pred_coords'], batch['coordinates'], batch['mask'])
    angle_loss = compute_angle_loss(outputs['pred_angles'], batch['dihedral_features'], batch['mask'])
    
    total_loss = (
        loss_weights['fape'] * fape_loss +
        loss_weights['confidence'] * conf_loss +
        loss_weights['angle'] * angle_loss
    )
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    gradient_norms = {}
    num_params_with_grad = 0
    num_params_total = 0
    
    # Group parameters by module
    module_gradients = {}
    
    for name, param in model.named_parameters():
        num_params_total += 1
        
        # Extract module name
        if '.' in name:
            module_name = name.split('.')[0]
        else:
            module_name = 'root'
        
        if module_name not in module_gradients:
            module_gradients[module_name] = {
                'total': 0,
                'with_grad': 0,
                'norm_sum': 0.0
            }
        
        module_gradients[module_name]['total'] += 1
        
        if param.grad is not None:
            num_params_with_grad += 1
            module_gradients[module_name]['with_grad'] += 1
            
            grad_norm = param.grad.norm().item()
            gradient_norms[name] = grad_norm
            module_gradients[module_name]['norm_sum'] += grad_norm
    
    # Print gradient flow summary
    print(f"Parameters with gradients: {num_params_with_grad}/{num_params_total} ({num_params_with_grad/num_params_total*100:.1f}%)")
    
    print("\nGradient flow by module:")
    for module_name, stats in module_gradients.items():
        if stats['with_grad'] > 0:
            avg_norm = stats['norm_sum'] / stats['with_grad']
            coverage = stats['with_grad'] / stats['total'] * 100
            print(f"  {module_name}: {stats['with_grad']}/{stats['total']} params with grad ({coverage:.1f}%), "
                  f"avg_norm={avg_norm:.6f}")
        else:
            print(f"  {module_name}: 0/{stats['total']} params with grad (0.0%), NO GRADIENTS")
    
    # Check for gradient issues
    if num_params_with_grad < num_params_total:
        print("\nWARNING: Some parameters do not have gradients!")
        
        # Find modules with no gradients
        for module_name, stats in module_gradients.items():
            if stats['with_grad'] == 0:
                print(f"  Module with no gradients: {module_name}")
    
    return num_params_with_grad, num_params_total, gradient_norms
```

### 6.5. Loss Value Sanity Checks

Verify that loss values are reasonable:

```python
def validate_loss_values(outputs, batch):
    """Validate that loss values are reasonable."""
    # Compute individual losses
    fape_loss = compute_fape_loss(outputs['pred_coords'], batch['coordinates'], batch['mask'])
    conf_loss = compute_confidence_loss(outputs['pred_confidence'], 
                                      outputs['pred_coords'], batch['coordinates'], batch['mask'])
    angle_loss = compute_angle_loss(outputs['pred_angles'], batch['dihedral_features'], batch['mask'])
    
    # Check for NaN or inf
    if torch.isnan(fape_loss) or torch.isinf(fape_loss):
        print("WARNING: FAPE loss is NaN or Inf!")
    if torch.isnan(conf_loss) or torch.isinf(conf_loss):
        print("WARNING: Confidence loss is NaN or Inf!")
    if torch.isnan(angle_loss) or torch.isinf(angle_loss):
        print("WARNING: Angle loss is NaN or Inf!")
    
    # Check for negative values (losses should be non-negative)
    if fape_loss < 0:
        print(f"WARNING: FAPE loss is negative: {fape_loss.item()}")
    if conf_loss < 0:
        print(f"WARNING: Confidence loss is negative: {conf_loss.item()}")
    if angle_loss < 0:
        print(f"WARNING: Angle loss is negative: {angle_loss.item()}")
    
    # Check for reasonable magnitude
    # These thresholds are approximate and may need adjustment
    if fape_loss > 100:
        print(f"WARNING: FAPE loss is unusually high: {fape_loss.item()}")
    if conf_loss > 10:
        print(f"WARNING: Confidence loss is unusually high: {conf_loss.item()}")
    if angle_loss > 10:
        print(f"WARNING: Angle loss is unusually high: {angle_loss.item()}")
    
    print(f"FAPE loss: {fape_loss.item():.4f}")
    print(f"Confidence loss: {conf_loss.item():.4f}")
    print(f"Angle loss: {angle_loss.item():.4f}")
    
    # Return all losses
    return {
        'fape': fape_loss.item(),
        'confidence': conf_loss.item(),
        'angle': angle_loss.item()
    }
```

## 7. Complete Integration Example

Here's a comprehensive script showing the entire integration process with proper error handling and validation:

```python
#!/usr/bin/env python
# integration_test.py - Comprehensive RNA 3D folding model integration test

import os
import argparse
import yaml
import time
import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.data_loading import create_data_loader
from src.models.rna_folding_model import RNAFoldingModel
from src.losses import compute_fape_loss, compute_confidence_loss, compute_angle_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Comprehensive RNA 3D folding integration test')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Path to output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # These settings may affect performance but ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def debug_shapes(tensors_dict, stage_name="", verbose=False):
    """Print shapes of tensors for debugging."""
    if not verbose:
        return
        
    print(f"\n--- Shape debugging at {stage_name} ---")
    for name, tensor in tensors_dict.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{name}: {tuple(tensor.shape)}")


def check_tensor_stats(tensor, name, verbose=False):
    """Check statistics of a tensor for debugging."""
    if not verbose:
        return
        
    stats = {
        'shape': tuple(tensor.shape),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item()
    }
    
    print(f"\n--- Tensor stats for {name} ---")
    for stat_name, value in stats.items():
        print(f"{stat_name}: {value}")
    
    return stats


def profile_memory(label="", verbose=False):
    """Print memory usage statistics."""
    if not torch.cuda.is_available() or not verbose:
        return {}
        
    print(f"\n--- Memory usage at {label} ---")
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Allocated: {allocated:.2f} MB")
    print(f"Max allocated: {max_allocated:.2f} MB")
    print(f"Reserved: {reserved:.2f} MB")
    
    return {
        'allocated': allocated,
        'max_allocated': max_allocated,
        'reserved': reserved
    }


def check_gradient_flow(model, threshold=1e-6, verbose=False):
    """
    Check gradient flow through the model.
    
    Args:
        model: PyTorch model
        threshold: Minimum gradient norm to consider "flowing"
        verbose: Whether to print detailed information
        
    Returns:
        Dict mapping parameter names to gradient norms
    """
    if not verbose:
        return {}
        
    print("\n--- Gradient flow check ---")
    gradient_dict = {}
    
    # Group by module
    module_stats = {}
    
    for name, param in model.named_parameters():
        # Extract module name
        if '.' in name:
            module_name = name.split('.')[0]
        else:
            module_name = 'root'
            
        if module_name not in module_stats:
            module_stats[module_name] = {
                'total': 0,
                'with_grad': 0,
                'zero_grad': 0,
                'small_grad': 0,
                'norm_sum': 0.0
            }
            
        module_stats[module_name]['total'] += 1
        
        if param.requires_grad:
            if param.grad is None:
                gradient_dict[name] = 0.0
                print(f"WARNING: {name} has no gradient")
            else:
                grad_norm = param.grad.norm().item()
                gradient_dict[name] = grad_norm
                
                module_stats[module_name]['with_grad'] += 1
                module_stats[module_name]['norm_sum'] += grad_norm
                
                if grad_norm == 0:
                    module_stats[module_name]['zero_grad'] += 1
                    if verbose:
                        print(f"WARNING: {name} has zero gradient")
                elif grad_norm < threshold:
                    module_stats[module_name]['small_grad'] += 1
                    if verbose:
                        print(f"WARNING: {name} has very small gradient: {grad_norm}")
    
    # Print module summary
    print("\nGradient flow by module:")
    for module_name, stats in module_stats.items():
        if stats['with_grad'] > 0:
            avg_norm = stats['norm_sum'] / stats['with_grad']
            zero_pct = stats['zero_grad'] / stats['with_grad'] * 100 if stats['with_grad'] > 0 else 0
            small_pct = stats['small_grad'] / stats['with_grad'] * 100 if stats['with_grad'] > 0 else 0
            
            print(f"  {module_name}: {stats['with_grad']}/{stats['total']} params with grad, "
                  f"avg_norm={avg_norm:.6f}, zero_grad={zero_pct:.1f}%, small_grad={small_pct:.1f}%")
    
    return gradient_dict


def run_integration_test(args):
    """Run the full integration test."""
    print(f"Starting integration test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device_str = args.device
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Using CPU instead.")
        device_str = 'cpu'
    
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    # Reset CUDA memory stats if applicable
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    # Create data directory paths (NO hardcoded paths)
    sequences_csv_path = os.path.join(args.data_dir, 'train_sequences.csv')
    labels_csv_path = os.path.join(args.data_dir, 'train_labels.csv')
    features_dir = os.path.join(args.data_dir, 'processed')
    
    # Validate paths
    for path in [sequences_csv_path, labels_csv_path, features_dir]:
        if not os.path.exists(path):
            print(f"ERROR: Path does not exist: {path}")
            return False
    
    # STAGE 1: Data Loading
    print("\n--- STAGE 1: Data Loading ---")
    profile_memory("before data loading", args.verbose)
    
    try:
        # Create data loader
        data_loader = create_data_loader(
            sequences_csv_path=sequences_csv_path,
            labels_csv_path=labels_csv_path,
            features_dir=features_dir,
            batch_size=args.batch_size,
            num_workers=0  # No multiprocessing for testing
        )
        print(f"Data loader created successfully with batch size {args.batch_size}")
        
        # Get a batch from the data loader
        for batch in data_loader:
            break
        
        # Print batch info
        print("\nBatch contents:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
        
        batch_size, seq_len = batch['sequence_int'].shape
        print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
        
    except Exception as e:
        print(f"ERROR during data loading: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    profile_memory("after data loading", args.verbose)
    
    # STAGE 2: Model Creation
    print("\n--- STAGE 2: Model Creation ---")
    profile_memory("before model creation", args.verbose)
    
    try:
        # Create model
        model = RNAFoldingModel(config).to(device)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model created successfully with {trainable_params:,} trainable parameters "
              f"out of {total_params:,} total parameters")
        
        if args.verbose:
            print("\nModel structure:")
            print(model)
        
    except Exception as e:
        print(f"ERROR during model creation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    profile_memory("after model creation", args.verbose)
    
    # STAGE 3: Forward Pass
    print("\n--- STAGE 3: Forward Pass ---")
    profile_memory("before forward pass", args.verbose)
    
    try:
        # Set model to evaluation mode for first test
        model.eval()
        
        # Move batch to device
        batch_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
        
        # Forward pass
        with torch.no_grad():
            print("Running model forward pass...")
            outputs = model(batch_on_device)
        
        # Print output info
        print("\nModel outputs:")
        for k, v in outputs.items():
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
            
            # Check for NaN or inf values
            if torch.isnan(v).any():
                print(f"    WARNING: {k} contains NaN values!")
            if torch.isinf(v).any():
                print(f"    WARNING: {k} contains Inf values!")
            
            # Print some stats
            print(f"    Range: [{v.min().item():.4f}, {v.max().item():.4f}], "
                  f"Mean: {v.mean().item():.4f}, Std: {v.std().item():.4f}")
            
            # Detailed tensor stats
            check_tensor_stats(v, k, args.verbose)
            
    except Exception as e:
        print(f"ERROR during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    profile_memory("after forward pass", args.verbose)
    
    # STAGE 4: Loss Computation
    print("\n--- STAGE 4: Loss Computation ---")
    profile_memory("before loss computation", args.verbose)
    
    try:
        # Extract loss weights from config
        loss_weights = config['training']['loss_weights']
        print(f"Loss weights: {loss_weights}")
        
        # Compute losses
        with torch.no_grad():
            # Compute FAPE loss
            fape_loss = compute_fape_loss(
                outputs['pred_coords'], 
                batch_on_device['coordinates'], 
                batch_on_device['mask']
            )
            print(f"FAPE loss: {fape_loss.item():.4f}")
            
            # Compute confidence loss
            conf_loss = compute_confidence_loss(
                outputs['pred_confidence'], 
                outputs['pred_coords'], 
                batch_on_device['coordinates'], 
                batch_on_device['mask']
            )
            print(f"Confidence loss: {conf_loss.item():.4f}")
            
            # Compute angle loss
            angle_loss = compute_angle_loss(
                outputs['pred_angles'], 
                batch_on_device['dihedral_features'], 
                batch_on_device['mask']
            )
            print(f"Angle loss: {angle_loss.item():.4f}")
            
            # Compute total loss
            total_loss = (
                loss_weights['fape'] * fape_loss +
                loss_weights['confidence'] * conf_loss +
                loss_weights['angle'] * angle_loss
            )
            print(f"Total weighted loss: {total_loss.item():.4f}")
            
    except Exception as e:
        print(f"ERROR during loss computation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    profile_memory("after loss computation", args.verbose)
    
    # STAGE 5: Training Mode and Gradient Flow
    print("\n--- STAGE 5: Training Mode and Gradient Flow ---")
    profile_memory("before gradient flow test", args.verbose)
    
    try:
        # Switch to training mode
        model.train()
        
        # Clear gradients
        model.zero_grad()
        
        # Forward pass (in training mode)
        outputs = model(batch_on_device)
        
        # Compute losses
        fape_loss = compute_fape_loss(
            outputs['pred_coords'], 
            batch_on_device['coordinates'], 
            batch_on_device['mask']
        )
        
        conf_loss = compute_confidence_loss(
            outputs['pred_confidence'], 
            outputs['pred_coords'], 
            batch_on_device['coordinates'], 
            batch_on_device['mask']
        )
        
        angle_loss = compute_angle_loss(
            outputs['pred_angles'], 
            batch_on_device['dihedral_features'], 
            batch_on_device['mask']
        )
        
        # Compute total loss
        total_loss = (
            loss_weights['fape'] * fape_loss +
            loss_weights['confidence'] * conf_loss +
            loss_weights['angle'] * angle_loss
        )
        
        print(f"Loss in training mode: {total_loss.item():.4f}")
        
        # Backward pass
        print("Running backward pass...")
        total_loss.backward()
        
        # Check gradient flow
        gradient_info = check_gradient_flow(model, verbose=True)
        
        # Check overall gradient health
        params_with_grad = sum(1 for param in model.parameters() 
                              if param.grad is not None and param.requires_grad)
        total_trainable = sum(1 for param in model.parameters() if param.requires_grad)
        
        print(f"\nParameters with gradients: {params_with_grad}/{total_trainable} "
              f"({params_with_grad/total_trainable*100:.1f}%)")
        
        if params_with_grad < total_trainable:
            print("WARNING: Some parameters are not receiving gradients!")
        
    except Exception as e:
        print(f"ERROR during gradient flow test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    profile_memory("after gradient flow test", args.verbose)
    
    # STAGE 6: Optimizer Step
    print("\n--- STAGE 6: Optimizer Step ---")
    profile_memory("before optimizer step", args.verbose)
    
    try:
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5)
        )
        
        # Store parameter values before update
        if args.verbose:
            params_before = {name: param.clone().detach().cpu() 
                            for name, param in model.named_parameters() 
                            if param.requires_grad}
        
        # Step optimizer
        optimizer.step()
        
        # Check if parameters changed
        if args.verbose:
            print("\nParameter changes after optimization step:")
            params_changed = 0
            params_unchanged = 0
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Check if parameter changed
                    before = params_before[name]
                    after = param.detach().cpu()
                    changed = not torch.allclose(before, after)
                    
                    if changed:
                        params_changed += 1
                        if args.verbose:
                            change_norm = (after - before).norm().item()
                            print(f"  {name}: changed, change_norm={change_norm:.6f}")
                    else:
                        params_unchanged += 1
            
            print(f"Parameters changed: {params_changed}/{params_changed + params_unchanged} "
                  f"({params_changed/(params_changed + params_unchanged)*100:.1f}%)")
            
            if params_unchanged > 0:
                print(f"WARNING: {params_unchanged} parameters did not change after optimizer step!")
        
    except Exception as e:
        print(f"ERROR during optimizer step: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    profile_memory("after optimizer step", args.verbose)
    
    # STAGE 7: Mini Training Loop
    print("\n--- STAGE 7: Mini Training Loop ---")
    profile_memory("before mini training loop", args.verbose)
    
    try:
        # Reset model and optimizer
        model = RNAFoldingModel(config).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-5)
        )
        
        # Set to training mode
        model.train()
        
        # Run a few steps of training
        num_mini_epochs = 3
        print(f"Running {num_mini_epochs} mini-epochs...")
        
        for epoch in range(num_mini_epochs):
            epoch_loss = 0.0
            
            # Create a mini data loader with just a few batches
            mini_loader = []
            for batch_idx, batch in enumerate(data_loader):
                mini_loader.append(batch)
                if batch_idx >= 2:  # Just use 3 batches
                    break
            
            for batch_idx, batch in enumerate(mini_loader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch)
                
                # Compute loss
                fape_loss = compute_fape_loss(
                    outputs['pred_coords'], batch['coordinates'], batch['mask']
                )
                conf_loss = compute_confidence_loss(
                    outputs['pred_confidence'], outputs['pred_coords'], 
                    batch['coordinates'], batch['mask']
                )
                angle_loss = compute_angle_loss(
                    outputs['pred_angles'], batch['dihedral_features'], batch['mask']
                )
                
                # Combine losses
                total_loss = (
                    loss_weights['fape'] * fape_loss +
                    loss_weights['confidence'] * conf_loss +
                    loss_weights['angle'] * angle_loss
                )
                
                # Backward pass
                total_loss.backward()
                
                # Optimizer step
                optimizer.step()
                
                # Accumulate loss
                epoch_loss += total_loss.item()
                
                print(f"Epoch {epoch+1}/{num_mini_epochs}, Batch {batch_idx+1}/{len(mini_loader)}, "
                      f"Loss: {total_loss.item():.4f}")
            
            # Calculate epoch average
            epoch_loss /= len(mini_loader)
            print(f"Epoch {epoch+1}/{num_mini_epochs} complete, Average loss: {epoch_loss:.4f}")
        
    except Exception as e:
        print(f"ERROR during mini training loop: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    profile_memory("after mini training loop", args.verbose)
    
    # STAGE 8: Checkpoint Saving and Loading
    print("\n--- STAGE 8: Checkpoint Saving and Loading ---")
    profile_memory("before checkpoint test", args.verbose)
    
    try:
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, 'test_checkpoint.pt')
        torch.save({
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'config': config
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        new_model = RNAFoldingModel(config).to(device)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        new_optimizer = optim.Adam(new_model.parameters())
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded successfully")
        
        # Verify loaded model produces same outputs
        new_model.eval()
        with torch.no_grad():
            original_outputs = model(batch)
            new_outputs = new_model(batch)
            
            # Compare outputs
            for key in original_outputs:
                orig = original_outputs[key]
                new = new_outputs[key]
                
                difference = torch.max(torch.abs(orig - new)).item()
                print(f"Max difference in {key}: {difference}")
                
                if difference > 1e-5:
                    print(f"WARNING: Significant difference in {key} after checkpoint loading!")
                else:
                    print(f"Checkpoint verification for {key}: PASSED")
        
    except Exception as e:
        print(f"ERROR during checkpoint test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    profile_memory("after checkpoint test", args.verbose)
    
    # Calculate total test time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\n--- Integration Test Summary ---")
    print(f"Test completed successfully!")
    print(f"Total test time: {int(minutes)}m {seconds:.2f}s")
    
    return True


if __name__ == '__main__':
    args = parse_args()
    success = run_integration_test(args)
    sys.exit(0 if success else 1)
```

## Conclusion

This Model Integration Workflow provides a comprehensive guide for unifying the separately implemented components of the RNA 3D folding pipeline. The workflow covers data flow integration, memory optimization, proper masking, gradient flow verification, and complete model assembly. By following this systematic approach, developers can ensure that all components work together correctly, producing a functional end-to-end pipeline capable of training and evaluation.

The integration process focuses on ensuring path parameterization is maintained throughout, proper device management is implemented, and masks are correctly propagated through all components. This workflow enables the team to move from individual component development to a cohesive system ready for training, validation, and eventually, Kaggle submission.

Next steps after successful integration include:
1. Running the full training pipeline with real data
2. Implementing advanced validation and evaluation metrics
3. Optimizing hyperparameters
4. Implementing inference and submission pipelines

For debugging integration issues, refer to the "Common Issues & Solutions" section, and for validating the integration, follow the steps in the "Validation Checkpoints" section.
