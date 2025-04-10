# Data Loading Examples

This document provides practical examples of using the data loading components for the RNA 3D folding project. It demonstrates how to load data, create batches, and integrate with the rest of the pipeline.

## Basic Usage Pattern

Here's the standard pattern for creating a data loader for training:

```python
from src.data_loading import create_data_loader

# Create data loader with all parameters explicitly passed (no hardcoded paths)
data_loader = create_data_loader(
    sequences_csv_path="/path/to/train_sequences.csv",
    labels_csv_path="/path/to/train_labels.csv",
    features_dir="/path/to/processed/features",
    batch_size=16,
    temporal_cutoff="2022-05-27",  # Filter sequences by publication date
    num_workers=4,
    shuffle=True
)

# Iterate through batches
for batch in data_loader:
    # Process batch
    # ...
```

For validation data:

```python
validation_loader = create_data_loader(
    sequences_csv_path="/path/to/validation_sequences.csv",
    labels_csv_path="/path/to/validation_labels.csv",
    features_dir="/path/to/processed/features",
    batch_size=16,
    use_validation_set=True,  # Ignores temporal_cutoff
    num_workers=4,
    shuffle=False
)
```

For test/inference (without labels):

```python
test_loader = create_data_loader(
    sequences_csv_path="/path/to/test_sequences.csv",
    labels_csv_path=None,  # No labels for test data
    features_dir="/path/to/processed/features",
    batch_size=1,  # Often use batch_size=1 for inference
    num_workers=4,
    shuffle=False
)
```

## Using Configuration

Integrate with the configuration system:

```python
import yaml

# Load configuration
with open("config/default_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create data loader using config
data_loader = create_data_loader(
    sequences_csv_path=config["data"]["sequences_csv_path"],
    labels_csv_path=config["data"]["labels_csv_path"],
    features_dir=config["data"]["features_dir"],
    batch_size=config["data"]["batch_size"],
    temporal_cutoff=config["data"].get("temporal_cutoff"),
    num_workers=config["data"].get("num_workers", 4),
    shuffle=config["data"].get("shuffle", True)
)
```

## Working with Batches

A batch contains the following tensors:

```python
# Get a batch from data loader
batch = next(iter(data_loader))

# Typical batch structure
print(f"Batch keys: {batch.keys()}")
# Output: Batch keys: ['target_ids', 'lengths', 'sequence_int', 'dihedral_features', 
#                      'pairing_probs', 'positional_entropy', 'coupling_matrix', 
#                      'accessibility', 'coordinates', 'mask']

# Target IDs (list of strings)
print(f"Target IDs: {batch['target_ids']}")
# Output: Target IDs: ['1A9N_R', '1CSL_A', ...]

# Sequence lengths (tensor of shape [batch_size])
print(f"Sequence lengths: {batch['lengths']}")
# Output: Sequence lengths: tensor([76, 28, 45, ...])

# Boolean mask for valid positions (tensor of shape [batch_size, max_seq_len])
print(f"Mask shape: {batch['mask'].shape}")
# Output: Mask shape: torch.Size([16, 76])
```

### Moving to Device

Move the batch to the appropriate device:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move all tensors to device
batch_on_device = {
    k: v.to(device) if isinstance(v, torch.Tensor) else v
    for k, v in batch.items()
}
```

## Working with Features

### Examining feature shapes

```python
# Sequence representation (integer encoded)
print(f"sequence_int: {batch['sequence_int'].shape}")
# Output: sequence_int: torch.Size([16, 76])

# Dihedral features (sin/cos of angles)
print(f"dihedral_features: {batch['dihedral_features'].shape}")
# Output: dihedral_features: torch.Size([16, 76, 4])

# Base pairing probability matrix
print(f"pairing_probs: {batch['pairing_probs'].shape}")
# Output: pairing_probs: torch.Size([16, 76, 76])

# Evolutionary coupling matrix
print(f"coupling_matrix: {batch['coupling_matrix'].shape}")
# Output: coupling_matrix: torch.Size([16, 76, 76])

# Ground truth coordinates for training
print(f"coordinates: {batch['coordinates'].shape}")
# Output: coordinates: torch.Size([16, 76, 3])
```

## Handling Variable Length Sequences

The dataloader handles variable-length sequences by padding to the maximum length in the batch and providing a boolean mask:

```python
# Check the actual sequence lengths
for i, length in enumerate(batch['lengths'][:3]):
    # Count valid positions in mask
    valid_positions = batch['mask'][i].sum().item()
    print(f"Sequence {i}: length={length.item()}, valid mask positions={valid_positions}")
    # Output example:
    # Sequence 0: length=76, valid mask positions=76
    # Sequence 1: length=28, valid mask positions=28 
    # Sequence 2: length=45, valid mask positions=45
    
    # Verify that padding values are zeros
    if length.item() < batch['sequence_int'].shape[1]:
        padding_sum = batch['sequence_int'][i, length:].sum().item()
        print(f"  Padding sum: {padding_sum}")  # Should be 0
```

## Using Masks in Attention

Here's how to use the mask in a transformer block:

```python
def apply_attention_with_mask(query, key, value, mask):
    """Apply attention with proper masking."""
    # Convert boolean mask to attention mask
    # mask shape: [batch_size, seq_len]
    # key_padding_mask expects False for valid positions in nn.MultiheadAttention
    key_padding_mask = ~mask
    
    attention = nn.MultiheadAttention(
        embed_dim=query.size(-1),
        num_heads=8,
        batch_first=True
    )
    
    # Apply attention with mask
    output, _ = attention(
        query=query,
        key=key,
        value=value,
        key_padding_mask=key_padding_mask
    )
    
    return output
```

## Using Masks in Loss Functions

When computing losses, use the mask to ignore padded positions:

```python
def compute_masked_loss(predictions, targets, mask):
    """Compute loss considering only valid positions."""
    # Calculate squared error at each position
    squared_error = (predictions - targets) ** 2
    
    # Apply mask
    masked_error = squared_error * mask.float().unsqueeze(-1)  # Add feature dimension
    
    # Sum error and divide by number of valid positions
    total_valid = mask.sum()
    loss = masked_error.sum() / (total_valid + 1e-10)
    
    return loss
```

## Direct Dataset and DataLoader Creation

If you need more control, you can create the dataset and dataloader separately:

```python
from src.data_loading import RNADataset, collate_fn
from torch.utils.data import DataLoader

# Create dataset
dataset = RNADataset(
    sequences_csv_path="/path/to/sequences.csv",
    labels_csv_path="/path/to/labels.csv",
    features_dir="/path/to/features",
    temporal_cutoff="2022-05-27"
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,  # Important! Use our custom collate_fn
    pin_memory=True         # Speeds up CPU to GPU transfers
)
```

## Multi-GPU Training Integration

For distributed training (future implementation):

```python
from torch.utils.data.distributed import DistributedSampler

def create_distributed_dataloader(config, is_training=True):
    """Create dataloader for distributed training."""
    # Create dataset
    dataset = RNADataset(
        sequences_csv_path=config["data"]["sequences_csv_path"],
        labels_csv_path=config["data"]["labels_csv_path"],
        features_dir=config["data"]["features_dir"],
        temporal_cutoff=config["data"].get("temporal_cutoff") if is_training else None,
        use_validation_set=not is_training
    )
    
    # Create distributed sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=is_training
    )
    
    # Create dataloader with sampler
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,  # Sampler handles shuffling
        sampler=sampler,
        num_workers=config["data"].get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader, sampler
```

## Feature Exploration

Exploring the loaded features for a specific sequence:

```python
# Get a specific item from the dataset
sample = dataset[0]

# Print target ID and sequence length
print(f"Target ID: {sample['target_id']}")
print(f"Sequence length: {sample['length']}")

# Examine dihedral features (sin/cos representation)
dihedral = sample['dihedral_features']
print(f"Dihedral features shape: {dihedral.shape}")
print(f"First few dihedral values:\n{dihedral[:5]}")

# Examine pairing probabilities
pairing = sample['pairing_probs']
print(f"Pairing probabilities shape: {pairing.shape}")
print(f"Pairing probability matrix sparsity: {(pairing > 0.5).float().mean().item():.4f}")

# Examine evolutionary coupling
coupling = sample['coupling_matrix']
print(f"Coupling matrix shape: {coupling.shape}")
print(f"Coupling matrix mean value: {coupling.mean().item():.6f}")

# For training data, check coordinates
if 'coordinates' in sample:
    coords = sample['coordinates']
    print(f"C1' coordinates shape: {coords.shape}")
    print(f"First few coordinates:\n{coords[:3]}")
```

## Common Pitfalls and Solutions

### 1. Memory Issues with Large Batches

For very long sequences, reduce batch size or use gradient accumulation:

```python
# Create dataloader with smaller batch size
dataloader = create_data_loader(
    sequences_csv_path=config["data"]["sequences_csv_path"],
    labels_csv_path=config["data"]["labels_csv_path"],
    features_dir=config["data"]["features_dir"],
    batch_size=1,  # Small batch size for long sequences
    # ...
)

# Use gradient accumulation in training loop
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(batch)
    
    # Calculate loss and scale by accumulation steps
    loss = compute_loss(outputs, batch) / ACCUMULATION_STEPS
    
    # Backward pass
    loss.backward()
    
    # Update weights every ACCUMULATION_STEPS
    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Missing Feature Files

Ensure all required feature files are present:

```python
import os

def verify_features_exist(target_ids, features_dir):
    """Check if all required feature files exist."""
    missing_files = []
    
    for target_id in target_ids:
        # Check thermodynamic features (required)
        thermo_path = os.path.join(
            features_dir, 
            "thermo_features", 
            f"{target_id}_thermo_features.npz"
        )
        
        if not os.path.exists(thermo_path):
            missing_files.append(thermo_path)
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} required feature files, including: "
            f"{missing_files[:5]}"
        )
    
    print(f"All required feature files found for {len(target_ids)} targets")
```

### 3. Handling Features of Different Scales

Normalize features if necessary:

```python
def normalize_features(batch):
    """Normalize features to have zero mean and unit variance."""
    # Get the mask for valid positions
    mask = batch['mask']
    
    # Normalize positional_entropy (example)
    if 'positional_entropy' in batch:
        # Calculate mean and std over valid positions
        mean = (batch['positional_entropy'] * mask.float()).sum() / mask.sum()
        std = torch.sqrt(
            ((batch['positional_entropy'] - mean) ** 2 * mask.float()).sum() / mask.sum()
        )
        
        # Normalize
        batch['positional_entropy'] = (batch['positional_entropy'] - mean) / (std + 1e-8)
    
    return batch
```

## Complete Training Loop Example

Here's a complete example of a training loop using the data loading components:

```python
def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Calculate loss
        loss = compute_loss(outputs, batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on dataloader."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate loss
            loss = compute_loss(outputs, batch)
            
            # Track loss
            total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(dataloader)


def main():
    # Load configuration
    config = load_config("config/default_config.yaml")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders
    train_loader = create_data_loader(
        sequences_csv_path=config["data"]["sequences_csv_path"],
        labels_csv_path=config["data"]["labels_csv_path"],
        features_dir=config["data"]["features_dir"],
        batch_size=config["data"]["batch_size"],
        temporal_cutoff=config["data"].get("temporal_cutoff"),
        num_workers=config["data"].get("num_workers", 4)
    )
    
    val_loader = create_data_loader(
        sequences_csv_path=config["data"]["val_sequences_csv_path"],
        labels_csv_path=config["data"]["val_labels_csv_path"],
        features_dir=config["data"]["features_dir"],
        batch_size=config["data"]["batch_size"],
        use_validation_set=True,
        num_workers=config["data"].get("num_workers", 4),
        shuffle=False
    )
    
    # Create model, optimizer, etc.
    model = create_model(config).to(device)
    optimizer = create_optimizer(model, config)
    
    # Training loop
    for epoch in range(config["training"]["epochs"]):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, device)
        
        # Print progress
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss)
```

## Conclusion

This guide demonstrated the key patterns for using the data loading components in the RNA 3D folding project. The examples cover most common use cases from basic initialization to integration with the training loop. Remember to always follow these key principles:

1. **No hardcoded paths**: Always pass paths as arguments
2. **Use the mask**: Apply masks for padding in attention and loss calculations
3. **Move to device**: Move all tensors to the appropriate device before use
4. **Batch processing**: Process data in batches for efficiency
5. **Feature verification**: Ensure all required features are available
