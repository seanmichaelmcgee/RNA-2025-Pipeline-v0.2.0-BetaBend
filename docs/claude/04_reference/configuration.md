# Configuration Reference Guide

This document outlines the configuration system for the RNA 3D folding project, covering file structure, key parameters, environment handling, Docker integration, and multi-GPU readiness.

## Configuration Philosophy

The project follows these configuration principles:

1. **Separation of Code and Configuration**: All adjustable parameters should be defined in configuration files, not hardcoded in source code
2. **Environment Adaptability**: Configuration should adapt to different environments (local, Docker, Kaggle) without code changes
3. **Reproducibility**: Configuration must support exact reproduction of experiments
4. **Scalability**: Configuration structure should anticipate future scaling needs (multi-GPU)
5. **Validation**: Critical parameters should be validated to prevent runtime errors

## Configuration File Structure

### Primary Configuration File

The project uses YAML as the configuration format (`config/default_config.yaml`):

```yaml
# RNA 3D Folding Model Configuration

# ===== General Settings =====
experiment_name: "rna3d_v1"
seed: 42
device: "cuda"  # Options: "cuda", "cpu", or specific "cuda:0"
precision: "float32"  # Options: "float32", "float16" (mixed precision)

# ===== Data Settings =====
data:
  sequences_csv_path: null  # Set via CLI/script
  labels_csv_path: null     # Set via CLI/script
  features_dir: null        # Set via CLI/script
  temporal_cutoff: "2022-05-27"  # Default cutoff date
  batch_size: 4
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2

# ===== Model Architecture =====
model:
  # Dimensions
  residue_embed_dim: 128    # Reduced for V1/development
  pair_embed_dim: 64        # Reduced for V1/development
  seq_embed_dim: 32
  
  # Transformer parameters
  num_transformer_blocks: 4  # Reduced for V1/development
  num_attention_heads: 4     # Reduced for V1/development
  ffn_dim: 256              # Reduced for V1/development
  dropout: 0.1
  
  # IPA module parameters (placeholder values for V1)
  num_ipa_iterations: 1
  ipa_dim: 16
  
  # Output parameters
  num_predictions: 1        # V1: single prediction; Final: 5 predictions

# ===== Training Settings =====
training:
  optimizer: "adamw"  # Options: "adam", "adamw", "sgd"
  learning_rate: 0.0005
  weight_decay: 0.01
  gradient_clip_val: 1.0
  lr_scheduler: "cosine"  # Options: "cosine", "step", "plateau", "none"
  warmup_steps: 100
  max_epochs: 100
  
  # Early stopping
  early_stopping: true
  patience: 10
  min_delta: 0.0001
  
  # Checkpointing
  checkpoint_dir: null  # Set via CLI/script
  save_top_k: 3
  save_last: true
  
  # Validation
  val_check_interval: 1.0  # Run validation every epoch
  
  # Multi-GPU (parameters for future implementation)
  strategy: "dp"  # Options: "dp" (DataParallel), "ddp" (DistributedDataParallel)
  sync_batchnorm: false
  find_unused_parameters: false

# ===== Loss Function Weights =====
loss_weights:
  fape: 1.0            # Coordinate loss weight
  confidence: 0.1      # Confidence prediction loss weight
  angle: 0.5           # Auxiliary angle prediction loss weight

# ===== Inference/Prediction =====
inference:
  temperature: 1.0      # Sampling temperature for diversity (future use)
  num_samples: 5        # V1: Use 1; Final: Use 5
  output_dir: null      # Set via CLI/script
```

### Environment-Specific Overrides

Create environment-specific configurations that extend the default:

```yaml
# config/dev_config.yaml - Local development overrides
device: "cuda:0"  # Specific GPU
model:
  residue_embed_dim: 64  # Further reduced for faster dev iterations
  pair_embed_dim: 32
  num_transformer_blocks: 2
```

```yaml
# config/kaggle_config.yaml - Kaggle-specific settings
device: "cuda"
data:
  sequences_csv_path: "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv" 
  features_dir: "/kaggle/input/rna-features/processed"
inference:
  output_dir: "/kaggle/working"
```

## Loading Configuration

### Configuration Loading Pattern

```python
import os
import yaml
import argparse
from typing import Dict, Any


def load_config(config_path: str, override_args: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with optional CLI overrides.
    
    Args:
        config_path: Path to YAML config file
        override_args: Dictionary of values to override from command line
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Apply CLI overrides
    if override_args:
        _update_config_with_overrides(config, override_args)
    
    # Validate critical parameters
    _validate_config(config)
    
    return config


def _update_config_with_overrides(config: Dict[str, Any], override_args: Dict[str, Any]) -> None:
    """Update config with CLI argument overrides using dot notation."""
    for key, value in override_args.items():
        if value is None:
            continue
            
        # Handle nested keys with dot notation (e.g., "model.dropout")
        if "." in key:
            parts = key.split(".")
            curr = config
            for part in parts[:-1]:
                if part not in curr:
                    curr[part] = {}
                curr = curr[part]
            curr[parts[-1]] = value
        else:
            config[key] = value


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate critical configuration parameters."""
    # Example validation
    if "model" in config:
        model_config = config["model"]
        if "num_attention_heads" in model_config and "residue_embed_dim" in model_config:
            if model_config["residue_embed_dim"] % model_config["num_attention_heads"] != 0:
                raise ValueError(
                    f"residue_embed_dim ({model_config['residue_embed_dim']}) must be divisible by "
                    f"num_attention_heads ({model_config['num_attention_heads']})"
                )
```

### Command-Line Integration
```python
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RNA 3D Structure Prediction")
    
    # Core arguments
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to config file")
    
    # Critical path arguments (always required, no defaults)
    parser.add_argument("--data.sequences_csv_path", type=str, required=True,
                        help="Path to sequences CSV file")
    parser.add_argument("--data.labels_csv_path", type=str,
                        help="Path to labels CSV file (required for training)")
    parser.add_argument("--data.features_dir", type=str, required=True,
                        help="Path to directory containing feature files")
    
    # Common overrides - specific options exposed for CLI convenience
    parser.add_argument("--model.residue_embed_dim", type=int,
                        help="Residue embedding dimension")
    parser.add_argument("--model.num_transformer_blocks", type=int,
                        help="Number of transformer blocks")
    parser.add_argument("--training.learning_rate", type=float,
                        help="Learning rate")
    parser.add_argument("--training.batch_size", type=int,
                        help="Batch size")
    
    # Add more parameters as needed...
    
    return parser.parse_args()
```

### Usage in Scripts

```python
# scripts/train.py
import os
import torch
from src.data_loading import RNADataset, collate_fn
from src.models.rna_folding_model import RNAFoldingModel

def main():
    # Parse args and load config
    args = parse_args()
    config = load_config(args.config, vars(args))
    
    # Set seed for reproducibility
    torch.manual_seed(config["seed"])
    
    # Set device
    device_str = config["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    
    # Create dataset using paths from config
    dataset = RNADataset(
        sequences_csv_path=config["data"]["sequences_csv_path"],
        labels_csv_path=config["data"]["labels_csv_path"],
        features_dir=config["data"]["features_dir"],
        temporal_cutoff=config["data"]["temporal_cutoff"]
    )
    
    # Create model using config parameters
    model = RNAFoldingModel(config["model"]).to(device)
    
    # Training loop...
```

## Docker Integration

### Environment Variables in Docker

Use Docker environment variables to adapt configuration at runtime:

```yaml
# In default_config.yaml
data:
  features_dir: "${FEATURES_DIR:-/app/data/processed}"  # Use env var with default
```

```python
# In configuration loading code
def _process_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Process environment variables in configuration strings."""
    import re
    import os
    
    def _replace_env_var(match):
        env_var = match.group(1)
        default = match.group(2) if match.group(2) else None
        return os.environ.get(env_var, default)
    
    def _process_value(value):
        if isinstance(value, str):
            # Match ${VAR:-default} pattern
            pattern = r'\${([^}:]+)(?::-(.*?))?}'
            return re.sub(pattern, _replace_env_var, value)
        return value
    
    # Process the config recursively
    def _process_dict(d):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = _process_dict(v)
            else:
                result[k] = _process_value(v)
        return result
    
    return _process_dict(config)
```

### Docker Run Command
```bash
docker run --gpus all -v $(pwd)/data:/app/data \
  -e FEATURES_DIR=/app/data/processed \
  -e MODEL_DIM=128 \
  rna-3d:dev python scripts/train.py \
  --config config/default_config.yaml \
  --data.sequences_csv_path /app/data/raw/train_sequences.csv \
  --data.labels_csv_path /app/data/raw/train_labels.csv
```

## Multi-GPU Readiness

### Configuration for Multi-GPU

Extend the configuration with multi-GPU parameters:

```yaml
# Multi-GPU settings (future implementation)
distributed:
  enabled: false  # Set to true for multi-GPU training
  backend: "nccl"  # Communication backend
  world_size: 0   # Number of processes (0 = auto-detect)
  rank: 0         # Process rank
  local_rank: 0   # Local process rank
  sync_bn: false  # Sync BatchNorm statistics
```

### Model Distribution Implementation Pattern

```python
def setup_distributed(config):
    """Set up distributed training if enabled in config."""
    if not config.get("distributed", {}).get("enabled", False):
        return False
    
    # Initialize distributed backend
    dist_config = config["distributed"]
    world_size = dist_config.get("world_size", 0)
    if world_size == 0:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            world_size = torch.cuda.device_count()
    
    # Use torch.distributed for multi-GPU
    if world_size > 1:
        torch.distributed.init_process_group(
            backend=dist_config.get("backend", "nccl"),
            init_method="env://",
            world_size=world_size,
            rank=dist_config.get("rank", 0)
        )
        torch.cuda.set_device(dist_config.get("local_rank", 0))
        return True
    
    return False
```

### Wrapping Models for Distribution

```python
def get_model(config, device):
    """Create model and wrap for distributed training if needed."""
    model = RNAFoldingModel(config["model"]).to(device)
    
    # Optionally enable sync_bn
    if config.get("distributed", {}).get("sync_bn", False):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Wrap model for distributed training
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config["distributed"]["local_rank"]],
            output_device=config["distributed"]["local_rank"],
            find_unused_parameters=config.get("distributed", {}).get(
                "find_unused_parameters", False
            )
        )
    
    return model
```

## Best Practices

### 1. Default Parameters & Documentation

Always include default values and document parameters:

```yaml
# Example documented parameter
dropout: 0.1  # Dropout probability after attention layers, range [0-0.5]
```

### 2. Type Validation

Validate parameter types during loading:

```python
def _validate_parameter_types(config):
    """Validate parameter types against expected types."""
    type_specs = {
        "seed": int,
        "model.residue_embed_dim": int,
        "model.dropout": float,
        "data.batch_size": int,
        # Add more as needed
    }
    
    for param_path, expected_type in type_specs.items():
        value = _get_nested_value(config, param_path)
        if value is not None and not isinstance(value, expected_type):
            raise TypeError(
                f"Parameter {param_path} should be of type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
```

### 3. Parameter Scaling

Scale dependent parameters automatically:

```python
def _apply_parameter_scaling(config):
    """Apply scaling rules to ensure parameter consistency."""
    # Example: Scale FFN dimension based on embedding dimension
    if "model" in config:
        model_config = config["model"]
        if "ffn_dim" not in model_config and "residue_embed_dim" in model_config:
            # Default FFN dimension to 4x embedding dimension
            model_config["ffn_dim"] = 4 * model_config["residue_embed_dim"]
```

### 4. Configuration Versioning

Track configuration versions for reproducibility:

```yaml
version: "1.0.0"  # Configuration schema version
created_at: "2025-04-09T15:30:00"
```

### 5. Config Presets

Create task-specific configuration presets:

```
config/
├── default_config.yaml       # Base configuration
├── presets/
│   ├── small.yaml           # Small model preset
│   ├── medium.yaml          # Medium model preset
│   └── large.yaml           # Large model preset
└── environments/
    ├── dev.yaml             # Development environment
    ├── test.yaml            # Testing environment
    └── kaggle.yaml          # Kaggle environment
```

### 6. Saving Configuration with Results

Always save the full configuration with results:

```python
def save_training_state(model, optimizer, config, epoch, metrics, save_dir):
    """Save model, optimizer state, and configuration."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config  # Save full configuration
    }, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))
    
    # Also save configuration as YAML for human readability
    with open(os.path.join(save_dir, f"config_epoch_{epoch}.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
```

## Example Usage Scenarios

### Example 1: Local Development

```bash
python scripts/train.py \
  --config config/default_config.yaml \
  --data.sequences_csv_path data/raw/train_sequences.csv \
  --data.labels_csv_path data/raw/train_labels.csv \
  --data.features_dir data/processed \
  --training.checkpoint_dir experiments/run1/checkpoints \
  --model.residue_embed_dim 64 \
  --model.num_transformer_blocks 2
```

### Example 2: Docker Container Run

```bash
docker run --rm -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/experiments:/app/experiments \
  rna-3d:v1 python scripts/train.py \
  --config config/default_config.yaml \
  --data.sequences_csv_path /app/data/raw/train_sequences.csv \
  --data.labels_csv_path /app/data/raw/train_labels.csv \
  --data.features_dir /app/data/processed \
  --training.checkpoint_dir /app/experiments/run1/checkpoints
```

### Example 3: Kaggle Notebook Integration

```python
# In Kaggle Notebook
import yaml
import sys
import os

# Load the configuration
with open("config/kaggle_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Override with Kaggle-specific paths
config["data"]["sequences_csv_path"] = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
config["data"]["features_dir"] = "/kaggle/input/rna-features/processed"
config["inference"]["output_dir"] = "/kaggle/working"

# Import the prediction module
from src.models.rna_folding_model import RNAFoldingModel
from src.data_loading import RNADataset, collate_fn

# Create model using config
model = RNAFoldingModel(config["model"])

# Load weights
model.load_state_dict(torch.load("/kaggle/input/rna-model-weights/model.pt"))

# Run inference
# ...
```

## Configuration File Location

In accordance with path parameterization principles:

1. Default location: `config/default_config.yaml`
2. Always pass configuration path as a command-line argument
3. Load different configurations for different environments
4. In scripts, never assume the configuration file location

```python
# In scripts
parser.add_argument("--config", type=str, required=True,
                   help="Path to configuration file")
```

## Conclusion

Following these configuration practices ensures:

1. Clean separation of code and configuration
2. Environment adaptability (local, Docker, Kaggle)
3. Reproducibility through configuration versioning
4. Scalability to multi-GPU setups
5. Clear documentation of parameters and their meanings

When implementing components, always reference configuration values rather than hardcoding parameters, and ensure all file paths are passed as arguments in accordance with the path parameterization principle.
