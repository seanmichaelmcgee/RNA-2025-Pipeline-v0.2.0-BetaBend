# CLAUDE.md for Data Pipeline Instance

This file provides Claude Code with guidance for the data loading pipeline implementation.

## Key References
- Main documentation: `docs/claude/03_code-instances/01_data_pipeline_kickoff.md`
- Implementation plan: `docs/claude/03_code-instances/instance_01_data/implementation_plan.md`
- Implementation journal: `docs/claude/03_code-instances/instance_01_data/implementation_journal.md`
- Feature specifications: `docs/claude/04_reference/feature_formats.md`

## Build & Test Commands

### Environment Setup
```bash
mamba env create -f environment.yml
mamba activate rna-3d-folding
```

### Data Loading Tests
```bash
# Run all data loading tests
python -m pytest tests/test_data_loading.py -v

# Run specific tests
python -m pytest tests/test_data_loading.py::TestRNADataset::test_initialization -v
python -m pytest tests/test_data_loading.py::TestFeatureLoading::test_load_precomputed_features -v
python -m pytest tests/test_data_loading.py::TestPartialFeatures::test_feature_availability -v

# Run tests with coverage
python -m pytest tests/test_data_loading.py --cov=src.data_loading
```

### Code Quality Commands
```bash
# Format code
black src/data_loading.py src/utils/padding.py
isort src/data_loading.py src/utils/padding.py

# Type checking
mypy src/data_loading.py src/utils/padding.py
```

## Feature File Structure

Feature files are stored in the `processed/` directory with the following structure:
```
processed/
├── dihedral_features/        # Backbone geometry features
│   └── {target_id}_dihedral_features.npz
├── thermo_features/          # Thermodynamic folding features
│   └── {target_id}_thermo_features.npz
└── mi_features/              # Mutual information/evolutionary features
    └── {target_id}_mi_features.npz
```

Example paths for test data:
```
data/processed/dihedral_features/1A51_A_dihedral_features.npz
data/processed/thermo_features/1A51_A_thermo_features.npz  
data/processed/mi_features/1A51_A_mi_features.npz
```

## Data Loading Code Style Guidelines

### Import Pattern
```python
# Standard library imports
import os
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable

# Third-party imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Local imports (always use relative imports)
from .utils.padding import pad_1d, pad_2d, pad_tensor
```

### Path Handling (CRITICAL)
- NEVER use hardcoded paths in source code
- ALWAYS use parameters for file paths
- Use `os.path.join()` for path construction
- Validate all path inputs

Example:
```python
def load_feature(features_dir: str, target_id: str, feature_type: str) -> Dict[str, np.ndarray]:
    """Load feature file for a given target_id and feature type."""
    # Correct path construction
    feature_path = os.path.join(features_dir, f"{feature_type}_features", f"{target_id}_{feature_type}_features.npz")
    
    # Path validation
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
        
    # Load feature
    return np.load(feature_path)
```

### Error Handling
- Use specific exception types
- Include context information in error messages
- Validate inputs before processing
- Handle missing features gracefully

## Key Implementation Details

### Feature Availability Detection
The feature availability detection system must implement:
- Scanning of features directory to identify available feature files
- Caching mechanism to avoid repeated filesystem operations
- Update mechanism to incorporate newly available features

### Metadata Generation
All batches must include metadata flags indicating:
- Feature presence (`has_dihedrals`, `has_msa`, etc.)
- Sequence properties (length, etc.)
- Training/validation split information

### Tensor Shapes
Key tensor shapes that must be maintained:
- `sequence_int`: (batch_size, max_seq_len)
- `dihedral_features`: (batch_size, max_seq_len, 4)
- `pairing_probs`: (batch_size, max_seq_len, max_seq_len)
- `positional_entropy`: (batch_size, max_seq_len)
- `coupling_matrix`: (batch_size, max_seq_len, max_seq_len)
- `coordinates`: (batch_size, max_seq_len, 3)
- `mask`: (batch_size, max_seq_len), type: bool, True for valid positions

## Test Data Reference

Example test sequences:
- 1A51_A - RNA small subunit
- 1F27_B - RNA/DNA complex
- 1LNG_A - tRNA fragment

Use these for feature availability testing and validation.