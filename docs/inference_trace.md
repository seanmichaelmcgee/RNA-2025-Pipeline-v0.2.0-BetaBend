# Kaggle Inference Notebook Dependency Trace

This document traces all dependencies required by the Kaggle inference notebook. It maps the complete dependency tree to ensure all necessary files are included in the Kaggle dataset.

## Top-Level Notebook
- `/notebooks/kaggle_inference_v2.1.ipynb`

## Direct Imports

### Standard Libraries (provided by Kaggle)
- `os`
- `sys`
- `json`
- `numpy`
- `pandas`
- `matplotlib.pyplot`
- `torch`
- `math`
- `pathlib.Path`
- `tqdm.notebook`
- `datetime` (used by execution log)
- `traceback` (used by execution log)

### Project-Specific Imports
- `src.models.rna_folding_model.RNAFoldingModel`
- `src.data_loading.RNADataset`
- `src.data_loading.collate_fn`
- `src.data_loading.create_data_loader`

## Dependency Tree

### 1. Core Model

#### `src/models/rna_folding_model.py`
Primary model implementation that imports:
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `src.models.embeddings.SequenceEmbedding`
- `src.models.embeddings.PositionalEncoding`
- `src.models.transformer_block.TransformerBlock`
- `src.models.ipa_module.IPAModule`

#### `src/models/embeddings.py`
Embedding implementations that import:
- `torch`
- `torch.nn`
- `math`
- `numpy`

#### `src/models/transformer_block.py`
Transformer implementations that import:
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `math`
- `einops` (for tensor manipulation)

#### `src/models/ipa_module.py`
Invariant Point Attention module that imports:
- `torch`
- `torch.nn`
- `torch.nn.functional`
- `math`
- `einops` (for tensor manipulation)

### 2. Data Handling

#### `src/data_loading.py`
Data loading utilities that import:
- `torch`
- `torch.utils.data`
- `numpy`
- `pandas`
- `os`
- `json`
- `warnings`
- `src.utils.padding.pad_sequences` (potentially)

#### `src/utils/padding.py`
Sequence padding utilities that import:
- `torch`
- `numpy`

### 3. Feature Processing

All feature processing is handled within the patched `fixed_load_precomputed_features` function in the notebook itself, which uses:
- `os`
- `numpy`
- `warnings`
- `pandas`

### 4. Metrics (Only if validation is performed)

#### `src/utils/structure_metrics.py`
Structure metrics that might be used for validation:
- `numpy`
- `torch`
- `scipy.spatial.distance`

## Additional Dependencies

### Custom Components in Notebook
The notebook also defines several custom components:
- `EnhancedPositionalEncoding` class (extends `nn.Module`)
- Functions for model loading, inference, and submission formatting

## Required Data Files
- Test sequences: CSV file with RNA sequences
- Feature files directory structure:
  - `/dihedral_features/{target_id}_dihedral_features.npz`
  - `/thermo_features/{target_id}_thermo_features.npz`
  - `/evolutionary_features/{target_id}_evolutionary_features.npz` (optional)
- Model checkpoint files:
  - Must include both model weights and configuration information

## Directory Structure for Kaggle
```
/kaggle/input/
├── stanford-rna-3d-folding/
│   └── test_sequences.csv
├── rna-3d-features/
│   ├── dihedral_features/
│   ├── thermo_features/
│   └── evolutionary_features/
├── rna-3d-models/
│   ├── best_model.pt
│   └── production_model.pt
└── rna-model-src/
    ├── src/
    │   ├── data_loading.py
    │   ├── models/
    │   │   ├── rna_folding_model.py
    │   │   ├── embeddings.py
    │   │   ├── transformer_block.py
    │   │   └── ipa_module.py
    │   └── utils/
    │       ├── padding.py
    │       └── structure_metrics.py
    └── (any other necessary files)
```

## Verification Checklist
- [ ] All required source files included in `rna-model-src` dataset
- [ ] All model checkpoint files included in `rna-3d-models` dataset
- [ ] Feature files structure matches expected paths
- [ ] Notebook paths updated to match Kaggle environment
- [ ] Dependencies like `einops` available in Kaggle environment
- [ ] Model checkpoint format compatible with loading method

## Common Issues
1. Missing source files - ensure all dependent modules are included
2. Path mismatches - verify all paths work in Kaggle environment
3. Incompatible module versions - especially PyTorch version differences
4. Memory management - Kaggle P100 GPU has memory limitations
5. Missing third-party libraries - check if all are available in Kaggle