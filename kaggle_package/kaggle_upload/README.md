# RNA 3D Structure Prediction - Kaggle Test Package

This package contains a minimal test version of the Kaggle submission files for RNA 3D structure prediction. It includes dummy model checkpoints for testing purposes.

## Testing Instructions

1. **Add diagnostic cells** to the notebook:
   - Import test cell (`import_test_cell.py`) after the initial imports
   - Data test cell (`data_test_cell.py`) after path configuration

2. **Simulated Kaggle environment setup**:
   - These folders simulate what would be available on Kaggle:
     - `/rna-model-src` - Source code for models
     - `/rna-3d-models` - Model checkpoint files (dummy versions)
     - `/rna-3d-features` - Empty directories for feature files 
     - `/stanford-rna-3d-folding` - Test sequence data

3. **Local testing**:
   - Place this directory under a path like `/kaggle/`
   - Run the notebook with paths set to Kaggle mode

## Structure

```
kaggle_upload/
├── kaggle_inference_v2.1.ipynb    # Main inference notebook
├── import_test_cell.py            # Test cell for imports
├── data_test_cell.py              # Test cell for data loading
├── USAGE.md                       # Usage instructions
├── rna-model-src/                 # Source code
├── rna-3d-models/                 # Model checkpoint files
│   ├── best_model.pt              # Dummy model checkpoint
│   └── production_model.pt        # Dummy model checkpoint
├── rna-3d-features/               # Feature directories
│   ├── dihedral_features/
│   ├── thermo_features/
│   └── evolutionary_features/     # Mutual Information features
└── stanford-rna-3d-folding/       # Test sequence data
    └── test_sequences.csv
```

## Notes

1. **Model checkpoints** are dummy versions with placeholder data
2. **Feature files** need to be populated with actual data or will fail loading
3. **Path detection** should automatically detect Kaggle-like environment
4. **Diagnostic cells** provide detailed error information for troubleshooting

For full instructions, see `USAGE.md`.