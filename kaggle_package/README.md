# RNA 3D Structure Prediction - Kaggle Package

This package contains a lean, refactored version of the RNA 2025 Pipeline for 3D structure prediction, designed to work both locally and on Kaggle without path modifications.

## Getting Started

1. **Validate the package**: Run `python validate_package.py` to verify all components work correctly
2. **Read usage guide**: See `USAGE.md` for detailed instructions and troubleshooting
3. **Add diagnostic cells**: Add the import and data test cells from `notebooks/import_test_cell.py` and `notebooks/data_test_cell.py` to your notebook
4. **Test locally**: Run the notebook from the `notebooks/` directory to test local execution
5. **Prepare for Kaggle**: Follow the upload instructions in `USAGE.md` for Kaggle submission

## Directory Structure

```
kaggle_package/
├── notebooks/
│   └── kaggle_inference_v2.1.ipynb  # Main inference notebook
├── data/
│   ├── raw/
│   │   └── test_sequences.csv       # Test sequences
│   └── processed/                   # Feature files
│       ├── dihedral_features/
│       ├── thermo_features/
│       └── evolutionary_features/
├── results/                         # Model checkpoints
│   ├── final_model/
│   │   └── run_20250423-072601/
│   │       └── checkpoints/
│   │           └── best_model.pt
│   └── production_run_1/
│       └── run_20250423-072209/
│           └── checkpoints/
│               └── best_model.pt
├── submissions/                     # Output directory for predictions
└── src/                             # Source code
    ├── models/
    │   ├── rna_folding_model.py
    │   ├── embeddings.py
    │   ├── transformer_block.py
    │   ├── ipa_module.py
    │   └── __init__.py
    ├── utils/
    │   ├── padding.py
    │   ├── structure_metrics.py 
    │   └── __init__.py
    ├── data_loading.py
    └── __init__.py
```

## Usage Instructions

### Local Execution

1. Navigate to the `notebooks` directory
2. Open and run `kaggle_inference_v2.1.ipynb`
3. The notebook will automatically detect it's running locally and use the proper paths

### Kaggle Execution

When uploading to Kaggle:

1. Upload the following directories as separate datasets:
   - `src` → Upload as dataset "rna-model-src"
   - `results` → Upload as dataset "rna-3d-models"
   - `data/processed` → Upload as dataset "rna-3d-features"

2. Upload the notebook `notebooks/kaggle_inference_v2.1.ipynb` to a new Kaggle notebook

3. Add the "stanford-rna-3d-folding" competition dataset

4. Add your uploaded datasets as input to the notebook

The notebook will automatically detect the Kaggle environment and use the appropriate paths.

## Path Handling

The package uses the following path logic to work in both environments:

```python
# Detect Kaggle environment
is_kaggle = os.path.exists('/kaggle')

if is_kaggle:
    # Kaggle environment paths
    TEST_SEQUENCES_PATH = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv"
    FEATURES_DIR = "/kaggle/input/rna-3d-features/"
    OUTPUT_DIR = "/kaggle/working/"
    
    # Model paths in Kaggle
    MODEL_PATHS = {
        "final_model": "/kaggle/input/rna-3d-models/best_model.pt",
        "production_run_1": "/kaggle/input/rna-3d-models/production_model.pt",
    }
else:
    # Local environment paths
    TEST_SEQUENCES_PATH = "../data/raw/test_sequences.csv"
    FEATURES_DIR = "../data/processed/"
    OUTPUT_DIR = "../submissions/"
    
    # Local model paths
    MODEL_PATHS = {
        "final_model": "../results/final_model/run_20250423-072601/checkpoints/best_model.pt",
        "production_run_1": "../results/production_run_1/run_20250423-072209/checkpoints/best_model.pt"
    }
```

## Notes on Feature Files

Feature files (in `data/processed/`) are not included in this package due to their size. For local execution, you would need to:

1. Generate feature files using the feature generation pipeline, or
2. Copy existing feature files to the appropriate directories

For Kaggle execution, feature files will either be:
1. Generated during the competition using Kaggle's provided tools, or
2. Uploaded as a separate dataset

## Memory Optimization

The notebook includes several memory optimization techniques for the Kaggle P100 GPU:
- Adaptive batch size based on sequence length
- Mixed precision inference
- Memory cleanup between operations
- Enhanced positional encoding for long sequences