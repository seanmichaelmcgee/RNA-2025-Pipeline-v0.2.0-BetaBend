# RNA 3D Structure Prediction - Kaggle Package Usage Guide

This document provides detailed instructions for testing, validating, and using the dual-purpose package for RNA 3D structure prediction.

## Package Overview

This package is designed to work seamlessly in both local and Kaggle environments without path modifications. It contains:

1. **Source Code**: Complete model implementation with proper module structure
2. **Model Checkpoints**: Trained models for structure prediction
3. **Test Data**: Sample sequences for validation
4. **Inference Notebook**: Main entry point for predictions

The directory structure is carefully organized to support dual-purpose usage with automatic environment detection.

## Step 1: Validate Package

Before using the package, validate that all components are properly set up:

```bash
cd /path/to/kaggle_package
python validate_package.py
```

This will verify:
- Directory structure is correct
- All required files are present
- Model loading works properly
- Basic inference functionality operates as expected

Fix any issues identified before proceeding.

## Step 2: Local Testing

To test the package locally:

1. Navigate to the notebook directory:
   ```bash
   cd /path/to/kaggle_package/notebooks
   ```

2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook kaggle_inference_v2.1.ipynb
   ```

3. **Add diagnostic cells** - Insert the test cells at the appropriate locations:
   - Add import test cell (from `import_test_cell.py`) after the initial imports
   - Add data test cell (from `data_test_cell.py`) after path configuration

4. Run the notebook cells to process test sequences and generate predictions

5. Check the submissions directory for output files

The diagnostic cells provide comprehensive testing of:
- All required library imports
- Project module imports
- Test sequence loading
- Feature file availability and loading
- Dataset and dataloader creation
- Basic functionality verification

## Step 3: Kaggle Submission

To use the package on Kaggle:

1. Prepare the directories for Kaggle:
   ```bash
   # Create and populate Kaggle-ready directories
   mkdir -p kaggle_upload/rna-model-src/src
   mkdir -p kaggle_upload/rna-3d-models
   mkdir -p kaggle_upload/rna-3d-features

   # Copy source code
   cp -r src/* kaggle_upload/rna-model-src/src/
   
   # Copy model checkpoints
   cp results/final_model/run_20250423-072601/checkpoints/best_model.pt kaggle_upload/rna-3d-models/
   cp results/production_run_1/run_20250423-072209/checkpoints/best_model.pt kaggle_upload/rna-3d-models/production_model.pt
   
   # Copy feature files (this will take longer)
   cp -r data/processed/* kaggle_upload/rna-3d-features/
   ```

2. Upload the following directories as separate datasets in Kaggle:
   - `kaggle_upload/rna-model-src` → Upload as dataset "rna-model-src"
   - `kaggle_upload/rna-3d-models` → Upload as dataset "rna-3d-models"
   - `kaggle_upload/rna-3d-features` → Upload as dataset "rna-3d-features"

3. Upload the notebook `notebooks/kaggle_inference_v2.1.ipynb` to a new Kaggle notebook

4. Add the "stanford-rna-3d-folding" competition dataset

5. Add your uploaded datasets as input to the notebook

6. Add the diagnostic cells from `import_test_cell.py` and `data_test_cell.py` to verify everything is working

7. Run the notebook and verify predictions

## Model Checkpoint Handling

This package includes special handling for model checkpoints:

1. The `src/utils/model_loader.py` module handles corrupted checkpoints gracefully
2. If a checkpoint contains only a 'dummy' key, it initializes from scratch using the configuration
3. Validation is performed to ensure the model can generate coordinate predictions

## Memory Optimization

The notebook includes several memory optimization techniques for Kaggle's P100 GPU:

1. PyTorch memory allocator configuration for reduced fragmentation 
2. Adaptive batch size based on sequence length
3. Mixed precision inference with torch.cuda.amp
4. Strategic tensor de-allocation between operations
5. Enhanced positional encoding implementation

## Troubleshooting

If you encounter issues:

1. **Model Loading Errors**:
   - Check that model checkpoint files are in the expected locations
   - Verify checkpoint file integrity
   - Try loading with different checkpoint files

2. **Memory Issues**:
   - Increase memory logging verbosity
   - Try processing with smaller batch sizes
   - Use sequence length limits for testing

3. **Import Errors**:
   - Verify module structure with `__init__.py` files
   - Check for proper dependencies in Kaggle
   - Ensure paths are correctly set up

## Known Issues

1. **Dummy Checkpoints**: Some checkpoints contain only placeholder data requiring initialization from scratch
2. **Long Sequences**: Sequences longer than 700nt may exceed GPU memory on Kaggle P100
3. **Feature Files**: Some feature files may have inconsistent formats

## Final Checklist

Before submission, verify:

- [ ] Package validation passes all checks
- [ ] Notebook runs locally without errors
- [ ] Model loading works with proper error handling
- [ ] Memory optimization is enabled for Kaggle
- [ ] Output format matches Kaggle submission requirements
- [ ] All dependencies are properly included