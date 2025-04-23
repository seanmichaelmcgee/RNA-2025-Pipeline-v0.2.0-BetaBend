# RNA 2025 Pipeline Workflow Improvements

## Recent Enhancements

We've implemented several key improvements to streamline our workflow and prepare for Kaggle submissions:

### 1. Training Report Generation

- Fixed the report generation script to properly parse our file structure
- Updated to handle both CSV and log file formats
- Generated comprehensive reports with loss curves, RMSD metrics, and GPU utilization
- Usage:
  ```bash
  mamba run -n rna-3d-folding python scripts/generate_training_report.py --training_dir="results/final_model/run_20250423-072601" --output_format=md
  ```

### 2. Enhanced Kaggle Inference

- Added support for multiple model checkpoints
- Implemented automatic model selection based on validation RMSD
- Created model ensembling capabilities with RMSD-based weighting
- Added detailed metadata tracking for submissions
- Usage: Run the `notebooks/kaggle_inference.ipynb` notebook after setting:
  ```python
  # For single best model (auto-selected)
  SELECTED_MODEL = None
  USE_ENSEMBLE = False
  
  # For ensemble of all models
  SELECTED_MODEL = None
  USE_ENSEMBLE = True
  
  # For specific model
  SELECTED_MODEL = "final_model"  # Choose from MODEL_PATHS keys
  USE_ENSEMBLE = False
  ```

## Next Steps

### Immediate Pipeline:

1. **Verify notebook execution**: Run the Kaggle inference notebook to catch and fix any runtime errors
2. **Test single model submission**: Generate a submission with our best current model
3. **Test ensemble submission**: Generate a submission using model ensembling
4. **Compare submissions**: Evaluate which approach yields better RMSD metrics

### Model Improvements:

1. **Increase model capacity**: Train a larger model with:
   - More transformer blocks (6-8 instead of 4)
   - Larger embedding dimensions (192-256)
   - More attention heads (8)

2. **Improve training strategy**:
   - Increase sequence length for training (better handling of longer RNAs)
   - Fine-tune learning rate and schedule
   - Adjust batch size for better gradient estimates
   - Longer training with early stopping

3. **Data enhancements**:
   - Include additional RNA families in training
   - Apply more aggressive data augmentation
   - Implement feature ablation studies to identify most important features

### Production Script Development:

Create a dedicated training script with configurable parameters:
- Easy-to-configure model architecture
- Automated hyperparameter tuning
- Checkpoint management and evaluation
- Direct integration with validation framework
- Simplified submission generation

## Expected Outcomes

By implementing these improvements, we expect to:
1. Reduce RMSD by 10-15% through model architecture improvements
2. Further reduce RMSD by 5-10% through ensemble methods
3. Establish a streamlined workflow from training to submission
4. Enable rapid iteration on model improvements