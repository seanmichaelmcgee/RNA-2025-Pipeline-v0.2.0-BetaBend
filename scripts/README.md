# RNA 3D Structure Prediction Scripts

This directory contains essential scripts for running various aspects of the RNA 3D structure prediction pipeline.

## Training Pipeline

The following scripts enable model training and evaluation:

### `train.py`

Complete training pipeline that implements:
- Data loading and batching 
- Forward/backward passes
- Loss calculation
- Checkpoint management
- Validation during training
- Learning rate scheduling
- Early stopping

**Usage:**
```bash
python scripts/train.py --sequences_csv data/raw/train_sequences.csv \
                       --labels_csv data/raw/train_labels.csv \
                       --features_dir data/processed/ \
                       --output_dir training_results \
                       [--additional_args]
```

### `validate_trained_model.py`

Comprehensive validation script that:
- Loads a trained model checkpoint
- Evaluates on validation data
- Calculates RMSD and TM-score metrics
- Generates performance visualizations

**Usage:**
```bash
python scripts/validate_trained_model.py --checkpoint path/to/model.pt \
                                       --sequences_csv data/raw/train_sequences.csv \
                                       --labels_csv data/raw/train_labels.csv \
                                       --features_dir data/processed/ \
                                       --generate_plots
```

### Helper Scripts

#### `run_training.sh`

Convenience script for launching training with common parameters.

**Usage:**
```bash
./scripts/run_training.sh [--additional_args]
```

#### `run_validation.sh`

Convenience script for running validation on trained models.

**Usage:**
```bash
./scripts/run_validation.sh --checkpoint training_results/checkpoints/best_model.pt
```

## Inference Pipeline

### `run_inference.py`

End-to-end inference script that:
- Loads a trained model
- Generates predictions for RNA sequences
- Saves output coordinates and metrics

### `quick_validation.py`

Fast validation script for checking basic model functionality.

## Output Directories

- `training_results/`: Contains model checkpoints and training logs
- `validation_results/`: Contains validation outputs and visualizations