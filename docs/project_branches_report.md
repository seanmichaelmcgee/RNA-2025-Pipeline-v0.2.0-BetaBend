# RNA-2025-Pipeline Project Branches and Development Report

## Project Overview

**Project**: RNA-2025-Pipeline  
**Version**: v0.2.0-BetaBend  
**Focus**: RNA 3D structure prediction for Stanford RNA 3D Folding competition  
**Primary Location**: `/home/smcgee/MLprojects/RNA_2025/Pipeline-v1-March27on/RNA-2025-Pipeline-v0.2.0-BetaBend/`

This report provides a comprehensive overview of the project branches, file structure, and development activities for the RNA-2025-Pipeline project.

## Branch Structure

The project is currently maintained on a single branch:

- **main**: Primary development branch where all active work is being done

The repository has one remote tracking branch:
- **remotes/origin/main**: The remote version of the main branch

We have been working exclusively in the `main` branch, with all feature development and bug fixes committed directly to it. The development follows a streamlined approach with each commit representing a specific feature enhancement or bug fix.

## Recent Development Timeline

Recent development has focused on the Kaggle inference notebook and package preparation. Here's a timeline of key commits:

| Commit Hash | Description | Date |
|-------------|-------------|------|
| 65116f1 | Add Kaggle package with dual-purpose structure and diagnostic tools | Apr 23, 2025 |
| b70f7d2 | Add inference dependency trace document | Apr 23, 2025 |
| 9446157 | Fix execution log generator to handle non-string history entries | Apr 23, 2025 |
| 6dbed93 | Add Kaggle submission format conversion | Apr 22, 2025 |
| 1ac91d4 | Add Kaggle environment detection and path adjustments | Apr 22, 2025 |
| 7415aef | Optimize Kaggle inference notebook for memory efficiency | Apr 22, 2025 |
| 65f2c32 | Fix missing nn import in Kaggle notebook | Apr 22, 2025 |
| 76c5931 | Fix tensor dimension mismatch for long sequences | Apr 22, 2025 |
| b717737 | Fix data loading for multi-structure feature files | Apr 22, 2025 |
| de53edd | Fix model loading to handle corrupted checkpoints | Apr 21, 2025 |
| 2789013 | Start debugging Kaggle inference notebook | Apr 21, 2025 |
| 61e4348 | Fix execution log generator for standard IPython | Apr 20, 2025 |
| 3f4409e | Add execution log generator to Kaggle notebook | Apr 20, 2025 |

## Directory Structure and Key Files

The project follows a modular directory structure:

### Source Code (`/src`)
- **Core ML Model Components**:
  - `/src/models/rna_folding_model.py`: Main RNA 3D structure prediction model
  - `/src/models/embeddings.py`: Sequence and feature embeddings
  - `/src/models/transformer_block.py`: Transformer architecture for RNA modeling
  - `/src/models/ipa_module.py`: Invariant Point Attention for 3D coordinate prediction
- **Data Handling**:
  - `/src/data_loading.py`: Dataset implementation and feature preprocessing
  - `/src/utils/padding.py`: Sequence padding utilities
- **Metrics and Evaluation**:
  - `/src/utils/structure_metrics.py`: Structure quality evaluation metrics
  - `/src/losses.py`: Loss functions for model training

### Notebooks (`/notebooks`)
- `/notebooks/kaggle_inference_v2.1.ipynb`: Latest Kaggle inference notebook
- `/notebooks/kaggle_inference_v2.0.ipynb`: Previous version of Kaggle notebook
- `/notebooks/production_run_analysis.ipynb`: Analysis of production training runs

### Kaggle Package (`/kaggle_package`)
- `/kaggle_package/kaggle_test_package.tar.gz`: Complete archive for Kaggle deployment
- `/kaggle_package/src/`: Source code in proper module structure
- `/kaggle_package/results/`: Model checkpoints
- `/kaggle_package/notebooks/`: Kaggle inference notebook and test cells
- `/kaggle_package/validate_package.py`: Validation script for package integrity

### Data (`/data`)
- `/data/raw/`: Raw RNA sequences and labels
- `/data/processed/`: Precomputed feature files (dihedral, thermodynamic, evolutionary)

### Training Results (`/results`)
- `/results/final_model/`: Best model checkpoint (RMSD ~7Å)
- `/results/production_run_1/`: Production training run results
- Model configuration uses:
  - Batch size: 32
  - Learning rate: 0.0005
  - Mixed precision: true
  - FAPE weight: 1.0
  - Confidence weight: 0.2
  - Angle weight: 0.5

### Submission Files (`/submissions`)
- Files formatted for Kaggle submission
- Includes both metadata and coordinate predictions

## Current Development Focus

The project is currently focused on:

1. **Kaggle Submission Preparation**:
   - Creating a robust package that works in both local and Kaggle environments
   - Fixing issues with model loading and inference
   - Optimizing memory usage for Kaggle's P100 GPU
   - Adding diagnostic tools for testing in Kaggle environment

2. **Model Loading Challenges**:
   - Handling corrupted model checkpoints with dummy state dictionaries
   - Implementing fallback mechanisms for model initialization from configuration

3. **Data Loading Improvements**:
   - Supporting multi-structure feature files with different key formats
   - Detecting and handling missing features gracefully

4. **Memory Optimization**:
   - Implementing mixed precision inference
   - Using adaptive batch sizes based on sequence length
   - Configuring memory allocator for efficient GPU utilization
   - Strategic tensor de-allocation and cache clearing

## Key Technical Components

### Dual-Purpose Architecture
- Automatically detects whether it's running locally or in Kaggle environment
- Adjusts file paths and loading mechanisms based on environment
- Eliminates the need for manual path adjustments between environments

### Robust Model Loader
- Handles corrupted model checkpoints with 'dummy' keys
- Maintains configuration/architecture even when weights are missing
- Implemented in `src/utils/model_loader.py`

### Memory Optimization Techniques
- Mixed precision inference using `torch.cuda.amp.autocast()`
- Adaptive batch sizing based on sequence length
- Memory allocator configuration for reduced fragmentation
- Strategic memory monitoring and management

### Diagnostic Tools
- Import test cell for verifying all required imports
- Data test cell for validating feature loading
- Package validation script for checking integrity

## Recent Challenges and Solutions

| Challenge | Solution | File Location |
|-----------|----------|---------------|
| Corrupted model checkpoints | Robust loader that initializes from config when state dict is corrupted | `/src/utils/model_loader.py` |
| Multi-structure feature files | Enhanced loader that handles different key formats | `/src/data_loading.py` |
| Tensor dimension mismatch | Enhanced positional encoding for long sequences | Implemented in notebook |
| Memory constraints on P100 GPU | Comprehensive memory optimization techniques | `/notebooks/kaggle_inference_v2.1.ipynb` |
| Path differences between environments | Dual-purpose architecture with environment detection | `/kaggle_package` structure |

## File Name

This report is saved as: `/docs/project_branches_report.md`
