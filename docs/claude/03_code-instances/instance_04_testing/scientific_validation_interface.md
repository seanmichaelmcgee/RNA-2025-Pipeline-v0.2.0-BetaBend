# Scientific Validation Framework Interface Documentation

**Version:** v1.0  
**Last Updated:** 2025-04-23  
**Component:** Tier 2 Scientific Validation Framework  
**Responsible Instance:** 04_testing  

## 1. Component Overview

The Scientific Validation Framework provides comprehensive tools for validating the RNA 3D folding model from a scientific perspective. It implements dual-mode validation to quantify the impact of feature availability differences between training and testing environments, along with detailed scientific analysis of structure quality metrics.

The framework consists of three main components:
1. Jupyter notebook for interactive validation and analysis
2. Python command-line script for programmatic validation
3. Shell script for simplified execution

## 2. Interface Specification

### 2.1 Jupyter Notebook Interface

**File:** `validation/tier2_scientific/validation_scientific.ipynb`

#### Input Parameters:
- **data_dir**: Path to data directory containing features
- **subset_size**: Number of sequences for validation (default: 12)
- **model_config**: Dictionary of model configuration parameters:
  ```python
  {
      "d_model": 256,        # Embedding dimension
      "d_feedforward": 1024, # Feedforward network dimension
      "num_layers": 4,       # Number of transformer layers
      "num_heads": 8,        # Number of attention heads
      "dropout": 0.1,        # Dropout rate
      "ipa_dropout": 0.1,    # IPA module dropout rate
      "use_checkpointing": False  # Whether to use gradient checkpointing
  }
  ```
- **results_dir**: Directory for output files
- **checkpoint_path**: Optional path to model checkpoint
- **metrics**: List of metrics to compute (default: ["rmsd", "tm_score"])

#### Output:
- **validation_results**: Dictionary containing validation results:
  ```python
  {
      "test_mode": {
          "mean_rmsd": float,          # Mean RMSD across all targets
          "mean_tm_score": float,      # Mean TM-score across all targets
          "target_ids": List[str],     # List of target IDs
          "rmsd_values": List[float],  # Per-target RMSD values
          "tm_scores": List[float],    # Per-target TM-scores
          "sequence_lengths": List[int]  # Per-target sequence lengths
      },
      "train_mode": {
          # Same structure as test_mode
      },
      "analysis": {
          "conclusion": {
              "overall_impact": str,    # Impact assessment
              "severity": str,          # Severity level
              "recommendation": str     # Recommendation for improvement
          }
      }
  }
  ```
- **Visualizations**:
  - Sequence length vs. quality plots
  - Quality metric distributions
  - Feature impact analysis
  - RNA family performance comparisons
- **Reports**:
  - Markdown report with comprehensive analysis
  - JSON file with serialized results

### 2.2 Command-line Script Interface

**File:** `validation/tier2_scientific/run_dual_mode_validation.py`

#### Command-line Arguments:
```
usage: run_dual_mode_validation.py [-h] [--checkpoint CHECKPOINT] [--data_dir DATA_DIR]
                                   [--output_dir OUTPUT_DIR] [--batch_size BATCH_SIZE]
                                   [--subset_size SUBSET_SIZE] [--cpu]
                                   [--rna-ids RNA_IDS [RNA_IDS ...]]
                                   [--rna-families RNA_FAMILIES [RNA_FAMILIES ...]]

Run scientific validation for RNA 3D folding model

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Path to model checkpoint
  --data_dir DATA_DIR   Path to data directory
  --output_dir OUTPUT_DIR
                        Directory for output files
  --batch_size BATCH_SIZE
                        Batch size for validation
  --subset_size SUBSET_SIZE
                        Number of sequences to use
  --cpu                 Force CPU usage (not recommended)
  --rna-ids RNA_IDS [RNA_IDS ...]
                        Specific RNA IDs to validate
  --rna-families RNA_FAMILIES [RNA_FAMILIES ...]
                        Specific RNA families to validate (e.g., tRNA, ribosomal)
```

#### Output:
- Console output with validation results summary
- Results files saved to output_dir:
  - `validation_scientific_results_TIMESTAMP.md`: Markdown report
  - `validation_scientific_results_TIMESTAMP.json`: JSON results
  - Visualization images (PNG format)

### 2.3 Shell Script Interface

**File:** `validation/tier2_scientific/run_scientific_validation.sh`

#### Command-line Arguments:
```
Usage: ./run_scientific_validation.sh [--checkpoint PATH] [--data_dir PATH] 
                                     [--batch_size N] [--subset_size N] [--cpu] 
                                     [--rna-ids IDS...] [--rna-families FAMILIES...]
```

#### Output:
- Same as the Python command-line script

## 3. Usage Examples

### 3.1 Running the Jupyter Notebook

```python
# Manual configuration in the notebook
CONFIG = {
    "data_dir": "/path/to/data",
    "subset_size": 15,  # Use 15 sequences
    "model_config": {
        "d_model": 256,
        "d_feedforward": 1024,
        "num_layers": 4,
        "num_heads": 8,
        "dropout": 0.1
    },
    "results_dir": "/path/to/results",
    "checkpoint_path": "/path/to/model.pt",
    "metrics": ["rmsd", "tm_score"]
}
```

### 3.2 Running the Command-line Script

```bash
# Basic usage
python validation/tier2_scientific/run_dual_mode_validation.py

# With specific configuration
python validation/tier2_scientific/run_dual_mode_validation.py \
  --checkpoint /path/to/model.pt \
  --data_dir /path/to/data \
  --output_dir /path/to/results \
  --batch_size 4 \
  --subset_size 20 \
  --rna-ids 1A51_A 2GDI_B 3V7E_P

# Force CPU usage
python validation/tier2_scientific/run_dual_mode_validation.py --cpu
```

### 3.3 Running the Shell Script

```bash
# Basic usage
./validation/tier2_scientific/run_scientific_validation.sh

# With specific configuration
./validation/tier2_scientific/run_scientific_validation.sh \
  --checkpoint /path/to/model.pt \
  --data_dir /path/to/data \
  --batch_size 4 \
  --subset_size 20 \
  --rna-families tRNA ribosomal
```

## 4. Error Handling

### 4.1 Common Error Conditions

| Error Condition | Handling Approach |
|-----------------|-------------------|
| Data directory not found | Attempts to find alternative data locations; fails with clear error if none found |
| Model checkpoint not found | Initializes model with random weights; provides warning message |
| CUDA not available | Falls back to CPU; provides warning message |
| Invalid RNA IDs | Ignores invalid IDs; proceeds with valid ones; provides warning |
| Memory overflow | Reduces batch size when possible; provides clear error if unrecoverable |
| Numeric instability | Applies robust calculation methods with epsilon values; logs warnings |

### 4.2 Error Reporting

Errors are reported through:
- Console messages with clear descriptions
- Exception traceback for debugging
- Log messages in the results directory

## 5. Performance Characteristics

### 5.1 Runtime Performance

- **Execution Time**: Typically <15 minutes for 12 sequences on GPU
- **Memory Usage**: ~4GB RAM with default settings
- **GPU Memory**: ~2GB VRAM with default settings
- **Scaling**: Approximately linear with sequence count (O(n))

### 5.2 Resource Management

- Configurable batch size to control memory usage
- Configurable subset size to control total runtime
- Option to use CPU when GPU unavailable

## 6. Dependencies

### 6.1 Required Python Packages

- PyTorch 2.1+
- NumPy
- matplotlib
- seaborn
- Jupyter (for notebook only)

### 6.2 Component Dependencies

- validation_runner.py
- structure_metrics.py (for RMSD and TM-score calculations)
- RNAFoldingModel class

## 7. Future Interface Extensions

Planned extensions for the interface include:
- Additional scientific metrics for RNA-specific structural properties
- Support for ensemble prediction validation
- Comparison with reference models
- Integration with RNA structure visualization tools
- Enhanced RNA family classification with database integration

## 8. Versioning

- Current version: v1.0 (2025-04-23)
- Interface changes will be documented in future versions
- Backward compatibility will be maintained whenever possible