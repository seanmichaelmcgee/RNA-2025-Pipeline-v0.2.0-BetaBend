# Validation Runner Interface Contract

## Overview
The ValidationRunner class serves as the primary interface for executing model validation in dual-mode (test and training modes). This contract defines how to interact with the ValidationRunner, expected inputs and outputs, and key functionality.

## Version
- Version: 1.0
- Last Updated: 2021-04-21
- Status: Implemented (In Progress)

## Dependencies
- `src.utils.structure_metrics`: For computing RMSD, TM-score and other structural metrics
- `src.losses`: For the stable_kabsch_align implementation
- `validation.validation_dataset`: For loading validation data in different modes

## Class: ValidationRunner

### Constructor

```python
def __init__(self, 
             model: nn.Module, 
             data_dir: Optional[str] = None,
             config: Optional[Dict[str, Any]] = None,
             device: Optional[str] = None) -> None:
    """
    Initialize validation runner.
    
    Args:
        model: Model to validate
        data_dir: Path to data directory. If None, will try to find it.
        config: Configuration dictionary. If None, uses default configuration.
        device: Device to run validation on (e.g., 'cuda', 'cpu')
    """
```

### Core Methods

#### `run_validation`
```python
def run_validation(self, subset_name: str, run_both_modes: bool = True) -> Dict[str, Any]:
    """
    Run validation in one or both modes.
    
    Args:
        subset_name: Validation subset to use ("technical", "scientific", "comprehensive")
        run_both_modes: Whether to run both test and train modes
            
    Returns:
        Dictionary with validation results
    """
```

#### `run_test_equivalent_mode`
```python
def run_test_equivalent_mode(self, subset_name: str) -> Dict[str, Any]:
    """
    Run validation using only test-available features.
    
    Args:
        subset_name: Validation subset name
            
    Returns:
        Dictionary with validation results
    """
```

#### `run_training_equivalent_mode`
```python
def run_training_equivalent_mode(self, subset_name: str) -> Dict[str, Any]:
    """
    Run validation using all training features (including pseudo-dihedrals).
    
    Args:
        subset_name: Validation subset name
            
    Returns:
        Dictionary with validation results
    """
```

#### `analyze_mode_differences`
```python
def analyze_mode_differences(self, 
                            test_results: Dict[str, Any], 
                            train_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze differences between test and training modes.
    
    Args:
        test_results: Results from test-equivalent mode
        train_results: Results from training-equivalent mode
            
    Returns:
        Dictionary with analysis metrics
    """
```

### Configuration Options

The ValidationRunner accepts a configuration dictionary with the following options:

```python
default_config = {
    "batch_size": 4,  # Default batch size
    "num_workers": 2, # Workers for data loading
    "max_targets": None, # Maximum number of targets to use (None = use tier default)
    "seed": 42,       # Random seed for reproducibility
    "verbose": True,  # Whether to show progress bars
    "max_sequence_length": 512, # Maximum sequence length for memory management
    "metrics": ["rmsd", "tm_score"], # Metrics to compute
    "save_results": True, # Whether to save results
    "results_dir": None, # Directory for saving results
    "image_format": "png", # Format for saving images
}
```

### Return Value Structure

The `run_validation` method returns a dictionary with the following structure:

```python
{
    "timestamp": "2025-04-21 12:34:56",  # Time of validation run
    "subset_name": "technical",  # Validation subset used
    "configuration": {
        # Configuration dictionary
    },
    "test_mode": {  # Results from test-equivalent mode
        "mode": "test_equivalent",
        "num_samples": 5,  # Number of samples processed
        "target_ids": ["id1", "id2", ...],  # List of target IDs
        "mean_rmsd": 3.456,  # Average RMSD value
        "median_rmsd": 3.123,  # Median RMSD value
        "min_rmsd": 1.234,  # Minimum RMSD value
        "max_rmsd": 5.678,  # Maximum RMSD value
        "rmsd_values": [3.1, 4.2, ...],  # List of all RMSD values
        "mean_tm_score": 0.763,  # Average TM-score (if computed)
        "tm_scores": [0.8, 0.7, ...],  # List of all TM-scores (if computed)
        "avg_per_residue_error": [...],  # Per-residue error data
        "evaluation_time_seconds": 12.34,  # Time taken for evaluation
        "mean_sequence_length": 124.5,  # Average sequence length
        "problematic_samples": [  # List of problematic samples
            {
                "id": "sample_id",
                "issue": "extreme_coordinates",
                "details": "Error description",
                "seq_len": 120,
                "pred_min": -1.5,
                "pred_max": 3.2,
                "true_min": -1e9,
                "true_max": 24.5
            }
        ]
    },
    "train_mode": {
        # Same structure as test_mode
    },
    "analysis": {  # Only present if both modes were run
        "rmsd": {
            "test_value": 3.456,
            "train_value": 3.123,
            "absolute_difference": 0.333,
            "relative_difference_percent": 10.67,
            "impact": "NEGATIVE",  # POSITIVE, NEGATIVE, or NEUTRAL
            "interpretation": "Missing dihedral features degrades performance"
        },
        "tm_score": {
            # Similar structure to rmsd
        },
        "per_residue": {
            "max_difference_position": 0.75,  # Position with maximum difference
            "max_difference_value": 1.234,  # Value of maximum difference
            "position_category": "end"  # end or middle
        },
        "conclusion": {
            "overall_impact": "NEGATIVE",  # POSITIVE, NEGATIVE, or NEUTRAL
            "severity": "MODERATE",  # MAJOR, MODERATE, MINOR, or NEGLIGIBLE
            "recommendation": "Consider using feature prediction to improve test performance."
        }
    }
}
```

## Usage Examples

### Basic Usage

```python
from validation.validation_runner import ValidationRunner
from src.models.rna_folding_model import RNAFoldingModel

# Initialize model and validator
model = RNAFoldingModel()
runner = ValidationRunner(
    model=model,
    data_dir="/path/to/data",
    config={"batch_size": 8, "max_targets": 5}
)

# Run validation in both modes
results = runner.run_validation("technical", run_both_modes=True)

# Print summary results
print(f"Test mode RMSD: {results['test_mode']['mean_rmsd']:.4f}")
print(f"Train mode RMSD: {results['train_mode']['mean_rmsd']:.4f}")
if "analysis" in results:
    print(f"Impact: {results['analysis']['conclusion']['overall_impact']}")
    print(f"Severity: {results['analysis']['conclusion']['severity']}")
```

### Running Only Test Mode

```python
# Run validation in test mode only
test_results = runner.run_validation("technical", run_both_modes=False)
# Or alternatively:
test_results = runner.run_test_equivalent_mode("technical")
```

## Error Handling

The ValidationRunner performs extensive error handling to ensure robustness:

1. For extreme coordinate values (outside ±500Å range), the sample is skipped and reported in `problematic_samples`
2. For invalid RMSD values (>100Å), the value is capped and reported
3. For numerical instabilities, appropriate messages are logged and reported
4. For out-of-memory errors during batch processing, batch size is automatically reduced

## Change Log

- **v1.0 (2025-04-21)**
  - Initial implementation
  - Added comprehensive error handling
  - Added problematic sample tracking
  - Added detailed reporting in markdown and JSON formats