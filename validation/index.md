# RNA 3D Folding Validation Framework

This directory contains the validation framework for the RNA 3D folding model. The framework includes tools for technical validation, scientific validation, and benchmark testing.

## Directory Structure

```
validation/
├── metrics/                      # Scientific validation metrics
│   ├── feature_importance/       # Feature importance analysis metrics
│   ├── rna_family/               # RNA family classification metrics
│   └── secondary_structure/      # Secondary structure evaluation metrics
├── results/                      # Shared validation results
├── rmsd_benchmark/               # RMSD validation against RNA-Puzzles
│   ├── predictions/              # Prediction models for each puzzle
│   ├── published_rmsd/           # Published RMSD reference values
│   ├── reference/                # Reference PDB structures
│   ├── results/                  # Validation results and reports
│   └── scripts/                  # RMSD validation scripts
├── tier1_technical/              # Technical validation framework
│   └── results/                  # Technical validation results
├── tier2_scientific/             # Scientific validation framework
│   └── results/                  # Scientific validation results
└── tier3_comprehensive/          # Comprehensive validation (combined)
    └── results/                  # Comprehensive validation results
```

## Key Components

### 1. Core Validation Files

| File | Description |
|------|-------------|
| [README.md](README.md) | Main validation framework documentation |
| [validation_runner.py](validation_runner.py) | Central validation runner implementation |
| [validation_dataset.py](validation_dataset.py) | Dataset handling for validation |
| [run_dual_mode_validation.sh](run_dual_mode_validation.sh) | Shell script to run the dual-mode validation |

### 2. Technical Documentation

| File | Description |
|------|-------------|
| [dual_mode_validation_README.md](dual_mode_validation_README.md) | Documentation for dual-mode validation approach |
| [dual_mode_implementation_summary.md](dual_mode_implementation_summary.md) | Implementation summary of dual-mode validation |
| [rmsd_calculation_improvements.md](rmsd_calculation_improvements.md) | RMSD calculation improvements including MDAnalysis integration |
| [validation_status_update_v2.md](validation_status_update_v2.md) | Latest validation status update |

### 3. Validation Frameworks

#### Technical Validation (Tier 1)

Located in [tier1_technical/](tier1_technical/), focuses on model correctness and numerical stability.

| File | Description |
|------|-------------|
| [debug_rmsd_calculation.py](tier1_technical/debug_rmsd_calculation.py) | Troubleshooting RMSD calculation issues |
| [debug_validation_runner.py](tier1_technical/debug_validation_runner.py) | Debugging validation runner |
| [run_validation.py](tier1_technical/run_validation.py) | Run technical validation |

#### Scientific Validation (Tier 2)

Located in [tier2_scientific/](tier2_scientific/), focuses on scientific relevance and accuracy.

| File | Description |
|------|-------------|
| [run_scientific_validation.py](tier2_scientific/run_scientific_validation.py) | Run scientific validation |
| [run_scientific_validation.sh](tier2_scientific/run_scientific_validation.sh) | Shell script for scientific validation |
| [validation_scientific_notebook.py](tier2_scientific/validation_scientific_notebook.py) | Notebook for scientific validation |

### 4. RMSD Benchmark

The [rmsd_benchmark/](rmsd_benchmark/) directory contains validation against RNA-Puzzles.

| File | Description |
|------|-------------|
| [rmsd_validator.py](rmsd_benchmark/scripts/rmsd_validator.py) | RMSD validation against analytical tests and RNA-Puzzles |
| [test_c1_vs_all_atom.py](rmsd_benchmark/scripts/test_c1_vs_all_atom.py) | Compare atom selection strategies |
| [run_validation.sh](rmsd_benchmark/scripts/run_validation.sh) | Run RMSD validation |

### 5. Metrics

The [metrics/](metrics/) directory contains implementations of scientific validation metrics.

| Module | Description |
|--------|-------------|
| [feature_importance](metrics/feature_importance/) | Analysis of feature importance |
| [rna_family](metrics/rna_family/) | RNA family classification metrics |
| [secondary_structure](metrics/secondary_structure/) | Secondary structure evaluation |

## Recent Improvements

1. **MDAnalysis Integration**: Enhanced RMSD calculation using MDAnalysis library's QCP algorithm
2. **Dual-Mode Validation**: Implemented framework for both feature-based and coordinate-based validation
3. **Scientific Metrics**: Added RNA family and secondary structure evaluation metrics
4. **RMSD Benchmark**: Added validation against RNA-Puzzles reference structures

## Usage

### Running Validation

```bash
# Activate environment
eval "$(mamba shell hook --shell bash)" && mamba activate rna-3d-folding

# Run dual-mode validation
./validation/run_dual_mode_validation.sh

# Run RMSD benchmark
cd validation/rmsd_benchmark/scripts
./run_validation.sh
```

### Viewing Results

Validation results are stored in the respective `results/` directories:

- Technical validation: `validation/tier1_technical/results/`
- Scientific validation: `validation/tier2_scientific/results/`
- RMSD benchmark: `validation/rmsd_benchmark/results/`

## Environment Requirements

- Python 3.10+
- PyTorch 2.1+
- MDAnalysis (for robust RMSD calculation)
- matplotlib, seaborn (for visualization)
- numpy, scipy, pandas (for data analysis)