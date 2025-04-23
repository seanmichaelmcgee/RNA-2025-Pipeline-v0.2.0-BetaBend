# RMSD Validation Benchmark

This directory contains tools and reference data for validating our RMSD calculation implementation against established standards. The validation framework compares our implementation with published RMSD values from RNA-Puzzles competitions.

## Directory Structure

```
rmsd_benchmark/
├── reference/        # Reference PDB structures from RNA-Puzzles
├── predictions/      # Prediction models for each puzzle
│   ├── puzzle5/      # Models for Puzzle 5
│   ├── puzzle6/      # Models for Puzzle 6
│   └── ...
├── published_rmsd/   # CSV files with published RMSD values
├── results/          # Validation results and reports
└── scripts/          # Processing and validation scripts
```

## Reference Structures

The `reference/` directory contains the experimental reference structures from RNA-Puzzles in PDB format:

| Puzzle | PDB ID | Name |
|--------|--------|------|
| 5 | 4P95 | Lariat-capping ribozyme |
| 6 | 4GXY | Adenosylcobalamin riboswitch |
| 10 | 4LCK | T-box-tRNA complex |
| 13 | 5KPY | 5-hydroxytryptophan riboswitch |
| 14 | 5DI4 | Hammerhead ribozyme |
| 15 | 5K7C | Pistol ribozyme |
| 16 | 5T5A | Twister sister ribozyme 1 |
| 17 | 5Y85 | Twister sister ribozyme 2 |
| 18 | 5NWQ | Guanidine-III riboswitch |

## Running the Validation

### Prerequisites

Before running validation, ensure you have:

1. Activated the RNA environment:
   ```bash
   mamba activate rna-3d-folding
   # or
   conda activate rna-3d-folding
   ```

2. Installed required packages:
   ```bash
   pip install requests tqdm
   ```

### Complete Validation Workflow

To run the complete validation process:

```bash
cd "/home/smcgee/MLprojects/RNA 2025/Pipeline-v1-March27on/RNA-2025-Pipeline-v0.2.0-BetaBend/"
./validation/rmsd_benchmark/scripts/run_complete_validation.sh
```

This script will:
1. Download real prediction models from RNA-Puzzles
2. Verify which models were successfully downloaded
3. Run analytical RMSD validation
4. Compare atom selection strategies (C1' vs P vs C4' vs all-heavy)

### Running Individual Components

You can also run individual validation components:

1. **Download prediction models**:
   ```bash
   ./validation/rmsd_benchmark/scripts/download_models.sh
   ```

2. **Verify downloaded models**:
   ```bash
   python validation/rmsd_benchmark/scripts/list_validated_pairs.py
   ```

3. **Run analytical validation**:
   ```bash
   ./validation/rmsd_benchmark/scripts/run_validation.sh --analytical-only
   ```

4. **Compare atom selection strategies**:
   ```bash
   ./validation/rmsd_benchmark/scripts/run_atom_comparison.sh --verified-only
   ```

## Validation Outputs

The validation generates:

1. `validated_pairs.txt`: List of puzzles with successfully downloaded models
2. `results/`: Analytical validation results
3. `results/atom_selection/`: Comparison of different atom selection strategies

## References

The validation is based on data from:

1. Cruz JA, Blanchet MF, Bujnicki JM, et al. RNA-Puzzles: a CASP-like evaluation of RNA three-dimensional structure prediction. RNA. 2012;18(4):610-625.
2. Miao Z, Adamiak RW, Blanchet MF, et al. RNA-Puzzles Round II: assessment of RNA structure prediction programs applied to three large RNA structures. RNA. 2015;21(6):1066-1084.
3. Miao Z, Adamiak RW, Antczak M, et al. RNA-Puzzles Round III: 3D RNA structure prediction of five riboswitches and one ribozyme. RNA. 2017;23(5):655-672.
4. Miao Z, Adamiak RW, Antczak M, et al. RNA-Puzzles Round IV: 3D structure predictions of four ribozymes and two aptamers. RNA. 2020;26(8):982-995.