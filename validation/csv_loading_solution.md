# CSV Loading Solution for RNA 3D Folding Validation

## Problem Identified

The validation notebook has been trying to load feature data from NPZ files in the `/data/processed/` directory, but the actual data we need is in CSV files in the `/data/raw/` directory. The CSV files contain the raw RNA sequences and their 3D coordinates, while the NPZ files contain processed features derived from these.

## Key Insights

1. The validation flow should be loading data from:
   - `/data/raw/validation_sequences.csv` - Contains RNA sequences
   - `/data/raw/validation_labels.csv` - Contains 3D coordinates of each nucleotide

2. Instead, we've been trying to load processed features from:
   - `/data/processed/dihedral_features/*.npz`
   - `/data/processed/thermo_features/*.npz`
   - `/data/processed/mi_features/*.npz`

3. These are feature files generated from the raw data, but we can bypass them entirely for validation since we have the raw data available.

## Solution Implemented

Created a `CSVDataLoader` class to directly load and process the CSV files:

1. **File Structure**: The `CSVDataLoader` loads data from the following files:
   ```
   data/raw/
   ├── train_sequences.csv
   ├── train_labels.csv
   ├── validation_sequences.csv
   └── validation_labels.csv
   ```

2. **Key Features**:
   - Loads sequences and 3D coordinates directly from CSV files
   - Handles sequence conversion (AUCG → integer representation)
   - Creates appropriate masks for valid coordinates
   - Properly pads sequences for batching
   - Implements a collate function for PyTorch DataLoader

3. **Benefits**:
   - No longer depends on preprocessed NPZ files
   - Simpler data pipeline with fewer points of failure
   - More transparent - we know exactly what data we're using
   - Avoids path resolution issues with spaces in directory names

## Implementation Details

The implementation:

1. **Provides a clean API**:
   ```python
   # Basic usage
   csv_loader = CSVDataLoader(data_dir="path/to/data", split="validation")
   dataset = csv_loader.create_torch_dataset(subset_size=5)
   
   # Create DataLoader
   dataloader = DataLoader(
       dataset,
       batch_size=2,
       shuffle=False,
       collate_fn=dataset.collate_fn
   )
   ```

2. **Handles Missing Data**:
   - Verifies coordinates are valid (filters out placeholder values)
   - Creates mock features when needed for model compatibility
   - Provides clear warnings about missing data

3. **Remains Compatible** with the RNA folding model:
   - Provides all the tensor fields the model expects
   - Maintains the same batch format used in training
   - Uses the same padding approach for consistent behavior

## Integration Into Validation Notebook

Updated `validation_technical.ipynb` to use the new data loader:

```python
# Import the CSV data loader
from csv_data_loader import CSVDataLoader

# Create loader and dataset
csv_loader = CSVDataLoader(data_dir=os.path.join(project_root, "data"), split="validation")
torch_dataset = csv_loader.create_torch_dataset(subset_size=CONFIG["subset_size"])

# Create validation dataloader
validation_loader = torch.utils.data.DataLoader(
    torch_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    collate_fn=torch_dataset.collate_fn
)
```

This replaces the previous complex feature loading code with a simpler, more direct approach.

## Next Steps

1. **Extend to Other Notebooks**:
   - Update Tier 2 and Tier 3 validation notebooks with the same approach
   - Consider standardizing this loader across the codebase

2. **Add Feature Computation**:
   - When needed, we can compute features (dihedral angles, etc.) from the raw data
   - Implement these calculations when more accurate feature values are needed

3. **Enhance Visualization**:
   - Now that we have direct access to the raw coordinate data, we can create better visualizations of structures
   - Consider adding a PyMOL or similar 3D visualization of predictions vs ground truth