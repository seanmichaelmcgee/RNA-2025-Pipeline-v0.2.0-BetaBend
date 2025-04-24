# Submission Format Conversion Plan

## Current vs. Expected Format Analysis

### Current Format (Our Submission)
- **Rows**: One row per target_id and model_id combination (5 rows per target)
- **Columns**: target_id, model_id, coordinates, confidence
- **Coordinate Storage**: JSON arrays with all residue coordinates for a single model
- **Example**: `R1107,0,"[[x1,y1,z1], [x2,y2,z2], ...]","[conf1, conf2, ...]"`

### Expected Format (Sample Submission)
- **Rows**: One row per residue in each target (e.g., R1107_1, R1107_2, etc.)
- **Columns**: ID, resname, resid, x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3, x_4, y_4, z_4, x_5, y_5, z_5
- **Coordinate Storage**: Each row contains the coordinates for that residue position across all 5 models
- **Example**: `R1107_1,G,1,x1_mod1,y1_mod1,z1_mod1,x1_mod2,y1_mod2,z1_mod2,...`

## Conversion Steps

1. **Load Data**:
   - Load the test sequences file to get RNA sequences and residue names
   - Load our current submission file with the predicted coordinates

2. **Parse Sequences**:
   - Extract the RNA sequence for each target_id
   - Create a mapping of residue position to residue name (A, C, G, U)

3. **Parse Coordinates**:
   - Load and parse the JSON coordinate arrays from our submission
   - Create a dictionary to store coordinates by target_id, model_id, and residue position

4. **Reformat Data**:
   - Initialize a new DataFrame with the expected columns
   - For each target_id and residue position:
     - Create a unique ID (target_id_position)
     - Get the residue name from the sequence
     - Get the coordinates for this residue from all 5 models
     - Combine into a single row with the expected format

5. **Validate Output**:
   - Ensure all residues have coordinates for all 5 models
   - Verify the total number of rows equals the total number of residues across all targets
   - Verify the column structure matches the sample submission

6. **Save Converted Submission**:
   - Write the converted DataFrame to a new CSV file

## Implementation Approach

```python
import pandas as pd
import numpy as np
import json

# 1. Load the data
test_sequences = pd.read_csv("data/raw/test_sequences.csv")
current_submission = pd.read_csv("submissions/submission_final_model_20250423-212450.csv")

# 2. Parse sequences to get residue names
sequence_dict = {}
for _, row in test_sequences.iterrows():
    target_id = row['target_id']
    sequence = row['sequence']
    sequence_dict[target_id] = sequence

# 3. Parse coordinates from current submission
coords_by_target_model_residue = {}
for _, row in current_submission.iterrows():
    target_id = row['target_id']
    model_id = row['model_id']
    
    # Parse coordinates JSON
    coords = json.loads(row['coordinates'])
    
    # Store the coordinates
    if target_id not in coords_by_target_model_residue:
        coords_by_target_model_residue[target_id] = {}
    
    coords_by_target_model_residue[target_id][model_id] = coords

# 4. Create the new submission format
new_rows = []

for target_id, sequence in sequence_dict.items():
    if target_id not in coords_by_target_model_residue:
        continue  # Skip if no predictions for this target
        
    for residue_idx, residue_name in enumerate(sequence):
        # 1-based residue indexing
        residue_pos = residue_idx + 1
        
        # Create the row ID
        row_id = f"{target_id}_{residue_pos}"
        
        # Initialize the row with basic info
        new_row = {
            'ID': row_id,
            'resname': residue_name,
            'resid': residue_pos
        }
        
        # Add coordinates for each model
        for model_id in range(5):
            if model_id not in coords_by_target_model_residue[target_id]:
                # If model missing, use zeros
                new_row[f'x_{model_id+1}'] = 0.0
                new_row[f'y_{model_id+1}'] = 0.0
                new_row[f'z_{model_id+1}'] = 0.0
            else:
                # Get coordinates for this residue from this model
                coords = coords_by_target_model_residue[target_id][model_id]
                
                if residue_idx < len(coords):
                    x, y, z = coords[residue_idx]
                    new_row[f'x_{model_id+1}'] = x
                    new_row[f'y_{model_id+1}'] = y
                    new_row[f'z_{model_id+1}'] = z
                else:
                    # Handle case where prediction is shorter than sequence
                    new_row[f'x_{model_id+1}'] = 0.0
                    new_row[f'y_{model_id+1}'] = 0.0
                    new_row[f'z_{model_id+1}'] = 0.0
        
        new_rows.append(new_row)

# Create and save the new submission dataframe
new_submission = pd.DataFrame(new_rows)
new_submission.to_csv("submissions/submission_reformatted.csv", index=False)
```

## Edge Cases to Handle

1. **Missing Models**: Ensure we handle cases where we might not have all 5 models for a target
2. **Sequence Length Mismatch**: Handle cases where the predicted coordinates array length doesn't match the sequence length
3. **Residue Name Case**: Ensure residue names are in the correct case (ACGU, not acgu)
4. **Coordinate Format**: Ensure the coordinates are formatted correctly (using appropriate decimal precision)
5. **ID Formatting**: Ensure the ID column follows the exact format from the sample submission

## Validation Checks

1. Verify the output has the same number of columns as the sample submission
2. Verify each target has the correct number of rows (equal to the sequence length)
3. Check that each residue has coordinates for all 5 models
4. Ensure residue names match the sequence data

## Testing Strategy

1. Convert a small subset of targets first to verify the format is correct
2. Compare the reformatted output to the sample submission to ensure structural matching
3. Process the full dataset after verification