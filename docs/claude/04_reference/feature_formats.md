# RNA Feature Formats Reference Guide

This document provides a comprehensive reference for the precomputed feature files used throughout the RNA 3D folding pipeline. It covers file formats, content structure, loading patterns, and usage in different pipeline stages.

## Overview of Feature Types

The RNA 3D folding pipeline uses three primary types of precomputed features:

1. **Dihedral Features**: Backbone geometry information in the form of pseudo-dihedral angles
2. **Thermodynamic Features**: RNA folding energetics and secondary structure probabilities
3. **Evolutionary Features (MI)**: Evolutionary coupling information derived from multiple sequence alignments

## File Organization and Naming Conventions

### Directory Structure
```
data/processed/
├── dihedral_features/
│   ├── {target_id}_dihedral_features.npz
│   └── ...
├── thermo_features/
│   ├── {target_id}_thermo_features.npz
│   └── ...
└── mi_features/
    ├── {target_id}_mi_features.npz
    └── ...
```

### Naming Pattern
- Dihedral: `{target_id}_dihedral_features.npz`
- Thermo: `{target_id}_thermo_features.npz`
- Evolutionary MI: `{target_id}_mi_features.npz` (within `mi_features/` directory)

## 1. Dihedral Feature Files

### Purpose
Dihedral features represent the RNA backbone geometry through pseudo-dihedral angles. They provide crucial geometric information for training and validation but are not available for test sequences.

### Key Arrays

| Array Name | Shape | Data Type | Description |
|------------|-------|-----------|-------------|
| `features` | `(N, 4)` | `float32` | sin/cos transformations of η and θ angles |
| `eta` | `(N,)` | `float32` | Raw eta (η) angles in degrees |
| `theta` | `(N,)` | `float32` | Raw theta (θ) angles in degrees |
| `feature_names` | `(4,)` | `string` | Column names: ['eta_sin', 'eta_cos', 'theta_sin', 'theta_cos'] |
| `metadata` | scalar | `string` | JSON-formatted metadata about the features |

### Data Format Details

- **Primary ML Input**: The `features` array is used as the primary input, containing:
  - Column 0: `eta_sin` - sine of eta angle
  - Column 1: `eta_cos` - cosine of eta angle
  - Column 2: `theta_sin` - sine of theta angle
  - Column 3: `theta_cos` - cosine of theta angle

- **Value Range**: All values in `features` are in the range [-1, 1]

- **Boundary Residues**: The first and last few residues may have NaN values (or zeros) as pseudo-dihedrals require 4 consecutive positions

### Example Data
```python
# features array - shape (20, 4)
features = [
 [0, 0, 0, 0],  # First residue (boundary)
 [0.3165691903994371, 0.9485694216502264, 0.38655602048917104, 0.9222659285821935],
 [0.38655602048917104, 0.9222659285821935, 0.19483134239692543, 0.9808367591091863],
 # ... more residues
 [0, 0, 0, 0]   # Last residue (boundary)
]

# Raw angle arrays
eta = [NaN, 18.455570404571613, 22.740373940772656, ...]  # shape (20,)
theta = [NaN, 22.740373940772656, 11.234871255751486, ...]  # shape (20,)

# Feature names
feature_names = ['eta_sin', 'eta_cos', 'theta_sin', 'theta_cos']

# Metadata
metadata = '{"feature_names": ["eta_sin", "eta_cos", "theta_sin", "theta_cos"], 
             "feature_description": "Pseudo-dihedral angle features in sin/cos encoding",
             "extraction_timestamp": "2025-04-13 00:10:44"}'
```

### Loading Pattern
```python
def load_dihedral_features(target_id: str, features_dir: str) -> Dict[str, np.ndarray]:
    """Load dihedral features from npz file."""
    file_path = os.path.join(features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
    
    if not os.path.exists(file_path):
        # Handle missing file (for test data or if extraction failed)
        warnings.warn(f"Dihedral features not found for {target_id}. Using zeros.")
        # You'll need sequence length from elsewhere to determine shape
        seq_len = get_sequence_length(target_id, features_dir)
        return {
            'features': np.zeros((seq_len, 4), dtype=np.float32)
        }
    
    with np.load(file_path) as data:
        # Extract the features array (primary ML input)
        features = data['features']
        
        # Convert NaN values to zeros if present
        if np.isnan(features).any():
            features = np.nan_to_num(features, nan=0.0)
        
        # Optional: extract raw angles if needed
        # eta = data.get('eta', None)
        # theta = data.get('theta', None)
        
        return {
            'features': features.astype(np.float32)
        }
```

### Usage in Architecture
- In V1, dihedral features are primarily used for **auxiliary supervision** during training
- They feed into the `angle_prediction_head` for multi-task learning
- They are NOT used as direct inputs during inference/prediction time
- The sin/cos representation avoids the periodic boundary problem with raw angles

## 2. Thermodynamic Feature Files

### Purpose
Thermodynamic features capture RNA folding energetics and secondary structure probabilities. They provide critical information about likely base pairings and structural stability.

### Key Arrays

#### Scalar Features (Global Properties)

| Array Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `mfe` | `float32` | Minimum Free Energy (kcal/mol) | -13.60 |
| `ensemble_energy` | `float32` | Free energy of the ensemble | -13.59 |
| `energy_gap` | `float32` | Difference between MFE and ensemble | 0.01 |
| `mfe_probability` | `float32` | Boltzmann probability of MFE structure | 0.974 |
| `gc_content` | `float32` | Fraction of G-C pairs in sequence | 0.65 |
| `paired_fraction` | `float32` | Fraction of paired nucleotides in MFE | 0.8 |
| `avg_stem_length` | `float32` | Average stem length | 8.0 |
| `free_energy_per_nucleotide` | `float32` | MFE normalized by sequence length | -0.68 |

#### Vector Features (Per-Residue)

| Array Name | Shape | Description | Value Range |
|------------|-------|-------------|-------------|
| `positional_entropy` | `(N,)` | Shannon entropy at each position | [0, log₂(4)] |
| `accessibility` | `(N,)` | Unpaired probability per nucleotide | [0, 1] |
| `sequence` | `string` | The RNA sequence | "GGACUAGCG..." |

#### Matrix Features (Pairwise)

| Array Name | Shape | Description | Properties |
|------------|-------|-------------|------------|
| `pairing_probs` | `(N, N)` | Base pair probabilities | Symmetric, values in [0, 1] |
| `base_pair_probs` | `(N, N)` | Alias for `pairing_probs` | Same as above |

#### Other Features

| Array Name | Type | Description |
|------------|------|-------------|
| `structure` or `mfe_structure` | `string` | Dot-bracket notation of MFE structure |
| `target_id` | `string` | Sequence ID |
| `processing_timestamp` | `string` | When features were computed |

### Example Data
```python
# Scalar features
mfe = -13.600000381469727
ensemble_energy = -13.590000381469727
mfe_probability = 0.974221294229328
gc_content = 0.65

# Vector features (shape = 20 for RNA of length 20)
positional_entropy = [0.9218817563464707, 0.8558744804008804, 0.8477366983300982, ...]
accessibility = [0.15, 0.14, 0.15, 0.18, 0.11, 0.10, 0.09, 0.08, 0.98, 0.99, ...]

# Matrix features (shape = 20x20)
pairing_probs = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0.02, 0.01, ...],
    [0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0.03, ...],
    # ... more rows
]

# Sequence
sequence = 'GGACUAGCGGAGGCUAGUCC'

# Structure
mfe_structure = '((((((((....))))))))'
```

### Loading Pattern
```python
def load_thermo_features(target_id: str, features_dir: str) -> Dict[str, np.ndarray]:
    """Load thermodynamic features from npz file."""
    file_path = os.path.join(features_dir, "thermo_features", f"{target_id}_thermo_features.npz")
    
    if not os.path.exists(file_path):
        raise ValueError(f"Thermodynamic features not found for {target_id}. These are required.")
    
    with np.load(file_path) as data:
        # Extract key arrays
        features = {}
        
        # Extract scalar features
        scalar_features = ['mfe', 'ensemble_energy', 'mfe_probability', 
                          'gc_content', 'paired_fraction']
        for key in scalar_features:
            if key in data:
                features[key] = float(data[key])
        
        # Extract vector features
        vector_features = ['positional_entropy', 'accessibility']
        for key in vector_features:
            # Handle potential key aliases
            actual_key = key
            if key == 'positional_entropy' and 'position_entropy' in data:
                actual_key = 'position_entropy'
            
            if actual_key in data:
                features[key] = data[actual_key].astype(np.float32)
        
        # Extract matrix features - CRITICAL for model
        if 'pairing_probs' in data:
            features['pairing_probs'] = data['pairing_probs'].astype(np.float32)
        elif 'base_pair_probs' in data:
            features['pairing_probs'] = data['base_pair_probs'].astype(np.float32)
        else:
            raise ValueError(f"No pairing probability matrix found for {target_id}")
        
        # Extract sequence if available
        if 'sequence' in data:
            features['sequence'] = str(data['sequence'])
        
        return features
```

### Usage in Architecture
- **Pairing Probabilities Matrix**: Critical input for the pair representation module; captures likely RNA secondary structure
- **Positional Entropy**: Important per-residue feature indicating local structural uncertainty
- **Accessibility**: Used as per-residue feature showing unpaired probability
- **Global Scalars**: Often used for conditioning either by:
  1. Broadcasting to all residues (e.g., `mfe` → tensor of shape (N,) with same value)
  2. Feeding into a global conditioning network

## 3. Evolutionary Coupling Features (MI)

### Purpose
Evolutionary coupling features (MI) capture co-evolutionary signals derived from multiple sequence alignments (MSAs). These indicate pairs of positions that may be in spatial proximity in the folded structure.

### Key Arrays

| Array Name | Shape | Data Type | Description |
|------------|-------|-----------|-------------|
| `coupling_matrix` | `(N, N)` | `float32` | Mutual information scores between residue pairs |
| `method` | scalar | `string` | Method used for MI calculation, e.g., "mutual_information" |
| `sequence_count` | scalar | `int` | Number of sequences in the MSA |
| `top_pairs` | `(P, 3)` | `float32` | Top P coupling pairs: [i, j, score] |
| `score_distance_correlation` | scalar | `float32` | Correlation between MI scores and distances |

### Example Data
```python
# Coupling matrix (shape = 20x20 for RNA of length 20)
coupling_matrix = [
    [0, 0.6212779434913056, 0.6212779434913056, ...],
    [0.6212779434913056, 0, 0.6212779434913056, ...],
    # ... more rows
]

method = 'mutual_information'

# Top coupling pairs [pos_i, pos_j, score]
top_pairs = [
    [0, 4, 0.6212779434913057],
    [0, 14, 0.6212779434913057],
    [0, 17, 0.6212779434913057],
    # ... more pairs
]

score_distance_correlation = 0.010928879659172563
```

### Important Considerations
- **Availability**: Not all sequences will have evolutionary features (depends on MSA availability)
- **Missing Data Handling**: Must be robust to missing files
- **Matrix Symmetry**: The coupling matrix should be symmetric (`coupling_matrix[i,j] == coupling_matrix[j,i]`)

### Loading Pattern
```python
def load_evolutionary_features(target_id: str, features_dir: str) -> Dict[str, np.ndarray]:
    """Load evolutionary coupling features from npz file."""
    file_path = os.path.join(features_dir, "mi_features", f"{target_id}_mi_features.npz")
    
    if not os.path.exists(file_path):
        warnings.warn(f"Evolutionary features not found for {target_id}. Using zeros.")
        # You'll need sequence length from elsewhere to determine shape
        seq_len = get_sequence_length(target_id, features_dir)
        return {
            'coupling_matrix': np.zeros((seq_len, seq_len), dtype=np.float32),
            'conservation': np.zeros(seq_len, dtype=np.float32) if include_conservation else None
        }
    
    with np.load(file_path) as data:
        features = {}
        
        # Extract coupling matrix (primary input)
        if 'coupling_matrix' in data:
            features['coupling_matrix'] = data['coupling_matrix'].astype(np.float32)
        else:
            raise ValueError(f"No coupling matrix found for {target_id}")
        
        # Extract method if available
        if 'method' in data:
            features['method'] = str(data['method'])
        
        # Extract top pairs if available
        if 'top_pairs' in data:
            features['top_pairs'] = data['top_pairs'].astype(np.float32)
        
        return features
```

### Usage in Architecture
- **Primary Input**: The `coupling_matrix` is a critical component for the pair representation module
- It directly influences the pair embeddings and pair updates in the transformer backbone
- The matrix captures likely spatial relationships between residues

## Integration in Data Loading

### Combined Loading Function

Here's a comprehensive function that loads all feature types and combines them into a single dictionary:

```python
def load_precomputed_features(target_id: str, features_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all precomputed features for a given target ID.
    
    Args:
        target_id: Target RNA identifier
        features_dir: Root directory for processed features
        
    Returns:
        Dictionary of feature dictionaries:
        {
            'dihedral': {'features': array(...)},
            'thermo': {
                'pairing_probs': array(...),
                'positional_entropy': array(...),
                ...
            },
            'evolutionary': {'coupling_matrix': array(...)}
        }
    """
    features = {}
    
    # Get sequence length (needed for creating default tensors)
    seq_len = get_sequence_length(target_id, features_dir)
    
    # 1. Load dihedral features (if available)
    try:
        features['dihedral'] = load_dihedral_features(target_id, features_dir)
    except Exception as e:
        warnings.warn(f"Error loading dihedral features for {target_id}: {e}")
        features['dihedral'] = {
            'features': np.zeros((seq_len, 4), dtype=np.float32)
        }
    
    # 2. Load thermodynamic features (required)
    try:
        features['thermo'] = load_thermo_features(target_id, features_dir)
    except Exception as e:
        raise ValueError(f"Failed to load thermodynamic features for {target_id}: {e}")
    
    # 3. Load evolutionary features (if available)
    try:
        features['evolutionary'] = load_evolutionary_features(target_id, features_dir)
    except Exception as e:
        warnings.warn(f"Error loading evolutionary features for {target_id}: {e}")
        features['evolutionary'] = {
            'coupling_matrix': np.zeros((seq_len, seq_len), dtype=np.float32)
        }
    
    return features
```

### Usage in Dataset Class
```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """Get a sample from the dataset."""
    target_id = self.target_ids[idx]
    
    # Load features
    features = load_precomputed_features(target_id, self.features_dir)
    
    # Convert to tensors
    sample = {
        'target_id': target_id,
        'sequence_int': torch.tensor(self.sequence_to_int(self.sequences[idx]), dtype=torch.long),
        'dihedral_features': torch.tensor(features['dihedral']['features'], dtype=torch.float32),
        'pairing_probs': torch.tensor(features['thermo']['pairing_probs'], dtype=torch.float32),
        'positional_entropy': torch.tensor(features['thermo']['positional_entropy'], dtype=torch.float32),
        'coupling_matrix': torch.tensor(features['evolutionary']['coupling_matrix'], dtype=torch.float32),
        'coordinates': torch.tensor(self.coordinates[target_id], dtype=torch.float32) if target_id in self.coordinates else None,
        'length': len(self.sequences[idx])
    }
    
    return sample
```

## Key Integration Considerations

### 1. Consistent Shape Handling
- Always check for shape consistency between different feature types
- For a given `target_id`, all feature arrays should have compatible shapes based on sequence length
- Ensure proper error messages when shape inconsistencies are detected

### 2. Missing Feature Handling
- **Dihedral features**: May be missing for test data; use zeros of appropriate shape
- **Thermodynamic features**: Required for all sequences; raise error if missing
- **Evolutionary features**: May be missing if no MSA available; use zeros of appropriate shape

### 3. Feature Transformation
- When processing to tensors, ensure correct data types (`torch.float32` for features)
- Consider normalizing features to similar ranges if not already in comparable ranges
- Handle NaN values consistently (typically by replacing with zeros)

### 4. Pipeline Integration
- **Data Loading**: These features are first processed by the `RNADataset` class
- **Embedding**: Features feed into component-specific embedding layers
- **Fusion**: The transformer backbone integrates information from all feature types
- **Structure Generation**: Final coordinates are predicted based on fused representations

## Reference Example Files

These file patterns are available in our project for testing and validation:
- `{target_id}_dihedral_features.npz` - Dihedral angle features
- `{target_id}_thermo_features.npz` - Thermodynamic features
- `{target_id}_mi_features.npz` - Mutual information/evolutionary coupling features

Use these files to validate your loading functions and verify expected array shapes and types.
