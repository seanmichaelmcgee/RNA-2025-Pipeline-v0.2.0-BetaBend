# RNA Feature Formats Reference Guide

This document provides a comprehensive reference for the precomputed feature files used throughout the RNA 3D folding pipeline. It covers file formats, content structure, loading patterns, and usage in different pipeline stages.

## Overview of Feature Types

The RNA 3D folding pipeline uses three primary types of precomputed features:

1. **Dihedral Features**: Backbone geometry information in the form of pseudo-dihedral angles
2. **Thermodynamic Features**: RNA folding energetics and secondary structure probabilities
3. **Evolutionary Features**: Evolutionary coupling information derived from multiple sequence alignments

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
    ├── {target_id}_features.npz
    └── ...
```

### Naming Pattern
- Dihedral: `{target_id}_dihedral_features.npz`
- Thermo: `{target_id}_thermo_features.npz`
- Evolutionary MI: `{target_id}_features.npz` (within `mi_features/`)

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

### Example Data (from 1A51_A_dihedral_features.npz)
```python
# features array - shape (41, 4)
features = [
 [0, 0, 0, 0],  # First residue (boundary)
 [-0.3710391936512465, 0.9286172068051683, 0.7956411540985328, 0.6057682344797842],
 [0.7956411540985328, 0.6057682344797842, 0.41479551457531194, -0.9099146559365896],
 # ... more residues
 [0, 0, 0, 0]   # Last residue (boundary)
]

# Raw angle arrays
eta = [NaN, -21.77972132590914, 52.71585626826523, ...]  # shape (41,)
theta = [NaN, 52.71585626826523, 155.49356027602423, ...]  # shape (41,)
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
| `mfe` | `float32` | Minimum Free Energy (kcal/mol) | -19.39 |
| `ensemble_energy` | `float32` | Free energy of the ensemble | -19.38 |
| `energy_gap` | `float32` | Difference between MFE and ensemble | 0.01 |
| `mfe_probability` | `float32` | Boltzmann probability of MFE structure | 0.406 |
| `gc_content` | `float32` | Fraction of G-C pairs in sequence | 0.658 |
| `paired_fraction` | `float32` | Fraction of paired nucleotides in MFE | 0.682 |
| `avg_pair_distance_mean` | `float32` | Mean base pairing distance | 16.21 |
| `free_energy_per_nucleotide` | `float32` | MFE normalized by sequence length | -0.473 |

#### Vector Features (Per-Residue)

| Array Name | Shape | Description | Value Range |
|------------|-------|-------------|-------------|
| `positional_entropy` | `(N,)` | Shannon entropy at each position | [0, log2(4)] |
| `position_entropy` | `(N,)` | Alias for `positional_entropy` | [0, log2(4)] |
| `accessibility` | `(N,)` | Unpaired probability per nucleotide | [0, 1] |
| `sequence` | `string` | The RNA sequence | "GGCCGA..." |

#### Matrix Features (Pairwise)

| Array Name | Shape | Description | Properties |
|------------|-------|-------------|------------|
| `pairing_probs` | `(N, N)` | Base pair probabilities | Symmetric, values in [0, 1] |
| `base_pair_probs` | `(N, N)` | Alias for `pairing_probs` | Same as above |

#### Other Features

| Array Name | Type | Description |
|------------|------|-------------|
| `structure` or `mfe_structure` | `string` | Dot-bracket notation of MFE structure |
| `seq_id` | `string` | Sequence ID |
| `processing_timestamp` | `string` | When features were computed |

### Example Data (from 1A51_A_thermo_features.npz)
```python
# Scalar features
mfe = -19.399999618530273
ensemble_energy = -19.389999618530272
mfe_probability = 0.4068769160792827
gc_content = 0.6585365853658537

# Vector features (shape = 41 for RNA of length 41)
positional_entropy = [0.9873037385331406, 0.9584614333282193, 0.9043900159501144, ...]
accessibility = [0.03810113393677772, 0.043191132112834096, 0.04019611206075213, ...]

# Matrix features (shape = 41x41)
pairing_probs = [
    [0, 0, 0, 0, 0.0239, ...],
    [0, 0, 0, 0, 0.0233, ...],
    # ... more rows
]

# Sequence
sequence = 'GGCCGAUGGUAGUGUGGGGUCUCCCCAUGCGAGAGUAGGCC'

# Structure
mfe_structure = '((((.((....((((((((...))))))))....)).))))'
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

## 3. Evolutionary Coupling Features

### Purpose
Evolutionary coupling features capture co-evolutionary signals derived from multiple sequence alignments (MSAs). These indicate pairs of positions that may be in spatial proximity in the folded structure.

### Key Arrays

| Array Name | Shape | Data Type | Description |
|------------|-------|-----------|-------------|
| `coupling_matrix` | `(N, N)` | `float32` | Mutual information scores between residue pairs |
| `method` | scalar | `string` | Method used for MI calculation, e.g., "mutual_information_enhanced" |
| `sequence_count` | scalar | `int` | Number of sequences in the MSA |
| `sequence_length` | scalar | `int` | Length of the target sequence |
| `conservation` | `(N,)` | `float32` | Per-position conservation scores (optional) |

### Example Data (from 1A51_A_features.npz)
```python
# Coupling matrix (shape = 40x40 for RNA of length 40)
coupling_matrix = [
    [0.00004353, 0.00150399, 0.00656933, ...],
    [0.00150399, 0.01123879, 0.02858577, ...],
    # ... more rows
]

method = 'mutual_information_enhanced'
sequence_count = 3000
sequence_length = 40
```

### Important Considerations
- **Availability**: Not all sequences will have evolutionary features (depends on MSA availability)
- **Missing Data Handling**: Must be robust to missing files
- **Matrix Symmetry**: The coupling matrix should be symmetric (`coupling_matrix[i,j] == coupling_matrix[j,i]`)

### Loading Pattern
```python
def load_evolutionary_features(target_id: str, features_dir: str) -> Dict[str, np.ndarray]:
    """Load evolutionary coupling features from npz file."""
    file_path = os.path.join(features_dir, "mi_features", f"{target_id}_features.npz")
    
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
        
        # Extract conservation values if needed
        if 'conservation' in data:
            features['conservation'] = data['conservation'].astype(np.float32)
        
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
        'coordinates': torch.tensor(self.coordinates[idx], dtype=torch.float32) if self.coordinates is not None else None,
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

These real examples are available in the project for testing and validation:
- `1A51_A_dihedral_features.npz.txt`
- `1A51_A_thermo_features.npz.txt`
- `1A51_A_features.npz.txt`

Use these files to validate your loading functions and verify expected array shapes and types.
