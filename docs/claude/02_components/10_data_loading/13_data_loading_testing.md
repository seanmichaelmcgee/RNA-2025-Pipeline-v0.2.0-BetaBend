# Data Loading Testing Guide with MI Matrix Validation

This guide outlines the testing approach for the data loading component (`src/data_loading.py`) with a special focus on the new uniform MI matrix detection and handling. It covers unit tests, integration tests, and common test cases to ensure the component functions correctly and properly identifies invalid evolutionary information.

## Test Structure

Create a comprehensive test file `tests/test_data_loading.py` with these major test groups:

1. Helper Function Tests
2. RNADataset Tests
3. Collate Function Tests 
4. End-to-End Tests
5. Edge Cases and Error Handling

## 1. Helper Function Tests

### 1.1 Test `load_coordinates`

```python
def test_load_coordinates():
    """Test loading coordinates from labels DataFrame."""
    # Create test data
    test_data = [
        {'ID': 'test1_1', 'resname': 'G', 'resid': 1, 'x_1': 1.0, 'y_1': 2.0, 'z_1': 3.0},
        {'ID': 'test1_2', 'resname': 'A', 'resid': 2, 'x_1': 4.0, 'y_1': 5.0, 'z_1': 6.0},
        {'ID': 'test1_3', 'resname': 'U', 'resid': 3, 'x_1': 7.0, 'y_1': 8.0, 'z_1': 9.0},
        {'ID': 'test2_1', 'resname': 'C', 'resid': 1, 'x_1': 10.0, 'y_1': 11.0, 'z_1': 12.0}
    ]
    labels_df = pd.DataFrame(test_data)
    
    # Test for first target
    coords, resnames = load_coordinates(labels_df, 'test1')
    
    # Verify shape and contents
    assert coords.shape == (3, 3)
    assert np.array_equal(coords, np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]))
    assert resnames == ['G', 'A', 'U']
    
    # Test for second target (single residue)
    coords, resnames = load_coordinates(labels_df, 'test2')
    assert coords.shape == (1, 3)
    assert resnames == ['C']
    
    # Test for non-existent target
    with pytest.raises(ValueError):
        load_coordinates(labels_df, 'test3')
```

### 1.2 Test `is_uniform_mi_matrix`

```python
def test_is_uniform_mi_matrix():
    """Test detection of uniform MI matrices."""
    # Case 1: Perfect uniform matrix (same value everywhere except diagonal)
    size = 5
    perfect_uniform = np.ones((size, size)) * 0.5
    np.fill_diagonal(perfect_uniform, 0)  # Zero diagonal
    assert is_uniform_mi_matrix(perfect_uniform)

    # Case 2: Nearly uniform matrix (tiny variations within epsilon)
    nearly_uniform = np.ones((size, size)) * 0.5
    np.fill_diagonal(nearly_uniform, 0)
    # Add tiny noise below epsilon threshold
    noise = np.random.rand(size, size) * 1e-7
    nearly_uniform += noise
    assert is_uniform_mi_matrix(nearly_uniform)

    # Case 3: Non-uniform matrix (variations above epsilon)
    non_uniform = np.random.rand(size, size)
    np.fill_diagonal(non_uniform, 0)
    assert not is_uniform_mi_matrix(non_uniform)

    # Case 4: Empty matrix
    empty_matrix = np.array([]).reshape(0, 0)
    assert not is_uniform_mi_matrix(empty_matrix)

    # Case 5: 1x1 matrix (edge case)
    singleton = np.array([[0]])
    assert not is_uniform_mi_matrix(singleton)

    # Case 6: 2x2 matrix with single off-diagonal pair
    tiny_matrix = np.array([[0, 0.5], [0.5, 0]])
    assert not is_uniform_mi_matrix(tiny_matrix)

    # Case 7: Test with different epsilon values
    borderline = np.ones((size, size)) * 0.5
    np.fill_diagonal(borderline, 0)
    # Add larger noise around 1e-5
    noise = np.random.rand(size, size) * 2e-5
    borderline += noise
    # Should be uniform with default epsilon (1e-6)
    assert not is_uniform_mi_matrix(borderline)
    # Should be uniform with larger epsilon
    assert is_uniform_mi_matrix(borderline, epsilon=1e-4)
```

### 1.3 Test `load_precomputed_features`

For thorough testing, we need to mock the file loading with controlled test data:

```python
@pytest.fixture
def mock_feature_data():
    """Create mock feature data."""
    seq_len = 5  # Small test sequence length
    
    # Mock dihedral features
    dihedral_features = {
        'features': np.random.rand(seq_len, 4).astype(np.float32),
        'eta': np.random.rand(seq_len).astype(np.float32),
        'theta': np.random.rand(seq_len).astype(np.float32)
    }
    
    # Some NaN values as typically found in real data
    dihedral_features['features'][0, :] = np.nan
    
    # Mock thermo features
    thermo_features = {
        'pairing_probs': np.random.rand(seq_len, seq_len).astype(np.float32),
        'positional_entropy': np.random.rand(seq_len).astype(np.float32),
        'accessibility': np.random.rand(seq_len).astype(np.float32),
        'mfe': -10.5,
        'ensemble_energy': -11.2,
        'mfe_probability': 0.7,
        'gc_content': 0.6,
        'paired_fraction': 0.8,
        'sequence': 'GACUG'
    }
    
    # Mock evolutionary features (varied/informative)
    evolutionary_features = {
        'coupling_matrix': np.random.rand(seq_len, seq_len).astype(np.float32),
        'conservation': np.random.rand(seq_len).astype(np.float32),
        'sequence_count': 100
    }
    
    # Mock uniform/uninformative MI matrix
    uniform_mi_features = {
        'coupling_matrix': np.ones((seq_len, seq_len)).astype(np.float32) * 0.5,
        'conservation': np.random.rand(seq_len).astype(np.float32),
        'sequence_count': 1  # Single sequence
    }
    # Zero diagonal as typically found in MI matrices
    np.fill_diagonal(uniform_mi_features['coupling_matrix'], 0)
    
    return {
        'dihedral': dihedral_features,
        'thermo': thermo_features,
        'evolutionary': evolutionary_features,
        'uniform_mi': uniform_mi_features
    }

def test_load_precomputed_features(mock_feature_data, tmp_path):
    """Test loading precomputed features."""
    # Set up temporary directories and files
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    dihedral_dir = features_dir / "dihedral_features"
    thermo_dir = features_dir / "thermo_features"
    mi_dir = features_dir / "mi_features"
    
    dihedral_dir.mkdir()
    thermo_dir.mkdir()
    mi_dir.mkdir()
    
    # Create mock NPZ files
    target_id = "test1"
    
    # Save dihedral features
    dihedral_path = dihedral_dir / f"{target_id}_dihedral_features.npz"
    np.savez(dihedral_path, **mock_feature_data['dihedral'])
    
    # Save thermo features
    thermo_path = thermo_dir / f"{target_id}_thermo_features.npz"
    np.savez(thermo_path, **mock_feature_data['thermo'])
    
    # Save evolutionary features
    mi_path = mi_dir / f"{target_id}_mi_features.npz"
    np.savez(mi_path, **mock_feature_data['evolutionary'])
    
    # Test loading all features
    features = load_precomputed_features(target_id, str(features_dir))
    
    # Verify structure and content
    assert set(features.keys()) == {'dihedral', 'thermo', 'evolutionary'}
    
    # Check dihedral features
    assert 'features' in features['dihedral']
    assert features['dihedral']['features'].shape == (5, 4)
    # Verify NaN handling
    assert not np.isnan(features['dihedral']['features']).any()
    
    # Check thermo features
    assert 'pairing_probs' in features['thermo']
    assert features['thermo']['pairing_probs'].shape == (5, 5)
    assert 'positional_entropy' in features['thermo']
    assert abs(features['thermo']['mfe'] - (-10.5)) < 1e-5
    
    # Check evolutionary features
    assert 'coupling_matrix' in features['evolutionary']
    assert features['evolutionary']['coupling_matrix'].shape == (5, 5)
    assert 'has_valid_mi' in features['evolutionary']
    assert features['evolutionary']['has_valid_mi'] == True
    
    # Now test with uniform MI matrix
    # Create a different target with uniform MI
    target_id2 = "test2"
    
    # Save dihedral and thermo features
    dihedral_path = dihedral_dir / f"{target_id2}_dihedral_features.npz"
    np.savez(dihedral_path, **mock_feature_data['dihedral'])
    thermo_path = thermo_dir / f"{target_id2}_thermo_features.npz"
    np.savez(thermo_path, **mock_feature_data['thermo'])
    
    # Save uniform MI data
    mi_path = mi_dir / f"{target_id2}_mi_features.npz"
    np.savez(mi_path, **mock_feature_data['uniform_mi'])
    
    # Test loading with uniform MI
    features2 = load_precomputed_features(target_id2, str(features_dir))
    
    # Verify uniform MI detection
    assert 'evolutionary' in features2
    assert 'coupling_matrix' in features2['evolutionary']
    assert 'has_valid_mi' in features2['evolutionary']
    assert features2['evolutionary']['has_valid_mi'] == False
    
    # Check that coupling matrix was zeroed out
    assert np.all(features2['evolutionary']['coupling_matrix'] == 0)
```

### 1.4 Test Missing Feature Handling

```python
def test_missing_features_handling(mock_feature_data, tmp_path):
    """Test handling of missing feature files."""
    # Set up minimal directory structure with only thermo features
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    thermo_dir = features_dir / "thermo_features"
    thermo_dir.mkdir()
    
    # Save only thermo features
    target_id = "test1"
    thermo_path = thermo_dir / f"{target_id}_thermo_features.npz"
    np.savez(thermo_path, **mock_feature_data['thermo'])
    
    # Test loading with missing dihedral and evolutionary features
    features = load_precomputed_features(target_id, str(features_dir))
    
    # Verify handling of missing features
    assert features['dihedral'] is None  # Should be None, will be created as zeros in __getitem__
    assert 'thermo' in features
    assert 'evolutionary' in features
    assert features['evolutionary'] is not None
    assert features['evolutionary']['coupling_matrix'].shape == (5, 5)
    assert np.all(features['evolutionary']['coupling_matrix'] == 0)  # Should be zeros
    assert 'has_valid_mi' in features['evolutionary']
    assert features['evolutionary']['has_valid_mi'] == False
    
    # Test missing thermodynamic features (required)
    with pytest.raises(ValueError):
        load_precomputed_features("missing_target", str(features_dir))
```

## 2. RNADataset Tests

### 2.1 Test Dataset Initialization and Filtering

```python
@pytest.fixture
def mock_sequences_df():
    """Create mock sequences DataFrame."""
    return pd.DataFrame({
        'target_id': ['seq1', 'seq2', 'seq3', 'seq4'],
        'sequence': ['GACUG', 'AUCGA', 'GCUAU', 'CAUAG'],
        'temporal_cutoff': ['2022-01-01', '2022-03-15', '2022-06-30', '2022-09-15']
    })

@pytest.fixture
def mock_labels_df():
    """Create mock labels DataFrame."""
    labels = []
    for target in ['seq1', 'seq2', 'seq3', 'seq4']:
        seq_len = 5
        for i in range(1, seq_len + 1):
            labels.append({
                'ID': f"{target}_{i}",
                'resname': ['G', 'A', 'C', 'U', 'G'][i-1],
                'resid': i,
                'x_1': float(i),
                'y_1': float(i + 1),
                'z_1': float(i + 2)
            })
    return pd.DataFrame(labels)

@patch('src.data_loading.load_coordinates')
@patch('pandas.read_csv')
def test_dataset_initialization(mock_read_csv, mock_load_coords, 
                                mock_sequences_df, mock_labels_df):
    """Test RNADataset initialization and filtering."""
    # Mock read_csv to return our test data
    mock_read_csv.side_effect = lambda path: (
        mock_sequences_df if 'sequences' in path else mock_labels_df
    )
    
    # Mock coordinates loading
    mock_load_coords.return_value = (np.ones((5, 3)), ['G', 'A', 'C', 'U', 'G'])
    
    # Test without temporal cutoff
    dataset = RNADataset(
        sequences_csv_path='dummy/sequences.csv',
        labels_csv_path='dummy/labels.csv',
        features_dir='dummy/features'
    )
    
    # Verify all sequences are included
    assert len(dataset) == 4
    assert set(dataset.target_ids) == {'seq1', 'seq2', 'seq3', 'seq4'}
    
    # Test with temporal cutoff
    dataset_cut = RNADataset(
        sequences_csv_path='dummy/sequences.csv',
        labels_csv_path='dummy/labels.csv',
        features_dir='dummy/features',
        temporal_cutoff='2022-05-01'
    )
    
    # Verify only sequences before cutoff are included
    assert len(dataset_cut) == 2
    assert set(dataset_cut.target_ids) == {'seq1', 'seq2'}
    
    # Test validation set (should ignore temporal cutoff)
    dataset_val = RNADataset(
        sequences_csv_path='dummy/sequences.csv',
        labels_csv_path='dummy/labels.csv',
        features_dir='dummy/features',
        temporal_cutoff='2022-05-01',
        use_validation_set=True
    )
    
    # Verify all sequences are included despite cutoff
    assert len(dataset_val) == 4
```

### 2.2 Test `__getitem__` with MI Matrix Handling

```python
@patch('src.data_loading.load_precomputed_features')
@patch('src.data_loading.load_coordinates')
@patch('pandas.read_csv')
def test_dataset_getitem_mi_handling(mock_read_csv, mock_load_coords, mock_load_features,
                         mock_sequences_df, mock_labels_df):
    """Test RNADataset __getitem__ method with different MI matrix cases."""
    # Mock read_csv to return our test data
    mock_read_csv.side_effect = lambda path: (
        mock_sequences_df if 'sequences' in path else mock_labels_df
    )
    
    # Mock coordinates loading
    mock_load_coords.return_value = (np.ones((5, 3)), ['G', 'A', 'C', 'U', 'G'])
    
    # Create dataset
    dataset = RNADataset(
        sequences_csv_path='dummy/sequences.csv',
        labels_csv_path='dummy/labels.csv',
        features_dir='dummy/features'
    )
    
    # Set filtered sequences explicitly for testing
    dataset.filtered_sequences = mock_sequences_df['target_id'].tolist()
    
    # Test with valid MI matrix
    # Create feature data with valid MI matrix
    feature_data_valid_mi = {
        'dihedral': {
            'features': np.random.rand(5, 4).astype(np.float32),
        },
        'thermo': {
            'pairing_probs': np.random.rand(5, 5).astype(np.float32),
            'positional_entropy': np.random.rand(5).astype(np.float32),
            'accessibility': np.random.rand(5).astype(np.float32),
            'mfe': -10.5,
        },
        'evolutionary': {
            'coupling_matrix': np.random.rand(5, 5).astype(np.float32),
            'conservation': np.random.rand(5).astype(np.float32),
            'has_valid_mi': True
        }
    }
    
    # Mock for first test
    mock_load_features.return_value = feature_data_valid_mi
    
    # Get item with valid MI
    sample = dataset[0]
    
    # Verify metadata flags
    assert 'meta' in sample
    assert 'has_dihedrals' in sample['meta']
    assert 'has_msa' in sample['meta']
    assert sample['meta']['has_dihedrals'] == True
    assert sample['meta']['has_msa'] == True
    
    # Test with uniform/invalid MI matrix
    # Create feature data with uniform MI matrix (detected as invalid)
    feature_data_invalid_mi = {
        'dihedral': {
            'features': np.random.rand(5, 4).astype(np.float32),
        },
        'thermo': {
            'pairing_probs': np.random.rand(5, 5).astype(np.float32),
            'positional_entropy': np.random.rand(5).astype(np.float32),
        },
        'evolutionary': {
            'coupling_matrix': np.zeros((5, 5)).astype(np.float32),  # Zero matrix
            'conservation': np.random.rand(5).astype(np.float32),
            'has_valid_mi': False  # Marked as invalid
        }
    }
    
    # Mock for second test
    mock_load_features.return_value = feature_data_invalid_mi
    
    # Get item with invalid MI
    sample = dataset[0]
    
    # Verify metadata flags
    assert 'meta' in sample
    assert 'has_msa' in sample['meta']
    assert sample['meta']['has_msa'] == False  # Should be False despite having feature file
    
    # Verify coupling matrix exists but is zeros
    assert 'coupling_matrix' in sample
    assert torch.all(sample['coupling_matrix'] == 0)
    
    # Test with missing evolutionary features
    # Create feature data with missing MI
    feature_data_missing_mi = {
        'dihedral': {
            'features': np.random.rand(5, 4).astype(np.float32),
        },
        'thermo': {
            'pairing_probs': np.random.rand(5, 5).astype(np.float32),
            'positional_entropy': np.random.rand(5).astype(np.float32),
        },
        'evolutionary': {
            'coupling_matrix': np.zeros((5, 5)).astype(np.float32),  # Default zeros
            'has_valid_mi': False  # Marked as invalid because missing
        }
    }
    
    # Mock for third test
    mock_load_features.return_value = feature_data_missing_mi
    
    # Get item with missing MI
    sample = dataset[0]
    
    # Verify metadata flags
    assert 'meta' in sample
    assert 'has_msa' in sample['meta']
    assert sample['meta']['has_msa'] == False
```

### 2.3 Test Feature Availability and Update Mechanism

```python
def test_feature_availability_and_update(mock_sequences_df, tmp_path):
    """Test detection of available features and update mechanism."""
    # Set up directory structure
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    thermo_dir = features_dir / "thermo_features"
    mi_dir = features_dir / "mi_features"
    dihedral_dir = features_dir / "dihedral_features"
    
    thermo_dir.mkdir()
    mi_dir.mkdir()
    dihedral_dir.mkdir()
    
    # Save sequences CSV
    sequences_csv = tmp_path / "sequences.csv"
    mock_sequences_df.to_csv(sequences_csv, index=False)
    
    # Initially create only some features
    # Create thermo features (required) for all sequences
    for i, target_id in enumerate(mock_sequences_df['target_id']):
        # Create minimal thermo data
        thermo_data = {
            'pairing_probs': np.random.rand(5, 5).astype(np.float32),
            'positional_entropy': np.random.rand(5).astype(np.float32)
        }
        thermo_path = thermo_dir / f"{target_id}_thermo_features.npz"
        np.savez(thermo_path, **thermo_data)
        
        # Create dihedral features for first two sequences only
        if i < 2:
            dihedral_data = {
                'features': np.random.rand(5, 4).astype(np.float32)
            }
            dihedral_path = dihedral_dir / f"{target_id}_dihedral_features.npz"
            np.savez(dihedral_path, **dihedral_data)
        
        # Create MI features with different properties:
        # seq1: valid MI
        # seq2: uniform/invalid MI
        # seq3: missing MI
        # seq4: missing MI (will be added later)
        if i == 0:  # Valid MI for seq1
            mi_data = {
                'coupling_matrix': np.random.rand(5, 5).astype(np.float32),
                'conservation': np.random.rand(5).astype(np.float32)
            }
            mi_path = mi_dir / f"{target_id}_mi_features.npz"
            np.savez(mi_path, **mi_data)
        elif i == 1:  # Uniform MI for seq2
            # Create uniform MI matrix
            mi_data = {
                'coupling_matrix': np.ones((5, 5)) * 0.5,
                'conservation': np.random.rand(5).astype(np.float32)
            }
            np.fill_diagonal(mi_data['coupling_matrix'], 0)  # Zero diagonal
            mi_path = mi_dir / f"{target_id}_mi_features.npz"
            np.savez(mi_path, **mi_data)
    
    # Create dataset requiring features
    dataset = RNADataset(
        sequences_csv_path=str(sequences_csv),
        features_dir=str(features_dir),
        require_features=True
    )
    
    # Should have all 4 sequences since all have thermo features
    assert len(dataset) == 4
    
    # Check metadata for each sequence
    sample0 = dataset[0]  # seq1 - should have valid MI
    assert sample0['meta']['has_dihedrals'] == True
    assert sample0['meta']['has_msa'] == True
    
    sample1 = dataset[1]  # seq2 - should have invalid MI
    assert sample1['meta']['has_dihedrals'] == True
    assert sample1['meta']['has_msa'] == False
    
    sample2 = dataset[2]  # seq3 - should have no MI
    assert sample2['meta']['has_dihedrals'] == False
    assert sample2['meta']['has_msa'] == False
    
    # Now add MI features for seq4
    target_id = mock_sequences_df['target_id'][3]  # seq4
    mi_data = {
        'coupling_matrix': np.random.rand(5, 5).astype(np.float32),
        'conservation': np.random.rand(5).astype(np.float32)
    }
    mi_path = mi_dir / f"{target_id}_mi_features.npz"
    np.savez(mi_path, **mi_data)
    
    # Call update_available_features
    dataset.update_available_features()
    
    # Check updated metadata
    sample3 = dataset[3]  # seq4 - should now have valid MI
    assert sample3['meta']['has_dihedrals'] == False
    assert sample3['meta']['has_msa'] == True
```

## 3. Collate Function Tests

### 3.1 Test Batch Collation with MI Metadata

```python
def test_collate_fn_with_mi_metadata():
    """Test batch collation handling of MI metadata flags."""
    # Create samples with different MI availability
    sample1 = {
        'target_id': 'seq1',
        'sequence_int': torch.tensor([0, 1, 2, 3, 4]),
        'dihedral_features': torch.rand(5, 4),
        'pairing_probs': torch.rand(5, 5),
        'positional_entropy': torch.rand(5),
        'coupling_matrix': torch.rand(5, 5),  # Non-zero coupling matrix
        'length': 5,
        'meta': {
            'has_dihedrals': torch.tensor(True),
            'has_msa': torch.tensor(True)  # Valid MI
        }
    }
    
    sample2 = {
        'target_id': 'seq2',
        'sequence_int': torch.tensor([4, 3, 2, 1, 0]),
        'dihedral_features': torch.rand(5, 4),
        'pairing_probs': torch.rand(5, 5),
        'positional_entropy': torch.rand(5),
        'coupling_matrix': torch.zeros(5, 5),  # Zero coupling matrix
        'length': 5,
        'meta': {
            'has_dihedrals': torch.tensor(True),
            'has_msa': torch.tensor(False)  # Invalid/uniform MI
        }
    }
    
    sample3 = {
        'target_id': 'seq3',
        'sequence_int': torch.tensor([2, 1, 0]),
        'dihedral_features': torch.zeros(3, 4),
        'pairing_probs': torch.rand(3, 3),
        'positional_entropy': torch.rand(3),
        'coupling_matrix': torch.zeros(3, 3),  # Zero coupling matrix 
        'length': 3,
        'meta': {
            'has_dihedrals': torch.tensor(False),
            'has_msa': torch.tensor(False)  # Missing MI
        }
    }
    
    # Collate samples
    batch = collate_fn([sample1, sample2, sample3])
    
    # Verify batch structure
    assert 'meta' in batch
    assert 'has_dihedrals' in batch['meta']
    assert 'has_msa' in batch['meta']
    
    # Verify metadata tensor shapes
    assert batch['meta']['has_dihedrals'].shape == (3,)
    assert batch['meta']['has_msa'].shape == (3,)
    
    # Verify metadata values
    assert torch.all(batch['meta']['has_dihedrals'] == torch.tensor([True, True, False]))
    assert torch.all(batch['meta']['has_msa'] == torch.tensor([True, False, False]))
    
    # Verify that coupling matrices are properly padded
    assert batch['coupling_matrix'].shape == (3, 5, 5)  # Padded to max_len=5
    # First sample should have non-zero values
    assert not torch.all(batch['coupling_matrix'][0] == 0)
    # Second and third samples should be all zeros
    assert torch.all(batch['coupling_matrix'][1] == 0)
    assert torch.all(batch['coupling_matrix'][2, :3, :3] == 0)
    # Third sample padding should be zeros
    assert torch.all(batch['coupling_matrix'][2, 3:, :] == 0)
    assert torch.all(batch['coupling_matrix'][2, :, 3:] == 0)
```

### 3.2 Test Variable-Length Collation

```python
def test_collate_fn_variable_length():
    """Test collate_fn with variable-length sequences."""
    # Create samples of different lengths
    samples = [
        {
            'target_id': 'seq1',
            'sequence_int': torch.tensor([0, 1, 2, 3, 4]),
            'dihedral_features': torch.rand(5, 4),
            'pairing_probs': torch.rand(5, 5),
            'positional_entropy': torch.rand(5),
            'coupling_matrix': torch.rand(5, 5),
            'coordinates': torch.rand(5, 3),
            'length': 5,
            'meta': {
                'has_dihedrals': torch.tensor(True),
                'has_msa': torch.tensor(True)
            }
        },
        {
            'target_id': 'seq2',
            'sequence_int': torch.tensor([4, 3, 2]),
            'dihedral_features': torch.rand(3, 4),
            'pairing_probs': torch.rand(3, 3),
            'positional_entropy': torch.rand(3),
            'coupling_matrix': torch.rand(3, 3),
            'coordinates': torch.rand(3, 3),
            'length': 3,
            'meta': {
                'has_dihedrals': torch.tensor(False),
                'has_msa': torch.tensor(False)
            }
        }
    ]
    
    # Collate samples
    batch = collate_fn(samples)
    
    # Verify batch structure and padding
    assert batch['sequence_int'].shape == (2, 5)  # Padded to max_len=5
    assert batch['dihedral_features'].shape == (2, 5, 4)
    assert batch['pairing_probs'].shape == (2, 5, 5)
    assert batch['coordinates'].shape == (2, 5, 3)
    
    # Check mask
    assert batch['mask'].shape == (2, 5)
    assert torch.all(batch['mask'][0])  # All positions valid for seq1
    assert torch.all(batch['mask'][1, :3])  # First 3 positions valid for seq2
    assert torch.all(~batch['mask'][1, 3:])  # Last 2 positions are padding for seq2
    
    # Check lengths
    assert torch.all(batch['lengths'] == torch.tensor([5, 3]))
    
    # Verify padding is zeros
    assert torch.all(batch['sequence_int'][1, 3:] == 0)
    assert torch.all(batch['dihedral_features'][1, 3:] == 0)
    assert torch.all(batch['pairing_probs'][1, 3:] == 0)
    assert torch.all(batch['pairing_probs'][1, :, 3:] == 0)
```

### 3.3 Test Edge Cases

```python
def test_collate_fn_edge_cases():
    """Test collate_fn with edge cases."""
    # Case 1: Single sample
    samples = [
        {
            'target_id': 'seq1',
            'sequence_int': torch.tensor([0, 1, 2]),
            'dihedral_features': torch.rand(3, 4),
            'pairing_probs': torch.rand(3, 3),
            'positional_entropy': torch.rand(3),
            'coupling_matrix': torch.rand(3, 3),
            'length': 3,
            'meta': {
                'has_dihedrals': torch.tensor(True),
                'has_msa': torch.tensor(True)
            }
        }
    ]
    
    batch = collate_fn(samples)
    assert batch['sequence_int'].shape == (1, 3)
    assert batch['mask'].shape == (1, 3)
    assert 'meta' in batch
    assert batch['meta']['has_msa'].shape == (1,)
    assert batch['meta']['has_msa'][0] == True
    
    # Case 2: Empty batch (should not happen in practice, but test for robustness)
    with pytest.raises(Exception):  # Exact exception depends on implementation
        collate_fn([])
    
    # Case 3: Missing optional tensors
    samples = [
        {
            'target_id': 'seq1',
            'sequence_int': torch.tensor([0, 1, 2]),
            'pairing_probs': torch.rand(3, 3),  # Only include required tensors
            'length': 3,
            'meta': {
                'has_dihedrals': torch.tensor(False),
                'has_msa': torch.tensor(False)
            }
        },
        {
            'target_id': 'seq2',
            'sequence_int': torch.tensor([3, 4]),
            'pairing_probs': torch.rand(2, 2),
            'length': 2,
            'meta': {
                'has_dihedrals': torch.tensor(False),
                'has_msa': torch.tensor(False)
            }
        }
    ]
    
    batch = collate_fn(samples)
    assert 'dihedral_features' not in batch  # Optional tensor not in any sample
    assert batch['sequence_int'].shape == (2, 3)
    assert batch['pairing_probs'].shape == (2, 3, 3)
    assert batch['mask'].shape == (2, 3)
    # Metadata should still be present
    assert 'meta' in batch
    assert 'has_msa' in batch['meta']
    assert torch.all(batch['meta']['has_msa'] == False)
    
    # Case a4: Mix of valid and uniform MI
    samples = [
        {
            'target_id': 'seq1',
            'sequence_int': torch.tensor([0, 1, 2]),
            'pairing_probs': torch.rand(3, 3), 
            'coupling_matrix': torch.rand(3, 3),  # Non-zero coupling
            'length': 3,
            'meta': {
                'has_dihedrals': torch.tensor(False),
                'has_msa': torch.tensor(True)  # Valid MSA
            }
        },
        {
            'target_id': 'seq2',
            'sequence_int': torch.tensor([3, 4]),
            'pairing_probs': torch.rand(2, 2),
            'coupling_matrix': torch.zeros(2, 2),  # Zero coupling from uniform MI
            'length': 2,
            'meta': {
                'has_dihedrals': torch.tensor(False),
                'has_msa': torch.tensor(False)  # Invalid/uniform MSA
            }
        }
    ]
    
    batch = collate_fn(samples)
    assert batch['coupling_matrix'].shape == (2, 3, 3)
    assert 'meta' in batch
    assert torch.all(batch['meta']['has_msa'] == torch.tensor([True, False]))
```

## 4. End-to-End Tests

Create an end-to-end test that combines all components:

```python
def test_data_loading_end_to_end(tmp_path):
    """Test the entire data loading pipeline end-to-end."""
    # Set up test directory structure
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    for subdir in ["dihedral_features", "thermo_features", "mi_features"]:
        (features_dir / subdir).mkdir()
    
    # Create test sequences CSV
    sequences_data = [
        {'target_id': 'seq1', 'sequence': 'GACUG', 'temporal_cutoff': '2022-01-01'},
        {'target_id': 'seq2', 'sequence': 'AUCGAU', 'temporal_cutoff': '2022-06-01'}
    ]
    sequences_df = pd.DataFrame(sequences_data)
    sequences_csv_path = tmp_path / "sequences.csv"
    sequences_df.to_csv(sequences_csv_path, index=False)
    
    # Create test labels CSV
    labels_data = []
    for seq_id, seq in [('seq1', 'GACUG'), ('seq2', 'AUCGAU')]:
        for i, base in enumerate(seq):
            labels_data.append({
                'ID': f"{seq_id}_{i+1}",
                'resname': base,
                'resid': i+1,
                'x_1': float(i),
                'y_1': float(i+1),
                'z_1': float(i+2)
            })
    labels_df = pd.DataFrame(labels_data)
    labels_csv_path = tmp_path / "labels.csv"
    labels_df.to_csv(labels_csv_path, index=False)
    
    # Create mock feature files for each sequence with different MI properties
    # seq1: Valid MI
    # seq2: Uniform/invalid MI
    for i, (seq_id, seq_len) in enumerate([('seq1', 5), ('seq2', 6)]):
        # Dihedral features
        dihedral_path = features_dir / "dihedral_features" / f"{seq_id}_dihedral_features.npz"
        np.savez(dihedral_path, 
                 features=np.random.rand(seq_len, 4).astype(np.float32))
        
        # Thermo features
        thermo_path = features_dir / "thermo_features" / f"{seq_id}_thermo_features.npz"
        np.savez(thermo_path,
                 pairing_probs=np.random.rand(seq_len, seq_len).astype(np.float32),
                 positional_entropy=np.random.rand(seq_len).astype(np.float32),
                 mfe=-10.0)
        
        # MI features with different properties
        mi_path = features_dir / "mi_features" / f"{seq_id}_mi_features.npz"
        if i == 0:  # seq1: valid MI
            np.savez(mi_path,
                    coupling_matrix=np.random.rand(seq_len, seq_len).astype(np.float32))
        else:  # seq2: uniform MI
            uniform_mi = np.ones((seq_len, seq_len)) * 0.5
            np.fill_diagonal(uniform_mi, 0)
            np.savez(mi_path,
                    coupling_matrix=uniform_mi.astype(np.float32))
    
    # Create dataset
    dataset = RNADataset(
        sequences_csv_path=str(sequences_csv_path),
        labels_csv_path=str(labels_csv_path),
        features_dir=str(features_dir)
    )
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Get a batch
    batch = next(iter(data_loader))
    
    # Verify batch structure
    assert len(batch['target_ids']) == 2
    assert batch['sequence_int'].shape == (2, 6)  # Padded to length of seq2
    assert batch['pairing_probs'].shape == (2, 6, 6)
    assert batch['coupling_matrix'].shape == (2, 6, 6)
    assert batch['mask'].shape == (2, 6)
    
    # Check mask correctness for different lengths
    assert torch.all(batch['mask'][0, :5])  # All 5 positions in seq1 are valid
    assert not batch['mask'][0, 5]  # Position 6 in seq1 is padding
    assert torch.all(batch['mask'][1])  # All 6 positions in seq2 are valid
    
    # Verify metadata flags
    assert 'meta' in batch
    assert 'has_msa' in batch['meta']
    assert batch['meta']['has_msa'].shape == (2,)
    assert batch['meta']['has_msa'][0] == True   # seq1 has valid MI
    assert batch['meta']['has_msa'][1] == False  # seq2 has uniform/invalid MI
    
    # Verify coupling matrices reflect has_msa status
    # seq1 should have non-zero values
    assert not torch.all(batch['coupling_matrix'][0, :5, :5] == 0)
    # seq2 should have all zeros despite having MI file (uniform MI)
    assert torch.all(batch['coupling_matrix'][1] == 0)
    
    # Test device transfer
    if torch.cuda.is_available():
        device = torch.device('cuda')
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        # Handle meta dictionary
        if 'meta' in batch_gpu:
            batch_gpu['meta'] = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch_gpu['meta'].items()}
        
        # Verify device
        for k, v in batch_gpu.items():
            if isinstance(v, torch.Tensor):
                assert v.device.type == 'cuda'
            elif k == 'meta':
                for meta_k, meta_v in batch_gpu['meta'].items():
                    if isinstance(meta_v, torch.Tensor):
                        assert meta_v.device.type == 'cuda'
```

## 5. Memory Usage Tests

It's important to test memory efficiency, especially for large datasets:

```python
def test_memory_efficiency():
    """Test memory usage patterns with large sequences."""
    # Skip if running in CI environment
    import os
    if os.environ.get('CI') == 'true':
        pytest.skip("Skipping memory test in CI environment")
    
    # Create a large dataset with mocked features
    with patch('src.data_loading.load_precomputed_features') as mock_load:
        with patch('src.data_loading.load_coordinates') as mock_coords:
            with patch('pandas.read_csv') as mock_read:
                # Mock large sequences
                seq_len = 500  # Large sequence
                num_sequences = 10
                
                # Create mock sequence data
                sequences = []
                for i in range(num_sequences):
                    sequences.append({
                        'target_id': f'seq{i}',
                        'sequence': 'G' * seq_len,
                        'temporal_cutoff': '2022-01-01'
                    })
                mock_read.return_value = pd.DataFrame(sequences)
                
                # Mock coordinates
                mock_coords.return_value = (np.zeros((seq_len, 3)), ['G'] * seq_len)
                
                # Mock feature loading to return large tensors with appropriate MI matrix properties
                def mock_load_large_features(target_id, features_dir):
                    # For even indices, create valid MI
                    # For odd indices, create uniform MI
                    target_idx = int(target_id.replace('seq', ''))
                    
                    if target_idx % 2 == 0:  # Valid MI
                        coupling_matrix = np.random.rand(seq_len, seq_len).astype(np.float32)
                        has_valid_mi = True
                    else:  # Uniform MI (invalid)
                        coupling_matrix = np.ones((seq_len, seq_len)).astype(np.float32) * 0.5
                        np.fill_diagonal(coupling_matrix, 0)
                        has_valid_mi = False
                    
                    return {
                        'dihedral': {'features': np.zeros((seq_len, 4), dtype=np.float32)},
                        'thermo': {
                            'pairing_probs': np.zeros((seq_len, seq_len), dtype=np.float32),
                            'positional_entropy': np.zeros(seq_len, dtype=np.float32)
                        },
                        'evolutionary': {
                            'coupling_matrix': coupling_matrix,
                            'has_valid_mi': has_valid_mi
                        }
                    }
                
                mock_load.side_effect = mock_load_large_features
                
                # Initialize dataset
                import tracemalloc
                tracemalloc.start()
                
                dataset = RNADataset(
                    sequences_csv_path='dummy',
                    labels_csv_path='dummy',
                    features_dir='dummy'
                )
                
                # Get first item and measure memory
                _ = dataset[0]
                current, peak = tracemalloc.get_traced_memory()
                
                print(f"Current memory usage: {current / 10**6:.2f}MB; Peak: {peak / 10**6:.2f}MB")
                
                # Create dataloader and batch
                loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=4,
                    collate_fn=collate_fn
                )
                
                # Get a batch and measure memory
                batch = next(iter(loader))
                current, peak = tracemalloc.get_traced_memory()
                
                print(f"After batch: Current: {current / 10**6:.2f}MB; Peak: {peak / 10**6:.2f}MB")
                tracemalloc.stop()
                
                # Output batch tensor shapes and memory usage
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key}: {value.shape}, {value.element_size() * value.nelement() / 10**6:.2f}MB")
                
                # Check metadata flags for MI validity
                assert 'meta' in batch
                assert 'has_msa' in batch['meta']
                assert batch['meta']['has_msa'].shape == (4,)  # Batch size 4
                # Every other sample should have valid MI
                assert torch.all(batch['meta']['has_msa'] == torch.tensor([True, False, True, False]))
                
                # No specific assertion, this is more for information
                # In a real test, you might set a maximum acceptable memory usage
```

## 6. Additional Tests

### 6.1 Test Sequence Conversion

```python
def test_sequence_to_int():
    """Test conversion of nucleotide sequences to integers."""
    dataset = RNADataset(
        sequences_csv_path='dummy',
        labels_csv_path='dummy',
        features_dir='dummy'
    )
    
    # Test standard nucleotides
    seq = "GACUTN"
    expected = [2, 0, 1, 3, 3, 4]  # G=2, A=0, C=1, U=3, T=3, N=4
    result = dataset.sequence_to_int(seq)
    assert result == expected
    
    # Test unknown nucleotides
    seq = "XYZW"
    expected = [4, 4, 4, 4]  # All mapped to 4 (unknown)
    result = dataset.sequence_to_int(seq)
    assert result == expected
```

### 6.2 Test Model Integration

```python
def test_dataloader_model_integration():
    """Test integration between dataloader and model."""
    # This test requires a minimal model implementation
    # Skip if some imports fail
    try:
        from src.models.embeddings import SequenceEmbedding
    except ImportError:
        pytest.skip("Skipping model integration test - embeddings not implemented yet")
    
    # Create a minimal model component that uses MI metadata
    class MinimalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_embedding = SequenceEmbedding(
                num_embeddings=5,  # 0-4 for ACGTU+N
                embedding_dim=16
            )
            
            # Different processing paths based on MI validity
            self.msa_path = torch.nn.Linear(16, 16)
            self.no_msa_path = torch.nn.Linear(16, 16)
        
        def forward(self, batch):
            # Extract inputs
            seq = batch['sequence_int']
            mask = batch['mask']
            has_msa = batch['meta']['has_msa']  # [batch_size]
            
            # Apply embedding
            seq_emb = self.seq_embedding(seq)  # [batch_size, seq_len, embed_dim]
            
            # Apply mask
            masked_emb = seq_emb * mask.unsqueeze(-1)
            
            # Different processing based on MI validity
            results = []
            for i, emb in enumerate(masked_emb):
                if has_msa[i]:
                    # Process with MSA path if valid MI
                    results.append(self.msa_path(emb))
                else:
                    # Process with non-MSA path if invalid/missing MI
                    results.append(self.no_msa_path(emb))
            
            output_emb = torch.stack(results)
            
            return {'embeddings': output_emb}
    
    # Create mock dataset
    with patch('src.data_loading.RNADataset.__getitem__') as mock_getitem:
        # Mock __getitem__ to return samples with different MI validity
        def mock_get_item(idx):
            # Alternate between valid and invalid MI
            if idx % 2 == 0:
                has_msa = True
            else:
                has_msa = False
                
            return {
                'target_id': f'seq{idx}',
                'sequence_int': torch.tensor([0, 1, 2, 3, 0]),
                'dihedral_features': torch.rand(5, 4),
                'pairing_probs': torch.rand(5, 5),
                'coupling_matrix': torch.rand(5, 5) if has_msa else torch.zeros(5, 5),
                'length': 5,
                'meta': {
                    'has_dihedrals': torch.tensor(True),
                    'has_msa': torch.tensor(has_msa)
                }
            }
        
        mock_getitem.side_effect = mock_get_item
        
        with patch('src.data_loading.RNADataset.__len__') as mock_len:
            mock_len.return_value = 10
            
            # Create dataset and data loader
            dataset = RNADataset(
                sequences_csv_path='dummy',
                labels_csv_path='dummy',
                features_dir='dummy'
            )
            
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=4,
                collate_fn=collate_fn
            )
            
            # Create model
            model = MinimalModel()
            
            # Run a batch through the model
            batch = next(iter(loader))
            outputs = model(batch)
            
            # Verify output
            assert 'embeddings' in outputs
            assert outputs['embeddings'].shape == (4, 5, 16)
            
            # Verify metadata was used correctly
            assert torch.all(batch['meta']['has_msa'] == torch.tensor([True, False, True, False]))
```

### 6.3 Test Edge Cases for MI Matrix Detection

```python
def test_edge_cases_mi_matrix_detection():
    """Test edge cases for MI matrix detection."""
    # Test near-uniform matrices with borderline deviations
    size = 10
    
    # Create matrices with different levels of uniformity
    # 1. Perfect uniform (should detect as uniform)
    uniform = np.ones((size, size)) * 0.5
    np.fill_diagonal(uniform, 0)
    assert is_uniform_mi_matrix(uniform)
    
    # 2. Nearly uniform with micro deviation (should still detect as uniform)
    micro_dev = uniform.copy()
    micro_dev[0, 1] += 1e-7  # Tiny deviation
    assert is_uniform_mi_matrix(micro_dev)
    
    # 3. Nearly uniform with small deviation (should not detect as uniform)
    small_dev = uniform.copy()
    small_dev[0, 1] += 1e-5  # Small but significant deviation
    assert not is_uniform_mi_matrix(small_dev)
    
    # 4. Test with different epsilon values
    borderline = uniform.copy()
    borderline[0, 1] += 5e-6  # Just above default epsilon
    assert not is_uniform_mi_matrix(borderline, epsilon=1e-6)
    assert is_uniform_mi_matrix(borderline, epsilon=1e-4)
    
    # 5. Test tiny matrix (2x2)
    tiny = np.array([[0, 0.5], [0.5, 0]])
    assert not is_uniform_mi_matrix(tiny)  # Not uniform by definition (too small)
    
    # 6. Test with diagonal values
    with_diag = np.ones((size, size)) * 0.5  # No zeros on diagonal
    assert not is_uniform_mi_matrix(with_diag)  # Diagonal affects calculation
    
    # 7. Test sparse matrices (mostly zeros)
    sparse = np.zeros((size, size))
    sparse[0, 1] = sparse[1, 0] = 0.5  # Only one non-zero off-diagonal value
    assert not is_uniform_mi_matrix(sparse)  # Should not be uniform
```

## Running the Tests

Execute the tests with:

```bash
# Run all data loading tests
pytest -xvs tests/test_data_loading.py

# Run specific test groups
pytest -xvs tests/test_data_loading.py::test_is_uniform_mi_matrix
pytest -xvs tests/test_data_loading.py::test_load_precomputed_features
pytest -xvs tests/test_data_loading.py::test_dataset_getitem_mi_handling
```

Use the `-v` flag for verbose output and `-s` to see print statements.

## Integration with CI/CD

These tests should be incorporated into your CI/CD pipeline to ensure data loading functionality is preserved across changes. Consider implementing the following in your CI configuration:

1. Fast tests: Run basic unit tests on each commit
2. Full test suite: Run all tests including end-to-end tests on pull requests
3. Memory tests: Run memory efficiency tests on a schedule or for releases

## Common Test Failures and Remediation

| Failure Pattern | Likely Cause | Remediation |
|-----------------|--------------|-------------|
| NaN values in tensors | Missing NaN handling in feature loading | Add `np.nan_to_num()` calls |
| Shape mismatch in batch | Inconsistent padding in collate_fn | Check all tensor shapes and padding logic |
| Memory errors with large sequences | Inefficient handling of large matrices | Use sparse matrices or lazy loading |
| Device errors in end-to-end tests | Inconsistent device management | Ensure all tensors moved to same device |
| Missing key errors | Inconsistent feature dictionary structure | Standardize feature access and provide defaults |
| **Uniform MI detection failures** | **Incorrect epsilon value or matrix handling** | **Adjust epsilon value or check matrix normalization** |
| **Metadata flag inconsistency** | **has_msa logic not matching has_valid_mi** | **Ensure has_msa only True when MI exists AND is valid** |
| **Invalid MI handling issues** | **Uniform detection not zeroing matrix** | **Verify uniform matrices are replaced with zeros** |
| **Batched metadata issues** | **Metadata not properly collected in collate_fn** | **Check meta dictionary handling in collate_fn** |

## Test Coverage Goals

Aim for >90% code coverage for the data loading component, ensuring:

1. All public methods and functions are tested
2. All branches of conditional logic are tested
3. Edge cases and error conditions are explicitly tested
4. Integration with dependent components is verified
5. **Uniform MI matrix detection with different thresholds is tested**
6. **Metadata generation and propagation for MI validity is verified**
7. **Mixed batches with both valid and invalid MI are tested**
