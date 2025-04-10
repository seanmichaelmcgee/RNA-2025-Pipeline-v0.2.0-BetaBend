# Data Loading Testing Guide

This guide outlines the testing approach for the data loading component (`src/data_loading.py`). It covers unit tests, integration tests, and common test cases to ensure the component functions correctly and handles edge cases properly.

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

### 1.2 Test `load_precomputed_features`

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
    
    # Mock evolutionary features
    evolutionary_features = {
        'coupling_matrix': np.random.rand(seq_len, seq_len).astype(np.float32),
        'conservation': np.random.rand(seq_len).astype(np.float32),
        'sequence_count': 100
    }
    
    return {
        'dihedral': dihedral_features,
        'thermo': thermo_features,
        'evolutionary': evolutionary_features
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
    mi_path = mi_dir / f"{target_id}_features.npz"
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
    assert 'conservation' in features['evolutionary']
```

### 1.3 Test Missing Feature Handling

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

### 2.2 Test `__getitem__`

```python
@patch('src.data_loading.load_precomputed_features')
@patch('src.data_loading.load_coordinates')
@patch('pandas.read_csv')
def test_dataset_getitem(mock_read_csv, mock_load_coords, mock_load_features,
                         mock_sequences_df, mock_labels_df, mock_feature_data):
    """Test RNADataset __getitem__ method."""
    # Mock read_csv to return our test data
    mock_read_csv.side_effect = lambda path: (
        mock_sequences_df if 'sequences' in path else mock_labels_df
    )
    
    # Mock coordinates loading
    mock_load_coords.return_value = (np.ones((5, 3)), ['G', 'A', 'C', 'U', 'G'])
    
    # Mock feature loading
    mock_load_features.return_value = mock_feature_data
    
    # Create dataset
    dataset = RNADataset(
        sequences_csv_path='dummy/sequences.csv',
        labels_csv_path='dummy/labels.csv',
        features_dir='dummy/features'
    )
    
    # Get an item
    sample = dataset[0]
    
    # Verify the sample structure
    assert sample['target_id'] == 'seq1'
    assert isinstance(sample['sequence_int'], torch.Tensor)
    assert sample['sequence_int'].shape == (5,)
    assert sample['sequence_int'].dtype == torch.long
    
    assert isinstance(sample['dihedral_features'], torch.Tensor)
    assert sample['dihedral_features'].shape == (5, 4)
    assert sample['dihedral_features'].dtype == torch.float32
    
    assert isinstance(sample['pairing_probs'], torch.Tensor)
    assert sample['pairing_probs'].shape == (5, 5)
    
    assert isinstance(sample['positional_entropy'], torch.Tensor)
    assert sample['positional_entropy'].shape == (5,)
    
    assert isinstance(sample['coupling_matrix'], torch.Tensor)
    assert sample['coupling_matrix'].shape == (5, 5)
    
    assert isinstance(sample['coordinates'], torch.Tensor)
    assert sample['coordinates'].shape == (5, 3)
    
    assert sample['length'] == 5
    
    # Test with missing features
    mock_feature_data_missing = {
        'dihedral': None,
        'thermo': mock_feature_data['thermo'],
        'evolutionary': None
    }
    mock_load_features.return_value = mock_feature_data_missing
    
    # Get item with missing features
    sample = dataset[0]
    
    # Verify default tensors are created
    assert isinstance(sample['dihedral_features'], torch.Tensor)
    assert sample['dihedral_features'].shape == (5, 4)
    assert torch.all(sample['dihedral_features'] == 0)
    
    assert isinstance(sample['coupling_matrix'], torch.Tensor)
    assert sample['coupling_matrix'].shape == (5, 5)
    assert torch.all(sample['coupling_matrix'] == 0)
```

## 3. Collate Function Tests

### 3.1 Test Basic Collation

```python
def test_collate_fn_basic():
    """Test basic functionality of collate_fn."""
    # Create samples of the same length
    samples = [
        {
            'target_id': 'seq1',
            'sequence_int': torch.tensor([0, 1, 2, 3, 4]),
            'dihedral_features': torch.rand(5, 4),
            'pairing_probs': torch.rand(5, 5),
            'positional_entropy': torch.rand(5),
            'coupling_matrix': torch.rand(5, 5),
            'coordinates': torch.rand(5, 3),
            'length': 5
        },
        {
            'target_id': 'seq2',
            'sequence_int': torch.tensor([4, 3, 2, 1, 0]),
            'dihedral_features': torch.rand(5, 4),
            'pairing_probs': torch.rand(5, 5),
            'positional_entropy': torch.rand(5),
            'coupling_matrix': torch.rand(5, 5),
            'coordinates': torch.rand(5, 3),
            'length': 5
        }
    ]
    
    # Collate samples
    batch = collate_fn(samples)
    
    # Verify batch structure
    assert 'target_ids' in batch
    assert batch['target_ids'] == ['seq1', 'seq2']
    
    assert 'sequence_int' in batch
    assert batch['sequence_int'].shape == (2, 5)
    
    assert 'dihedral_features' in batch
    assert batch['dihedral_features'].shape == (2, 5, 4)
    
    assert 'pairing_probs' in batch
    assert batch['pairing_probs'].shape == (2, 5, 5)
    
    assert 'coordinates' in batch
    assert batch['coordinates'].shape == (2, 5, 3)
    
    assert 'mask' in batch
    assert batch['mask'].shape == (2, 5)
    assert torch.all(batch['mask'])  # All positions are valid
    
    assert 'lengths' in batch
    assert torch.all(batch['lengths'] == 5)
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
            'length': 5
        },
        {
            'target_id': 'seq2',
            'sequence_int': torch.tensor([4, 3, 2]),
            'dihedral_features': torch.rand(3, 4),
            'pairing_probs': torch.rand(3, 3),
            'positional_entropy': torch.rand(3),
            'coupling_matrix': torch.rand(3, 3),
            'coordinates': torch.rand(3, 3),
            'length': 3
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
            'length': 3
        }
    ]
    
    batch = collate_fn(samples)
    assert batch['sequence_int'].shape == (1, 3)
    assert batch['mask'].shape == (1, 3)
    
    # Case 2: Empty batch (should not happen in practice, but test for robustness)
    with pytest.raises(Exception):  # Exact exception depends on implementation
        collate_fn([])
    
    # Case 3: Missing optional tensors
    samples = [
        {
            'target_id': 'seq1',
            'sequence_int': torch.tensor([0, 1, 2]),
            'pairing_probs': torch.rand(3, 3),  # Only include required tensors
            'length': 3
        },
        {
            'target_id': 'seq2',
            'sequence_int': torch.tensor([3, 4]),
            'pairing_probs': torch.rand(2, 2),
            'length': 2
        }
    ]
    
    batch = collate_fn(samples)
    assert 'dihedral_features' not in batch  # Optional tensor not in any sample
    assert batch['sequence_int'].shape == (2, 3)
    assert batch['pairing_probs'].shape == (2, 3, 3)
    assert batch['mask'].shape == (2, 3)
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
    
    # Create mock feature files for each sequence
    for seq_id, seq_len in [('seq1', 5), ('seq2', 6)]:
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
        
        # Evolutionary features (only for seq1 to test missing feature handling)
        if seq_id == 'seq1':
            mi_path = features_dir / "mi_features" / f"{seq_id}_features.npz"
            np.savez(mi_path,
                    coupling_matrix=np.random.rand(seq_len, seq_len).astype(np.float32))
    
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
    assert batch['mask'].shape == (2, 6)
    
    # Check mask correctness for different lengths
    assert torch.all(batch['mask'][0, :5])  # All 5 positions in seq1 are valid
    assert not batch['mask'][0, 5]  # Position 6 in seq1 is padding
    assert torch.all(batch['mask'][1])  # All 6 positions in seq2 are valid
    
    # Test device transfer
    if torch.cuda.is_available():
        device = torch.device('cuda')
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        # Verify device
        for k, v in batch_gpu.items():
            if isinstance(v, torch.Tensor):
                assert v.device.type == 'cuda'
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
                
                # Mock feature loading to return large tensors
                def mock_load_large_features(target_id, features_dir):
                    return {
                        'dihedral': {'features': np.zeros((seq_len, 4), dtype=np.float32)},
                        'thermo': {
                            'pairing_probs': np.zeros((seq_len, seq_len), dtype=np.float32),
                            'positional_entropy': np.zeros(seq_len, dtype=np.float32)
                        },
                        'evolutionary': {
                            'coupling_matrix': np.zeros((seq_len, seq_len), dtype=np.float32)
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
    
    # Create a minimal model component that uses dataloader output
    class MinimalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seq_embedding = SequenceEmbedding(
                num_embeddings=5,  # 0-4 for ACGTU+N
                embedding_dim=16
            )
        
        def forward(self, batch):
            # Extract inputs
            seq = batch['sequence_int']
            mask = batch['mask']
            
            # Apply embedding
            seq_emb = self.seq_embedding(seq)
            
            # Apply mask
            masked_emb = seq_emb * mask.unsqueeze(-1)
            
            return {'embeddings': masked_emb}
    
    # Create mock dataset
    with patch('src.data_loading.RNADataset.__getitem__') as mock_getitem:
        # Mock __getitem__ to return compatible sample
        mock_getitem.return_value = {
            'target_id': 'seq1',
            'sequence_int': torch.tensor([0, 1, 2, 3, 0]),
            'dihedral_features': torch.rand(5, 4),
            'pairing_probs': torch.rand(5, 5),
            'length': 5
        }
        
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
                batch_size=2,
                collate_fn=collate_fn
            )
            
            # Create model
            model = MinimalModel()
            
            # Run a batch through the model
            batch = next(iter(loader))
            outputs = model(batch)
            
            # Verify output
            assert 'embeddings' in outputs
            assert outputs['embeddings'].shape == (2, 5, 16)
```

## Running the Tests

Execute the tests with:

```bash
# Run all data loading tests
pytest -xvs tests/test_data_loading.py

# Run specific test groups
pytest -xvs tests/test_data_loading.py::test_load_coordinates
pytest -xvs tests/test_data_loading.py::test_collate_fn_variable_length
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

## Test Coverage Goals

Aim for >90% code coverage for the data loading component, ensuring:

1. All public methods and functions are tested
2. All branches of conditional logic are tested
3. Edge cases and error conditions are explicitly tested
4. Integration with dependent components is verified
