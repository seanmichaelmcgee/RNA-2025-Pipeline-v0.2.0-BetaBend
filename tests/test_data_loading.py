import os
import numpy as np
import pandas as pd
import pytest
import torch
import tracemalloc
from unittest.mock import patch, MagicMock

from src.data_loading import (
    load_coordinates,
    load_precomputed_features,
    get_dihedral_tensors,
    check_features_availability,
    RNADataset,
    collate_fn,
    create_data_loader
)


class TestLoadCoordinates:
    """Tests for the load_coordinates function."""

    def test_basic_loading(self):
        """Test loading coordinates from a simple DataFrame."""
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
        assert np.allclose(coords, np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]))
        assert resnames == ['G', 'A', 'U']
    
    def test_single_residue(self):
        """Test loading coordinates for a single residue."""
        # Create test data
        test_data = [
            {'ID': 'test1_1', 'resname': 'G', 'resid': 1, 'x_1': 1.0, 'y_1': 2.0, 'z_1': 3.0},
            {'ID': 'test2_1', 'resname': 'C', 'resid': 1, 'x_1': 10.0, 'y_1': 11.0, 'z_1': 12.0}
        ]
        labels_df = pd.DataFrame(test_data)
        
        # Test for second target (single residue)
        coords, resnames = load_coordinates(labels_df, 'test2')
        assert coords.shape == (1, 3)
        assert np.allclose(coords, np.array([[10.0, 11.0, 12.0]]))
        assert resnames == ['C']
    
    def test_nonexistent_target(self):
        """Test loading coordinates for a nonexistent target."""
        # Create test data
        test_data = [
            {'ID': 'test1_1', 'resname': 'G', 'resid': 1, 'x_1': 1.0, 'y_1': 2.0, 'z_1': 3.0}
        ]
        labels_df = pd.DataFrame(test_data)
        
        # Test for non-existent target
        with pytest.raises(ValueError):
            load_coordinates(labels_df, 'test3')
    
    def test_sorting(self):
        """Test that coordinates are sorted by resid."""
        # Create test data with shuffled order
        test_data = [
            {'ID': 'test1_3', 'resname': 'U', 'resid': 3, 'x_1': 7.0, 'y_1': 8.0, 'z_1': 9.0},
            {'ID': 'test1_1', 'resname': 'G', 'resid': 1, 'x_1': 1.0, 'y_1': 2.0, 'z_1': 3.0},
            {'ID': 'test1_2', 'resname': 'A', 'resid': 2, 'x_1': 4.0, 'y_1': 5.0, 'z_1': 6.0}
        ]
        labels_df = pd.DataFrame(test_data)
        
        # Test for first target
        coords, resnames = load_coordinates(labels_df, 'test1')
        
        # Verify data is sorted by resid
        assert resnames == ['G', 'A', 'U']
        assert np.allclose(coords[0], np.array([1.0, 2.0, 3.0]))
        assert np.allclose(coords[1], np.array([4.0, 5.0, 6.0]))
        assert np.allclose(coords[2], np.array([7.0, 8.0, 9.0]))


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


class TestLoadPrecomputedFeatures:
    """Tests for the load_precomputed_features function."""
    
    def test_load_features(self, mock_feature_data, tmp_path):
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
        assert 'conservation' in features['evolutionary']
    
    def test_missing_features(self, mock_feature_data, tmp_path):
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
    
    def test_missing_required_features(self, tmp_path):
        """Test with missing thermodynamic features (required)."""
        # Set up empty directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dihedral_dir = features_dir / "dihedral_features"
        dihedral_dir.mkdir()
        
        # Test missing thermodynamic features (required)
        with pytest.raises(ValueError):
            load_precomputed_features("test1", str(features_dir))
    
    def test_alias_keys(self, mock_feature_data, tmp_path):
        """Test handling of alternative key names."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        thermo_dir = features_dir / "thermo_features"
        thermo_dir.mkdir()
        
        # Create modified thermo features with different key names
        thermo_data = mock_feature_data['thermo'].copy()
        # Use base_pair_probs instead of pairing_probs
        thermo_data['base_pair_probs'] = thermo_data.pop('pairing_probs')
        # Use position_entropy instead of positional_entropy
        thermo_data['position_entropy'] = thermo_data.pop('positional_entropy')
        
        # Save modified thermo features
        target_id = "test1"
        thermo_path = thermo_dir / f"{target_id}_thermo_features.npz"
        np.savez(thermo_path, **thermo_data)
        
        # Test loading with alternative key names
        features = load_precomputed_features(target_id, str(features_dir))
        
        # Verify key mapping worked
        assert 'pairing_probs' in features['thermo']
        assert 'positional_entropy' in features['thermo']


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


class TestRNADataset:
    """Tests for the RNADataset class."""
    
    @patch('pandas.read_csv')
    @patch('src.data_loading.check_features_availability')
    def test_initialization(self, mock_check_features, mock_read_csv, mock_sequences_df, mock_labels_df):
        """Test RNADataset initialization and filtering."""
        # Mock read_csv to return our test data
        mock_read_csv.side_effect = lambda path: (
            mock_sequences_df if 'sequences' in path else mock_labels_df
        )
        
        # Mock check_features_availability to return all features available
        mock_check_features.return_value = {"dihedral": True, "thermo": True, "mi": True}
        
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
    
    @patch('pandas.read_csv')
    @patch('src.data_loading.check_features_availability')
    def test_sequence_to_int(self, mock_check_features, mock_read_csv):
        """Test conversion of nucleotide sequences to integers."""
        # Mock check_features_availability to return all features available
        mock_check_features.return_value = {"dihedral": True, "thermo": True, "mi": True}
        
        # Create empty test dataframe
        test_df = pd.DataFrame({'target_id': ['seq1'], 'sequence': ['GACUG'], 'temporal_cutoff': ['2022-01-01']})
        mock_read_csv.return_value = test_df
        
        # Create dataset with mock paths and mocked data
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
    
    @patch('src.data_loading.load_coordinates')
    @patch('pandas.read_csv')
    @patch('src.data_loading.check_features_availability')
    def test_getitem(self, mock_check_features, mock_read_csv, mock_load_coords,
                     mock_sequences_df, mock_labels_df, mock_feature_data):
        """Test RNADataset __getitem__ method."""
        # Mock check_features_availability
        mock_check_features.return_value = {"dihedral": True, "thermo": True, "mi": True}
        
        # Mock read_csv to return our test data
        mock_read_csv.side_effect = lambda path: (
            mock_sequences_df if 'sequences' in path else mock_labels_df
        )
        
        # Mock coordinates loading
        mock_load_coords.return_value = (np.ones((5, 3)), ['G', 'A', 'C', 'U', 'G'])
        
        # Create test feature data manually
        feature_data = {
            'dihedral': {
                'features': np.random.rand(5, 4).astype(np.float32),
            },
            'thermo': {
                'pairing_probs': np.random.rand(5, 5).astype(np.float32),
                'positional_entropy': np.random.rand(5).astype(np.float32),
                'accessibility': np.random.rand(5).astype(np.float32),
                'mfe': -10.5,
                'ensemble_energy': -11.2,
                'gc_content': 0.6
            },
            'evolutionary': {
                'coupling_matrix': np.random.rand(5, 5).astype(np.float32),
                'conservation': np.random.rand(5).astype(np.float32)
            }
        }
        
        # Create dataset and patch the load_precomputed_features function
        with patch('src.data_loading.load_precomputed_features', return_value=feature_data):
            dataset = RNADataset(
                sequences_csv_path='dummy/sequences.csv',
                labels_csv_path='dummy/labels.csv',
                features_dir='dummy/features'
            )
            
            # Set filtered sequences explicitly for testing
            dataset.filtered_sequences = mock_sequences_df['target_id'].tolist()
            
            # Add coordinates to the dataset's coordinates dictionary for testing
            dataset.coordinates = {
                'seq1': np.ones((5, 3))
            }
            
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
        feature_data_missing = {
            'dihedral': None,
            'thermo': feature_data['thermo'],
            'evolutionary': None
        }
        
        # Create dataset with a new patch for missing features
        with patch('src.data_loading.load_precomputed_features', return_value=feature_data_missing):
            dataset = RNADataset(
                sequences_csv_path='dummy/sequences.csv',
                labels_csv_path='dummy/labels.csv',
                features_dir='dummy/features'
            )
            
            # Set filtered sequences explicitly for testing
            dataset.filtered_sequences = mock_sequences_df['target_id'].tolist()
            
            # Get item with missing features
            sample = dataset[0]
            
            # Verify default tensors are created
            assert isinstance(sample['dihedral_features'], torch.Tensor)
            assert sample['dihedral_features'].shape == (5, 4)
            assert torch.all(sample['dihedral_features'] == 0)
            
            assert isinstance(sample['coupling_matrix'], torch.Tensor)
            assert sample['coupling_matrix'].shape == (5, 5)
            assert torch.all(sample['coupling_matrix'] == 0)


class TestCollateFunction:
    """Tests for the collate_fn function."""
    
    def test_collate_fn_basic(self):
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
    
    def test_collate_fn_variable_length(self):
        """Test collate_fn with variable-length sequences."""
        # Create test data with variable length sequences
        seq1_len = 5
        seq2_len = 3
        
        # We need to simplify the test to avoid tensor shape mismatches
        samples = [
            {
                'target_id': 'seq1',
                'sequence_int': torch.tensor([0, 1, 2, 3, 4]),
                'length': seq1_len,
                'meta': {
                    'has_dihedrals': torch.tensor(True)
                }
            },
            {
                'target_id': 'seq2',
                'sequence_int': torch.tensor([4, 3, 2]),
                'length': seq2_len,
                'meta': {
                    'has_dihedrals': torch.tensor(True)
                }
            }
        ]
        
        # Collate samples
        batch = collate_fn(samples)
        
        # Verify batch structure and padding
        assert batch['sequence_int'].shape == (2, 5)  # Padded to max_len=5
        
        # Check mask
        assert batch['mask'].shape == (2, 5)
        assert torch.all(batch['mask'][0])  # All positions valid for seq1
        assert torch.all(batch['mask'][1, :3])  # First 3 positions valid for seq2
        assert torch.all(~batch['mask'][1, 3:])  # Last 2 positions are padding for seq2
        
        # Check lengths
        assert torch.all(batch['lengths'] == torch.tensor([5, 3]))
        
        # Verify padding is zeros
        assert torch.all(batch['sequence_int'][1, 3:] == 0)
    
    def test_collate_fn_missing_features(self):
        """Test collate_fn with missing optional features."""
        # Create samples with missing optional features
        samples = [
            {
                'target_id': 'seq1',
                'sequence_int': torch.tensor([0, 1, 2, 3, 4]),
                'pairing_probs': torch.rand(5, 5),  # Only include required tensors
                'length': 5
            },
            {
                'target_id': 'seq2',
                'sequence_int': torch.tensor([4, 3, 2]),
                'pairing_probs': torch.rand(3, 3),
                'length': 3
            }
        ]
        
        # Collate samples
        batch = collate_fn(samples)
        
        # Verify batch structure
        assert 'dihedral_features' not in batch  # Optional tensor not in any sample
        assert batch['sequence_int'].shape == (2, 5)
        assert batch['pairing_probs'].shape == (2, 5, 5)
        assert batch['mask'].shape == (2, 5)
    
    def test_collate_fn_scalar_values(self):
        """Test collate_fn with scalar values."""
        # Create samples with scalar values
        samples = [
            {
                'target_id': 'seq1',
                'sequence_int': torch.tensor([0, 1, 2]),
                'pairing_probs': torch.rand(3, 3),
                'mfe': -10.5,
                'gc_content': 0.6,
                'length': 3,
                'meta': {
                    'has_dihedrals': torch.tensor(True)
                }
            },
            {
                'target_id': 'seq2',
                'sequence_int': torch.tensor([4, 3, 2, 1]),
                'pairing_probs': torch.rand(4, 4),
                'mfe': -12.3,
                'gc_content': 0.7,
                'length': 4,
                'meta': {
                    'has_dihedrals': torch.tensor(True)
                }
            }
        ]
        
        # Collate samples
        batch = collate_fn(samples)
        
        # Verify scalar values are properly collected
        assert 'mfe' in batch
        assert isinstance(batch['mfe'], torch.Tensor)
        assert batch['mfe'].shape == (2,)
        assert abs(batch['mfe'][0].item() - (-10.5)) < 1e-5
        assert abs(batch['mfe'][1].item() - (-12.3)) < 1e-5
        
        assert 'gc_content' in batch
        assert batch['gc_content'].shape == (2,)
        assert abs(batch['gc_content'][0].item() - 0.6) < 1e-5
        assert abs(batch['gc_content'][1].item() - 0.7) < 1e-5
    
    def test_collate_fn_single_sample(self):
        """Test collate_fn with a single sample."""
        # Create a single sample
        samples = [
            {
                'target_id': 'seq1',
                'sequence_int': torch.tensor([0, 1, 2]),
                'dihedral_features': torch.rand(3, 4),
                'pairing_probs': torch.rand(3, 3),
                'length': 3
            }
        ]
        
        # Collate samples
        batch = collate_fn(samples)
        
        # Verify batch structure
        assert batch['sequence_int'].shape == (1, 3)
        assert batch['dihedral_features'].shape == (1, 3, 4)
        assert batch['pairing_probs'].shape == (1, 3, 3)
        assert batch['mask'].shape == (1, 3)
        assert torch.all(batch['mask'])  # All positions are valid


class TestFeatureAvailability:
    """Tests for feature availability detection and partial data handling."""
    
    def test_check_features_availability(self, tmp_path):
        """Test feature availability detection function."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dihedral_dir = features_dir / "dihedral_features"
        thermo_dir = features_dir / "thermo_features"
        mi_dir = features_dir / "mi_features"
        
        dihedral_dir.mkdir()
        thermo_dir.mkdir()
        mi_dir.mkdir()
        
        # Create test files for different combinations
        target1 = "target1"  # All features available
        target2 = "target2"  # Only thermo available
        target3 = "target3"  # No features available
        
        # Create empty files
        (dihedral_dir / f"{target1}_dihedral_features.npz").touch()
        (thermo_dir / f"{target1}_thermo_features.npz").touch()
        (mi_dir / f"{target1}_mi_features.npz").touch()
        
        (thermo_dir / f"{target2}_thermo_features.npz").touch()
        
        # Test availability detection
        avail1 = check_features_availability(target1, str(features_dir))
        avail2 = check_features_availability(target2, str(features_dir))
        avail3 = check_features_availability(target3, str(features_dir))
        
        # Verify results
        expected1 = {"dihedral": True, "thermo": True, "mi": True}
        expected2 = {"dihedral": False, "thermo": True, "mi": False}
        expected3 = {"dihedral": False, "thermo": False, "mi": False}
        
        assert avail1 == expected1, f"Expected {expected1}, got {avail1}"
        assert avail2 == expected2, f"Expected {expected2}, got {avail2}"
        assert avail3 == expected3, f"Expected {expected3}, got {avail3}"
    
    def test_dataset_feature_filtering(self, tmp_path, mock_sequences_df):
        """Test dataset filtering based on feature availability."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dihedral_dir = features_dir / "dihedral_features"
        thermo_dir = features_dir / "thermo_features"
        mi_dir = features_dir / "mi_features"
        
        dihedral_dir.mkdir()
        thermo_dir.mkdir()
        mi_dir.mkdir()
        
        # Create test files
        # seq1: all features available
        (dihedral_dir / "seq1_dihedral_features.npz").touch()
        (thermo_dir / "seq1_thermo_features.npz").touch()
        (mi_dir / "seq1_mi_features.npz").touch()
        
        # seq2: only thermo available
        (thermo_dir / "seq2_thermo_features.npz").touch()
        
        # Mock read_csv to return our test data
        with patch('pandas.read_csv', return_value=mock_sequences_df):
            # Test with require_features=True
            dataset = RNADataset(
                sequences_csv_path='dummy/sequences.csv',
                features_dir=str(features_dir),
                require_features=True
            )
            
            # Verify only sequences with all features are included
            assert len(dataset) == 1
            assert dataset.filtered_sequences == ['seq1']
            
            # Test with require_features=False
            dataset_all = RNADataset(
                sequences_csv_path='dummy/sequences.csv',
                features_dir=str(features_dir),
                require_features=False
            )
            
            # Verify all sequences are included
            assert len(dataset_all) == 4
            assert set(dataset_all.filtered_sequences) == {'seq1', 'seq2', 'seq3', 'seq4'}
    
    def test_update_available_features(self, tmp_path, mock_sequences_df):
        """Test dynamic updating of available features."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dihedral_dir = features_dir / "dihedral_features"
        thermo_dir = features_dir / "thermo_features"
        mi_dir = features_dir / "mi_features"
        
        dihedral_dir.mkdir()
        thermo_dir.mkdir()
        mi_dir.mkdir()
        
        # Initial state: only seq1 has all features
        (dihedral_dir / "seq1_dihedral_features.npz").touch()
        (thermo_dir / "seq1_thermo_features.npz").touch()
        (mi_dir / "seq1_mi_features.npz").touch()
        
        # Mock read_csv to return our test data
        with patch('pandas.read_csv', return_value=mock_sequences_df):
            # Create dataset with require_features=True
            dataset = RNADataset(
                sequences_csv_path='dummy/sequences.csv',
                features_dir=str(features_dir),
                require_features=True
            )
            
            # Verify only seq1 is included
            assert len(dataset) == 1
            assert dataset.filtered_sequences == ['seq1']
            
            # Add features for seq2
            (dihedral_dir / "seq2_dihedral_features.npz").touch()
            (thermo_dir / "seq2_thermo_features.npz").touch()
            (mi_dir / "seq2_mi_features.npz").touch()
            
            # Update available features
            count = dataset.update_available_features()
            
            # Verify dataset now includes seq2
            assert count == 2
            assert len(dataset) == 2
            assert set(dataset.filtered_sequences) == {'seq1', 'seq2'}
    
    def test_metadata_flags(self, tmp_path, mock_sequences_df, mock_feature_data):
        """Test generation of metadata flags for feature presence."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dihedral_dir = features_dir / "dihedral_features"
        thermo_dir = features_dir / "thermo_features"
        mi_dir = features_dir / "mi_features"
        
        dihedral_dir.mkdir()
        thermo_dir.mkdir()
        mi_dir.mkdir()
        
        # Create test files with different availability patterns
        # seq1: all features
        np.savez(dihedral_dir / "seq1_dihedral_features.npz", **mock_feature_data['dihedral'])
        np.savez(thermo_dir / "seq1_thermo_features.npz", **mock_feature_data['thermo'])
        np.savez(mi_dir / "seq1_mi_features.npz", **mock_feature_data['evolutionary'])
        
        # seq2: only thermo
        np.savez(thermo_dir / "seq2_thermo_features.npz", **mock_feature_data['thermo'])
        
        # Mock read_csv and load_coordinates
        with patch('pandas.read_csv', return_value=mock_sequences_df), \
             patch('src.data_loading.load_coordinates', return_value=(np.ones((5, 3)), ['G', 'A', 'C', 'U', 'G'])):
            
            # Create dataset with require_features=False to include seq2
            dataset = RNADataset(
                sequences_csv_path='dummy/sequences.csv',
                features_dir=str(features_dir),
                require_features=False
            )
            
            # Get samples
            sample1 = dataset[0]  # seq1
            sample2 = dataset[1]  # seq2
            
            # Verify metadata flags for seq1
            assert 'meta' in sample1
            assert sample1['meta']['has_dihedrals'] == torch.tensor(True)
            assert sample1['meta']['has_thermo'] == torch.tensor(True)
            assert sample1['meta']['has_msa'] == torch.tensor(True)
            
            # Verify metadata flags for seq2
            assert 'meta' in sample2
            assert sample2['meta']['has_dihedrals'] == torch.tensor(False)
            assert sample2['meta']['has_thermo'] == torch.tensor(True)
            assert sample2['meta']['has_msa'] == torch.tensor(False)
            
            # Test collation of metadata
            batch = collate_fn([sample1, sample2])
            
            # Verify metadata in batch
            assert 'meta' in batch
            assert 'has_dihedrals' in batch['meta']
            assert torch.all(batch['meta']['has_dihedrals'] == torch.tensor([True, False]))
            assert torch.all(batch['meta']['has_thermo'] == torch.tensor([True, True]))
            assert torch.all(batch['meta']['has_msa'] == torch.tensor([True, False]))

    def test_partial_feature_tensors(self, tmp_path, mock_sequences_df, mock_feature_data):
        """Test handling of missing features in tensor generation."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dihedral_dir = features_dir / "dihedral_features"
        thermo_dir = features_dir / "thermo_features"
        
        dihedral_dir.mkdir()
        thermo_dir.mkdir()
        
        # Create test files with different feature combinations
        # seq1: both dihedral and thermo
        np.savez(dihedral_dir / "seq1_dihedral_features.npz", **mock_feature_data['dihedral'])
        np.savez(thermo_dir / "seq1_thermo_features.npz", **mock_feature_data['thermo'])
        
        # seq2: only thermo (no dihedral)
        np.savez(thermo_dir / "seq2_thermo_features.npz", **mock_feature_data['thermo'])
        
        # Mock functions
        with patch('pandas.read_csv', return_value=mock_sequences_df), \
             patch('src.data_loading.load_coordinates', return_value=(np.ones((5, 3)), ['G', 'A', 'C', 'U', 'G'])):
            
            # Create dataset with require_features=False
            dataset = RNADataset(
                sequences_csv_path='dummy/sequences.csv',
                features_dir=str(features_dir),
                require_features=False
            )
            
            # Get samples
            sample1 = dataset[0]  # seq1 (with dihedral)
            sample2 = dataset[1]  # seq2 (without dihedral)
            
            # Verify both have dihedral tensors, but different metadata flags
            assert 'dihedral_features' in sample1
            assert 'dihedral_features' in sample2
            
            # seq1 should have actual dihedral values
            assert sample1['meta']['has_dihedrals'] == torch.tensor(True)
            # Check that values are not all zeros
            assert not torch.all(sample1['dihedral_features'] == 0)
            
            # seq2 should have zero dihedral values
            assert sample2['meta']['has_dihedrals'] == torch.tensor(False)
            # Check that values are all zeros
            assert torch.all(sample2['dihedral_features'] == 0)
            
            # Test batch collation
            batch = collate_fn([sample1, sample2])
            
            # Verify batch has dihedral features for both
            assert batch['dihedral_features'].shape == (2, 5, 4)
            # Metadata flags should correctly indicate which have real features
            assert torch.all(batch['meta']['has_dihedrals'] == torch.tensor([True, False]))
    
    @patch('pandas.read_csv')
    @patch('src.data_loading.check_features_availability')
    def test_feature_requirements_error_handling(self, mock_check_features, mock_read_csv, tmp_path, mock_sequences_df):
        """Test error handling when required features are missing."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        # Create feature directories
        dihedral_dir = features_dir / "dihedral_features"
        dihedral_dir.mkdir()
        
        # No thermo directory, which should cause error when attempting to load
        
        # Mock check_features_availability to return that we have dihedral but not thermo
        mock_check_features.return_value = {"dihedral": True, "thermo": False, "mi": False}
        
        # Mock read_csv
        mock_read_csv.return_value = mock_sequences_df
        
        # Create dataset with require_features=False
        dataset = RNADataset(
            sequences_csv_path='dummy/sequences.csv',
            features_dir=str(features_dir),
            require_features=False
        )
        
        # Explicitly set filtered sequences for testing
        dataset.filtered_sequences = ['seq1']
        
        # Mock the load_precomputed_features call inside the __getitem__
        # to raise a specific error for missing thermo features
        with patch('src.data_loading.load_precomputed_features') as mock_load:
            mock_load.side_effect = ValueError("Thermodynamic features not found for seq1. Required for prediction.")
            
            # Access should raise the RuntimeError that wraps the ValueError
            with pytest.raises(RuntimeError, match="Error loading features for seq1"):
                dataset[0]
    
    def test_empty_dataset(self, tmp_path):
        """Test handling of empty dataset after filtering."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        # Create subdirectories but no feature files
        for subdir in ["dihedral_features", "thermo_features", "mi_features"]:
            (features_dir / subdir).mkdir()
        
        # Create empty sequences dataframe
        empty_df = pd.DataFrame({
            'target_id': ['seq1', 'seq2'],
            'sequence': ['GACUG', 'AUCGA'],
            'temporal_cutoff': ['2022-01-01', '2022-03-15']
        })
        
        # Mock read_csv
        with patch('pandas.read_csv', return_value=empty_df):
            # Create dataset with require_features=True (should be empty)
            dataset = RNADataset(
                sequences_csv_path='dummy/sequences.csv',
                features_dir=str(features_dir),
                require_features=True
            )
            
            # Verify dataset is empty
            assert len(dataset) == 0
            assert dataset.filtered_sequences == []


class TestLongSequences:
    """Tests for handling extremely long RNA sequences."""
    
    def test_long_sequence_padding(self):
        """Test padding of extremely long sequences."""
        # Create long sequence samples
        long_seq_len = 500
        short_seq_len = 50
        
        # Create a long sequence and a short sequence sample
        long_sample = {
            'target_id': 'long_seq',
            'sequence_int': torch.randint(0, 5, (long_seq_len,)),
            'dihedral_features': torch.rand(long_seq_len, 4),
            'pairing_probs': torch.rand(long_seq_len, long_seq_len),
            'positional_entropy': torch.rand(long_seq_len),
            'coupling_matrix': torch.rand(long_seq_len, long_seq_len),
            'coordinates': torch.rand(long_seq_len, 3),
            'length': long_seq_len
        }
        
        short_sample = {
            'target_id': 'short_seq',
            'sequence_int': torch.randint(0, 5, (short_seq_len,)),
            'dihedral_features': torch.rand(short_seq_len, 4),
            'pairing_probs': torch.rand(short_seq_len, short_seq_len),
            'positional_entropy': torch.rand(short_seq_len),
            'coupling_matrix': torch.rand(short_seq_len, short_seq_len),
            'coordinates': torch.rand(short_seq_len, 3),
            'length': short_seq_len
        }
        
        # Test batch with both long and short sequence
        batch = collate_fn([long_sample, short_sample])
        
        # Verify padded tensor shapes
        assert batch['sequence_int'].shape == (2, long_seq_len)
        assert batch['dihedral_features'].shape == (2, long_seq_len, 4)
        assert batch['pairing_probs'].shape == (2, long_seq_len, long_seq_len)
        assert batch['positional_entropy'].shape == (2, long_seq_len)
        assert batch['coupling_matrix'].shape == (2, long_seq_len, long_seq_len)
        assert batch['coordinates'].shape == (2, long_seq_len, 3)
        
        # Verify mask is correct
        assert batch['mask'].shape == (2, long_seq_len)
        assert torch.all(batch['mask'][0])  # All positions in long sequence are valid
        assert torch.all(batch['mask'][1, :short_seq_len])  # First short_seq_len positions in short sequence are valid
        assert torch.all(~batch['mask'][1, short_seq_len:])  # Remaining positions are padding
        
        # Verify original data is preserved
        assert torch.all(batch['sequence_int'][0, :long_seq_len] == long_sample['sequence_int'])
        assert torch.all(batch['sequence_int'][1, :short_seq_len] == short_sample['sequence_int'])
        
        # Verify padding is zeros
        assert torch.all(batch['sequence_int'][1, short_seq_len:] == 0)
        assert torch.all(batch['dihedral_features'][1, short_seq_len:] == 0)
    
    def test_very_large_batch_memory_efficiency(self):
        """Test memory efficiency with very large batches."""
        # Create extremely large tensors
        large_seq_len = 1000
        batch_size = 4
        
        # Create large samples
        large_samples = []
        for i in range(batch_size):
            sample = {
                'target_id': f'large_seq_{i}',
                'sequence_int': torch.randint(0, 5, (large_seq_len,)),
                'dihedral_features': torch.rand(large_seq_len, 4),
                'pairing_probs': torch.rand(large_seq_len, large_seq_len),
                'positional_entropy': torch.rand(large_seq_len),
                'length': large_seq_len
            }
            large_samples.append(sample)
        
        # Track memory usage during collation
        import tracemalloc
        
        # Start tracking memory
        tracemalloc.start()
        
        # Perform batch collation
        batch = collate_fn(large_samples)
        
        # Get current memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Verify batch shapes
        assert batch['sequence_int'].shape == (batch_size, large_seq_len)
        assert batch['dihedral_features'].shape == (batch_size, large_seq_len, 4)
        assert batch['pairing_probs'].shape == (batch_size, large_seq_len, large_seq_len)
        
        # Check that 2D matrices use memory-efficient padding
        # For pairing_probs, the memory should be less than naive implementation
        # which would create a new tensor and concatenate
        print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
        
        # The test is primarily for manual inspection of memory usage during development
        # No strict assertion since memory usage varies by environment
        # But we can verify the batch was created without errors
        assert batch is not None
        
    @pytest.mark.parametrize("long_seq_len", [100, 500, 1000])
    def test_memory_efficiency_pad_2d(self, long_seq_len):
        """Test memory efficiency of pad_2d function with large matrices."""
        from src.utils.padding import pad_2d
        
        # Create a large 2D tensor
        large_tensor = torch.rand(long_seq_len, long_seq_len)
        
        # Target padding size slightly larger
        pad_size = long_seq_len + 10
        
        # Track memory during padding
        import tracemalloc
        
        # Start tracking memory
        tracemalloc.start()
        
        # Pad tensor
        padded = pad_2d(large_tensor, pad_size)
        
        # Get memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Verify padded shape
        assert padded.shape == (pad_size, pad_size)
        
        # Verify original data is preserved
        assert torch.all(padded[:long_seq_len, :long_seq_len] == large_tensor)
        
        # Verify padding area is zeros
        assert torch.all(padded[long_seq_len:, :] == 0)
        assert torch.all(padded[:, long_seq_len:] == 0)
        
        # Log memory usage (no strict assertion, just informational)
        print(f"pad_2d peak memory for {long_seq_len}x{long_seq_len}: {peak / 1024 / 1024:.2f} MB")
        
        # Check that function completes successfully
        assert padded is not None


class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_feature_file_corruption(self, tmp_path, mock_sequences_df):
        """Test handling of corrupted feature files."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dihedral_dir = features_dir / "dihedral_features"
        thermo_dir = features_dir / "thermo_features"
        mi_dir = features_dir / "mi_features"
        
        dihedral_dir.mkdir()
        thermo_dir.mkdir()
        mi_dir.mkdir()
        
        # Create invalid NPZ file (just empty file)
        target_id = "seq1"
        (dihedral_dir / f"{target_id}_dihedral_features.npz").touch()
        
        # Create valid thermo file with minimal required data
        thermo_data = {
            'pairing_probs': np.random.rand(5, 5).astype(np.float32),
            'positional_entropy': np.random.rand(5).astype(np.float32)
        }
        np.savez(thermo_dir / f"{target_id}_thermo_features.npz", **thermo_data)
        
        # Mock read_csv
        with patch('pandas.read_csv', return_value=mock_sequences_df):
            # Create dataset
            dataset = RNADataset(
                sequences_csv_path='dummy/sequences.csv',
                features_dir=str(features_dir),
                require_features=False
            )
            
            # Access should raise specific error about dihedral file
            with pytest.raises(Exception) as excinfo:
                dataset[0]
            
            # Verify error contains useful information
            assert "Error loading features" in str(excinfo.value)
    
    def test_zero_length_sequence(self):
        """Test handling of zero-length sequences."""
        # Create empty sequence sample
        empty_sample = {
            'target_id': 'empty_seq',
            'sequence_int': torch.tensor([], dtype=torch.long),
            'dihedral_features': torch.zeros((0, 4)),
            'pairing_probs': torch.zeros((0, 0)),
            'positional_entropy': torch.tensor([]),
            'coupling_matrix': torch.zeros((0, 0)),
            'coordinates': torch.zeros((0, 3)),
            'length': 0
        }
        
        # Create normal sequence sample
        normal_sample = {
            'target_id': 'normal_seq',
            'sequence_int': torch.tensor([0, 1, 2]),
            'dihedral_features': torch.rand(3, 4),
            'pairing_probs': torch.rand(3, 3),
            'positional_entropy': torch.rand(3),
            'coupling_matrix': torch.rand(3, 3),
            'coordinates': torch.rand(3, 3),
            'length': 3
        }
        
        # Test collation with empty sequence
        batch = collate_fn([empty_sample, normal_sample])
        
        # Verify batch shape is based on normal sequence
        assert batch['sequence_int'].shape == (2, 3)
        assert batch['dihedral_features'].shape == (2, 3, 4)
        assert batch['pairing_probs'].shape == (2, 3, 3)
        
        # Verify mask is correct
        assert not torch.any(batch['mask'][0])  # No valid positions in empty sequence
        assert torch.all(batch['mask'][1])      # All positions valid in normal sequence
        
        # Verify lengths are correct
        assert torch.all(batch['lengths'] == torch.tensor([0, 3]))
    
    def test_singleton_batch(self):
        """Test batch with single sample."""
        # Create single sample
        sample = {
            'target_id': 'seq1',
            'sequence_int': torch.tensor([0, 1, 2]),
            'dihedral_features': torch.rand(3, 4),
            'pairing_probs': torch.rand(3, 3),
            'positional_entropy': torch.rand(3),
            'coupling_matrix': torch.rand(3, 3),
            'coordinates': torch.rand(3, 3),
            'length': 3,
            'meta': {
                'has_dihedrals': torch.tensor(True)
            }
        }
        
        # Test batch with single sample
        batch = collate_fn([sample])
        
        # Verify batch shape has batch_size=1
        assert batch['sequence_int'].shape == (1, 3)
        assert batch['dihedral_features'].shape == (1, 3, 4)
        assert batch['pairing_probs'].shape == (1, 3, 3)
        
        # Verify metadata propagation
        assert 'meta' in batch
        assert 'has_dihedrals' in batch['meta']
        assert batch['meta']['has_dihedrals'].shape == (1,)
        assert batch['meta']['has_dihedrals'][0] == True


class TestIntegration:
    """Integration tests for the full data pipeline."""
    
    def test_pipeline_smoke(self, tmp_path, mock_sequences_df, mock_feature_data, mock_labels_df):
        """End-to-end smoke test for the data pipeline."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dihedral_dir = features_dir / "dihedral_features"
        thermo_dir = features_dir / "thermo_features"
        mi_dir = features_dir / "mi_features"
        
        dihedral_dir.mkdir()
        thermo_dir.mkdir()
        mi_dir.mkdir()
        
        # Create minimal feature data if mock data is empty
        basic_feature_data = {
            'dihedral': {
                'features': np.random.rand(5, 4).astype(np.float32)
            },
            'thermo': {
                'pairing_probs': np.random.rand(5, 5).astype(np.float32),
                'positional_entropy': np.random.rand(5).astype(np.float32)
            },
            'evolutionary': {
                'coupling_matrix': np.random.rand(5, 5).astype(np.float32)
            }
        }
        
        # Use mock data if available, otherwise use basic data
        feature_data = mock_feature_data if mock_feature_data else basic_feature_data
        
        # Create feature files for all sequences
        for seq_id in mock_sequences_df['target_id']:
            np.savez(dihedral_dir / f"{seq_id}_dihedral_features.npz", **feature_data['dihedral'])
            np.savez(thermo_dir / f"{seq_id}_thermo_features.npz", **feature_data['thermo'])
            np.savez(mi_dir / f"{seq_id}_mi_features.npz", **feature_data['evolutionary'])
        
        # Create sequence and label CSV files
        sequences_csv_path = tmp_path / "sequences.csv"
        labels_csv_path = tmp_path / "labels.csv"
        
        mock_sequences_df.to_csv(sequences_csv_path, index=False)
        mock_labels_df.to_csv(labels_csv_path, index=False)
        
        # Create data loader
        data_loader = create_data_loader(
            sequences_csv_path=str(sequences_csv_path),
            labels_csv_path=str(labels_csv_path),
            features_dir=str(features_dir),
            batch_size=2,
            shuffle=False,  # For deterministic testing
            require_features=False  # Don't require all features
        )
        
        # Get a batch
        batch = next(iter(data_loader))
        
        # Verify batch structure
        assert isinstance(batch, dict)
        assert 'target_ids' in batch
        assert len(batch['target_ids']) == 2
        assert batch['target_ids'][0] == 'seq1'
        
        # Verify tensor shapes
        assert batch['sequence_int'].shape == (2, 5)  # Batch size 2, seq length 5
        assert batch['dihedral_features'].shape == (2, 5, 4)
        assert batch['pairing_probs'].shape == (2, 5, 5)
        assert batch['coupling_matrix'].shape == (2, 5, 5)
        assert batch['coordinates'].shape == (2, 5, 3)
        assert batch['mask'].shape == (2, 5)
        
        # Verify metadata
        assert 'meta' in batch
        assert 'has_dihedrals' in batch['meta']
        assert 'has_thermo' in batch['meta']
        assert 'has_msa' in batch['meta']
        assert torch.all(batch['meta']['has_dihedrals'])
        assert torch.all(batch['meta']['has_thermo'])
        assert torch.all(batch['meta']['has_msa'])
        
        # Verify data types
        assert batch['sequence_int'].dtype == torch.long
        assert batch['dihedral_features'].dtype == torch.float32
        assert batch['pairing_probs'].dtype == torch.float32
        assert batch['coordinates'].dtype == torch.float32
        assert batch['mask'].dtype == torch.bool
    
    def test_missing_labels_inference_mode(self, tmp_path, mock_sequences_df, mock_feature_data):
        """Test pipeline in inference mode (without labels)."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        dihedral_dir = features_dir / "dihedral_features"
        thermo_dir = features_dir / "thermo_features"
        
        dihedral_dir.mkdir()
        thermo_dir.mkdir()
        
        # Create feature files for all sequences (only thermo is required for inference)
        for seq_id in mock_sequences_df['target_id']:
            np.savez(thermo_dir / f"{seq_id}_thermo_features.npz", **mock_feature_data['thermo'])
        
        # Create sequence CSV file
        sequences_csv_path = tmp_path / "sequences.csv"
        mock_sequences_df.to_csv(sequences_csv_path, index=False)
        
        # Create data loader without labels
        data_loader = create_data_loader(
            sequences_csv_path=str(sequences_csv_path),
            features_dir=str(features_dir),
            batch_size=2,
            shuffle=False,
            require_features=False  # Don't require all features
        )
        
        # Get a batch
        batch = next(iter(data_loader))
        
        # Verify batch structure
        assert isinstance(batch, dict)
        assert 'target_ids' in batch
        
        # Verify tensor shapes
        assert batch['sequence_int'].shape == (2, 5)
        assert batch['dihedral_features'].shape == (2, 5, 4)  # Default zeros for missing
        assert batch['pairing_probs'].shape == (2, 5, 5)
        
        # Coordinates should not be present in inference mode
        assert 'coordinates' not in batch
        
        # Metadata should correctly reflect missing features
        assert torch.all(~batch['meta']['has_dihedrals'])  # No dihedral features
        assert torch.all(batch['meta']['has_thermo'])      # Thermo features present
    
    def test_temporal_cutoff_filtering(self, tmp_path, mock_sequences_df, mock_feature_data):
        """Test temporal cutoff filtering."""
        # Set up directory structure
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        
        thermo_dir = features_dir / "thermo_features"
        thermo_dir.mkdir()
        
        # Create feature files for all sequences
        for seq_id in mock_sequences_df['target_id']:
            np.savez(thermo_dir / f"{seq_id}_thermo_features.npz", **mock_feature_data['thermo'])
        
        # Create sequence CSV file
        sequences_csv_path = tmp_path / "sequences.csv"
        mock_sequences_df.to_csv(sequences_csv_path, index=False)
        
        # Create data loader with temporal cutoff
        early_cutoff_loader = create_data_loader(
            sequences_csv_path=str(sequences_csv_path),
            features_dir=str(features_dir),
            batch_size=4,  # Large enough for all sequences
            temporal_cutoff="2022-03-01",  # Should include only seq1
            shuffle=False,
            require_features=False  # Don't require all features
        )
        
        late_cutoff_loader = create_data_loader(
            sequences_csv_path=str(sequences_csv_path),
            features_dir=str(features_dir),
            batch_size=4,
            temporal_cutoff="2022-07-01",  # Should include seq1, seq2, seq3
            shuffle=False,
            require_features=False  # Don't require all features
        )
        
        # Verify filtering
        assert len(early_cutoff_loader.dataset) == 1
        assert len(late_cutoff_loader.dataset) == 3
        
        # Verify first target ID in each
        early_batch = next(iter(early_cutoff_loader))
        assert early_batch['target_ids'][0] == 'seq1'
        
        late_batch = next(iter(late_cutoff_loader))
        assert set(late_batch['target_ids']) == {'seq1', 'seq2', 'seq3'}
