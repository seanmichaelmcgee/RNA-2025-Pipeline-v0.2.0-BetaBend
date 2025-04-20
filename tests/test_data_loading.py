import os
import numpy as np
import pandas as pd
import pytest
import torch
from unittest.mock import patch, MagicMock

from src.data_loading import (
    load_coordinates,
    load_precomputed_features,
    RNADataset,
    collate_fn
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
    def test_initialization(self, mock_read_csv, mock_sequences_df, mock_labels_df):
        """Test RNADataset initialization and filtering."""
        # Mock read_csv to return our test data
        mock_read_csv.side_effect = lambda path: (
            mock_sequences_df if 'sequences' in path else mock_labels_df
        )
        
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
    
    def test_sequence_to_int(self):
        """Test conversion of nucleotide sequences to integers."""
        # Create dataset with mock paths
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
    
    @patch('src.data_loading.load_precomputed_features')
    @patch('src.data_loading.load_coordinates')
    @patch('pandas.read_csv')
    def test_getitem(self, mock_read_csv, mock_load_coords, mock_load_features,
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
                'length': 3
            },
            {
                'target_id': 'seq2',
                'sequence_int': torch.tensor([4, 3, 2, 1]),
                'pairing_probs': torch.rand(4, 4),
                'mfe': -12.3,
                'gc_content': 0.7,
                'length': 4
            }
        ]
        
        # Collate samples
        batch = collate_fn(samples)
        
        # Verify scalar values are properly collected
        assert 'mfe' in batch
        assert isinstance(batch['mfe'], torch.Tensor)
        assert batch['mfe'].shape == (2,)
        assert batch['mfe'][0].item() == -10.5
        assert batch['mfe'][1].item() == -12.3
        
        assert 'gc_content' in batch
        assert batch['gc_content'].shape == (2,)
        assert batch['gc_content'][0].item() == 0.6
        assert batch['gc_content'][1].item() == 0.7
    
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
        assert avail1 == {"dihedral": True, "thermo": True, "mi": True}
        assert avail2 == {"dihedral": False, "thermo": True, "mi": False}
        assert avail3 == {"dihedral": False, "thermo": False, "mi": False}
    
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
