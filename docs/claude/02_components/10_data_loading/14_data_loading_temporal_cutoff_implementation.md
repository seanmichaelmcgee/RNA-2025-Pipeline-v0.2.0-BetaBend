# RNA 3D Folding: Temporal Cutoff Implementation Guide

## 1. Overview

This document provides comprehensive guidance for implementing temporal cutoff filtering in the RNA dataset loading pipeline for the Stanford RNA 3D Folding Kaggle competition. Proper handling of temporal boundaries is critical for competition compliance and scientific integrity.

## 2. Competition Requirements

The Kaggle competition rules specify important temporal cutoff requirements:

- Each RNA sequence entry includes a `temporal_cutoff` field in YYYY-MM-DD format representing publication date
- For the Early Sharing Prize, submissions must only use information available before specified cutoff dates
- The default safe cutoff date is 2022-05-27 (predates test sequences)
- Training on structures published after the cutoff date could constitute data leakage

The optimal validation strategy involves:
1. **Train set**: Sequences from `train_sequences.csv` published *before* the temporal cutoff
2. **Extended validation set**:
   - All sequences from `validation_sequences.csv` (12 CASP15 targets)
   - Sequences from `train_sequences.csv` published *on or after* the temporal cutoff

## 3. Implementation Specifications

### 3.1 RNADataset Class

Implement `RNADataset` with flexible temporal split options:

```python
class RNADataset(torch.utils.data.Dataset):
    """
    Dataset for RNA 3D structure prediction with temporal split options.
    
    This dataset handles:
    1. Loading RNA sequences, coordinates, and precomputed features
    2. Filtering sequences based on temporal cutoff date
    3. Supporting different split modes (train, validation, validation_extended)
    """
    
    def __init__(
        self,
        sequences_csv_path: str,
        labels_csv_path: str,
        features_dir: str,
        temporal_cutoff: Optional[str] = None,
        split_mode: str = "train"  # Options: "train", "validation", "validation_extended"
    ):
        """
        Initialize RNA dataset with flexible temporal splitting options.
        
        Args:
            sequences_csv_path: Path to CSV with RNA sequences
            labels_csv_path: Path to CSV with ground truth coordinates
            features_dir: Directory containing precomputed features
            temporal_cutoff: Date string (YYYY-MM-DD) to filter sequences
                            For training: include sequences published before this date
                            For validation_extended: include train sequences published on or after this date
            split_mode: Dataset split mode:
                       "train" - uses sequences before temporal_cutoff
                       "validation" - uses all sequences from validation_sequences.csv
                       "validation_extended" - combines validation set with train sequences after temporal_cutoff
        
        Raises:
            ValueError: If temporal_cutoff is missing when required, has invalid format,
                       or if no sequences remain after filtering
        """
        self.sequences_csv_path = sequences_csv_path
        self.labels_csv_path = labels_csv_path
        self.features_dir = features_dir
        self.split_mode = split_mode
        
        # Load labels DataFrame
        self.labels_df = pd.read_csv(labels_csv_path)
        
        # Validate temporal_cutoff format if provided
        if temporal_cutoff is not None:
            try:
                pd.to_datetime(temporal_cutoff)
            except ValueError:
                raise ValueError(f"Invalid temporal_cutoff format: {temporal_cutoff}. "
                                f"Expected format: YYYY-MM-DD")
        
        # Load sequences DataFrame
        sequences_df = pd.read_csv(sequences_csv_path)
        
        # Verify required columns exist
        required_cols = ['target_id', 'sequence']
        if temporal_cutoff is not None:
            required_cols.append('temporal_cutoff')
            
        missing_cols = [col for col in required_cols if col not in sequences_df.columns]
        if missing_cols:
            raise ValueError(f"Sequences CSV missing required columns: {', '.join(missing_cols)}")
        
        # Store original count for logging
        original_count = len(sequences_df)
        
        # Apply filtering based on split mode
        if split_mode == "train":
            if temporal_cutoff is None:
                raise ValueError("temporal_cutoff must be provided for training split")
                
            # Convert string dates to datetime objects for comparison
            sequences_df['temporal_cutoff_dt'] = pd.to_datetime(sequences_df['temporal_cutoff'])
            cutoff_date = pd.to_datetime(temporal_cutoff)
            
            # Filter to include only sequences published before the cutoff
            filtered_df = sequences_df[sequences_df['temporal_cutoff_dt'] < cutoff_date]
            
            filtered_count = len(filtered_df)
            print(f"Training split: Using {filtered_count}/{original_count} sequences published before {temporal_cutoff}")
            
        elif split_mode == "validation":
            # For standard validation, use all sequences in validation_sequences.csv
            filtered_df = sequences_df
            print(f"Validation split: Using all {len(filtered_df)} sequences from validation set")
            
        elif split_mode == "validation_extended":
            if temporal_cutoff is None:
                raise ValueError("temporal_cutoff must be provided for extended validation split")
                
            # Check if we're using train_sequences.csv for extended validation
            is_train_csv = "train" in os.path.basename(sequences_csv_path).lower()
            
            if is_train_csv:
                # Convert string dates to datetime objects for comparison
                sequences_df['temporal_cutoff_dt'] = pd.to_datetime(sequences_df['temporal_cutoff'])
                cutoff_date = pd.to_datetime(temporal_cutoff)
                
                # Filter to include only sequences published on or after the cutoff
                filtered_df = sequences_df[sequences_df['temporal_cutoff_dt'] >= cutoff_date]
                
                filtered_count = len(filtered_df)
                print(f"Extended validation (from train): Using {filtered_count}/{original_count} sequences "
                     f"published on or after {temporal_cutoff}")
            else:
                # Using validation_sequences.csv - include all sequences
                filtered_df = sequences_df
                print(f"Extended validation (validation set): Using all {len(filtered_df)} sequences")
        else:
            raise ValueError(f"Invalid split_mode: {split_mode}. Must be 'train', 'validation', "
                            f"or 'validation_extended'")
        
        # If no sequences remain after filtering, raise an error
        if len(filtered_df) == 0:
            raise ValueError(f"No sequences remain after filtering for {split_mode} with "
                            f"temporal_cutoff={temporal_cutoff}")
        
        # Store filtered sequences
        self.target_ids = filtered_df['target_id'].tolist()
        self.sequences = filtered_df['sequence'].tolist()
        
        # Define nucleotide mapping
        self.nuc_to_int = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3, 'N': 4}
        
        # Log completion of initialization
        print(f"Initialized {split_mode} dataset with {len(self.target_ids)} sequences")
```

### 3.2 Creating Data Loaders with Temporal Splits

Implement a function to create data loaders that respect temporal boundaries:

```python
def create_data_loaders(config):
    """
    Create training and validation data loaders with temporal splitting.
    
    Args:
        config: Configuration dictionary containing paths and parameters:
               - config['data']['train_sequences_csv'] - Path to training sequences CSV
               - config['data']['train_labels_csv'] - Path to training labels CSV
               - config['data']['validation_sequences_csv'] - Path to validation sequences CSV
               - config['data']['validation_labels_csv'] - Path to validation labels CSV
               - config['data']['features_dir'] - Path to features directory
               - config['data']['temporal_cutoff'] - Cutoff date for temporal splitting
               - config['data']['batch_size'] - Batch size for data loaders
               - config['data']['num_workers'] - Number of workers for data loaders
        
    Returns:
        Dictionary containing data loaders:
        - 'train_loader': Training data (sequences before temporal cutoff)
        - 'val_loader': Validation data (all validation sequences)
        - 'val_extended_loader': Extended validation (train sequences after cutoff)
        - 'val_combined_loader': Combined validation (validation + extended)
    """
    # Training data - sequences before temporal cutoff
    train_dataset = RNADataset(
        sequences_csv_path=config['data']['train_sequences_csv'],
        labels_csv_path=config['data']['train_labels_csv'],
        features_dir=config['data']['features_dir'],
        temporal_cutoff=config['data']['temporal_cutoff'],
        split_mode="train"
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    
    # Standard validation data - all sequences from validation_sequences.csv
    val_dataset = RNADataset(
        sequences_csv_path=config['data']['validation_sequences_csv'],
        labels_csv_path=config['data']['validation_labels_csv'],
        features_dir=config['data']['features_dir'],
        split_mode="validation"
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    
    # Extended validation - train sequences after temporal cutoff
    val_extended_dataset = RNADataset(
        sequences_csv_path=config['data']['train_sequences_csv'],
        labels_csv_path=config['data']['train_labels_csv'],
        features_dir=config['data']['features_dir'],
        temporal_cutoff=config['data']['temporal_cutoff'],
        split_mode="validation_extended"
    )
    
    val_extended_loader = torch.utils.data.DataLoader(
        val_extended_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    
    # Combined validation dataset
    val_combined_dataset = CombinedRNADataset([val_dataset, val_extended_dataset])
    
    val_combined_loader = torch.utils.data.DataLoader(
        val_combined_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_fn
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'val_extended_loader': val_extended_loader,
        'val_combined_loader': val_combined_loader
    }
```

### 3.3 Combined Dataset Implementation

For the combined validation set, implement a helper class:

```python
class CombinedRNADataset(torch.utils.data.Dataset):
    """
    Combines multiple RNA datasets into a single dataset.
    Useful for creating the combined validation set.
    """
    
    def __init__(self, datasets):
        """
        Initialize combined dataset.
        
        Args:
            datasets: List of RNADataset instances to combine
        """
        self.datasets = datasets
        self.dataset_sizes = [len(dataset) for dataset in datasets]
        self.cumulative_sizes = [0] + list(itertools.accumulate(self.dataset_sizes))
        
    def __len__(self):
        """Return total number of samples across all datasets."""
        return self.cumulative_sizes[-1]
    
    def __getitem__(self, idx):
        """
        Get item from the appropriate dataset.
        
        Args:
            idx: Global index
            
        Returns:
            Sample from the appropriate dataset
        """
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        sample_idx = idx - self.cumulative_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]
```

## 4. Unit Testing Guidelines

Create comprehensive unit tests to verify temporal cutoff functionality:

```python
class TestTemporalSplitting:
    """Tests for RNADataset temporal splitting functionality."""
    
    def setup_method(self):
        """Create test data with diverse dates."""
        # Create test data with diverse dates
        self.test_data = {
            'target_id': ['seq1', 'seq2', 'seq3', 'seq4'],
            'sequence': ['GACUG', 'AUCGA', 'GCUAU', 'CAUAG'],
            'temporal_cutoff': ['2022-01-01', '2022-03-15', '2022-06-30', '2022-09-15']
        }
        
        # Create mock coordinates data
        self.coords_data = []
        for i, target in enumerate(self.test_data['target_id']):
            seq = self.test_data['sequence'][i]
            for j, base in enumerate(seq):
                self.coords_data.append({
                    'ID': f"{target}_{j+1}",
                    'resname': base,
                    'resid': j+1,
                    'x_1': float(i+j),
                    'y_1': float(i+j+1),
                    'z_1': float(i+j+2)
                })
    
    def test_train_split(self, tmp_path):
        """Test training split with temporal cutoff."""
        # Create temp files for testing
        tmp_dir = tmp_path / "data"
        tmp_dir.mkdir()
        
        # Save CSVs
        seq_csv = tmp_dir / 'sequences.csv'
        pd.DataFrame(self.test_data).to_csv(seq_csv, index=False)
        
        labels_csv = tmp_dir / 'labels.csv'
        pd.DataFrame(self.coords_data).to_csv(labels_csv, index=False)
        
        # Create feature dir
        features_dir = tmp_dir / 'features'
        features_dir.mkdir()
        
        # Test training split with cutoff (should only include sequences before 2022-05-01)
        dataset = RNADataset(
            sequences_csv_path=str(seq_csv),
            labels_csv_path=str(labels_csv),
            features_dir=str(features_dir),
            temporal_cutoff='2022-05-01',
            split_mode="train"
        )
        
        # Verify filtering
        assert len(dataset) == 2
        assert set(dataset.target_ids) == {'seq1', 'seq2'}
    
    def test_validation_split(self, tmp_path):
        """Test validation split (should include all sequences)."""
        # Create temp files
        tmp_dir = tmp_path / "data"
        tmp_dir.mkdir()
        
        # Save CSVs
        seq_csv = tmp_dir / 'sequences.csv'
        pd.DataFrame(self.test_data).to_csv(seq_csv, index=False)
        
        labels_csv = tmp_dir / 'labels.csv'
        pd.DataFrame(self.coords_data).to_csv(labels_csv, index=False)
        
        # Create feature dir
        features_dir = tmp_dir / 'features'
        features_dir.mkdir()
        
        # Test validation split (should include all sequences)
        dataset = RNADataset(
            sequences_csv_path=str(seq_csv),
            labels_csv_path=str(labels_csv),
            features_dir=str(features_dir),
            split_mode="validation"
        )
        
        # Verify all sequences included
        assert len(dataset) == 4
        assert set(dataset.target_ids) == {'seq1', 'seq2', 'seq3', 'seq4'}
    
    def test_validation_extended_split(self, tmp_path):
        """Test extended validation split (should include sequences on or after cutoff)."""
        # Create temp files
        tmp_dir = tmp_path / "data"
        tmp_dir.mkdir()
        
        # Save CSVs
        seq_csv = tmp_dir / 'sequences.csv'
        pd.DataFrame(self.test_data).to_csv(seq_csv, index=False)
        
        labels_csv = tmp_dir / 'labels.csv'
        pd.DataFrame(self.coords_data).to_csv(labels_csv, index=False)
        
        # Create feature dir
        features_dir = tmp_dir / 'features'
        features_dir.mkdir()
        
        # Test extended validation split (should include sequences on or after 2022-05-01)
        dataset = RNADataset(
            sequences_csv_path=str(seq_csv),
            labels_csv_path=str(labels_csv),
            features_dir=str(features_dir),
            temporal_cutoff='2022-05-01',
            split_mode="validation_extended"
        )
        
        # Verify filtering
        assert len(dataset) == 2
        assert set(dataset.target_ids) == {'seq3', 'seq4'}
    
    def test_combined_dataset(self, tmp_path):
        """Test combined dataset functionality."""
        # Create temp files
        tmp_dir = tmp_path / "data"
        tmp_dir.mkdir()
        
        # Save CSVs
        train_seq_csv = tmp_dir / 'train_sequences.csv'
        pd.DataFrame(self.test_data).to_csv(train_seq_csv, index=False)
        
        val_data = {
            'target_id': ['val1', 'val2'],
            'sequence': ['AAAAA', 'CCCCC'],
            'temporal_cutoff': ['2021-01-01', '2021-02-01']
        }
        val_seq_csv = tmp_dir / 'val_sequences.csv'
        pd.DataFrame(val_data).to_csv(val_seq_csv, index=False)
        
        # Combined labels
        labels_data = self.coords_data.copy()
        for i, target in enumerate(val_data['target_id']):
            seq = val_data['sequence'][i]
            for j, base in enumerate(seq):
                labels_data.append({
                    'ID': f"{target}_{j+1}",
                    'resname': base,
                    'resid': j+1,
                    'x_1': float(i+j+10),
                    'y_1': float(i+j+11),
                    'z_1': float(i+j+12)
                })
        
        labels_csv = tmp_dir / 'labels.csv'
        pd.DataFrame(labels_data).to_csv(labels_csv, index=False)
        
        # Create feature dir
        features_dir = tmp_dir / 'features'
        features_dir.mkdir()
        
        # Create validation dataset
        val_dataset = RNADataset(
            sequences_csv_path=str(val_seq_csv),
            labels_csv_path=str(labels_csv),
            features_dir=str(features_dir),
            split_mode="validation"
        )
        
        # Create extended validation dataset
        val_extended_dataset = RNADataset(
            sequences_csv_path=str(train_seq_csv),
            labels_csv_path=str(labels_csv),
            features_dir=str(features_dir),
            temporal_cutoff='2022-05-01',
            split_mode="validation_extended"
        )
        
        # Create combined dataset
        combined_dataset = CombinedRNADataset([val_dataset, val_extended_dataset])
        
        # Verify combined size
        assert len(combined_dataset) == len(val_dataset) + len(val_extended_dataset)
        assert len(combined_dataset) == 4  # 2 validation + 2 extended validation
```

## 5. Error Handling

Implement robust error handling for all temporal cutoff operations:

1. **Invalid date format**:
   ```python
   try:
       pd.to_datetime(temporal_cutoff)
   except ValueError:
       raise ValueError(f"Invalid temporal_cutoff format: {temporal_cutoff}. "
                       f"Expected format: YYYY-MM-DD")
   ```

2. **Missing columns**:
   ```python
   required_cols = ['target_id', 'sequence']
   if temporal_cutoff is not None:
       required_cols.append('temporal_cutoff')
       
   missing_cols = [col for col in required_cols if col not in sequences_df.columns]
   if missing_cols:
       raise ValueError(f"Sequences CSV missing required columns: {', '.join(missing_cols)}")
   ```

3. **Empty dataset after filtering**:
   ```python
   if len(filtered_df) == 0:
       raise ValueError(f"No sequences remain after filtering for {split_mode} "
                       f"with temporal_cutoff={temporal_cutoff}")
   ```

4. **Missing required cutoff date**:
   ```python
   if split_mode == "train" and temporal_cutoff is None:
       raise ValueError("temporal_cutoff must be provided for training split")
   ```

## 6. Integration with Training Loop

Example script showing how to integrate the temporal split data loaders into a training loop:

```python
def main(config_path):
    """
    Main training function using temporal splitting.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders with temporal splitting
    loaders = create_data_loaders(config)
    train_loader = loaders['train_loader']
    val_combined_loader = loaders['val_combined_loader']
    
    # Create model
    model = RNAFoldingModel(config['model'])
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop
    for epoch in range(config['training']['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Compute loss
            loss = compute_combined_loss(outputs, batch, config['loss_weights'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_combined_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(batch)
                
                # Compute loss
                loss = compute_combined_loss(outputs, batch, config['loss_weights'])
                
                val_loss += loss.item()
        
        # Print metrics
        print(f"Epoch {epoch+1}/{config['training']['max_epochs']} - "
             f"Train Loss: {train_loss/len(train_loader):.4f}, "
             f"Val Loss: {val_loss/len(val_combined_loader):.4f}")
```

## 7. Best Practices for Temporal Split Implementation

1. **Log filtering results clearly**:
   - Report before/after counts when applying temporal filters
   - Include the cutoff date in log messages
   - Document which split mode is being used

2. **Handle temporal boundary exactly**:
   - `<` for "before" (excludes the cutoff date)
   - `>=` for "on or after" (includes the cutoff date)

3. **Use configuration-driven approach**:
   - Store temporal cutoff in configuration
   - Default to a safe cutoff date (2022-05-27) but allow override

4. **Verify CSV format during loading**:
   - Check for required columns before attempting to filter
   - Validate date format before conversion

5. **Handle edge cases gracefully**:
   - Empty datasets after filtering
   - Missing temporal information
   - Invalid date formats

## 8. Configuration Example

Example configuration snippet showing temporal cutoff settings:

```yaml
data:
  train_sequences_csv: "data/raw/train_sequences.csv"
  train_labels_csv: "data/raw/train_labels.csv"
  validation_sequences_csv: "data/raw/validation_sequences.csv"
  validation_labels_csv: "data/raw/validation_labels.csv"
  features_dir: "data/processed"
  temporal_cutoff: "2022-05-27"  # Default safe cutoff date
  batch_size: 16
  num_workers: 4
```

## 9. Summary

This implementation guide provides a comprehensive approach to handling temporal cutoffs in the RNA 3D folding pipeline:

1. **RNADataset Class**: Flexible implementation with three split modes (train, validation, validation_extended)
2. **Data Loader Creation**: Functions to create appropriate loaders for each data split
3. **Combined Validation**: Method to create a unified validation set from multiple sources
4. **Robust Error Handling**: Comprehensive checks for data validity and consistency
5. **Unit Tests**: Verification of correct temporal filtering behavior
6. **Integration Example**: Sample code showing how to use these components in training

By following this guide, the data loading implementation will fully comply with the Kaggle competition requirements for temporal cutoffs, preventing data leakage while maximizing available validation data.
