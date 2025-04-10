#!/usr/bin/env python
"""
Test script for the data loading pipeline.

This script demonstrates how to use the data loading components
and verifies their functionality with example data.
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path

from src.data_loading import RNADataset, collate_fn, create_data_loader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test RNA data loading pipeline')
    
    # Critical path arguments (no hardcoded paths!)
    parser.add_argument('--sequences_csv', type=str, required=True,
                        help='Path to sequences CSV file')
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='Path to labels CSV file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Path to directory containing feature files')
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for data loader')
    parser.add_argument('--temporal_cutoff', type=str, default=None,
                        help='Temporal cutoff date (YYYY-MM-DD) for filtering sequences')
    parser.add_argument('--use_validation', action='store_true',
                        help='Use validation dataset instead of training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading')
    
    return parser.parse_args()


def main():
    """Main function to test data loading pipeline."""
    args = parse_args()
    
    # Print data loading parameters
    print(f"Loading data from:")
    print(f"  Sequences CSV: {args.sequences_csv}")
    print(f"  Labels CSV: {args.labels_csv}")
    print(f"  Features directory: {args.features_dir}")
    if args.temporal_cutoff:
        print(f"  Temporal cutoff: {args.temporal_cutoff}")
    if args.use_validation:
        print("  Using validation dataset")
    
    # Check if paths exist
    for path in [args.sequences_csv, args.labels_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
    
    if not os.path.exists(args.features_dir):
        raise FileNotFoundError(f"Features directory not found: {args.features_dir}")
    
    # Create device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Method 1: Create dataset and data loader directly
    print("\nMethod 1: Manual dataset and loader creation")
    dataset = RNADataset(
        sequences_csv_path=args.sequences_csv,
        labels_csv_path=args.labels_csv,
        features_dir=args.features_dir,
        temporal_cutoff=args.temporal_cutoff,
        use_validation_set=args.use_validation
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Method 2: Use the convenience function
    print("\nMethod 2: Using create_data_loader convenience function")
    data_loader2 = create_data_loader(
        sequences_csv_path=args.sequences_csv,
        labels_csv_path=args.labels_csv,
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        temporal_cutoff=args.temporal_cutoff,
        use_validation_set=args.use_validation,
        num_workers=args.num_workers
    )
    
    # Print dataset information
    print(f"\nDataset contains {len(dataset)} sequences")
    
    # Get a sample from the dataset
    print("\nSample item from dataset:")
    sample = dataset[0]
    print(f"  Target ID: {sample['target_id']}")
    print(f"  Sequence length: {sample['length']}")
    
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Get a batch from the data loader
    print("\nProcessing one batch from data loader:")
    batch = next(iter(data_loader))
    
    # Move batch to device (CPU or GPU)
    batch_on_device = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    
    # Print batch information
    print(f"  Batch contains {len(batch['target_ids'])} sequences")
    print(f"  Target IDs: {batch['target_ids']}")
    print(f"  Sequence lengths: {batch['lengths'].tolist()}")
    
    # Print tensor shapes
    print("\nBatch tensor shapes:")
    for key, value in batch_on_device.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
    
    # Verify masks are correct
    print("\nVerifying mask correctness:")
    for i, length in enumerate(batch['lengths']):
        mask_sum = batch['mask'][i].sum().item()
        print(f"  Sequence {i}, length={length.item()}, valid positions in mask={mask_sum}")
        assert mask_sum == length.item(), f"Mask doesn't match sequence length: {mask_sum} vs {length.item()}"
    
    print("\nData loading test completed successfully!")


if __name__ == "__main__":
    main()
