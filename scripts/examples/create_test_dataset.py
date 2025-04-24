#!/usr/bin/env python3
"""
Create a small test dataset for RNA 3D structure prediction training.

This script extracts a subset of sequences and labels from the full dataset
to create a small test dataset for quick validation of the training pipeline.
"""

import os
import argparse
import random
import shutil
import pandas as pd
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create a small test dataset for training validation'
    )
    
    parser.add_argument('--input_seq_csv', type=str, default='data/raw/train_sequences.csv',
                       help='Path to input training sequences CSV')
    parser.add_argument('--input_label_csv', type=str, default='data/raw/train_labels.csv',
                       help='Path to input training labels CSV')
    parser.add_argument('--features_dir', type=str, default='data/processed/',
                       help='Path to processed features directory')
    parser.add_argument('--output_dir', type=str, default='data/test_data/',
                       help='Output directory for test dataset')
    parser.add_argument('--num_sequences', type=int, default=10,
                       help='Number of sequences to include in test dataset')
    parser.add_argument('--max_seq_len', type=int, default=150,
                       help='Maximum sequence length to include')
    parser.add_argument('--require_features', action='store_true',
                       help='Only include sequences with all feature files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function to create test dataset."""
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Get absolute paths
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    input_seq_csv = os.path.join(base_dir, args.input_seq_csv) 
    input_label_csv = os.path.join(base_dir, args.input_label_csv)
    features_dir = os.path.join(base_dir, args.features_dir)
    output_dir = os.path.join(base_dir, args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed', 'dihedral_features'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed', 'thermo_features'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'processed', 'mi_features'), exist_ok=True)
    
    # Read sequences
    print(f"Reading sequences from {input_seq_csv}")
    sequences_df = pd.read_csv(input_seq_csv)
    
    # Filter sequences by length
    sequences_df['length'] = sequences_df['sequence'].apply(len)
    sequences_df = sequences_df[sequences_df['length'] <= args.max_seq_len]
    
    # Check for feature availability if required
    if args.require_features:
        print("Checking for features availability...")
        valid_ids = []
        
        for idx, row in sequences_df.iterrows():
            rna_id = row['id']
            
            # Check if all feature files exist
            dihedral_path = os.path.join(features_dir, 'dihedral_features', f"{rna_id}_dihedral_features.npz")
            thermo_path = os.path.join(features_dir, 'thermo_features', f"{rna_id}_thermo_features.npz")
            mi_path = os.path.join(features_dir, 'mi_features', f"{rna_id}_mi_features.npz")
            
            if os.path.exists(dihedral_path) and os.path.exists(thermo_path) and os.path.exists(mi_path):
                valid_ids.append(rna_id)
        
        # Filter sequences by available features
        sequences_df = sequences_df[sequences_df['id'].isin(valid_ids)]
    
    # Randomly sample sequences
    if len(sequences_df) <= args.num_sequences:
        print(f"Only {len(sequences_df)} sequences match criteria - using all of them")
        sampled_sequences = sequences_df
    else:
        print(f"Randomly sampling {args.num_sequences} from {len(sequences_df)} matching sequences")
        sampled_sequences = sequences_df.sample(n=args.num_sequences, random_state=args.seed)
    
    # Get the selected IDs
    selected_ids = sampled_sequences['id'].tolist()
    print(f"Selected {len(selected_ids)} sequences: {selected_ids}")
    
    # Read and filter labels
    print(f"Reading labels from {input_label_csv}")
    labels_df = pd.read_csv(input_label_csv)
    filtered_labels = labels_df[labels_df['id'].isin(selected_ids)]
    
    # Save sequences and labels
    output_seq_path = os.path.join(output_dir, 'test_sequences.csv')
    output_label_path = os.path.join(output_dir, 'test_labels.csv')
    sampled_sequences.to_csv(output_seq_path, index=False)
    filtered_labels.to_csv(output_label_path, index=False)
    
    print(f"Saved {len(sampled_sequences)} sequences to {output_seq_path}")
    print(f"Saved {len(filtered_labels)} label entries to {output_label_path}")
    
    # Copy feature files
    if args.require_features:
        for rna_id in selected_ids:
            # Copy dihedral features
            src_path = os.path.join(features_dir, 'dihedral_features', f"{rna_id}_dihedral_features.npz")
            dst_path = os.path.join(output_dir, 'processed', 'dihedral_features', f"{rna_id}_dihedral_features.npz")
            shutil.copy2(src_path, dst_path)
            
            # Copy thermo features
            src_path = os.path.join(features_dir, 'thermo_features', f"{rna_id}_thermo_features.npz")
            dst_path = os.path.join(output_dir, 'processed', 'thermo_features', f"{rna_id}_thermo_features.npz")
            shutil.copy2(src_path, dst_path)
            
            # Copy MI features
            src_path = os.path.join(features_dir, 'mi_features', f"{rna_id}_mi_features.npz")
            dst_path = os.path.join(output_dir, 'processed', 'mi_features', f"{rna_id}_mi_features.npz")
            shutil.copy2(src_path, dst_path)
        
        print(f"Copied feature files for {len(selected_ids)} sequences")
    
    # Generate a sample script to run training on this test dataset
    script_path = os.path.join(output_dir, 'run_test_training.sh')
    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
# Run enhanced training on test dataset

# Change to the project root directory to fix import issues
cd "{base_dir}"

python scripts/train_enhanced_model_v2.py \\
    --train_csv {os.path.relpath(output_seq_path, base_dir)} \\
    --labels_csv {os.path.relpath(output_label_path, base_dir)} \\
    --features_dir {os.path.relpath(os.path.join(output_dir, 'processed'), base_dir)} \\
    --mixed_precision \\
    --gradient_checkpointing \\
    --debug \\
    --batch_size 2 \\
    --epochs 5 \\
    --output_dir {os.path.relpath(os.path.join(output_dir, 'results'), base_dir)}
""")
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    print(f"Created sample training script: {script_path}")
    print("\nTest dataset creation complete!")

if __name__ == "__main__":
    main()