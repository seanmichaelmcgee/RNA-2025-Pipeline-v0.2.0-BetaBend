#!/usr/bin/env python3
"""
Reformat our submission to match the expected Kaggle submission format.

This script converts the submission format from:
target_id,model_id,coordinates,confidence

To the required format:
ID,resname,resid,x_1,y_1,z_1,x_2,y_2,z_2,x_3,y_3,z_3,x_4,y_4,z_4,x_5,y_5,z_5

Where each row corresponds to a single residue position across all 5 models.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reformat submission to match Kaggle's expected format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the current submission file",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="../data/raw/test_sequences.csv",
        help="Path to the test sequences file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the reformatted submission (default: input_reformatted.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output during processing",
    )
    args = parser.parse_args()

    # Set default output path if not provided
    if not args.output:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_reformatted.csv")

    return args


def load_data(submission_path, sequences_path, verbose=False):
    """Load the submission and sequence data."""
    if verbose:
        print(f"Loading submission from: {submission_path}")
        print(f"Loading sequences from: {sequences_path}")

    submission_df = pd.read_csv(submission_path)
    sequences_df = pd.read_csv(sequences_path)

    if verbose:
        print(f"Loaded {len(submission_df)} submission entries")
        print(f"Loaded {len(sequences_df)} sequence entries")

    return submission_df, sequences_df


def parse_sequences(sequences_df, verbose=False):
    """Extract sequence information by target_id."""
    sequence_dict = {}
    for _, row in sequences_df.iterrows():
        target_id = row["target_id"]
        sequence = row["sequence"]
        sequence_dict[target_id] = sequence

    if verbose:
        print(f"Parsed sequences for {len(sequence_dict)} targets")
        for target_id, seq in sequence_dict.items():
            print(f"  {target_id}: length={len(seq)}, starts with {seq[:10]}...")

    return sequence_dict


def parse_coordinates(submission_df, verbose=False):
    """Parse and organize the coordinate data from the submission."""
    coords_by_target_model = {}
    for _, row in submission_df.iterrows():
        target_id = row["target_id"]
        model_id = int(row["model_id"])  # Ensure model_id is an integer

        # Parse coordinates JSON
        coords = json.loads(row["coordinates"])

        # Store the coordinates
        if target_id not in coords_by_target_model:
            coords_by_target_model[target_id] = {}

        coords_by_target_model[target_id][model_id] = coords

    if verbose:
        print(f"Parsed coordinates for {len(coords_by_target_model)} targets")
        for target_id, models in coords_by_target_model.items():
            print(f"  {target_id}: {len(models)} models")
            for model_id, coords in models.items():
                print(f"    Model {model_id}: {len(coords)} residues")

    return coords_by_target_model


def reformat_submission(sequence_dict, coords_by_target_model, verbose=False):
    """Reformat the submission to match the expected format."""
    new_rows = []

    for target_id, sequence in sequence_dict.items():
        if target_id not in coords_by_target_model:
            if verbose:
                print(f"Warning: No predictions found for target {target_id}")
            continue

        if verbose:
            print(f"Processing target {target_id} with sequence length {len(sequence)}")

        for residue_idx, residue_name in enumerate(sequence):
            # 1-based residue indexing
            residue_pos = residue_idx + 1

            # Create the row ID
            row_id = f"{target_id}_{residue_pos}"

            # Initialize the row with basic info
            new_row = {
                "ID": row_id,
                "resname": residue_name,
                "resid": residue_pos,
            }

            # Add coordinates for each model
            for model_id in range(5):  # Always process models 0-4
                if model_id not in coords_by_target_model[target_id]:
                    if verbose:
                        print(f"Warning: Model {model_id} missing for {target_id}")
                    # If model missing, use zeros
                    new_row[f"x_{model_id+1}"] = 0.0
                    new_row[f"y_{model_id+1}"] = 0.0
                    new_row[f"z_{model_id+1}"] = 0.0
                else:
                    # Get coordinates for this residue from this model
                    coords = coords_by_target_model[target_id][model_id]

                    if residue_idx < len(coords):
                        x, y, z = coords[residue_idx]
                        new_row[f"x_{model_id+1}"] = float(x)  # Ensure float format
                        new_row[f"y_{model_id+1}"] = float(y)
                        new_row[f"z_{model_id+1}"] = float(z)
                    else:
                        if verbose:
                            print(
                                f"Warning: Residue {residue_pos} out of range for {target_id} model {model_id}"
                            )
                        # Handle case where prediction is shorter than sequence
                        new_row[f"x_{model_id+1}"] = 0.0
                        new_row[f"y_{model_id+1}"] = 0.0
                        new_row[f"z_{model_id+1}"] = 0.0

            new_rows.append(new_row)

    # Create the new submission dataframe
    columns = ["ID", "resname", "resid"]
    for model_id in range(1, 6):  # Models 1-5 in output
        columns.extend([f"x_{model_id}", f"y_{model_id}", f"z_{model_id}"])

    new_submission = pd.DataFrame(new_rows, columns=columns)

    if verbose:
        print(f"Created {len(new_submission)} rows in the reformatted submission")
        print(f"Column names: {new_submission.columns.tolist()}")

    return new_submission


def validate_submission(new_submission, sequence_dict, verbose=False):
    """Validate the reformatted submission."""
    if verbose:
        print("Validating reformatted submission...")

    # Check if all targets are included
    submission_targets = set(
        [row.split("_")[0] for row in new_submission["ID"].unique()]
    )
    missing_targets = set(sequence_dict.keys()) - submission_targets
    if missing_targets:
        print(f"Warning: Missing targets in submission: {missing_targets}")

    # Check residue counts per target
    target_counts = {}
    for row_id in new_submission["ID"]:
        target_id = row_id.split("_")[0]
        target_counts[target_id] = target_counts.get(target_id, 0) + 1

    for target_id, count in target_counts.items():
        if target_id in sequence_dict and count != len(sequence_dict[target_id]):
            print(
                f"Warning: Target {target_id} has {count} residues in submission but {len(sequence_dict[target_id])} in sequence"
            )

    # Check for null values in coordinate columns
    for col in new_submission.columns:
        if col.startswith("x_") or col.startswith("y_") or col.startswith("z_"):
            null_count = new_submission[col].isnull().sum()
            if null_count > 0:
                print(f"Warning: {null_count} null values in column {col}")

    if verbose:
        print("Validation complete.")


def main():
    """Main function to reformat the submission."""
    args = parse_arguments()

    # Load data
    submission_df, sequences_df = load_data(
        args.input, args.sequences, verbose=args.verbose
    )

    # Parse sequences
    sequence_dict = parse_sequences(sequences_df, verbose=args.verbose)

    # Parse coordinates
    coords_by_target_model = parse_coordinates(submission_df, verbose=args.verbose)

    # Reformat submission
    new_submission = reformat_submission(
        sequence_dict, coords_by_target_model, verbose=args.verbose
    )

    # Validate submission
    validate_submission(new_submission, sequence_dict, verbose=args.verbose)

    # Save the reformatted submission
    new_submission.to_csv(args.output, index=False)
    print(f"Reformatted submission saved to: {args.output}")


if __name__ == "__main__":
    main()