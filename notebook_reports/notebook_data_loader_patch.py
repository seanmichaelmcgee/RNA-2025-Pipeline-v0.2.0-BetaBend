# Insert this code at the beginning of the notebook, after importing data_loading

# Patch the data loading function to handle inconsistent feature file formats
def fixed_load_precomputed_features(
    target_id, features_dir, temporal_cutoff=None
):
    """
    Enhanced version of load_precomputed_features that handles inconsistent feature file formats.
    """
    import os
    import numpy as np
    import warnings
    import pandas as pd
    
    features = {}

    # 1. Load dihedral features
    dihedral_path = os.path.join(
        features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz"
    )
    if os.path.exists(dihedral_path):
        try:
            with np.load(dihedral_path) as data:
                # Check feature generation date if available for temporal cutoff
                if temporal_cutoff is not None and "metadata" in data:
                    try:
                        metadata_str = str(data["metadata"])
                        if "extraction_timestamp" in metadata_str:
                            timestamp_part = metadata_str.split("extraction_timestamp")[
                                1
                            ].split("'")[1]
                            generation_date = timestamp_part.split()[0]

                            if pd.to_datetime(generation_date) > pd.to_datetime(
                                temporal_cutoff
                            ):
                                warnings.warn(
                                    f"Dihedral features for {target_id} were generated after the temporal cutoff. Using zeros."
                                )
                                features["dihedral"] = None
                                return features
                    except (KeyError, IndexError, ValueError):
                        pass

                # Handle different feature file formats
                if "features" in data:
                    # Standard format
                    features["dihedral"] = {"features": data["features"].astype(np.float32)}
                elif "struct_1_features" in data:
                    # Multi-structure format with numbered structures
                    features["dihedral"] = {"features": data["struct_1_features"].astype(np.float32)}
                else:
                    # Unknown format - warn and use None
                    warnings.warn(f"Dihedral features file for {target_id} has unexpected format. Using zeros.")
                    features["dihedral"] = None
                    return features
                
                # Handle NaN values if present
                if features["dihedral"] is not None and np.isnan(features["dihedral"]["features"]).any():
                    features["dihedral"]["features"] = np.nan_to_num(
                        features["dihedral"]["features"], nan=0.0
                    )
        except Exception as e:
            # Handle any errors in loading
            warnings.warn(f"Error loading dihedral features for {target_id}: {str(e)}. Using zeros.")
            features["dihedral"] = None
    else:
        # For test data or if file is missing
        features["dihedral"] = None
        warnings.warn(f"Dihedral features not found for {target_id}. Using zeros.")

    # 2. Load thermodynamic features (required)
    thermo_path = os.path.join(
        features_dir, "thermo_features", f"{target_id}_thermo_features.npz"
    )
    if not os.path.exists(thermo_path):
        raise ValueError(
            f"Thermodynamic features not found for {target_id}. Required for prediction."
        )

    try:
        with np.load(thermo_path) as data:
            # Check feature generation date if available for temporal cutoff
            if temporal_cutoff is not None and "generation_date" in data:
                generation_date = str(data["generation_date"])
                if pd.to_datetime(generation_date) > pd.to_datetime(temporal_cutoff):
                    warnings.warn(
                        f"Thermo features for {target_id} were generated after the temporal cutoff. Using zeros."
                    )
                    features["thermo"] = None
                    return features

            # Extract key arrays and scalar values
            thermo_features = {}

            # Get pairing probabilities matrix (critical)
            if "pairing_probs" in data:
                thermo_features["pairing_probs"] = data["pairing_probs"].astype(np.float32)
            elif "base_pair_probs" in data:
                thermo_features["pairing_probs"] = data["base_pair_probs"].astype(np.float32)
            else:
                raise ValueError(f"No pairing probabilities found in {target_id} thermo features")

            # Handle NaN values in pairing probabilities
            if np.isnan(thermo_features["pairing_probs"]).any():
                thermo_features["pairing_probs"] = np.nan_to_num(
                    thermo_features["pairing_probs"], nan=0.0
                )

            # Get positional entropy (optional)
            if "positional_entropy" in data:
                thermo_features["positional_entropy"] = data["positional_entropy"].astype(
                    np.float32
                )
            else:
                # Calculate from pairing probabilities if missing
                pair_probs = thermo_features["pairing_probs"]
                row_entropies = -np.sum(
                    pair_probs * np.log2(pair_probs + 1e-10), axis=1
                )
                thermo_features["positional_entropy"] = row_entropies

            # Get accessibility (optional)
            if "accessibility" in data:
                thermo_features["accessibility"] = data["accessibility"].astype(np.float32)
            else:
                # Calculate from pairing probabilities if missing
                pair_probs = thermo_features["pairing_probs"]
                accessibilities = 1.0 - np.sum(pair_probs, axis=1)
                thermo_features["accessibility"] = np.maximum(0.0, accessibilities)

            features["thermo"] = thermo_features
    except Exception as e:
        raise ValueError(f"Error loading thermodynamic features for {target_id}: {str(e)}")

    # 3. Load evolutionary coupling features (optional)
    mi_path = os.path.join(
        features_dir, "evolutionary_features", f"{target_id}_evolutionary_features.npz"
    )
    if os.path.exists(mi_path):
        try:
            with np.load(mi_path) as data:
                # Check feature generation date if available for temporal cutoff
                if temporal_cutoff is not None and "generation_date" in data:
                    generation_date = str(data["generation_date"])
                    if pd.to_datetime(generation_date) > pd.to_datetime(temporal_cutoff):
                        warnings.warn(
                            f"Evolutionary features for {target_id} were generated after the temporal cutoff. Using zeros."
                        )
                        features["evolutionary"] = None
                        return features

                # Extract key arrays and metadata
                evol_features = {}

                # Get coupling matrix (required for this feature type)
                if "coupling_matrix" in data:
                    coupling_matrix = data["coupling_matrix"].astype(np.float32)
                    
                    # Check if the matrix is valid (not all zeros or constant)
                    is_valid = not np.allclose(coupling_matrix, 0.0)
                    evol_features["has_valid_mi"] = is_valid
                    
                    if is_valid:
                        evol_features["coupling_matrix"] = coupling_matrix
                    else:
                        # Zero matrix case - still provide the matrix but flag it
                        evol_features["coupling_matrix"] = coupling_matrix
                        warnings.warn(f"Coupling matrix for {target_id} is all zeros or constant.")
                else:
                    # No coupling matrix found
                    evol_features["has_valid_mi"] = False
                    evol_features["coupling_matrix"] = None

                features["evolutionary"] = evol_features
        except Exception as e:
            # Handle errors in evolutionary feature loading
            warnings.warn(f"Error loading evolutionary features for {target_id}: {str(e)}. Proceeding without them.")
            features["evolutionary"] = None
    else:
        # No evolutionary features available
        features["evolutionary"] = None

    return features

# Apply the patch
from src import data_loading
original_load_precomputed_features = data_loading.load_precomputed_features
data_loading.load_precomputed_features = fixed_load_precomputed_features
print("Successfully patched data_loading.load_precomputed_features with fixed version")