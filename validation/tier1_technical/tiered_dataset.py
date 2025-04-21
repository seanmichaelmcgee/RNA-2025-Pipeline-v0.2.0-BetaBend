import os
import torch
import numpy as np
from src.data_loading import RNADataset

# Tier-specific configuration
TIER_CONFIG = {
    "tier1": {
        "required_features": ["dihedral", "thermo"],
        "optional_features": ["mi"],
        "sample_size": 5,
        "allow_mock_data": True,
        "verify_all_features": False,
        "description": "Fast technical validation"
    },
    "tier2": {
        "required_features": ["dihedral", "thermo"],
        "optional_features": ["mi"],
        "sample_size": 15,
        "allow_mock_data": False,
        "verify_all_features": True,
        "description": "Comprehensive scientific validation"
    },
    "tier3": {
        "required_features": ["dihedral", "thermo", "mi"],
        "optional_features": [],
        "sample_size": None,  # Use all available
        "allow_mock_data": False,
        "verify_all_features": True,
        "description": "Full model evaluation"
    }
}

def debug_feature_path(target_id, feature_type, data_dir):
    """Print and verify the exact path being used to load features."""
    if feature_type == "dihedral":
        path = os.path.join(data_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
    elif feature_type == "thermo":
        path = os.path.join(data_dir, "thermo_features", f"{target_id}_thermo_features.npz")
    elif feature_type == "mi":
        path = os.path.join(data_dir, "mi_features", f"{target_id}_mi_features.npz")
    else:
        path = None
        
    if path:
        exists = os.path.exists(path)
        print(f"Path for {target_id} {feature_type}: {path}")
        print(f"File exists: {exists}")
        
        if exists:
            # Try to open and list contents
            try:
                with np.load(path) as data:
                    print(f"File contains keys: {list(data.keys())}")
                    
                    # Check shapes of key elements
                    if feature_type == "dihedral" and "features" in data:
                        print(f"Dihedral features shape: {data['features'].shape}")
                    elif feature_type == "thermo" and "pairing_probs" in data:
                        print(f"Pairing probs shape: {data['pairing_probs'].shape}")
                    elif feature_type == "mi" and "coupling_matrix" in data:
                        print(f"Coupling matrix shape: {data['coupling_matrix'].shape}")
            except Exception as e:
                print(f"Error opening file: {e}")
    
    return path, exists if path else (None, False)

def inspect_npz_file(file_path):
    """Inspect an NPZ file and return detailed information about its contents."""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
        
    try:
        with np.load(file_path) as data:
            info = {
                "keys": list(data.keys()),
                "shapes": {k: data[k].shape for k in data.keys()},
                "dtypes": {k: str(data[k].dtype) for k in data.keys()},
                "file_size": os.path.getsize(file_path) / 1024,  # KB
            }
            return info
    except Exception as e:
        return f"Error inspecting file: {e}"

def test_feature_loading(target_ids, data_dir, required_features=None, verbose=True):
    """Test feature loading for a list of target IDs."""
    if required_features is None:
        required_features = ["dihedral", "thermo"]  # Default minimum requirements
        
    results = []
    
    for target_id in target_ids:
        result = {
            "target_id": target_id,
            "status": "unknown",
            "errors": [],
            "features_found": [],
            "sequence_length": None
        }
        
        try:
            # Test dihedral features
            dihedral_path = os.path.join(data_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
            if os.path.exists(dihedral_path):
                result["features_found"].append("dihedral")
                with np.load(dihedral_path) as data:
                    if "features" in data:
                        result["sequence_length"] = len(data["features"])
            
            # Test thermo features
            thermo_path = os.path.join(data_dir, "thermo_features", f"{target_id}_thermo_features.npz")
            if os.path.exists(thermo_path):
                result["features_found"].append("thermo")
                if result["sequence_length"] is None:
                    with np.load(thermo_path) as data:
                        if "pairing_probs" in data:
                            result["sequence_length"] = len(data["pairing_probs"])
            
            # Test MI features
            mi_path = os.path.join(data_dir, "mi_features", f"{target_id}_mi_features.npz")
            if os.path.exists(mi_path):
                result["features_found"].append("mi")
                if result["sequence_length"] is None:
                    with np.load(mi_path) as data:
                        if "coupling_matrix" in data:
                            result["sequence_length"] = len(data["coupling_matrix"])
            
            # Check if we have all required features
            if all(feat in result["features_found"] for feat in required_features):
                result["status"] = "pass"
            else:
                result["status"] = "fail"
                missing = [feat for feat in required_features if feat not in result["features_found"]]
                result["errors"].append(f"Missing required features: {missing}")
                
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(str(e))
        
        results.append(result)
    
    if verbose:
        # Count statistics
        total = len(results)
        passed = sum(1 for r in results if r["status"] == "pass")
        failed = sum(1 for r in results if r["status"] == "fail")
        errored = sum(1 for r in results if r["status"] == "error")
        
        print(f"Test Results: {passed}/{total} passed, {failed} failed, {errored} errored")
        
        # Feature availability
        feature_counts = {
            "dihedral": sum(1 for r in results if "dihedral" in r["features_found"]),
            "thermo": sum(1 for r in results if "thermo" in r["features_found"]),
            "mi": sum(1 for r in results if "mi" in r["features_found"])
        }
        
        print("\nFeature Availability:")
        for feat, count in feature_counts.items():
            print(f"  {feat}: {count}/{total} ({count/total*100:.1f}%)")
        
        # Display detailed results for failed/errored tests
        if failed + errored > 0:
            print("\nDetailed Results for Failed/Errored Tests:")
            for result in results:
                if result["status"] != "pass":
                    print(f"{result['target_id']}: {result['status'].upper()}")
                    print(f"  Features found: {', '.join(result['features_found'])}")
                    for error in result["errors"]:
                        print(f"  Error: {error}")
        
    return results

def find_valid_ids_for_validation(data_dir, tier="tier1", subset_size=None, seed=42):
    """Find IDs that meet the requirements for a specific validation tier."""
    np.random.seed(seed)
    
    # Get tier configuration
    tier_config = TIER_CONFIG[tier]
    required_features = tier_config["required_features"]
    
    # Use configured sample size if not specified
    if subset_size is None:
        subset_size = tier_config["sample_size"]
        if subset_size is None:  # For tier3 which uses all available
            subset_size = float('inf')  # Will be limited by available data
    
    # Get all dihedral files as baseline
    dihedral_dir = os.path.join(data_dir, "dihedral_features")
    all_files = [f for f in os.listdir(dihedral_dir) if f.endswith("_dihedral_features.npz")]
    all_ids = [f.split("_dihedral")[0] for f in all_files]
    
    # Test random IDs to find enough valid ones
    test_size = min(50, len(all_ids))  # Test a reasonable number
    test_ids = np.random.choice(all_ids, test_size, replace=False)
    test_results = test_feature_loading(test_ids, data_dir, required_features, verbose=False)
    
    # Filter for passing IDs
    valid_ids = [r["target_id"] for r in test_results if r["status"] == "pass"]
    
    # If we don't have enough valid IDs, try more
    if len(valid_ids) < subset_size and len(valid_ids) < len(all_ids):
        remaining_ids = [id for id in all_ids if id not in test_ids]
        if remaining_ids:
            more_test_ids = np.random.choice(remaining_ids, 
                                           min(100, len(remaining_ids)), 
                                           replace=False)
            more_results = test_feature_loading(more_test_ids, data_dir, required_features, verbose=False)
            more_valid = [r["target_id"] for r in more_results if r["status"] == "pass"]
            valid_ids.extend(more_valid)
    
    # Print tier-specific information
    print(f"\nValidation Tier: {tier} - {tier_config['description']}")
    print(f"Required features: {required_features}")
    print(f"Optional features: {tier_config['optional_features']}")
    print(f"Found {len(valid_ids)} valid IDs out of {len(all_ids)} total IDs")
    
    # Take the requested subset
    if len(valid_ids) <= subset_size:
        final_ids = valid_ids
    else:
        final_ids = np.random.choice(valid_ids, subset_size, replace=False).tolist()
    
    print(f"Selected {len(final_ids)} IDs for validation")
    return final_ids

def load_features_for_validation(target_id, data_dir, tier="tier1"):
    """Load features for validation, with robust error handling and tier-specific behavior."""
    tier_config = TIER_CONFIG[tier]
    allow_mock_data = tier_config["allow_mock_data"]
    required_features = tier_config["required_features"]
    
    features = {}
    sequence_length = None
    
    # Load dihedral features
    dihedral_path = os.path.join(data_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
    if os.path.exists(dihedral_path):
        try:
            with np.load(dihedral_path) as data:
                if "features" in data:
                    features["dihedral"] = {"features": data["features"]}
                    sequence_length = len(data["features"])
        except Exception as e:
            print(f"Warning: Error loading dihedral features for {target_id}: {e}")
    
    # Load thermo features
    thermo_path = os.path.join(data_dir, "thermo_features", f"{target_id}_thermo_features.npz")
    if os.path.exists(thermo_path):
        try:
            with np.load(thermo_path) as data:
                thermo_data = {}
                if "pairing_probs" in data:
                    thermo_data["pairing_probs"] = data["pairing_probs"]
                    if sequence_length is None:
                        sequence_length = len(data["pairing_probs"])
                
                for key in ["positional_entropy", "accessibility"]:
                    if key in data:
                        thermo_data[key] = data[key]
                
                features["thermo"] = thermo_data
        except Exception as e:
            print(f"Warning: Error loading thermo features for {target_id}: {e}")
    
    # Load MI features
    mi_path = os.path.join(data_dir, "mi_features", f"{target_id}_mi_features.npz")
    if os.path.exists(mi_path):
        try:
            with np.load(mi_path) as data:
                mi_data = {}
                if "coupling_matrix" in data:
                    mi_data["coupling_matrix"] = data["coupling_matrix"]
                    if sequence_length is None:
                        sequence_length = len(data["coupling_matrix"])
                
                features["mi"] = mi_data
        except Exception as e:
            print(f"Warning: Error loading MI features for {target_id}: {e}")
    
    # If sequence length is still None, use a default
    if sequence_length is None:
        sequence_length = 100
        print(f"Warning: Could not determine sequence length for {target_id}, using default: {sequence_length}")
    
    # Create mock features if allowed by tier and missing required features
    for feature_type in required_features:
        if feature_type not in features or not features[feature_type]:
            if allow_mock_data:
                if feature_type == "dihedral":
                    features["dihedral"] = {
                        "features": np.zeros((sequence_length, 4))
                    }
                    print(f"Warning: Created mock dihedral features for {target_id}")
                elif feature_type == "thermo":
                    features["thermo"] = {
                        "pairing_probs": np.zeros((sequence_length, sequence_length)),
                        "positional_entropy": np.zeros(sequence_length),
                        "accessibility": np.zeros(sequence_length)
                    }
                    print(f"Warning: Created mock thermo features for {target_id}")
                elif feature_type == "mi":
                    features["mi"] = {
                        "coupling_matrix": np.zeros((sequence_length, sequence_length))
                    }
                    print(f"Warning: Created mock MI features for {target_id}")
            else:
                # Raise error if mock data not allowed in this tier
                raise ValueError(f"Missing required feature '{feature_type}' for ID '{target_id}' in tier '{tier}' which does not allow mock data")
    
    return features, sequence_length

class TieredRNADataset(RNADataset):
    """Custom RNA dataset for validation with tier-specific behavior."""
    
    def __init__(self, data_dir, tier="tier1", filter_ids=None, custom_config=None):
        """
        Initialize a tiered validation dataset.
        
        Args:
            data_dir: Path to data directory
            tier: Validation tier ("tier1", "tier2", or "tier3")
            filter_ids: List of target IDs to include (will be validated against tier requirements)
            custom_config: Optional custom configuration to override tier defaults
        """
        # Validate tier
        if tier not in TIER_CONFIG and custom_config is None:
            raise ValueError(f"Invalid tier: {tier}. Must be one of {list(TIER_CONFIG.keys())}")
        
        # Use empty paths to skip default loading
        super().__init__(
            sequences_csv_path="",  # Will be set up manually
            labels_csv_path=None,   # Will be set up manually
            features_dir=data_dir,
            split_fn=None,
            temporal_cutoff=None,
            use_validation_set=True,  # Always use validation set for validation
            require_features=False
        )
        
        # Store tier and config
        self.tier = tier
        self.tier_config = custom_config if custom_config else TIER_CONFIG[tier]
        self.required_features = self.tier_config["required_features"]
        self.allow_mock_data = self.tier_config["allow_mock_data"]
        
        # Validate filter_ids against tier requirements
        if filter_ids:
            valid_results = test_feature_loading(
                filter_ids, 
                data_dir, 
                required_features=self.required_features, 
                verbose=False
            )
            
            valid_ids = [r["target_id"] for r in valid_results if r["status"] == "pass"]
            
            if not self.allow_mock_data:
                # Use only fully valid IDs
                self.target_ids = valid_ids.copy()
                self.filtered_sequences = valid_ids.copy()
            else:
                # Can use all IDs with mock data filling in gaps
                self.target_ids = filter_ids.copy()
                self.filtered_sequences = filter_ids.copy()
        else:
            # No filter IDs provided, find valid ones for this tier
            subset_size = self.tier_config["sample_size"]
            self.target_ids = find_valid_ids_for_validation(
                data_dir, 
                tier=tier, 
                subset_size=subset_size
            )
            self.filtered_sequences = self.target_ids.copy()
        
        # For collate_fn to work properly
        self.sequences = ["A" * 100] * len(self.target_ids)  # Dummy sequences
        
        # Create an availability cache
        self._availability_cache = {}
        for target_id in self.target_ids:
            self._availability_cache[target_id] = self._check_feature_availability(target_id)
        
        print(f"Created TieredRNADataset for tier '{tier}' with {len(self.target_ids)} samples")
    
    def _check_feature_availability(self, target_id):
        """Check which features are available for a target ID."""
        availability = {
            "dihedral": False, 
            "thermo": False, 
            "mi": False
        }
        
        # Check dihedral features
        dihedral_path = os.path.join(self.features_dir, "dihedral_features", f"{target_id}_dihedral_features.npz")
        if os.path.exists(dihedral_path):
            availability["dihedral"] = True
        
        # Check thermo features
        thermo_path = os.path.join(self.features_dir, "thermo_features", f"{target_id}_thermo_features.npz")
        if os.path.exists(thermo_path):
            availability["thermo"] = True
        
        # Check MI features
        mi_path = os.path.join(self.features_dir, "mi_features", f"{target_id}_mi_features.npz")
        if os.path.exists(mi_path):
            availability["mi"] = True
        
        return availability
    
    def get_tier_stats(self):
        """Get statistics about feature availability across the dataset for this tier."""
        total = len(self.target_ids)
        if total == 0:
            return {"error": "No IDs in dataset"}
        
        # Count features
        feature_counts = {
            "dihedral": sum(1 for tid in self.target_ids if self._availability_cache[tid]["dihedral"]),
            "thermo": sum(1 for tid in self.target_ids if self._availability_cache[tid]["thermo"]),
            "mi": sum(1 for tid in self.target_ids if self._availability_cache[tid]["mi"])
        }
        
        # Calculate percentages
        feature_percentages = {
            k: (v / total * 100) for k, v in feature_counts.items()
        }
        
        # Count IDs with all required features
        all_required = sum(
            1 for tid in self.target_ids 
            if all(self._availability_cache[tid][feat] for feat in self.required_features)
        )
        
        return {
            "tier": self.tier,
            "description": self.tier_config["description"],
            "total_ids": total,
            "feature_counts": feature_counts,
            "feature_percentages": feature_percentages,
            "ids_with_all_required": all_required,
            "ids_with_all_required_percent": (all_required / total * 100) if total > 0 else 0,
            "required_features": self.required_features,
            "optional_features": self.tier_config["optional_features"],
            "allow_mock_data": self.allow_mock_data
        }
            
    # Override __getitem__ to use robust feature loading with tier-specific behavior
    def __getitem__(self, idx):
        """Get features for a single RNA sequence with tier-specific handling."""
        # Get target_id from filtered sequences
        target_id = self.filtered_sequences[idx]
        
        # Load features with robust handling and tier-specific behavior
        features, sequence_length = load_features_for_validation(
            target_id, 
            self.features_dir,
            tier=self.tier
        )
        
        # Create sample dictionary with all required keys
        sample = {
            "target_id": target_id,
            "ids": target_id,
            "sequence_int": torch.zeros(sequence_length, dtype=torch.long),
            "length": sequence_length,
            "mask": torch.ones(sequence_length, dtype=torch.bool),
            "atom_mask": torch.ones(sequence_length, dtype=torch.bool),
            "angle_mask": torch.ones(sequence_length, dtype=torch.bool),
        }
            
        # Add dihedral features
        if "dihedral" in features:
            sample["dihedral_features"] = torch.tensor(
                features["dihedral"]["features"], dtype=torch.float32
            )
            sample["dihedral_angles"] = torch.tensor(
                features["dihedral"]["features"], dtype=torch.float32
            )
        else:
            # Create default zero tensor
            sample["dihedral_features"] = torch.zeros(
                (sequence_length, 4), dtype=torch.float32
            )
            sample["dihedral_angles"] = torch.zeros(
                (sequence_length, 4), dtype=torch.float32
            )
            
        # Add thermodynamic features
        if "thermo" in features:
            if "pairing_probs" in features["thermo"]:
                sample["pairing_probs"] = torch.tensor(
                    features["thermo"]["pairing_probs"], dtype=torch.float32
                )
            else:
                sample["pairing_probs"] = torch.zeros(
                    (sequence_length, sequence_length), dtype=torch.float32
                )
                
            if "positional_entropy" in features["thermo"]:
                sample["positional_entropy"] = torch.tensor(
                    features["thermo"]["positional_entropy"], dtype=torch.float32
                )
            else:
                sample["positional_entropy"] = torch.zeros(
                    sequence_length, dtype=torch.float32
                )
                
            if "accessibility" in features["thermo"]:
                sample["accessibility"] = torch.tensor(
                    features["thermo"]["accessibility"], dtype=torch.float32
                )
            else:
                sample["accessibility"] = torch.zeros(
                    sequence_length, dtype=torch.float32
                )
        else:
            # Create default zero tensors
            sample["pairing_probs"] = torch.zeros(
                (sequence_length, sequence_length), dtype=torch.float32
            )
            sample["positional_entropy"] = torch.zeros(
                sequence_length, dtype=torch.float32
            )
            sample["accessibility"] = torch.zeros(
                sequence_length, dtype=torch.float32
            )
            
        # Add evolutionary features
        if "mi" in features and "coupling_matrix" in features["mi"]:
            sample["coupling_matrix"] = torch.tensor(
                features["mi"]["coupling_matrix"], dtype=torch.float32
            )
        else:
            # Create default zero tensor
            sample["coupling_matrix"] = torch.zeros(
                (sequence_length, sequence_length), dtype=torch.float32
            )
            
        # Simulate atom positions for testing
        sample["atom_positions"] = torch.rand(
            (sequence_length, 3), dtype=torch.float32
        )
        
        return sample
        
    # Override collate function to ensure all required keys are preserved
    def collate_fn(self, batch):
        """Collate batch of samples, ensuring all required keys for the model are preserved."""
        # Get batch size and maximum sequence length
        batch_size = len(batch)
        max_len = max(sample["length"] for sample in batch)
        
        # Extract IDs
        ids = [sample["target_id"] for sample in batch]
        
        # Initialize output dictionary with required fields
        output = {
            "ids": ids,
            "lengths": torch.tensor([sample["length"] for sample in batch], dtype=torch.long),
        }
        
        # Required keys for the model (based on RNAFoldingModel.forward)
        required_keys = [
            "sequence_int",
            "dihedral_features",
            "pairing_probs",
            "positional_entropy",
            "accessibility",
            "coupling_matrix",
            "mask"
        ]
        
        # Process each tensor in the batch, ensuring required keys are included
        for key in batch[0].keys():
            if key in ["target_id", "length", "ids"]:
                continue  # Already processed
                
            if isinstance(batch[0][key], torch.Tensor):
                # Process tensor based on its shape
                sample_shape = batch[0][key].shape
                
                if len(sample_shape) == 1:
                    # 1D tensor (sequence, per-residue features)
                    padded = []
                    for sample in batch:
                        tensor = sample[key]
                        padded_tensor = torch.zeros(max_len, dtype=tensor.dtype, device=tensor.device)
                        padded_tensor[:len(tensor)] = tensor
                        padded.append(padded_tensor)
                    output[key] = torch.stack(padded)
                    
                elif len(sample_shape) == 2 and sample_shape[0] == sample_shape[1]:
                    # 2D square matrix (L, L)
                    padded = []
                    for sample in batch:
                        tensor = sample[key]
                        padded_tensor = torch.zeros(
                            (max_len, max_len), dtype=tensor.dtype, device=tensor.device
                        )
                        padded_tensor[:len(tensor), :len(tensor)] = tensor
                        padded.append(padded_tensor)
                    output[key] = torch.stack(padded)
                    
                elif len(sample_shape) == 2:
                    # 2D tensor with feature dimension (L, D)
                    feature_dim = sample_shape[1]
                    padded = []
                    for sample in batch:
                        tensor = sample[key]
                        padded_tensor = torch.zeros(
                            (max_len, feature_dim), dtype=tensor.dtype, device=tensor.device
                        )
                        padded_tensor[:len(tensor)] = tensor
                        padded.append(padded_tensor)
                    output[key] = torch.stack(padded)
        
        # Create masks if not already included
        if "mask" not in output:
            mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
            for i, sample in enumerate(batch):
                mask[i, :sample["length"]] = True
            output["mask"] = mask
        
        if "atom_mask" not in output:
            atom_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
            for i, sample in enumerate(batch):
                atom_mask[i, :sample["length"]] = True
            output["atom_mask"] = atom_mask
        
        if "angle_mask" not in output:
            angle_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
            for i, sample in enumerate(batch):
                angle_mask[i, :sample["length"]] = True
            output["angle_mask"] = angle_mask
        
        # Ensure all required keys are present
        for key in required_keys:
            if key not in output:
                print(f"Warning: Required key '{key}' not in output batch. Adding dummy tensor.")
                # Add a dummy tensor with appropriate shape
                if key in ["sequence_int"]:
                    # 1D integer tensor
                    output[key] = torch.zeros((batch_size, max_len), dtype=torch.long)
                elif key in ["dihedral_features"]:
                    # 2D tensor with feature dimension
                    output[key] = torch.zeros((batch_size, max_len, 4), dtype=torch.float32)
                elif key in ["pairing_probs", "coupling_matrix"]:
                    # 2D square matrix
                    output[key] = torch.zeros((batch_size, max_len, max_len), dtype=torch.float32)
                elif key in ["positional_entropy", "accessibility"]:
                    # 1D float tensor
                    output[key] = torch.zeros((batch_size, max_len), dtype=torch.float32)
        
        return output