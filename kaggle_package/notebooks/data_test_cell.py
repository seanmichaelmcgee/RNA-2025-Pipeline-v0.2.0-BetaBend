"""
Data Loading Test Cell

Copy this into the notebook after path initialization to test data loading
"""

print("=== Testing Data Loading ===")

# Check test sequences
try:
    print(f"\nChecking test sequences at: {TEST_SEQUENCES_PATH}")
    if os.path.exists(TEST_SEQUENCES_PATH):
        test_seqs_df = pd.read_csv(TEST_SEQUENCES_PATH)
        print(f"✅ Successfully loaded test sequences: {len(test_seqs_df)} sequences found")
        print(f"✅ Sample sequences: {test_seqs_df['sequence_id'].values[:5]}")
    else:
        print(f"❌ Test sequences file not found at: {TEST_SEQUENCES_PATH}")
except Exception as e:
    print(f"❌ Error loading test sequences: {str(e)}")

# Check features directory
print(f"\nChecking features directory at: {FEATURES_DIR}")
if os.path.exists(FEATURES_DIR):
    print(f"✅ Features directory exists")
    
    # Check each feature subdirectory
    for subdir in ['dihedral_features', 'thermo_features', 'evolutionary_features']:
        feature_dir = os.path.join(FEATURES_DIR, subdir)
        if os.path.exists(feature_dir):
            feature_files = os.listdir(feature_dir)
            print(f"✅ {subdir}: Directory exists with {len(feature_files)} files")
            if feature_files:
                print(f"   Sample files: {feature_files[:3]}")
        else:
            print(f"❌ {subdir}: Directory not found")
else:
    print(f"❌ Features directory not found at: {FEATURES_DIR}")

# Test loading specific feature files
print("\n=== Testing Feature Loading ===")

if 'test_seqs_df' in locals() and len(test_seqs_df) > 0:
    # Get a sample sequence ID
    sample_id = test_seqs_df['sequence_id'].values[0]
    print(f"Testing feature loading for sample ID: {sample_id}")
    
    # Test the feature loading function with robust error handling
    try:
        from src.data_loading import load_precomputed_features
        # Patch the function if needed
        if 'fixed_load_precomputed_features' in globals():
            print("Using patched feature loading function")
            features = fixed_load_precomputed_features(sample_id, FEATURES_DIR)
        else:
            print("Using standard feature loading function")
            features = load_precomputed_features(sample_id, FEATURES_DIR)
            
        # Check what features were loaded
        print("\nFeatures loaded:")
        for feature_type, feature_data in features.items():
            if feature_data is not None:
                print(f"✅ {feature_type}: Loaded successfully")
                # Print shape if available
                if isinstance(feature_data, dict) and 'features' in feature_data:
                    shape = feature_data['features'].shape
                    print(f"   Shape: {shape}")
                    # Check for NaN values
                    if hasattr(feature_data['features'], 'size'):
                        nan_count = np.isnan(feature_data['features']).sum()
                        if nan_count > 0:
                            print(f"⚠️ Warning: {nan_count} NaN values found")
                        else:
                            print(f"✅ No NaN values found")
            else:
                print(f"❌ {feature_type}: Not loaded or not available")
                
    except Exception as e:
        print(f"❌ Error testing feature loading: {str(e)}")
        traceback.print_exc()
else:
    print("❌ Cannot test feature loading without test sequences")

# Test creating a dataset and dataloader
print("\n=== Testing Dataset Creation ===")
try:
    if 'test_seqs_df' in locals() and len(test_seqs_df) > 0:
        # Create a small test dataset
        sample_ids = test_seqs_df['sequence_id'].values[:3]
        sample_sequences = test_seqs_df['sequence'].values[:3]
        
        print(f"Creating dataset with {len(sample_ids)} samples")
        
        # Create dataset with robust error handling
        try:
            dataset = RNADataset(
                sequence_ids=sample_ids,
                sequences=sample_sequences,
                features_dir=FEATURES_DIR,
                labels=None  # No labels for test data
            )
            print(f"✅ Dataset created successfully with {len(dataset)} samples")
            
            # Test accessing an item
            try:
                sample = dataset[0]
                print("✅ Successfully accessed dataset item")
                print(f"   Keys in sample: {list(sample.keys())}")
                for k, v in sample.items():
                    if hasattr(v, 'shape'):
                        print(f"   {k}: shape {v.shape}")
                    else:
                        print(f"   {k}: {type(v)}")
            except Exception as e:
                print(f"❌ Error accessing dataset item: {str(e)}")
                
            # Test creating a dataloader
            try:
                dataloader = create_data_loader(
                    dataset,
                    batch_size=2,
                    shuffle=False,
                    num_workers=0
                )
                print(f"✅ DataLoader created successfully")
                
                # Test iterating the dataloader
                try:
                    batch = next(iter(dataloader))
                    print("✅ Successfully retrieved batch from dataloader")
                    print(f"   Keys in batch: {list(batch.keys())}")
                    for k, v in batch.items():
                        if hasattr(v, 'shape'):
                            print(f"   {k}: shape {v.shape}")
                        else:
                            print(f"   {k}: {type(v)}")
                except Exception as e:
                    print(f"❌ Error iterating dataloader: {str(e)}")
                    
            except Exception as e:
                print(f"❌ Error creating dataloader: {str(e)}")
                
        except Exception as e:
            print(f"❌ Error creating dataset: {str(e)}")
            traceback.print_exc()
    else:
        print("❌ Cannot test dataset creation without test sequences")
except Exception as e:
    print(f"❌ Error in dataset testing: {str(e)}")
    traceback.print_exc()

print("\nData loading test complete!")