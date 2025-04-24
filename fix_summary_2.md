# RNA 3D Structure Training Pipeline Fixes

## Issues Fixed

1. **CurriculumManager Missing Parameter**
   - The `get_filtered_dataset` method in `pipeline/src/utils/curriculum.py` did not accept the `length_key` parameter that was being passed from `train_enhanced_model_fixed.py`.
   - Modified the method signature to accept and use this optional function for extracting length from a sample.

2. **TypeError in Loss Function**
   - Fixed a `TypeError` in `compute_angle_loss` in `src/losses.py` which was failing when receiving a list of tensors instead of a single tensor for `true_angles`.
   - Enhanced the function to handle both tensor and list inputs by checking the input type and properly converting list inputs to a tensor with proper device placement.

## Validation

- Created and ran a minimal test script that successfully completes the following:
  - Loads datasets
  - Applies curriculum filtering
  - Creates dataloaders
  - Performs one epoch of training
  - Validates on a test set
  - Saves checkpoint

## Next Steps

1. The full production training script can now be executed with:
   ```bash
   ./scripts/run_production_training_fixed.sh
   ```

2. There are still SVD-related warnings during training, but they're just warnings about falling back to a simpler alignment method and don't block the training process.

3. Future improvements could include:
   - Updating the SVD implementation to properly handle half-precision (mixed precision) tensors
   - Creating additional validation steps during training to monitor model quality
   - Adding more error handling for edge cases in the dataset

## Technical Details

1. **CurriculumManager Fix**:
   ```python
   def get_filtered_dataset(self, dataset: Dataset, length_key=None) -> Dataset:
       # Now properly handles the length_key parameter
       if length_key is not None:
           try:
               length = length_key(sample)
           except Exception as e:
               logger.warning(f"Error using length_key function on sample {i}: {e}")
               length = None
   ```

2. **Loss Function Fix**:
   ```python
   def compute_angle_loss(
       pred_angles: torch.Tensor,
       true_angles: Union[torch.Tensor, List[torch.Tensor]],
       # ...
   ):
       # Now handles both tensor and list inputs
       if isinstance(true_angles, list):
           # Convert list to a proper tensor
           true_angles_tensor = torch.zeros(batch_size, seq_len, num_features, 
                                          dtype=dtype, device=device)
           # ...process list items...
           true_angles = true_angles_tensor
   ```

These changes ensure the curriculum learning is properly implemented and the training process can handle varying data formats coming from the collation function.