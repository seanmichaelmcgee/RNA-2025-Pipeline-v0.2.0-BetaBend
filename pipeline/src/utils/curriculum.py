"""
Enhanced Curriculum Learning for RNA Structure Prediction

This module provides specialized curriculum learning strategies for RNA structure 
prediction, with a focus on sequence length-based curriculum and adaptive batch sizing.
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader

# Initialize logging
logger = logging.getLogger(__name__)

class CurriculumManager:
    """
    Curriculum learning manager for RNA 3D structure prediction, with emphasis on
    sequence length progression and adaptive batch sizing.
    """
    
    def __init__(self,
                sequence_stages: List[int] = [50, 100, 150, 200, 250, 300],
                epochs_per_stage: int = 10,
                plateau_patience: int = 3,
                min_improvement: float = 0.005,
                base_batch_size: int = 8,
                base_grad_accum_steps: int = 1,
                batch_adaptive: bool = True):
        """
        Initialize curriculum learning manager.
        
        Args:
            sequence_stages: Maximum sequence lengths for each curriculum stage
            epochs_per_stage: Minimum epochs before considering stage advancement
            plateau_patience: Epochs without improvement to trigger advancement
            min_improvement: Minimum relative improvement to not trigger plateau
            base_batch_size: Base batch size for the first curriculum stage
            base_grad_accum_steps: Base gradient accumulation steps
            batch_adaptive: Whether to dynamically adapt batch size with sequence length
        """
        self.sequence_stages = sequence_stages
        self.epochs_per_stage = epochs_per_stage
        self.plateau_patience = plateau_patience
        self.min_improvement = min_improvement
        self.base_batch_size = base_batch_size
        self.base_grad_accum_steps = base_grad_accum_steps
        self.batch_adaptive = batch_adaptive
        
        # Initialize tracking variables
        self.current_stage = 0
        self.epochs_at_current_stage = 0
        self.best_loss_at_stage = float('inf')
        self.plateau_counter = 0
        
        # Track batch parameters for each stage
        self.stage_batch_params = {}
        
        # Initialize batch parameters for the current stage
        self.curr_batch_size = base_batch_size
        self.curr_grad_accum_steps = base_grad_accum_steps
        
        # Track stage transition history
        self.stage_history = []
        
        logger.info(f"Initialized curriculum learning with {len(sequence_stages)} stages: {sequence_stages}")
        logger.info(f"Starting with stage 0, max length {self.get_current_max_length()}")
    
    def get_current_max_length(self) -> int:
        """Get maximum sequence length for the current curriculum stage."""
        if self.current_stage < len(self.sequence_stages):
            return self.sequence_stages[self.current_stage]
        return self.sequence_stages[-1]  # Use the final stage if out of bounds
    
    def get_batch_params(self) -> Tuple[int, int]:
        """Get batch size and gradient accumulation steps for current stage."""
        # Check if we have cached parameters for this stage
        if self.current_stage in self.stage_batch_params:
            return (
                self.stage_batch_params[self.current_stage]["batch_size"],
                self.stage_batch_params[self.current_stage]["grad_accum_steps"]
            )
        
        # Calculate batch parameters based on sequence length if adaptive
        if self.batch_adaptive:
            # Scale batch size inverse quadratically with sequence length
            max_len = self.get_current_max_length()
            base_len = self.sequence_stages[0]
            
            # Using quadratic scaling (attention is O(n²))
            scaling_factor = (base_len / max_len) ** 2
            
            # Apply scaling to base batch size
            batch_size = max(1, int(self.base_batch_size * scaling_factor))
            
            # Adjust to power of 2 for better GPU performance
            batch_size = 2 ** int(np.log2(batch_size))
            
            # Adjust gradient accumulation to maintain effective batch size
            if batch_size < self.base_batch_size:
                grad_accum_steps = max(
                    self.base_grad_accum_steps, 
                    self.base_grad_accum_steps * (self.base_batch_size // batch_size)
                )
            else:
                grad_accum_steps = self.base_grad_accum_steps
            
            # Store for future reference
            self.stage_batch_params[self.current_stage] = {
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps
            }
            
            # Update current parameters
            self.curr_batch_size = batch_size
            self.curr_grad_accum_steps = grad_accum_steps
            
            return batch_size, grad_accum_steps
        
        # If not adaptive, use base values
        return self.base_batch_size, self.base_grad_accum_steps
    
    def update_batch_params(self, batch_size: int, grad_accum_steps: int):
        """Manually update batch parameters for the current stage."""
        self.stage_batch_params[self.current_stage] = {
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps
        }
        self.curr_batch_size = batch_size
        self.curr_grad_accum_steps = grad_accum_steps
    
    def get_filtered_dataset(self, dataset: Dataset) -> Dataset:
        """
        Filter dataset to only include sequences within the current stage's max length.
        
        Args:
            dataset: The full dataset to filter
            
        Returns:
            Filtered dataset with sequences up to the current max length
        """
        max_length = self.get_current_max_length()
        
        # Find sequences within the length limit
        valid_indices = []
        
        # Try to determine sequence lengths
        for i in range(len(dataset)):
            try:
                # Try different sequence length attributes/keys
                sample = dataset[i]
                length = None
                
                # Dictionary-like access
                if hasattr(sample, 'get'):
                    # Try common keys for sequence length
                    if 'length' in sample:
                        length = sample['length']
                    elif 'sequence_length' in sample:
                        length = sample['sequence_length']
                    elif 'sequence_int' in sample and isinstance(sample['sequence_int'], torch.Tensor):
                        length = len(sample['sequence_int'])
                    elif 'sequence' in sample:
                        if isinstance(sample['sequence'], str):
                            length = len(sample['sequence'])
                        elif isinstance(sample['sequence'], torch.Tensor):
                            length = len(sample['sequence'])
                
                # If found length, check against max
                if length is not None and length <= max_length:
                    valid_indices.append(i)
                
                # Limit how many samples we check for large datasets
                if i >= 1000 and len(valid_indices) >= 10:
                    logger.info(f"Dataset filtering limited to first 1000 examples (found {len(valid_indices)} within limit)")
                    # Continue checking but less frequently
                    if i % 100 != 0:
                        continue
                        
            except Exception as e:
                logger.warning(f"Error accessing sample {i}: {e}")
        
        if not valid_indices:
            logger.warning(f"No sequences found with length <= {max_length}, using full dataset")
            valid_indices = list(range(len(dataset)))
        
        # Create and return subset dataset
        filtered_dataset = Subset(dataset, valid_indices)
        logger.info(f"Stage {self.current_stage}: filtered dataset contains {len(filtered_dataset)} sequences (max length {max_length})")
        
        return filtered_dataset
    
    def update_stage(self, epoch: int, epoch_loss: float, 
                    num_sequences_at_next_stage: Optional[int] = None) -> bool:
        """
        Update curriculum stage based on epochs and loss improvement.
        
        Args:
            epoch: Current global epoch number
            epoch_loss: Validation loss for current epoch
            num_sequences_at_next_stage: Optional count of sequences available in next stage
            
        Returns:
            True if stage was updated, False otherwise
        """
        # Increment epochs at current stage
        self.epochs_at_current_stage += 1
        
        # Check if we're already at the final stage
        if self.current_stage >= len(self.sequence_stages) - 1:
            logger.info(f"Already at final curriculum stage ({self.current_stage})")
            return False
        
        # Check minimum epochs requirement
        minimum_epochs_met = self.epochs_at_current_stage >= self.epochs_per_stage
        
        # Check for plateau/improvement
        relative_improvement = 0.0
        if self.best_loss_at_stage != float('inf'):
            relative_improvement = (self.best_loss_at_stage - epoch_loss) / self.best_loss_at_stage
        
        if epoch_loss < self.best_loss_at_stage:
            self.best_loss_at_stage = epoch_loss
            
            # Reset plateau counter if improvement exceeds threshold
            if relative_improvement >= self.min_improvement:
                self.plateau_counter = 0
            else:
                # Small improvement but not enough
                self.plateau_counter += 1
        else:
            # No improvement
            self.plateau_counter += 1
        
        # Determine if we should advance to the next stage
        plateau_triggered = self.plateau_counter >= self.plateau_patience
        should_advance = minimum_epochs_met and (plateau_triggered or 
                                              self.epochs_at_current_stage >= 2*self.epochs_per_stage)
        
        # Additional check: ensure we have enough sequences at the next stage
        if should_advance and num_sequences_at_next_stage is not None:
            min_sequences = 10  # Minimum threshold for advancing
            if num_sequences_at_next_stage < min_sequences:
                logger.warning(f"Not advancing to stage {self.current_stage + 1} due to insufficient data "
                             f"({num_sequences_at_next_stage} sequences)")
                should_advance = False
        
        # Advance stage if conditions are met
        if should_advance:
            # Record stage transition
            from_stage = self.current_stage
            from_length = self.get_current_max_length()
            
            # Update stage
            self.current_stage += 1
            
            # Get new sequence length
            to_stage = self.current_stage
            to_length = self.get_current_max_length()
            
            # Record in history
            transition = {
                'epoch': epoch,
                'from_stage': from_stage,
                'to_stage': to_stage,
                'from_length': from_length,
                'to_length': to_length,
                'epochs_spent': self.epochs_at_current_stage,
                'best_loss': self.best_loss_at_stage,
                'plateau_reached': plateau_triggered
            }
            self.stage_history.append(transition)
            
            # Reset stage tracking variables
            self.epochs_at_current_stage = 0
            self.plateau_counter = 0
            self.best_loss_at_stage = float('inf')
            
            logger.info(f"Advanced to curriculum stage {self.current_stage}, max sequence length now {to_length}")
            return True
            
        # Log if we're not advancing but met some conditions
        if minimum_epochs_met and self.plateau_counter > 0:
            logger.info(f"Stage {self.current_stage}: plateau counter {self.plateau_counter}/{self.plateau_patience}, "
                      f"epoch {self.epochs_at_current_stage}/{self.epochs_per_stage}")
        
        return False
    
    def get_estimated_batch_size(self) -> int:
        """Estimate appropriate batch size for current stage based on sequence length."""
        if not self.batch_adaptive:
            return self.base_batch_size
        
        # Calculate based on sequence length
        max_len = self.get_current_max_length()
        base_len = self.sequence_stages[0]
        
        # Using quadratic scaling (attention is O(n²))
        scaling_factor = (base_len / max_len) ** 2
        
        # Apply scaling to base batch size
        batch_size = max(1, int(self.base_batch_size * scaling_factor))
        
        # Adjust to power of 2 for better GPU performance
        batch_size = 2 ** int(np.log2(batch_size))
        
        return batch_size
    
    def get_estimated_grad_accum(self, batch_size: int) -> int:
        """Calculate gradient accumulation steps to maintain effective batch size."""
        if batch_size >= self.base_batch_size:
            return self.base_grad_accum_steps
        
        # Increase grad accumulation to compensate for smaller batch
        factor = self.base_batch_size / batch_size
        return max(self.base_grad_accum_steps, int(self.base_grad_accum_steps * factor))
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Create state dictionary for checkpointing curriculum state."""
        return {
            'current_stage': self.current_stage,
            'epochs_at_current_stage': self.epochs_at_current_stage,
            'best_loss_at_stage': float(self.best_loss_at_stage),
            'plateau_counter': self.plateau_counter,
            'stage_history': self.stage_history,
            'stage_batch_params': self.stage_batch_params,
            'curr_batch_size': self.curr_batch_size,
            'curr_grad_accum_steps': self.curr_grad_accum_steps
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load curriculum state from state dictionary."""
        self.current_stage = state_dict.get('current_stage', 0)
        self.epochs_at_current_stage = state_dict.get('epochs_at_current_stage', 0)
        self.best_loss_at_stage = state_dict.get('best_loss_at_stage', float('inf'))
        self.plateau_counter = state_dict.get('plateau_counter', 0)
        self.stage_history = state_dict.get('stage_history', [])
        self.stage_batch_params = state_dict.get('stage_batch_params', {})
        self.curr_batch_size = state_dict.get('curr_batch_size', self.base_batch_size)
        self.curr_grad_accum_steps = state_dict.get('curr_grad_accum_steps', self.base_grad_accum_steps)
        
        logger.info(f"Loaded curriculum state: stage {self.current_stage}, "
                   f"epochs at stage {self.epochs_at_current_stage}")
        logger.info(f"Current max length: {self.get_current_max_length()}")


def create_length_limited_collate_fn(max_seq_length: int, base_collate_fn: Callable) -> Callable:
    """
    Create a collate function that limits sequence length in batches.
    
    Args:
        max_seq_length: Maximum allowed sequence length
        base_collate_fn: Original collate function to wrap
        
    Returns:
        Collate function with length limiting
    """
    def length_limited_collate(batch):
        # Get batch size
        batch_size = len(batch)
        
        # Apply sequence length limit to each sample if needed
        for sample in batch:
            if isinstance(sample, dict) and 'length' in sample and sample['length'] > max_seq_length:
                # Trim all sequence-related tensors
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        # Trim 1D tensors
                        if len(value.shape) == 1 and value.shape[0] > max_seq_length:
                            sample[key] = value[:max_seq_length]
                        # Trim 2D tensors
                        elif len(value.shape) == 2:
                            if key.endswith('_matrix') or 'pair' in key:
                                # This is likely a pairwise tensor (e.g. contact matrix)
                                sample[key] = value[:max_seq_length, :max_seq_length]
                            elif value.shape[0] > max_seq_length:
                                sample[key] = value[:max_seq_length, :]
                        # Trim 3D tensors
                        elif len(value.shape) == 3 and value.shape[0] > max_seq_length:
                            sample[key] = value[:max_seq_length, :, :]
                
                # Update the length
                sample['length'] = max_seq_length
        
        # Use the original collate function now that lengths are restricted
        return base_collate_fn(batch)
    
    return length_limited_collate


def analyze_dataset_lengths(dataset: Dataset) -> Dict[str, Any]:
    """
    Analyze sequence length distribution in a dataset.
    
    Args:
        dataset: Dataset to analyze
        
    Returns:
        Statistics about sequence lengths in the dataset
    """
    # Sample up to 1000 sequences from the dataset
    max_samples = min(1000, len(dataset))
    sequence_lengths = []
    
    for i in range(max_samples):
        try:
            # Try to get sequence length
            sample = dataset[i]
            length = None
            
            # Try different sequence length attributes/keys
            if hasattr(sample, 'get'):
                # Try common keys for sequence length
                if 'length' in sample:
                    length = sample['length']
                elif 'sequence_length' in sample:
                    length = sample['sequence_length']
                elif 'sequence_int' in sample and isinstance(sample['sequence_int'], torch.Tensor):
                    length = len(sample['sequence_int'])
                elif 'sequence' in sample:
                    if isinstance(sample['sequence'], str):
                        length = len(sample['sequence'])
                    elif isinstance(sample['sequence'], torch.Tensor):
                        length = len(sample['sequence'])
            
            if length is not None:
                sequence_lengths.append(length)
                
        except Exception as e:
            logger.warning(f"Error accessing sample {i}: {e}")
    
    if not sequence_lengths:
        return {"error": "Could not determine sequence lengths"}
    
    # Calculate statistics
    lengths = np.array(sequence_lengths)
    stats = {
        "count": len(sequence_lengths),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "sampled": max_samples,
        "total": len(dataset)
    }
    
    # Count sequences available at different length thresholds
    counts_by_threshold = {}
    sequences_available = {}
    
    length_thresholds = [50, 100, 150, 200, 250, 300, 400, 500]
    for threshold in length_thresholds:
        count = np.sum(lengths <= threshold)
        fraction = count / len(lengths)
        counts_by_threshold[threshold] = {
            "count": int(count),
            "fraction": float(fraction),
            "estimated_total": int(fraction * len(dataset))
        }
        sequences_available[threshold] = int(fraction * len(dataset))
    
    stats["counts_by_threshold"] = counts_by_threshold
    stats["sequences_available"] = sequences_available
    
    return stats


class CurriculumDataLoader(DataLoader):
    """
    DataLoader that integrates curriculum learning with dynamic batch sizing.
    """
    
    def __init__(self, dataset: Dataset, curriculum_manager: CurriculumManager, **kwargs):
        """
        Initialize curriculum-aware DataLoader.
        
        Args:
            dataset: The dataset to load from
            curriculum_manager: Curriculum manager for sequence length limits
            **kwargs: Additional arguments for DataLoader
        """
        self.curriculum_manager = curriculum_manager
        
        # Override batch_size with curriculum-specific value
        batch_size, _ = curriculum_manager.get_batch_params()
        if "batch_size" in kwargs and kwargs["batch_size"] != batch_size:
            logger.info(f"Overriding batch size {kwargs['batch_size']} with curriculum value {batch_size}")
        
        kwargs["batch_size"] = batch_size
        
        # Get max sequence length for this curriculum stage
        self.max_seq_length = curriculum_manager.get_current_max_length()
        
        # If a custom collate function is provided, wrap it with length limiting
        if "collate_fn" in kwargs and kwargs["collate_fn"] is not None:
            base_collate_fn = kwargs["collate_fn"]
            kwargs["collate_fn"] = create_length_limited_collate_fn(
                self.max_seq_length, base_collate_fn
            )
        
        # Get filtered dataset for current stage
        filtered_dataset = curriculum_manager.get_filtered_dataset(dataset)
        
        # Initialize DataLoader with the filtered dataset
        super().__init__(filtered_dataset, **kwargs)
        
        logger.info(f"Created curriculum DataLoader for stage {curriculum_manager.current_stage}")
        logger.info(f"Maximum sequence length: {self.max_seq_length}, batch size: {batch_size}")
        logger.info(f"Dataset size: {len(filtered_dataset)} sequences")
