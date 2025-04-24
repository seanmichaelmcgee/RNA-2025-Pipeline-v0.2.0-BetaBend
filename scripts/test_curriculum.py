#!/usr/bin/env python3
"""
Test script for curriculum.py to verify the syntax error fix
"""

import sys
import os
from pathlib import Path

# Add project root to path for importing project modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import the fixed curriculum module
from pipeline.src.utils.curriculum import CurriculumManager

def test_curriculum_manager():
    """Test basic functionality of CurriculumManager."""
    print("Testing CurriculumManager initialization...")
    
    # Create a simple test dataset
    class TestDataset:
        def __init__(self, lengths):
            self.lengths = lengths
            
        def __len__(self):
            return len(self.lengths)
            
        def __getitem__(self, index):
            return {"length": self.lengths[index], "sequence": "A" * self.lengths[index]}
    
    # Create dataset with varied sequence lengths
    test_lengths = [20, 50, 100, 150, 200, 250, 300, 350, 400]
    dataset = TestDataset(test_lengths)
    
    # Initialize curriculum manager
    sequence_stages = [50, 100, 150, 200, 250, 300]
    curriculum = CurriculumManager(
        sequence_stages=sequence_stages,
        epochs_per_stage=2,
        plateau_patience=1,
        batch_adaptive=True
    )
    
    # Test get_current_max_length
    max_length = curriculum.get_current_max_length()
    print(f"Current max length: {max_length}")
    assert max_length == sequence_stages[0], f"Expected {sequence_stages[0]}, got {max_length}"
    
    # Test batch parameters
    batch_size, grad_accum_steps = curriculum.get_batch_params()
    print(f"Batch params: size={batch_size}, grad_accum={grad_accum_steps}")
    
    # Test update_stage
    for i in range(5):
        # Simulate training for a few epochs
        updated = curriculum.update_stage(i, 1.0 - i*0.1)
        print(f"Epoch {i}, updated: {updated}, stage: {curriculum.current_stage}, max_length: {curriculum.get_current_max_length()}")
    
    # Test get_filtered_dataset
    filtered_dataset = curriculum.get_filtered_dataset(dataset)
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    # Test batch size adaptation
    for stage in range(len(sequence_stages)):
        curriculum.current_stage = stage
        batch_size, grad_accum = curriculum.get_batch_params()
        max_len = curriculum.get_current_max_length()
        print(f"Stage {stage}, max_len: {max_len}, batch_size: {batch_size}, grad_accum: {grad_accum}")
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    success = test_curriculum_manager()
    sys.exit(0 if success else 1)