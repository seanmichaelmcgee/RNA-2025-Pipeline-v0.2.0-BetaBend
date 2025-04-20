import pytest
import torch
import numpy as np
from src.utils.padding import pad_1d, pad_2d, pad_tensor


class TestPaddingUtils:
    """Tests for padding utility functions."""
    
    def test_pad_1d(self):
        """Test padding of 1D tensors."""
        # Test basic padding
        x = torch.tensor([1, 2, 3])
        padded = pad_1d(x, 5)
        assert padded.shape == (5,)
        assert torch.all(padded == torch.tensor([1, 2, 3, 0, 0]))
        
        # Test custom pad value
        padded = pad_1d(x, 5, pad_value=-1)
        assert torch.all(padded == torch.tensor([1, 2, 3, -1, -1]))
        
        # Test truncation
        padded = pad_1d(x, 2)
        assert padded.shape == (2,)
        assert torch.all(padded == torch.tensor([1, 2]))
        
        # Test empty tensor
        x = torch.tensor([])
        padded = pad_1d(x, 3)
        assert padded.shape == (3,)
        assert torch.all(padded == torch.tensor([0, 0, 0]))
        
        # Test device preservation
        if torch.cuda.is_available():
            x = torch.tensor([1, 2, 3], device='cuda')
            padded = pad_1d(x, 5)
            assert padded.device == x.device
        
        # Test dtype preservation
        x = torch.tensor([1, 2, 3], dtype=torch.float16)
        padded = pad_1d(x, 5)
        assert padded.dtype == torch.float16
    
    def test_pad_2d(self):
        """Test padding of 2D tensors."""
        # Test square tensor padding
        x = torch.tensor([[1, 2], [3, 4]])
        padded = pad_2d(x, 4)
        assert padded.shape == (4, 4)
        expected = torch.tensor([
            [1, 2, 0, 0],
            [3, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        assert torch.all(padded == expected)
        
        # Test feature tensor padding (L, D)
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
        padded = pad_2d(x, 4)
        assert padded.shape == (4, 3)  # Should preserve feature dim
        expected = torch.tensor([
            [1, 2, 3],
            [4, 5, 6],
            [0, 0, 0],
            [0, 0, 0]
        ])
        assert torch.all(padded == expected)
        
        # Test truncation for square tensor
        x = torch.ones((5, 5))
        padded = pad_2d(x, 3)
        assert padded.shape == (3, 3)
        assert torch.all(padded == torch.ones((3, 3)))
        
        # Test truncation for feature tensor
        x = torch.ones((5, 2))
        padded = pad_2d(x, 3)
        assert padded.shape == (3, 2)
        assert torch.all(padded == torch.ones((3, 2)))
        
        # Test empty tensor
        x = torch.tensor([]).reshape(0, 3)
        padded = pad_2d(x, 2)
        assert padded.shape == (2, 3)
        assert torch.all(padded == torch.zeros((2, 3)))
        
        # Test custom pad value
        x = torch.tensor([[1, 2], [3, 4]])
        padded = pad_2d(x, 3, pad_value=-1)
        expected = torch.tensor([
            [1, 2, -1],
            [3, 4, -1],
            [-1, -1, -1]
        ])
        assert torch.all(padded == expected)
    
    def test_pad_tensor(self):
        """Test padding of arbitrary tensors."""
        # Test 3D tensor padding
        x = torch.ones((2, 3, 4))
        target_shape = (3, 5, 6)
        padded = pad_tensor(x, target_shape)
        assert padded.shape == target_shape
        
        # Check original content is preserved
        assert torch.all(padded[:2, :3, :4] == 1)
        
        # Check padding is zeros
        assert torch.all(padded[2, :, :] == 0)
        assert torch.all(padded[:, 3:, :] == 0)
        assert torch.all(padded[:, :, 4:] == 0)
        
        # Test truncation
        x = torch.ones((5, 5, 5))
        target_shape = (3, 4, 2)
        padded = pad_tensor(x, target_shape)
        assert padded.shape == target_shape
        assert torch.all(padded == 1)  # Should be all ones from original
        
        # Test complex case with mixed truncation and padding
        x = torch.ones((2, 6, 3))
        target_shape = (4, 4, 5)
        padded = pad_tensor(x, target_shape)
        assert padded.shape == target_shape
        assert torch.all(padded[:2, :4, :3] == 1)
        assert torch.all(padded[2:, :, :] == 0)
        assert torch.all(padded[:, 4:, :] == 0)
        assert torch.all(padded[:, :, 3:] == 0)
        
        # Test with custom pad value
        x = torch.ones((2, 2))
        target_shape = (3, 3)
        padded = pad_tensor(x, target_shape, pad_value=-1)
        expected = torch.tensor([
            [1, 1, -1],
            [1, 1, -1],
            [-1, -1, -1]
        ])
        assert torch.all(padded == expected)