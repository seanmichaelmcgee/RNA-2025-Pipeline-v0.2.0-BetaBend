import pytest
import torch
import math
import numpy as np
from src.models.embeddings import (
    SequenceEmbedding,
    PositionalEncoding,
    RelativePositionalEncoding,
    EmbeddingModule
)


class TestSequenceEmbedding:
    """Tests for SequenceEmbedding module."""
    
    def test_initialization(self):
        """Test that the module initializes correctly."""
        # Default parameters
        config = {
            'num_embeddings': 5,
            'seq_embed_dim': 32
        }
        embedding = SequenceEmbedding(config)
        
        # Check attributes
        assert embedding.num_embeddings == 5
        assert embedding.embedding_dim == 32
        assert embedding.embedding.weight.shape == (5, 32)
    
    def test_forward(self):
        """Test the forward pass."""
        config = {
            'num_embeddings': 5,
            'seq_embed_dim': 32
        }
        embedding = SequenceEmbedding(config)
        
        # Create a batch of sequences
        sequences = torch.tensor([
            [0, 1, 2, 3, 4],  # A, C, G, U, N
            [4, 3, 2, 1, 0]   # N, U, G, C, A
        ])
        
        # Forward pass
        output = embedding(sequences)
        
        # Check shape
        assert output.shape == (2, 5, 32)
        
        # Check that padding (index 4) produces zeros
        # This assumes padding_idx=4 in config
        config['padding_idx'] = 4
        embedding_with_padding = SequenceEmbedding(config)
        output_with_padding = embedding_with_padding(sequences)
        
        # First sequence, last position should be all zeros
        assert torch.all(output_with_padding[0, 4] == 0)
        # Second sequence, first position should be all zeros
        assert torch.all(output_with_padding[1, 0] == 0)
    
    def test_device_compatibility(self):
        """Test that the module works on different devices."""
        config = {
            'num_embeddings': 5,
            'seq_embed_dim': 32
        }
        embedding = SequenceEmbedding(config)
        
        # Create a batch of sequences
        sequences = torch.tensor([
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0]
        ])
        
        # Forward pass on CPU
        output_cpu = embedding(sequences)
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            embedding = embedding.to('cuda')
            sequences = sequences.to('cuda')
            
            # Forward pass on GPU
            output_gpu = embedding(sequences)
            
            # Check that output is on the correct device
            assert output_gpu.device.type == 'cuda'
            
            # Check that output has the same shape
            assert output_gpu.shape == output_cpu.shape


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""
    
    def test_initialization(self):
        """Test that the module initializes correctly."""
        config = {
            'residue_embed_dim': 128,
            'max_len': 500
        }
        pos_encoding = PositionalEncoding(config)
        
        # Check attributes
        assert pos_encoding.embed_dim == 128
        assert pos_encoding.max_len == 500
        assert pos_encoding.pe.shape == (1, 500, 128)
    
    def test_forward(self):
        """Test the forward pass."""
        config = {
            'residue_embed_dim': 128,
            'max_len': 500
        }
        pos_encoding = PositionalEncoding(config)
        
        # Request encodings for a specific sequence length
        seq_len = 10
        output = pos_encoding(seq_len)
        
        # Check shape
        assert output.shape == (1, 10, 128)
        
        # Check pattern properties
        # 1. Different positions should have different encodings
        for i in range(seq_len - 1):
            assert not torch.allclose(output[0, i], output[0, i+1])
        
        # 2. The encoding should follow a sinusoidal pattern
        # Positions that are far apart should have more different encodings
        diffs = []
        for i in range(seq_len - 1):
            diff = torch.norm(output[0, i] - output[0, i+1]).item()
            diffs.append(diff)
        
        # Check device compatibility
        if torch.cuda.is_available():
            pos_encoding = pos_encoding.to('cuda')
            output = pos_encoding(seq_len)
            assert output.device.type == 'cuda'


class TestRelativePositionalEncoding:
    """Tests for RelativePositionalEncoding module."""
    
    def test_initialization(self):
        """Test that the module initializes correctly."""
        config = {
            'max_relative_position': 32,
            'rel_pos_dim': 32
        }
        rel_pos_encoding = RelativePositionalEncoding(config)
        
        # Check attributes
        assert rel_pos_encoding.max_relative_position == 32
        assert rel_pos_encoding.rel_pos_dim == 32
        assert rel_pos_encoding.embeddings.weight.shape == (65, 32)  # 2*32 + 1
    
    def test_forward(self):
        """Test the forward pass."""
        config = {
            'max_relative_position': 32,
            'rel_pos_dim': 32
        }
        rel_pos_encoding = RelativePositionalEncoding(config)
        
        # Request encodings for a specific sequence length
        seq_len = 10
        output = rel_pos_encoding(seq_len)
        
        # Check shape
        assert output.shape == (10, 10, 32)
        
        # Check symmetry properties
        # The relative position (i,j) should be the negative of (j,i)
        # BUT the actual encodings need not be symmetrical for pairs with the same absolute distance
        # because the actual encoding pattern is sinusoidal, not identical
        
        # Test that diagonal elements are identical (distance=0)
        for i in range(seq_len):
            assert torch.allclose(output[i, i], output[0, 0])
        
        # Test device compatibility
        if torch.cuda.is_available():
            rel_pos_encoding = rel_pos_encoding.to('cuda')
            output = rel_pos_encoding(seq_len)
            assert output.device.type == 'cuda'


class TestEmbeddingModule:
    """Tests for the complete EmbeddingModule."""
    
    def test_initialization(self):
        """Test that the module initializes correctly."""
        config = {
            'seq_embed_dim': 32,
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'max_relative_position': 32,
            'rel_pos_dim': 32,
            'use_conservation': False  # Set to False for consistent inputs
        }
        embedding_module = EmbeddingModule(config)
        
        # Check that all sub-modules are initialized
        assert isinstance(embedding_module.sequence_embedding, SequenceEmbedding)
        assert isinstance(embedding_module.positional_encoding, PositionalEncoding)
        assert isinstance(embedding_module.relative_pos_encoding, RelativePositionalEncoding)
        
        # Check projection layers
        # With use_conservation=False, residue_in_dim should be seq_embed + dihedral + pos_entropy + accessibility
        expected_residue_in_dim = config['seq_embed_dim'] + 4 + 1 + 1  # 38
        assert embedding_module.residue_in_dim == expected_residue_in_dim
        assert embedding_module.residue_projection.in_features == expected_residue_in_dim
        assert embedding_module.residue_projection.out_features == config['residue_embed_dim']
        
        # Pair input dim is pairing_probs(1) + coupling_matrix(1) + rel_pos_dim
        expected_pair_in_dim = 1 + 1 + config['rel_pos_dim']  # 34
        # The pair projection layer should use this input dimension
        assert embedding_module.pair_projection.in_features == expected_pair_in_dim
        assert embedding_module.pair_projection.out_features == config['pair_embed_dim']
    
    def test_forward(self):
        """Test the forward pass with a mock batch."""
        config = {
            'seq_embed_dim': 32,
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'max_relative_position': 32,
            'rel_pos_dim': 32,
            'num_embeddings': 5,
            'use_conservation': False  # Set to false for this test
        }
        embedding_module = EmbeddingModule(config)
        
        # Create a mock batch
        batch_size = 2
        seq_len = 10
        
        batch = {
            'sequence_int': torch.randint(0, 5, (batch_size, seq_len)),
            'dihedral_features': torch.rand(batch_size, seq_len, 4),
            'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
            'positional_entropy': torch.rand(batch_size, seq_len),
            'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
            'accessibility': torch.rand(batch_size, seq_len),
            'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
        }
        
        # Set some positions as padding
        batch['mask'][0, -2:] = False
        
        # Forward pass
        residue_repr, pair_repr, mask = embedding_module(batch)
        
        # Check shapes
        assert residue_repr.shape == (batch_size, seq_len, config['residue_embed_dim'])
        assert pair_repr.shape == (batch_size, seq_len, seq_len, config['pair_embed_dim'])
        assert mask.shape == (batch_size, seq_len)
        
        # Check mask handling
        # Masked positions in residue representation should be zero
        assert torch.all(residue_repr[0, -2:] == 0)
        
        # Masked positions in pair representation should be zero
        # Any pair involving a masked position should be zero
        assert torch.all(pair_repr[0, -2:, :] == 0)
        assert torch.all(pair_repr[0, :, -2:] == 0)
        
        # Test device compatibility
        if torch.cuda.is_available():
            embedding_module = embedding_module.to('cuda')
            batch = {k: v.to('cuda') for k, v in batch.items()}
            
            residue_repr, pair_repr, mask = embedding_module(batch)
            
            assert residue_repr.device.type == 'cuda'
            assert pair_repr.device.type == 'cuda'
            assert mask.device.type == 'cuda'
    
    def test_with_conservation(self):
        """Test forward pass with conservation feature."""
        config = {
            'seq_embed_dim': 32,
            'residue_embed_dim': 128,
            'pair_embed_dim': 64,
            'max_relative_position': 32,
            'rel_pos_dim': 32,
            'num_embeddings': 5,
            'use_conservation': True
        }
        embedding_module = EmbeddingModule(config)
        
        # Check the input dimension includes conservation
        expected_residue_in_dim = config['seq_embed_dim'] + 4 + 1 + 1 + 1  # +1 for conservation
        assert embedding_module.residue_in_dim == expected_residue_in_dim
        assert embedding_module.residue_projection.in_features == expected_residue_in_dim
        
        # Create a mock batch
        batch_size = 2
        seq_len = 10
        
        batch = {
            'sequence_int': torch.randint(0, 5, (batch_size, seq_len)),
            'dihedral_features': torch.rand(batch_size, seq_len, 4),
            'pairing_probs': torch.rand(batch_size, seq_len, seq_len),
            'positional_entropy': torch.rand(batch_size, seq_len),
            'coupling_matrix': torch.rand(batch_size, seq_len, seq_len),
            'accessibility': torch.rand(batch_size, seq_len),
            'conservation': torch.rand(batch_size, seq_len),
            'mask': torch.ones(batch_size, seq_len, dtype=torch.bool)
        }
        
        # Forward pass
        residue_repr, pair_repr, mask = embedding_module(batch)
        
        # Check shapes
        assert residue_repr.shape == (batch_size, seq_len, config['residue_embed_dim'])
        assert pair_repr.shape == (batch_size, seq_len, seq_len, config['pair_embed_dim'])
        
        # Test without conservation
        config['use_conservation'] = False
        embedding_module_no_cons = EmbeddingModule(config)
        
        # Expected input dim without conservation
        expected_residue_in_dim_no_cons = config['seq_embed_dim'] + 4 + 1 + 1  # No conservation
        assert embedding_module_no_cons.residue_in_dim == expected_residue_in_dim_no_cons
        
        # Remove conservation from batch
        batch_no_cons = batch.copy()
        del batch_no_cons['conservation']
        
        # Forward pass should still work
        residue_repr, pair_repr, mask = embedding_module_no_cons(batch_no_cons)
        
        # Check shapes
        assert residue_repr.shape == (batch_size, seq_len, config['residue_embed_dim'])
        assert pair_repr.shape == (batch_size, seq_len, seq_len, config['pair_embed_dim'])