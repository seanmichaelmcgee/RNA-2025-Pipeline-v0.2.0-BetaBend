"""
RNA 3D Structure Prediction Model (V1)

This module implements the main model architecture for RNA 3D structure prediction,
integrating embeddings, transformer blocks, and the IPA module for coordinate prediction.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import EmbeddingModule
from .transformer_block import TransformerBlock
from .ipa_module import IPAModule


class RNAFoldingModel(nn.Module):
    """
    End-to-end RNA 3D folding model that combines embeddings, transformer blocks,
    and a coordinate prediction module.

    The model takes RNA sequence and feature data and predicts 3D coordinates,
    confidence scores, and torsion angles.
    """

    def __init__(self, config: Dict):
        """
        Initialize the RNA folding model.

        Args:
            config: Dictionary containing model parameters
                - num_blocks: Number of transformer blocks
                - residue_embed_dim: Dimension of residue embeddings
                - pair_embed_dim: Dimension of pair embeddings
                - num_attention_heads: Number of attention heads in transformer
                - dropout: Dropout probability
                - confidence_output_dim: Dimension of confidence output (usually 1)
                - angles_output_dim: Dimension of angle output (usually 4)
                - ffn_dim: Hidden dimension for feed-forward networks
                - seq_embed_dim: Dimension of sequence embeddings
                - max_relative_position: Maximum relative distance to consider for positional encoding
        """
        super().__init__()

        # Extract parameters from config
        self.num_blocks = config.get("num_blocks", 4)
        self.residue_dim = config.get("residue_embed_dim", 128)
        self.pair_dim = config.get("pair_embed_dim", 64)
        self.confidence_output_dim = config.get("confidence_output_dim", 1)
        self.angles_output_dim = config.get("angles_output_dim", 4)

        # Initialize the embedding module
        self.embedding_module = EmbeddingModule(config)

        # Initialize transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(self.num_blocks)]
        )

        # Initialize IPA module for coordinate prediction
        self.ipa_module = IPAModule(config)

        # Initialize confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(self.residue_dim, self.residue_dim // 2),
            nn.ReLU(),
            nn.Linear(self.residue_dim // 2, self.confidence_output_dim),
        )

        # Initialize angle prediction head
        self.angle_head = nn.Sequential(
            nn.Linear(self.residue_dim, self.residue_dim // 2),
            nn.ReLU(),
            nn.Linear(self.residue_dim // 2, self.angles_output_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RNA folding model.

        Args:
            batch: Dictionary of input tensors from the data loader
                - sequence_int: Integer-encoded RNA sequence (batch_size, seq_len)
                - dihedral_features: Dihedral angle features (batch_size, seq_len, 4)
                - pairing_probs: Pairing probabilities (batch_size, seq_len, seq_len)
                - positional_entropy: Positional entropy (batch_size, seq_len)
                - accessibility: Solvent accessibility (batch_size, seq_len)
                - coupling_matrix: Evolutionary coupling (batch_size, seq_len, seq_len)
                - mask: Boolean mask (batch_size, seq_len)

        Returns:
            Dictionary of output tensors:
                - pred_coords: Predicted coordinates (batch_size, seq_len, 3)
                - pred_confidence: Predicted confidence scores (batch_size, seq_len)
                - pred_angles: Predicted torsion angles (batch_size, seq_len, 4)
        """
        # Validate required inputs
        required_keys = [
            "sequence_int",
            "dihedral_features",
            "pairing_probs",
            "positional_entropy",
            "accessibility",
            "coupling_matrix",
            "mask",
        ]

        for key in required_keys:
            if key not in batch:
                raise ValueError(f"Input batch missing required key: {key}")

        # Extract mask for convenience
        mask = batch["mask"]  # (batch_size, seq_len)

        # Create initial representations using embedding module
        residue_repr, pair_repr, mask = self.embedding_module(batch)
        # residue_repr: (batch_size, seq_len, residue_dim)
        # pair_repr: (batch_size, seq_len, seq_len, pair_dim)

        # Process through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            # Update residue and pair representations
            residue_repr, pair_repr = block(residue_repr, pair_repr, mask)

        # Generate 3D coordinates using IPA module
        pred_coords = self.ipa_module(
            residue_repr, pair_repr, mask
        )  # (batch_size, seq_len, 3)

        # Predict per-residue confidence
        confidence_logits = self.confidence_head(
            residue_repr
        )  # (batch_size, seq_len, 1)
        pred_confidence = confidence_logits.squeeze(-1)  # (batch_size, seq_len)

        # Predict angles
        pred_angles = self.angle_head(residue_repr)  # (batch_size, seq_len, 4)

        # Apply mask to outputs
        if mask is not None:
            mask_float = mask.float().unsqueeze(-1)
            pred_coords = pred_coords * mask_float
            pred_confidence = pred_confidence * mask.float()
            pred_angles = pred_angles * mask_float

        # Return all predictions
        outputs = {
            "pred_coords": pred_coords,
            "pred_confidence": pred_confidence,
            "pred_angles": pred_angles,
        }

        return outputs
