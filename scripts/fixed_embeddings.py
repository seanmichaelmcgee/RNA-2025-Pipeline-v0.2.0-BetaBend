"""
Fix for the embeddings module to handle sequences longer than max_len.
"""

import os
import sys
import torch
import math
import logging
from pathlib import Path

# Add project root to path
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.models.embeddings import PositionalEncoding

class EnhancedPositionalEncoding(PositionalEncoding):
    """
    Enhanced version of PositionalEncoding that handles sequences longer than max_len.
    """
    
    def __init__(self, config):
        """Initialize with parent constructor but add dynamic extension capability."""
        super().__init__(config)
        # Keep track of the original formula parameters
        self.div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim)
        )
    
    def extend_pe(self, new_max_len):
        """Dynamically extend the positional encoding to handle longer sequences."""
        # Create new positions
        old_max_len = self.pe.size(1)
        if new_max_len <= old_max_len:
            return  # No need to extend
            
        logging.info(f"Extending positional encoding from {old_max_len} to {new_max_len}")
        
        # Generate positions for the new entries
        position = torch.arange(old_max_len, new_max_len).unsqueeze(1).float()
        
        # Create new encodings
        pe_extension = torch.zeros(new_max_len - old_max_len, self.embed_dim, 
                                  device=self.pe.device)
        pe_extension[:, 0::2] = torch.sin(position * self.div_term)
        pe_extension[:, 1::2] = torch.cos(position * self.div_term)
        
        # Concatenate with existing buffer
        new_pe = torch.cat([self.pe.squeeze(0), pe_extension], dim=0).unsqueeze(0)
        
        # Replace the buffer
        self.pe = new_pe
        self.max_len = new_max_len
    
    def forward(self, seq_len):
        """Get positional encodings with automatic extension if needed."""
        if seq_len > self.max_len:
            # If sequence is longer than our current max, extend the encoding
            new_max_len = max(seq_len, int(self.max_len * 1.5))  # Grow by 50% to reduce frequent extensions
            self.extend_pe(new_max_len)
            
        return self.pe[:, :seq_len]

def patch_embeddings_module():
    """Patch the embeddings module to use our enhanced positional encoding."""
    from src.models import embeddings
    
    # Save original class for reference
    original_pos_encoding = embeddings.PositionalEncoding
    
    # Replace with our enhanced version
    embeddings.PositionalEncoding = EnhancedPositionalEncoding
    
    # Also patch any existing instances in loaded models
    def patch_existing_model(model):
        """Recursively patch positional encoding in an existing model."""
        for name, module in model.named_children():
            if isinstance(module, original_pos_encoding):
                # Get the config from the existing module
                config = {
                    'residue_embed_dim': module.embed_dim,
                    'max_len': module.max_len
                }
                # Create enhanced version with same parameters
                enhanced_module = EnhancedPositionalEncoding(config)
                # Copy the buffer
                enhanced_module.pe = module.pe.clone()
                # Replace in the model
                setattr(model, name, enhanced_module)
            else:
                # Recursively check children
                patch_existing_model(module)
    
    # Return a function that can be used to patch a model
    return patch_existing_model

if __name__ == "__main__":
    # Create a test config
    config = {
        'residue_embed_dim': 128,
        'max_len': 500
    }
    
    # Create original and enhanced encodings
    original = PositionalEncoding(config)
    enhanced = EnhancedPositionalEncoding(config)
    
    # Test with normal length
    seq_len = 300
    pos1 = original(seq_len)
    pos2 = enhanced(seq_len)
    print(f"Normal case (seq_len={seq_len}):")
    print(f"Original shape: {pos1.shape}")
    print(f"Enhanced shape: {pos2.shape}")
    print(f"Equal: {torch.allclose(pos1, pos2)}")
    
    # Test with longer sequence
    try:
        seq_len = 600
        pos1 = original(seq_len)
        print(f"Original with seq_len={seq_len}: {pos1.shape}")
    except Exception as e:
        print(f"Original failed with seq_len={seq_len}: {str(e)}")
    
    # Enhanced should handle it
    pos2 = enhanced(seq_len)
    print(f"Enhanced with seq_len={seq_len}: {pos2.shape}")
    
    # Test very long sequence
    seq_len = 1000
    pos2 = enhanced(seq_len)
    print(f"Enhanced with seq_len={seq_len}: {pos2.shape}")