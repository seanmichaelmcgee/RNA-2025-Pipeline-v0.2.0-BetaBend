# Add this code to the notebook to fix the positional encoding for long sequences

# Enhanced positional encoding that can handle longer sequences
class EnhancedPositionalEncoding(nn.Module):
    """
    Enhanced version of PositionalEncoding that handles sequences longer than max_len.
    """
    
    def __init__(self, config):
        """Initialize positional encoding with extendable length."""
        super().__init__()

        # Extract parameters from config
        self.embed_dim = config.get("residue_embed_dim", 128)
        self.max_len = config.get("max_len", 500)

        # Create constant positional encoding matrix
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        self.div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim)
        )

        pe = torch.zeros(self.max_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)

        # Register buffer (not a parameter, but part of state)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, embed_dim)
    
    def extend_pe(self, new_max_len):
        """Dynamically extend the positional encoding to handle longer sequences."""
        # Create new positions
        old_max_len = self.pe.size(1)
        if new_max_len <= old_max_len:
            return  # No need to extend
            
        print(f"Extending positional encoding from {old_max_len} to {new_max_len}")
        
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

def patch_model_positional_encoding(model):
    """Patch a model's positional encoding to handle long sequences."""
    if hasattr(model, 'embedding_module') and hasattr(model.embedding_module, 'positional_encoding'):
        # Get original module
        orig_pe = model.embedding_module.positional_encoding
        
        # Create config for new module
        config = {
            'residue_embed_dim': orig_pe.embed_dim,
            'max_len': orig_pe.max_len
        }
        
        # Create enhanced module
        enhanced_pe = EnhancedPositionalEncoding(config)
        
        # Copy the existing buffer
        enhanced_pe.pe = orig_pe.pe.clone()
        
        # Replace in the model
        model.embedding_module.positional_encoding = enhanced_pe
        
        print(f"Patched model with enhanced positional encoding (max_len: {enhanced_pe.max_len})")
        return True
    else:
        print("Warning: Could not find positional encoding in model structure")
        return False

# Apply this patch to each model
for model_name, model_info in models.items():
    if 'model' in model_info:
        model = model_info['model']
        patch_model_positional_encoding(model)
        
print("All models patched successfully with enhanced positional encoding")