# PyTorch Patterns for RNA 3D Structure Prediction

This reference guide documents essential PyTorch patterns and best practices specifically tailored for implementing the RNA 3D folding architecture. It covers model structure, tensor operations, attention mechanisms, optimization strategies, and scaling considerations.

## 1. Model Architecture Patterns

### 1.1 Module Organization

Structure your model following PyTorch's compositional design pattern:

```python
class RNAFoldingModel(nn.Module):
    """Main RNA folding model that combines all components."""
    
    def __init__(self, config):
        super().__init__()
        
        # Extract config parameters
        residue_dim = config['residue_embed_dim']
        pair_dim = config['pair_embed_dim']
        num_blocks = config['num_transformer_blocks']
        
        # Embeddings
        self.sequence_embedding = SequenceEmbedding(config)
        self.positional_encoding = PositionalEncoding(residue_dim)
        
        # Projection layers
        self.residue_projection = nn.Linear(in_features=self.calculate_residue_input_dim(),
                                           out_features=residue_dim)
        self.pair_projection = nn.Linear(in_features=self.calculate_pair_input_dim(),
                                        out_features=pair_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(num_blocks)
        ])
        
        # IPA module (placeholder for V1)
        self.ipa_module = IPAModule(config)
        
        # Output heads
        self.confidence_head = self._build_confidence_head(residue_dim)
        self.angle_prediction_head = self._build_angle_head(residue_dim)
        
    def _build_confidence_head(self, input_dim):
        """Build the confidence prediction head."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def _build_angle_head(self, input_dim):
        """Build the angle prediction head."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 4)  # sin/cos of eta and theta
        )
    
    def forward(self, batch):
        """Forward pass through the full model."""
        # Implementation details...
```

### 1.2 Forward Method Implementation

Follow this structured pattern for complex forward methods:

```python
def forward(self, batch):
    """
    Forward pass through the RNA folding model.
    
    Args:
        batch: Dictionary containing:
            - sequence_int: (B, L) integer sequence representation
            - pairing_probs: (B, L, L) base pair probabilities
            - dihedral_features: (B, L, 4) dihedral angle sin/cos features
            - positional_entropy: (B, L) per-position entropy
            - coupling_matrix: (B, L, L) evolutionary coupling matrix
            - mask: (B, L) boolean mask for valid positions
            
    Returns:
        Dictionary containing:
            - pred_coords: (B, L, 3) predicted C1' coordinates
            - pred_confidence: (B, L) confidence scores per residue
            - pred_angles: (B, L, 4) predicted dihedral angles sin/cos
    """
    # 1. Extract inputs and validate shapes
    sequence = batch['sequence_int']
    pairing_probs = batch['pairing_probs']
    mask = batch['mask']
    device = sequence.device
    
    batch_size, seq_len = sequence.shape
    
    # 2. Shape assertions (can be disabled in production)
    self._validate_input_shapes(batch)
    
    # 3. Initial embeddings
    # 3.1 Sequence embedding
    seq_embedding = self.sequence_embedding(sequence)
    
    # 3.2 Positional encoding
    pos_encoding = self.positional_encoding(seq_len).to(device)
    pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
    
    # 3.3 Combine all residue features
    residue_features = torch.cat([
        seq_embedding,
        batch['dihedral_features'],
        batch['positional_entropy'].unsqueeze(-1),
        # Additional features...
    ], dim=-1)
    
    # 4. Initial projections
    residue_repr = self.residue_projection(residue_features)
    pair_repr = self.pair_projection(self._prepare_pair_features(batch))
    
    # 5. Apply transformer blocks
    for block in self.transformer_blocks:
        residue_repr, pair_repr = block(residue_repr, pair_repr, mask)
    
    # 6. Generate 3D coordinates
    coords = self.ipa_module(residue_repr, pair_repr, mask)
    
    # 7. Compute auxiliary outputs
    confidence = self.confidence_head(residue_repr).squeeze(-1)
    angles = self.angle_prediction_head(residue_repr)
    
    # 8. Return all outputs
    return {
        'pred_coords': coords,
        'pred_confidence': confidence,
        'pred_angles': angles
    }
```

### 1.3 Component-Specific Module Pattern

Each component should follow this pattern:

```python
class TransformerBlock(nn.Module):
    """
    Transformer block with residue and pair representation updates.
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract dimensions
        self.residue_dim = config['residue_embed_dim']
        self.pair_dim = config['pair_embed_dim']
        self.heads = config['num_attention_heads']
        self.dropout_rate = config['dropout']
        
        # Attention for residue update
        self.residue_attention_norm = nn.LayerNorm(self.residue_dim)
        self.residue_attention = nn.MultiheadAttention(
            embed_dim=self.residue_dim,
            num_heads=self.heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        # Feedforward for residue
        self.residue_ffn_norm = nn.LayerNorm(self.residue_dim)
        self.residue_ffn = nn.Sequential(
            nn.Linear(self.residue_dim, config.get('ffn_dim', self.residue_dim * 4)),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(config.get('ffn_dim', self.residue_dim * 4), self.residue_dim)
        )
        
        # Pair update components
        self.pair_update_norm = nn.LayerNorm(self.pair_dim)
        self.pair_update = self._build_pair_update(config)
    
    def _build_pair_update(self, config):
        """Build the simplified pair update network."""
        # Input: outer product of residue embeddings + current pair repr
        input_dim = 2 * self.residue_dim + self.pair_dim
        return nn.Sequential(
            nn.Linear(input_dim, self.pair_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.pair_dim, self.pair_dim)
        )
    
    def forward(self, residue_repr, pair_repr, mask=None):
        """
        Forward pass through the transformer block.
        
        Args:
            residue_repr: (B, L, D_res) residue representations
            pair_repr: (B, L, L, D_pair) pair representations
            mask: (B, L) padding mask, True for valid positions
            
        Returns:
            Updated residue and pair representations
        """
        # Convert mask to attention mask if provided
        attn_mask = None
        if mask is not None:
            # Create attention mask (B, L, L)
            attn_mask = torch.bmm(mask.float().unsqueeze(-1), 
                                  mask.float().unsqueeze(1))
            # Convert to boolean mask expected by PyTorch attention
            attn_mask = (1.0 - attn_mask) * -10000.0
        
        # 1. Residue update with self-attention
        # Pre-normalization architecture
        res_norm = self.residue_attention_norm(residue_repr)
        
        # Self-attention with optional mask
        attn_output, _ = self.residue_attention(
            query=res_norm, 
            key=res_norm, 
            value=res_norm,
            key_padding_mask=~mask if mask is not None else None,
            need_weights=False
        )
        
        # Residual connection
        residue_repr = residue_repr + attn_output
        
        # 2. Residue update with feedforward
        res_norm = self.residue_ffn_norm(residue_repr)
        residue_repr = residue_repr + self.residue_ffn(res_norm)
        
        # 3. Pair representation update
        pair_norm = self.pair_update_norm(pair_repr)
        
        # Create outer product of residue embeddings
        batch_size, seq_len, res_dim = residue_repr.shape
        
        # For each pair i,j: concatenate h_i, h_j, and pair_repr
        # Extract features for each position
        h_i = residue_repr.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (B, L, L, D_res)
        h_j = residue_repr.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (B, L, L, D_res)
        
        # Concatenate along feature dimension
        pair_input = torch.cat([h_i, h_j, pair_norm], dim=-1)  # (B, L, L, 2*D_res + D_pair)
        
        # Apply pair update
        pair_update = self.pair_update(pair_input)
        
        # Apply mask to update if provided
        if mask is not None:
            # Create 2D mask for pairs
            pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # (B, L, L)
            pair_mask = pair_mask.unsqueeze(-1)  # (B, L, L, 1)
            pair_update = pair_update * pair_mask
        
        # Residual connection
        pair_repr = pair_repr + pair_update
        
        return residue_repr, pair_repr
```

## 2. Tensor Operations and Manipulation

### 2.1 Shape Handling

Use explicit shape operations and document dimensions:

```python
def prepare_pair_features(self, features_dict):
    """
    Prepare pair features by concatenating and reshaping.
    
    Args:
        features_dict: Dictionary containing pair features:
            - pairing_probs: (B, L, L) base pair probabilities
            - coupling_matrix: (B, L, L) evolutionary coupling scores
            
    Returns:
        Tensor of shape (B, L, L, F) where F is the total feature dimension
    """
    # Extract batch size and sequence length
    pairing_probs = features_dict['pairing_probs']
    batch_size, seq_len, _ = pairing_probs.shape
    
    # Concatenate features along new dimension
    pair_features = torch.cat([
        pairing_probs.unsqueeze(-1),              # (B, L, L, 1)
        features_dict['coupling_matrix'].unsqueeze(-1),  # (B, L, L, 1)
        # Add more pair features as needed
    ], dim=-1)  # Result: (B, L, L, num_features)
    
    return pair_features
```

### 2.2 Masking Patterns

Implement proper masking for variable sequence lengths:

```python
def create_padding_mask(self, sequences, padding_idx=0):
    """
    Create boolean mask for valid positions (not padding).
    
    Args:
        sequences: (B, L) integer tensor of sequence indices
        padding_idx: Integer value representing padding
        
    Returns:
        Boolean mask of shape (B, L), True for valid positions
    """
    return sequences != padding_idx  # True for valid positions

def apply_mask_to_logits(self, logits, mask):
    """
    Apply mask to attention logits.
    
    Args:
        logits: (B, L, L) unnormalized attention scores
        mask: (B, L) boolean mask, True for valid positions
        
    Returns:
        Masked logits with large negative values in invalid positions
    """
    # Create 2D mask from 1D mask
    mask_2d = torch.bmm(mask.float().unsqueeze(-1), 
                        mask.float().unsqueeze(1))  # (B, L, L)
    
    # Apply mask: -10000 for padding tokens
    masked_logits = logits.masked_fill(~(mask_2d.bool()), -10000.0)
    return masked_logits
```

### 2.3 Efficient Operations

Use vectorized operations and avoid loops:

```python
# EFFICIENT - vectorized
def pairwise_distances(coords):
    """
    Compute pairwise distances between all residues efficiently.
    
    Args:
        coords: (B, L, 3) coordinates
        
    Returns:
        (B, L, L) pairwise distances
    """
    # Calculate squared differences along coordinate dimension
    diffs = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, L, L, 3)
    squared_diffs = diffs.pow(2)  # (B, L, L, 3)
    
    # Sum across coordinate dimension and take sqrt
    squared_distances = squared_diffs.sum(dim=-1)  # (B, L, L)
    distances = torch.sqrt(squared_distances + 1e-8)  # Add epsilon for numerical stability
    
    return distances

# INEFFICIENT - avoid this
def pairwise_distances_inefficient(coords):
    """Inefficient implementation - avoid!"""
    batch_size, seq_len, _ = coords.shape
    distances = torch.zeros(batch_size, seq_len, seq_len, device=coords.device)
    
    # Loop-based implementation - slow!
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(seq_len):
                distances[b, i, j] = torch.norm(coords[b, i] - coords[b, j])
    
    return distances
```

### 2.4 Device Management

Handle devices consistently:

```python
def transfer_batch_to_device(batch, device):
    """
    Transfer all tensors in batch to the target device.
    
    Args:
        batch: Dictionary of tensors or nested dictionaries
        device: torch.device to transfer to
        
    Returns:
        Batch with all tensors moved to device
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    
    if isinstance(batch, dict):
        return {
            key: transfer_batch_to_device(value, device)
            for key, value in batch.items()
        }
    
    if isinstance(batch, list):
        return [transfer_batch_to_device(item, device) for item in batch]
    
    # If not a tensor or container, return as is
    return batch
```

## 3. Attention Mechanisms and Transformer Patterns

### 3.1 Multi-Head Attention Usage

Use PyTorch's built-in attention with proper masking:

```python
def self_attention_layer(self, query, key, value, mask=None):
    """
    Apply multi-head self-attention with proper masking.
    
    Args:
        query, key, value: (B, L, D) tensors
        mask: (B, L) boolean mask, True for valid positions
        
    Returns:
        Attention output of shape (B, L, D)
    """
    # 1. Create attention mask if needed
    attn_mask = None
    if mask is not None:
        # Create a mask of shape (B, L)
        key_padding_mask = ~mask  # nn.MultiheadAttention expects False for valid positions
    else:
        key_padding_mask = None
    
    # 2. Apply attention
    attn_output, _ = self.attention(
        query=query,
        key=key,
        value=value,
        key_padding_mask=key_padding_mask,
        need_weights=False
    )
    
    return attn_output
```

### 3.2 Pair-Aware Attention Implementation

For the RNA folding transformer with pair bias (V2+):

```python
class PairBiasedAttention(nn.Module):
    """
    Attention mechanism with pair representation bias.
    Note: This is a more advanced implementation for V2+.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projection matrices
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.pair_bias_proj = nn.Linear(embed_dim, num_heads)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key, value, pair_repr, mask=None):
        """
        Apply attention with pair bias.
        
        Args:
            query, key, value: (B, L, D) tensors 
            pair_repr: (B, L, L, D_pair) pair representation
            mask: (B, L) boolean mask, True for valid positions
            
        Returns:
            Attention output with shape (B, L, D)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project query, key, value
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (B, H, L, D_head)
        k = k.transpose(1, 2)  # (B, H, L, D_head)
        v = v.transpose(1, 2)  # (B, H, L, D_head)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)
        
        # Add pair bias to attention scores
        pair_bias = self.pair_bias_proj(pair_repr)  # (B, L, L, H)
        pair_bias = pair_bias.permute(0, 3, 1, 2)  # (B, H, L, L)
        scores = scores + pair_bias
        
        # Apply mask if provided
        if mask is not None:
            # Create 2D mask for attention scores
            attn_mask = torch.bmm(mask.float().unsqueeze(-1), 
                                 mask.float().unsqueeze(1))  # (B, L, L)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, L, L)
            scores = scores.masked_fill(~(attn_mask.bool()), -10000.0)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights
        output = torch.matmul(attn_weights, v)  # (B, H, L, D_head)
        
        # Transpose and reshape
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(output)
        
        return output
```

### 3.3 Transformer Layer with Pre-Layer Normalization

Use the modern pre-LN transformer architecture:

```python
class PreLNTransformerLayer(nn.Module):
    """Transformer layer with pre-layer normalization."""
    
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        # Attention components
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # FFN components
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """Forward pass with pre-normalization."""
        # Pre-LN for attention
        norm_x = self.norm1(x)
        
        # Self-attention
        attn_output, _ = self.self_attn(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            key_padding_mask=~mask if mask is not None else None,
            need_weights=False
        )
        
        # Residual connection
        x = x + self.dropout(attn_output)
        
        # Pre-LN for FFN
        norm_x = self.norm2(x)
        
        # Feed-forward network
        ffn_output = self.ffn(norm_x)
        
        # Residual connection
        x = x + self.dropout(ffn_output)
        
        return x
```

## 4. IPA Module Implementation Patterns

For the Invariant Point Attention module (placeholder for V1, full implementation for V2+):

### 4.1 IPA Module Placeholder (V1)

```python
class IPAModulePlaceholder(nn.Module):
    """Placeholder for the Invariant Point Attention module (V1)."""
    
    def __init__(self, config):
        super().__init__()
        self.residue_dim = config['residue_embed_dim']
        
        # Simple linear projection from residue representation to 3D coordinates
        self.coord_projection = nn.Sequential(
            nn.Linear(self.residue_dim, self.residue_dim // 2),
            nn.ReLU(),
            nn.Linear(self.residue_dim // 2, 3)  # Output: x, y, z coordinates
        )
    
    def forward(self, residue_repr, pair_repr=None, mask=None):
        """
        Forward pass through IPA placeholder.
        
        Args:
            residue_repr: (B, L, D_res) residue representations
            pair_repr: (B, L, L, D_pair) pair representations (unused in placeholder)
            mask: (B, L) boolean mask (True for valid positions)
            
        Returns:
            (B, L, 3) predicted coordinates for each residue
        """
        # Simple linear projection to coordinates
        coords = self.coord_projection(residue_repr)  # (B, L, 3)
        
        # Apply mask if provided
        if mask is not None:
            coords = coords * mask.unsqueeze(-1).float()
        
        return coords
```

### 4.2 Full IPA Module Pattern (V2+)

The full IPA module with invariant point attention:

```python
class IPAModule(nn.Module):
    """
    Invariant Point Attention module for 3D coordinate prediction.
    Reference: AlphaFold 2 and IPA papers
    Note: This is for future V2+ implementation.
    """
    def __init__(self, config):
        super().__init__()
        self.residue_dim = config['residue_embed_dim']
        self.pair_dim = config['pair_embed_dim']
        self.ipa_dim = config.get('ipa_dim', 16)
        self.num_heads = config.get('ipa_heads', 4)
        self.num_iterations = config.get('num_ipa_iterations', 8)
        
        # Components to be implemented in V2+
        # ...
    
    def forward(self, residue_repr, pair_repr, mask=None):
        """
        Predict 3D coordinates using invariant point attention.
        
        Args:
            residue_repr: (B, L, D_res) residue representations
            pair_repr: (B, L, L, D_pair) pair representations
            mask: (B, L) boolean mask (True for valid positions)
            
        Returns:
            (B, L, 3) predicted coordinates
        """
        batch_size, seq_len, _ = residue_repr.shape
        device = residue_repr.device
        
        # Initialize coordinates and frames
        # For V1, initialize with a simple linear projection
        coords = self._initialize_coords(residue_repr, mask)
        
        # For future V2+ implementation
        # Run multiple iterations of IPA
        # for _ in range(self.num_iterations):
        #     # Update coordinates with IPA
        #     coords = self._ipa_iteration(coords, residue_repr, pair_repr, mask)
        
        return coords
    
    def _initialize_coords(self, residue_repr, mask=None):
        """Initialize coordinates from residue representations."""
        # Simple initialization for V1
        batch_size, seq_len, _ = residue_repr.shape
        
        # Project residue features to coordinates
        coords = self.coord_projection(residue_repr)  # (B, L, 3)
        
        # Apply mask if provided
        if mask is not None:
            coords = coords * mask.unsqueeze(-1).float()
        
        return coords
```

## 5. Loss Function Implementations

### 5.1 Coordinate Loss (FAPE Proxy)

```python
def compute_fape_loss_proxy(pred_coords, true_coords, mask=None, clamp_value=10.0):
    """
    Compute Frame-Aligned Point Error (FAPE) loss proxy.
    For V1, this is a simple clamped L2 loss after Kabsch alignment.
    
    Args:
        pred_coords: (B, L, 3) predicted coordinates
        true_coords: (B, L, 3) ground truth coordinates
        mask: (B, L) boolean mask (True for valid positions)
        clamp_value: Maximum distance error to consider
        
    Returns:
        Scalar loss value
    """
    batch_size, seq_len, _ = pred_coords.shape
    
    # Default mask: all positions are valid
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=pred_coords.device)
    
    # Initialize loss
    total_loss = 0.0
    
    # Process each sequence in the batch separately
    for b in range(batch_size):
        # Extract valid coordinates for this sequence
        valid_mask = mask[b]
        if not valid_mask.any():
            continue  # Skip if no valid positions
            
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        
        # Perform Kabsch alignment
        p_aligned = _kabsch_align(p_valid, t_valid)
        
        # Calculate clamped L2 distance
        dist = torch.norm(p_aligned - t_valid, dim=1)
        clamped_dist = torch.clamp(dist, max=clamp_value)
        
        # Average over valid positions
        seq_loss = clamped_dist.mean()
        total_loss += seq_loss
    
    # Average over batch
    loss = total_loss / batch_size
    return loss

def _kabsch_align(P, Q):
    """
    Align points P to points Q using Kabsch algorithm.
    
    Args:
        P: (N, 3) points to align
        Q: (N, 3) reference points
        
    Returns:
        P_aligned: (N, 3) aligned points
    """
    # Center the points
    p_mean = P.mean(dim=0, keepdim=True)
    q_mean = Q.mean(dim=0, keepdim=True)
    P_centered = P - p_mean
    Q_centered = Q - q_mean
    
    # Compute covariance matrix
    C = torch.matmul(P_centered.transpose(-2, -1), Q_centered)
    
    # Compute optimal rotation using SVD
    U, _, Vt = torch.linalg.svd(C)
    V = Vt.transpose(-2, -1)
    
    # Ensure proper rotation (no reflection)
    det = torch.det(torch.matmul(V, U.transpose(-2, -1)))
    if det < 0:
        V[:, 2] = -V[:, 2]
    
    # Compute rotation matrix
    R = torch.matmul(V, U.transpose(-2, -1))
    
    # Apply rotation and translation
    P_aligned = torch.matmul(P_centered, R) + q_mean
    
    return P_aligned
```

### 5.2 Confidence Loss

```python
def compute_confidence_loss(pred_confidence, pred_coords, true_coords, mask=None):
    """
    Compute confidence prediction loss.
    Uses predicted confidence scores to estimate model uncertainty.
    
    Args:
        pred_confidence: (B, L) predicted confidence scores (logits)
        pred_coords: (B, L, 3) predicted coordinates
        true_coords: (B, L, 3) ground truth coordinates
        mask: (B, L) boolean mask (True for valid positions)
        
    Returns:
        Scalar loss value
    """
    batch_size, seq_len = pred_confidence.shape
    
    # Default mask: all positions are valid
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=pred_confidence.device)
    
    # Calculate residue-wise error (as proxy for lDDT)
    with torch.no_grad():
        # Compute per-residue coordinate error
        coord_error = torch.norm(pred_coords - true_coords, dim=2)  # (B, L)
        
        # Convert to per-residue lDDT-like score in [0, 1]
        # Higher is better (1.0 = perfect prediction)
        lddt_proxy = torch.exp(-coord_error / 3.0)  # Simple exponential proxy
        
        # Ensure values are in [0, 1]
        lddt_proxy = torch.clamp(lddt_proxy, 0.0, 1.0)
        
        # Create confidence targets
        # 1 = high confidence (low error), 0 = low confidence (high error)
        conf_targets = lddt_proxy
    
    # Apply sigmoid to predicted logits
    pred_probs = torch.sigmoid(pred_confidence)
    
    # Calculate MSE loss
    squared_error = (pred_probs - conf_targets) ** 2
    
    # Apply mask and average
    masked_se = squared_error * mask.float()
    loss = masked_se.sum() / (mask.sum() + 1e-8)
    
    return loss
```

### 5.3 Angle Prediction Loss

```python
def compute_angle_loss(pred_angles, true_angles, mask=None):
    """
    Compute loss for dihedral angle predictions.
    
    Args:
        pred_angles: (B, L, 4) predicted sin/cos of angles [sin(η), cos(η), sin(θ), cos(θ)]
        true_angles: (B, L, 4) true sin/cos of angles
        mask: (B, L) boolean mask (True for valid positions)
        
    Returns:
        Scalar loss value
    """
    batch_size, seq_len, _ = pred_angles.shape
    
    # Default mask: all positions are valid
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=pred_angles.device)
    
    # Handle NaNs in true angles (typically at boundaries)
    angle_mask = mask.clone()
    if torch.isnan(true_angles).any():
        # Create mask for non-NaN angles
        nan_mask = ~torch.isnan(true_angles).any(dim=2)  # (B, L)
        angle_mask = angle_mask & nan_mask
    
    # Expand mask to match angle dimensions
    expanded_mask = angle_mask.unsqueeze(-1).expand_as(pred_angles)  # (B, L, 4)
    
    # Replace NaNs with zeros in true angles
    true_angles_clean = torch.nan_to_num(true_angles, nan=0.0)
    
    # Calculate mean squared error
    squared_error = (pred_angles - true_angles_clean) ** 2
    
    # Apply mask and calculate mean
    masked_se = squared_error * expanded_mask.float()
    total_elements = expanded_mask.sum() + 1e-8
    mse = masked_se.sum() / total_elements
    
    return mse
```

## 6. Optimization and Training Patterns

### 6.1 Gradient Norm Monitoring and Clipping

```python
def train_step(model, batch, optimizer, loss_fn, clip_grad_norm=1.0):
    """
    Perform a single training step with gradient monitoring and clipping.
    
    Args:
        model: PyTorch model
        batch: Input batch
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        clip_grad_norm: Maximum gradient norm
        
    Returns:
        Loss value and gradient statistics
    """
    # Forward pass
    outputs = model(batch)
    loss = loss_fn(outputs, batch)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip_grad_norm
    )
    
    # Step optimizer
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'grad_norm': grad_norm.item(),
        'grad_norm_clipped': grad_norm.item() > clip_grad_norm
    }
```

### 6.2 Learning Rate Scheduling

```python
def create_optimizer_and_scheduler(model, config):
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        
    Returns:
        Optimizer and scheduler
    """
    # Extract parameters
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    # Create optimizer
    optimizer_name = config['training'].get('optimizer', 'adamw')
    if optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Create scheduler
    scheduler_name = config['training'].get('lr_scheduler', 'cosine')
    if scheduler_name.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['max_epochs']
        )
    elif scheduler_name.lower() == 'step':
        step_size = config['training'].get('lr_step_size', 30)
        gamma = config['training'].get('lr_gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif scheduler_name.lower() == 'plateau':
        patience = config['training'].get('lr_patience', 5)
        factor = config['training'].get('lr_factor', 0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=True
        )
    elif scheduler_name.lower() == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return optimizer, scheduler
```

### 6.3 Mixed Precision Training

```python
def train_epoch_mixed_precision(model, dataloader, optimizer, loss_fn, scaler, device):
    """
    Train for one epoch using mixed precision.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        scaler: GradScaler for mixed precision
        device: torch.device
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = transfer_batch_to_device(batch, device)
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(batch)
            loss = loss_fn(outputs, batch)
        
        # Backward pass with scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping with scaling
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Step with scaling
        scaler.step(optimizer)
        scaler.update()
        
        # Log loss
        total_loss += loss.item()
    
    # Return average loss
    return total_loss / len(dataloader)
```

## 7. Scaling and Multi-GPU Patterns

### 7.1 Gradient Checkpointing for Memory Efficiency

```python
class MemoryEfficientTransformerBlock(nn.Module):
    """
    Transformer block with gradient checkpointing for memory efficiency.
    """
    def __init__(self, config):
        super().__init__()
        # Same initialization as regular TransformerBlock
        # ...
    
    def _attention_block(self, x, mask=None):
        """Attention block for checkpointing."""
        # Attention implementation
        # ...
        return x
    
    def _ffn_block(self, x):
        """Feed-forward block for checkpointing."""
        # FFN implementation
        # ...
        return x
    
    def forward(self, x, mask=None):
        """Forward pass with gradient checkpointing."""
        if self.training:
            # Use gradient checkpointing to save memory
            x = torch.utils.checkpoint.checkpoint(
                self._attention_block, x, mask
            )
            x = torch.utils.checkpoint.checkpoint(
                self._ffn_block, x
            )
        else:
            # Regular forward pass during inference
            x = self._attention_block(x, mask)
            x = self._ffn_block(x)
        
        return x
```

### 7.2 DataParallel and DistributedDataParallel Compatibility

```python
def create_model(config, device=None, distributed=False):
    """
    Create model with DataParallel or DistributedDataParallel wrapping if needed.
    
    Args:
        config: Configuration dictionary
        device: torch.device or None
        distributed: Whether to use DistributedDataParallel
        
    Returns:
        PyTorch model (possibly wrapped)
    """
    # Create base model
    model = RNAFoldingModel(config['model'])
    
    if device is not None:
        model = model.to(device)
    
    # Wrap model for multi-GPU training
    if distributed and torch.cuda.device_count() > 1:
        if torch.distributed.is_initialized():
            # Use DistributedDataParallel
            local_rank = config.get('local_rank', 0)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=config.get('find_unused_parameters', False)
            )
        else:
            # Use DataParallel
            model = torch.nn.DataParallel(model)
    
    return model
```

### 7.3 DistributedSampler Setup

```python
def create_data_loaders(dataset, config, distributed=False):
    """
    Create train and validation data loaders with optional distributed sampling.
    
    Args:
        dataset: PyTorch Dataset
        config: Configuration dictionary
        distributed: Whether to use distributed training
        
    Returns:
        DataLoader instance(s)
    """
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    
    # Create sampler
    sampler = None
    if distributed and torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            shuffle=True
        )
        shuffle = False  # Sampling is handled by the sampler
    else:
        shuffle = True
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config['data'].get('pin_memory', True),
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    return data_loader
```

## 8. Testing and Debugging Patterns

### 8.1 Shape Validation Helpers

```python
def validate_tensor_shapes(tensors_dict, expected_shapes):
    """
    Validate that tensors have expected shapes.
    
    Args:
        tensors_dict: Dictionary of tensors
        expected_shapes: Dictionary mapping tensor names to expected shapes
                        Use -1 for dynamic dimensions
                        
    Returns:
        True if all shapes match, raises ValueError otherwise
    """
    for name, expected_shape in expected_shapes.items():
        if name not in tensors_dict:
            raise ValueError(f"Missing tensor: {name}")
        
        tensor = tensors_dict[name]
        actual_shape = tensor.shape
        
        if len(actual_shape) != len(expected_shape):
            raise ValueError(
                f"Tensor {name} has {len(actual_shape)} dimensions, "
                f"expected {len(expected_shape)}"
            )
        
        for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
            if expected != -1 and actual != expected:
                raise ValueError(
                    f"Tensor {name} has shape {actual_shape}, "
                    f"but dimension {i} should be {expected}"
                )
    
    return True
```

### 8.2 Memory Profiling

```python
def profile_memory_usage(model, batch, device):
    """
    Profile memory usage during forward and backward pass.
    
    Args:
        model: PyTorch model
        batch: Input batch
        device: torch.device
        
    Returns:
        Dictionary of memory statistics
    """
    # Ensure GPU is clean
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Move batch to device
    batch = transfer_batch_to_device(batch, device)
    
    # Record initial memory
    initial_mem = torch.cuda.memory_allocated(device)
    
    # Forward pass
    outputs = model(batch)
    
    # Record forward memory
    forward_mem = torch.cuda.memory_allocated(device)
    
    # Compute loss and backward
    loss = 0
    for output in outputs.values():
        if isinstance(output, torch.Tensor) and output.requires_grad:
            loss = loss + output.sum()
    
    loss.backward()
    
    # Record backward memory
    backward_mem = torch.cuda.memory_allocated(device)
    peak_mem = torch.cuda.max_memory_allocated(device)
    
    # Clean up
    del outputs, loss
    torch.cuda.empty_cache()
    
    return {
        'initial_mem_mb': initial_mem / (1024 ** 2),
        'forward_mem_mb': (forward_mem - initial_mem) / (1024 ** 2),
        'backward_mem_mb': (backward_mem - forward_mem) / (1024 ** 2),
        'total_mem_mb': (backward_mem - initial_mem) / (1024 ** 2),
        'peak_mem_mb': peak_mem / (1024 ** 2)
    }
```

### 8.3 Deterministic Training

```python
def set_deterministic(seed=42, deterministic=True):
    """
    Set deterministic behavior for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to enable deterministic algorithms
        
    Returns:
        None
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Make CuDNN deterministic (slower)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Use CuDNN benchmark for performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
```

## 9. Kaggle Integration Patterns

### 9.1 Model Loading and Inference

```python
def load_model_for_inference(model_path, config, device):
    """
    Load model from checkpoint for Kaggle inference.
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration dictionary
        device: torch.device
        
    Returns:
        Loaded model in evaluation mode
    """
    # Create model
    model = RNAFoldingModel(config['model'])
    
    # Load state dict
    if device.type == 'cpu':
        state_dict = torch.load(model_path, map_location='cpu')
    else:
        state_dict = torch.load(model_path)
    
    # Handle DataParallel or DDP state dict
    if 'module.' in list(state_dict.keys())[0]:
        # Remove 'module.' prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_state_dict[key.replace('module.', '')] = value
        state_dict = new_state_dict
    
    # Load state dict and move to device
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    return model
```

### 9.2 Kaggle Submission CSV Generation

```python
def generate_submission_file(predictions, target_ids, output_path):
    """
    Generate submission file in Kaggle format.
    
    Args:
        predictions: Dictionary mapping target_ids to coordinate predictions
                    Each prediction has shape (num_models, seq_len, 3)
        target_ids: List of target IDs
        output_path: Path to save submission CSV
        
    Returns:
        None (writes file to output_path)
    """
    # Collect all rows for submission
    rows = []
    
    for target_id in target_ids:
        # Get coordinates for all 5 models
        coords = predictions[target_id]  # (5, L, 3)
        seq_len = coords.shape[1]
        
        # For each residue
        for i in range(seq_len):
            row = {
                'ID': f"{target_id}_{i+1}",  # 1-indexed residue IDs
                'resname': get_residue_name(target_id, i),
                'resid': i + 1  # 1-indexed
            }
            
            # Add coordinates for all 5 models
            for model_idx in range(5):
                x, y, z = coords[model_idx, i]
                row[f'x_{model_idx+1}'] = x.item()
                row[f'y_{model_idx+1}'] = y.item()
                row[f'z_{model_idx+1}'] = z.item()
            
            rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"Submission file saved to {output_path}")
```

## 10. Conclusion and Best Practices Summary

### Key Takeaways

1. **Model Organization**: Use compositional design with `nn.Module` inheritance and clear component separation
2. **Tensor Operations**: Document shapes, use vectorized operations, avoid loops
3. **Masking**: Handle variable sequence lengths consistently across components
4. **Attention**: Use `nn.MultiheadAttention` for standard cases, implement custom attention for pair bias
5. **Memory Management**: Monitor usage, employ checkpointing for large models
6. **Scaling**: Design for multi-GPU from the start with proper device management
7. **Testing**: Validate shapes, use deterministic mode for debugging, profile memory

### Critical Patterns for RNA 3D Folding

1. Proper handling of 1D (residue) and 2D (pair) representations
2. Consistent masking throughout the pipeline for variable sequences
3. Efficient fusion of multiple feature types
4. Strategic memory management for the transformer backbone
5. Appropriate shape transformations for coordinate prediction

These patterns provide a foundation for implementing both the V1 model with simplified components and laying groundwork for the full V2+ architecture with advanced features like pair-bias attention and the full IPA module.
