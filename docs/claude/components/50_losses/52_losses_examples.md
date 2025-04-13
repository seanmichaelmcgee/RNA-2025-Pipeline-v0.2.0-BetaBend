Okay, I can help with that. I'll go through the document, identify the core functions and examples, remove redundant definitions and introductory/meta text, consolidate imports, structure it logically, and format it as a Markdown file.

I will keep functions that represent different *versions* (like the simplified FAPE proxy vs. the full FAPE, or the confidence loss with proxy targets vs. lDDT targets) as they seem intentionally distinct. I will also keep the more robust `stable_kabsch_align` alongside the simpler `kabsch_align` as they are presented in different contexts (basic use vs. stability improvement).

```markdown
# RNA 3D Structure Prediction Loss Functions and Implementation Guide

This document provides Python code examples for loss functions used in RNA 3D structure prediction, along with strategies for implementation, optimization, troubleshooting, and best practices.

## Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import math
import unittest
from typing import Optional, Dict, Tuple, List, Union, Any

# Optional imports for visualization/analysis (handle ImportError)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import PCA
except ImportError:
    plt = None
    Axes3D = None
    PCA = None
    logging.warning("matplotlib or sklearn not found. Visualization functions may not work.")

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:
    pearsonr = None
    spearmanr = None
    logging.warning("scipy not found. Correlation calculations may not work.")

try:
    import pandas as pd
except ImportError:
    pd = None
    logging.warning("pandas not found. CSV formatting/loading may not work.")

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    logging.warning("TensorBoard not available. Logging to TensorBoard will be disabled.")

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
```

## Section 1: Core Loss Functions

### 1.1 Basic Losses (V1 / Proxies)

These are implementations often used as starting points or simpler proxies.

```python
def kabsch_align(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Align points P to points Q using Kabsch algorithm (Simple Version).

    Args:
        P: Moving points, shape (N, 3)
        Q: Fixed points, shape (N, 3)

    Returns:
        P_aligned: Aligned points, shape (N, 3)
    """
    # Center the points
    p_mean = P.mean(dim=0, keepdim=True)
    q_mean = Q.mean(dim=0, keepdim=True)
    P_centered = P - p_mean
    Q_centered = Q - q_mean

    # Compute covariance matrix
    C = torch.matmul(P_centered.transpose(-2, -1), Q_centered)

    # Compute optimal rotation using SVD
    try:
        U, _, Vt = torch.linalg.svd(C)
        V = Vt.transpose(-2, -1)

        # Ensure proper rotation (no reflection)
        d = torch.det(torch.matmul(V, U.transpose(-2, -1)))
        if d < 0:
            V[:, 2] = -V[:, 2]

        # Compute rotation matrix
        R = torch.matmul(V, U.transpose(-2, -1))

        # Apply rotation and translation
        P_aligned = torch.matmul(P_centered, R) + q_mean

    except RuntimeError:
        # Fallback for numerical instability
        logging.warning("SVD failed in simple Kabsch, returning unaligned points.")
        P_aligned = P.clone()  # Return unaligned coordinates as fallback

    return P_aligned

def compute_fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_value: float = 10.0
) -> torch.Tensor:
    """
    Compute a simplified FAPE loss proxy based on clamped L2 distance
    after global Kabsch alignment.

    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        clamp_value: Maximum distance error to consider (Å)

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device

    # Create default mask if not provided (all positions valid)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Initialize loss
    total_loss = 0.0
    total_valid_sequences = 0

    # Process each sequence in the batch separately for Kabsch alignment
    for b in range(batch_size):
        # Extract valid coordinates for this sequence
        valid_mask = mask[b]
        valid_count = valid_mask.sum()

        if valid_count < 3:  # Need at least 3 points for meaningful alignment
            continue

        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]

        # Perform Kabsch alignment
        p_aligned = kabsch_align(p_valid, t_valid)

        # Calculate clamped L2 distance
        dist = torch.norm(p_aligned - t_valid, dim=1)
        clamped_dist = torch.clamp(dist, max=clamp_value)

        # Average over valid positions
        seq_loss = clamped_dist.mean()
        total_loss += seq_loss
        total_valid_sequences += 1

    # Handle case where all sequences in batch have insufficient valid points
    if total_valid_sequences == 0:
        return torch.tensor(0.0, device=device)

    # Average over valid sequences in batch
    loss = total_loss / total_valid_sequences
    return loss

def compute_confidence_loss(
    pred_confidence: torch.Tensor,
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    target_type: str = 'lddt_proxy',
    scaling_factor: float = 3.0
) -> torch.Tensor:
    """
    Compute confidence prediction loss using proxy targets (V1 approach).

    Train the model to predict a per-residue confidence score (similar to pLDDT)
    that correlates with the actual per-residue accuracy based on aligned coordinate error.

    Args:
        pred_confidence: Predicted confidence scores (logits), shape (batch_size, seq_len)
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        target_type: Type of target to use ('lddt_proxy' or 'distance_based')
        scaling_factor: Scaling factor for exponential target computation ('lddt_proxy')

    Returns:
        Scalar loss value
    """
    batch_size, seq_len = pred_confidence.shape
    device = pred_confidence.device

    # Create default mask if not provided (all positions valid)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Calculate residue-wise error (as proxy for lDDT)
    with torch.no_grad():
        # First align structures globally, batch by batch
        aligned_pred_coords = torch.zeros_like(pred_coords)
        for b in range(batch_size):
            valid_mask = mask[b]
            valid_count = valid_mask.sum()

            if valid_count >= 3:  # Need at least 3 points for alignment
                p_valid = pred_coords[b, valid_mask]
                t_valid = true_coords[b, valid_mask]
                aligned_pred_coords[b, valid_mask] = kabsch_align(p_valid, t_valid)
            else:
                # Not enough points, use original coords (error will be calculated but might be large)
                aligned_pred_coords[b] = pred_coords[b]

        # Compute per-residue coordinate error
        coord_error = torch.norm(aligned_pred_coords - true_coords, dim=2)

        if target_type == 'lddt_proxy':
            # Convert to per-residue lDDT-like score in [0, 1]
            # Higher is better (1.0 = perfect prediction)
            conf_targets = torch.exp(-coord_error / scaling_factor)
            conf_targets = torch.clamp(conf_targets, 0.0, 1.0) # Ensure [0, 1]

        elif target_type == 'distance_based':
            # Alternative approach: directly scale distances to [0, 1] range
            # 0 = high error, 1 = low error
            max_dist = 15.0  # Maximum distance to consider (Å)
            conf_targets = 1.0 - torch.clamp(coord_error / max_dist, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown target_type: {target_type}")

    # Apply sigmoid to predicted logits (if the model outputs logits)
    pred_probs = torch.sigmoid(pred_confidence)

    # Calculate MSE loss
    squared_error = (pred_probs - conf_targets) ** 2

    # Apply mask and average
    masked_se = squared_error * mask.float()
    num_valid = mask.sum()

    if num_valid == 0:
        return torch.tensor(0.0, device=device)

    loss = masked_se.sum() / num_valid

    return loss

def compute_angle_loss(
    pred_angles: torch.Tensor,
    true_angles: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = 'mse'
) -> torch.Tensor:
    """
    Compute loss for dihedral angle predictions.

    Compare the predicted sin/cos representations of angles with the true values.

    Args:
        pred_angles: Predicted sin/cos of angles [sin(η), cos(η), sin(θ), cos(θ)],
                    shape (batch_size, seq_len, 4)
        true_angles: True sin/cos of angles, shape (batch_size, seq_len, 4)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        loss_type: Loss function to use ('mse', 'cosine', or 'mae')

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, num_features = pred_angles.shape
    device = pred_angles.device

    # Create default mask if not provided (all positions valid)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Handle NaNs in true angles (typically at boundaries)
    angle_mask = mask.clone()
    if torch.isnan(true_angles).any():
        # Create mask for non-NaN angles
        nan_mask = ~torch.isnan(true_angles).any(dim=2)  # (batch_size, seq_len)
        angle_mask = angle_mask & nan_mask

    # Expand mask to match angle dimensions
    expanded_mask = angle_mask.unsqueeze(-1).expand_as(pred_angles)  # (batch_size, seq_len, 4)

    # Replace NaNs with zeros in true angles
    true_angles_clean = torch.nan_to_num(true_angles, nan=0.0)

    # Calculate appropriate loss
    if loss_type == 'mse':
        # Mean squared error
        squared_error = (pred_angles - true_angles_clean) ** 2
        masked_se = squared_error * expanded_mask.float()
        total_elements = expanded_mask.sum().item()

        if total_elements == 0:
            return torch.tensor(0.0, device=device)

        loss = masked_se.sum() / total_elements

    elif loss_type == 'cosine':
        # 1 - cos similarity (for angle vectors)
        # Group sin/cos pairs: (sin_eta, cos_eta) and (sin_theta, cos_theta)
        pred_eta = pred_angles[:, :, 0:2]
        pred_theta = pred_angles[:, :, 2:4]
        true_eta = true_angles_clean[:, :, 0:2]
        true_theta = true_angles_clean[:, :, 2:4]

        # Create mask for each angle type
        eta_mask = expanded_mask[:, :, 0:2]
        theta_mask = expanded_mask[:, :, 2:4]

        # Calculate cosine similarity (dot product of normalized vectors)
        # Normalize first to ensure vectors are on unit circle
        pred_eta_norm = F.normalize(pred_eta, p=2, dim=2)
        true_eta_norm = F.normalize(true_eta, p=2, dim=2)
        pred_theta_norm = F.normalize(pred_theta, p=2, dim=2)
        true_theta_norm = F.normalize(true_theta, p=2, dim=2)

        # Calculate dot products
        dot_eta = torch.sum(pred_eta_norm * true_eta_norm, dim=2)
        dot_theta = torch.sum(pred_theta_norm * true_theta_norm, dim=2)

        # 1 - cos similarity gives a value that's 0 when angles are identical
        # and 2 when they're opposite
        eta_loss = (1.0 - dot_eta) * angle_mask.float()
        theta_loss = (1.0 - dot_theta) * angle_mask.float()

        num_valid = angle_mask.sum().item()
        if num_valid == 0:
            return torch.tensor(0.0, device=device)

        loss = (eta_loss.sum() + theta_loss.sum()) / (num_valid * 2) # Average over two angles

    elif loss_type == 'mae':
        # Mean absolute error
        abs_error = torch.abs(pred_angles - true_angles_clean)
        masked_ae = abs_error * expanded_mask.float()
        total_elements = expanded_mask.sum().item()

        if total_elements == 0:
            return torch.tensor(0.0, device=device)

        loss = masked_ae.sum() / total_elements

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss

def compute_combined_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined loss from multiple loss components using basic losses.

    Args:
        outputs: Dictionary of model outputs:
            - pred_coords: Predicted coordinates (batch_size, seq_len, 3)
            - pred_confidence: Predicted confidence (batch_size, seq_len)
            - pred_angles: Predicted angles (batch_size, seq_len, 4)
        batch: Dictionary of ground truth data:
            - coordinates: True coordinates (batch_size, seq_len, 3)
            - dihedral_features: True angle features (batch_size, seq_len, 4)
            - mask: Boolean mask (batch_size, seq_len)
        loss_weights: Dictionary of loss weights:
            - fape: Weight for coordinate loss
            - confidence: Weight for confidence loss
            - angle: Weight for angle loss

    Returns:
        Tuple of:
        - total_loss: Combined weighted loss (scalar)
        - loss_components: Dictionary of individual loss components
    """
    # Extract model outputs
    pred_coords = outputs['pred_coords']
    pred_confidence = outputs['pred_confidence']
    pred_angles = outputs['pred_angles']

    # Extract ground truth
    true_coords = batch['coordinates']
    true_angles = batch['dihedral_features']
    mask = batch['mask']

    # Compute individual losses (using the basic versions defined above)
    fape_loss_val = compute_fape_loss(pred_coords, true_coords, mask)
    confidence_loss_val = compute_confidence_loss(pred_confidence, pred_coords, true_coords, mask)
    angle_loss_val = compute_angle_loss(pred_angles, true_angles, mask)

    # Extract weights with defaults
    fape_weight = loss_weights.get('fape', 1.0)
    confidence_weight = loss_weights.get('confidence', 0.1)
    angle_weight = loss_weights.get('angle', 0.5)

    # Combine losses using weights
    total_loss = (
        fape_weight * fape_loss_val +
        confidence_weight * confidence_loss_val +
        angle_weight * angle_loss_val
    )

    # Return total loss and individual components
    loss_components = {
        'fape': fape_loss_val,
        'confidence': confidence_loss_val,
        'angle': angle_loss_val,
        'total': total_loss # Store the calculated total loss here too
    }

    # Ensure components requiring grad still have it
    loss_components_tensors = {
        'fape': fape_loss_val,
        'confidence': confidence_loss_val,
        'angle': angle_loss_val,
    }

    return total_loss, loss_components_tensors # Return tensors for gradient tracking
```

### 1.2 Advanced Core Losses (V2+ / Full Implementations)

These represent more sophisticated loss functions, potentially replacing the basic proxies in future versions.

```python
def compute_lddt_target(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    cutoffs: List[float] = [0.5, 1.0, 2.0, 4.0],
    per_residue: bool = True
) -> torch.Tensor:
    """
    Compute lDDT (local Distance Difference Test) score.

    lDDT is a more accurate measure of local structure quality than simple RMSD.
    This is a full implementation that could replace the simplified proxy in V2+.

    Args:
        pred_coords: Predicted coordinates (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates (batch_size, seq_len, 3)
        mask: Boolean mask (batch_size, seq_len)
        cutoffs: Distance cutoffs for lDDT calculation (in Å)
        per_residue: Whether to return per-residue scores

    Returns:
        lDDT scores (batch_size, seq_len) if per_residue=True,
        otherwise (batch_size,)
    """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device

    # Default mask (all valid)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    all_lddt_scores = []

    for b in range(batch_size):
        # Extract valid coordinates for this batch element
        valid_mask = mask[b]
        valid_indices = torch.where(valid_mask)[0]
        num_valid = valid_indices.size(0)

        if num_valid <= 1:
            # Not enough valid atoms for pairwise distances
            lddt_b = torch.zeros(seq_len if per_residue else 1, device=device)
            if not per_residue and lddt_b.dim() > 0: # Ensure scalar for per-batch
                lddt_b = lddt_b.squeeze()
            all_lddt_scores.append(lddt_b)
            continue

        # Extract valid coordinates
        p_valid = pred_coords[b, valid_mask]  # (num_valid, 3)
        t_valid = true_coords[b, valid_mask]  # (num_valid, 3)

        # Compute all pairwise distances
        p_dists = torch.cdist(p_valid, p_valid)  # (num_valid, num_valid)
        t_dists = torch.cdist(t_valid, t_valid)  # (num_valid, num_valid)

        # Initialize per-residue lDDT scores
        lddt_residue_scores = torch.zeros(num_valid, device=device)

        # For each valid residue i
        for i in range(num_valid):
            # Define pairs to consider: other residues within 15 Å in true structure
            # Exclude self and immediate neighbors (|i-j| <= 1 in sequence space)
            local_cutoff = 15.0
            ref_dists_i = t_dists[i] # Distances from residue i in true structure

            # Sequence indices corresponding to valid atoms
            seq_indices = torch.arange(num_valid, device=device)

            # Mask for residues within 15Å and not immediate neighbors
            pair_mask_i = (ref_dists_i < local_cutoff) & \
                          (ref_dists_i > 1e-6) & \
                          (torch.abs(seq_indices - i) > 1)

            if not pair_mask_i.any():
                continue # No valid pairs for this residue

            # Get distances for valid pairs
            ref_dists_pairs = ref_dists_i[pair_mask_i]
            mod_dists_pairs = p_dists[i, pair_mask_i]

            # Calculate distance differences
            dist_diff = torch.abs(mod_dists_pairs - ref_dists_pairs)

            # Score based on cutoffs
            score = 0.0
            for cutoff in cutoffs:
                score += (dist_diff < cutoff).float()

            lddt_residue_scores[i] = score.mean() / len(cutoffs) # Average over cutoffs

        # Create full lddt tensor (with zeros for masked positions)
        if per_residue:
            lddt_b = torch.zeros(seq_len, device=device)
            lddt_b[valid_mask] = lddt_residue_scores
            all_lddt_scores.append(lddt_b)
        else:
            # Average over valid residues for batch score
            batch_score = lddt_residue_scores.mean() if num_valid > 0 else torch.tensor(0.0, device=device)
            all_lddt_scores.append(batch_score)


    # Stack all scores
    if per_residue:
        return torch.stack(all_lddt_scores)  # (batch_size, seq_len)
    else:
        # Ensure stack returns (batch_size,) even if some batches had 0 valid
        result = torch.stack([s.reshape(1) if s.dim() == 0 else s for s in all_lddt_scores]).squeeze()
        # Handle case where batch_size is 1 and result becomes scalar
        if batch_size == 1 and result.dim() == 0:
            result = result.unsqueeze(0)
        return result


def compute_lddt_confidence_loss(
    pred_confidence: torch.Tensor,
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    logits: bool = True
) -> torch.Tensor:
    """
    Compute confidence loss using proper lDDT targets (V2+ approach).

    Replaces the simpler proxy target in V1 with actual lDDT calculation.

    Args:
        pred_confidence: Predicted confidence scores, shape (batch_size, seq_len)
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len)
        logits: Whether pred_confidence contains logits (True) or probabilities (False)

    Returns:
        Confidence loss scalar
    """
    # Compute lDDT targets
    with torch.no_grad():
        # Ensure we get per-residue lDDT scores
        lddt_targets = compute_lddt_target(
            pred_coords, true_coords, mask, per_residue=True
        ) # Shape (batch_size, seq_len)

    # Process predictions (apply sigmoid if logits)
    if logits:
        pred_probs = torch.sigmoid(pred_confidence)
    else:
        pred_probs = pred_confidence

    # Default mask (all valid)
    if mask is None:
        batch_size, seq_len = pred_confidence.shape
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=pred_confidence.device)

    # Compute MSE loss (can also use BCE or others)
    squared_error = (pred_probs - lddt_targets) ** 2

    # Apply mask
    masked_se = squared_error * mask.float()

    # Compute mean loss over valid positions
    num_valid = mask.sum()
    if num_valid == 0:
        return torch.tensor(0.0, device=pred_confidence.device)

    loss = masked_se.sum() / num_valid

    return loss

def generate_local_frames(coords: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Generate local reference frames from C1' coordinates for RNA.

    This is a simplified approach that uses consecutive triplets of atoms
    (N(i-1), C1'(i), C1'(i+1)) to define local coordinate systems, similar to protein backbone frames.
    Requires at least 3 valid coordinates per sequence.

    Args:
        coords: C1' coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len)

    Returns:
        Local frames (rigid transforms), shape (batch_size, seq_len, 4, 4)
    """
    batch_size, seq_len, _ = coords.shape
    device = coords.device
    dtype = coords.dtype

    # Default mask (all valid)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Pad coordinates for easier boundary handling
    coords_padded = F.pad(coords, (0, 0, 1, 1), value=float('nan')) # Pad sequence dimension
    mask_padded = F.pad(mask, (1, 1), value=False)

    # Get coordinates for i-1, i, i+1
    coords_im1 = coords_padded[:, :-2]      # (batch_size, seq_len, 3)
    coords_i   = coords_padded[:, 1:-1]     # (batch_size, seq_len, 3)
    coords_ip1 = coords_padded[:, 2:]       # (batch_size, seq_len, 3)

    # Vectors for frame construction
    v_i_im1 = coords_i - coords_im1
    v_i_ip1 = coords_i - coords_ip1 # Note: AlphaFold uses C-N vector (ip1 - i)

    # Z-axis (Normal to the plane formed by i-1, i, i+1)
    # Cross product is robust to collinear points (results in zero vector)
    z_axis = F.normalize(torch.cross(v_i_im1, v_i_ip1, dim=2), dim=2)

    # X-axis (Vector from i to i+1, similar to protein C-N vector)
    x_axis = F.normalize(coords_ip1 - coords_i, dim=2)

    # Y-axis (Orthogonal to X and Z)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=2), dim=2)

    # Re-compute Z axis to ensure orthogonality if vectors were near parallel
    z_axis = F.normalize(torch.cross(x_axis, y_axis, dim=2), dim=2)


    # Stack axes to form rotation matrix (3x3)
    # R = [x_axis, y_axis, z_axis]^T transposed because axes are columns
    rotation = torch.stack([x_axis, y_axis, z_axis], dim=-1) # (batch_size, seq_len, 3, 3)

    # Translation vector (origin is C1' coordinate)
    translation = coords_i.unsqueeze(-1) # (batch_size, seq_len, 3, 1)

    # Construct 4x4 transformation matrix
    frames = torch.zeros(batch_size, seq_len, 4, 4, device=device, dtype=dtype)
    frames[:, :, :3, :3] = rotation
    frames[:, :, :3, 3] = coords_i # Translation part
    frames[:, :, 3, 3] = 1.0

    # Apply mask: set frames for invalid positions to identity
    identity = torch.eye(4, device=device, dtype=dtype).expand(batch_size, seq_len, 4, 4)
    frames = torch.where(mask.view(batch_size, seq_len, 1, 1), frames, identity)

    # Handle cases where vectors might be NaN due to masking or collinearity
    frames = torch.nan_to_num(frames, nan=0.0) # Replace NaN rotations with 0
    # Ensure diagonal is 1 for potentially zeroed-out matrices
    frames[..., torch.arange(4), torch.arange(4)] = torch.where(
        mask.view(batch_size, seq_len, 1).expand(-1,-1,4),
        frames[..., torch.arange(4), torch.arange(4)],
        torch.tensor(1.0, device=device, dtype=dtype)
    )
    frames[..., 3,3] = 1.0 # Ensure bottom right is always 1

    return frames


def prepare_frames_for_fape(
    model_output: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare frames for FAPE loss computation, compatible with future
    AlphaFold-style backbone frame generation.

    Args:
        model_output: Dictionary containing:
            - pred_coords: Predicted C1' coordinates
            - pred_frames (optional): Predicted frames if using advanced model
        batch: Dictionary containing:
            - coordinates: True C1' coordinates
            - mask: Boolean mask

    Returns:
        Tuple of:
        - pred_coords: Predicted coordinates
        - pred_frames: Predicted frames
        - true_coords: True coordinates
        - true_frames: True frames
    """
    # Extract data
    pred_coords = model_output['pred_coords']
    true_coords = batch['coordinates']
    mask = batch['mask']

    # Check if model already predicts frames
    if 'pred_frames' in model_output:
        # Advanced model with explicit frame prediction
        pred_frames = model_output['pred_frames']
    else:
        # Generate frames from coordinates
        pred_frames = generate_local_frames(pred_coords, mask)

    # Generate ground truth frames
    true_frames = generate_local_frames(true_coords, mask)

    return pred_coords, pred_frames, true_coords, true_frames


def compute_full_fape_loss(
    pred_coords: torch.Tensor,
    pred_frames: torch.Tensor,
    true_coords: torch.Tensor,
    true_frames: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_value: float = 10.0,
    epsilon: float = 1e-8,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute full Frame-Aligned Point Error (FAPE) loss (AlphaFold-style).

    Uses explicit coordinate frames for proper local alignment, providing
    better invariance to global transforms and measuring local structure fit.

    Args:
        pred_coords: Predicted coordinates (batch_size, seq_len, 3)
        pred_frames: Predicted frames (rigid transforms), shape (batch_size, seq_len, 4, 4)
        true_coords: Ground truth coordinates (batch_size, seq_len, 3)
        true_frames: Ground truth frames (rigid transforms), shape (batch_size, seq_len, 4, 4)
        mask: Boolean mask (batch_size, seq_len)
        clamp_value: Maximum distance error to consider (Å)
        epsilon: Small constant for numerical stability (e.g., in norm calculation)
        reduction: How to reduce the loss ('mean', 'sum', 'none')

    Returns:
        FAPE loss scalar or tensor based on reduction.
    """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device
    dtype = pred_coords.dtype

    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Reshape coordinates for broadcasting: (batch_size, seq_len, 1, 3) and (batch_size, 1, seq_len, 3)
    pred_coords_i = pred_coords.unsqueeze(2) # (B, N, 1, 3) - reference frame i
    true_coords_i = true_coords.unsqueeze(2) # (B, N, 1, 3)
    pred_coords_j = pred_coords.unsqueeze(1) # (B, 1, N, 3) - point j
    true_coords_j = true_coords.unsqueeze(1) # (B, 1, N, 3)

    # Prepare frames for inverse transformation
    # Inverse of a rigid transform [R|t] is [R^T | -R^T*t]
    pred_R_inv = pred_frames[:, :, :3, :3].transpose(-1, -2) # (B, N, 3, 3)
    pred_t = pred_frames[:, :, :3, 3] # (B, N, 3)
    pred_t_inv = -torch.matmul(pred_R_inv, pred_t.unsqueeze(-1)).squeeze(-1) # (B, N, 3)

    true_R_inv = true_frames[:, :, :3, :3].transpose(-1, -2) # (B, N, 3, 3)
    true_t = true_frames[:, :, :3, 3] # (B, N, 3)
    true_t_inv = -torch.matmul(true_R_inv, true_t.unsqueeze(-1)).squeeze(-1) # (B, N, 3)

    # Expand inverse frames for broadcasting
    pred_R_inv_i = pred_R_inv.unsqueeze(2) # (B, N, 1, 3, 3)
    pred_t_inv_i = pred_t_inv.unsqueeze(2) # (B, N, 1, 3)
    true_R_inv_i = true_R_inv.unsqueeze(2) # (B, N, 1, 3, 3)
    true_t_inv_i = true_t_inv.unsqueeze(2) # (B, N, 1, 3)

    # Transform points j into the local frame of i
    # Point_local = R_inv * (Point_global - t) = R_inv * Point_global + (-R_inv * t)
    # Unsqueeze point_j to (B, 1, N, 1, 3) for matmul with R_inv_i
    pred_coords_j_in_i = torch.matmul(pred_coords_j.unsqueeze(-2), pred_R_inv_i) # (B, N, N, 1, 3)
    pred_coords_j_in_i = pred_coords_j_in_i.squeeze(-2) + pred_t_inv_i # (B, N, N, 3)

    true_coords_j_in_i = torch.matmul(true_coords_j.unsqueeze(-2), true_R_inv_i) # (B, N, N, 1, 3)
    true_coords_j_in_i = true_coords_j_in_i.squeeze(-2) + true_t_inv_i # (B, N, N, 3)

    # Compute the L2 norm of the difference (FAPE error)
    error_dist = torch.sqrt(
        torch.sum((pred_coords_j_in_i - true_coords_j_in_i)**2, dim=-1) + epsilon # (B, N, N)
    )

    # Clamp the error
    clamped_error = torch.clamp(error_dist, max=clamp_value)

    # Create pairwise mask from the sequence mask
    # Valid pair (i, j) if both mask[i] and mask[j] are True
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) # (B, N, N)

    # Apply the mask (set errors for invalid pairs to 0)
    masked_error = clamped_error * pair_mask.float()

    # Perform reduction
    if reduction == 'none':
        # Return per-pair errors, possibly averaged over i for each j? AlphaFold averages over j for each i.
        # Let's return average over j for each residue i
        num_valid_pairs_per_i = pair_mask.sum(dim=2).float() # (B, N)
        loss = masked_error.sum(dim=2) / (num_valid_pairs_per_i + epsilon) # (B, N)
        loss = loss * mask.float() # Mask out invalid residues
    elif reduction == 'sum':
        loss = masked_error.sum()
    else: # 'mean'
        # Average over all valid pairs in the batch
        num_valid_pairs = pair_mask.sum()
        if num_valid_pairs == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        loss = masked_error.sum() / num_valid_pairs

    return loss
```

## Section 2: Advanced Loss Strategies

Techniques to combine, weight, or enhance the core losses.

```python
def compute_ensemble_fape_loss(
    ensemble_pred_coords: List[torch.Tensor],
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ensemble_strategy: str = 'mean',
    clamp_value: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute FAPE loss using an ensemble of model predictions (using simplified FAPE).

    Args:
        ensemble_pred_coords: List of predicted coordinates, each with shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        ensemble_strategy: Strategy for ensemble combination ('mean', 'min', or 'weighted')
        clamp_value: Maximum distance error to consider

    Returns:
        Tuple of:
        - ensemble_loss: Scalar loss value
        - individual_losses: Tensor of losses for each ensemble member
    """
    device = true_coords.device
    num_models = len(ensemble_pred_coords)

    if num_models == 0:
        raise ValueError("Empty ensemble provided")

    # Compute loss for each ensemble member using the simplified FAPE proxy
    individual_losses = torch.zeros(num_models, device=device)
    for i, pred_coords in enumerate(ensemble_pred_coords):
        # Use compute_fape_loss (simplified proxy) here
        individual_losses[i] = compute_fape_loss(pred_coords, true_coords, mask, clamp_value)

    # Combine losses according to strategy
    if ensemble_strategy == 'mean':
        # Simple average of all ensemble members
        ensemble_loss = individual_losses.mean()

    elif ensemble_strategy == 'min':
        # Use only the best prediction (min loss)
        ensemble_loss = individual_losses.min()

    elif ensemble_strategy == 'weighted':
        # Weight inversely proportional to loss value
        # Lower loss = higher weight
        weights = 1.0 / (individual_losses + 1e-8)
        weights = weights / weights.sum()  # Normalize weights
        ensemble_loss = (individual_losses * weights).sum()

    else:
        raise ValueError(f"Unknown ensemble_strategy: {ensemble_strategy}")

    return ensemble_loss, individual_losses


def compute_ensemble_prediction(
    ensemble_pred_coords: List[torch.Tensor],
    ensemble_pred_confidence: List[torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    strategy: str = 'confidence_weighted'
) -> torch.Tensor:
    """
    Compute a consensus prediction from an ensemble of models.

    Args:
        ensemble_pred_coords: List of predicted coordinates, each with shape (batch_size, seq_len, 3)
        ensemble_pred_confidence: List of predicted confidence scores (logits), each with shape (batch_size, seq_len)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        strategy: Strategy for combining predictions ('confidence_weighted', 'mean', or 'median')

    Returns:
        Consensus coordinates, shape (batch_size, seq_len, 3)
    """
    if not ensemble_pred_coords:
        raise ValueError("Empty ensemble provided")

    device = ensemble_pred_coords[0].device
    batch_size, seq_len, _ = ensemble_pred_coords[0].shape
    num_models = len(ensemble_pred_coords)

    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Stack predictions from all models
    stacked_coords = torch.stack(ensemble_pred_coords, dim=0)  # (num_models, batch_size, seq_len, 3)

    # Combine according to strategy
    if strategy == 'confidence_weighted':
        if not ensemble_pred_confidence or len(ensemble_pred_confidence) != num_models:
            raise ValueError("Confidence scores must be provided for confidence_weighted strategy.")

        # Convert confidence scores (logits) to weights (probabilities)
        conf_probs = [torch.sigmoid(conf) for conf in ensemble_pred_confidence]
        stacked_conf = torch.stack(conf_probs, dim=0)  # (num_models, batch_size, seq_len)

        # Apply weights along model dimension
        weights = stacked_conf.unsqueeze(-1)  # (num_models, batch_size, seq_len, 1)
        weighted_sum = (stacked_coords * weights).sum(dim=0)  # (batch_size, seq_len, 3)
        weight_sum = weights.sum(dim=0) + 1e-8  # (batch_size, seq_len, 1)

        consensus_coords = weighted_sum / weight_sum  # (batch_size, seq_len, 3)

    elif strategy == 'mean':
        # Simple average across models
        consensus_coords = stacked_coords.mean(dim=0)  # (batch_size, seq_len, 3)

    elif strategy == 'median':
        # Median across models (more robust to outliers)
        # Note: torch.median returns (values, indices)
        consensus_coords = torch.median(stacked_coords, dim=0)[0]  # (batch_size, seq_len, 3)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply mask to ensure padded positions are zero
    mask_3d = mask.unsqueeze(-1).expand(-1, -1, 3)  # (batch_size, seq_len, 3)
    consensus_coords = consensus_coords * mask_3d.float()

    return consensus_coords


class CurriculumLossManager:
    """
    Manages curriculum learning for loss functions.

    Handles the progressive adjustment of loss weights and potentially difficulty focus
    throughout training.
    """

    def __init__(
        self,
        initial_weights: Dict[str, float],
        final_weights: Dict[str, float],
        total_epochs: int,
        curriculum_type: str = 'linear',
        difficulty_curriculum: bool = False # Set to False initially, add compute_difficulty_weighted_loss if True
    ):
        """
        Initialize curriculum loss manager.

        Args:
            initial_weights: Initial loss weights
            final_weights: Final loss weights
            total_epochs: Total number of training epochs
            curriculum_type: Type of weight schedule ('linear', 'exponential', or 'step')
            difficulty_curriculum: Whether to use difficulty-based curriculum (Not fully implemented below)
        """
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.total_epochs = total_epochs
        self.curriculum_type = curriculum_type
        self.difficulty_curriculum = difficulty_curriculum

        # Validate weights
        for key in self.final_weights:
            if key not in self.initial_weights:
                raise ValueError(f"Loss component {key} is missing from initial_weights")

        for key in self.initial_weights:
            if key not in self.final_weights:
                raise ValueError(f"Loss component {key} is missing from final_weights")

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        Get current loss weights based on curriculum schedule.

        Args:
            epoch: Current epoch number (0-indexed)

        Returns:
            Current loss weights
        """
        # Ensure epoch is in valid range
        epoch = max(0, min(epoch, self.total_epochs - 1))

        # Calculate progress through curriculum
        progress = epoch / max(1, self.total_epochs - 1) # Avoid division by zero if total_epochs=1

        # Apply schedule based on curriculum type
        if self.curriculum_type == 'linear':
            weights = self._linear_schedule(progress)
        elif self.curriculum_type == 'exponential':
            weights = self._exponential_schedule(progress)
        elif self.curriculum_type == 'step':
            weights = self._step_schedule(progress)
        else:
            raise ValueError(f"Unknown curriculum_type: {self.curriculum_type}")

        return weights

    def _linear_schedule(self, progress: float) -> Dict[str, float]:
        """Linear interpolation between initial and final weights."""
        weights = {}
        for key, initial_value in self.initial_weights.items():
            final_value = self.final_weights[key]
            weights[key] = initial_value + progress * (final_value - initial_value)
        return weights

    def _exponential_schedule(self, progress: float) -> Dict[str, float]:
        """Exponential interpolation for faster weight changes near beginning."""
        weights = {}
        for key, initial_value in self.initial_weights.items():
            final_value = self.final_weights[key]
            # Exponential curve: uses simple power, adjust exponent for curve shape
            curve_progress = progress ** 2 # Starts slow, accelerates
            weights[key] = initial_value + curve_progress * (final_value - initial_value)
        return weights

    def _step_schedule(self, progress: float) -> Dict[str, float]:
        """Step-wise schedule with discrete transitions."""
        weights = {}
        # Define steps (e.g., 3 steps at 0%, 50%, 100% progress)
        steps = [0.0, 0.5, 1.0]
        num_steps = len(steps) - 1

        # Find current interval index
        current_interval = 0
        for i in range(num_steps):
            if progress >= steps[i] and progress < steps[i+1]:
                current_interval = i
                break
        if progress == steps[-1]: # Handle exact endpoint
             current_interval = num_steps - 1

        # Interpolate between initial and final based on interval
        interval_progress = current_interval / max(1, num_steps - 1) if num_steps > 1 else 1.0

        for key, initial_value in self.initial_weights.items():
            final_value = self.final_weights[key]
            weights[key] = initial_value + interval_progress * (final_value - initial_value)

        return weights

    def compute_difficulty_weighted_loss(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
        epoch: int,
        base_loss_fn: callable = compute_fape_loss, # Use simplified FAPE as base
        clamp_value: float = 10.0,
        max_error_focus: float = 15.0
    ) -> torch.Tensor:
        """
        Compute difficulty-weighted loss based on curriculum progress.

        Early in training, focus on easy examples (low error). Gradually shift focus to harder ones.
        Uses Kabsch-aligned per-residue error as difficulty measure.

        Args:
            pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
            true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
            mask: Boolean mask, shape (batch_size, seq_len)
            epoch: Current epoch number
            base_loss_fn: Function to compute base loss (e.g., FAPE proxy)
            clamp_value: Clamping value for the base loss calculation.
            max_error_focus: Error level considered 'hardest' for weighting.

        Returns:
            Difficulty-weighted loss value
        """
        if not self.difficulty_curriculum:
            # If difficulty curriculum is disabled, just use standard base loss
            return base_loss_fn(pred_coords, true_coords, mask, clamp_value=clamp_value)

        batch_size, seq_len, _ = pred_coords.shape
        device = pred_coords.device

        # Calculate per-residue errors after alignment
        per_residue_error = torch.zeros((batch_size, seq_len), device=device)
        with torch.no_grad(): # Don't need gradients through error calculation itself
             for b in range(batch_size):
                 valid_mask = mask[b]
                 valid_count = valid_mask.sum()
                 if valid_count >= 3:
                     p_valid = pred_coords[b, valid_mask]
                     t_valid = true_coords[b, valid_mask]
                     p_aligned = kabsch_align(p_valid, t_valid)
                     errors = torch.norm(p_aligned - t_valid, dim=1)
                     per_residue_error[b, valid_mask] = errors
                 # else: errors remain 0 for invalid sequences/residues

        # Progress through curriculum (0 to 1)
        progress = epoch / max(1, self.total_epochs - 1)

        # Create difficulty weights based on error and progress
        # Normalized error (0=easy, 1=hard)
        normalized_error = torch.clamp(per_residue_error / max_error_focus, 0.0, 1.0)

        # Weight schedule:
        # progress=0 => weight = 1 - normalized_error (focus on easy)
        # progress=1 => weight = normalized_error (focus on hard)
        difficulty_weights = progress * normalized_error + (1 - progress) * (1 - normalized_error)

        # Ensure weights are 0 for masked positions
        difficulty_weights = difficulty_weights * mask.float()

        # --- Compute the base loss (e.g., FAPE) ---
        # This part *does* need gradients wrt pred_coords
        total_loss = 0.0
        total_valid_sequences = 0
        for b in range(batch_size):
            valid_mask = mask[b]
            valid_count = valid_mask.sum()
            if valid_count < 3:
                continue

            p_valid = pred_coords[b, valid_mask]
            t_valid = true_coords[b, valid_mask]
            p_aligned = kabsch_align(p_valid, t_valid) # Alignment needs gradients

            # Calculate clamped L2 distance (the actual loss term)
            dist = torch.norm(p_aligned - t_valid, dim=1)
            clamped_dist = torch.clamp(dist, max=clamp_value) # This is the per-residue loss

            # Get weights for this sequence's valid residues
            seq_weights = difficulty_weights[b, valid_mask]

            # Apply difficulty weights to the loss term
            # Normalize weights for this sequence
            weight_sum = seq_weights.sum() + 1e-8
            normalized_seq_weights = seq_weights / weight_sum

            weighted_seq_loss = (clamped_dist * normalized_seq_weights).sum()

            total_loss += weighted_seq_loss
            total_valid_sequences += 1

        if total_valid_sequences == 0:
            return torch.tensor(0.0, device=device)

        # Average over valid sequences
        final_loss = total_loss / total_valid_sequences
        return final_loss


class ExternalMetricLoss:
    """
    Integrates external structural biology metrics (e.g., TM-score, RMSD) into loss computation.
    Requires external tools or libraries to compute metrics. This is a placeholder structure.
    """

    def __init__(
        self,
        use_tm_score: bool = False,
        use_rmsd: bool = False,
        tm_weight: float = 0.5,
        rmsd_weight: float = 0.5,
        # cache_dir: str = '/tmp/metric_cache' # Caching implementation omitted
    ):
        """
        Initialize external metric loss.

        Args:
            use_tm_score: Whether to use TM-score
            use_rmsd: Whether to use RMSD
            tm_weight: Weight for TM-score component
            rmsd_weight: Weight for RMSD component
        """
        self.use_tm_score = use_tm_score
        self.use_rmsd = use_rmsd
        self.tm_weight = tm_weight
        self.rmsd_weight = rmsd_weight
        # self.cache_dir = cache_dir
        # if (use_tm_score or use_rmsd) and not os.path.exists(cache_dir):
        #     os.makedirs(cache_dir)

    def compute_loss(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        mask: torch.Tensor,
        target_ids: List[str], # Assumes target IDs are available
        base_loss: torch.Tensor # Existing loss to augment
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss incorporating external metric values.

        NOTE: This requires detaching tensors and running external non-differentiable
              computations (like TM-score). It's primarily useful for evaluation
              or potentially reinforcement learning style updates, not direct gradient descent.

        Args:
            pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
            true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
            mask: Boolean mask, shape (batch_size, seq_len)
            target_ids: List of target IDs in batch (for potential caching/logging)
            base_loss: Base loss value from other loss functions (already computed)

        Returns:
            Tuple of:
            - combined_loss: Base loss potentially adjusted by external metrics (may not be differentiable)
            - metrics: Dictionary of computed metrics (non-tensor floats)
        """
        batch_size = pred_coords.shape[0]
        device = pred_coords.device # Loss adjustment should be on same device

        # Initialize metrics dictionary
        metrics = {
            'tm_score': 0.0,
            'rmsd': 0.0
        }

        # Skip external metrics if not requested
        if not (self.use_tm_score or self.use_rmsd):
            return base_loss, metrics

        # Initialize metric values (as python lists to accumulate floats)
        tm_scores_list = []
        rmsds_list = []

        # Process each structure in batch
        for b in range(batch_size):
            target_id = target_ids[b] if target_ids else f"batch_{b}"

            # Extract valid coordinates
            valid_mask = mask[b]
            valid_count = valid_mask.sum().item()

            if valid_count < 3:  # Need at least 3 valid positions
                if self.use_tm_score: tm_scores_list.append(0.0)
                if self.use_rmsd: rmsds_list.append(float('inf')) # High RMSD for invalid
                continue

            # Detach and move to CPU for external calculation
            pred_valid_np = pred_coords[b, valid_mask].detach().cpu().numpy()
            true_valid_np = true_coords[b, valid_mask].detach().cpu().numpy()

            # Compute metrics using external tools or internal implementations
            if self.use_tm_score:
                # Placeholder - replace with actual TM-score call
                tm_score = self._compute_tm_score(pred_valid_np, true_valid_np, target_id)
                tm_scores_list.append(tm_score)

            if self.use_rmsd:
                # Placeholder - replace with actual RMSD call (after alignment)
                rmsd = self._compute_rmsd(pred_valid_np, true_valid_np, target_id)
                rmsds_list.append(rmsd)

        # Compute average metrics over batch
        avg_tm_score = np.mean(tm_scores_list) if tm_scores_list else 0.0
        avg_rmsd = np.mean([r for r in rmsds_list if r != float('inf')]) if any(r != float('inf') for r in rmsds_list) else 0.0

        # Update metrics dictionary (non-tensor values)
        metrics['tm_score'] = avg_tm_score
        metrics['rmsd'] = avg_rmsd

        # --- Loss Adjustment (Non-Differentiable Part) ---
        # This part modifies the loss based on metrics but gradients won't flow back
        # through the metric calculation. Use with caution.
        combined_loss = base_loss # Start with the differentiable base loss

        if self.use_tm_score and tm_scores_list:
            # TM-score is [0,1] where 1 is best, so use (1 - TM_score) as penalty
            # Average TM penalty across batch, convert to tensor on correct device
            tm_penalty = torch.tensor(1.0 - avg_tm_score, device=device, dtype=base_loss.dtype)
            combined_loss = combined_loss + self.tm_weight * tm_penalty

        if self.use_rmsd and any(r != float('inf') for r in rmsds_list):
            # RMSD is unbounded, lower is better
            # Scale to avoid dominating the loss, convert to tensor
            rmsd_scale = 0.1  # Adjust based on expected RMSD range
            rmsd_penalty = torch.tensor(avg_rmsd * rmsd_scale, device=device, dtype=base_loss.dtype)
            combined_loss = combined_loss + self.rmsd_weight * rmsd_penalty

        return combined_loss, metrics

    def _compute_tm_score(
        self,
        pred_coords_np: np.ndarray,
        true_coords_np: np.ndarray,
        target_id: str # For potential caching/logging
    ) -> float:
        """
        Placeholder: Compute TM-score using external tool (e.g., US-align) or approximation.

        Args:
            pred_coords_np: Predicted coordinates as NumPy array, shape (N, 3)
            true_coords_np: True coordinates as NumPy array, shape (N, 3)
            target_id: Target ID

        Returns:
            TM-score value (float)
        """
        # In practice: write PDB files, run US-align subprocess, parse output.
        # Using a simplified RMSD-based approximation for demonstration:
        N = pred_coords_np.shape[0]
        if N < 3: return 0.0

        rmsd_val = self._compute_rmsd(pred_coords_np, true_coords_np, target_id)

        # TM-score scale factor (d0) depends on length N
        d0 = 1.24 * np.cbrt(max(0, N - 15)) - 1.8
        d0 = max(d0, 0.5) # Ensure positive scale

        tm_score = 1.0 / (1.0 + (rmsd_val / d0) ** 2)
        return float(tm_score)

    def _compute_rmsd(
        self,
        pred_coords_np: np.ndarray,
        true_coords_np: np.ndarray,
        target_id: str # For potential caching/logging
    ) -> float:
        """
        Placeholder: Compute RMSD after Kabsch alignment using NumPy.

        Args:
            pred_coords_np: Predicted coordinates as NumPy array, shape (N, 3)
            true_coords_np: True coordinates as NumPy array, shape (N, 3)
            target_id: Target ID

        Returns:
            RMSD value (float)
        """
        if pred_coords_np.shape[0] < 1: return float('inf')

        # Center coordinates
        pred_mean = np.mean(pred_coords_np, axis=0)
        true_mean = np.mean(true_coords_np, axis=0)
        pred_centered = pred_coords_np - pred_mean
        true_centered = true_coords_np - true_mean

        # Compute covariance matrix
        H = pred_centered.T @ true_centered

        # Compute optimal rotation via SVD
        try:
            U, _, Vt = np.linalg.svd(H)
            V = Vt.T
            # Ensure proper rotation (handle reflection)
            d = np.sign(np.linalg.det(V @ U.T))
            Z = np.identity(3)
            Z[2, 2] = d
            R = V @ Z @ U.T
        except np.linalg.LinAlgError:
             # SVD failed, use identity rotation
             R = np.identity(3)


        # Align coordinates
        pred_aligned = pred_centered @ R + true_mean

        # Compute RMSD
        diff = pred_aligned - true_coords_np
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

        return float(rmsd)


def compute_position_weighted_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
    clamp_value: float = 10.0,
    position_weights: Optional[torch.Tensor] = None,
    weighting_strategy: str = 'uniform', # Default to uniform if no weights provided
    strategy_params: Optional[Dict] = None # For strategies needing params
) -> torch.Tensor:
    """
    Compute coordinate loss (FAPE proxy) with position-dependent weighting.

    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len)
        clamp_value: Maximum distance error to consider for FAPE proxy.
        position_weights: Optional pre-computed weights, shape (batch_size, seq_len).
                          If provided, weighting_strategy is ignored.
        weighting_strategy: Strategy for generating weights if not provided
                           ('secondary_structure', 'conservation',
                            'distance_from_center', 'uniform').
        strategy_params: Dictionary of parameters for the weighting strategy. E.g.,
                         {'secondary_structure': {'ss_mask': tensor, 'stem_weight': 1.5}}
                         {'conservation': {'scores': tensor, 'scale': 0.5}}

    Returns:
        Weighted loss value scalar.
    """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device

    # Generate weights if not provided
    if position_weights is None:
        if weighting_strategy == 'uniform':
            position_weights = torch.ones((batch_size, seq_len), device=device)
        elif weighting_strategy == 'distance_from_center':
            center_idx = seq_len // 2
            indices = torch.arange(seq_len, device=device).float()
            # Example: linear decay from center (max weight 1.5 at center, min 1.0 at ends)
            max_weight = 1.5
            min_weight = 1.0
            relative_pos = torch.abs(indices - center_idx) / max(1, seq_len / 2)
            pos_weights_1d = min_weight + (max_weight - min_weight) * (1.0 - relative_pos)
            position_weights = pos_weights_1d.unsqueeze(0).expand(batch_size, -1)
        elif weighting_strategy == 'secondary_structure':
            # Requires secondary structure information, e.g., a mask for stems
            # Example: weight stems higher
            if strategy_params is None or 'ss_mask' not in strategy_params['secondary_structure']:
                 raise ValueError("Secondary structure mask needed for ss weighting.")
            ss_mask = strategy_params['secondary_structure']['ss_mask'] # (batch, seq_len) bool/float
            stem_weight = strategy_params['secondary_structure'].get('stem_weight', 1.5)
            loop_weight = 1.0
            position_weights = torch.where(ss_mask.bool(), stem_weight, loop_weight) * torch.ones_like(mask, dtype=torch.float32)
        elif weighting_strategy == 'conservation':
             # Requires conservation scores
            if strategy_params is None or 'scores' not in strategy_params['conservation']:
                 raise ValueError("Conservation scores needed for conservation weighting.")
            cons_scores = strategy_params['conservation']['scores'] # (batch, seq_len) range [0, 1]
            scale = strategy_params['conservation'].get('scale', 0.5) # How much conservation affects weight
            position_weights = 1.0 + scale * cons_scores # Base weight 1, scaled by conservation
        else:
            raise ValueError(f"Unknown weighting_strategy: {weighting_strategy}")

    # Apply sequence mask to weights (masked positions have zero weight)
    position_weights = position_weights * mask.float()

    # Normalize weights *within each sequence* so total weight per sequence is constant
    # This prevents sequences with generally higher weights from dominating the batch loss
    weight_sums = position_weights.sum(dim=1, keepdim=True) + 1e-8
    normalized_weights = position_weights / weight_sums

    # Compute per-residue loss after Kabsch alignment (FAPE proxy)
    total_weighted_loss = 0.0
    total_valid_sequences = 0

    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum()
        if valid_count < 3:
            continue

        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]
        p_aligned = kabsch_align(p_valid, t_valid) # Needs gradients

        # Per-residue errors
        errors = torch.norm(p_aligned - t_valid, dim=1)
        clamped_errors = torch.clamp(errors, max=clamp_value) # This is the per-residue loss

        # Get normalized weights for valid residues of this sequence
        seq_norm_weights = normalized_weights[b, valid_mask]

        # Apply weights to the per-residue loss for this sequence
        # We sum here because weights are normalized to sum to 1 per sequence
        weighted_sequence_loss = (clamped_errors * seq_norm_weights).sum()

        total_weighted_loss += weighted_sequence_loss
        total_valid_sequences += 1

    if total_valid_sequences == 0:
        return torch.tensor(0.0, device=device)

    # Average the weighted losses across sequences in the batch
    final_loss = total_weighted_loss / total_valid_sequences
    return final_loss


class AdaptiveLossWeights(nn.Module):
    """
    Module for learning adaptive weights for different loss components.

    Uses the approach from "Multi-Task Learning Using Uncertainty to Weigh Losses..."
    by Kendall, Gal, and Cipolla (2018). Learns log variances for each task.
    """

    def __init__(
        self,
        loss_names: List[str],
        initial_log_vars: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize adaptive loss weights.

        Args:
            loss_names: List of names for the loss components to be weighted.
            initial_log_vars: Optional dictionary of initial log variances. Defaults to 0.0.
        """
        super().__init__()

        self.loss_names = loss_names
        num_losses = len(loss_names)

        # Initialize learnable log variances (one per loss component)
        if initial_log_vars is None:
            init_vals = torch.zeros(num_losses, dtype=torch.float32)
        else:
            init_vals = torch.tensor([initial_log_vars.get(name, 0.0) for name in loss_names], dtype=torch.float32)

        self.log_vars = nn.Parameter(init_vals)

    def forward(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the adaptively weighted total loss.

        Args:
            losses: Dictionary mapping loss names to their computed tensor values.

        Returns:
            Tuple of:
            - total_weighted_loss: The combined loss scalar.
            - current_weights: Dictionary of the calculated weights (1 / (2*sigma^2))
                               and log variances used.
        """
        if set(losses.keys()) != set(self.loss_names):
            raise ValueError(f"Input losses {list(losses.keys())} do not match expected {self.loss_names}")

        total_loss = 0.0
        current_weights = {}

        for i, name in enumerate(self.loss_names):
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var) # precision = 1 / sigma^2

            # Weighted loss term: L_i / (2 * sigma_i^2) + log(sigma_i)
            # = L_i * exp(-log_var_i) / 2 + log_var_i / 2
            loss_term = 0.5 * precision * losses[name] + 0.5 * log_var
            total_loss += loss_term

            # Store weights for logging/analysis
            # Weight is effectively 1 / (2 * sigma^2)
            current_weights[f'{name}_weight'] = (0.5 * precision).item()
            current_weights[f'{name}_log_var'] = log_var.item()

        return total_loss, current_weights


class StructureSampler:
    """
    Generates diverse structure predictions using confidence-guided sampling or other strategies.
    Requires modifications to the model's forward pass or internal mechanisms for some strategies.
    """

    def __init__(
        self,
        num_samples: int = 5,
        sampling_strategy: str = 'noise_injection', # 'noise_injection', 'dropout', 'temperature'
        noise_scale: float = 0.5, # For 'noise_injection'
        temperature: float = 1.0, # For 'temperature' scaling (if model supports it)
        use_confidence_for_noise: bool = False # Scale noise by inverse confidence
    ):
        """
        Initialize structure sampler.

        Args:
            num_samples: Number of structure samples to generate.
            sampling_strategy: Strategy for generating samples.
                               'noise_injection': Adds noise to intermediate representations or inputs.
                               'dropout': Runs model in train mode with dropout enabled.
                               'temperature': Scales logits before softmax (requires model support).
            noise_scale: Scale for noise addition in 'noise_injection'.
            temperature: Temperature parameter for 'temperature' scaling.
            use_confidence_for_noise: If True and strategy is 'noise_injection', scale noise
                                      inversely proportional to predicted confidence.
        """
        self.num_samples = num_samples
        self.sampling_strategy = sampling_strategy
        self.noise_scale = noise_scale
        self.temperature = temperature
        self.use_confidence_for_noise = use_confidence_for_noise

        if sampling_strategy not in ['noise_injection', 'dropout', 'temperature']:
             raise ValueError(f"Unsupported sampling_strategy: {sampling_strategy}")

    def generate_samples(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate multiple structure samples.

        Args:
            model: RNA folding model.
            batch: Input batch.
            device: Device to run model on.

        Returns:
            Tuple of:
            - pred_coords_list: List of coordinate predictions [num_samples, (B, N, 3)].
            - pred_conf_list: List of confidence predictions [num_samples, (B, N)].
        """
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        pred_coords_list = []
        pred_conf_list = []

        original_mode = model.training # Store original mode

        if self.sampling_strategy == 'dropout':
            model.train() # Enable dropout for sampling
        else:
            model.eval() # Standard eval mode for other strategies

        for i in range(self.num_samples):
            torch.manual_seed(i) # For reproducible sampling if needed

            with torch.no_grad():
                if self.sampling_strategy == 'noise_injection':
                    # Requires modifying the batch or having hooks into the model
                    # Simplified: Add noise to input coords if available, or just run model multiple times
                    # This example assumes model takes optional noise scale
                    outputs = self._noise_injection_forward(model, batch_on_device, i)

                elif self.sampling_strategy == 'dropout':
                    # Simply run forward pass with model in train() mode
                    outputs = model(batch_on_device)

                elif self.sampling_strategy == 'temperature':
                     # Requires model to support temperature scaling internally
                     # This is a placeholder call
                     outputs = self._temperature_scaled_forward(model, batch_on_device, self.temperature)

                else: # Should not happen due to init check
                    outputs = model(batch_on_device)

            pred_coords = outputs.get('pred_coords')
            pred_confidence = outputs.get('pred_confidence') # Logits

            if pred_coords is not None: pred_coords_list.append(pred_coords)
            if pred_confidence is not None: pred_conf_list.append(pred_confidence)

        model.train(original_mode) # Restore original mode

        return pred_coords_list, pred_conf_list


    def _noise_injection_forward(self, model, batch, seed):
        """ Helper for noise injection - requires model support or modification """
        # --- Strategy 1: Add noise to initial coordinates (if model uses them) ---
        if 'initial_coords' in batch:
            coords_to_noise = batch['initial_coords']
        elif 'coords' in batch: # Fallback to target coords? Risky.
             coords_to_noise = batch['coords']
        else:
             # Cannot inject noise at input level
             logging.warning("Noise injection requires 'initial_coords' in batch or model support.")
             return model(batch)

        mask = batch.get('mask')
        pred_coords_base = model(batch)['pred_coords'] # Get base prediction for confidence
        pred_conf_base = model(batch)['pred_confidence'] # Logits

        noise_scale = self.noise_scale
        if self.use_confidence_for_noise and pred_conf_base is not None:
            conf_probs = torch.sigmoid(pred_conf_base) # (B, N)
            inv_conf = 1.0 - conf_probs # Low conf = high value
            noise_scaling_factor = inv_conf.unsqueeze(-1) * noise_scale # (B, N, 1)
        else:
            noise_scaling_factor = noise_scale

        torch.manual_seed(seed)
        noise = torch.randn_like(coords_to_noise) * noise_scaling_factor

        if mask is not None:
            noise = noise * mask.unsqueeze(-1).float()

        perturbed_batch = batch.copy()
        perturbed_batch['initial_coords'] = coords_to_noise + noise

        # Run model with perturbed input
        outputs = model(perturbed_batch)
        return outputs

        # --- Strategy 2: Model supports noise injection internally ---
        # outputs = model(batch, noise_scale=self.noise_scale, seed=seed)
        # return outputs


    def _temperature_scaled_forward(self, model, batch, temperature):
         """ Helper for temperature scaling - requires model support """
         # This usually happens internally before a softmax
         # e.g., model(batch, temperature=temperature)
         logging.warning("Temperature scaling requires internal model support.")
         # Fallback: run normally
         outputs = model(batch)
         return outputs


def format_kaggle_submission(
    pred_coords_list: List[torch.Tensor],
    target_ids: List[str],
    sequences: List[str],
    mask_list: Optional[List[torch.Tensor]] = None, # Optional mask list if available
    output_path: str = 'submission.csv'
) -> None:
    """
    Format multiple structure predictions (e.g., 5 models/samples) for Kaggle submission CSV.

    Args:
        pred_coords_list: List of coordinate predictions, each tensor shape (batch_size, seq_len, 3).
                          Typically 5 tensors for Kaggle RNA challenge.
        target_ids: List of target IDs corresponding to batch dim (len=batch_size).
        sequences: List of RNA sequences corresponding to batch dim (len=batch_size).
        mask_list: Optional list of masks corresponding to pred_coords_list. If provided,
                   only unmasked coordinates are written. Assumes masks are (batch_size, seq_len).
        output_path: Path to output CSV file.
    """
    if pd is None:
        logging.error("pandas is required to format Kaggle submission.")
        return

    num_models = len(pred_coords_list)
    if num_models == 0:
        logging.error("No prediction coordinates provided.")
        return

    batch_size = pred_coords_list[0].shape[0]
    if len(target_ids) != batch_size or len(sequences) != batch_size:
        raise ValueError("Length of target_ids and sequences must match batch size.")

    # Convert to CPU and numpy
    coords_numpy = [p.detach().cpu().numpy() for p in pred_coords_list]
    masks_numpy = None
    if mask_list:
        if len(mask_list) != num_models:
             logging.warning("Mask list length doesn't match prediction list length. Ignoring masks.")
        else:
            masks_numpy = [m.detach().cpu().numpy() for m in mask_list]


    rows = []
    nuc_map = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U', 'T': 'U'} # Handle T if present

    for b in range(batch_size):
        target_id = target_ids[b]
        sequence = sequences[b]
        seq_len = len(sequence)

        for i in range(seq_len):
            # Check masks if provided
            is_valid = True
            if masks_numpy:
                 # Check if valid in *at least one* model's mask for this position
                 is_valid = any(masks_numpy[m][b, i] for m in range(num_models))

            if not is_valid:
                 continue # Skip masked residues entirely

            resname = nuc_map.get(sequence[i].upper(), 'X') # Use upper case, default 'X'
            row = {
                'ID': f"{target_id}_{i+1}",  # Kaggle uses 1-based indexing
                'resname': resname,
                'resid': i + 1             # 1-based residue index
            }

            # Add coordinates from all models/samples
            for m in range(num_models):
                coords = coords_numpy[m][b, i]
                # Handle potential NaNs/Infs from prediction if needed
                row[f'x_{m+1}'] = float(np.nan_to_num(coords[0]))
                row[f'y_{m+1}'] = float(np.nan_to_num(coords[1]))
                row[f'z_{m+1}'] = float(np.nan_to_num(coords[2]))

            rows.append(row)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"Submission formatted for {len(rows)} residues saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving submission file: {e}")

```

## Section 3: Implementation, Optimization, and Troubleshooting

Guidance on practical aspects of using these loss functions.

### 3.1 Troubleshooting

Detecting and handling common issues like NaNs and gradient problems.

```python
def detect_and_handle_nan_loss(
    loss: torch.Tensor,
    loss_components: Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor] # For logging context
) -> Tuple[bool, torch.Tensor]:
    """
    Detect NaN in loss values and respond appropriately.

    Args:
        loss: Total loss tensor (scalar)
        loss_components: Dictionary of individual loss components (tensors)
        model: The model being trained
        optimizer: The optimizer
        batch: Input batch that produced the loss (for logging)

    Returns:
        Tuple of:
        - bool: Whether step should be skipped (True if NaN detected)
        - torch.Tensor: Original loss (if no NaN) or a zero tensor (if NaN and skipping)
    """
    skip_step = False
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        logging.warning(f"NaN or Inf detected in total loss: {loss.item()}!")
        skip_step = True

        # Identify problematic components
        for name, comp_loss in loss_components.items():
            if torch.isnan(comp_loss).any() or torch.isinf(comp_loss).any():
                 logging.warning(f"  Component '{name}' has NaN/Inf: {comp_loss.item()}")

        # Check parameters and gradients for NaNs/Infs
        for name, param in model.named_parameters():
             if param.grad is not None:
                 if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                     logging.warning(f"  NaN/Inf gradient detected in parameter: {name}")
             if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                 logging.warning(f"  NaN/Inf value detected in parameter: {name}")


        # Strategy: Skip the update for this batch
        logging.warning("Skipping optimizer step due to NaN/Inf loss.")
        optimizer.zero_grad() # Zero gradients to prevent propagation
        # Return a zero tensor for the loss to avoid errors downstreams if loss is used after backward
        loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=False)

    return skip_step, loss


def train_step_with_nan_handling(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    loss_weights: Dict[str, float],
    gradient_clip_val: Optional[float] = 1.0 # Optional gradient clipping
) -> Optional[Dict[str, float]]:
    """
    Training step with combined loss calculation and NaN detection/handling.

    Args:
        model: RNA folding model
        batch: Input data batch moved to the correct device
        optimizer: PyTorch optimizer
        loss_weights: Loss component weights
        gradient_clip_val: Value for gradient clipping (None to disable)

    Returns:
        Dictionary of loss values (floats) or None if step was skipped due to NaN/Inf.
    """
    model.train() # Ensure model is in training mode
    optimizer.zero_grad()

    # Forward pass
    outputs = model(batch)

    # Compute combined loss (assuming a function like compute_combined_loss exists)
    # This function should return the total loss tensor and a dict of component tensors
    try:
        total_loss, loss_components_tensors = compute_combined_loss(outputs, batch, loss_weights)
         # Example: loss_components_tensors = {'fape': fape_t, 'conf': conf_t, 'angle': angle_t}
    except Exception as e:
        logging.error(f"Error during loss computation: {e}")
        # Decide how to handle: skip step or raise error
        return None # Skip step

    # Check for NaN/Inf *before* backward pass
    skip_step, safe_loss = detect_and_handle_nan_loss(
        total_loss, loss_components_tensors, model, optimizer, batch
    )

    if skip_step:
        return None # Indicate step was skipped

    # Backward pass
    try:
        safe_loss.backward()
    except RuntimeError as e:
        logging.error(f"RuntimeError during backward pass: {e}")
        # Potentially due to operations not supporting grads or other issues
        optimizer.zero_grad() # Ensure grads are zeroed if backward failed
        return None # Skip step


    # Optional: Gradient Clipping (after backward, before step)
    if gradient_clip_val is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val)

    # Analyze gradients (Optional: good for debugging)
    # analyze_gradients(model) # Assuming analyze_gradients function exists

    # Update weights
    optimizer.step()

    # Return loss values as floats for logging
    loss_values_float = {k: v.item() for k, v in loss_components_tensors.items()}
    loss_values_float['total'] = safe_loss.item() # Log the potentially modified total loss

    return loss_values_float


def analyze_gradients(model: nn.Module, norm_threshold: float = 10.0) -> Dict[str, Dict[str, float]]:
    """
    Analyze gradient statistics per parameter to diagnose potential issues.

    Args:
        model: The model after loss.backward() has been called.
        norm_threshold: Threshold to flag potentially exploding gradients.

    Returns:
        Dictionary with gradient statistics per parameter.
    """
    gradient_stats = {}
    total_norm = 0.0

    for name, param in model.named_parameters():
        if param.grad is None:
            # logging.debug(f"No gradient for parameter: {name}")
            continue

        grad = param.grad.detach()
        if torch.isnan(grad).any() or torch.isinf(grad).any():
             logging.warning(f"NaN/Inf gradient detected in parameter: {name}")
             stats = {'has_nan_inf': True}
        else:
            norm = grad.norm().item()
            total_norm += norm**2
            stats = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'norm': norm,
                'magnitude_mean': torch.abs(grad).mean().item(),
                'zero_fraction': (grad == 0).float().mean().item(),
                'is_exploding': norm > norm_threshold,
                'is_vanishing': norm < 1e-7 and param.requires_grad # Check vanishing only if grad expected
            }

        gradient_stats[name] = stats

    total_norm = total_norm**0.5
    gradient_stats['overall'] = {'total_norm': total_norm}
    # log_gradient_issues(gradient_stats) # Optionally log issues directly

    return gradient_stats


def log_gradient_issues(gradient_stats: Dict[str, Dict[str, float]]) -> bool:
    """
    Log gradient issues based on analysis and return whether any issues were detected.

    Args:
        gradient_stats: Output from analyze_gradients.

    Returns:
        True if issues (NaN/Inf, exploding, vanishing) were detected, False otherwise.
    """
    has_issues = False
    vanishing_params = []
    exploding_params = []
    nan_inf_params = []

    for name, stats in gradient_stats.items():
        if name == 'overall': continue # Skip overall stats

        if stats.get('has_nan_inf', False):
            nan_inf_params.append(name)
            has_issues = True
        elif stats.get('is_exploding', False):
            exploding_params.append(f"{name} (norm: {stats['norm']:.2e})")
            has_issues = True
        elif stats.get('is_vanishing', False):
             vanishing_params.append(f"{name} (norm: {stats['norm']:.2e})")
             # Don't necessarily set has_issues=True for vanishing, it's often expected for some params
             # Log it as a warning instead.

    if nan_inf_params:
        logging.warning(f"NaN/Inf gradients detected in: {nan_inf_params}")
    if exploding_params:
        logging.warning(f"Exploding gradients detected in: {exploding_params}")
        logging.warning("Consider gradient clipping or smaller learning rate.")
    if vanishing_params:
         logging.warning(f"Vanishing gradients detected in: {vanishing_params}")
         # logging.warning("Consider checking network architecture or normalization layers.")


    return has_issues # Only returns True for critical issues (NaN/Inf, Exploding)


def adaptive_gradient_handling(
    model: nn.Module,
    optimizer: torch.optim.Optimizer, # Not used here, but might be in more complex scenarios
    clip_norm: Optional[float] = 1.0
) -> None:
    """
    Analyze gradients (after backward) and apply gradient clipping if needed.

    Args:
        model: The model after loss.backward().
        optimizer: PyTorch optimizer.
        clip_norm: Gradient clipping norm. If None, clipping is skipped.
    """
    # Analyze gradients first to see if clipping is warranted
    grad_stats = analyze_gradients(model)
    issues_detected = log_gradient_issues(grad_stats) # Logs warnings based on analysis

    # Apply gradient clipping regardless of analysis (common practice)
    # Or apply conditionally: if issues_detected:
    if clip_norm is not None:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        # Log the norm after clipping if desired
        # logging.debug(f"Gradient norm after clipping: {total_norm:.4f}")


class LossTracker:
    """Track and visualize multiple loss components during training."""

    def __init__(self, component_names: List[str], window_size: int = 100):
        """
        Initialize loss tracker.

        Args:
            component_names: Names of loss components to track (e.g., ['fape', 'confidence', 'angle', 'total'])
            window_size: Window size for moving average calculation.
        """
        self.components = component_names
        self.window_size = window_size
        # Store history as lists
        self.history = {name: [] for name in component_names}
        # Store running averages
        self.running_avg = {name: 0.0 for name in component_names}
        # Store weights if provided
        self.weights_history = [] # List of tuples (step, weights_dict)
        self.step = 0

    def update(self, loss_values: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> None:
        """
        Update loss tracker with new values from a training step.

        Args:
            loss_values: Dictionary of loss values (floats) for the current step.
                         Should contain keys matching component_names.
            weights: Optional dictionary of loss weights used for this step.
        """
        self.step += 1

        # Update loss history and running average
        for name in self.components:
            if name in loss_values:
                value = loss_values[name]
                self.history[name].append(value)

                # Keep history within window size
                if len(self.history[name]) > self.window_size:
                    self.history[name].pop(0)

                # Update running average
                # Use numpy for potentially better numerical stability with many floats
                self.running_avg[name] = np.mean(self.history[name]) if self.history[name] else 0.0
            # else: Handle missing loss component? Log warning?

        # Store weights if provided
        if weights:
            self.weights_history.append((self.step, weights.copy())) # Store a copy

    def get_relative_contributions(self) -> Dict[str, float]:
        """
        Calculate relative contribution of each component to the *weighted* total loss,
        based on the running average loss and the *latest* recorded weights.

        Returns:
            Dictionary mapping component names to relative contributions (summing to ~1.0).
        """
        if not self.weights_history:
            logging.warning("No weights recorded, cannot calculate relative contributions.")
            # Fallback: assume equal weight or return raw averages?
            # Returning ratio based on raw running averages:
            total_avg = sum(self.running_avg.get(name, 0.0) for name in self.components if name != 'total')
            if total_avg > 1e-8:
                 return {name: self.running_avg.get(name, 0.0) / total_avg
                         for name in self.components if name != 'total'}
            else:
                 return {name: 0.0 for name in self.components if name != 'total'}


        latest_step, latest_weights = self.weights_history[-1]

        # Calculate weighted contributions using running averages
        weighted_values = {}
        total_weighted_avg = 0.0

        for name in self.components:
            # Exclude 'total' loss itself from contribution calculation
            if name != 'total' and name in latest_weights and name in self.running_avg:
                weight = latest_weights[name]
                avg_loss = self.running_avg[name]
                contribution = weight * avg_loss
                weighted_values[name] = contribution
                total_weighted_avg += contribution

        # Calculate relative contributions
        if total_weighted_avg > 1e-8:
            relative_contributions = {name: val / total_weighted_avg
                                     for name, val in weighted_values.items()}
        else:
            # Avoid division by zero if all weighted contributions are tiny
            num_components = len(weighted_values)
            relative_contributions = {name: 1.0 / num_components if num_components > 0 else 0.0
                                     for name in weighted_values}

        return relative_contributions

    def suggest_weight_adjustments(self, target_contribution: float = -1.0) -> Dict[str, float]:
        """
        Suggest weight adjustments to balance loss components towards equal contribution.
        Uses a simple heuristic based on current contributions and weights.

        Args:
            target_contribution: Target relative contribution per component.
                                 Defaults to 1 / num_components.

        Returns:
            Dictionary of suggested *new* weights (not adjustments factors).
        """
        relative_contributions = self.get_relative_contributions()
        if not relative_contributions or not self.weights_history:
            logging.warning("Cannot suggest adjustments without contributions or weights.")
            return {}

        latest_step, current_weights = self.weights_history[-1]
        num_components = len(relative_contributions)

        if target_contribution < 0:
            target = 1.0 / max(1, num_components) # Target equal contribution
        else:
            target = target_contribution


        suggested_weights = {}
        for name, current_contrib in relative_contributions.items():
            current_weight = current_weights.get(name, 0.0)

            if current_contrib > 1e-8 and current_weight > 1e-8:
                # Adjustment factor: (target / current_contrib)
                # Apply softly, e.g., using sqrt or a dampening factor
                # Simple scaling: new_weight = current_weight * (target / current_contrib)
                adjustment_factor = (target / current_contrib)

                # Dampen adjustment to prevent wild swings (e.g., limit factor)
                adjustment_factor = np.clip(adjustment_factor, 0.5, 2.0) # Limit change to 2x up/down

                suggested_weights[name] = current_weight * adjustment_factor
            else:
                # Keep weight same if contribution or weight is near zero
                suggested_weights[name] = current_weight

        # Optional: Re-normalize suggested weights (e.g., to keep sum constant)
        # total_suggested_weight = sum(suggested_weights.values())
        # if total_suggested_weight > 1e-8:
        #     normalization = sum(current_weights.values()) / total_suggested_weight
        #     suggested_weights = {k: v * normalization for k, v in suggested_weights.items()}

        return suggested_weights


def adaptive_loss_balancing(
    epoch: int,
    tracker: LossTracker,
    current_weights: Dict[str, float],
    adjust_frequency: int = 10, # Adjust every 10 epochs
    learning_rate: float = 0.1 # How much to move towards suggested weights
) -> Dict[str, float]:
    """
    Adaptively balance loss weights based on tracking history using simple heuristics.

    Args:
        epoch: Current training epoch.
        tracker: LossTracker instance containing history.
        current_weights: The currently used loss weights.
        adjust_frequency: How often (in epochs) to adjust weights.
        learning_rate: Controls how quickly weights adapt (0=no change, 1=jump to suggestion).

    Returns:
        Updated weights dictionary for the next epochs.
    """
    # Only adjust periodically and not on the first epoch
    if epoch == 0 or epoch % adjust_frequency != 0:
        return current_weights

    # Get suggested *new* weights
    suggested_weights = tracker.suggest_weight_adjustments()

    if not suggested_weights:
        return current_weights # No suggestion available

    # Apply adjustments gradually towards the suggestion
    new_weights = {}
    updated = False
    for name, current_weight in current_weights.items():
        suggested = suggested_weights.get(name)
        if suggested is not None:
            # Move towards suggested weight: new = current + lr * (suggested - current)
            new_weight = current_weight + learning_rate * (suggested - current_weight)
            # Ensure weights remain non-negative
            new_weights[name] = max(0.0, new_weight)
            if abs(new_weights[name] - current_weight) > 1e-6:
                 updated = True
        else:
            # Keep weight if no suggestion was made for it
            new_weights[name] = current_weight

    if updated:
        logging.info(f"Epoch {epoch}: Adjusting loss weights -> { {k: f'{v:.3f}' for k,v in new_weights.items()} }")
        return new_weights
    else:
        # Return the original dict if no changes were made
        return current_weights


def plot_loss_components(tracker: LossTracker, output_path: Optional[str] = None) -> None:
    """
    Plot loss component history (running average) and relative contributions.

    Args:
        tracker: LossTracker instance with recorded data.
        output_path: Path to save plot. If None, defaults to 'loss_components.png'.
    """
    if plt is None:
        logging.warning("Matplotlib not available for loss visualization.")
        return

    if output_path is None:
        output_path = 'loss_components.png'

    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        # Plot 1: Running average loss history
        steps = list(range(1, tracker.step + 1))
        for name in tracker.components:
             # Get running average history (need to recompute from history list)
             avg_history = []
             current_history = tracker.history[name]
             for k in range(1, len(current_history) + 1):
                 window = current_history[max(0, k - tracker.window_size):k]
                 avg_history.append(np.mean(window) if window else 0.0)

             # Ensure history length matches steps length
             if len(avg_history) == len(steps):
                  ax1.plot(steps, avg_history, label=f'{name} (avg {tracker.window_size})')
             elif len(avg_history) > 0:
                 logging.warning(f"Length mismatch for {name}: steps={len(steps)}, history={len(avg_history)}")


        ax1.set_title(f'Running Average Loss ({tracker.window_size}-step window)')
        # ax1.set_xlabel('Training Step') # X-axis label can be shared if desired
        ax1.set_ylabel('Loss Value')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log') # Log scale often helpful for losses

        # Plot 2: Relative contributions (bar chart of latest values)
        contributions = tracker.get_relative_contributions()
        comp_names = list(contributions.keys())
        comp_values = list(contributions.values())

        ax2.bar(comp_names, comp_values)
        ax2.set_title('Latest Relative Weighted Loss Contributions')
        ax2.set_xlabel('Loss Component')
        ax2.set_ylabel('Relative Contribution')
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.set_ylim(0, 1.05) # Set y-limit for relative scale


        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.suptitle('Loss Component Analysis', fontsize=16)
        plt.savefig(output_path)
        plt.close(fig) # Close the figure to free memory

        logging.info(f"Loss visualization saved to {output_path}")

    except Exception as e:
        logging.error(f"Error generating loss plot: {e}")

```

### 3.2 Implementation Pitfalls

Addressing memory usage and numerical stability.

```python
def compute_memory_efficient_fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_value: float = 10.0,
    chunk_size: int = 16 # Smaller chunk size for Kabsch
) -> torch.Tensor:
    """
    Memory-efficient FAPE loss (simplified proxy) using batched Kabsch alignment.

    Processes sequences individually and potentially chunks large alignments if needed,
    though simple FAPE usually doesn't benefit much from chunking *within* a sequence.
    Main saving is processing batch elements one by one.

    Args:
        pred_coords: Predicted coordinates (batch_size, seq_len, 3)
        true_coords: True coordinates (batch_size, seq_len, 3)
        mask: Boolean mask (batch_size, seq_len)
        clamp_value: Maximum distance error
        chunk_size: (Less relevant here) Max points for Kabsch if needed, usually not limiting.

    Returns:
        FAPE loss scalar
    """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device

    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Initialize loss accumulator
    total_loss = 0.0
    total_valid_sequences = 0

    # Process each batch element separately to reduce peak memory
    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()

        if valid_count < 3:
            continue # Skip if not enough points for alignment

        # Extract valid coordinates for this sequence only
        p_valid = pred_coords[b, valid_mask] # (num_valid, 3)
        t_valid = true_coords[b, valid_mask] # (num_valid, 3)

        # Perform Kabsch alignment (on CPU if necessary for very large N?)
        # Usually GPU Kabsch is fine unless N > ~10000
        try:
            p_aligned = kabsch_align(p_valid, t_valid) # Keep on original device

            # Calculate clamped distances
            dists = torch.norm(p_aligned - t_valid, dim=1)
            clamped_dists = torch.clamp(dists, max=clamp_value)

            # Compute mean loss for this sequence
            if clamped_dists.numel() > 0:
                sequence_loss = clamped_dists.mean()
                total_loss += sequence_loss
                total_valid_sequences += 1
        except RuntimeError as e:
             logging.warning(f"Kabsch failed for batch item {b}: {e}. Skipping.")
             continue


    # Return average loss over valid sequences
    if total_valid_sequences > 0:
        return total_loss / total_valid_sequences
    else:
        return torch.tensor(0.0, device=device)


# Note: Memory-efficient versions for confidence and angle loss primarily involve
# processing sequences/chunks serially if intermediate tensors become too large.
# The provided basic implementations are generally memory-efficient already unless
# sequence lengths are extremely large (>>10k residues).
# Full FAPE loss (compute_full_fape_loss) is memory intensive due to N x N matrices.
# Chunking strategies might be needed there for very long sequences.

def stable_kabsch_align(P: torch.Tensor, Q: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable Kabsch alignment with SVD fallback and epsilon checks.

    Args:
        P: Moving points (N, 3)
        Q: Fixed points (N, 3)
        epsilon: Small constant for numerical stability

    Returns:
        P_aligned: Aligned points (N, 3)
    """
    if P.shape[0] < 1: # Handle empty input
        return P

    # Center the points
    p_mean = P.mean(dim=0, keepdim=True)
    q_mean = Q.mean(dim=0, keepdim=True)
    P_centered = P - p_mean
    Q_centered = Q - q_mean

    # Check for degenerate cases (all points coincident)
    p_norm = torch.norm(P_centered)
    q_norm = torch.norm(Q_centered)

    if p_norm < epsilon or q_norm < epsilon:
        # All points are basically coincident, only apply translation
        logging.debug("Degenerate input to Kabsch (points coincident). Applying translation only.")
        return P - p_mean + q_mean # Align centers

    # Compute covariance matrix
    C = torch.matmul(P_centered.transpose(-2, -1), Q_centered)

    try:
        # Compute optimal rotation using SVD
        U, S, Vt = torch.linalg.svd(C) # S contains singular values
        V = Vt.transpose(-2, -1)

        # Check for near-zero singular values which might cause instability
        if torch.any(S < epsilon):
             logging.debug("Near-zero singular values in Kabsch SVD.")

        # Ensure proper rotation (handle reflection case)
        det = torch.det(torch.matmul(V, U.transpose(-2, -1)))
        if det < (0.0 - epsilon): # Allow for slight numerical imprecision around -1
            # Reflection detected, correct it
            V_corrected = V.clone()
            V_corrected[:, -1] = -V_corrected[:, -1] # Flip the column corresponding to smallest singular value
            R = torch.matmul(V_corrected, U.transpose(-2, -1))
            # Verify correction
            # corrected_det = torch.det(R)
            # logging.debug(f"Corrected reflection in Kabsch: det {det:.3f} -> {corrected_det:.3f}")
        else:
            R = torch.matmul(V, U.transpose(-2, -1))


        # Apply rotation and translation
        P_aligned = torch.matmul(P_centered, R) + q_mean

    except RuntimeError as e:
        # SVD failed (less common with PyTorch >= 1.8)
        logging.warning(f"SVD failed in stable Kabsch: {e}. Returning translation-aligned points.")
        P_aligned = P - p_mean + q_mean # Fallback to center alignment

    # Final check for NaNs in output (shouldn't happen with checks, but be safe)
    if torch.isnan(P_aligned).any():
         logging.error("NaN detected in Kabsch output despite checks! Returning input P.")
         return P # Safest fallback

    return P_aligned


def robust_distance_calculation(
    pred: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Numerically stable distance calculation using squared differences first.

    Args:
        pred: Predicted coordinates (*, 3)
        target: Target coordinates (*, 3)
        epsilon: Small constant added before sqrt for stability

    Returns:
        Distances tensor with shape (*)
    """
    # Calculate squared differences element-wise
    squared_diff = (pred - target)**2 # (*, 3)

    # Sum squared differences across the coordinate dimension (dim=-1)
    sum_sq_diff = torch.sum(squared_diff, dim=-1) # (*)

    # Add epsilon before taking the square root to avoid sqrt(0) or sqrt(negative) due to precision
    # Clamp sum_sq_diff to be non-negative just in case
    stable_sum_sq_diff = torch.clamp(sum_sq_diff, min=0.0) + epsilon

    distances = torch.sqrt(stable_sum_sq_diff)

    return distances


def compute_stable_fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_value: float = 10.0,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Numerically stable FAPE loss (simplified proxy) implementation using stable Kabsch
    and robust distance calculation.

    Args:
        pred_coords: Predicted coordinates (batch_size, seq_len, 3)
        true_coords: True coordinates (batch_size, seq_len, 3)
        mask: Boolean mask (batch_size, seq_len)
        clamp_value: Maximum distance error
        epsilon: Small constant for numerical stability

    Returns:
        FAPE loss scalar
    """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device

    # Default mask
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # --- Input Sanitization ---
    # Check for NaNs/Infs in inputs and log warnings
    if torch.isnan(pred_coords).any() or torch.isinf(pred_coords).any():
        logging.warning("NaN/Inf detected in predicted coordinates. Replacing with 0.")
        pred_coords = torch.nan_to_num(pred_coords, nan=0.0, posinf=0.0, neginf=0.0)
    if torch.isnan(true_coords).any() or torch.isinf(true_coords).any():
        logging.warning("NaN/Inf detected in true coordinates. Replacing with 0.")
        true_coords = torch.nan_to_num(true_coords, nan=0.0, posinf=0.0, neginf=0.0)

    # Initialize loss accumulator
    total_loss = 0.0
    total_valid_sequences = 0

    # Process each sequence separately
    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()

        if valid_count < 3: # Need at least 3 for stable Kabsch
            continue

        # Extract valid coordinates
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]

        try:
            # Apply stable Kabsch alignment
            p_aligned = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)

            # Compute robust distances
            distances = robust_distance_calculation(p_aligned, t_valid, epsilon=epsilon)

            # Apply stable clamping (torch.minimum is often more stable than clamp)
            clamped_distances = torch.minimum(
                distances,
                torch.tensor(clamp_value, device=device, dtype=distances.dtype)
            )

            # Compute mean loss for this sequence
            if clamped_distances.numel() > 0:
                sequence_loss = clamped_distances.mean()

                # Final check for NaN/Inf in the sequence loss itself
                if torch.isnan(sequence_loss) or torch.isinf(sequence_loss):
                    logging.warning(f"NaN/Inf computed for sequence loss (batch {b}), skipping.")
                    continue

                total_loss += sequence_loss
                total_valid_sequences += 1
            else:
                 logging.debug(f"No distances calculated for batch item {b} (likely due to mask).")


        except Exception as e:
            logging.error(f"Error computing stable FAPE for batch item {b}: {e}")
            # Optionally log more details about p_valid, t_valid here if debugging
            continue # Skip this sequence


    # Return average loss over valid sequences
    if total_valid_sequences > 0:
        return total_loss / total_valid_sequences
    else:
        # If no sequences were valid or all failed, return zero loss
        logging.warning("No valid sequences processed for stable FAPE loss.")
        return torch.tensor(0.0, device=device)

```

### 3.3 Performance Optimization

Techniques to speed up loss computations.

```python
def use_scaled_losses(
    loss_dict: Dict[str, torch.Tensor],
    scale_factor: float = 128.0
) -> Dict[str, torch.Tensor]:
    """
    Scale losses before backward pass, typically used with AMP (Automatic Mixed Precision)
    to prevent gradient underflow with float16 gradients.

    Args:
        loss_dict: Dictionary of computed loss component tensors.
        scale_factor: Factor to scale losses by.

    Returns:
        Dictionary of scaled losses.
    """
    # This function itself doesn't optimize computation, but is part of a performance strategy (AMP)
    return {
        name: loss * scale_factor
        for name, loss in loss_dict.items()
    }
    # Remember to unscale gradients before optimizer.step() if using manual AMP scaling,
    # or use torch.cuda.amp.GradScaler which handles this automatically.

def precompute_distance_matrices(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Precompute pairwise distance matrices, useful if multiple losses (like lDDT, full FAPE)
    depend on these.

    Args:
        pred_coords: Predicted coordinates (batch_size, seq_len, 3)
        true_coords: True coordinates (batch_size, seq_len, 3)
        mask: Boolean mask (batch_size, seq_len)

    Returns:
        Tuple of:
        - pred_dists: Predicted distance matrix (batch_size, seq_len, seq_len)
        - true_dists: True distance matrix (batch_size, seq_len, seq_len)
        - pair_mask: Valid pairs mask (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device

    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Compute pairwise distances using cdist (efficient)
    # cdist handles batching internally
    pred_dists = torch.cdist(pred_coords, pred_coords) # (batch_size, seq_len, seq_len)
    true_dists = torch.cdist(true_coords, true_coords) # (batch_size, seq_len, seq_len)

    # Create pairwise mask (pair (i,j) is valid if both mask[i] and mask[j] are True)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) # (batch_size, seq_len, seq_len)

    # Optional: Mask distances where pairs are invalid (set to 0 or NaN)
    # pred_dists = pred_dists * pair_mask.float()
    # true_dists = true_dists * pair_mask.float()

    return pred_dists, true_dists, pair_mask


# Note: compute_lddt_from_dists and compute_coord_errors_from_dists
#       are placeholders. Real implementations would take the precomputed
#       matrices as input to calculate lDDT or RMSD proxies.
#       These are complex and omitted here for brevity but demonstrate the concept
#       of reusing distance matrices.

def compute_lddt_from_dists(
    pred_dists: torch.Tensor,
    true_dists: torch.Tensor,
    pair_mask: torch.Tensor,
    cutoffs: List[float] = [0.5, 1.0, 2.0, 4.0],
    epsilon: float = 1e-8
) -> torch.Tensor:
    """ Placeholder: Compute lDDT scores from distance matrices """
    # Vectorized implementation would go here...
    # Example logic:
    # diff = torch.abs(pred_dists - true_dists) # (B, N, N)
    # scores = torch.zeros_like(diff)
    # for cutoff in cutoffs:
    #      # Consider only pairs where true_dist < 15A (local neighborhood)
    #      local_mask = (true_dists < 15.0) & pair_mask
    #      preserved = (diff < cutoff) & local_mask
    #      scores += preserved.float()
    #
    # num_local_pairs = local_mask.sum(dim=[1,2], keepdim=True).float() # Per batch count
    # lddt = scores.sum(dim=[1,2]) / (num_local_pairs * len(cutoffs) + epsilon) # Batch average lDDT
    # Per-residue lDDT is more complex, requires summing over j for each i.

    # Returning placeholder based on average distance diff
    batch_size, seq_len, _ = pred_dists.shape
    diff = torch.abs(pred_dists - true_dists) * pair_mask.float()
    avg_diff_per_residue = diff.sum(dim=2) / (pair_mask.sum(dim=2).float() + epsilon) # (B, N)
    # Crude lDDT proxy: lower diff = higher score
    lddt_proxy = torch.exp(-avg_diff_per_residue / 5.0) # Scale factor 5 is arbitrary
    return lddt_proxy * mask.float() # Apply original mask


def compute_coord_errors_from_dists(
    pred_dists: torch.Tensor,
    true_dists: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """ Placeholder: Compute per-residue coordinate error proxy from dist matrices """
    # True RMSD requires alignment. This is just a proxy based on distance differences.
    batch_size, seq_len, _ = pred_dists.shape

    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) # (B, N, N)
    diff_sq = (pred_dists - true_dists)**2 * pair_mask.float() # (B, N, N)

    # Mean squared distance difference per residue
    num_valid_pairs = pair_mask.sum(dim=2).float() + epsilon # (B, N)
    mean_sq_diff = diff_sq.sum(dim=2) / num_valid_pairs # (B, N)

    # Take sqrt to get RMS-like error proxy
    error_proxy = torch.sqrt(mean_sq_diff)

    return error_proxy * mask.float() # Apply original mask


def compute_unified_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float],
    precompute_dists: bool = True,
    use_lddt_target: bool = False # Flag to use lDDT for confidence
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute all losses with potential optimization by sharing precomputed distances.

    Args:
        outputs: Model outputs (pred_coords, pred_confidence, pred_angles)
        batch: Input batch (coordinates, dihedral_features, mask)
        loss_weights: Dictionary of loss weights
        precompute_dists: Whether to precompute distance matrices.
        use_lddt_target: If True, use lDDT (from distances) for confidence target.
                         If False, use coord error proxy (from distances or Kabsch).

    Returns:
        Tuple of:
        - total_loss: Combined weighted loss scalar.
        - loss_components: Dictionary of individual loss component tensors.
    """
    # Extract data
    pred_coords = outputs['pred_coords']
    pred_conf = outputs['pred_confidence'] # Logits
    pred_angles = outputs['pred_angles']
    true_coords = batch['coordinates']
    true_angles = batch['dihedral_features']
    mask = batch['mask']
    device = pred_coords.device

    loss_components = {}

    # --- FAPE Loss ---
    # Always compute basic FAPE proxy for this example
    loss_components['fape'] = compute_stable_fape_loss(pred_coords, true_coords, mask)

    # --- Confidence Loss ---
    if use_lddt_target:
        # Option 1: Use lDDT target (requires distance matrices)
        if not precompute_dists:
            # Compute lDDT target directly if distances weren't precomputed
            with torch.no_grad():
                conf_targets = compute_lddt_target(pred_coords, true_coords, mask, per_residue=True)
        else:
             # Compute distance matrices if not already done (or recompute here)
             pred_dists, true_dists, pair_mask = precompute_distance_matrices(
                 pred_coords, true_coords, mask
             )
             with torch.no_grad():
                 conf_targets = compute_lddt_from_dists(pred_dists, true_dists, pair_mask)

        pred_probs = torch.sigmoid(pred_conf)
        squared_error = (pred_probs - conf_targets) ** 2
        masked_se = squared_error * mask.float()
        num_valid = mask.sum()
        loss_components['confidence'] = masked_se.sum() / (num_valid + 1e-8) if num_valid > 0 else torch.tensor(0.0, device=device)

    else:
        # Option 2: Use basic proxy confidence loss (which uses Kabsch internally)
        loss_components['confidence'] = compute_confidence_loss(pred_conf, pred_coords, true_coords, mask)


    # --- Angle Loss ---
    loss_components['angle'] = compute_angle_loss(pred_angles, true_angles, mask)

    # --- Combine Losses ---
    total_loss = torch.tensor(0.0, device=device)
    for name, loss in loss_components.items():
        total_loss += loss_weights.get(name, 1.0) * loss

    loss_components['total'] = total_loss

    # Return tensors for gradient tracking
    return total_loss, {k: v for k, v in loss_components.items() if k != 'total'}

```

### 3.4 FAQ

Guidance on choosing and debugging losses.

```python
def choose_coordinate_loss(
    loss_type: str,
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    # Add args for FAPE/Huber if needed
    clamp_value: float = 10.0,
    huber_delta: float = 1.0,
    pred_frames: Optional[torch.Tensor] = None, # For full FAPE
    true_frames: Optional[torch.Tensor] = None, # For full FAPE
) -> torch.Tensor:
    """
    Choose and compute appropriate coordinate loss based on requirements.

    Args:
        loss_type: Type of loss ('l1', 'l2', 'huber', 'fape_proxy', 'fape_full').
        pred_coords, true_coords: Coordinate tensors.
        mask: Boolean mask.
        clamp_value: Clamp value for FAPE proxy.
        huber_delta: Delta for Huber loss.
        pred_frames, true_frames: Frame tensors required for 'fape_full'.

    Returns:
        Coordinate loss tensor.
    """
    if loss_type == 'l1':
        # L1 loss: Robust to outliers, less sensitive to large errors.
        diff = torch.abs(pred_coords - true_coords)
        if mask is not None:
            loss = (diff * mask.unsqueeze(-1).float()).sum() / (mask.sum() * 3 + 1e-8)
        else:
            loss = diff.mean()
        logging.debug("Using L1 coordinate loss.")
        return loss

    elif loss_type == 'l2':
        # L2 loss (MSE): Sensitive to outliers, penalizes large errors heavily.
        diff_sq = (pred_coords - true_coords) ** 2
        if mask is not None:
            loss = (diff_sq * mask.unsqueeze(-1).float()).sum() / (mask.sum() * 3 + 1e-8)
        else:
            loss = diff_sq.mean()
        logging.debug("Using L2 coordinate loss.")
        return loss

    elif loss_type == 'huber':
        # Huber loss: Mix of L1 and L2, robust but smooth near zero.
        logging.debug(f"Using Huber coordinate loss with delta={huber_delta}.")
        if mask is not None:
             # Apply mask before reduction
             loss = F.huber_loss(pred_coords * mask.unsqueeze(-1).float(),
                                 true_coords * mask.unsqueeze(-1).float(),
                                 reduction='sum', delta=huber_delta)
             loss = loss / (mask.sum() + 1e-8) # Normalize manually
        else:
             loss = F.huber_loss(pred_coords, true_coords, reduction='mean', delta=huber_delta)
        return loss


    elif loss_type == 'fape_proxy':
        # Simplified FAPE: Alignment-based, handles global transforms. Good default for structures.
        logging.debug(f"Using FAPE proxy coordinate loss with clamp={clamp_value}.")
        # Use stable version
        return compute_stable_fape_loss(pred_coords, true_coords, mask, clamp_value=clamp_value)

    elif loss_type == 'fape_full':
        # Full FAPE: Requires frames, measures local frame alignment. More advanced.
        logging.debug(f"Using Full FAPE coordinate loss with clamp={clamp_value}.")
        if pred_frames is None or true_frames is None:
             raise ValueError("Predicted and true frames are required for 'fape_full' loss.")
        return compute_full_fape_loss(pred_coords, pred_frames, true_coords, true_frames, mask, clamp_value=clamp_value)

    else:
        raise ValueError(f"Unknown coordinate loss type: {loss_type}")


def choose_confidence_loss(
    loss_type: str,
    pred_confidence: torch.Tensor, # Assumes logits
    targets: torch.Tensor, # Target values in [0,1]
    mask: Optional[torch.Tensor] = None,
    # Add args for BCE/Focal if needed
    pos_weight: Optional[torch.Tensor] = None, # For BCE
    focal_gamma: float = 2.0, # For Focal
    focal_alpha: Optional[float] = None # For Focal
) -> torch.Tensor:
    """
    Choose and compute appropriate confidence loss based on requirements.

    Args:
        loss_type: Type of loss ('mse', 'bce', 'focal').
        pred_confidence: Predicted confidence logits (batch_size, seq_len).
        targets: Target values in [0,1] (batch_size, seq_len).
        mask: Boolean mask (batch_size, seq_len).
        pos_weight: Weight for positive class in BCE.
        focal_gamma: Focusing parameter gamma for Focal loss.
        focal_alpha: Alpha weighting factor for Focal loss.

    Returns:
        Confidence loss scalar.
    """
    num_valid = mask.sum().item() if mask is not None else pred_confidence.numel()
    if num_valid == 0:
        return torch.tensor(0.0, device=pred_confidence.device)

    epsilon = 1e-8

    if loss_type == 'mse':
        # MSE: Standard regression loss on probabilities.
        logging.debug("Using MSE confidence loss.")
        pred_probs = torch.sigmoid(pred_confidence)
        squared_error = (pred_probs - targets) ** 2
        if mask is not None:
            loss = (squared_error * mask.float()).sum() / (num_valid + epsilon)
        else:
            loss = squared_error.mean()
        return loss

    elif loss_type == 'bce':
        # BCEWithLogitsLoss: Probabilistic interpretation, often better calibrated.
        logging.debug("Using BCE confidence loss.")
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_confidence,
            targets,
            reduction='none',
            pos_weight=pos_weight # Optional weighting for positive class
        )
        if mask is not None:
            loss = (bce_loss * mask.float()).sum() / (num_valid + epsilon)
        else:
            loss = bce_loss.mean()
        return loss

    elif loss_type == 'focal':
        # Focal loss: Focuses on hard examples, good for imbalance.
        logging.debug(f"Using Focal confidence loss with gamma={focal_gamma}, alpha={focal_alpha}.")
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_confidence, targets, reduction='none'
        )
        pred_probs = torch.sigmoid(pred_confidence)
        p_t = pred_probs * targets + (1 - pred_probs) * (1 - targets) # Probability of correct prediction
        focal_modulator = (1.0 - p_t) ** focal_gamma

        if focal_alpha is not None:
            alpha_weight = focal_alpha * targets + (1 - focal_alpha) * (1 - targets)
            focal_loss = alpha_weight * focal_modulator * bce_loss
        else:
            focal_loss = focal_modulator * bce_loss

        if mask is not None:
            loss = (focal_loss * mask.float()).sum() / (num_valid + epsilon)
        else:
            loss = focal_loss.mean()
        return loss

    else:
        raise ValueError(f"Unknown confidence loss type: {loss_type}")

# --- Debugging Functions ---
# (debug_fape_loss, debug_confidence_loss, debug_angle_loss remain largely the same
#  as provided in the input, focusing on step-by-step calculation and checking NaNs/Infs.
#  Ensure they use the stable/robust versions where appropriate for debugging.)

def debug_fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_value: float = 10.0,
    epsilon: float = 1e-8
) -> Dict:
    """ Debug FAPE proxy loss using stable implementations. """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device
    debug_info = {}

    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    debug_info['inputs'] = {
        'pred_isnan': torch.isnan(pred_coords).any().item(),
        'true_isnan': torch.isnan(true_coords).any().item(),
        'pred_norm': torch.norm(pred_coords).item(),
        'true_norm': torch.norm(true_coords).item(),
        'mask_sum': mask.sum().item()
    }

    # Sanitize inputs for debugging calculation
    pred_clean = torch.nan_to_num(pred_coords, nan=0.0)
    true_clean = torch.nan_to_num(true_coords, nan=0.0)

    seq_debug_info = []
    seq_losses = []

    for b in range(batch_size):
        seq_info = {'batch_idx': b}
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()
        seq_info['valid_count'] = valid_count

        if valid_count < 3:
            seq_info['status'] = 'Skipped (insufficient points)'
            seq_losses.append(0.0)
            seq_debug_info.append(seq_info)
            continue

        p_valid = pred_clean[b, valid_mask]
        t_valid = true_clean[b, valid_mask]
        seq_info['coords_extracted'] = True

        try:
            p_aligned = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
            seq_info['kabsch_success'] = True
            seq_info['aligned_isnan'] = torch.isnan(p_aligned).any().item()

            distances = robust_distance_calculation(p_aligned, t_valid, epsilon=epsilon)
            seq_info['dist_success'] = True
            seq_info['dist_isnan'] = torch.isnan(distances).any().item()
            seq_info['dist_max'] = distances.max().item() if distances.numel() > 0 else 0.0

            clamped_distances = torch.minimum(
                distances, torch.tensor(clamp_value, device=device, dtype=distances.dtype)
            )
            seq_info['clamp_success'] = True

            if clamped_distances.numel() > 0:
                seq_loss = clamped_distances.mean()
                seq_info['seq_loss'] = seq_loss.item()
                if torch.isnan(seq_loss) or torch.isinf(seq_loss):
                     seq_info['status'] = 'Error (NaN/Inf loss)'
                     seq_losses.append(0.0)
                else:
                     seq_info['status'] = 'Success'
                     seq_losses.append(seq_loss.item())
            else:
                 seq_info['status'] = 'Skipped (no distances)'
                 seq_losses.append(0.0)

        except Exception as e:
            seq_info['status'] = f'Error ({type(e).__name__})'
            seq_info['error_msg'] = str(e)
            seq_losses.append(0.0)

        seq_debug_info.append(seq_info)

    debug_info['sequences'] = seq_debug_info
    valid_losses = [l for l in seq_losses if l > 0] # Crude check for valid losses
    final_loss = np.mean(valid_losses) if valid_losses else 0.0
    debug_info['calculated_loss'] = final_loss

    return debug_info

# debug_confidence_loss and debug_angle_loss would follow similar patterns,
# breaking down calculations and checking for NaNs/Infs at each step.

```

### 3.5 Visualization Tools

Functions to visualize loss landscapes, calibration, etc.

```python
def visualize_loss_breakdown(
    loss_components: Dict[str, float], # Should contain floats
    loss_weights: Dict[str, float],
    title: str = 'Loss Component Breakdown',
    output_path: Optional[str] = None
) -> None:
    """ Visualize raw losses and weighted contributions. """
    if plt is None:
        logging.warning("Matplotlib not available.")
        return

    if output_path is None:
        output_path = f"{title.lower().replace(' ', '_')}.png"

    raw_losses = {k: v for k, v in loss_components.items() if k != 'total'}
    weighted_losses = {k: v * loss_weights.get(k, 1.0) for k, v in raw_losses.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(raw_losses.keys(), raw_losses.values())
    ax1.set_title('Raw Loss Values')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(weighted_losses.keys(), weighted_losses.values())
    ax2.set_title('Weighted Loss Contributions')
    ax2.set_ylabel('Weighted Loss')
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close(fig)
    logging.info(f"Loss breakdown visualization saved to {output_path}")


def analyze_confidence_prediction_quality(
    pred_confidence: torch.Tensor, # Logits
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
    conf_target_fn: callable = None # Function to calculate target confidence
) -> Optional[Dict[str, float]]:
    """ Analyze correlation and calibration of confidence predictions. """
    if pearsonr is None or spearmanr is None:
         logging.warning("SciPy not found, cannot calculate correlations.")
         return None

    pred_probs = torch.sigmoid(pred_confidence)[mask].detach().cpu().numpy()

    # Calculate actual confidence targets
    if conf_target_fn:
         with torch.no_grad():
             actual_conf = conf_target_fn(pred_coords, true_coords, mask)[mask].detach().cpu().numpy()
    else:
        # Default: use simple distance-based proxy
        with torch.no_grad():
            errors = torch.norm(pred_coords - true_coords, dim=-1)
            targets = torch.exp(-errors / 3.0)
            actual_conf = torch.clamp(targets, 0.0, 1.0)[mask].detach().cpu().numpy()

    if len(pred_probs) == 0:
        logging.warning("No valid samples for confidence analysis.")
        return None

    results = {}
    # Correlation
    results['pearson_r'], results['pearson_p'] = pearsonr(pred_probs, actual_conf)
    results['spearman_r'], results['spearman_p'] = spearmanr(pred_probs, actual_conf)

    # MSE
    results['mse'] = np.mean((pred_probs - actual_conf)**2)

    # Calibration (ECE - Expected Calibration Error)
    num_bins = 10
    bin_limits = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]
    ece = 0.0
    results['bins'] = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (pred_probs >= bin_lower) & (pred_probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(actual_conf[in_bin])
            confidence_in_bin = np.mean(pred_probs[in_bin])
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
            results['bins'].append({
                'lower': bin_lower, 'upper': bin_upper,
                'accuracy': accuracy_in_bin, 'confidence': confidence_in_bin,
                'count': np.sum(in_bin)
            })
        else:
             results['bins'].append({
                 'lower': bin_lower, 'upper': bin_upper,
                 'accuracy': 0, 'confidence': 0, 'count': 0
             })


    results['ece'] = ece
    return results


def visualize_confidence_calibration(
    analysis_results: Dict,
    output_path: str = 'confidence_calibration.png'
) -> None:
    """ Visualize confidence calibration (reliability diagram). """
    if plt is None:
        logging.warning("Matplotlib not available.")
        return
    if not analysis_results or 'bins' not in analysis_results:
         logging.warning("Invalid analysis results for calibration plot.")
         return

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    accuracies = [b['accuracy'] for b in analysis_results['bins']]
    confidences = [b['confidence'] for b in analysis_results['bins']]
    counts = np.array([b['count'] for b in analysis_results['bins']])
    total_count = counts.sum()
    proportions = counts / total_count if total_count > 0 else counts

    # Reliability diagram
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(confidences, accuracies, 'o-', label='Model Calibration')

    # Optional: Bar chart showing gap between confidence and accuracy per bin
    # gap_ax = ax.twinx()
    # gaps = np.abs(np.array(confidences) - np.array(accuracies))
    # gap_ax.bar(confidences, gaps, width=0.1, alpha=0.3, color='red', label='Calibration Gap')
    # gap_ax.set_ylabel('Abs(Accuracy - Confidence)', color='red')

    ax.set_xlabel('Predicted Confidence (Bin Midpoint)')
    ax.set_ylabel('Actual Confidence (Accuracy in Bin)')
    ax.set_title(f"Reliability Diagram (ECE: {analysis_results['ece']:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logging.info(f"Confidence calibration visualization saved to {output_path}")


# visualize_loss_landscape and visualize_training_trajectory remain complex
# examples involving model manipulation, PCA, etc. They are kept as provided
# in the input, assuming the user understands their usage context.

def visualize_loss_landscape(
    model: torch.nn.Module,
    loss_fn: callable, # Should take model output and batch, return scalar loss
    batch: Dict[str, torch.Tensor], # Batch moved to correct device
    param_groups_to_vary: Optional[List[str]] = None, # Names of params/modules
    directions: Optional[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = None, # Precomputed directions
    alpha_range: List[float] = [-1.0, 1.0],
    beta_range: List[float] = [-1.0, 1.0],
    n_points: int = 21, # Odd number includes center
    output_path: str = 'loss_landscape.png'
) -> None:
    """ Visualize the loss landscape along two directions from current model parameters. """
    if plt is None or Axes3D is None:
        logging.warning("Matplotlib (with 3D) not available.")
        return

    model.eval()
    device = next(model.parameters()).device

    # --- 1. Get Original Parameters and Filter ---
    original_params = {}
    params_to_vary = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.clone().detach()
                is_included = True
                if param_groups_to_vary:
                     is_included = any(group in name for group in param_groups_to_vary)
                if is_included:
                    params_to_vary[name] = original_params[name]

    if not params_to_vary:
        logging.warning("No parameters selected for variation.")
        return

    # --- 2. Define Directions ---
    if directions is None:
        # Create random directions, scaled by parameter norm
        dir1, dir2 = {}, {}
        for name, param in params_to_vary.items():
            # Direction 1
            rand_dir1 = torch.randn_like(param)
            param_norm = torch.norm(param) + 1e-10
            dir1_norm = torch.norm(rand_dir1) + 1e-10
            dir1[name] = rand_dir1 * (param_norm / dir1_norm)
            # Direction 2
            rand_dir2 = torch.randn_like(param)
            dir2_norm = torch.norm(rand_dir2) + 1e-10
            dir2[name] = rand_dir2 * (param_norm / dir2_norm)
            # Optional: Orthogonalize dir2 w.r.t dir1 (Gram-Schmidt)
            # dot_prod = torch.sum(dir1[name] * dir2[name])
            # dir2[name] = dir2[name] - (dot_prod / (dir1_norm**2)) * dir1[name]
        directions = (dir1, dir2)
    else:
        # Ensure provided directions only contain keys present in params_to_vary
        dir1 = {k: v for k, v in directions[0].items() if k in params_to_vary}
        dir2 = {k: v for k, v in directions[1].items() if k in params_to_vary}
        directions = (dir1, dir2)


    # --- 3. Evaluate Loss on Grid ---
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
    betas = np.linspace(beta_range[0], beta_range[1], n_points)
    loss_surface = np.zeros((n_points, n_points))

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Temporarily update model parameters
            with torch.no_grad():
                 for name, param in model.named_parameters():
                     if name in params_to_vary:
                         pert = torch.zeros_like(param)
                         if name in directions[0]: pert += alpha * directions[0][name]
                         if name in directions[1]: pert += beta * directions[1][name]
                         param.copy_(original_params[name] + pert)

            # Evaluate loss
            with torch.no_grad():
                try:
                    outputs = model(batch)
                    loss = loss_fn(outputs, batch).item()
                    # Handle potential NaN/Inf from loss function itself
                    if np.isnan(loss) or np.isinf(loss):
                        loss = loss_surface.max() if loss_surface.size > 0 else 1e6 # Assign high value
                except Exception as e:
                     logging.warning(f"Error evaluating loss at alpha={alpha}, beta={beta}: {e}")
                     loss = loss_surface.max() if loss_surface.size > 0 else 1e6 # Assign high value

            loss_surface[i, j] = loss

    # --- 4. Restore Original Parameters ---
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_params:
                param.copy_(original_params[name])

    # --- 5. Plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)

    # Use log scale for Z axis if loss range is large
    z_min, z_max = np.min(loss_surface), np.max(loss_surface)
    if z_max / (z_min + 1e-9) > 100: # Heuristic for large range
         loss_plot = np.log(loss_surface + 1e-9 - z_min) # Shift, log, avoid log(0)
         z_label = 'Log(Loss - MinLoss)'
    else:
         loss_plot = loss_surface
         z_label = 'Loss'


    surf = ax.plot_surface(alpha_grid, beta_grid, loss_plot, cmap='viridis', alpha=0.8)
    # Contour plot on the 'floor'
    # cset = ax.contourf(alpha_grid, beta_grid, loss_plot, zdir='z', offset=np.min(loss_plot)*0.9, cmap='viridis', alpha=0.5)

    # Mark the center point (original parameters)
    center_idx = n_points // 2
    center_loss = loss_plot[center_idx, center_idx]
    ax.scatter([0], [0], [center_loss], color='red', s=100, depthshade=True, label='Original Params')

    ax.set_xlabel('Direction 1 (alpha)')
    ax.set_ylabel('Direction 2 (beta)')
    ax.set_zlabel(z_label)
    title_str = 'Loss Landscape'
    if param_groups_to_vary:
        title_str += f' (varying {", ".join(param_groups_to_vary)})'
    ax.set_title(title_str)
    ax.legend()
    fig.colorbar(surf, shrink=0.5, aspect=10, label=z_label)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logging.info(f"Loss landscape visualization saved to {output_path}")

def visualize_training_trajectory(
    model_checkpoints: List[str], # List of paths to saved model state_dicts
    model_class: type, # The class of the model (e.g., RNAFoldingModel)
    model_config: Dict, # Config dict needed to instantiate the model
    pca_components: int = 2, # Number of PCA components (2 or 3)
    param_groups_to_include: Optional[List[str]] = None, # Filter parameters
    output_path: str = 'training_trajectory.png'
) -> None:
    """ Visualize the training trajectory in parameter space using PCA. """
    if PCA is None or plt is None:
        logging.warning("sklearn or matplotlib not available.")
        return
    if pca_components not in [2, 3]:
         raise ValueError("pca_components must be 2 or 3")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    param_vectors = []

    logging.info(f"Loading {len(model_checkpoints)} checkpoints...")
    for i, ckpt_path in enumerate(model_checkpoints):
        try:
            model = model_class(model_config).to(device)
            checkpoint = torch.load(ckpt_path, map_location=device)
            # Adjust key if necessary (e.g., 'model_state_dict' or just the state_dict)
            state_dict_key = 'model_state_dict' if 'model_state_dict' in checkpoint else None
            if state_dict_key:
                model.load_state_dict(checkpoint[state_dict_key])
            else: # Assume checkpoint is the state_dict itself
                model.load_state_dict(checkpoint)

            model.eval()

            # Extract and flatten relevant parameters
            current_params = []
            with torch.no_grad():
                for name, param in model.named_parameters():
                     is_included = True
                     if param_groups_to_include:
                          is_included = any(group in name for group in param_groups_to_include)
                     if is_included and param.requires_grad:
                          current_params.append(param.detach().view(-1))

            if not current_params:
                 logging.warning(f"No parameters included for checkpoint {i}. Skipping.")
                 continue

            param_vectors.append(torch.cat(current_params).cpu().numpy())
        except Exception as e:
            logging.error(f"Failed to load checkpoint {ckpt_path}: {e}")
            continue

    if len(param_vectors) < 2:
        logging.error("Need at least 2 valid checkpoints to visualize trajectory.")
        return

    logging.info("Performing PCA...")
    param_matrix = np.array(param_vectors)
    pca = PCA(n_components=pca_components)
    reduced_params = pca.fit_transform(param_matrix)
    explained_variance = pca.explained_variance_ratio_

    logging.info(f"PCA explained variance: {explained_variance}")

    # --- Plotting ---
    logging.info("Plotting trajectory...")
    fig = plt.figure(figsize=(10, 8))
    indices = np.arange(len(reduced_params))
    cmap = plt.cm.viridis

    if pca_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(reduced_params[:, 0], reduced_params[:, 1], c=indices, cmap=cmap, s=50)
        ax.plot(reduced_params[:, 0], reduced_params[:, 1], 'k--', alpha=0.5)
        ax.set_xlabel(f'PCA Component 1 ({explained_variance[0]:.2%})')
        ax.set_ylabel(f'PCA Component 2 ({explained_variance[1]:.2%})')
    else: # 3D
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_params[:, 0], reduced_params[:, 1], reduced_params[:, 2], c=indices, cmap=cmap, s=50)
        ax.plot(reduced_params[:, 0], reduced_params[:, 1], reduced_params[:, 2], 'k--', alpha=0.5)
        ax.set_xlabel(f'PCA Component 1 ({explained_variance[0]:.2%})')
        ax.set_ylabel(f'PCA Component 2 ({explained_variance[1]:.2%})')
        ax.set_zlabel(f'PCA Component 3 ({explained_variance[2]:.2%})')

    # Add checkpoint numbers
    for i, point in enumerate(reduced_params):
        if pca_components == 2:
            ax.text(point[0], point[1], str(i), fontsize=9)
        else:
            ax.text(point[0], point[1], point[2], str(i), fontsize=9)

    # Add colorbar
    cbar = fig.colorbar(scatter, label='Checkpoint Index (Epoch Order)')

    title_str = 'Training Trajectory in Parameter Space (PCA)'
    if param_groups_to_include:
        title_str += f'\n(Parameters: {", ".join(param_groups_to_include)})'
    ax.set_title(title_str)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    logging.info(f"Training trajectory visualization saved to {output_path}")

```

## Section 4: Best Practices and Examples

Unit tests, monitoring, and example usage patterns.

### 4.1 Unit Testing Frameworks

Examples using Python's `unittest`.

```python
# --- Unit Tests ---
# Note: These tests assume the loss functions (compute_fape_loss, etc.)
#       are importable or defined in the same scope.

class TestFAPELoss(unittest.TestCase):
    """Unit tests for FAPE loss function (Simplified Proxy)."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.true_coords = torch.randn(self.batch_size, self.seq_len, 3, device=self.device) * 10
        self.pred_coords_equal = self.true_coords.clone()
        self.pred_coords_translated = self.true_coords.clone() + torch.tensor([10.0, -5.0, 2.0], device=self.device)
        self.pred_coords_noisy = self.true_coords.clone() + torch.randn_like(self.true_coords) * 0.5 # Small noise

        self.mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool, device=self.device)
        self.mask[1, -1] = False # Mask last position of second sequence

    def test_identical_coordinates(self):
        loss = compute_stable_fape_loss(self.pred_coords_equal, self.true_coords)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_translated_coordinates(self):
        loss = compute_stable_fape_loss(self.pred_coords_translated, self.true_coords)
        self.assertAlmostEqual(loss.item(), 0.0, places=4, msg="FAPE should be invariant to translation")

    def test_mask_respected(self):
        loss_masked = compute_stable_fape_loss(self.pred_coords_noisy, self.true_coords, self.mask)
        # Compute loss for the first sequence only (fully unmasked)
        loss_seq0 = compute_stable_fape_loss(self.pred_coords_noisy[0:1], self.true_coords[0:1], self.mask[0:1])
        # Compute loss for the second sequence only (partially masked)
        loss_seq1 = compute_stable_fape_loss(self.pred_coords_noisy[1:2], self.true_coords[1:2], self.mask[1:2])
        # Expected average
        expected_loss = (loss_seq0 + loss_seq1) / 2.0
        self.assertAlmostEqual(loss_masked.item(), expected_loss.item(), places=5)

    def test_clamping(self):
        pred_outlier = self.true_coords.clone()
        pred_outlier[0, 0] += 100.0 # Large error
        loss_clamp10 = compute_stable_fape_loss(pred_outlier, self.true_coords, clamp_value=10.0)
        loss_clamp100 = compute_stable_fape_loss(pred_outlier, self.true_coords, clamp_value=100.0)
        self.assertLess(loss_clamp10.item(), loss_clamp100.item(), "Lower clamp should result in lower loss")

    def test_gradient_flow(self):
        pred = self.pred_coords_noisy.clone().requires_grad_(True)
        loss = compute_stable_fape_loss(pred, self.true_coords, self.mask)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertGreater(pred.grad.abs().sum().item(), 0)

    def test_insufficient_points(self):
        mask_few = torch.zeros_like(self.mask)
        mask_few[0, :2] = True # Only 2 points
        loss = compute_stable_fape_loss(self.pred_coords_noisy, self.true_coords, mask_few)
        self.assertEqual(loss.item(), 0.0, "Loss should be 0 if no sequence has enough points for alignment")

    def test_all_masked_batch(self):
        mask_all_false = torch.zeros_like(self.mask)
        loss = compute_stable_fape_loss(self.pred_coords_noisy, self.true_coords, mask_all_false)
        self.assertEqual(loss.item(), 0.0)


class TestConfidenceLoss(unittest.TestCase):
    """Unit tests for confidence loss function (Proxy Target)."""

    def setUp(self):
        self.batch_size = 2
        self.seq_len = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.true_coords = torch.randn(self.batch_size, self.seq_len, 3, device=self.device) * 10
        self.pred_coords_perfect = self.true_coords.clone()
        self.pred_coords_bad = self.true_coords.clone() + 5.0 # Large error (~5A)
        # Confidence logits
        self.pred_conf_high = torch.full((self.batch_size, self.seq_len), 5.0, device=self.device) # ~1 prob
        self.pred_conf_low = torch.full((self.batch_size, self.seq_len), -5.0, device=self.device) # ~0 prob
        self.mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool, device=self.device)

    def test_perfect_pred_high_conf(self):
        # Perfect prediction should have target confidence near 1
        # High predicted confidence should result in low loss
        loss = compute_confidence_loss(self.pred_conf_high, self.pred_coords_perfect, self.true_coords)
        self.assertLess(loss.item(), 0.1, "Perfect pred, high conf -> low loss")

    def test_perfect_pred_low_conf(self):
        # Perfect prediction should have target confidence near 1
        # Low predicted confidence should result in high loss (~(0-1)^2=1)
        loss = compute_confidence_loss(self.pred_conf_low, self.pred_coords_perfect, self.true_coords)
        self.assertGreater(loss.item(), 0.8, "Perfect pred, low conf -> high loss")

    def test_bad_pred_high_conf(self):
        # Bad prediction should have target confidence near 0
        # High predicted confidence should result in high loss (~(1-0)^2=1)
        loss = compute_confidence_loss(self.pred_conf_high, self.pred_coords_bad, self.true_coords)
        self.assertGreater(loss.item(), 0.8, "Bad pred, high conf -> high loss")

    def test_bad_pred_low_conf(self):
        # Bad prediction should have target confidence near 0
        # Low predicted confidence should result in low loss
        loss = compute_confidence_loss(self.pred_conf_low, self.pred_coords_bad, self.true_coords)
        self.assertLess(loss.item(), 0.1, "Bad pred, low conf -> low loss")

    def test_gradient_flow(self):
        pred_conf = (torch.randn(self.batch_size, self.seq_len, device=self.device)
                     .requires_grad_(True))
        pred_coords = self.pred_coords_perfect.clone().requires_grad_(True) # Also check grad flow through coords
        loss = compute_confidence_loss(pred_conf, pred_coords, self.true_coords)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(pred_conf.grad)
        self.assertGreater(pred_conf.grad.abs().sum().item(), 0)
        # Gradient should also flow to coordinates used for target calculation if needed (but not here due to no_grad context)
        # self.assertIsNotNone(pred_coords.grad) # This would fail as target calc is no_grad


class TestAngleLoss(unittest.TestCase):
    """Unit tests for angle loss function."""
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Generate angles theta, eta in radians
        angles_rad = torch.rand(self.batch_size, self.seq_len, 2, device=self.device) * 2 * math.pi
        # Convert to sin/cos pairs [sin(eta), cos(eta), sin(theta), cos(theta)]
        self.true_angles = torch.cat([
            torch.sin(angles_rad[:,:,0:1]), torch.cos(angles_rad[:,:,0:1]),
            torch.sin(angles_rad[:,:,1:2]), torch.cos(angles_rad[:,:,1:2])
        ], dim=2) # Shape (B, N, 4)

        self.pred_angles_perfect = self.true_angles.clone()
        self.pred_angles_opposite = -self.true_angles.clone() # Opposite direction sin/cos

        self.true_angles_with_nan = self.true_angles.clone()
        self.true_angles_with_nan[0, 0, :] = float('nan') # NaN at start
        self.mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool, device=self.device)

    def test_perfect_prediction(self):
        loss_mse = compute_angle_loss(self.pred_angles_perfect, self.true_angles, loss_type='mse')
        loss_mae = compute_angle_loss(self.pred_angles_perfect, self.true_angles, loss_type='mae')
        loss_cos = compute_angle_loss(self.pred_angles_perfect, self.true_angles, loss_type='cosine')
        self.assertAlmostEqual(loss_mse.item(), 0.0, places=6)
        self.assertAlmostEqual(loss_mae.item(), 0.0, places=6)
        self.assertAlmostEqual(loss_cos.item(), 0.0, places=6)

    def test_opposite_prediction_mse_mae(self):
        # MSE should be high, MAE should be average abs diff
        loss_mse = compute_angle_loss(self.pred_angles_opposite, self.true_angles, loss_type='mse')
        loss_mae = compute_angle_loss(self.pred_angles_opposite, self.true_angles, loss_type='mae')
        # Expected MSE = mean((-t - t)^2) = mean(4*t^2) = 4 * mean(t^2)
        # Since mean(sin^2 + cos^2) = 1, mean(t^2) across 4 features = 0.5 -> Expected MSE ~ 2.0
        self.assertGreater(loss_mse.item(), 1.5)
        self.assertLess(loss_mse.item(), 2.5)
        self.assertGreater(loss_mae.item(), 0.5) # Avg |sin|, |cos| > 0.5

    def test_opposite_prediction_cosine(self):
        # Cosine loss (1 - cos_sim) should be near max (2.0) for opposite vectors
        loss_cos = compute_angle_loss(self.pred_angles_opposite, self.true_angles, loss_type='cosine')
        self.assertAlmostEqual(loss_cos.item(), 2.0, places=5)

    def test_nan_handling(self):
        pred = torch.randn_like(self.true_angles_with_nan)
        loss = compute_angle_loss(pred, self.true_angles_with_nan, self.mask)
        self.assertFalse(torch.isnan(loss).item(), "Loss should not be NaN when handling NaNs in target")
        self.assertFalse(torch.isinf(loss).item())

    def test_gradient_flow(self):
        pred_angles = self.true_angles.clone().requires_grad_(True)
        loss = compute_angle_loss(pred_angles, self.true_angles)
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(pred_angles.grad)
        self.assertNotEqual(pred_angles.grad.abs().sum().item(), 0) # Grad might be zero if loss=0

# To run tests:
# if __name__ == '__main__':
#    unittest.main(argv=['first-arg-is-ignored'], exit=False) # In notebook/interactive
#    # Or run from command line: python -m unittest your_module_name.py
```

### 4.2 Monitoring, Logging, and Early Stopping

Classes for tracking training progress.

```python
class EarlyStopping:
    """
    Early stopping utility to stop training when a monitored metric stops improving.
    """
    def __init__(
        self,
        patience: int = 10,
        verbose: bool = True,
        delta: float = 0.0, # Minimum change to qualify as improvement
        mode: str = 'min', # 'min' for loss, 'max' for accuracy/metrics
        checkpoint_path: str = 'checkpoint.pt' # Path to save the best model
    ):
        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")

        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metric = float('inf') if mode == 'min' else -float('inf')

    def __call__(self, metric_value: float, model: nn.Module, epoch: int,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[Any] = None) -> bool:
        """
        Call instance with current metric value. Saves model if improvement, stops if patience exceeded.

        Args:
            metric_value: The metric to monitor (e.g., validation loss).
            model: The model being trained.
            epoch: Current epoch number.
            optimizer: Optimizer state to save (optional).
            scheduler: LR scheduler state to save (optional).

        Returns:
            bool: True if training should stop, False otherwise.
        """
        score = metric_value

        improvement = False
        if self.mode == 'min':
            if score < self.best_metric - self.delta:
                improvement = True
        else: # mode == 'max'
            if score > self.best_metric + self.delta:
                improvement = True

        if improvement:
            if self.verbose:
                logging.info(f"Metric improved ({self.best_metric:.6f} --> {score:.6f}). Saving model...")
            self.best_metric = score
            self.save_checkpoint(metric_value, model, epoch, optimizer, scheduler)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                     logging.info("Early stopping triggered.")

        return self.early_stop

    def save_checkpoint(self, metric_value: float, model: nn.Module, epoch: int,
                       optimizer: Optional[torch.optim.Optimizer],
                       scheduler: Optional[Any]) -> None:
        """Saves model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            f'best_{"loss" if self.mode == "min" else "metric"}': self.best_metric,
        }
        if optimizer:
             checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
             checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        try:
            torch.save(checkpoint, self.checkpoint_path)
        except Exception as e:
            logging.error(f"Error saving checkpoint to {self.checkpoint_path}: {e}")


class LossHistory:
    """ Track and log loss values using TensorBoard and optionally CSV. """
    def __init__(
        self,
        log_dir: str = 'logs/rna_folding',
        use_tensorboard: bool = True,
        save_csv: bool = False
    ):
        self.log_dir = log_dir
        self.save_csv_flag = save_csv
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = None
        if use_tensorboard and SummaryWriter:
            try:
                self.writer = SummaryWriter(log_dir=self.log_dir)
            except Exception as e:
                 logging.warning(f"Could not initialize TensorBoard SummaryWriter: {e}")
        elif use_tensorboard:
             logging.warning("TensorBoard requested but not available.")

        self.history = {} # Stores lists of losses like {'train_total': [], 'val_total': []}
        self.epochs = []

    def update(self, epoch: int, losses: Dict[str, float]) -> None:
        """ Update history with losses from one epoch. losses dict keys should indicate phase e.g., 'train_fape', 'val_total'. """
        self.epochs.append(epoch)
        for name, value in losses.items():
             if name not in self.history:
                 self.history[name] = []
             # Pad previous epochs with NaN if this metric wasn't logged before
             pad_len = len(self.epochs) - 1 - len(self.history[name])
             if pad_len > 0:
                  self.history[name].extend([float('nan')] * pad_len)
             self.history[name].append(value)

             # Log to TensorBoard
             if self.writer:
                 try:
                      # Format tag for TensorBoard (e.g., Loss/train/fape)
                      parts = name.split('_')
                      tag = f"Loss/{parts[0]}/{'_'.join(parts[1:])}" if len(parts) > 1 else f"Loss/{name}"
                      self.writer.add_scalar(tag, value, epoch)
                 except Exception as e:
                      logging.warning(f"TensorBoard logging failed for {name}: {e}")

    def save_csv(self, csv_path: Optional[str] = None) -> None:
        """ Save loss history to a CSV file. """
        if not self.save_csv_flag or pd is None:
            if self.save_csv_flag: logging.warning("Pandas not available for CSV saving.")
            return

        if csv_path is None:
             csv_path = os.path.join(self.log_dir, 'loss_history.csv')

        # Ensure all history lists have the same length (pad with NaN)
        max_len = len(self.epochs)
        data_for_df = {'epoch': self.epochs}
        for name, values in self.history.items():
             padded_values = values + [float('nan')] * (max_len - len(values))
             data_for_df[name] = padded_values

        try:
             df = pd.DataFrame(data_for_df)
             df.to_csv(csv_path, index=False)
             logging.debug(f"Loss history saved to {csv_path}")
        except Exception as e:
             logging.error(f"Error saving loss history to CSV: {e}")

    def plot_losses(self, save_path: Optional[str] = None) -> None:
        """ Plot loss history trends. """
        if plt is None:
            logging.warning("Matplotlib not available for plotting.")
            return

        if not self.history:
            logging.warning("No loss history to plot.")
            return

        if save_path is None:
            save_path = os.path.join(self.log_dir, 'loss_history.png')

        # Determine metrics to plot (e.g., group by train/val)
        train_metrics = sorted([k for k in self.history if k.startswith('train_')])
        val_metrics = sorted([k for k in self.history if k.startswith('val_')])
        other_metrics = sorted([k for k in self.history if not k.startswith('train_') and not k.startswith('val_')])

        num_plots = len(train_metrics) + len(other_metrics)
        if num_plots == 0: return

        # Simple plotting: one plot per metric pair (train/val) or single metric
        fig, axs = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
        if num_plots == 1: axs = [axs] # Make it iterable

        plot_idx = 0
        epochs_array = np.array(self.epochs)

        # Plot train/val pairs
        for train_key in train_metrics:
            base_key = '_'.join(train_key.split('_')[1:])
            val_key = f'val_{base_key}'

            ax = axs[plot_idx]
            train_values = np.array(self.history[train_key])
            ax.plot(epochs_array[:len(train_values)], train_values, label='Train', marker='.')

            if val_key in self.history:
                val_values = np.array(self.history[val_key])
                ax.plot(epochs_array[:len(val_values)], val_values, label='Validation', marker='.')

            ax.set_title(f'{base_key.replace("_", " ").title()} Loss')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            # ax.set_yscale('log') # Optional log scale
            plot_idx += 1

        # Plot other metrics
        for key in other_metrics:
             ax = axs[plot_idx]
             values = np.array(self.history[key])
             ax.plot(epochs_array[:len(values)], values, label=key, marker='.')
             ax.set_title(f'{key.replace("_", " ").title()}')
             ax.set_ylabel('Value')
             ax.legend()
             ax.grid(True, alpha=0.3)
             plot_idx += 1


        axs[-1].set_xlabel('Epoch') # Label only the last x-axis
        plt.suptitle('Training Loss History', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.close(fig)
        logging.info(f"Loss history plot saved to {save_path}")


    def close(self) -> None:
        """ Close the TensorBoard writer. """
        if self.writer:
            try:
                self.writer.close()
            except Exception as e:
                 logging.warning(f"Error closing TensorBoard writer: {e}")

```

### 4.3 Example Usage

Putting components together in training or inference loops.

```python
# Placeholder for the actual RNA folding model class
class RNAFoldingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Dummy layers for example
        self.config = config
        self.linear = nn.Linear(config.get('input_dim', 128), config.get('hidden_dim', 256))
        self.coord_head = nn.Linear(config.get('hidden_dim', 256), 3) # Predicts 3D coords per residue
        self.conf_head = nn.Linear(config.get('hidden_dim', 256), 1) # Predicts confidence logit
        self.angle_head = nn.Linear(config.get('hidden_dim', 256), 4) # Predicts sin/cos for 2 angles

    def forward(self, batch):
        # Dummy forward pass using sequence length from mask
        mask = batch['mask']
        batch_size, seq_len = mask.shape
        device = mask.device
        # Create dummy features based on sequence length
        # In reality, this would use sequence embeddings, pair features etc.
        dummy_features = torch.randn(batch_size, seq_len, self.config.get('input_dim', 128), device=device)
        hidden = F.relu(self.linear(dummy_features))

        # Apply mask in hidden state? Often done before output heads
        hidden = hidden * mask.unsqueeze(-1).float()

        pred_coords = self.coord_head(hidden)
        pred_confidence = self.conf_head(hidden).squeeze(-1) # Remove last dim
        pred_angles = self.angle_head(hidden)

        # Apply mask to outputs - ensure padding regions have zero/default outputs
        pred_coords = pred_coords * mask.unsqueeze(-1).float()
        pred_confidence = pred_confidence * mask.float()
        pred_angles = pred_angles * mask.unsqueeze(-1).float()


        return {
            'pred_coords': pred_coords,
            'pred_confidence': pred_confidence,
            'pred_angles': pred_angles,
        }

# --- Example Training Step (using basic combined loss) ---
def train_step(model, batch, optimizer, loss_weights, grad_clip=1.0):
    model.train()
    optimizer.zero_grad()
    outputs = model(batch)
    total_loss, loss_components = compute_combined_loss(outputs, batch, loss_weights)

    # Handle potential NaNs before backward
    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
         logging.warning("NaN/Inf loss detected in train_step. Skipping batch.")
         return None # Skip step

    total_loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    # Return detached loss values for logging
    return {k: v.item() for k, v in loss_components.items()} | {'total': total_loss.item()}

# --- Example Training Loop with Curriculum ---
def train_with_curriculum(model, train_loader, val_loader, num_epochs, config):
    # Define curriculum manager
    initial_weights = config.get('initial_loss_weights', {'fape': 1.0, 'confidence': 0.05, 'angle': 0.1})
    final_weights = config.get('final_loss_weights', {'fape': 1.0, 'confidence': 0.2, 'angle': 0.5})
    curriculum_type = config.get('curriculum_type', 'linear')

    curriculum = CurriculumLossManager(
        initial_weights=initial_weights,
        final_weights=final_weights,
        total_epochs=num_epochs,
        curriculum_type=curriculum_type
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
    loss_history = LossHistory(log_dir=config.get('log_dir', 'logs/curriculum_run'))
    early_stopping = EarlyStopping(patience=config.get('patience', 15),
                                  checkpoint_path=os.path.join(config.get('checkpoint_dir', 'checkpoints'), 'best_curriculum.pt'))
    device = next(model.parameters()).device

    logging.info("Starting training with curriculum...")
    for epoch in range(num_epochs):
        current_weights = curriculum.get_weights(epoch)
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss Weights: { {k:f'{v:.3f}' for k,v in current_weights.items()} }")

        # Training Phase
        model.train()
        epoch_train_losses = {'fape': 0.0, 'confidence': 0.0, 'angle': 0.0, 'total': 0.0}
        num_train_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # Use the basic train_step defined above
            loss_dict = train_step(model, batch, optimizer, current_weights, grad_clip=config.get('grad_clip', 1.0))
            if loss_dict: # If step wasn't skipped
                 for k, v in loss_dict.items(): epoch_train_losses[k] += v
                 num_train_batches += 1

        # Average training losses for the epoch
        avg_train_losses = {f'train_{k}': v / max(1, num_train_batches) for k, v in epoch_train_losses.items()}
        logging.info(f"  Avg Train Losses: { {k: f'{v:.4f}' for k,v in avg_train_losses.items()} }")


        # Validation Phase
        model.eval()
        epoch_val_losses = {'fape': 0.0, 'confidence': 0.0, 'angle': 0.0, 'total': 0.0}
        num_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(batch)
                # Use compute_combined_loss directly for validation
                total_loss, loss_components = compute_combined_loss(outputs, batch, current_weights)
                if not (torch.isnan(total_loss).any() or torch.isinf(total_loss).any()):
                     epoch_val_losses['total'] += total_loss.item()
                     for k, v in loss_components.items(): epoch_val_losses[k] += v.item()
                     num_val_batches += 1

        # Average validation losses
        avg_val_losses = {f'val_{k}': v / max(1, num_val_batches) for k, v in epoch_val_losses.items()}
        logging.info(f"  Avg Val Losses: { {k: f'{v:.4f}' for k,v in avg_val_losses.items()} }")

        # Update history and check early stopping
        all_epoch_losses = {**avg_train_losses, **avg_val_losses}
        loss_history.update(epoch, all_epoch_losses)
        val_metric = avg_val_losses.get('val_total', float('inf')) # Monitor total validation loss
        if early_stopping(val_metric, model, epoch, optimizer):
            break

        # Save history periodically
        if epoch % config.get('save_history_freq', 10) == 0:
             loss_history.save_csv()


    loss_history.close()
    logging.info("Training finished.")
    # Load best model
    # best_checkpoint = torch.load(early_stopping.checkpoint_path)
    # model.load_state_dict(best_checkpoint['model_state_dict'])
    return model


# --- Example Inference with Multiple Predictions (using StructureSampler) ---
def generate_submission(model, test_loader, config, output_path='submission.csv'):
    sampler_config = config.get('sampler', {})
    sampler = StructureSampler(
        num_samples=sampler_config.get('num_samples', 5),
        sampling_strategy=sampler_config.get('strategy', 'noise_injection'),
        noise_scale=sampler_config.get('noise_scale', 0.1),
        use_confidence_for_noise=sampler_config.get('use_confidence', False)
    )

    device = next(model.parameters()).device
    model.eval() # Sampler might override this temporarily (e.g., for dropout)

    all_pred_coords_lists = [] # List to hold coords from all batches
    all_target_ids = []
    all_sequences = []
    all_masks = [] # Collect masks if needed for formatting

    logging.info("Generating samples for submission...")
    with torch.no_grad(): # Most sampling happens within no_grad context
        for batch in test_loader:
            # Generate samples (list of B,N,3 tensors)
            # Note: batch needs required inputs for the model
            # For noise injection, ensure 'initial_coords' or similar is present if needed
            pred_coords_list, _ = sampler.generate_samples(model, batch, device) # Ignore confidence list for now

            if not pred_coords_list: continue # Skip if sampling failed

            # Need to collect batch data corresponding to predictions
            # Detach and move results for this batch to CPU list
            # Important: Ensure the list structure matches format_kaggle_submission
            # format_kaggle_submission expects a list of tensors, where each tensor is (B, N, 3) for ONE sample/model
            # Our sampler currently returns [sample1_coords(B,N,3), sample2_coords(B,N,3), ...]
            # We need to transpose this structure over batches.

            # For simplicity here, let's assume we process batches sequentially and accumulate
            # We need to store the B dimension info.
            all_target_ids.extend(batch['target_ids']) # Assuming batch has this key
            all_sequences.extend(batch['sequences'])   # Assuming batch has this key
            if 'mask' in batch: all_masks.append(batch['mask'].cpu())

            # Store coords: Need list of lists [batch][sample](N,3) -> transpose later
            batch_coords_by_sample = [sample_coords.cpu() for sample_coords in pred_coords_list] # [num_samples](B, N, 3)
            all_pred_coords_lists.append(batch_coords_by_sample)


    logging.info("Consolidating predictions...")
    # Consolidate collected results: Transpose [batch][sample](B,N,3) -> [sample](all_B, N, 3)
    num_samples = sampler.num_samples
    final_coords_list = [[] for _ in range(num_samples)] # [sample] -> list of batch tensors
    final_mask_list = [[] for _ in range(num_samples)] if all_masks else None

    for batch_idx, batch_coords_samples in enumerate(all_pred_coords_lists):
        for sample_idx in range(num_samples):
             final_coords_list[sample_idx].append(batch_coords_samples[sample_idx])
             if final_mask_list:
                 final_mask_list[sample_idx].append(all_masks[batch_idx])


    # Concatenate batch tensors for each sample
    try:
        final_coords_list = [torch.cat(tensors, dim=0) for tensors in final_coords_list] # [sample](total_B, N, 3)
        final_mask_list = [torch.cat(tensors, dim=0) for tensors in final_mask_list] if final_mask_list else None
    except Exception as e:
        logging.error(f"Error concatenating batch results: {e}. Check sequence lengths.")
        return

    logging.info("Formatting submission file...")
    # Format for Kaggle submission
    format_kaggle_submission(
        pred_coords_list=final_coords_list,
        target_ids=all_target_ids,
        sequences=all_sequences,
        mask_list=final_mask_list, # Pass consolidated masks
        output_path=output_path
    )

```

## Conclusion

This document provides a comprehensive set of tools and best practices for implementing, debugging, and optimizing loss functions in the RNA 3D structure prediction pipeline. The code examples address common challenges such as NaN values, exploding gradients, and numerical instability, while also providing utilities for monitoring and visualizing loss behaviors.

By following these patterns, you can ensure that your loss functions are robust, numerically stable, and properly configured for training the RNA 3D folding model. The test frameworks and visualization tools will help diagnose issues early and maintain model quality throughout the development process.

Remember that the core loss components (FAPE proxy, confidence, and angle losses) work together to provide a balanced training objective that captures both structural accuracy and model uncertainty. Properly tuning the relative weights of these components is critical for achieving optimal model performance.

For future improvements, consider implementing more sophisticated loss formulations as described in the Future Extensions section (like full FAPE or lDDT-based losses), which can provide more accurate structural evaluation and potentially better gradient signals for learning.
```
