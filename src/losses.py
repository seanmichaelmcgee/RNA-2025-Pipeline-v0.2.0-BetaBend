# START OF FILE: src/losses.py
"""
Loss Functions for RNA 3D Structure Prediction (V1 Implementation)

This module implements the loss functions used for training the V1 RNA folding model.
It includes a simplified FAPE proxy loss for coordinates, a confidence prediction loss
using a derived target, and an auxiliary angle prediction loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Dict, Tuple, List, Union

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Helper Functions ---

def stable_kabsch_align(P: torch.Tensor, Q: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Numerically stable Kabsch alignment with SVD fallback and epsilon checks.
    Aligns points P to points Q.

    Args:
        P: Moving points, shape (N, 3)
        Q: Fixed points, shape (N, 3)
        epsilon: Small constant for numerical stability

    Returns:
        P_aligned: Aligned points, shape (N, 3)
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
        # Use full_matrices=False for potentially better stability with some versions/devices
        U, S, Vt = torch.linalg.svd(C, full_matrices=False)
        V = Vt.transpose(-2, -1)

        # Check for near-zero singular values which might cause instability
        if torch.any(S < epsilon):
             logging.debug("Near-zero singular values in Kabsch SVD.")

        # Ensure proper rotation (handle reflection case)
        det = torch.det(torch.matmul(V, U.transpose(-2, -1)))
        R = torch.matmul(V, U.transpose(-2, -1))
        if det < (0.0 - epsilon): # Allow for slight numerical imprecision around -1
            # Reflection detected, correct it by flipping the sign along the axis
            # corresponding to the smallest singular value (last column of V).
            V_corrected = V.clone()
            V_corrected[:, -1] = -V_corrected[:, -1]
            R = torch.matmul(V_corrected, U.transpose(-2, -1))
            # Verify correction
            # corrected_det = torch.det(R)
            # logging.debug(f"Corrected reflection in Kabsch: det {det:.3f} -> {corrected_det:.3f}")

        # Apply rotation and translation
        P_aligned = torch.matmul(P_centered, R) + q_mean

    except RuntimeError as e:
        # SVD failed (less common with PyTorch >= 1.8)
        logging.warning(f"SVD failed in stable Kabsch: {e}. Returning translation-aligned points.")
        P_aligned = P - p_mean + q_mean # Fallback to center alignment

    # Final check for NaNs in output
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

# --- Core V1 Loss Functions ---

def compute_stable_fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_value: float = 10.0,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute a simplified FAPE loss proxy (V1) based on clamped L2 distance
    after global Kabsch alignment, using stable implementations.

    Args:
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        clamp_value: Maximum distance error to consider (Å)
        epsilon: Small constant for numerical stability

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device
    dtype = pred_coords.dtype

    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # --- Input Sanitization ---
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
        return torch.tensor(0.0, device=device, dtype=dtype)


def compute_confidence_loss(
    pred_confidence: torch.Tensor,
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = 'mse', # 'mse' or 'bce'
    target_type: str = 'lddt_proxy',
    scaling_factor: float = 3.0, # For 'lddt_proxy'
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute confidence prediction loss (V1 Proxy Target).

    Trains the model to predict a per-residue confidence score (similar to pLDDT)
    that correlates with the actual per-residue accuracy based on a proxy derived
    from coordinate error after global alignment.

    Args:
        pred_confidence: Predicted confidence scores (logits), shape (batch_size, seq_len)
        pred_coords: Predicted coordinates, shape (batch_size, seq_len, 3)
        true_coords: Ground truth coordinates, shape (batch_size, seq_len, 3)
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        loss_type: Loss function to use ('mse' or 'bce').
        target_type: Type of proxy target ('lddt_proxy' or 'distance_based').
        scaling_factor: Scaling factor for 'lddt_proxy' target computation (Å).
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar loss value
    """
    batch_size, seq_len = pred_confidence.shape
    device = pred_confidence.device
    dtype = pred_confidence.dtype

    # Create default mask if not provided (all positions valid)
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # --- Calculate residue-wise error target (proxy for lDDT) ---
    # This calculation should NOT contribute to gradients wrt pred_coords
    with torch.no_grad():
        # First align structures globally, batch by batch
        aligned_pred_coords = torch.zeros_like(pred_coords)
        per_residue_error = torch.zeros((batch_size, seq_len), device=device, dtype=dtype)

        for b in range(batch_size):
            valid_mask = mask[b]
            valid_count = valid_mask.sum().item()

            if valid_count >= 3:  # Need at least 3 points for alignment
                p_valid = pred_coords[b, valid_mask]
                t_valid = true_coords[b, valid_mask]
                aligned_p_valid = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)
                aligned_pred_coords[b, valid_mask] = aligned_p_valid

                # Compute per-residue coordinate error only for valid residues
                errors = robust_distance_calculation(aligned_p_valid, t_valid, epsilon=epsilon)
                per_residue_error[b, valid_mask] = errors
            else:
                # Not enough points for alignment, calculate raw error or assign high error
                # Calculate raw error for any valid points
                if valid_count > 0:
                     raw_errors = robust_distance_calculation(
                         pred_coords[b, valid_mask],
                         true_coords[b, valid_mask],
                         epsilon=epsilon
                     )
                     per_residue_error[b, valid_mask] = raw_errors
                # Implicitly, error for invalid points remains 0, but they will be masked out

        # --- Compute confidence target based on error ---
        if target_type == 'lddt_proxy':
            # Convert error to per-residue lDDT-like score in [0, 1]
            # Higher score means better prediction (lower error)
            conf_targets = torch.exp(-per_residue_error / scaling_factor)
            conf_targets = torch.clamp(conf_targets, 0.0, 1.0) # Ensure [0, 1]

        elif target_type == 'distance_based':
            # Alternative: directly scale distances to [0, 1] range
            # 1 = low error, 0 = high error
            max_dist = 15.0  # Maximum distance to consider (Å)
            conf_targets = 1.0 - torch.clamp(per_residue_error / max_dist, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown target_type for confidence loss: {target_type}")

        # Mask out targets for padded positions (set to 0, loss will be ignored via mask)
        conf_targets = conf_targets * mask.float()

    # --- Compute Loss ---
    if loss_type == 'mse':
        # Apply sigmoid to predicted logits
        pred_probs = torch.sigmoid(pred_confidence)
        # Calculate MSE loss
        squared_error = (pred_probs - conf_targets) ** 2
        masked_loss = squared_error * mask.float()
    elif loss_type == 'bce':
        # Calculate BCE loss
        masked_loss = F.binary_cross_entropy_with_logits(
            pred_confidence, conf_targets, reduction='none'
        ) * mask.float()
    else:
        raise ValueError(f"Unknown loss_type for confidence loss: {loss_type}")

    # Compute mean loss over valid positions
    num_valid = mask.sum().item()
    if num_valid == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    loss = masked_loss.sum() / num_valid

    return loss


def compute_angle_loss(
    pred_angles: torch.Tensor,
    true_angles: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = 'mse', # 'mse', 'cosine', or 'mae'
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Compute loss for dihedral angle predictions (V1).

    Compares the predicted sin/cos representations of angles with the true values.

    Args:
        pred_angles: Predicted sin/cos of angles [sin(η), cos(η), sin(θ), cos(θ)],
                     shape (batch_size, seq_len, 4)
        true_angles: True sin/cos of angles, shape (batch_size, seq_len, 4)
                     May contain NaNs for boundary residues.
        mask: Boolean mask, shape (batch_size, seq_len), True for valid positions
        loss_type: Loss function to use ('mse', 'cosine', or 'mae').
        epsilon: Small constant for numerical stability.

    Returns:
        Scalar loss value
    """
    batch_size, seq_len, num_features = pred_angles.shape
    device = pred_angles.device
    dtype = pred_angles.dtype

    # Create default mask if not provided
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Handle NaNs in true angles (typically at boundaries)
    angle_mask = mask.clone()
    true_is_nan = torch.isnan(true_angles)
    if true_is_nan.any():
        # Create mask for non-NaN angles across all 4 features
        nan_mask = ~true_is_nan.any(dim=2)  # (batch_size, seq_len)
        angle_mask = angle_mask & nan_mask

    # Expand mask to match angle dimensions
    expanded_mask = angle_mask.unsqueeze(-1).expand_as(pred_angles)  # (batch_size, seq_len, 4)

    # Number of valid elements for averaging
    num_valid_elements = expanded_mask.sum().item()
    if num_valid_elements == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Replace NaNs with zeros in true angles (masked out anyway)
    true_angles_clean = torch.nan_to_num(true_angles, nan=0.0)

    # --- Compute Loss ---
    if loss_type == 'mse':
        squared_error = (pred_angles - true_angles_clean) ** 2
        masked_loss_sum = (squared_error * expanded_mask.float()).sum()
        loss = masked_loss_sum / num_valid_elements

    elif loss_type == 'cosine':
        # Cosine similarity loss: 1 - cos(angle_diff)
        # Group sin/cos pairs: [sin(eta), cos(eta)] and [sin(theta), cos(theta)]
        pred_eta = pred_angles[:, :, 0:2]
        pred_theta = pred_angles[:, :, 2:4]
        true_eta = true_angles_clean[:, :, 0:2]
        true_theta = true_angles_clean[:, :, 2:4]

        # Normalize vectors to ensure they are on unit circle (robustness)
        pred_eta_norm = F.normalize(pred_eta, p=2, dim=2, eps=epsilon)
        true_eta_norm = F.normalize(true_eta, p=2, dim=2, eps=epsilon)
        pred_theta_norm = F.normalize(pred_theta, p=2, dim=2, eps=epsilon)
        true_theta_norm = F.normalize(true_theta, p=2, dim=2, eps=epsilon)

        # Calculate cosine similarity (element-wise dot product)
        cos_sim_eta = torch.sum(pred_eta_norm * true_eta_norm, dim=2) # (B, L)
        cos_sim_theta = torch.sum(pred_theta_norm * true_theta_norm, dim=2) # (B, L)

        # Loss = 1 - cos_sim. Average over eta and theta.
        eta_loss_term = (1.0 - cos_sim_eta)
        theta_loss_term = (1.0 - cos_sim_theta)

        # Apply mask
        masked_eta_loss = eta_loss_term * angle_mask.float()
        masked_theta_loss = theta_loss_term * angle_mask.float()

        # Sum losses and divide by number of valid angles (num_valid_residues * 2)
        num_valid_angles = angle_mask.sum().item() * 2
        if num_valid_angles == 0:
             return torch.tensor(0.0, device=device, dtype=dtype)

        loss = (masked_eta_loss.sum() + masked_theta_loss.sum()) / num_valid_angles

    elif loss_type == 'mae':
        abs_error = torch.abs(pred_angles - true_angles_clean)
        masked_loss_sum = (abs_error * expanded_mask.float()).sum()
        loss = masked_loss_sum / num_valid_elements

    else:
        raise ValueError(f"Unknown angle loss_type: {loss_type}")

    return loss


def compute_combined_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined loss from multiple loss components using V1 losses.

    Args:
        outputs: Dictionary of model outputs:
            - pred_coords: Predicted coordinates (batch_size, seq_len, 3)
            - pred_confidence: Predicted confidence logits (batch_size, seq_len)
            - pred_angles: Predicted angles sin/cos (batch_size, seq_len, 4)
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
        - total_loss: Combined weighted loss (scalar tensor for backprop)
        - loss_components_tensors: Dictionary of individual loss component tensors
                                   (useful for logging/analysis, still attached to graph)
    """
    # Extract model outputs
    pred_coords = outputs['pred_coords']
    pred_confidence = outputs['pred_confidence']
    pred_angles = outputs['pred_angles']

    # Extract ground truth and mask
    true_coords = batch['coordinates']
    # V1 uses dihedral_features directly as sin/cos targets
    true_angles = batch['dihedral_features']
    mask = batch['mask']
    device = pred_coords.device

    loss_components_tensors = {}

    # Compute individual losses (using the V1 stable/proxy versions defined above)
    fape_loss_val = compute_stable_fape_loss(pred_coords, true_coords, mask)
    loss_components_tensors['fape'] = fape_loss_val

    # Use V1 proxy confidence loss
    confidence_loss_val = compute_confidence_loss(pred_confidence, pred_coords, true_coords, mask)
    loss_components_tensors['confidence'] = confidence_loss_val

    # Use angle loss
    angle_loss_val = compute_angle_loss(pred_angles, true_angles, mask)
    loss_components_tensors['angle'] = angle_loss_val

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

    # Add total loss to the dictionary for logging purposes if needed
    # loss_components_tensors['total'] = total_loss # This might be redundant

    # Return total loss tensor and dictionary of component tensors
    return total_loss, loss_components_tensors