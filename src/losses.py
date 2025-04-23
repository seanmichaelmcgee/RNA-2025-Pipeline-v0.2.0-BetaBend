# START OF FILE: src/losses.py
"""
Loss Functions for RNA 3D Structure Prediction (V1 Implementation)

This module implements the loss functions used for training the V1 RNA folding model.
It includes a simplified FAPE proxy loss for coordinates, a confidence prediction loss
using a derived target, and an auxiliary angle prediction loss.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic logging setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Helper Functions ---


def stable_kabsch_align(
    P: torch.Tensor, Q: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
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
    if P.shape[0] < 1:  # Handle empty input
        return P
    
    # Validate input - we need at least 3 points for reliable alignment
    if P.shape[0] < 3:
        logging.warning("Too few points for Kabsch alignment. Need at least 3 points.")
        # Return centered points at least
        p_mean = P.mean(dim=0, keepdim=True)
        q_mean = Q.mean(dim=0, keepdim=True)
        return P - p_mean + q_mean

    # Direct check for identical inputs
    if torch.allclose(P, Q, atol=1e-7):
        return P.clone()

    # Center the points
    p_mean = P.mean(dim=0, keepdim=True)
    q_mean = Q.mean(dim=0, keepdim=True)
    P_centered = P - p_mean
    Q_centered = Q - q_mean

    # Check for degenerate cases (all points coincident)
    p_var = torch.var(P, dim=0).sum()  # Sum of variances in each dimension
    q_var = torch.var(Q, dim=0).sum()
    
    # More reliable test for degeneracy using variance
    if p_var < epsilon or q_var < epsilon:
        # All points are basically coincident, only apply translation
        logging.debug(
            "Degenerate input to Kabsch (points coincident). Applying translation only."
        )
        return P - p_mean + q_mean  # Align centers

    # Compute covariance matrix
    C = torch.matmul(P_centered.transpose(-2, -1), Q_centered)

    try:
        # ========== SIMPLIFIED ROBUST APPROACH ==========
        # This handles all cases (full-rank, planar, collinear) more consistently
        
        # Compute SVD of covariance matrix
        try:
            # Try full SVD first
            U, S, Vt = torch.linalg.svd(C, full_matrices=False)
            V = Vt.transpose(-2, -1)
            
            # The key issue is handling reflections consistently
            # We'll use a simpler and more robust approach based on QCP method
            
            # First compute optimal rotation without worrying about reflections
            R_raw = torch.matmul(V, U.transpose(-2, -1))
            
            # Check determinant to identify reflections
            det = torch.det(R_raw)
            
            if det < 0:
                # We have a reflection case
                # Create a proper rotation matrix by flipping the last column of V
                # (corresponding to smallest singular value)
                V_fixed = V.clone()
                V_fixed[:, -1] = -V_fixed[:, -1]  # Flip the right singular vector for smallest singular value
                
                # Recompute rotation matrix
                R = torch.matmul(V_fixed, U.transpose(-2, -1))
                
                # Verify determinant is close to 1 now
                new_det = torch.det(R)
                if abs(new_det - 1.0) > 0.01:  # More lenient check
                    logging.warning(f"Reflection correction failed: det={new_det:.5f}. Using emergency fix.")
                    # Emergency fix: force it to be a proper rotation
                    R = R / new_det
            else:
                # Already a proper rotation
                R = R_raw
                
        except RuntimeError as svd_error:
            # If SVD fails, try a direct orthogonalization approach for simple cases
            logging.warning(f"SVD failed: {svd_error}, trying orthogonalization directly")
            
            # For the rotation-only case (common with rotated structures)
            # Try using Gram-Schmidt orthogonalization directly
            
            # For direct approach, check if we have >= 4 non-coplanar points for a robust fit
            if P.shape[0] >= 4:
                # Construct basis vectors using the first points
                v1 = P_centered[1] - P_centered[0]  # First basis vector
                v1 = v1 / (torch.norm(v1) + epsilon)
                
                # Find a point not collinear with the first vector
                dots = torch.abs(torch.sum(P_centered * v1.unsqueeze(0), dim=1))
                norms = torch.norm(P_centered, dim=1)
                
                # Look for a point that's not collinear (dot product not close to norm)
                non_collinear_idx = 2  # Default to third point
                for i in range(1, P.shape[0]):
                    if abs(dots[i] - norms[i]) > epsilon:
                        non_collinear_idx = i
                        break
                
                # Calculate second basis vector using the non-collinear point
                v2_unnorm = P_centered[non_collinear_idx] - torch.sum(P_centered[non_collinear_idx] * v1) * v1
                v2 = v2_unnorm / (torch.norm(v2_unnorm) + epsilon)
                
                # Third basis vector via cross product
                v3 = torch.linalg.cross(v1, v2)
                v3 = v3 / (torch.norm(v3) + epsilon)
                
                # Construct rotation matrix
                R_p = torch.stack([v1, v2, v3], dim=1)
                
                # Repeat for Q
                v1_q = Q_centered[1] - Q_centered[0]
                v1_q = v1_q / (torch.norm(v1_q) + epsilon)
                
                # Use same non-collinear point for consistency
                v2_q_unnorm = Q_centered[non_collinear_idx] - torch.sum(Q_centered[non_collinear_idx] * v1_q) * v1_q
                v2_q = v2_q_unnorm / (torch.norm(v2_q_unnorm) + epsilon)
                
                v3_q = torch.linalg.cross(v1_q, v2_q)
                v3_q = v3_q / (torch.norm(v3_q) + epsilon)
                
                R_q = torch.stack([v1_q, v2_q, v3_q], dim=1)
                
                # Final rotation from P to Q
                R = torch.matmul(R_q, R_p.transpose(-2, -1))
            else:
                # Simpler case - try to align principal axes directly
                # Create orthogonal basis from P and Q
                
                # Get principal axes - largest variance directions
                p_centered_T = P_centered.transpose(0, 1)  # (3, N)
                q_centered_T = Q_centered.transpose(0, 1)  # (3, N)
                
                # Simple covariance-based approach for small matrices
                p_cov = torch.matmul(p_centered_T, p_centered_T.transpose(0, 1))
                q_cov = torch.matmul(q_centered_T, q_centered_T.transpose(0, 1))
                
                # Get eigenvectors - these will form our basis
                try:
                    p_eig = torch.linalg.eigh(p_cov)[1]  # Eigenvectors
                    q_eig = torch.linalg.eigh(q_cov)[1]  # Eigenvectors
                    
                    # Rotation from P basis to Q basis
                    R = torch.matmul(q_eig, p_eig.transpose(-2, -1))
                    
                    # Check for proper rotation
                    det_R = torch.det(R)
                    if det_R < 0:
                        # Flip the last column to ensure proper rotation
                        R[:, -1] = -R[:, -1]
                except RuntimeError as eig_error:
                    logging.warning(f"Eigendecomposition failed: {eig_error}, using identity rotation")
                    # Fallback to identity rotation (translation-only alignment)
                    R = torch.eye(3, device=P.device)
        
        # Final validation - ensure R is a valid rotation matrix
        orthogonality_check = torch.matmul(R, R.transpose(-2, -1))
        identity = torch.eye(3, device=P.device)
        
        # Check if R is orthogonal (R*R^T = I)
        is_orthogonal = torch.allclose(orthogonality_check, identity, atol=1e-5)
        
        # Check if determinant is 1 (proper rotation)
        det_R = torch.det(R)
        is_proper = abs(det_R - 1.0) < 1e-5
        
        if not (is_orthogonal and is_proper):
            logging.warning(
                f"Rotation matrix not valid: orthogonal={is_orthogonal}, proper={is_proper}, det={det_R:.5f}"
            )
            # Fix orthogonality issues with a robust approach
            # Using SVD to find the nearest orthogonal matrix
            try:
                U_fix, _, Vt_fix = torch.linalg.svd(R, full_matrices=False)
                R = torch.matmul(U_fix, Vt_fix)  # This is guaranteed to be orthogonal
                
                # Ensure determinant is 1
                det_R = torch.det(R)
                if det_R < 0:
                    # Flip last column
                    U_fix[:, -1] = -U_fix[:, -1]
                    R = torch.matmul(U_fix, Vt_fix)
            except RuntimeError:
                logging.warning("Failed to fix rotation matrix, falling back to translation-only alignment")
                return P - p_mean + q_mean
        
        # Apply rotation and translation
        P_aligned = torch.matmul(P_centered, R) + q_mean
            
    except Exception as e:
        # Handle any other exceptions
        logging.warning(f"Exception in Kabsch alignment: {e}. Falling back to translation alignment.")
        P_aligned = P - p_mean + q_mean  # Fallback to center alignment
        
    # Final validation - check for NaNs or Inf values
    if torch.isnan(P_aligned).any() or torch.isinf(P_aligned).any():
        logging.error("NaN/Inf detected in Kabsch output despite safeguards! Returning translated points.")
        return P - p_mean + q_mean  # Even safer fallback - just translation
        
    return P_aligned


def robust_distance_calculation(
    pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8
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
    # Direct check for identical inputs to return exactly zero
    if torch.all(torch.eq(pred, target)):
        return torch.zeros(pred.shape[:-1], device=pred.device, dtype=pred.dtype)

    # Calculate squared differences element-wise
    squared_diff = (pred - target) ** 2  # (*, 3)

    # Sum squared differences across the coordinate dimension (dim=-1)
    sum_sq_diff = torch.sum(squared_diff, dim=-1)  # (*)

    # Check for exact zeros - where all coordinates match exactly
    exact_zeros = torch.all(pred == target, dim=-1)
    
    # For elements that aren't exact zeros, compute with sqrt
    # Add epsilon inside sqrt for numerical stability with small but non-zero distances
    distances = torch.where(
        exact_zeros,
        torch.zeros_like(sum_sq_diff),
        torch.sqrt(sum_sq_diff + epsilon)
    )
    
    # Lower threshold for detectable differences - using much smaller epsilon
    # This allows very small differences to be detected
    tiny_threshold = 1e-15
    tiny_diffs = sum_sq_diff < tiny_threshold
    
    # For extremely small differences, use even more precise calculation
    if tiny_diffs.any():
        # For these tiny values, do a direct calculation without epsilon
        precise_distances = torch.sqrt(sum_sq_diff)
        distances = torch.where(
            tiny_diffs & ~exact_zeros,  # Only for tiny non-zero differences
            precise_distances,
            distances
        )
    
    # Check for NaNs explicitly - if any input was NaN, the output should be NaN
    input_has_nan = torch.isnan(pred).any(dim=-1) | torch.isnan(target).any(dim=-1)
    if input_has_nan.any():
        distances = torch.where(
            input_has_nan,
            torch.tensor(float('nan'), device=distances.device, dtype=distances.dtype),
            distances
        )
    
    return distances


# --- Core V1 Loss Functions ---


def compute_stable_fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_value: float = 10.0,
    epsilon: float = 1e-8,
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
    device = pred_coords.device
    dtype = pred_coords.dtype
    
    # Handle different shapes of true_coords
    if len(true_coords.shape) == 4:
        # If true_coords is (batch_size, seq_len, seq_len, 3), extract diagonal
        logging.info(f"True coords has shape {true_coords.shape}, extracting diagonal")
        # Extract the diagonal entries (i==j) to get (batch_size, seq_len, 3)
        batch_size, seq_len1, seq_len2, coords_dim = true_coords.shape
        if seq_len1 != seq_len2:
            logging.warning(f"Unexpected true_coords shape: {true_coords.shape}")
        
        # Create indices for the diagonal
        diag_indices = torch.arange(min(seq_len1, seq_len2), device=device)
        # Extract diagonal for each batch
        true_coords_diag = true_coords[:, diag_indices, diag_indices, :]
        true_coords = true_coords_diag
    
    # Now both should be (batch_size, seq_len, 3)
    batch_size, seq_len, _ = pred_coords.shape
    
    # Handle different sequence lengths - trim to minimum length
    if true_coords.shape[1] != seq_len:
        min_seq_len = min(seq_len, true_coords.shape[1])
        logging.warning(f"Sequence length mismatch: pred={seq_len}, true={true_coords.shape[1]}, using {min_seq_len}")
        pred_coords = pred_coords[:, :min_seq_len, :]
        true_coords = true_coords[:, :min_seq_len, :]
        if mask is not None:
            mask = mask[:, :min_seq_len]
    
    # Early return for identical inputs (addresses zero loss issue)
    # Note: using allclose with a small tolerance to account for floating point precision
    try:
        if torch.allclose(pred_coords, true_coords, atol=1e-7):
            return torch.tensor(0.0, device=device, dtype=dtype)
    except RuntimeError as e:
        logging.warning(f"Error in allclose check: {e} - shape1={pred_coords.shape}, shape2={true_coords.shape}")
        # Continue with regular processing

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
    total_loss = torch.tensor(0.0, device=device, dtype=dtype)
    total_valid_sequences = 0

    # Process each sequence separately
    for b in range(batch_size):
        valid_mask = mask[b]
        valid_count = valid_mask.sum().item()

        if valid_count < 3:  # Need at least 3 for stable Kabsch
            continue

        # Extract valid coordinates
        p_valid = pred_coords[b, valid_mask]
        t_valid = true_coords[b, valid_mask]

        # Special case handling for degenerate/coincident points situation
        is_coincident_pred = torch.allclose(
            p_valid - p_valid.mean(dim=0, keepdim=True),
            torch.zeros_like(p_valid),
            atol=1e-5,
        )

        is_coincident_target = torch.allclose(
            t_valid - t_valid.mean(dim=0, keepdim=True),
            torch.zeros_like(t_valid),
            atol=1e-5,
        )

        # If both point sets are degenerate, use a special case
        if is_coincident_pred and is_coincident_target:
            # If both are coincident points, just compare the centers directly
            pred_center = p_valid.mean(dim=0)
            target_center = t_valid.mean(dim=0)
            center_distance = torch.norm(pred_center - target_center)

            # Apply clamping to the center distance (same as normal flow)
            center_distance = torch.minimum(
                center_distance,
                torch.tensor(clamp_value, device=device, dtype=center_distance.dtype),
            )

            total_loss += center_distance
            total_valid_sequences += 1
            continue

        # If only one set is degenerate, we still expect an approximation of the score
        # rather than failing with NaN values
        if is_coincident_pred and not is_coincident_target:
            # For the case when predicted structure is all coincident points
            # But target structure is not, this is clearly a poor prediction
            # Return the maximum loss value (clamped)
            total_loss = total_loss + torch.tensor(clamp_value, device=device, dtype=dtype)
            total_valid_sequences += 1
            continue

        # Proceed with normal calculation for non-degenerate cases
        try:
            # Apply stable Kabsch alignment
            p_aligned = stable_kabsch_align(p_valid, t_valid, epsilon=epsilon)

            # Compute robust distances
            distances = robust_distance_calculation(p_aligned, t_valid, epsilon=epsilon)

            # Apply stable clamping (torch.minimum is often more stable than clamp)
            clamped_distances = torch.minimum(
                distances,
                torch.tensor(clamp_value, device=device, dtype=distances.dtype),
            )

            # Compute mean loss for this sequence
            if clamped_distances.numel() > 0:
                sequence_loss = clamped_distances.mean()

                # Final check for NaN/Inf in the sequence loss itself
                if torch.isnan(sequence_loss) or torch.isinf(sequence_loss):
                    logging.warning(
                        f"NaN/Inf computed for sequence loss (batch {b}), skipping."
                    )
                    continue

                total_loss += sequence_loss
                total_valid_sequences += 1
            else:
                logging.debug(
                    f"No distances calculated for batch item {b} (likely due to mask)."
                )

        except Exception as e:
            logging.error(f"Error computing stable FAPE for batch item {b}: {e}")
            # Optionally log more details about p_valid, t_valid here if debugging
            continue  # Skip this sequence

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
    loss_type: str = "mse",  # 'mse' or 'bce'
    target_type: str = "lddt_proxy",
    scaling_factor: float = 3.0,  # For 'lddt_proxy'
    epsilon: float = 1e-8,
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
        # Initialize per_residue_error and properly handle the mask
        # Start with "worst possible error" (high value) for all positions
        # This ensures masked positions have an appropriately high error by default
        per_residue_error = (
            torch.ones((batch_size, seq_len), device=device, dtype=dtype) * 100.0
        )

        for b in range(batch_size):
            valid_mask = mask[b]
            valid_count = valid_mask.sum().item()

            if valid_count >= 3:  # Need at least 3 points for alignment
                p_valid = pred_coords[b, valid_mask]
                t_valid = true_coords[b, valid_mask]

                # Check for degenerate/coincident points
                is_coincident_pred = torch.allclose(
                    p_valid - p_valid.mean(dim=0, keepdim=True),
                    torch.zeros_like(p_valid),
                    atol=1e-5,
                )

                is_coincident_target = torch.allclose(
                    t_valid - t_valid.mean(dim=0, keepdim=True),
                    torch.zeros_like(t_valid),
                    atol=1e-5,
                )

                if is_coincident_pred and is_coincident_target:
                    # Both are coincident points - compare centers
                    pred_center = p_valid.mean(dim=0)
                    target_center = t_valid.mean(dim=0)
                    center_distance = torch.norm(pred_center - target_center)

                    # Set the same error for all valid residues
                    per_residue_error[b, valid_mask] = center_distance

                elif is_coincident_pred and not is_coincident_target:
                    # Predicted structure is all coincident but target isn't
                    # This is clearly a poor prediction - assign high error
                    per_residue_error[b, valid_mask] = 20.0  # High error value

                else:
                    # Standard case - align and calculate distances
                    try:
                        aligned_p_valid = stable_kabsch_align(
                            p_valid, t_valid, epsilon=epsilon
                        )

                        # Compute per-residue coordinate error for valid residues
                        errors = robust_distance_calculation(
                            aligned_p_valid, t_valid, epsilon=epsilon
                        )
                        per_residue_error[b, valid_mask] = errors
                    except Exception as e:
                        logging.error(
                            f"Error in confidence loss computation for batch {b}: {e}"
                        )
                        # Keep the high error values for this batch

            elif valid_count > 0:  # Some valid points but not enough for alignment
                # Calculate raw error without alignment
                raw_errors = robust_distance_calculation(
                    pred_coords[b, valid_mask],
                    true_coords[b, valid_mask],
                    epsilon=epsilon,
                )
                per_residue_error[b, valid_mask] = raw_errors

            # For completely invalid sequences (valid_count==0), keep the high error

        # --- Compute confidence target based on error ---
        if target_type == "lddt_proxy":
            # Convert error to per-residue lDDT-like score in [0, 1]
            # Higher score means better prediction (lower error)
            # Base formula: exp(-err/scale) maps [0,inf) -> (0,1]
            # Error of 0 Å -> score of 1.0
            # Error of 2*scale Å -> score of ~0.5
            # Error of 5*scale Å -> score of ~0.2
            conf_targets = torch.exp(-per_residue_error / scaling_factor)
            conf_targets = torch.clamp(conf_targets, 0.0, 1.0)  # Ensure [0, 1]

        elif target_type == "distance_based":
            # Alternative: directly scale distances to [0, 1] range
            # 1 = low error, 0 = high error
            max_dist = 15.0  # Maximum distance to consider (Å)
            conf_targets = 1.0 - torch.clamp(per_residue_error / max_dist, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown target_type for confidence loss: {target_type}")

        # Create a float mask for masked operations
        float_mask = mask.float()

        # Set invalid positions (masked) to have zero target
        # This is crucial as we only want to compute loss for valid positions
        # First make sure conf_targets and mask are on same device
        conf_targets = conf_targets.to(device)
        float_mask = float_mask.to(device)

        # Use masked_fill to ensure precise masking behavior
        conf_targets = conf_targets.masked_fill(~mask, 0.0)

    # --- Compute Loss ---
    if loss_type == "mse":
        # Apply sigmoid to predicted logits
        pred_probs = torch.sigmoid(pred_confidence)

        # Calculate MSE loss only for valid positions
        squared_error = (pred_probs - conf_targets) ** 2
        masked_loss = squared_error * float_mask
    elif loss_type == "bce":
        # Calculate BCE loss
        masked_loss = (
            F.binary_cross_entropy_with_logits(
                pred_confidence, conf_targets, reduction="none"
            )
            * float_mask
        )
    else:
        raise ValueError(f"Unknown loss_type for confidence loss: {loss_type}")

    # Compute mean loss over valid positions
    num_valid = float_mask.sum().item()
    if num_valid == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    loss = masked_loss.sum() / num_valid

    return loss


def compute_angle_loss(
    pred_angles: torch.Tensor,
    true_angles: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = "mse",  # 'mse', 'cosine', or 'mae'
    epsilon: float = 1e-8,
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
    expanded_mask = angle_mask.unsqueeze(-1).expand_as(
        pred_angles
    )  # (batch_size, seq_len, 4)

    # Number of valid elements for averaging
    num_valid_elements = expanded_mask.sum().item()
    if num_valid_elements == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)

    # Replace NaNs with zeros in true angles (masked out anyway)
    true_angles_clean = torch.nan_to_num(true_angles, nan=0.0)

    # --- Compute Loss ---
    if loss_type == "mse":
        squared_error = (pred_angles - true_angles_clean) ** 2
        masked_loss_sum = (squared_error * expanded_mask.float()).sum()
        loss = masked_loss_sum / num_valid_elements

    elif loss_type == "cosine":
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
        cos_sim_eta = torch.sum(pred_eta_norm * true_eta_norm, dim=2)  # (B, L)
        cos_sim_theta = torch.sum(pred_theta_norm * true_theta_norm, dim=2)  # (B, L)

        # Loss = 1 - cos_sim. Average over eta and theta.
        eta_loss_term = 1.0 - cos_sim_eta
        theta_loss_term = 1.0 - cos_sim_theta

        # Apply mask
        masked_eta_loss = eta_loss_term * angle_mask.float()
        masked_theta_loss = theta_loss_term * angle_mask.float()

        # Sum losses and divide by number of valid angles (num_valid_residues * 2)
        num_valid_angles = angle_mask.sum().item() * 2
        if num_valid_angles == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        loss = (masked_eta_loss.sum() + masked_theta_loss.sum()) / num_valid_angles

    elif loss_type == "mae":
        abs_error = torch.abs(pred_angles - true_angles_clean)
        masked_loss_sum = (abs_error * expanded_mask.float()).sum()
        loss = masked_loss_sum / num_valid_elements

    else:
        raise ValueError(f"Unknown angle loss_type: {loss_type}")

    return loss


def compute_combined_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_weights: Dict[str, float],
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
    pred_coords = outputs["pred_coords"]
    pred_confidence = outputs["pred_confidence"]
    pred_angles = outputs["pred_angles"]

    # Extract ground truth and mask
    true_coords = batch["coordinates"]
    # V1 uses dihedral_features directly as sin/cos targets
    true_angles = batch["dihedral_features"]
    mask = batch["mask"]
    device = pred_coords.device

    loss_components_tensors = {}

    # Compute individual losses (using the V1 stable/proxy versions defined above)
    fape_loss_val = compute_stable_fape_loss(pred_coords, true_coords, mask)
    loss_components_tensors["fape"] = fape_loss_val

    # Use V1 proxy confidence loss
    confidence_loss_val = compute_confidence_loss(
        pred_confidence, pred_coords, true_coords, mask
    )
    loss_components_tensors["confidence"] = confidence_loss_val

    # Use angle loss
    angle_loss_val = compute_angle_loss(pred_angles, true_angles, mask)
    loss_components_tensors["angle"] = angle_loss_val

    # Extract weights with defaults
    fape_weight = loss_weights.get("fape", 1.0)
    confidence_weight = loss_weights.get("confidence", 0.1)
    angle_weight = loss_weights.get("angle", 0.5)

    # Combine losses using weights
    total_loss = (
        fape_weight * fape_loss_val
        + confidence_weight * confidence_loss_val
        + angle_weight * angle_loss_val
    )

    # Add total loss to the dictionary for logging purposes if needed
    # loss_components_tensors['total'] = total_loss # This might be redundant

    # Return total loss tensor and dictionary of component tensors
    return total_loss, loss_components_tensors
