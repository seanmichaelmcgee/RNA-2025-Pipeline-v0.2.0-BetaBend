# START OF FILE: docs/claude/workflows/advanced_training_techniques.md

# Advanced Training Techniques for RNA 3D Folding

**Version:** 1.0
**Date:** 2024-04-15
**Purpose:** This document describes advanced techniques that can be applied during the training process of the RNA 3D folding model to potentially improve robustness, accuracy, handle difficult examples, or generate diverse predictions. These techniques build upon the foundational V1 architecture and basic loss functions.

## 1. Introduction

While the core V1 pipeline establishes a functional baseline, achieving state-of-the-art performance often requires more sophisticated training strategies. This guide covers several advanced techniques that can be implemented and experimented with, typically in later stages of development (post-V1), once the basic pipeline is stable.

These techniques include:
-   **Ensemble Methods**: Combining multiple models or predictions.
-   **Curriculum Learning**: Progressively increasing training difficulty.
-   **Adaptive Loss Weighting**: Automatically balancing multiple loss components.
-   **Position-Specific Loss Weighting**: Focusing loss on specific sequence regions.
-   **Confidence Target Refinement**: Using more accurate metrics like lDDT for confidence supervision.
-   **Structure Sampling**: Generating diverse structural predictions.
-   **Integrating External Metrics**: Incorporating structural biology metrics (use with caution).

## 2. Ensemble Methods

Ensembles combine predictions from multiple models or multiple runs of the same model to improve robustness and potentially accuracy.

### 2.1 Ensemble Loss Calculation

During training or evaluation, you can compute the loss across an ensemble.

```python
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from src.losses import compute_stable_fape_loss # Assuming V1 FAPE proxy

def compute_ensemble_fape_loss(
    ensemble_pred_coords: List[torch.Tensor],
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    ensemble_strategy: str = 'mean', # 'mean', 'min', 'weighted'
    clamp_value: float = 10.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute FAPE loss using an ensemble of model predictions (using simplified FAPE).

    Args:
        ensemble_pred_coords: List of predicted coords tensors [M, (B, N, 3)].
        true_coords: Ground truth coords tensor (B, N, 3).
        mask: Boolean mask tensor (B, N).
        ensemble_strategy: How to combine individual losses.
        clamp_value: Clamp value for FAPE proxy.

    Returns:
        Tuple of (ensemble_loss_scalar, individual_losses_tensor)
    """
    device = true_coords.device
    num_models = len(ensemble_pred_coords)

    if num_models == 0:
        raise ValueError("Empty ensemble provided")

    # Compute loss for each ensemble member
    individual_losses = torch.zeros(num_models, device=device)
    for i, pred_coords in enumerate(ensemble_pred_coords):
        # Use the stable V1 FAPE proxy
        individual_losses[i] = compute_stable_fape_loss(
            pred_coords, true_coords, mask, clamp_value
        )

    # Combine losses
    if ensemble_strategy == 'mean':
        ensemble_loss = individual_losses.mean()
    elif ensemble_strategy == 'min':
        ensemble_loss = individual_losses.min()
    elif ensemble_strategy == 'weighted':
        # Example: Weight inversely proportional to loss
        weights = 1.0 / (individual_losses + 1e-8)
        weights = weights / weights.sum()
        ensemble_loss = (individual_losses * weights).sum()
    else:
        raise ValueError(f"Unknown ensemble_strategy: {ensemble_strategy}")

    return ensemble_loss, individual_losses
```

### 2.2 Ensemble Prediction Generation

Combine coordinates from multiple models/samples into a consensus structure.

```python
def compute_ensemble_prediction(
    ensemble_pred_coords: List[torch.Tensor],
    ensemble_pred_confidence: Optional[List[torch.Tensor]] = None, # Logits
    mask: Optional[torch.Tensor] = None,
    strategy: str = 'confidence_weighted' # 'confidence_weighted', 'mean', 'median'
) -> torch.Tensor:
    """
    Compute a consensus prediction from an ensemble.

    Args:
        ensemble_pred_coords: List of predicted coords tensors [M, (B, N, 3)].
        ensemble_pred_confidence: Optional list of confidence logits [M, (B, N)].
                                   Required for 'confidence_weighted'.
        mask: Boolean mask tensor (B, N).
        strategy: Strategy for combining predictions.

    Returns:
        Consensus coordinates tensor (B, N, 3).
    """
    if not ensemble_pred_coords:
        raise ValueError("Empty ensemble provided")

    device = ensemble_pred_coords[0].device
    batch_size, seq_len, _ = ensemble_pred_coords[0].shape
    num_models = len(ensemble_pred_coords)

    # Default mask
    if mask is None:
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    # Stack predictions
    stacked_coords = torch.stack(ensemble_pred_coords, dim=0) # (M, B, N, 3)

    if strategy == 'confidence_weighted':
        if not ensemble_pred_confidence or len(ensemble_pred_confidence) != num_models:
            raise ValueError("Confidence scores required for confidence_weighted strategy.")

        conf_probs = [torch.sigmoid(conf) for conf in ensemble_pred_confidence]
        stacked_conf = torch.stack(conf_probs, dim=0) # (M, B, N)
        weights = stacked_conf.unsqueeze(-1) # (M, B, N, 1)

        weighted_sum = (stacked_coords * weights).sum(dim=0)
        weight_sum = weights.sum(dim=0) + 1e-8
        consensus_coords = weighted_sum / weight_sum

    elif strategy == 'mean':
        consensus_coords = stacked_coords.mean(dim=0)
    elif strategy == 'median':
        consensus_coords = torch.median(stacked_coords, dim=0)[0]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Apply mask
    mask_3d = mask.unsqueeze(-1).expand_as(consensus_coords)
    consensus_coords = consensus_coords * mask_3d.float()

    return consensus_coords
```

**Usage**: Ensembles are typically used during inference/evaluation. The loss calculation can inform strategies like snapshot ensembling during training.

## 3. Curriculum Learning

Gradually increase the difficulty of the training process.

### 3.1 Loss Weight Scheduling

Adjust the weights of different loss components over epochs.

```python
class CurriculumLossManager:
    """Manages curriculum learning for loss weights."""

    def __init__(
        self,
        initial_weights: Dict[str, float],
        final_weights: Dict[str, float],
        total_epochs: int,
        curriculum_type: str = 'linear', # 'linear', 'exponential', 'step'
    ):
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.total_epochs = total_epochs
        self.curriculum_type = curriculum_type
        # Validate keys match
        # ... (validation logic as provided before) ...

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """Get current loss weights based on curriculum schedule."""
        epoch = max(0, min(epoch, self.total_epochs - 1))
        progress = epoch / max(1, self.total_epochs - 1)

        if self.curriculum_type == 'linear':
            weights = self._linear_schedule(progress)
        # ... (implement _linear_schedule, _exponential_schedule, _step_schedule) ...
        else:
            raise ValueError(f"Unknown curriculum_type: {self.curriculum_type}")
        return weights

    def _linear_schedule(self, progress: float) -> Dict[str, float]:
        weights = {}
        for key, initial_value in self.initial_weights.items():
            final_value = self.final_weights[key]
            weights[key] = initial_value + progress * (final_value - initial_value)
        return weights
    # ... (add other schedules) ...

# --- Example Usage in Training Loop ---
# curriculum = CurriculumLossManager(...)
# for epoch in range(num_epochs):
#     current_weights = curriculum.get_weights(epoch)
#     # ... train using current_weights ...
```

### 3.2 Difficulty-Weighted Loss

Focus training on examples or regions based on their current difficulty (e.g., prediction error).

```python
def compute_difficulty_weighted_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
    epoch: int,
    total_epochs: int,
    base_loss_fn: callable = compute_stable_fape_loss,
    clamp_value: float = 10.0,
    max_error_focus: float = 15.0 # Error level considered 'hardest'
) -> torch.Tensor:
    """Compute difficulty-weighted loss (focus shifts from easy to hard)."""
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device

    # Calculate per-residue errors (proxy for difficulty)
    per_residue_error = torch.zeros((batch_size, seq_len), device=device)
    with torch.no_grad():
        # ... (Kabsch alignment and error calculation per sequence, as before) ...
        for b in range(batch_size):
            # ... calculate errors for valid points ...
            # per_residue_error[b, valid_mask] = errors

    # Curriculum progress
    progress = epoch / max(1, total_epochs - 1)

    # Difficulty weights (shift focus from low error to high error)
    normalized_error = torch.clamp(per_residue_error / max_error_focus, 0.0, 1.0)
    # Weight = interpolation between (1 - error) and (error)
    difficulty_weights = progress * normalized_error + (1 - progress) * (1 - normalized_error)
    difficulty_weights = difficulty_weights * mask.float() # Apply mask

    # --- Compute Base Loss (with gradients) ---
    total_weighted_loss = 0.0
    total_weight_sum = 0.0

    for b in range(batch_size):
        # ... (Kabsch align, calculate per-residue clamped_errors as before) ...
        # valid_mask = mask[b]
        # clamped_errors = ... # Calculate FAPE per residue for valid points

        # Get weights for this sequence's valid residues
        seq_weights = difficulty_weights[b, valid_mask]

        # Apply weights to the loss term (clamped_errors)
        weighted_errors = clamped_errors * seq_weights
        total_weighted_loss += weighted_errors.sum()
        total_weight_sum += seq_weights.sum() # Use sum of weights for normalization

    # Average the weighted loss
    final_loss = total_weighted_loss / (total_weight_sum + 1e-8)
    return final_loss
```

**Usage**: Replace the standard loss call in the training step with the difficulty-weighted version.

## 4. Adaptive Loss Weighting

Learn the relative importance of different loss components automatically.

```python
class AdaptiveLossWeights(nn.Module):
    """Learns loss weights based on uncertainty (Kendall et al., 2018)."""

    def __init__(
        self,
        loss_names: List[str],
        initial_log_vars: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.loss_names = loss_names
        num_losses = len(loss_names)
        # ... (Initialization as provided before) ...
        init_vals = torch.zeros(num_losses, dtype=torch.float32) # Start with log_var=0 => weight=0.5
        if initial_log_vars:
            init_vals = torch.tensor([initial_log_vars.get(name, 0.0) for name in loss_names])
        self.log_vars = nn.Parameter(init_vals)


    def forward(
        self,
        losses: Dict[str, torch.Tensor] # Dict of individual loss tensors
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the adaptively weighted total loss."""
        # ... (Forward calculation as provided before) ...
        total_loss = 0.0
        current_weights = {}
        for i, name in enumerate(self.loss_names):
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            # Loss = L_i * exp(-log_var_i) / 2 + log_var_i / 2
            loss_term = 0.5 * precision * losses[name] + 0.5 * log_var
            total_loss += loss_term
            current_weights[f'{name}_weight'] = (0.5 * precision).item()
            current_weights[f'{name}_log_var'] = log_var.item()
        return total_loss, current_weights

# --- Example Usage in Training ---
# adaptive_weighter = AdaptiveLossWeights(loss_names=['fape', 'confidence', 'angle']).to(device)
# optimizer.add_param_group({'params': adaptive_weighter.parameters()}) # Add log_vars to optimizer

# In training step:
#   outputs = model(batch)
#   # Compute individual *unweighted* loss tensors
#   loss_components_tensors = {
#       'fape': compute_stable_fape_loss(...),
#       'confidence': compute_confidence_loss(...),
#       'angle': compute_angle_loss(...)
#   }
#   # Compute total loss using adaptive weights
#   total_loss, current_adaptive_weights = adaptive_weighter(loss_components_tensors)
#   # loss_history.update(..., current_adaptive_weights) # Log weights
#   total_loss.backward()
#   optimizer.step()
```

**Usage**: Replace fixed `loss_weights` with this learnable module. Add its parameters to the optimizer.

## 5. Position-Specific Loss Weighting

Emphasize or de-emphasize certain regions of the RNA during training.

```python
def compute_position_weighted_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
    clamp_value: float = 10.0,
    position_weights: Optional[torch.Tensor] = None, # (B, N)
    weighting_strategy: str = 'uniform',
    strategy_params: Optional[Dict] = None
) -> torch.Tensor:
    """Compute coordinate loss (FAPE proxy) with position-dependent weighting."""
    batch_size, seq_len, _ = pred_coords.shape
    device = pred_coords.device

    # Generate weights if not provided
    if position_weights is None:
        # ... (Implement strategies: 'uniform', 'distance_from_center', 'secondary_structure', 'conservation') ...
        # Example: Weight structured regions higher
        if weighting_strategy == 'secondary_structure':
             ss_mask = strategy_params['ss_mask'] # Boolean mask for stem regions
             stem_weight = strategy_params.get('stem_weight', 1.5)
             position_weights = torch.where(ss_mask, stem_weight, 1.0)
        else: # Default uniform
            position_weights = torch.ones((batch_size, seq_len), device=device)

    # Apply sequence mask
    position_weights = position_weights * mask.float()

    # Normalize weights per sequence
    weight_sums = position_weights.sum(dim=1, keepdim=True) + 1e-8
    normalized_weights = position_weights / weight_sums # (B, N)

    # Compute per-residue FAPE proxy loss (needs alignment)
    total_weighted_loss = 0.0
    total_valid_sequences = 0

    for b in range(batch_size):
        # ... (Kabsch align, calculate clamped_errors for valid residues) ...
        # clamped_errors = ... # (num_valid,)
        # seq_norm_weights = normalized_weights[b, valid_mask] # (num_valid,)

        # Apply weights to the per-residue loss
        # weighted_sequence_loss = (clamped_errors * seq_norm_weights).sum()
        # total_weighted_loss += weighted_sequence_loss
        # total_valid_sequences += 1
        pass # Placeholder for calculation loop

    # final_loss = total_weighted_loss / max(1, total_valid_sequences)
    # return final_loss
    return torch.tensor(0.0) # Placeholder return
```

**Usage**: Provide `position_weights` tensor or specify a `weighting_strategy` (requires passing necessary info like secondary structure masks or conservation scores in the batch).

## 6. Confidence Target Refinement (Using lDDT)

Improve confidence prediction by using a more accurate target like lDDT instead of the simple distance proxy.

```python
# Assume compute_lddt_target function exists (implementation provided previously)
# from src.losses import compute_lddt_target

def compute_lddt_confidence_loss(
    pred_confidence: torch.Tensor, # Logits
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    logits: bool = True,
    loss_type: str = 'mse' # 'mse' or 'bce'
) -> torch.Tensor:
    """Compute confidence loss using proper lDDT targets."""
    # Compute lDDT targets
    with torch.no_grad():
        lddt_targets = compute_lddt_target(
            pred_coords, true_coords, mask, per_residue=True
        ) # (B, N)

    # Process predictions
    if logits:
        pred_probs = torch.sigmoid(pred_confidence)
    else:
        pred_probs = pred_confidence

    # Default mask
    if mask is None: mask = torch.ones_like(pred_confidence, dtype=torch.bool)

    # Compute loss
    if loss_type == 'mse':
        loss_unreduced = (pred_probs - lddt_targets) ** 2
    elif loss_type == 'bce':
        loss_unreduced = F.binary_cross_entropy_with_logits(
            pred_confidence, lddt_targets, reduction='none'
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # Apply mask and average
    masked_loss = loss_unreduced * mask.float()
    num_valid = mask.sum()
    loss = masked_loss.sum() / (num_valid + 1e-8) if num_valid > 0 else torch.tensor(0.0)

    return loss

# --- Example Usage ---
# In compute_combined_loss:
#   loss_components['confidence'] = compute_lddt_confidence_loss(
#       outputs['pred_confidence'], outputs['pred_coords'], batch['coordinates'], mask
#   )
```

**Usage**: Replace the call to `compute_confidence_loss` with `compute_lddt_confidence_loss` in your combined loss calculation. Ensure `compute_lddt_target` is available (likely in `src/losses.py` or `src/utils/metrics.py`).

## 7. Structure Sampling

Generate multiple diverse structure predictions for a single input sequence. Required for the Kaggle submission format (5 predictions).

```python
class StructureSampler:
    """Generates diverse structure predictions."""

    def __init__(
        self,
        num_samples: int = 5,
        sampling_strategy: str = 'dropout', # 'dropout', 'noise_injection'
        noise_scale: float = 0.1, # For 'noise_injection'
        use_confidence_for_noise: bool = False
    ):
        self.num_samples = num_samples
        self.sampling_strategy = sampling_strategy
        self.noise_scale = noise_scale
        self.use_confidence_for_noise = use_confidence_for_noise
        # ... (validation) ...

    def generate_samples(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        device: torch.device
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate multiple structure samples."""
        # ... (Implementation as provided before) ...
        # Key logic: Loop num_samples times, apply strategy (dropout/noise), run model.forward
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        pred_coords_list = []
        pred_conf_list = []
        original_mode = model.training

        if self.sampling_strategy == 'dropout': model.train()
        else: model.eval()

        for i in range(self.num_samples):
            with torch.no_grad():
                 if self.sampling_strategy == 'dropout':
                     outputs = model(batch_on_device)
                 elif self.sampling_strategy == 'noise_injection':
                     # Requires model modification or input perturbation
                     # perturbed_batch = self._add_noise(batch_on_device, model, i)
                     # outputs = model(perturbed_batch)
                     outputs = model(batch_on_device) # Placeholder
                 else:
                     outputs = model(batch_on_device)

            pred_coords_list.append(outputs.get('pred_coords'))
            pred_conf_list.append(outputs.get('pred_confidence')) # Logits

        model.train(original_mode)
        return pred_coords_list, pred_conf_list

    # ... (_add_noise helper implementation needed if using noise injection) ...
```

**Usage**: Use during inference (`scripts/predict.py`) to generate the 5 required predictions for the Kaggle submission format.

## 8. Integrating External Metrics (Advanced/Non-Differentiable)

Incorporate metrics like TM-score or RMSD calculated by external tools, primarily for evaluation or potentially advanced techniques like reinforcement learning (gradients won't flow through the external calculation).

```python
class ExternalMetricLoss:
    """Integrates external metrics (TM-score, RMSD) into loss computation."""

    def __init__(
        self,
        use_tm_score: bool = True, tm_weight: float = 0.1,
        use_rmsd: bool = True, rmsd_weight: float = 0.05,
        # ... (External tool path, caching etc.) ...
    ):
        # ... (Initialization) ...
        self.use_tm_score = use_tm_score
        self.tm_weight = tm_weight
        # ... (Implement _compute_tm_score and _compute_rmsd using subprocess calls
        #      to tools like US-align or internal NumPy implementations) ...

    def compute_loss(
        self, pred_coords, true_coords, mask, target_ids, base_loss
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # ... (Implementation as provided before) ...
        # 1. Calculate metrics (TM-score, RMSD) externally (non-differentiable)
        # 2. Compute penalty based on metrics (e.g., 1 - TM_score)
        # 3. Add weighted penalty to the base_loss
        # 4. Return combined loss and raw metrics dictionary
        metrics = {} # Populate with calculated TM/RMSD values
        combined_loss = base_loss # Start with differentiable loss
        # Add non-differentiable penalty terms based on metrics
        # combined_loss = combined_loss + self.tm_weight * tm_penalty + ...
        return combined_loss, metrics
```

**Usage**: Typically used during evaluation phases or potentially in RL settings, not standard gradient descent.

## Conclusion

These advanced techniques offer avenues for improving the performance, robustness, and utility of the RNA 3D folding model beyond the V1 baseline. They should be considered and implemented iteratively after establishing and validating the core pipeline. Careful experimentation and analysis are required to determine the effectiveness of each technique for this specific task. Remember to test thoroughly and monitor the impact on both training dynamics and final prediction quality.
