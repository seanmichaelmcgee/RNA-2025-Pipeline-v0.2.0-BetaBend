# %% [markdown]
# # RNA 3D Folding Model - Technical Validation (Tier 1)
# 
# This notebook implements Tier 1 validation for the RNA 3D folding model. The focus is on fast verification of model functionality and basic technical performance.
# 
# **Characteristics:**
# - Quick to run (<5 minutes)
# - Uses small subset of data (3-5 sequences)
# - Focuses on shape checks, gradient flow, and basic metrics

# %%
import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(os.getcwd()).parent.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.models.rna_folding_model import RNAFoldingModel
from src.data_loading import RNADataset
from src.utils.structure_metrics import compute_rmsd, compute_tm_score, compute_per_residue_rmsd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Configuration
# 
# Define validation parameters and model configuration.

# %%
# Validation configuration
CONFIG = {
    # Data configuration
    "data_dir": os.path.join(project_root, "data", "processed"),
    "subset_size": 5,  # Number of sequences to use for validation
    
    # Model configuration
    "model_config": {
        "d_model": 256,
        "d_feedforward": 1024,
        "num_layers": 4,
        "num_heads": 8,
        "dropout": 0.1,
        "ipa_dropout": 0.1,
        "use_checkpointing": False
    },
    
    # Paths
    "results_dir": os.path.join(project_root, "validation", "tier1_technical", "results"),
    "checkpoint_path": None,  # Path to model checkpoint (if available)
    
    # Runtime options
    "random_seed": 42,
    "batch_size": 2,
    "num_workers": 2
}

# Create results directory if it doesn't exist
os.makedirs(CONFIG["results_dir"], exist_ok=True)

# %% [markdown]
# ## 2. Create Small Validation Dataset
# 
# Load a small subset of validation data for quick technical validation.

# %%
def create_validation_subset(data_dir, subset_size=5, split="validation", seed=42):
    """Create a small subset of validation data for quick technical validation."""
    np.random.seed(seed)
    
    # Get list of validation files
    dihedral_dir = os.path.join(data_dir, "dihedral_features")
    all_files = [f for f in os.listdir(dihedral_dir) if f.endswith("_dihedral_features.npz")]
    
    # Load validation IDs from the processing summary
    summary_path = os.path.join(data_dir, f"{split}_processing_summary.json")
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    valid_ids = set(summary["processed_ids"])
    valid_files = [f for f in all_files if f.split("_dihedral")[0] in valid_ids]
    
    # Select a random subset
    if len(valid_files) <= subset_size:
        subset_files = valid_files
    else:
        subset_files = np.random.choice(valid_files, size=subset_size, replace=False)
    
    # Get sample IDs
    sample_ids = [f.split("_dihedral")[0] for f in subset_files]
    
    print(f"Created {split} subset with {len(sample_ids)} samples: {sample_ids}")
    return sample_ids

# Create validation subset
validation_ids = create_validation_subset(
    CONFIG["data_dir"], 
    CONFIG["subset_size"],
    seed=CONFIG["random_seed"]
)

# Create validation dataset
validation_dataset = RNADataset(
    data_dir=CONFIG["data_dir"],
    split="validation",
    filter_ids=validation_ids
)

# Create validation dataloader
validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    collate_fn=validation_dataset.collate_fn
)

# %% [markdown]
# ## 3. Initialize Model
# 
# Create model instance and load checkpoint if available.

# %%
def initialize_model(config, checkpoint_path=None):
    """Initialize model and load checkpoint if available."""
    model = RNAFoldingModel(
        d_model=config["model_config"]["d_model"],
        d_feedforward=config["model_config"]["d_feedforward"],
        num_layers=config["model_config"]["num_layers"],
        num_heads=config["model_config"]["num_heads"],
        dropout=config["model_config"]["dropout"],
        ipa_dropout=config["model_config"]["ipa_dropout"],
        use_checkpointing=config["model_config"]["use_checkpointing"]
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Initializing model with random weights")
    
    model = model.to(device)
    return model

# Initialize model
model = initialize_model(CONFIG, CONFIG["checkpoint_path"])

# %% [markdown]
# ## 4. Model Shape Check
# 
# Verify input and output tensor shapes with a sample batch.

# %%
def check_model_shapes(model, dataloader):
    """Check model input and output shapes with a sample batch."""
    batch = next(iter(dataloader))
    
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Set model to eval mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch)
    
    # Print input shapes
    print("\nInput shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Print output shapes
    print("\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
            
    # Return batch and outputs for further analysis
    return batch, outputs

batch, outputs = check_model_shapes(model, validation_loader)

# %% [markdown]
# ## 5. Gradient Flow Check
# 
# Verify gradient flow through the model.

# %%
def check_gradient_flow(model, batch):
    """Verify gradient flow through the model."""
    # Set model to train mode
    model.train()
    
    # Clear gradients
    model.zero_grad()
    
    # Forward pass
    outputs = model(batch)
    
    # Compute loss
    from src.losses import compute_fape_loss, compute_angle_loss
    
    # FAPE loss
    fape_loss = compute_fape_loss(
        pred_coords=outputs["final_atom_positions"],
        true_coords=batch["atom_positions"],
        mask=batch["atom_mask"],
        clamp_distance=10.0,
        reduction="mean"
    )
    
    # Angle loss (if angles are predicted)
    angle_loss = 0.0
    if "predicted_angles" in outputs:
        angle_loss = compute_angle_loss(
            predicted_angles=outputs["predicted_angles"],
            target_angles=batch["dihedral_angles"],
            angle_mask=batch["angle_mask"],
            reduction="mean"
        )
    
    # Total loss
    total_loss = fape_loss + 0.5 * angle_loss
    print(f"\nLoss values:")
    print(f"  FAPE loss: {fape_loss.item():.4f}")
    print(f"  Angle loss: {angle_loss if isinstance(angle_loss, float) else angle_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    print("\nGradient flow check:")
    
    # Parameters that should always have gradients
    critical_params = [
        "transformer_stack",
        "ipa_module"
    ]
    
    # Parameters that may not have gradients
    allowed_no_grad_params = [
        "embedding_module.relative_pos_encoding.embeddings.weight",
        "embedding_module.sequence_embedding.embedding.weight"
    ]
    
    # Track parameters with/without gradients
    parameters_with_gradient = []
    parameters_without_gradient = []
    
    for name, param in model.named_parameters():
        if param.grad is None:
            if name in allowed_no_grad_params:
                # This parameter is allowed to have no gradient
                continue
            parameters_without_gradient.append(name)
        else:
            # Check if gradient is non-zero
            if param.grad.abs().sum().item() > 0:
                parameters_with_gradient.append(name)
            else:
                parameters_without_gradient.append(name)
    
    # Check critical parameters
    critical_missing_grad = []
    for critical in critical_params:
        if not any(critical in param for param in parameters_with_gradient):
            critical_missing_grad.append(critical)
    
    # Print results
    print(f"  Parameters with gradients: {len(parameters_with_gradient)}")
    print(f"  Parameters without gradients: {len(parameters_without_gradient)}")
    
    if critical_missing_grad:
        print(f"  WARNING: Critical components missing gradients: {critical_missing_grad}")
        # Show the parameters without gradients for debugging
        print("  Parameters without gradients:")
        for name in parameters_without_gradient:
            print(f"    {name}")
    else:
        print("  All critical components have gradients. ✓")
    
    # Return gradient status
    gradient_status = {
        "parameters_with_gradient": parameters_with_gradient,
        "parameters_without_gradient": parameters_without_gradient,
        "critical_missing_grad": critical_missing_grad,
        "loss": {
            "fape": fape_loss.item(),
            "angle": angle_loss if isinstance(angle_loss, float) else angle_loss.item(),
            "total": total_loss.item()
        }
    }
    
    return gradient_status

# Check gradient flow
gradient_status = check_gradient_flow(model, batch)

# %% [markdown]
# ## 6. Structure Metrics Evaluation
# 
# Evaluate model predictions with structure metrics (RMSD, TM-score, per-residue RMSD).

# %%
def evaluate_structure_metrics(model, dataloader):
    """Evaluate model predictions with structure metrics."""
    model.eval()
    
    results = {
        "rmsd": [],
        "tm_score": [],
        "per_residue_rmsd": [],
        "ids": [],
        "sequence_lengths": []
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Get predictions and true coordinates
            pred_coords = outputs["final_atom_positions"]
            true_coords = batch["atom_positions"]
            atom_mask = batch["atom_mask"]
            
            # Calculate RMSD
            rmsd = compute_rmsd(pred_coords, true_coords, atom_mask)
            
            # Calculate TM-score
            tm_score = compute_tm_score(pred_coords, true_coords, atom_mask)
            
            # Calculate per-residue RMSD
            per_res_rmsd = compute_per_residue_rmsd(pred_coords, true_coords, atom_mask)
            
            # Store results
            for i in range(len(batch["ids"])):
                results["ids"].append(batch["ids"][i])
                results["rmsd"].append(rmsd[i].item())
                results["tm_score"].append(tm_score[i].item())
                results["per_residue_rmsd"].append(per_res_rmsd[i].detach().cpu().numpy())
                results["sequence_lengths"].append(int(atom_mask[i].sum().item() // 3))  # 3 atoms per residue
    
    # Print summary
    print("\nStructure Metrics Summary:")
    print(f"  Mean RMSD: {np.mean(results['rmsd']):.4f} Å")
    print(f"  Mean TM-score: {np.mean(results['tm_score']):.4f}")
    
    # Create detailed results table
    print("\nDetailed Results:")
    print(f"{'ID':<15} {'Length':<10} {'RMSD (Å)':<12} {'TM-score':<12}")
    print("-" * 50)
    for i in range(len(results["ids"])):
        print(f"{results['ids'][i]:<15} {results['sequence_lengths'][i]:<10} {results['rmsd'][i]:<12.4f} {results['tm_score'][i]:<12.4f}")
    
    return results

# Evaluate structure metrics
structure_metrics = evaluate_structure_metrics(model, validation_loader)

# %% [markdown]
# ## 7. Visualize Per-Residue RMSD
# 
# Plot per-residue RMSD for each sequence.

# %%
def visualize_per_residue_rmsd(results):
    """Visualize per-residue RMSD for each sequence."""
    num_sequences = len(results["ids"])
    fig, axes = plt.subplots(num_sequences, 1, figsize=(10, 4 * num_sequences))
    
    # Handle case with only one sequence
    if num_sequences == 1:
        axes = [axes]
    
    for i in range(num_sequences):
        per_res_rmsd = results["per_residue_rmsd"][i]
        seq_id = results["ids"][i]
        seq_length = results["sequence_lengths"][i]
        
        ax = axes[i]
        ax.plot(range(1, len(per_res_rmsd) + 1), per_res_rmsd, marker='o', linestyle='-')
        ax.axhline(y=np.mean(per_res_rmsd), color='r', linestyle='--', 
                   label=f'Mean RMSD: {np.mean(per_res_rmsd):.2f} Å')
        
        ax.set_xlabel('Residue Position')
        ax.set_ylabel('RMSD (Å)')
        ax.set_title(f'Per-Residue RMSD for {seq_id} (Length: {seq_length})')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(CONFIG["results_dir"], "per_residue_rmsd.png")
    plt.savefig(fig_path)
    print(f"\nSaved per-residue RMSD visualization to {fig_path}")
    
    return fig

# Visualize per-residue RMSD
per_residue_plot = visualize_per_residue_rmsd(structure_metrics)

# %% [markdown]
# ## 8. Memory and Performance Benchmarking
# 
# Measure memory usage and inference time.

# %%
def benchmark_model(model, dataloader, num_runs=3):
    """Benchmark model memory usage and inference time."""
    model.eval()
    
    # Get a batch for benchmarking
    batch = next(iter(dataloader))
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Measure inference time
    times = []
    
    print("\nBenchmarking model inference time...")
    with torch.no_grad():
        # Warmup run
        _ = model(batch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        # Timed runs
        for run in range(num_runs):
            start_time = time.time()
            _ = model(batch)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # Measure memory usage
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        
        # Forward pass to measure memory
        with torch.no_grad():
            _ = model(batch)
        
        # Get memory stats
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
        memory_str = f"{peak_memory:.2f} MB"
    else:
        memory_str = "N/A (CPU only)"
    
    print(f"\nPerformance Metrics:")
    print(f"  Inference time (avg of {num_runs} runs): {avg_time:.4f} ± {std_time:.4f} seconds")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Peak memory usage: {memory_str}")
    
    # Save results
    benchmark_results = {
        "inference_time": {
            "mean": avg_time,
            "std": std_time,
            "unit": "seconds"
        },
        "batch_size": CONFIG["batch_size"],
        "peak_memory": memory_str
    }
    
    return benchmark_results

# Benchmark model
benchmark_results = benchmark_model(model, validation_loader)

# %% [markdown]
# ## 9. Save Validation Results
# 
# Save validation results to a JSON file for future reference.

# %%
def save_validation_results(structure_metrics, gradient_status, benchmark_results):
    """Save validation results to a JSON file."""
    # Prepare results for saving
    results = {
        "validation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "config": CONFIG,
        "structure_metrics": {
            "rmsd": {
                "mean": float(np.mean(structure_metrics["rmsd"])),
                "std": float(np.std(structure_metrics["rmsd"])),
                "per_sequence": [
                    {"id": id, "length": length, "rmsd": rmsd} 
                    for id, length, rmsd in zip(
                        structure_metrics["ids"], 
                        structure_metrics["sequence_lengths"], 
                        structure_metrics["rmsd"]
                    )
                ]
            },
            "tm_score": {
                "mean": float(np.mean(structure_metrics["tm_score"])),
                "std": float(np.std(structure_metrics["tm_score"])),
                "per_sequence": [
                    {"id": id, "length": length, "tm_score": tm_score} 
                    for id, length, tm_score in zip(
                        structure_metrics["ids"], 
                        structure_metrics["sequence_lengths"], 
                        structure_metrics["tm_score"]
                    )
                ]
            },
            # Per-residue RMSD summary (only means to save space)
            "per_residue_rmsd": {
                "per_sequence": [
                    {"id": id, "length": length, "mean_rmsd": float(np.mean(per_res_rmsd))} 
                    for id, length, per_res_rmsd in zip(
                        structure_metrics["ids"], 
                        structure_metrics["sequence_lengths"], 
                        structure_metrics["per_residue_rmsd"]
                    )
                ]
            }
        },
        "gradient_status": {
            "parameters_with_gradient_count": len(gradient_status["parameters_with_gradient"]),
            "parameters_without_gradient_count": len(gradient_status["parameters_without_gradient"]),
            "critical_missing_grad": gradient_status["critical_missing_grad"],
            "loss": gradient_status["loss"]
        },
        "benchmark_results": benchmark_results
    }
    
    # Save to file
    results_file = os.path.join(CONFIG["results_dir"], "validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nValidation results saved to {results_file}")
    return results

# Save validation results
validation_results = save_validation_results(structure_metrics, gradient_status, benchmark_results)

# %% [markdown]
# ## 10. Validation Summary
# 
# Display a summary of validation results.

# %%
def display_summary(validation_results):
    """Display a summary of validation results."""
    print("\n" + "=" * 50)
    print("RNA 3D Folding Model - Technical Validation Summary")
    print("=" * 50)
    print(f"Validation performed on {validation_results['validation_time']}")
    print(f"Device: {validation_results['device']}")
    print("\nStructure Metrics:")
    print(f"  Mean RMSD: {validation_results['structure_metrics']['rmsd']['mean']:.4f} Å")
    print(f"  Mean TM-score: {validation_results['structure_metrics']['tm_score']['mean']:.4f}")
    
    print("\nGradient Flow:")
    if validation_results['gradient_status']['critical_missing_grad']:
        print(f"  ❌ Critical components missing gradients: {validation_results['gradient_status']['critical_missing_grad']}")
    else:
        print(f"  ✓ All critical components have gradients")
    print(f"  Parameters with gradients: {validation_results['gradient_status']['parameters_with_gradient_count']}")
    print(f"  Total loss: {validation_results['gradient_status']['loss']['total']:.4f}")
    
    print("\nPerformance:")
    print(f"  Inference time: {validation_results['benchmark_results']['inference_time']['mean']:.4f} seconds")
    print(f"  Peak memory usage: {validation_results['benchmark_results']['peak_memory']}")
    
    print("\nSummary:")
    if validation_results['gradient_status']['critical_missing_grad']:
        print("  ❌ FAILED: Gradient flow issues detected")
    else:
        print("  ✓ PASSED: Technical validation successful")
    print("=" * 50)

# Display summary
display_summary(validation_results)


