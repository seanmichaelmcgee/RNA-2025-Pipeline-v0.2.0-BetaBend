#!/usr/bin/env python3
"""
Enhanced Training Script for RNA 3D Structure Prediction

This script provides a configurable way to train improved RNA folding models
with focus on:
1. Increased model capacity
2. Better hyperparameter settings
3. Advanced training strategies
4. Automated validation and checkpoint management
"""

import os
import sys
import argparse
import json
import time
import logging
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add project root to path for importing project modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.models.rna_folding_model import RNAFoldingModel
from src.data_loading import RNADataset, collate_fn, create_data_loader
from src.losses import compute_fape_loss, compute_confidence_loss, compute_angle_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train an enhanced RNA folding model')
    
    # Data parameters
    parser.add_argument('--train_csv', type=str, default='data/raw/train_sequences.csv',
                       help='Path to training sequences CSV')
    parser.add_argument('--features_dir', type=str, default='data/processed/',
                       help='Path to processed features directory')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--temporal_cutoff', type=str, default='2022-05-01',
                       help='Temporal cutoff date for features')
    parser.add_argument('--max_seq_len', type=int, default=256,
                       help='Maximum sequence length for training')
    
    # Model architecture parameters
    parser.add_argument('--num_blocks', type=int, default=6,
                       help='Number of transformer blocks')
    parser.add_argument('--residue_embed_dim', type=int, default=192,
                       help='Residue embedding dimension')
    parser.add_argument('--pair_embed_dim', type=int, default=64,
                       help='Pair embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=512,
                       help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['plateau', 'cosine', 'none'],
                       help='LR scheduler type')
    
    # Loss weights
    parser.add_argument('--fape_weight', type=float, default=1.0,
                       help='Weight for FAPE loss')
    parser.add_argument('--confidence_weight', type=float, default=0.1,
                       help='Weight for confidence loss')
    parser.add_argument('--angle_weight', type=float, default=0.5,
                       help='Weight for angle loss')
    
    # Training options
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every N epochs')
    parser.add_argument('--output_dir', type=str, default='results/enhanced_model',
                       help='Output directory for saving models and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--validate_checkpoints', action='store_true',
                       help='Run full validation on best checkpoints')
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_output_dirs(args):
    """Setup output directories for logs and model checkpoints."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    # Setup file logging
    file_handler = logging.FileHandler(os.path.join(run_dir, "logs", "training.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # Save config
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    return run_dir

def create_model(args):
    """Create model with specified architecture."""
    model_config = {
        'num_blocks': args.num_blocks,
        'residue_embed_dim': args.residue_embed_dim,
        'pair_embed_dim': args.pair_embed_dim,
        'num_attention_heads': args.num_heads,
        'ff_dim': args.ff_dim,
        'dropout': args.dropout,
    }
    
    model = RNAFoldingModel(model_config)
    return model, model_config

def create_dataloaders(args):
    """Create training and validation data loaders."""
    # Create full dataset
    dataset = RNADataset(
        sequences_csv_path=args.train_csv,
        features_dir=args.features_dir,
        temporal_cutoff=args.temporal_cutoff,
        max_seq_len=args.max_seq_len,
    )
    
    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    logger.info(f"Dataset split: {train_size} training, {val_size} validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, device, loss_weights):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    fape_losses = 0
    conf_losses = 0
    angle_losses = 0
    
    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
               for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        
        # Compute losses
        fape_loss = compute_fape_loss(
            pred_coords=outputs["pred_coords"],
            true_coords=batch["coords"],
            pred_confidence=torch.sigmoid(outputs["pred_confidence"]),
            lengths=batch["lengths"],
        )
        
        confidence_loss = compute_confidence_loss(
            pred_confidence=outputs["pred_confidence"],
            true_coords=batch["coords"],
            pred_coords=outputs["pred_coords"],
            lengths=batch["lengths"],
        )
        
        angle_loss = compute_angle_loss(
            pred_coords=outputs["pred_coords"],
            true_coords=batch["coords"],
            lengths=batch["lengths"],
        )
        
        # Combine losses
        loss = (
            loss_weights["fape"] * fape_loss
            + loss_weights["confidence"] * confidence_loss
            + loss_weights["angle"] * angle_loss
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        fape_losses += fape_loss.item()
        conf_losses += confidence_loss.item()
        angle_losses += angle_loss.item()
    
    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_fape = fape_losses / len(train_loader)
    avg_conf = conf_losses / len(train_loader)
    avg_angle = angle_losses / len(train_loader)
    
    return {
        "loss": avg_loss,
        "fape_loss": avg_fape,
        "confidence_loss": avg_conf,
        "angle_loss": avg_angle,
    }

def validate(model, val_loader, device, loss_weights):
    """Validate the model."""
    model.eval()
    total_loss = 0
    fape_losses = 0
    conf_losses = 0
    angle_losses = 0
    all_rmsd = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Compute losses
            fape_loss = compute_fape_loss(
                pred_coords=outputs["pred_coords"],
                true_coords=batch["coords"],
                pred_confidence=torch.sigmoid(outputs["pred_confidence"]),
                lengths=batch["lengths"],
            )
            
            confidence_loss = compute_confidence_loss(
                pred_confidence=outputs["pred_confidence"],
                true_coords=batch["coords"],
                pred_coords=outputs["pred_coords"],
                lengths=batch["lengths"],
            )
            
            angle_loss = compute_angle_loss(
                pred_coords=outputs["pred_coords"],
                true_coords=batch["coords"],
                lengths=batch["lengths"],
            )
            
            # Combine losses
            loss = (
                loss_weights["fape"] * fape_loss
                + loss_weights["confidence"] * confidence_loss
                + loss_weights["angle"] * angle_loss
            )
            
            # Track metrics
            total_loss += loss.item()
            fape_losses += fape_loss.item()
            conf_losses += confidence_loss.item()
            angle_losses += angle_loss.item()
            
            # Calculate RMSD for each sequence in batch
            for i in range(len(batch["lengths"])):
                seq_len = batch["lengths"][i].item()
                pred_coords_i = outputs["pred_coords"][i, :seq_len].cpu().numpy()
                true_coords_i = batch["coords"][i, :seq_len].cpu().numpy()
                
                # Basic RMSD calculation
                rmsd = np.sqrt(np.mean(np.sum((pred_coords_i - true_coords_i) ** 2, axis=1)))
                all_rmsd.append(rmsd)
    
    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    avg_fape = fape_losses / len(val_loader)
    avg_conf = conf_losses / len(val_loader)
    avg_angle = angle_losses / len(val_loader)
    avg_rmsd = np.mean(all_rmsd)
    
    return {
        "loss": avg_loss,
        "fape_loss": avg_fape,
        "confidence_loss": avg_conf,
        "angle_loss": avg_angle,
        "rmsd": avg_rmsd,
    }

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, run_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    
    if is_best:
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": metrics["loss"],
        "val_rmsd": metrics["rmsd"],
        "best_val_metrics": metrics,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def create_optimizer(model, args):
    """Create optimizer and scheduler."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
    else:
        scheduler = None
    
    return optimizer, scheduler

def resume_from_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Resume training from checkpoint."""
    logger.info(f"Resuming from checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1
    
    return model, optimizer, scheduler, start_epoch

def log_metrics(metrics, epoch, prefix="train"):
    """Log metrics to console."""
    metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
    logger.info(f"{prefix.capitalize()} Epoch {epoch}: {metrics_str}")

def save_training_log(train_log, val_log, run_dir):
    """Save training logs to CSV file."""
    log_df = pd.DataFrame({
        "epoch": range(1, len(train_log) + 1),
        "train_loss": [log["loss"] for log in train_log],
        "val_loss": [log["loss"] for log in val_log],
        "val_rmsd": [log["rmsd"] for log in val_log],
    })
    
    log_path = os.path.join(run_dir, "training_log.csv")
    log_df.to_csv(log_path, index=False)
    logger.info(f"Training log saved to {log_path}")

def plot_metrics(train_log, val_log, run_dir):
    """Plot training and validation metrics."""
    epochs = range(1, len(train_log) + 1)
    
    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [log["loss"] for log in train_log], 'b-', label='Training Loss')
    plt.plot(epochs, [log["loss"] for log in val_log], 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Plot RMSD
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [log["rmsd"] for log in val_log], 'g-', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('RMSD (Å)')
    plt.title('Validation RMSD')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_metrics.png"), dpi=300)
    plt.close()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup output directories
    run_dir = setup_output_dirs(args)
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Create model
    model, model_config = create_model(args)
    model = model.to(device)
    logger.info(f"Created model with config: {model_config}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(args)
    logger.info(f"Created dataloaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, args)
    
    # Set loss weights
    loss_weights = {
        "fape": args.fape_weight,
        "confidence": args.confidence_weight,
        "angle": args.angle_weight,
    }
    logger.info(f"Using loss weights: {loss_weights}")
    
    # Optionally resume from checkpoint
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        model, optimizer, scheduler, start_epoch = resume_from_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        logger.info(f"Resumed training from epoch {start_epoch}")
    
    # Initialize tracking variables
    train_log = []
    val_log = []
    best_val_loss = float('inf')
    best_val_rmsd = float('inf')
    patience_counter = 0
    
    # Main training loop
    logger.info(f"Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        
        # Train for one epoch
        train_metrics = train_epoch(model, train_loader, optimizer, device, loss_weights)
        train_log.append(train_metrics)
        log_metrics(train_metrics, epoch + 1, prefix="train")
        
        # Validate
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = validate(model, val_loader, device, loss_weights)
            val_log.append(val_metrics)
            log_metrics(val_metrics, epoch + 1, prefix="val")
            
            # Update learning rate if using ReduceLROnPlateau
            if args.scheduler == 'plateau' and scheduler is not None:
                scheduler.step(val_metrics["loss"])
            
            # Save checkpoint if best so far
            is_best = False
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                is_best = True
                patience_counter = 0
            elif val_metrics["rmsd"] < best_val_rmsd:
                best_val_rmsd = val_metrics["rmsd"]
                is_best = True
                patience_counter = 0
            else:
                patience_counter += 1
            
            if is_best:
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, run_dir, is_best=True)
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, run_dir, is_best=False)
        else:
            # Keep validation log in sync with training log
            if val_log:
                val_log.append(val_log[-1])
            else:
                # Initialize with empty validation
                val_log.append({
                    "loss": float('inf'),
                    "fape_loss": float('inf'),
                    "confidence_loss": float('inf'),
                    "angle_loss": float('inf'),
                    "rmsd": float('inf'),
                })
        
        # Update learning rate if using CosineAnnealingLR
        if args.scheduler == 'cosine' and scheduler is not None:
            scheduler.step()
        
        # Check early stopping
        if patience_counter >= args.patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Log epoch time
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, 
                   val_log[-1], run_dir, is_best=False)
    
    # Save training log and plots
    save_training_log(train_log, val_log, run_dir)
    plot_metrics(train_log, val_log, run_dir)
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}, Best RMSD: {best_val_rmsd:.6f}")
    logger.info(f"All outputs saved to {run_dir}")
    
    # Optionally run full validation on best checkpoint
    if args.validate_checkpoints:
        best_checkpoint_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
        if os.path.exists(best_checkpoint_path):
            logger.info("Running full validation on best checkpoint...")
            # Here you could call a validation script or function
            # validate_checkpoint(best_checkpoint_path)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())