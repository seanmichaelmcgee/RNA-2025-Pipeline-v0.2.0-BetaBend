#!/usr/bin/env python3
"""
RNA 3D Structure Prediction Training Script (V1)

This script implements a complete training pipeline for the RNA folding model:
1. Loads training data using the data loading framework
2. Initializes and configures the RNAFoldingModel
3. Sets up optimizer and learning rate scheduler
4. Implements forward/backward passes and gradient updates
5. Tracks and logs training metrics
6. Periodically saves checkpoints and validates model performance
7. Implements early stopping

Usage:
    python scripts/train.py --sequences_csv data/sequences.csv --labels_csv data/labels.csv --features_dir data/features/ 
                           [--output_dir results/] [--checkpoint path/to/model.pt]
                           [--batch_size 8] [--num_epochs 100] [--lr 0.001]
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to Python path so we can import modules
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.data_loading import create_data_loader, RNADataset
from src.models.rna_folding_model import RNAFoldingModel
from src.losses import compute_combined_loss
from src.utils.structure_metrics import compute_rmsd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RNA 3D Structure Prediction Training')
    
    # Data paths
    parser.add_argument('--sequences_csv', type=str, required=True,
                        help='Path to sequences CSV file')
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='Path to structure labels CSV file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing feature files (thermo, dihedral, mi)')
    parser.add_argument('--output_dir', type=str, default='./training_results',
                        help='Directory to save training results, logs, and checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory to save model checkpoints (defaults to output_dir/checkpoints)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Directory to save tensorboard logs (defaults to output_dir/logs)')
    
    # Loading existing model (optional)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint to resume training from')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    
    # Loss weights
    parser.add_argument('--fape_weight', type=float, default=1.0,
                        help='Weight for the FAPE loss')
    parser.add_argument('--confidence_weight', type=float, default=0.1,
                        help='Weight for the confidence prediction loss')
    parser.add_argument('--angle_weight', type=float, default=0.5,
                        help='Weight for the angle prediction loss')
    
    # Validation parameters
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--val_frequency', type=int, default=1,
                        help='Frequency (in epochs) to run validation')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs without improvement to trigger early stopping')
    
    # Dataset filtering
    parser.add_argument('--max_seq_length', type=int, default=200,
                        help='Maximum sequence length to consider for training')
    parser.add_argument('--min_seq_length', type=int, default=10,
                        help='Minimum sequence length to consider for training')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use (for quick testing)')
    
    # Device selection
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu). If None, use cuda if available.')
    
    # Debugging
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging and small subset of data')
    
    return parser.parse_args()


def load_model(config_path: Optional[str], checkpoint_path: Optional[str], device: torch.device) -> Tuple[RNAFoldingModel, Dict, int]:
    """
    Initialize the RNA folding model, optionally from a checkpoint.
    
    Args:
        config_path: Path to model configuration JSON file (optional)
        checkpoint_path: Path to model weights checkpoint file (optional)
        device: Device to load the model on
        
    Returns:
        Tuple of:
        - Initialized model
        - Model config dictionary
        - Starting epoch (0 for new models, next epoch for checkpoint models)
    """
    # Default configuration as fallback
    default_config = {
        "num_blocks": 4,
        "residue_embed_dim": 128,
        "pair_embed_dim": 64,
        "num_attention_heads": 4,
        "dropout": 0.1,
        "ffn_dim": 512,
        "max_relative_position": 32,
        "seq_embed_dim": 16,
        "num_embeddings": 5,  # A, C, G, U, padding
        "padding_idx": 4,
        "ipa_dim": 64,
        "max_len": 500,
        "use_conservation": False
    }
    
    # Initialize config
    config = None
    checkpoint = None
    start_epoch = 0
    
    # First, try to load checkpoint if provided (to get config from it)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logger.info(f"Loading model checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Extract config from checkpoint if available
            if isinstance(checkpoint, dict) and "model_config" in checkpoint:
                config = checkpoint["model_config"]
                logger.info("Using model configuration from checkpoint")
                
            # Extract starting epoch if available
            if isinstance(checkpoint, dict) and "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1  # Start from next epoch
                logger.info(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            checkpoint = None
    
    # If no config from checkpoint, try loading from config file
    if config is None and config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            logger.info("Using model configuration from config file")
    
    # Fall back to default config if still not set
    if config is None:
        config = default_config
        logger.info("Using default model configuration")
    
    # Create model with the determined config
    model = RNAFoldingModel(config).to(device)
    
    # Load state dict if available in checkpoint
    if checkpoint is not None:
        try:
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
                logger.info("Successfully loaded model weights from checkpoint")
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
                logger.info("Successfully loaded model weights from checkpoint")
            else:
                # Assume checkpoint is just the state_dict
                model.load_state_dict(checkpoint)
                logger.info("Successfully loaded model weights directly from checkpoint")
        except RuntimeError as e:
            logger.error(f"Error loading model weights: {e}")
            logger.warning("Using model with random initialization")
    else:
        logger.info("Using model with default initialization (no checkpoint loaded)")
    
    return model, config, start_epoch


def create_train_val_dataloaders(args):
    """
    Create training and validation data loaders.
    
    Args:
        args: Command-line arguments
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Load sequences data
    import pandas as pd
    import random
    
    # Read and filter sequences based on length
    sequences_df = pd.read_csv(args.sequences_csv)
    sequences_df['length'] = sequences_df['sequence'].apply(len)
    
    filtered_df = sequences_df[(sequences_df['length'] >= args.min_seq_length) & 
                              (sequences_df['length'] <= args.max_seq_length)]
    
    if args.debug:
        logger.debug(f"Filtered from {len(sequences_df)} to {len(filtered_df)} sequences with length between {args.min_seq_length} and {args.max_seq_length}")
    
    # Limit samples if max_samples is specified
    if args.max_samples is not None and len(filtered_df) > args.max_samples:
        if args.debug:
            logger.debug(f"Limiting dataset to {args.max_samples} random samples")
        filtered_df = filtered_df.sample(args.max_samples, random_state=42)
    
    # Split into train and validation sets
    if args.val_fraction > 0:
        # Stratified sampling based on sequence length to get good distribution
        # First, create sequence length bins (if possible)
        try:
            # Try to create bins, may fail if not enough unique values
            filtered_df['length_bin'] = pd.qcut(filtered_df['length'], 
                                                min(5, len(filtered_df['length'].unique())), 
                                                labels=False)
            
            # Now split each bin into train/val
            train_indices = []
            val_indices = []
            
            for bin_idx in filtered_df['length_bin'].unique():
                bin_df = filtered_df[filtered_df['length_bin'] == bin_idx]
                bin_indices = bin_df.index.tolist()
                random.Random(42).shuffle(bin_indices)  # Fixed seed for reproducibility
                
                # Split based on val_fraction (ensure at least 1 sample in train if possible)
                split_idx = max(1, int(len(bin_indices) * (1 - args.val_fraction)))
                # Handle corner case of having only 1 sample in bin
                split_idx = min(split_idx, len(bin_indices))
                
                train_indices.extend(bin_indices[:split_idx])
                if split_idx < len(bin_indices):  # Only add validation if we have samples left
                    val_indices.extend(bin_indices[split_idx:])
            
            train_df = filtered_df.loc[train_indices]
            val_df = filtered_df.loc[val_indices]
        except Exception as e:
            # Fallback to simple random split if stratified sampling fails
            logger.warning(f"Stratified sampling failed: {e}. Using random split.")
            # Simple random split with guaranteed minimum 1 training sample
            filtered_df = filtered_df.sample(frac=1, random_state=42)  # Shuffle
            split_idx = max(1, int(len(filtered_df) * (1 - args.val_fraction)))
            
            train_df = filtered_df.iloc[:split_idx]
            val_df = filtered_df.iloc[split_idx:]
        
        # Ensure we have at least one sample in each split
        if len(train_df) == 0 and len(filtered_df) > 0:
            # Move one sample from validation to training if necessary
            train_df = val_df.head(1)
            val_df = val_df.iloc[1:] if len(val_df) > 1 else pd.DataFrame(columns=filtered_df.columns)
        
        logger.info(f"Split dataset into {len(train_df)} training and {len(val_df)} validation samples")
    else:
        train_df = filtered_df
        val_df = pd.DataFrame(columns=filtered_df.columns)  # Empty DataFrame with same structure
        logger.info(f"Using all {len(train_df)} samples for training (no validation split)")
    
    # Custom split function for train data
    def train_split_fn(df):
        # Return only the sequences in our training set
        return df[df['target_id'].isin(train_df['target_id'])]
    
    # Custom split function for validation data
    def val_split_fn(df):
        # Return only the sequences in our validation set
        return df[df['target_id'].isin(val_df['target_id'])]
    
    # Create data loaders with our custom split functions
    train_loader = create_data_loader(
        sequences_csv_path=args.sequences_csv,
        labels_csv_path=args.labels_csv,
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        split_fn=train_split_fn,
        require_features=True,
    )
    
    # Create validation loader only if we have validation samples
    if len(val_df) > 0:
        val_loader = create_data_loader(
            sequences_csv_path=args.sequences_csv,
            labels_csv_path=args.labels_csv,
            features_dir=args.features_dir,
            batch_size=args.batch_size,
            shuffle=False,  # No need to shuffle validation data
            num_workers=args.num_workers,
            split_fn=val_split_fn,
            require_features=True,
        )
    else:
        # Create an empty validation loader (for consistency)
        val_loader = create_data_loader(
            sequences_csv_path=args.sequences_csv,
            labels_csv_path=args.labels_csv,
            features_dir=args.features_dir,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            split_fn=lambda df: df.head(0),  # Empty DataFrame
            require_features=True,
        )
    
    logger.info(f"Created data loaders with {len(train_loader.dataset)} training and {len(val_loader.dataset)} validation samples")
    
    return train_loader, val_loader


def validate_model(model, val_loader, loss_weights, device):
    """
    Validate the model on the validation dataset.
    
    Args:
        model: The RNA folding model
        val_loader: Validation data loader
        loss_weights: Dictionary of loss weights for combined loss
        device: Device to run validation on
        
    Returns:
        dict: Dictionary with validation metrics
    """
    # Check if validation loader has any samples
    if len(val_loader.dataset) == 0:
        logger.warning("No validation samples available. Skipping validation.")
        # Return default metrics
        return {
            "val_loss": float('nan'),
            "val_fape": float('nan'),
            "val_confidence": float('nan'),
            "val_angle": float('nan'),
            "val_rmsd": float('nan'),
            "rmsd_values": []
        }
    
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0.0
    total_val_fape = 0.0
    total_val_confidence = 0.0
    total_val_angle = 0.0
    rmsd_values = []
    
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch_idx, batch in enumerate(val_loader):
            # Move batch to device
            batch_on_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            
            # Run forward pass
            outputs = model(batch_on_device)
            
            # Compute loss
            loss, loss_components = compute_combined_loss(outputs, batch_on_device, loss_weights)
            
            # Accumulate losses
            total_val_loss += loss.item()
            total_val_fape += loss_components["fape"].item()
            total_val_confidence += loss_components["confidence"].item()
            total_val_angle += loss_components["angle"].item()
            
            # Calculate RMSD for each sample in the batch
            pred_coords = outputs["pred_coords"]
            true_coords = batch_on_device["coordinates"]
            mask = batch_on_device["mask"]
            
            batch_rmsd = compute_rmsd(
                pred_coords, true_coords, mask
            )
            
            # Collect RMSD values for each sample in batch
            for i in range(batch_rmsd.size(0)):
                rmsd_values.append(batch_rmsd[i].item())
    
    # Calculate mean metrics
    num_batches = len(val_loader)
    if num_batches > 0:
        mean_val_loss = total_val_loss / num_batches
        mean_val_fape = total_val_fape / num_batches
        mean_val_confidence = total_val_confidence / num_batches
        mean_val_angle = total_val_angle / num_batches
    else:
        mean_val_loss = float('nan')
        mean_val_fape = float('nan')
        mean_val_confidence = float('nan')
        mean_val_angle = float('nan')
        
    mean_rmsd = np.mean(rmsd_values) if rmsd_values else float('nan')
    
    # Return validation metrics
    return {
        "val_loss": mean_val_loss,
        "val_fape": mean_val_fape,
        "val_confidence": mean_val_confidence,
        "val_angle": mean_val_angle,
        "val_rmsd": mean_rmsd,
        "rmsd_values": rmsd_values
    }


def save_checkpoint(model, optimizer, epoch, val_metrics, config, loss_weights, checkpoint_path):
    """
    Save a model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        epoch: Current epoch number
        val_metrics: Validation metrics
        config: Model configuration
        loss_weights: Loss weights
        checkpoint_path: Path to save the checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_config": config,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_metrics": val_metrics,
        "loss_weights": loss_weights
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def train_epoch(model, train_loader, optimizer, loss_weights, device, clip_grad_norm=1.0):
    """
    Train the model for one epoch.
    
    Args:
        model: The RNA folding model
        train_loader: Training data loader
        optimizer: Optimizer
        loss_weights: Dictionary of loss weights for combined loss
        device: Device to train on
        clip_grad_norm: Maximum norm for gradient clipping
        
    Returns:
        dict: Dictionary with training metrics for this epoch
    """
    model.train()  # Set model to training mode
    total_train_loss = 0.0
    total_fape_loss = 0.0
    total_confidence_loss = 0.0
    total_angle_loss = 0.0
    
    # Progress bar for tracking training
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch_on_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v 
            for k, v in batch.items()
        }
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_on_device)
        
        # Compute loss
        loss, loss_components = compute_combined_loss(outputs, batch_on_device, loss_weights)
        
        # Backward pass and optimize
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        # Update weights
        optimizer.step()
        
        # Accumulate losses
        total_train_loss += loss.item()
        total_fape_loss += loss_components["fape"].item()
        total_confidence_loss += loss_components["confidence"].item()
        total_angle_loss += loss_components["angle"].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'fape': f"{loss_components['fape'].item():.4f}",
        })
    
    # Calculate mean losses
    num_batches = len(train_loader)
    mean_train_loss = total_train_loss / num_batches
    mean_fape_loss = total_fape_loss / num_batches
    mean_confidence_loss = total_confidence_loss / num_batches
    mean_angle_loss = total_angle_loss / num_batches
    
    # Return training metrics
    return {
        "train_loss": mean_train_loss,
        "train_fape": mean_fape_loss,
        "train_confidence": mean_confidence_loss,
        "train_angle": mean_angle_loss
    }


def main():
    """Main function that runs the training process."""
    args = parse_args()
    
    # Set up output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up checkpoint directory
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up log directory for tensorboard
    log_dir = args.log_dir or os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Configure logging to file
    log_file = os.path.join(args.output_dir, f"training_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    # Set up tensorboard writer
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, timestamp))
    
    # Set up device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Debug mode - use smaller dataset
    if args.debug:
        logger.setLevel(logging.DEBUG)
        if args.max_samples is None:
            args.max_samples = 100
        logger.debug(f"Debug mode enabled. Using max {args.max_samples} samples.")
    
    # Create data loaders
    train_loader, val_loader = create_train_val_dataloaders(args)
    
    # Initialize model, get config and starting epoch
    model, config, start_epoch = load_model(None, args.checkpoint, device)
    
    # Log model configuration
    logger.info(f"Model configuration: {json.dumps(config, indent=2)}")
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # If resuming training, load optimizer state
    if args.checkpoint is not None and start_epoch > 0:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("Loaded optimizer state from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")
    
    # Set up loss weights
    loss_weights = {
        "fape": args.fape_weight,
        "confidence": args.confidence_weight,
        "angle": args.angle_weight
    }
    
    # Log training parameters
    logger.info(f"Training parameters:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Gradient clipping: {args.clip_grad_norm}")
    logger.info(f"  Loss weights: {loss_weights}")
    logger.info(f"  Max epochs: {args.num_epochs}")
    
    # Track best validation loss for early stopping
    best_val_loss = float('inf')
    best_val_rmsd = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        # Update learning rate based on epoch
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} (lr={current_lr:.2e})")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_weights=loss_weights,
            device=device,
            clip_grad_norm=args.clip_grad_norm
        )
        
        # Log training metrics
        logger.info(f"  Train loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"  Train FAPE loss: {train_metrics['train_fape']:.4f}")
        logger.info(f"  Train confidence loss: {train_metrics['train_confidence']:.4f}")
        logger.info(f"  Train angle loss: {train_metrics['train_angle']:.4f}")
        
        # Add training metrics to tensorboard
        tb_writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
        tb_writer.add_scalar('FAPE/train', train_metrics['train_fape'], epoch)
        tb_writer.add_scalar('Confidence/train', train_metrics['train_confidence'], epoch)
        tb_writer.add_scalar('Angle/train', train_metrics['train_angle'], epoch)
        
        # Run validation if it's time
        if (epoch + 1) % args.val_frequency == 0:
            val_metrics = validate_model(
                model=model,
                val_loader=val_loader,
                loss_weights=loss_weights,
                device=device
            )
            
            # Log validation metrics
            logger.info(f"  Val loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val FAPE loss: {val_metrics['val_fape']:.4f}")
            logger.info(f"  Val confidence loss: {val_metrics['val_confidence']:.4f}")
            logger.info(f"  Val angle loss: {val_metrics['val_angle']:.4f}")
            logger.info(f"  Val RMSD: {val_metrics['val_rmsd']:.4f} Å")
            
            # Add validation metrics to tensorboard
            tb_writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
            tb_writer.add_scalar('FAPE/val', val_metrics['val_fape'], epoch)
            tb_writer.add_scalar('Confidence/val', val_metrics['val_confidence'], epoch)
            tb_writer.add_scalar('Angle/val', val_metrics['val_angle'], epoch)
            tb_writer.add_scalar('RMSD/val', val_metrics['val_rmsd'], epoch)
            
            # Update learning rate scheduler based on validation loss
            scheduler.step(val_metrics['val_loss'])
            
            # Save checkpoint for every validation
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_metrics=val_metrics,
                config=config,
                loss_weights=loss_weights,
                checkpoint_path=checkpoint_path
            )
            
            # Check for best model (based on validation loss)
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_val_rmsd = val_metrics['val_rmsd']
                best_epoch = epoch
                
                # Save best model
                best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    config=config,
                    loss_weights=loss_weights,
                    checkpoint_path=best_model_path
                )
                
                logger.info(f"  New best model saved (val_loss: {best_val_loss:.4f}, val_rmsd: {best_val_rmsd:.4f} Å)")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                logger.info(f"  No improvement for {epochs_without_improvement} epochs (best: {best_val_loss:.4f} at epoch {best_epoch+1})")
            
            # Early stopping check
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                break
        
        # Always save latest model at the end of epoch
        latest_model_path = os.path.join(checkpoint_dir, "latest_model.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_metrics=val_metrics if ((epoch + 1) % args.val_frequency == 0 and 'val_metrics' in locals()) else {},
            config=config,
            loss_weights=loss_weights,
            checkpoint_path=latest_model_path
        )
    
    # Training finished
    logger.info("Training finished!")
    if best_epoch >= 0:
        logger.info(f"Best model at epoch {best_epoch+1}: val_loss={best_val_loss:.4f}, val_rmsd={best_val_rmsd:.4f} Å")
    
    # Close tensorboard writer
    tb_writer.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())