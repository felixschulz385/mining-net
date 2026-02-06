"""Training script for mining segmentation UNet model."""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.unet import build_unet
from network.config import NetworkConfig
from network.utils import (
    combined_loss,
    dice_loss,
    focal_loss,
    DiceCoefficient,
    IoU,
    augment_tile,
    prepare_for_training,
    plot_training_history
)
from data.data_loader import MiningSegmentationDataLoader
from data.config import Config as DataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MiningSegmentationTrainer:
    """Trainer for mining segmentation model."""
    
    def __init__(
        self,
        network_config: Optional[NetworkConfig] = None,
        data_config: Optional[DataConfig] = None
    ):
        """Initialize trainer.
        
        Args:
            network_config: Network configuration
            data_config: Data configuration
        """
        self.network_config = network_config or NetworkConfig()
        self.data_config = data_config or DataConfig()
        
        # Create necessary directories
        self.network_config.create_directories()
        
        # Setup device
        if self.network_config.DEVICE == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif self.network_config.DEVICE == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("Using Apple Silicon GPU")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Model will be initialized in setup_model
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    def setup_model(self):
        """Build and setup the model."""
        logger.info("Building UNet model...")
        
        # Build model
        self.model = build_unet(
            in_channels=self.network_config.IN_CHANNELS,
            num_classes=self.network_config.NUM_CLASSES,
            filters_base=self.network_config.FILTERS_BASE,
            depth=self.network_config.DEPTH,
            dropout_rate=self.network_config.DROPOUT_RATE
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.network_config.LEARNING_RATE
        )
        
        # Setup learning rate scheduler (ReduceLROnPlateau is better for convergence)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.network_config.REDUCE_LR_FACTOR,
            patience=self.network_config.REDUCE_LR_PATIENCE,
            min_lr=self.network_config.MIN_LR
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def get_loss_function(self):
        """Get loss function based on configuration."""
        if self.network_config.LOSS_TYPE == 'bce':
            return nn.BCELoss()
        elif self.network_config.LOSS_TYPE == 'dice':
            return dice_loss
        elif self.network_config.LOSS_TYPE == 'focal':
            return lambda pred, true: focal_loss(
                pred, true,
                alpha=self.network_config.FOCAL_ALPHA,
                gamma=self.network_config.FOCAL_GAMMA
            )
        elif self.network_config.LOSS_TYPE == 'combined':
            return lambda pred, true: combined_loss(
                pred, true,
                bce_weight=self.network_config.BCE_WEIGHT,
                dice_weight=self.network_config.DICE_WEIGHT
            )
        else:
            raise ValueError(f"Unknown loss type: {self.network_config.LOSS_TYPE}")
    
    def create_datasets(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None
    ):
        """Create training and validation datasets.
        
        Args:
            countries: Filter by country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Creating datasets...")
        
        # Create full dataset with filters and normalization
        # auto_compute_stats=True will compute statistics once during init if not provided
        full_dataset = MiningSegmentationDataLoader(
            config=self.data_config,
            countries=countries,
            years=years,
            cluster_ids=cluster_ids,
            bands=self.network_config.INPUT_BANDS,
            normalize=self.network_config.NORMALIZE_INPUTS,
            band_means=self.network_config.BAND_MEANS,
            band_stds=self.network_config.BAND_STDS,
            auto_compute_stats=True,  # Compute once during init for speed
            stats_samples=100  # Use 100 samples for quick stats
        )
        
        logger.info(f"Found {len(full_dataset)} tiles")
        
        # Split into train and validation
        n_total = len(full_dataset)
        n_val = int(n_total * self.network_config.VALIDATION_SPLIT)
        n_train = n_total - n_val
        
        # Random split
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Training tiles: {len(train_dataset)}, Validation tiles: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.network_config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.network_config.NUM_WORKERS,
            pin_memory=self.network_config.PIN_MEMORY
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.network_config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.network_config.NUM_WORKERS,
            pin_memory=self.network_config.PIN_MEMORY
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, loss_fn, epoch):
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
            loss_fn: Loss function
            epoch: Current epoch number
            
        Returns:
            Dict of training metrics
        """
        self.model.train()
        
        epoch_loss = 0.0
        dice_metric = DiceCoefficient()
        iou_metric = IoU()
        total_grad_norm = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for features, labels in pbar:
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = loss_fn(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norm before clipping (for monitoring)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.network_config.GRADIENT_CLIP_VALUE if hasattr(self.network_config, 'GRADIENT_CLIP_VALUE') else float('inf')
            )
            total_grad_norm += grad_norm.item()
            num_batches += 1
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            dice_metric.update(predictions.detach(), labels)
            iou_metric.update(predictions.detach(), labels)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        metrics = {
            'loss': epoch_loss / len(train_loader),
            'dice_coefficient': dice_metric.compute(),
            'iou': iou_metric.compute(),
            'avg_grad_norm': total_grad_norm / num_batches if num_batches > 0 else 0.0
        }
        
        return metrics
    
    def validate(self, val_loader, loss_fn):
        """Validate the model.
        
        Args:
            val_loader: Validation data loader
            loss_fn: Loss function
            
        Returns:
            Dict of validation metrics
        """
        self.model.eval()
        
        val_loss = 0.0
        dice_metric = DiceCoefficient()
        iou_metric = IoU()
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc="Validating"):
                # Move to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                predictions = self.model(features)
                loss = loss_fn(predictions, labels)
                
                # Update metrics
                val_loss += loss.item()
                dice_metric.update(predictions, labels)
                iou_metric.update(predictions, labels)
        
        metrics = {
            'val_loss': val_loss / len(val_loader),
            'val_dice_coefficient': dice_metric.compute(),
            'val_iou': iou_metric.compute()
        }
        
        return metrics
    
    def train(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None,
        run_name: Optional[str] = None
    ):
        """Train the model.
        
        Args:
            countries: Filter by country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            run_name: Name for this training run
            
        Returns:
            Training history dict
        """
        # Setup model
        self.setup_model()
        
        # Create datasets
        train_loader, val_loader = self.create_datasets(countries, years, cluster_ids)
        
        # Get loss function
        loss_fn = self.get_loss_function()
        
        # Setup run name
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup tensorboard
        log_dir = self.network_config.LOG_DIR / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir)
        
        # Setup checkpointing
        checkpoint_dir = self.network_config.CHECKPOINT_DIR / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training: {run_name}")
        logger.info(f"Checkpoints: {checkpoint_dir}")
        logger.info(f"Logs: {log_dir}")
        
        # Training history
        history = {
            'loss': [],
            'dice_coefficient': [],
            'iou': [],
            'val_loss': [],
            'val_dice_coefficient': [],
            'val_iou': [],
            'lr': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.network_config.EPOCHS):
            # Train
            train_metrics = self.train_epoch(train_loader, loss_fn, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader, loss_fn)
            
            # Update history
            history['loss'].append(train_metrics['loss'])
            history['dice_coefficient'].append(train_metrics['dice_coefficient'])
            history['iou'].append(train_metrics['iou'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_dice_coefficient'].append(val_metrics['val_dice_coefficient'])
            history['val_iou'].append(val_metrics['val_iou'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log to tensorboard
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['val_loss'], epoch)
            writer.add_scalar('Dice/train', train_metrics['dice_coefficient'], epoch)
            writer.add_scalar('Dice/val', val_metrics['val_dice_coefficient'], epoch)
            writer.add_scalar('IoU/train', train_metrics['iou'], epoch)
            writer.add_scalar('IoU/val', val_metrics['val_iou'], epoch)
            writer.add_scalar('LearningRate', history['lr'][-1], epoch)
            writer.add_scalar('Diagnostics/avg_grad_norm', train_metrics.get('avg_grad_norm', 0), epoch)
            
            # Log to console
            logger.info(
                f"Epoch {epoch + 1}/{self.network_config.EPOCHS} - "
                f"Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Dice: {train_metrics['dice_coefficient']:.4f} - "
                f"Val Dice: {val_metrics['val_dice_coefficient']:.4f} - "
                f"GradNorm: {train_metrics.get('avg_grad_norm', 0):.4f} - "
                f"LR: {history['lr'][-1]:.2e}"
            )
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'history': history
            }, checkpoint_path)
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                best_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss
                }, best_path)
                logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.network_config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Learning rate scheduling (ReduceLROnPlateau needs val_loss)
            self.scheduler.step(val_metrics['val_loss'])
        
        writer.close()
        logger.info("Training completed!")
        
        # Plot training history
        plot_path = checkpoint_dir / "training_history.png"
        plot_training_history(history, save_path=plot_path)
        
        return history
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train UNet for mining segmentation')
    parser.add_argument(
        '--countries',
        nargs='+',
        help='Country codes to filter (e.g., ZAF USA)'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        help='Years to filter (e.g., 2019 2020)'
    )
    parser.add_argument(
        '--cluster-ids',
        nargs='+',
        type=int,
        help='Cluster IDs to filter'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        help='Name for this training run'
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = NetworkConfig()
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    # Initialize trainer
    trainer = MiningSegmentationTrainer(network_config=config)
    
    # Train model
    trainer.train(
        countries=args.countries,
        years=args.years,
        cluster_ids=args.cluster_ids,
        run_name=args.run_name
    )


if __name__ == "__main__":
    main()
