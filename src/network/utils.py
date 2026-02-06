"""Utility functions for training and evaluation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# Loss Functions
# ============================================================================

def dice_loss(y_pred, y_true, smooth=1e-6):
    """Dice loss for binary segmentation.
    
    Args:
        y_pred: Predicted mask (batch, 1, H, W)
        y_true: Ground truth mask (batch, 1, H, W)
        smooth: Smoothing factor
        
    Returns:
        Dice loss value
    """
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)
    
    intersection = (y_pred_f * y_true_f).sum()
    dice = (2. * intersection + smooth) / (y_pred_f.sum() + y_true_f.sum() + smooth)
    
    return 1 - dice


def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance.
    
    Args:
        y_pred: Predicted mask (batch, 1, H, W)
        y_true: Ground truth mask (batch, 1, H, W)
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss value
    """
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)
    
    # Calculate focal loss
    pt_1 = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
    pt_0 = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
    
    loss = -torch.mean(
        alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)
    ) - torch.mean(
        (1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0)
    )
    
    return loss


def combined_loss(y_pred, y_true, bce_weight=0.5, dice_weight=0.5):
    """Combined BCE and Dice loss.
    
    Args:
        y_pred: Predicted mask (batch, 1, H, W)
        y_true: Ground truth mask (batch, 1, H, W)
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        
    Returns:
        Combined loss value
    """
    bce = F.binary_cross_entropy(y_pred, y_true)
    dice = dice_loss(y_pred, y_true)
    
    return bce_weight * bce + dice_weight * dice


# ============================================================================
# Metrics
# ============================================================================

class DiceCoefficient:
    """Dice coefficient metric for segmentation."""
    
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.intersection = 0.0
        self.union = 0.0
    
    def update(self, y_pred, y_true):
        """Update metric with batch predictions.
        
        Args:
            y_pred: Predicted mask (batch, 1, H, W)
            y_true: Ground truth mask (batch, 1, H, W)
        """
        y_pred = (y_pred > 0.5).float()
        y_true = y_true.float()
        
        intersection = (y_pred * y_true).sum().item()
        union = y_pred.sum().item() + y_true.sum().item()
        
        self.intersection += intersection
        self.union += union
    
    def compute(self):
        """Compute dice coefficient."""
        return (2. * self.intersection + self.smooth) / (self.union + self.smooth)


class IoU:
    """Intersection over Union (IoU) metric."""
    
    def __init__(self, smooth=1e-6):
        self.smooth = smooth
        self.reset()
    
    def reset(self):
        """Reset metric state."""
        self.intersection = 0.0
        self.union = 0.0
    
    def update(self, y_pred, y_true):
        """Update metric with batch predictions.
        
        Args:
            y_pred: Predicted mask (batch, 1, H, W)
            y_true: Ground truth mask (batch, 1, H, W)
        """
        y_pred = (y_pred > 0.5).float()
        y_true = y_true.float()
        
        intersection = (y_pred * y_true).sum().item()
        union = y_pred.sum().item() + y_true.sum().item() - intersection
        
        self.intersection += intersection
        self.union += union
    
    def compute(self):
        """Compute IoU."""
        return (self.intersection + self.smooth) / (self.union + self.smooth)


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_tile(features, labels, config):
    """Apply data augmentation to a single tile.
    
    Args:
        features: Input features (C, H, W) as torch tensor
        labels: Ground truth mask (1, H, W) as torch tensor
        config: NetworkConfig instance
        
    Returns:
        Augmented features and labels
    """
    # Random horizontal flip
    if config.FLIP_HORIZONTAL:
        if torch.rand(1).item() > 0.5:
            features = torch.flip(features, dims=[2])
            labels = torch.flip(labels, dims=[2])
    
    # Random vertical flip
    if config.FLIP_VERTICAL:
        if torch.rand(1).item() > 0.5:
            features = torch.flip(features, dims=[1])
            labels = torch.flip(labels, dims=[1])
    
    # Random 90-degree rotations
    if config.ROTATE_90:
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            features = torch.rot90(features, k, dims=[1, 2])
            labels = torch.rot90(labels, k, dims=[1, 2])
    
    # Random brightness adjustment (only on features)
    if config.BRIGHTNESS_DELTA > 0:
        delta = torch.rand(1).item() * config.BRIGHTNESS_DELTA * 2 - config.BRIGHTNESS_DELTA
        features = features + delta
        features = torch.clamp(features, 0, 1)
    
    # Random contrast adjustment (only on features)
    if config.CONTRAST_RANGE is not None:
        factor = torch.rand(1).item() * (config.CONTRAST_RANGE[1] - config.CONTRAST_RANGE[0]) + config.CONTRAST_RANGE[0]
        mean = features.mean(dim=(1, 2), keepdim=True)
        features = (features - mean) * factor + mean
        features = torch.clamp(features, 0, 1)
    
    return features, labels


# ============================================================================
# Visualization
# ============================================================================

def visualize_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    band_indices: Tuple[int, int, int] = (3, 2, 1),  # NIR, Red, Green for false color
    num_samples: int = 4,
    save_path: Optional[Path] = None
):
    """Visualize model predictions.
    
    Args:
        features: Input features (N, C, H, W) or (N, H, W, C)
        labels: Ground truth masks (N, 1, H, W) or (N, H, W)
        predictions: Predicted masks (N, 1, H, W)
        band_indices: Which bands to use for RGB visualization
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    num_samples = min(num_samples, len(features))
    
    # Convert from (N, C, H, W) to (N, H, W, C) if needed
    if features.ndim == 4 and features.shape[1] < features.shape[2]:
        features = np.transpose(features, (0, 2, 3, 1))
    
    # Handle predictions shape
    if predictions.ndim == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0, :, :]  # (N, H, W)
    
    # Handle labels shape
    if labels.ndim == 3:
        pass  # Already (N, H, W)
    elif labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0, :, :]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        # RGB composite (normalized)
        # Extract selected bands: features[i] is (H, W, C)
        rgb = np.stack([features[i, :, :, b] for b in band_indices], axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title('False Color Composite')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(labels[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred_mask = predictions[i]
        axes[i, 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
        
        # Overlay
        axes[i, 3].imshow(rgb)
        axes[i, 3].imshow(pred_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
        axes[i, 3].set_title('Prediction Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def plot_training_history(history: dict, save_path: Optional[Path] = None):
    """Plot training history.
    
    Args:
        history: Dict with training metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    if 'loss' in history:
        axes[0, 0].plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice coefficient
    if 'dice_coefficient' in history:
        axes[0, 1].plot(history['dice_coefficient'], label='Train Dice')
    if 'val_dice_coefficient' in history:
        axes[0, 1].plot(history['val_dice_coefficient'], label='Val Dice')
    if 'dice_coefficient' in history or 'val_dice_coefficient' in history:
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Coefficient')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    if 'iou' in history:
        axes[1, 0].plot(history['iou'], label='Train IoU')
    if 'val_iou' in history:
        axes[1, 0].plot(history['val_iou'], label='Val IoU')
    if 'iou' in history or 'val_iou' in history:
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('Intersection over Union')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


# ============================================================================
# Data Processing
# ============================================================================

def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """Normalize features to [0, 1] range.
    
    Args:
        features: Input features (C, H, W)
        
    Returns:
        Normalized features
    """
    # Simple min-max normalization per band
    min_val = features.min()
    max_val = features.max()
    
    normalized = (features - min_val) / (max_val - min_val + 1e-8)
    return normalized


def prepare_for_training(features, labels):
    """Prepare data for training with PyTorch.
    
    Args:
        features: Input features (H, W, C) as numpy array
        labels: Ground truth mask (H, W) as numpy array
        
    Returns:
        Processed features and labels as torch tensors (C, H, W) and (1, H, W)
    """
    # Convert to torch tensors
    features = torch.from_numpy(features).float()
    labels = torch.from_numpy(labels).float()
    
    # Transpose features from (H, W, C) to (C, H, W)
    if features.ndim == 3:
        features = features.permute(2, 0, 1)
    
    # Add channel dimension to labels if needed
    if labels.ndim == 2:
        labels = labels.unsqueeze(0)  # (1, H, W)
    
    return features, labels
