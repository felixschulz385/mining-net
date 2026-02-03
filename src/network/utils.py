"""Utility functions for training and evaluation."""

import tensorflow as tf
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# Loss Functions
# ============================================================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice loss for binary segmentation.
    
    Args:
        y_true: Ground truth mask (batch, H, W, 1)
        y_pred: Predicted mask (batch, H, W, 1)
        smooth: Smoothing factor
        
    Returns:
        Dice loss value
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    return 1 - dice


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss value
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # Calculate focal loss
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    
    loss = -tf.reduce_mean(
        alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)
    ) - tf.reduce_mean(
        (1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)
    )
    
    return loss


def combined_loss(y_true, y_pred, bce_weight=0.5, dice_weight=0.5):
    """Combined BCE and Dice loss.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        
    Returns:
        Combined loss value
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    dice = dice_loss(y_true, y_pred)
    
    return bce_weight * bce + dice_weight * dice


# ============================================================================
# Metrics
# ============================================================================

class DiceCoefficient(tf.keras.metrics.Metric):
    """Dice coefficient metric for segmentation."""
    
    def __init__(self, name='dice_coefficient', smooth=1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
    
    def result(self):
        return (2. * self.intersection + self.smooth) / (self.union + self.smooth)
    
    def reset_state(self):
        self.intersection.assign(0.)
        self.union.assign(0.)


class IoU(tf.keras.metrics.Metric):
    """Intersection over Union (IoU) metric."""
    
    def __init__(self, name='iou', smooth=1e-6, **kwargs):
        super().__init__(name=name, **kwargs)
        self.smooth = smooth
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)
    
    def result(self):
        return (self.intersection + self.smooth) / (self.union + self.smooth)
    
    def reset_state(self):
        self.intersection.assign(0.)
        self.union.assign(0.)


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_tile(features, labels, config):
    """Apply data augmentation to a single tile.
    
    Args:
        features: Input features (H, W, C)
        labels: Ground truth mask (H, W, 1) or (H, W)
        config: NetworkConfig instance
        
    Returns:
        Augmented features and labels with consistent shapes
    """
    # Ensure labels have channel dimension
    if len(labels.shape) == 2:
        labels = labels[..., tf.newaxis]
    
    # Random horizontal flip
    if config.FLIP_HORIZONTAL:
        if tf.random.uniform(()) > 0.5:
            features = tf.image.flip_left_right(features)
            labels = tf.image.flip_left_right(labels)
    
    # Random vertical flip
    if config.FLIP_VERTICAL:
        if tf.random.uniform(()) > 0.5:
            features = tf.image.flip_up_down(features)
            labels = tf.image.flip_up_down(labels)
    
    # Random 90-degree rotations
    if config.ROTATE_90:
        k = tf.random.uniform((), 0, 4, dtype=tf.int32)
        features = tf.image.rot90(features, k)
        labels = tf.image.rot90(labels, k)
    
    # Random brightness adjustment (only on features)
    if config.BRIGHTNESS_DELTA > 0:
        features = tf.image.random_brightness(features, config.BRIGHTNESS_DELTA)
    
    # Random contrast adjustment (only on features)
    if config.CONTRAST_RANGE is not None:
        features = tf.image.random_contrast(
            features,
            config.CONTRAST_RANGE[0],
            config.CONTRAST_RANGE[1]
        )
    
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
        features: Input features (N, H, W, C)
        labels: Ground truth masks (N, H, W)
        predictions: Predicted masks (N, H, W, 1)
        band_indices: Which bands to use for RGB visualization
        num_samples: Number of samples to visualize
        save_path: Path to save figure
    """
    num_samples = min(num_samples, len(features))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_samples):
        # RGB composite (normalized)
        rgb = features[i, :, :, band_indices]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title('False Color Composite')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(labels[i], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred_mask = predictions[i, :, :, 0]
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


def plot_training_history(history, save_path: Optional[Path] = None):
    """Plot training history.
    
    Args:
        history: Keras History object or dict with training metrics
        save_path: Path to save figure
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
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

def normalize_features(features: tf.Tensor) -> tf.Tensor:
    """Normalize features to [0, 1] range.
    
    Args:
        features: Input features
        
    Returns:
        Normalized features
    """
    # Simple min-max normalization per band
    min_val = tf.reduce_min(features, axis=[0, 1], keepdims=True)
    max_val = tf.reduce_max(features, axis=[0, 1], keepdims=True)
    
    normalized = (features - min_val) / (max_val - min_val + 1e-8)
    return normalized


def prepare_for_training(features, labels):
    """Prepare data for training.
    
    Args:
        features: Input features (H, W, C)
        labels: Ground truth mask (H, W)
        
    Returns:
        Processed features and labels
    """
    # Ensure correct dtypes
    features = tf.cast(features, tf.float32)
    labels = tf.cast(labels, tf.float32)
    
    # Add channel dimension to labels if needed
    if len(labels.shape) == 2:
        labels = labels[..., tf.newaxis]
    
    return features, labels
