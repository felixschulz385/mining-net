"""Mining segmentation network package.

This package provides a comprehensive UNet architecture and training pipeline
for semantic segmentation of mining areas from multi-spectral Landsat imagery.

Components:
    - unet: UNet model architecture
    - train: Training pipeline
    - evaluate: Evaluation and inference
    - config: Model configuration
    - utils: Utility functions for loss, metrics, and visualization
"""

from .unet import UNet, build_unet
from .config import NetworkConfig
from .utils import (
    dice_loss,
    focal_loss,
    combined_loss,
    DiceCoefficient,
    IoU,
    augment_tile,
    visualize_predictions,
    plot_training_history
)

__all__ = [
    'UNet',
    'build_unet',
    'NetworkConfig',
    'dice_loss',
    'focal_loss',
    'combined_loss',
    'DiceCoefficient',
    'IoU',
    'augment_tile',
    'visualize_predictions',
    'plot_training_history'
]

__version__ = '2.0.0'  # Updated for PyTorch
