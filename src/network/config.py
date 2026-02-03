"""Training configuration for mining segmentation."""

from pathlib import Path
from typing import Optional


class NetworkConfig:
    """Configuration for network training."""
    
    # Model architecture
    INPUT_SHAPE = (64, 64, 7)  # (height, width, channels)
    NUM_CLASSES = 1  # Binary segmentation
    FILTERS_BASE = 64  # Base number of filters in UNet
    DEPTH = 2  # Number of pooling operations
    DROPOUT_RATE = 0.1
    
    # Input bands
    INPUT_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
    LABEL_BAND = 'mining_footprint'
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    LEARNING_RATE_DECAY = 0.95  # Exponential decay rate
    DECAY_STEPS = 1000  # Steps between decay
    
    # Data augmentation
    USE_AUGMENTATION = True
    FLIP_HORIZONTAL = True
    FLIP_VERTICAL = True
    ROTATE_90 = True
    BRIGHTNESS_DELTA = 0.1
    CONTRAST_RANGE = (0.8, 1.2)
    
    # Loss function weights
    LOSS_TYPE = 'combined'  # 'bce', 'dice', 'focal', 'combined'
    BCE_WEIGHT = 0.5
    DICE_WEIGHT = 0.5
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Class weighting for imbalanced data
    USE_CLASS_WEIGHTS = True
    POSITIVE_CLASS_WEIGHT = 10.0  # Higher weight for rare mining pixels
    
    # Validation split
    VALIDATION_SPLIT = 0.2
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 15
    REDUCE_LR_PATIENCE = 5
    REDUCE_LR_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # Checkpointing
    CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "models" / "checkpoints"
    SAVE_BEST_ONLY = True
    SAVE_WEIGHTS_ONLY = False
    
    # TensorBoard
    LOG_DIR = Path(__file__).parent.parent.parent / "logs"
    UPDATE_FREQ = 'epoch'
    
    # Mixed precision training (for faster training on GPUs with Tensor Cores)
    USE_MIXED_PRECISION = True
    
    # Multi-GPU training
    USE_MULTI_GPU = False
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for training."""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
