"""Training configuration for mining segmentation."""

from pathlib import Path
from typing import Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


class NetworkConfig:
    """Configuration for network training."""
    
    # Model architecture (optimized for A100 40GB)
    IN_CHANNELS = 7  # Number of input channels (Landsat bands)
    INPUT_SIZE = 64  # Height and width of input tiles
    NUM_CLASSES = 1  # Binary segmentation
    FILTERS_BASE = 128  # Base number of filters in UNet (increased from 64 for A100)
    DEPTH = 5  # Number of pooling operations (increased from 4 for deeper model)
    DROPOUT_RATE = 0.2  # Increased dropout for larger model
    
    # Input bands
    INPUT_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
    LABEL_BAND = 'mining_footprint'
    
    # Input normalization (critical for convergence)
    # Set to None to compute from data, or provide precomputed values
    NORMALIZE_INPUTS = True
    # Example statistics (update with actual values from your data)
    BAND_MEANS = None  # Will be computed if None
    BAND_STDS = None   # Will be computed if None
    
    # Training parameters
    BATCH_SIZE = 64  # Increased from 32 for A100 40GB with AMP
    EPOCHS = 100
    LEARNING_RATE = 1e-3  # Increased for better initial convergence
    LEARNING_RATE_DECAY = 0.95  # Exponential decay rate
    DECAY_STEPS = 1000  # Steps between decay
    
    # Gradient clipping (prevents explosion)
    GRADIENT_CLIP_VALUE = 1.0  # Clip gradients to this max norm
    
    # Mixed precision training (critical for A100 performance)
    USE_AMP = True  # Automatic Mixed Precision (float16/bfloat16)
    AMP_DTYPE = 'bfloat16'  # 'float16' or 'bfloat16' (bfloat16 recommended for A100)
    
    # Gradient accumulation (for effective larger batch sizes)
    GRADIENT_ACCUMULATION_STEPS = 1  # Accumulate gradients over N steps
    # Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    
    # Data augmentation
    USE_AUGMENTATION = True
    FLIP_HORIZONTAL = True
    FLIP_VERTICAL = True
    ROTATE_90 = True
    BRIGHTNESS_DELTA = 0.1
    CONTRAST_RANGE = (0.8, 1.2)
    
    # Loss function weights
    LOSS_TYPE = 'combined'  # 'bce', 'dice', 'focal', 'combined'
    BCE_WEIGHT = 0.3  # Lower BCE weight (less sensitive to class imbalance)
    DICE_WEIGHT = 0.7  # Higher Dice weight (better for sparse targets)
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
    
    # Checkpointing (uses general config)
    CHECKPOINT_DIR = Config.CHECKPOINTS_DIR
    SAVE_BEST_ONLY = True
    
    # TensorBoard / Logging (uses general config)
    LOG_DIR = Config.LOGS_DIR
    
    # Device configuration
    DEVICE = 'cuda'  # 'cuda', 'cpu', or 'mps' (for Apple Silicon) - auto-detected in trainer
    NUM_WORKERS = 4  # DataLoader workers (set to 4 based on system recommendation)
    PIN_MEMORY = True  # Pin memory for faster GPU transfer
    PREFETCH_FACTOR = 2  # Number of batches to prefetch per worker
    PERSISTENT_WORKERS = True  # Keep workers alive between epochs (faster for large datasets)
    
    # Memory optimization
    EMPTY_CACHE_EVERY_N_STEPS = 100  # Clear CUDA cache periodically to prevent fragmentation
    
    # torch.compile optimization (PyTorch 2.0+)
    USE_COMPILE = False  # Enable torch.compile for faster training
    COMPILE_MODE = 'default'  # 'default', 'reduce-overhead', 'max-autotune'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories for training."""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
