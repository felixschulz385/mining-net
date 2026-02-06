"""General configuration for mining-net project."""

from pathlib import Path


class Config:
    """General configuration settings for the mining-net project."""
    
    # Base paths
    BASE_DIR = Path("C:\\Users\\schulz0022\\Documents\\mining-net")
    DATA_DIR = BASE_DIR / "data"
    
    # Database
    DB_PATH = DATA_DIR / "mining_segmentation.db"
    
    # Memory-mapped data storage
    MMAP_DIR = DATA_DIR / "landsat_mmap"
    
    # Model paths
    MODELS_DIR = BASE_DIR / "models"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    
    # Logging
    LOGS_DIR = BASE_DIR / "logs"
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
