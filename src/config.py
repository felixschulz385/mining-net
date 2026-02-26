"""General configuration for mining-net project."""

import os
from pathlib import Path


def _get_base_dir():
    """Get base directory from environment variable or raise error.
    
    Device-specific paths should not be committed to git.
    Set MINING_NET_BASE_DIR environment variable or create local_settings.py.
    """
    # Try environment variable first
    env_path = os.getenv('MINING_NET_BASE_DIR')
    if env_path:
        return Path(env_path)
    
    # Try to import local settings
    try:
        from .local_settings import BASE_DIR as local_base_dir
        return Path(local_base_dir)
    except ImportError:
        pass
    
    # Raise informative error
    raise RuntimeError(
        "BASE_DIR not configured. Please set one of:\n"
        "  1. Environment variable: MINING_NET_BASE_DIR\n"
        "  2. Create src/local_settings.py with BASE_DIR = '/path/to/mining-net'\n"
        "\nNote: local_settings.py is gitignored for device-specific configuration."
    )


class Config:
    """General configuration settings for the mining-net project."""
    
    # Base paths (device-specific, not committed to git)
    BASE_DIR = _get_base_dir()
    DATA_DIR = BASE_DIR / "data_nobackup"
    
    # Database
    DB_PATH = DATA_DIR / "mining_segmentation.db"
    
    # Zarr data storage (unified group for all tiles)
    ZARR_DIR = DATA_DIR / "landsat_zarr"
    ZARR_STORE_PATH = ZARR_DIR / "data.zarr"
    
    # Legacy: Memory-mapped data storage (deprecated)
    MMAP_DIR = DATA_DIR / "landsat_mmap"
    
    # Model paths
    MODELS_DIR = BASE_DIR / "models"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    
    # Logging
    LOGS_DIR = BASE_DIR / "logs"

    # Compression
    COMPRESS_DOWNLOADS: bool = True
    COMPRESS_CODEC: str = "zstd"
    COMPRESS_LEVEL: int = 6
    COMPRESS_KEEP_RAW: bool = False
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
