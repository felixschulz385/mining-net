"""Configuration for mining segmentation downloader."""

import os
from pathlib import Path
from typing import Optional

from ..config import Config as BaseConfig


def _get_base_dir():
    """Get base directory from environment variable or local settings.
    
    Device-specific paths should not be committed to git.
    """
    # Try environment variable first
    env_path = os.getenv('MINING_NET_BASE_DIR')
    if env_path:
        return Path(env_path)
    
    # Try to import local settings (one directory up from src/data/)
    try:
        from ..local_settings import BASE_DIR as local_base_dir
        return Path(local_base_dir)
    except ImportError:
        pass
    
    # Fallback for HPC/scicore
    return Path("/scicore/home/meiera/schulz0022/projects/mining-net")


class Config(BaseConfig):
    """Configuration settings."""
    
    # Override BASE_DIR with data-specific logic
    BASE_DIR = _get_base_dir()
    DATA_DIR = BASE_DIR / "data"
    DB_PATH = DATA_DIR / "mining_segmentation.db"
    
    # Google Earth Engine
    GEE_PROJECT = "ee-growthandheat"
    GEE_COLLECTION = "LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL"
    GEE_SCALE = 30  # meters
    GEE_CRS = "EPSG:4326"
    GEE_MAX_PIXELS = 1e13
    
    # Google Drive
    DRIVE_FOLDER = "landsat_mining"
    DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
    
    # Local paths (device-specific, respects environment and local_settings.py)
    DOWNLOAD_DIR = DATA_DIR / "downloads"
    ARCHIVE_DIR = DATA_DIR / "archives"
    MINING_FILE = DATA_DIR / "global_mining_polygons_v2.gpkg"  # Path to mining polygons GeoPackage
    CREDENTIALS_FILE = BASE_DIR / "secrets" / "client_secret.json"
    TOKEN_FILE = BASE_DIR / "secrets" / "token.pickle"
    
    # HPC/Scicore settings
    HPC_HOST = "transfer12.scicore.unibas.ch"
    HPC_USER = "schulz0022"
    HPC_BASE_PATH = "/scicore/home/meiera/schulz0022/projects/mining-net"
    HPC_DATA_PATH = f"{HPC_BASE_PATH}/data_nobackup"
    HPC_ZARR_PATH = f"{HPC_DATA_PATH}/landsat_mmap"
    HPC_BACKUP_PATH = f"{HPC_DATA_PATH}/backups"
    SSH_KEY = Path.home() / ".ssh" / "id_ed25519_scicore"
    
    # Worker settings
    WORKER_SLEEP_INTERVAL = 30  # seconds
    BATCH_SIZE = 10  # tasks per batch
    COMPRESSION_BATCH_SIZE = 100  # files per zip
    MAX_RETRIES = 3
    MAX_SUBMITTED_TASKS = 10  # maximum number of tasks submitted to GEE
    
    # Geometry settings
    BUFFER_SIZE = 0.01  # degrees
    
    # Geobox settings (world grid aligned to 30m Landsat)
    WORLD_GEOBOX_RESOLUTION = 0.000269495  # ~30m at equator
    WORLD_GEOBOX_TILE_SIZE = [256, 256]  # tiles per dimension
    
    # Ground truth year - only this year has labels
    GROUND_TRUTH_YEAR = 2019  # Year with available ground truth labels
    
    # Task status constants
    STATUS_PENDING = "pending"
    STATUS_SUBMITTED = "submitted"
    STATUS_COMPLETED = "completed"
    STATUS_DOWNLOADED = "downloaded"
    STATUS_FAILED = "failed"
    
    # Legacy statuses (deprecated, kept for backward compatibility)
    STATUS_STORED = "stored"  # Legacy: stored in Zarr format (deprecated)
    STATUS_REPROJECTED = "reprojected"  # Old MMAP format (deprecated)
    STATUS_COMPRESSED = "compressed"  # Deprecated
    STATUS_UPLOADED = "uploaded"  # Deprecated
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        super().ensure_dirs()
        cls.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
