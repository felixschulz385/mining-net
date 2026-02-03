"""Configuration for mining segmentation downloader."""

from pathlib import Path
from typing import Optional


class Config:
    """Configuration settings."""
    
    # Google Earth Engine
    GEE_PROJECT = "ee-growthandheat"
    GEE_COLLECTION = "LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL"
    GEE_SCALE = 30  # meters
    GEE_CRS = "EPSG:4326"
    GEE_MAX_PIXELS = 1e13
    
    # Google Drive
    DRIVE_FOLDER = "landsat_mining"
    DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
    
    # Local paths
    BASE_DIR = Path("/Users/felixschulz/Library/CloudStorage/OneDrive-Personal/Dokumente/Job/UNI/Basel/Research/mining-net")
    DATA_DIR = BASE_DIR / "data"
    DOWNLOAD_DIR = DATA_DIR / "downloads"
    ARCHIVE_DIR = DATA_DIR / "archives"
    DB_PATH = DATA_DIR / "mining_segmentation.db"
    MINING_FILE = DATA_DIR / "global_mining_polygons_v2.gpkg"  # Path to mining polygons GeoPackage
    CREDENTIALS_FILE = BASE_DIR / "secrets" / "client_secret.json"
    TOKEN_FILE = BASE_DIR / "secrets" / "token.pickle"
    
    # Worker settings
    WORKER_SLEEP_INTERVAL = 30  # seconds
    BATCH_SIZE = 10  # tasks per batch
    COMPRESSION_BATCH_SIZE = 100  # files per zip
    MAX_RETRIES = 3
    MAX_SUBMITTED_TASKS = 100  # maximum number of tasks submitted to GEE
    
    # Geometry settings
    BUFFER_SIZE = 0.05  # degrees
    
    # Geobox settings (world grid aligned to 30m Landsat)
    WORLD_GEOBOX_RESOLUTION = 0.000269495  # ~30m at equator
    WORLD_GEOBOX_TILE_SIZE = [64, 64]  # tiles per dimension
    
    # Task status constants
    STATUS_PENDING = "pending"
    STATUS_SUBMITTED = "submitted"
    STATUS_COMPLETED = "completed"
    STATUS_DOWNLOADED = "downloaded"
    STATUS_REPROJECTED = "reprojected"
    STATUS_COMPRESSED = "compressed"
    STATUS_UPLOADED = "uploaded"
    STATUS_FAILED = "failed"
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
