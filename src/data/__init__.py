"""Mining segmentation data downloader for Landsat imagery.

This module provides a multi-worker system to download Landsat annual composites
for mining regions from Google Earth Engine with automatic clustering, 
geobox alignment, and zarr storage.
"""

from .config import Config
from .database import DownloadDatabase
from .clustering import create_clusters_and_tiles
from .janitor import JanitorWorker

__version__ = "0.1.0"

__all__ = [
    'Config',
    'DownloadDatabase',
    'create_clusters_and_tiles',
    'JanitorWorker',
]
