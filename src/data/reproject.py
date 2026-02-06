"""Reprojection worker to convert downloaded GeoTIFFs to memory-mapped PyTorch format."""

import time
import logging
from typing import Optional, List, Tuple
from pathlib import Path
import warnings
import json

import numpy as np
import torch
import xarray as xr
import rioxarray as rxr
from odc.geo.geobox import GeoBox, GeoboxTiles, geobox_union_conservative
from odc.geo.geom import Geometry
from odc.geo.xr import rasterize
from shapely.geometry import shape
import rasterio.env

from .database import DownloadDatabase
from .config import Config

# Silence GDAL warnings about unsupported warp options
warnings.filterwarnings('ignore', message='.*CPLE_NotSupported.*warp options.*')

# Suppress rasterio._env logger warnings about DTYPE
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class ReprojectionWorker:
    """Worker to reproject downloaded files to memory-mapped format."""
    
    def __init__(self, db: DownloadDatabase, config: Optional[Config] = None, countries: Optional[List[str]] = None, mining_file: Optional[str] = None):
        """Initialize reprojection worker.
        
        Args:
            db: Database instance
            config: Configuration instance
            countries: Optional list of ISO3 country codes to filter tasks
            mining_file: Unused (kept for backward compatibility)
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.worker_name = "reprojection"
        
        # Setup world geobox
        self.world_geobox = GeoBox.from_bbox(
            [-180, -90, 180, 90],
            resolution=self.config.WORLD_GEOBOX_RESOLUTION,
            crs=4326
        )
        self.world_geobox_tiles = GeoboxTiles(
            self.world_geobox, 
            self.config.WORLD_GEOBOX_TILE_SIZE
        )
        
        # Setup MMAP output directory
        self.mmap_path = self.config.DATA_DIR / "landsat_mmap"
        self.mmap_path.mkdir(exist_ok=True, parents=True)
        
        # Bands to process
        self.bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
        
        logger.info(f"Initialized reprojection worker with MMAP output: {self.mmap_path}")
    
    def reproject_to_mmap(self, task_data: dict) -> bool:
        """Reproject downloaded file to grid and save as memory-mapped PyTorch tensors.
        
        Reprojects entire cluster once, then extracts and stores individual tiles for efficiency.
        
        Args:
            task_data: Task data from database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            geometry_hash = task_data['geometry_hash']
            year = task_data['year']
            country_code = task_data['country_code']
            cluster_id = task_data.get('cluster_id', 0)
            filepath = Path(task_data['local_filepath'])
            
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return False
            
            # Get tiles for this task
            tiles = self.db.get_tiles_for_task(geometry_hash, year)
            
            if not tiles:
                logger.warning(f"No tiles found for {geometry_hash} {year}")
                return False
            
            # Load the image
            logger.info(f"Processing {len(tiles)} tiles to MMAP format for {country_code} {year}")
            image = rxr.open_rasterio(filepath)
            
            # Get mining footprint from database
            mining_footprint_json = self.db.get_mining_footprint(geometry_hash, year)
            mining_geom = shape(mining_footprint_json) if mining_footprint_json else None
            
            logger.info(f"Starting reprojection: {len(tiles)} tiles from {filepath.name}")
            
            # Reconstruct world geobox
            world_geobox = GeoBox.from_bbox(
                [-180, -90, 180, 90],
                resolution=self.config.WORLD_GEOBOX_RESOLUTION,
                crs=4326
            )
            world_geobox_tiles = GeoboxTiles(
                world_geobox,
                tile_shape=self.config.WORLD_GEOBOX_TILE_SIZE
            )
            
            # Compute union geobox of all tiles
            tile_geoboxes = [world_geobox_tiles[tile['tile_ix'], tile['tile_iy']] for tile in tiles]
            union_geobox = geobox_union_conservative(tile_geoboxes)
            
            logger.info(f"Union geobox: {union_geobox.shape} covering {len(tiles)} tiles")
            
            # Reproject image to union geobox once
            reprojected = image.odc.reproject(union_geobox)
            logger.debug(f"Reprojected to union geobox")
            
            # Round coordinates for alignment
            reprojected.coords["latitude"] = reprojected.coords["latitude"].values.round(5)
            reprojected.coords["longitude"] = reprojected.coords["longitude"].values.round(5)
            
            # Convert to xarray Dataset with band names
            conversion_dict = {1: 'blue', 2: 'green', 3: 'red', 4: 'nir', 5: 'swir1', 6: 'swir2', 7: 'thermal'}
            reprojected = reprojected.to_dataset(dim="band").rename(conversion_dict)
            
            # Rasterize mining footprint for full union once
            mining_footprint_union = None
            if mining_geom:
                mining_footprint_union = rasterize(Geometry(mining_geom, crs=4326), union_geobox)
            
            # Extract and store individual tiles
            for tile in tiles:
                tile_ix = tile['tile_ix']
                tile_iy = tile['tile_iy']
                
                # Get tile geobox
                tile_geobox = world_geobox_tiles[tile_ix, tile_iy]
                bounds = tile_geobox.boundingbox
                
                # Extract band data for this tile from reprojected union
                band_arrays = [
                    reprojected[band].sel(
                        latitude=slice(bounds.top, bounds.bottom),
                        longitude=slice(bounds.left, bounds.right)
                    ).values
                    for band in self.bands
                ]
                
                # Stack bands to (H, W, C)
                features = np.stack(band_arrays, axis=-1).astype(np.float32)
                
                # Extract mining footprint for this tile
                if mining_footprint_union is not None:
                    labels = mining_footprint_union.sel(
                        latitude=slice(bounds.top, bounds.bottom),
                        longitude=slice(bounds.left, bounds.right)
                    ).values.astype(np.float32)
                else:
                    labels = np.zeros(tile_geobox.shape, dtype=np.float32)
                
                if labels.ndim == 2:
                    labels = labels[..., np.newaxis]
                
                # Replace NaN with 0
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Convert to torch tensors: (H, W, C) -> (C, H, W)
                features_tensor = torch.from_numpy(features).float().permute(2, 0, 1)
                labels_tensor = torch.from_numpy(labels).float().permute(2, 0, 1)
                
                # Create tile directory: cluster_id/year/tile_ix_tile_iy/
                tile_dir = self.mmap_path / str(cluster_id) / str(year) / f"{tile_ix}_{tile_iy}"
                tile_dir.mkdir(parents=True, exist_ok=True)
                
                # Save tensors
                torch.save(features_tensor, tile_dir / "features.pt")
                torch.save(labels_tensor, tile_dir / "labels.pt")
                
                # Save metadata
                metadata = {
                    "tile_ix": tile_ix,
                    "tile_iy": tile_iy,
                    "year": year,
                    "cluster_id": cluster_id,
                    "geometry_hash": geometry_hash,
                    "features_shape": list(features_tensor.shape),
                    "labels_shape": list(labels_tensor.shape),
                    "bands": self.bands,
                    "dtype": "float32"
                }
                
                with open(tile_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Mark tile as written in database with correct cluster_id
                self.db.mark_tile_mmap_written(tile_ix, tile_iy, geometry_hash, year, cluster_id)
            
            logger.info(f"✓ Processed and saved {len(tiles)} tiles to MMAP format")
            
            # Close the image file handle before deleting
            image.close()
            
            # Delete local file after reprojection
            try:
                filepath.unlink()
                logger.info(f"  Deleted local file: {filepath}")
            except Exception as e:
                logger.warning(f"  Could not delete {filepath}: {e}")
            
            # Update database
            self.db.update_task_status(
                geometry_hash, year, self.config.STATUS_REPROJECTED,
            )
            
            self.db.increment_worker_counter(self.worker_name, "tasks_processed")
            return True
            
        except Exception as e:
            logger.error(f"Error reprojecting file: {e}", exc_info=True)
            self.db.increment_worker_counter(self.worker_name, "errors")
            return False
    
    def run(self, continuous: bool = True):
        """Run the reprojection worker.
        
        Args:
            continuous: If True, run continuously; otherwise run once
        """
        logger.info(f"Starting {self.worker_name} worker")
        
        while True:
            self.db.update_worker_heartbeat(self.worker_name, "running")
            
            # Get downloaded tasks
            downloaded_tasks = self.db.get_tasks_by_status(
                self.config.STATUS_DOWNLOADED,
                limit=self.config.BATCH_SIZE
            )
            
            # Filter by countries if specified
            if self.countries and downloaded_tasks:
                downloaded_tasks = [t for t in downloaded_tasks if t['country_code'] in self.countries]
            
            if downloaded_tasks:
                logger.info(f"Processing {len(downloaded_tasks)} downloaded tasks")
                for task in downloaded_tasks:
                    self.reproject_to_mmap(task)
            else:
                logger.debug("No downloaded tasks to reproject")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")