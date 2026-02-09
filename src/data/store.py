"""Storage worker to convert downloaded GeoTIFFs to Zarr format.

This worker replaces the old reprojection worker. Instead of creating memory-mapped
PyTorch files (.pt), it creates a single large Zarr group with all tiles.

The worker:
1. Loads downloaded GeoTIFF files
2. Reprojects to the world geobox grid
3. Appends tiles to global Zarr arrays (features, labels, indices)
4. Maintains index arrays for cluster_id, tile_ix, tile_iy, year
5. Uses chunking for efficient batch access
"""

import time
import logging
from typing import Optional, List
from pathlib import Path
import warnings
import json

import numpy as np
import zarr
import xarray as xr
import rioxarray as rxr
from odc.geo.geobox import GeoBox, GeoboxTiles, geobox_union_conservative
from odc.geo.geom import Geometry
from odc.geo.xr import rasterize
from shapely.geometry import shape

from .database import DownloadDatabase
from .config import Config

# Silence GDAL warnings about unsupported warp options
warnings.filterwarnings('ignore', message='.*CPLE_NotSupported.*warp options.*')

# Suppress rasterio._env logger warnings about DTYPE
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class StorageWorker:
    """Worker to store downloaded files in Zarr format."""
    
    def __init__(self, db: DownloadDatabase, config: Optional[Config] = None, countries: Optional[List[str]] = None, mining_file: Optional[str] = None):
        """Initialize storage worker.
        
        Args:
            db: Database instance
            config: Configuration instance
            countries: Optional list of ISO3 country codes to filter tasks
            mining_file: Unused (kept for backward compatibility)
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.worker_name = "storage"
        
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
        
        # Setup Zarr output directory
        self.zarr_path = self.config.DATA_DIR / "landsat_zarr"
        self.zarr_path.mkdir(exist_ok=True, parents=True)
        
        # Bands to process
        self.bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
        
        # Initialize or open Zarr group
        self._init_zarr_store()
        
        logger.info(f"Initialized storage worker with Zarr output: {self.zarr_path}")
    
    def _init_zarr_store(self):
        """Initialize or open the global Zarr store with resizable arrays."""
        store_path = self.zarr_path / "data.zarr"
        metadata_path = self.zarr_path / "store_metadata.json"
        
        # Load or create metadata tracking current position
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.store_metadata = json.load(f)
        else:
            self.store_metadata = {
                "next_index": 0,
                "tile_size": self.config.WORLD_GEOBOX_TILE_SIZE,
                "n_bands": len(self.bands),
                "chunk_size": 8,
                "format_version": "1.0"
            }
            with open(metadata_path, 'w') as f:
                json.dump(self.store_metadata, f, indent=2)
        
        self.metadata_path = metadata_path
        tile_h, tile_w = self.config.WORLD_GEOBOX_TILE_SIZE
        n_bands = len(self.bands)
        chunk_size = self.store_metadata["chunk_size"]
        
        # Open or create Zarr group
        if store_path.exists():
            self.zarr_group = zarr.open_group(store=str(store_path), mode='r+')
            logger.info(f"Opened existing Zarr store at {store_path}")
        else:
            # Create new Zarr group with initial arrays
            self.zarr_group = zarr.open_group(store=str(store_path), mode='w')
            
            # Create resizable arrays with initial size
            initial_size = 1000  # Will grow as needed
            
            self.zarr_group.create_array(
                'features',
                shape=(initial_size, n_bands, tile_h, tile_w),
                chunks=(chunk_size, n_bands, tile_h, tile_w),
                dtype=np.float32,
                compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
            )
            
            self.zarr_group.create_array(
                'labels',
                shape=(initial_size, 1, tile_h, tile_w),
                chunks=(chunk_size, 1, tile_h, tile_w),
                dtype=np.float32,
                compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
            )
            
            # Index arrays
            self.zarr_group.create_array(
                'cluster_ids',
                shape=(initial_size,),
                chunks=(chunk_size * 1000,),  # Larger chunks for index arrays
                dtype=np.int64,
                compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
            )
            
            self.zarr_group.create_array(
                'tile_ix',
                shape=(initial_size,),
                chunks=(chunk_size * 1000,),
                dtype=np.int32,
                compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
            )
            
            self.zarr_group.create_array(
                'tile_iy',
                shape=(initial_size,),
                chunks=(chunk_size * 1000,),
                dtype=np.int32,
                compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
            )
            
            self.zarr_group.create_array(
                'years',
                shape=(initial_size,),
                chunks=(chunk_size * 1000,),
                dtype=np.int32,
                compressors=zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
            )
            
            logger.info(f"Created new Zarr store at {store_path}")
    
    def _ensure_capacity(self, n_new_tiles: int):
        """Ensure Zarr arrays have enough capacity for new tiles."""
        next_idx = self.store_metadata["next_index"]
        current_size = self.zarr_group['features'].shape[0]
        required_size = next_idx + n_new_tiles
        
        if required_size > current_size:
            # Resize to accommodate new tiles with some buffer
            new_size = max(required_size, int(current_size * 1.5))
            logger.info(f"Resizing Zarr arrays from {current_size} to {new_size}")
            
            for array_name in ['features', 'labels', 'cluster_ids', 'tile_ix', 'tile_iy', 'years']:
                array = self.zarr_group[array_name]
                new_shape = list(array.shape)
                new_shape[0] = new_size
                array.resize(tuple(new_shape))
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.store_metadata, f, indent=2)
    
    def store_to_zarr(self, task_data: dict) -> bool:
        """Store downloaded file to grid and save as Zarr arrays.
        
        Reprojects entire cluster once, then extracts and stores individual tiles to global Zarr.
        
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
            logger.info(f"Processing {len(tiles)} tiles to Zarr format for {country_code} {year}")
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
            
            # Ensure we have capacity for these tiles
            self._ensure_capacity(len(tiles))
            
            # Get starting index for this batch
            start_idx = self.store_metadata["next_index"]
            
            # Extract and store individual tiles to global Zarr arrays
            for i, tile in enumerate(tiles):
                tile_ix = tile['tile_ix']
                tile_iy = tile['tile_iy']
                current_idx = start_idx + i
                
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
                
                # Stack bands to (C, H, W)
                features = np.stack(band_arrays, axis=0).astype(np.float32)
                
                # Extract mining footprint for this tile
                if mining_footprint_union is not None:
                    labels = mining_footprint_union.sel(
                        latitude=slice(bounds.top, bounds.bottom),
                        longitude=slice(bounds.left, bounds.right)
                    ).values.astype(np.float32)
                else:
                    labels = np.zeros(tile_geobox.shape, dtype=np.float32)
                
                if labels.ndim == 2:
                    labels = labels[np.newaxis, ...]  # Add channel dimension (1, H, W)
                
                # Replace NaN with 0
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Write to global Zarr arrays at current index
                self.zarr_group['features'][current_idx] = features
                self.zarr_group['labels'][current_idx] = labels
                self.zarr_group['cluster_ids'][current_idx] = cluster_id
                self.zarr_group['tile_ix'][current_idx] = tile_ix
                self.zarr_group['tile_iy'][current_idx] = tile_iy
                self.zarr_group['years'][current_idx] = year
                
                # Mark tile as written in database
                self.db.mark_tile_stored(tile_ix, tile_iy, geometry_hash, year, cluster_id)
            
            # Update next index
            self.store_metadata["next_index"] = start_idx + len(tiles)
            self._save_metadata()
            
            logger.info(f"✓ Processed and saved {len(tiles)} tiles to Zarr (indices {start_idx} to {start_idx + len(tiles) - 1})")
            
            # Close the image file handle before deleting
            image.close()
            
            # Delete local file after storage
            try:
                filepath.unlink()
                logger.info(f"  Deleted local file: {filepath}")
            except Exception as e:
                logger.warning(f"  Could not delete {filepath}: {e}")
            
            # Update database - set local_filepath to NULL since file is deleted
            self.db.update_task_status(
                geometry_hash, year, self.config.STATUS_STORED,
                local_filepath=None
            )
            
            self.db.increment_worker_counter(self.worker_name, "tasks_processed")
            return True
            
        except Exception as e:
            logger.error(f"Error storing file: {e}", exc_info=True)
            self.db.increment_worker_counter(self.worker_name, "errors")
            return False
    
    def run(self, continuous: bool = True):
        """Run the storage worker.
        
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
                    self.store_to_zarr(task)
            else:
                logger.debug("No downloaded tasks to store")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")


# Deprecated alias for backward compatibility
ReprojectionWorker = StorageWorker