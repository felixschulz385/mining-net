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
import functools
import gc

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


def retry_on_permission_error(max_retries=5, initial_delay=0.5, backoff=2.0):
    """Decorator to retry operations on Windows permission errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except PermissionError as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(
                        f"PermissionError on attempt {attempt + 1}/{max_retries}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator


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
        
        # Initialize store structure (but don't keep group open)
        self._init_zarr_store()
        self.store_path = self.zarr_path / "data.zarr"
        
        logger.info(f"Initialized storage worker with Zarr output: {self.zarr_path}")
    
    def _init_zarr_store(self):
        """Initialize or open the global Zarr store with resizable arrays.
        
        Uses xarray to create the initial structure, then opens with plain zarr for writing.
        """
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
            # Open existing store with plain zarr
            self.zarr_group = zarr.open_group(store=str(store_path), mode='r+', zarr_format=3)
            logger.info(f"Opened existing Zarr store at {store_path}")
        else:
            # Create new store using xarray with empty arrays, then open with zarr
            logger.info(f"Creating new Zarr store at {store_path}")
            
            # Create xarray Dataset with proper structure using 0-length arrays
            ds = xr.Dataset(
                data_vars={
                    'features': (
                        ['tile', 'channel', 'y', 'x'],
                        np.zeros((0, n_bands, tile_h, tile_w), dtype=np.float32),
                        {'long_name': 'Landsat features', 'units': 'reflectance'}
                    ),
                    'labels': (
                        ['tile', 'label_channel', 'y', 'x'],
                        np.zeros((0, 1, tile_h, tile_w), dtype=np.float32),
                        {'long_name': 'Mining labels', 'units': 'binary'}
                    ),
                },
                coords={
                    'tile': np.array([], dtype=np.int64),
                    'channel': np.arange(n_bands),
                    'y': np.arange(tile_h),
                    'x': np.arange(tile_w),
                    'label_channel': [0],
                }
            )
            
            # Add non-dimension coordinates
            ds = ds.assign_coords(
                cluster_id=('tile', np.array([], dtype=np.int64)),
                tile_ix=('tile', np.array([], dtype=np.int32)),
                tile_iy=('tile', np.array([], dtype=np.int32)),
                year=('tile', np.array([], dtype=np.int32)),
            )
            
            # Add metadata
            ds.attrs['description'] = 'Mining segmentation dataset from Landsat imagery'
            ds.attrs['tile_shape'] = f"{tile_h}x{tile_w}"
            ds.attrs['n_channels'] = n_bands
            
            # Write to zarr with xarray (structure only, no computation)
            ds.to_zarr(
                str(store_path),
                mode='w',
                encoding={
                    'features': {
                        'chunks': (chunk_size, n_bands, tile_h, tile_w),
                        'compressor': zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
                    },
                    'labels': {
                        'chunks': (chunk_size, 1, tile_h, tile_w),
                        'compressor': zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
                    },
                    'cluster_id': {
                        'chunks': (chunk_size,),
                        'compressor': zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
                    },
                    'tile_ix': {
                        'chunks': (chunk_size,),
                        'compressor': zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
                    },
                    'tile_iy': {
                        'chunks': (chunk_size,),
                        'compressor': zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
                    },
                    'year': {
                        'chunks': (chunk_size,),
                        'compressor': zarr.codecs.BloscCodec(cname='zstd', clevel=0, shuffle="shuffle", blocksize=0),
                    },
                },
                consolidated=False,
                zarr_format=3,
                compute=False
            )
            
            logger.info(f"Created Zarr structure with xarray")
            # Don't keep group open - we'll open it per operation to avoid async issues
    
    def _open_zarr_group(self):
        """Open zarr group for operations. Opens fresh each time to avoid async shutdown issues."""
        return zarr.open_group(store=str(self.store_path), mode='r+', zarr_format=3)
    
    def _close_zarr_group(self, zarr_group):
        """Properly close a zarr group with synchronization and cleanup.
        
        Args:
            zarr_group: The zarr group to close
        """
        try:
            # Ensure all pending writes complete
            if hasattr(zarr_group.store, 'sync'):
                zarr_group.store.sync()
            
            # Close the store if it has a close method
            if hasattr(zarr_group.store, 'close'):
                zarr_group.store.close()
        except Exception as e:
            logger.warning(f"Error during zarr group cleanup: {e}")
        finally:
            # Delete reference to allow garbage collection
            del zarr_group
            # Force garbage collection to release handles
            gc.collect()
            # Give Windows time to release file locks (longer for async operations)
            time.sleep(0.5)
    
    @retry_on_permission_error(max_retries=5, initial_delay=1.0)
    def _ensure_capacity(self, n_new_tiles: int):
        """Ensure Zarr arrays have enough capacity for new tiles.
        
        Note: Resizing is done using plain zarr after xarray creates the initial structure.
        """
        next_idx = self.store_metadata["next_index"]
        
        # Open zarr group fresh for this operation
        zarr_group = self._open_zarr_group()
        
        try:
            current_size = zarr_group['features'].shape[0]
            required_size = next_idx + n_new_tiles
            
            if required_size > current_size:
                # Resize to accommodate new tiles with some buffer
                new_size = max(required_size, int(current_size * 1.5))
                logger.info(f"Resizing Zarr arrays from {current_size} to {new_size}")
                
                # Resize data arrays and coordinate arrays using plain zarr
                for array_name in ['features', 'labels', 'cluster_id', 'tile_ix', 'tile_iy', 'year', 'tile']:
                    array = zarr_group[array_name]
                    new_shape = list(array.shape)
                    new_shape[0] = new_size
                    array.resize(tuple(new_shape))
                
                # Update tile coordinate with new indices
                zarr_group['tile'][current_size:new_size] = np.arange(current_size, new_size)
        finally:
            # Properly close the zarr group
            self._close_zarr_group(zarr_group)
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.store_metadata, f, indent=2)
    
    @retry_on_permission_error(max_retries=5, initial_delay=1.0)
    def _write_tiles(self, start_idx, features_list, labels_list, cluster_ids, tile_ixs, tile_iys, years):
        """Write all tiles to zarr arrays at once.
        
        Args:
            start_idx: Starting index in zarr arrays
            features_list: List of feature arrays
            labels_list: List of label arrays
            cluster_ids: List of cluster IDs
            tile_ixs: List of tile x indices
            tile_iys: List of tile y indices
            years: List of years
        """
        zarr_group = self._open_zarr_group()
        
        try:
            # Stack all tiles into single arrays for batch writing
            n_tiles = len(features_list)
            end_idx = start_idx + n_tiles
            
            # Stack features: (n_tiles, C, H, W)
            features_batch = np.stack(features_list, axis=0)
            
            # Stack labels: (n_tiles, 1, H, W) 
            labels_batch = np.stack(labels_list, axis=0)
            
            # Convert metadata to arrays
            cluster_ids_batch = np.array(cluster_ids, dtype=np.int64)
            tile_ixs_batch = np.array(tile_ixs, dtype=np.int32)
            tile_iys_batch = np.array(tile_iys, dtype=np.int32)
            years_batch = np.array(years, dtype=np.int32)
            
            # Write all tiles at once to slices
            zarr_group['features'][start_idx:end_idx] = features_batch
            zarr_group['labels'][start_idx:end_idx] = labels_batch
            zarr_group['cluster_id'][start_idx:end_idx] = cluster_ids_batch
            zarr_group['tile_ix'][start_idx:end_idx] = tile_ixs_batch
            zarr_group['tile_iy'][start_idx:end_idx] = tile_iys_batch
            zarr_group['year'][start_idx:end_idx] = years_batch
            
            # Sync to ensure all writes complete
            if hasattr(zarr_group.store, 'sync'):
                zarr_group.store.sync()
            # Wait for async operations to finish
            time.sleep(0.5)
        finally:
            # Properly close the zarr group with synchronization
            self._close_zarr_group(zarr_group)
    
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
                logger.warning(
                    f"Downloaded file not found: {filepath}. "
                    f"Demoting task from DOWNLOADED to COMPLETED to re-download."
                )
                self.db.update_task_status(
                    cluster_id,
                    year,
                    self.config.STATUS_COMPLETED
                )
                return False
            
            # Get tiles for this cluster
            tiles = self.db.get_tiles_for_cluster(cluster_id)
            
            if not tiles:
                logger.warning(f"No tiles found for {geometry_hash} {year}")
                return False
            
            # Load the image
            logger.info(f"Processing {len(tiles)} tiles to Zarr format for {country_code} {year}")
            image = rxr.open_rasterio(filepath)
            
            # Get mining footprint from database (uses cluster_id)
            mining_footprint_json = self.db.get_mining_footprint(cluster_id)
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
            
            # Rasterize mining footprint only for ground truth year (optimization)
            mining_footprint_union = None
            if year == self.config.GROUND_TRUTH_YEAR and mining_geom:
                mining_footprint_union = rasterize(Geometry(mining_geom, crs=4326), union_geobox)
                logger.info(f"Rasterized mining footprint for ground truth year {year}")
            
            # Ensure we have capacity for these tiles
            self._ensure_capacity(len(tiles))
            
            # Get starting index for this batch
            start_idx = self.store_metadata["next_index"]
            
            # Prepare all tiles first (compute all, then write all at once)
            all_features = []
            all_labels = []
            all_cluster_ids = []
            all_tile_ixs = []
            all_tile_iys = []
            all_years = []
            
            # Extract all tiles
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
                
                # Stack bands to (C, H, W)
                features = np.stack(band_arrays, axis=0).astype(np.float32)
                
                # Extract mining footprint for this tile only if this is the ground truth year
                if year == self.config.GROUND_TRUTH_YEAR and mining_footprint_union is not None:
                    labels = mining_footprint_union.sel(
                        latitude=slice(bounds.top, bounds.bottom),
                        longitude=slice(bounds.left, bounds.right)
                    ).values.astype(np.float32)
                else:
                    # For non-ground truth years, store empty labels
                    labels = np.zeros(tile_geobox.shape, dtype=np.float32)
                
                if labels.ndim == 2:
                    labels = labels[np.newaxis, ...]  # Add channel dimension (1, H, W)
                
                # Replace NaN with 0
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                
                all_features.append(features)
                all_labels.append(labels)
                all_cluster_ids.append(cluster_id)
                all_tile_ixs.append(tile_ix)
                all_tile_iys.append(tile_iy)
                all_years.append(year)
            
            # Write all tiles at once
            self._write_tiles(
                start_idx,
                all_features,
                all_labels,
                all_cluster_ids,
                all_tile_ixs,
                all_tile_iys,
                all_years
            )            
            # Update next index
            self.store_metadata["next_index"] = start_idx + len(tiles)
            self._save_metadata()
            
            logger.info(f"✓ Processed and saved {len(tiles)} tiles to Zarr (indices {start_idx} to {start_idx + len(tiles) - 1})")
            
            # Close the image file handle
            image.close()
            
            # Keep the downloaded file on disk (don't delete)
            
            # Update database - cluster/year is now stored
            self.db.update_task_status(
                cluster_id, year, self.config.STATUS_STORED
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
            
            # Get downloaded tasks with country filtering and filenames
            downloaded_tasks = self.db.get_tasks_by_status(
                self.config.STATUS_DOWNLOADED,
                limit=self.config.BATCH_SIZE,
                countries=self.countries,
                include_filenames=True
            )
            
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