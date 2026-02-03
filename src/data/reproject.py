"""Reprojection worker to convert downloaded GeoTIFFs to zarr format."""

import time
import logging
from typing import Optional, List, Tuple
from pathlib import Path

import zarr
from zarr.codecs import BloscCodec
import numpy as np
import dask.array as da
import xarray as xr
import rioxarray as rxr
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.geom import Geometry
from odc.geo.xr import rasterize
from shapely.geometry import shape

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


def create_global_zarr_dataset(
    output_path: str,
    world_geobox: GeoBox,
    bands: List[str] = None,
    chunk_size: Tuple[int, int] = (512, 512),
    dtype: str = 'float32'
) -> None:
    """Create an empty global zarr dataset for Landsat bands and mining footprints.
    
    Args:
        output_path: Path to output zarr directory
        world_geobox: Global GeoBox defining spatial structure
        bands: List of band names (default: Landsat bands)
        chunk_size: Chunk size for zarr storage (y, x)
        dtype: Data type for bands
    
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if chunk_size[0] <= 0 or chunk_size[1] <= 0:
        raise ValueError(f"Invalid chunk size: {chunk_size}")
    if dtype not in ['float32', 'float64', 'int16', 'uint16']:
        raise ValueError(f"Invalid dtype: {dtype}. Must be one of: float32, float64, int16, uint16")
    
    if bands is None:
        bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
    
    if not bands:
        raise ValueError("At least one band must be specified")
    
    logger.info(f"Creating global zarr dataset at {output_path}")
    logger.info(f"  Resolution: {world_geobox.resolution}")
    logger.info(f"  Shape: {world_geobox.shape}")
    logger.info(f"  Bounds: {world_geobox.boundingbox}")
    logger.info(f"  Bands: {bands}")
    logger.info(f"  Chunk size: {chunk_size}")
    
    # Define spatial dimensions
    height, width = world_geobox.shape
    
    # Create coordinate arrays
    y_coords = world_geobox.coords["latitude"].values.round(5)
    x_coords = world_geobox.coords["longitude"].values.round(5)
    
    # Create empty dataset with all bands
    data_vars = {}
    
    # Add Landsat bands
    for band in bands:
        data_vars[band] = (
            ['latitude', 'longitude'],
            da.full((height, width), np.nan, dtype=dtype, chunks=chunk_size),
            {
                'long_name': f'Landsat {band} band',
                'units': 'reflectance' if band != 'thermal' else 'kelvin'
            }
        )
    
    # Add mining footprint
    data_vars['mining_footprint'] = (
        ['latitude', 'longitude'],
        da.zeros((height, width), dtype='uint8', chunks=chunk_size),
        {
            'long_name': 'Mining area footprint',
            'description': 'Binary mask: 1 = mining area, 0 = no mining',
            'units': 'boolean'
        }
    )
    
    # Create xarray Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'latitude': ('latitude', y_coords, {'units': 'degrees_north'}),
            'longitude': ('longitude', x_coords, {'units': 'degrees_east'})
        },
        attrs={
            'crs': str(world_geobox.crs),
        }
    )
    
    # Write to zarr with chunking and compression, but don't compute data
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            'chunks': chunk_size,
            'compressor': BloscCodec()
        }
    
    ds.to_zarr(output_path, mode='w', encoding=encoding, consolidated=False, compute=False)
    
    logger.info(f"Global zarr dataset structure created successfully")
    logger.info(f"  Total size: {height} x {width} = {height * width:,} pixels")
    logger.info(f"  Estimated storage (compressed): ~{(height * width * len(bands) * 4 / 10) / 1e9:.2f} GB")


class ReprojectionWorker:
    """Worker to reproject downloaded files to zarr."""
    
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
        
        # Setup global zarr
        self.global_zarr_path = self.config.DATA_DIR / "global_landsat.zarr"
        self._ensure_global_zarr()
        
        logger.info("Initialized reprojection worker")
    
    def _ensure_global_zarr(self):
        """Ensure global zarr dataset exists, create if needed."""
        if not self.global_zarr_path.exists():
            logger.info("Creating global zarr dataset...")
            create_global_zarr_dataset(
                str(self.global_zarr_path),
                self.world_geobox,
                chunk_size=(512, 512)
            )
        else:
            logger.info(f"Using existing global zarr: {self.global_zarr_path}")
    
    def reproject_to_zarr(self, task_data: dict) -> bool:
        """Reproject downloaded file to grid and save as zarr.
        
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
            logger.info(f"Reprojecting {len(tiles)} tiles to global zarr for {country_code} {year}")
            image = rxr.open_rasterio(filepath)
            
            # Get mining footprint from database (will rasterize per tile)
            mining_footprint_json = self.db.get_mining_footprint(geometry_hash, year)
            
            # Reproject each tile and write to global zarr using xarray region writing
            for tile in tiles:
                tile_ix = tile['tile_ix']
                tile_iy = tile['tile_iy']
                
                # Get tile geobox
                tile_geobox = self.world_geobox_tiles[tile_ix, tile_iy]
                
                # Reproject image to this tile
                reprojected = image.odc.reproject(tile_geobox)
                
                # Round resolution of coords
                reprojected.coords["latitude"] = reprojected.coords["latitude"].values.round(5)
                reprojected.coords["longitude"] = reprojected.coords["longitude"].values.round(5)
                
                # Convert to xarray Dataset with band names
                conversion_dict = {1: 'blue', 2: 'green', 3: 'red', 4: 'nir', 5: 'swir1', 6: 'swir2', 7: 'thermal'}
                reprojected = reprojected.to_dataset(dim="band").rename(conversion_dict)
                
                # Rasterize mining footprint for this tile if available
                if mining_footprint_json is not None:
                    mining_geom = shape(mining_footprint_json)
                    tile_mining_footprint = rasterize(
                        Geometry(mining_geom, crs=4326),
                        tile_geobox
                    )
                    # Round coordinates for alignment
                    tile_mining_footprint.coords['latitude'] = tile_mining_footprint.coords['latitude'].values.round(5)
                    tile_mining_footprint.coords['longitude'] = tile_mining_footprint.coords['longitude'].values.round(5)
                    reprojected['mining_footprint'] = tile_mining_footprint
                
                # Write to global zarr using coordinate-based region indexing
                # xarray will handle the alignment automatically
                reprojected.drop_vars(['spatial_ref'], errors='ignore')\
                    .to_zarr(
                    str(self.global_zarr_path),
                    region="auto",
                    mode='r+'
                )
                
                # Mark tile as written
                self.db.mark_tile_written(tile_ix, tile_iy, geometry_hash, year)
            
            logger.info(f"✓ Reprojected {len(tiles)} tiles to global zarr")
            
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
                    self.reproject_to_zarr(task)
            else:
                logger.debug("No downloaded tasks to reproject")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")