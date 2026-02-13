"""Google Earth Engine export worker."""

import time
import logging
from typing import List, Optional
import json
import ee
import geopandas as gpd
from odc.geo.geobox import GeoBox, GeoboxTiles
from odc.geo.geom import Geometry

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


class GEEExportWorker:
    """Worker to submit export tasks to Google Earth Engine."""
    
    # Rate limiting delay between task submissions (seconds)
    TASK_SUBMISSION_DELAY = 1
    
    def __init__(self, db: DownloadDatabase, config: Optional[Config] = None, countries: Optional[List[str]] = None):
        """Initialize GEE export worker.
        
        Args:
            db: Database instance
            config: Configuration instance
            countries: Optional list of ISO3 country codes to filter tasks
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.worker_name = "gee_export"
        
        # Initialize Earth Engine
        ee.Initialize(project=self.config.GEE_PROJECT)
        logger.info("Initialized Google Earth Engine")
        
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
        logger.info("Initialized world geobox")
    
    def submit_task(self, task_data: dict) -> bool:
        """Submit a single export task to GEE.
        
        Args:
            task_data: Task data from database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            year = task_data['year']
            country_code = task_data['country_code']
            cluster_id = task_data.get('cluster_id', 0)
            
            # Parse geometry
            geometry = json.loads(task_data['geometry_json'])
            
            # Get geobox-aligned bounding box
            geom = Geometry(geometry, crs=4326)
            tiles = list(self.world_geobox_tiles.tiles(geom))
            
            if not tiles:
                logger.warning(f"No tiles for {country_code} {year}")
                return False
            
            # Use total bounds of all tiles as query region
            tile_geoms = [
                self.world_geobox_tiles[tile_ix, tile_iy].extent.geom
                for tile_ix, tile_iy in tiles
            ]
            query_geom = gpd.GeoSeries(tile_geoms).unary_union
            roi = ee.Geometry(query_geom.__geo_interface__)
            
            # Get image for the year
            dataset = ee.ImageCollection(self.config.GEE_COLLECTION) \
                .filterDate(f'{year}-01-01', f'{year+1}-01-01') \
                .filterBounds(roi)
            
            year_image = dataset.first()
            
            if year_image is None:
                logger.warning(f"No image for {country_code} {year}")
                self.db.update_task_status(
                    cluster_id, year, self.config.STATUS_FAILED
                )
                return False
            
            # Clip to ROI
            clipped_image = year_image.clip(roi)
            
            # Create filename using cluster_id in hex format
            cluster_id_hex = format(cluster_id, 'x')[:8]
            filename = f"LANDSAT_C02_T1_L2_{country_code}_{cluster_id_hex}_{year}"
            
            # Create export task
            task = ee.batch.Export.image.toDrive(
                image=clipped_image,
                description=filename,
                folder=self.config.DRIVE_FOLDER,
                fileNamePrefix=filename,
                scale=self.config.GEE_SCALE,
                crs=self.config.GEE_CRS,
                fileFormat='GeoTIFF',
                maxPixels=self.config.GEE_MAX_PIXELS
            )
            
            # Start the task
            task.start()
            gee_task_id = task.id
            
            logger.info(f"Submitted task {gee_task_id} for {country_code} {year} ({len(tiles)} tiles)")
            
            # Update database with GEE task ID
            self.db.update_task_status(
                cluster_id, year, self.config.STATUS_SUBMITTED, gee_task_id=gee_task_id
            )
            
            self.db.increment_worker_counter(self.worker_name, "tasks_processed")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting task: {e}", exc_info=True)
            self.db.update_task_status(
                task_data['cluster_id'], 
                task_data['year'], 
                self.config.STATUS_FAILED
            )
            self.db.increment_worker_counter(self.worker_name, "errors")
            return False
    
    def run(self, continuous: bool = True):
        """Run the export worker.
        
        Args:
            continuous: If True, run continuously; otherwise run once
        """
        logger.info(f"Starting {self.worker_name} worker")
        
        while True:
            self.db.update_worker_heartbeat(self.worker_name, "running")
            
            # Check number of currently submitted tasks
            submitted_tasks = self.db.get_tasks_by_status(self.config.STATUS_SUBMITTED)
            num_submitted = len(submitted_tasks)
            
            # Skip submitting new tasks if at limit
            if num_submitted >= self.config.MAX_SUBMITTED_TASKS:
                logger.info(f"At maximum submitted tasks limit ({num_submitted}/{self.config.MAX_SUBMITTED_TASKS}), waiting...")
            else:
                # Get available cluster/year combinations from cluster and tiles tables
                available_slots = self.config.MAX_SUBMITTED_TASKS - num_submitted
                batch_size = min(self.config.BATCH_SIZE, available_slots)
                
                # Get pending tasks with geometry and filtering
                pending_tasks = self.db.get_tasks_by_status(
                    self.config.STATUS_PENDING,
                    limit=batch_size,
                    countries=self.countries,
                    include_geometry=True
                )
                
                if pending_tasks:
                    logger.info(f"Processing {len(pending_tasks)} pending tasks ({num_submitted}/{self.config.MAX_SUBMITTED_TASKS} submitted)")
                    
                    for task_data in pending_tasks:
                        self.submit_task(task_data)
                        time.sleep(self.TASK_SUBMISSION_DELAY)  # Rate limiting
                else:
                    logger.debug("No pending tasks")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
