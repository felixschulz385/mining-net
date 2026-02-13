"""Janitor worker to verify Zarr index integrity.

The Janitor worker verifies that all tiles marked as STORED in the database
are properly indexed in the Zarr store.
"""

import time
import logging
from typing import Optional, List, Set, Tuple, Dict
from pathlib import Path
from collections import defaultdict
import numpy as np
import zarr

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


class JanitorWorker:
    """Worker to verify Zarr index integrity."""
    
    def __init__(
        self, 
        db: DownloadDatabase, 
        config: Optional[Config] = None,
        countries: Optional[List[str]] = None,
        clean: bool = False
    ):
        """Initialize janitor worker.
        
        Args:
            db: Database instance
            config: Configuration instance
            countries: Optional list of ISO3 country codes to filter tasks
            clean: If True, demote tasks with missing tiles
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.clean = clean
        self.worker_name = "janitor"
        
        mode = "CLEAN" if self.clean else "CHECK"
        logger.info(f"Janitor running in {mode} mode")
    
    def _get_zarr_tiles_from_index(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        """Get all tiles from Zarr index arrays.
        
        Returns:
            Dict mapping (cluster_id, year) -> set of (tile_ix, tile_iy) tuples
        """
        zarr_path = self.config.DATA_DIR / "landsat_zarr" / "data.zarr"
        
        if not zarr_path.exists():
            logger.warning(f"Zarr store not found: {zarr_path}")
            return {}
        
        try:
            group = zarr.open_group(store=str(zarr_path), mode='r', zarr_format=3)
            
            # Read index arrays
            cluster_ids = group['cluster_id'][:]
            years = group['year'][:]
            tile_ixs = group['tile_ix'][:]
            tile_iys = group['tile_iy'][:]
            
            # Find valid entries (non-zero cluster_id indicates written tile)
            valid_mask = cluster_ids != 0
            
            # Build mapping
            tiles_by_cluster_year = defaultdict(set)
            for i in np.where(valid_mask)[0]:
                cluster_id = int(cluster_ids[i])
                year = int(years[i])
                tile_ix = int(tile_ixs[i])
                tile_iy = int(tile_iys[i])
                tiles_by_cluster_year[(cluster_id, year)].add((tile_ix, tile_iy))
            
            total_tiles = sum(len(tiles) for tiles in tiles_by_cluster_year.values())
            logger.info(
                f"Found {len(tiles_by_cluster_year)} cluster/year combinations "
                f"with {total_tiles} tiles in Zarr index"
            )
            
            return dict(tiles_by_cluster_year)
            
        except Exception as e:
            logger.error(f"Error reading Zarr index: {e}", exc_info=True)
            return {}
    
    def check_stored_tasks_integrity(self) -> Tuple[int, int]:
        """Verify that all STORED tasks have their tiles in the Zarr index.
        
        Returns:
            Tuple of (issues_found, issues_fixed)
        """
        logger.info("Checking STORED tasks integrity...")
        
        # Get all stored tasks
        if self.countries:
            stored_tasks = []
            for country in self.countries:
                tasks = self.db.get_tasks_by_status(
                    self.config.STATUS_STORED,
                    countries=[country]
                )
                stored_tasks.extend(tasks)
        else:
            stored_tasks = self.db.get_tasks_by_status(self.config.STATUS_STORED)
        
        if not stored_tasks:
            logger.info("No STORED tasks found")
            return 0, 0
        
        logger.info(f"Checking {len(stored_tasks)} STORED tasks")
        
        # Get Zarr index
        zarr_tiles = self._get_zarr_tiles_from_index()
        
        issues_found = 0
        issues_fixed = 0
        
        for task in stored_tasks:
            cluster_id = task['cluster_id']
            year = task['year']
            country_code = task['country_code']
            
            # Get expected tiles from database
            db_tiles = self.db.get_tiles_for_cluster(cluster_id)
            expected_tiles = {(tile['tile_ix'], tile['tile_iy']) for tile in db_tiles}
            
            # Get actual tiles from Zarr
            actual_tiles = zarr_tiles.get((cluster_id, year), set())
            
            # Check if all expected tiles are present
            missing_tiles = expected_tiles - actual_tiles
            
            if missing_tiles:
                issues_found += 1
                logger.warning(
                    f"Task {country_code} cluster={cluster_id} year={year}: "
                    f"{len(missing_tiles)}/{len(expected_tiles)} tiles missing from Zarr index"
                )
                
                if self.clean:
                    logger.info(f"  Demoting to DOWNLOADED to re-store")
                    self.db.update_task_status(
                        cluster_id,
                        year,
                        self.config.STATUS_DOWNLOADED
                    )
                    issues_fixed += 1
        
        if issues_found == 0:
            logger.info("✓ All STORED tasks have complete tiles in Zarr index")
        else:
            logger.warning(f"Found {issues_found} tasks with missing tiles")
            if self.clean:
                logger.info(f"Fixed {issues_fixed} tasks by demoting to DOWNLOADED")
        
        return issues_found, issues_fixed
    
    def run(self, continuous: bool = True):
        """Run the janitor worker.
        
        Args:
            continuous: If True, run continuously; otherwise run once
        """
        logger.info(f"Starting {self.worker_name} worker")
        
        while True:
            self.db.update_worker_heartbeat(self.worker_name, "running")
            
            try:
                issues_found, issues_fixed = self.check_stored_tasks_integrity()
                
                if issues_found > 0:
                    logger.info(
                        f"Integrity check complete: "
                        f"{issues_found} issues found, {issues_fixed} fixed"
                    )
            except Exception as e:
                logger.error(f"Error during integrity check: {e}", exc_info=True)
                self.db.increment_worker_counter(self.worker_name, "errors")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL * 10)  # Run less frequently
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
