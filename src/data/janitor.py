"""Janitor worker to verify Zarr index and downloaded-file integrity.

The Janitor worker validates Zarr index entries and checks downloaded GeoTIFFs
and their metadata. It can optionally repair metadata, demote tasks, or
remove orphan files when run with `clean=True`.
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
        

    
    def check_downloaded_tasks_integrity(self) -> Tuple[int, int]:
        """Verify downloaded files exist, repair missing metadata, and optionally clean orphan files.
        
        Returns:
            Tuple of (issues_found, issues_fixed)
        """
        logger.info("Checking DOWNLOADED tasks and download-directory integrity...")

        # Fetch downloaded tasks (optionally filtered by country)
        if self.countries:
            downloaded_tasks = []
            for country in self.countries:
                tasks = self.db.get_tasks_by_status(
                    self.config.STATUS_DOWNLOADED,
                    countries=[country],
                    include_filenames=True
                )
                downloaded_tasks.extend(tasks)
        else:
            downloaded_tasks = self.db.get_tasks_by_status(
                self.config.STATUS_DOWNLOADED, include_filenames=True
            )

        if not downloaded_tasks:
            logger.info("No DOWNLOADED tasks found")
            return 0, 0

        issues_found = 0
        issues_fixed = 0

        expected_names = set()

        for task in downloaded_tasks:
            cluster_id = task['cluster_id']
            year = task['year']
            country_code = task['country_code']
            local_fp = Path(task.get('local_filepath', ''))

            if not local_fp:
                logger.warning(f"Task missing local filepath metadata: {cluster_id}/{year}")
                issues_found += 1
                continue

            expected_names.add(local_fp.name)

            if not local_fp.exists():
                issues_found += 1
                logger.warning(
                    f"Downloaded file missing for task {country_code} cluster={cluster_id} year={year}: {local_fp}"
                )
                if self.clean:
                    logger.info("  Demoting to COMPLETED to trigger re-download")
                    self.db.update_task_status(cluster_id, year, self.config.STATUS_COMPLETED)
                    issues_fixed += 1
                continue

            # Ensure metadata JSON exists alongside TIFF; recreate if missing
            meta_path = local_fp.with_suffix('.json')
            if not meta_path.exists():
                issues_found += 1
                logger.warning(f"Metadata JSON missing for {local_fp}")
                if self.clean:
                    try:
                        tiles = self.db.get_tiles_for_cluster(cluster_id)
                        footprint = self.db.get_mining_footprint(cluster_id)
                        metadata = {
                            "cluster_id": cluster_id,
                            "country_code": country_code,
                            "year": year,
                            "tiles": [{"tile_ix": t["tile_ix"], "tile_iy": t["tile_iy"]} for t in tiles],
                            "footprint": footprint
                        }
                        with open(meta_path, 'w', encoding='utf-8') as mf:
                            import json
                            json.dump(metadata, mf, indent=2)
                        logger.info(f"  Recreated metadata: {meta_path}")
                        issues_fixed += 1
                    except Exception as e:
                        logger.warning(f"  Could not recreate metadata for {local_fp}: {e}")

        # Detect orphan TIFFs in the download directory (conservative: only files starting with 'LANDSAT_')
        download_dir = Path(self.config.DOWNLOAD_DIR)
        if download_dir.exists():
            for p in download_dir.glob('LANDSAT_*'):
                if p.suffix.lower() not in ('.tif', '.tiff'):
                    continue
                if p.name not in expected_names:
                    issues_found += 1
                    logger.warning(f"Orphan downloaded file: {p}")
                    if self.clean:
                        try:
                            p.unlink()
                            meta = p.with_suffix('.json')
                            if meta.exists():
                                meta.unlink()
                            logger.info(f"  Removed orphan file: {p}")
                            issues_fixed += 1
                        except Exception as e:
                            logger.warning(f"  Could not remove orphan file {p}: {e}")

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
                d_found, d_fixed = self.check_downloaded_tasks_integrity()

                issues_found = d_found
                issues_fixed = d_fixed

                if issues_found > 0:
                    logger.info(
                        f"Integrity check complete: {issues_found} issues found, {issues_fixed} fixed"
                    )
            except Exception as e:
                logger.error(f"Error during integrity check: {e}", exc_info=True)
                self.db.increment_worker_counter(self.worker_name, "errors")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL * 10)  # Run less frequently
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
