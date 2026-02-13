"""Task generation worker for mining segmentation."""

import logging
import time
from tqdm import tqdm
import numpy as np
from typing import List, Optional
from odc.geo.geobox import GeoBox, GeoboxTiles

from .database import DownloadDatabase
from .config import Config
from .clustering import create_clusters_and_tiles

logger = logging.getLogger(__name__)


class TaskGeneratorWorker:
    """Worker to generate pending tasks for clusters across years."""
    
    def __init__(
        self, 
        db: DownloadDatabase, 
        config: Optional[Config] = None, 
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ):
        """Initialize task generator worker.
        
        Args:
            db: Database instance
            config: Configuration instance
            countries: Optional list of ISO3 country codes to filter clusters
            years: Optional list of years to generate tasks for (default: 1984-2023)
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.years = years or list(range(1984, 2024))  # Landsat 5+ years
        self.worker_name = "task_generator"
        
        logger.info(f"Initialized TaskGeneratorWorker for years {min(self.years)}-{max(self.years)}")
        if countries:
            logger.info(f"Filtering for countries: {', '.join(countries)}")
    
    def run(self, continuous: bool = True):
        """Run the task generator worker.
        
        Args:
            continuous: Whether to run continuously or just once
        """
        logger.info(f"Starting task generator worker (continuous={continuous})")
        
        while True:
            try:
                self.db.update_worker_heartbeat(self.worker_name, "running")
                
                # Get all cluster IDs (optionally filtered by country)
                cluster_ids = self.db.get_all_cluster_ids(self.countries)
                
                if not cluster_ids:
                    logger.warning("No clusters found in database")
                    if not continuous:
                        break
                    time.sleep(self.config.WORKER_SLEEP_INTERVAL)
                    continue
                                
                # Prepare all tasks to create
                all_tasks = []
                for cluster_id in cluster_ids:
                    for year in self.years:
                        all_tasks.append((cluster_id, year, self.config.STATUS_PENDING))
                
                # Batch create tasks
                if all_tasks:
                    tasks_created = self.db.create_tasks(all_tasks)
                    
                    if tasks_created > 0:
                        logger.info(f"Created {tasks_created} new tasks")
                        self.db.increment_worker_counter(self.worker_name, "tasks_processed")
                    
                    # Show task summary by status
                    task_summary = self.db.get_task_summary(self.countries)
                    if task_summary:
                        summary_str = ", ".join([f"{status}: {count}" for status, count in task_summary.items()])
                        logger.info(f"Task summary: {summary_str}")
                else:
                    logger.info("No tasks to create")
                
                if not continuous:
                    break
                
                # Sleep before next iteration
                logger.info(f"Sleeping for {self.config.WORKER_SLEEP_INTERVAL}s...")
                time.sleep(self.config.WORKER_SLEEP_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("Task generator worker interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in task generator worker: {e}", exc_info=True)
                self.db.increment_worker_counter(self.worker_name, "errors")
                self.db.update_worker_heartbeat(self.worker_name, "error")
                
                if not continuous:
                    raise
                
                time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        logger.info("Task generator worker stopped")
