"""Status checker worker to monitor GEE task completion."""

import time
import logging
from typing import Optional, List
import ee

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


class StatusCheckerWorker:
    """Worker to check status of GEE export tasks."""
    
    # Rate limiting delay between status checks (seconds)
    STATUS_CHECK_DELAY = 0.5
    
    def __init__(self, db: DownloadDatabase, config: Optional[Config] = None, countries: Optional[List[str]] = None):
        """Initialize status checker worker.
        
        Args:
            db: Database instance
            config: Configuration instance
            countries: Optional list of ISO3 country codes to filter tasks
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.worker_name = "status_checker"
        
        # Initialize Earth Engine
        ee.Initialize(project=self.config.GEE_PROJECT)
        logger.info("Initialized Google Earth Engine")
    
    def check_task_status(self, task_data: dict) -> bool:
        """Check status of a single GEE task.
        
        Args:
            task_data: Task data from database
            
        Returns:
            True if status updated, False otherwise
        """
        try:
            geometry_hash = task_data['geometry_hash']
            year = task_data['year']
            gee_task_id = task_data['gee_task_id']
            
            # Get task status from GEE
            status = ee.data.getTaskStatus(gee_task_id)[0]
            
            state = status.get('state')
            
            if state == 'COMPLETED':
                logger.info(f"Task {gee_task_id} completed for {task_data['country_code']} {year}")
                self.db.update_task_status(
                    geometry_hash, year, self.config.STATUS_COMPLETED
                )
                self.db.increment_worker_counter(self.worker_name, "tasks_processed")
                return True
                
            elif state in ['FAILED', 'CANCELLED']:
                error_msg = status.get('error_message', 'Unknown error')
                logger.error(f"Task {gee_task_id} {state}: {error_msg}")
                self.db.update_task_status(
                    geometry_hash, year, self.config.STATUS_FAILED,
                    error_message=error_msg
                )
                self.db.increment_worker_counter(self.worker_name, "errors")
                return True
                
            elif state in ['RUNNING', 'READY']:
                logger.debug(f"Task {gee_task_id} in state: {state}")
                return False
                
            else:
                logger.debug(f"Task {gee_task_id} in state: {state}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking task status: {e}", exc_info=True)
            self.db.increment_worker_counter(self.worker_name, "errors")
            return False
    
    def run(self, continuous: bool = True):
        """Run the status checker worker.
        
        Args:
            continuous: If True, run continuously; otherwise run once
        """
        logger.info(f"Starting {self.worker_name} worker")
        
        while True:
            self.db.update_worker_heartbeat(self.worker_name, "running")
            
            # Get submitted tasks
            submitted_tasks = self.db.get_tasks_by_status(self.config.STATUS_SUBMITTED)
            
            # Filter by countries if specified
            if self.countries and submitted_tasks:
                submitted_tasks = [t for t in submitted_tasks if t['country_code'] in self.countries]
            
            if submitted_tasks:
                logger.info(f"Checking status of {len(submitted_tasks)} tasks")
                for task in submitted_tasks:
                    self.check_task_status(task)
                    time.sleep(self.STATUS_CHECK_DELAY)  # Rate limiting
            else:
                logger.debug("No tasks to check")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
