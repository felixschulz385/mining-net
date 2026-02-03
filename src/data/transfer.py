"""Transfer worker to upload zarr datasets to HPC via SCP."""

import time
import logging
from typing import Optional, List
from pathlib import Path

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


class TransferWorker:
    """Worker to transfer zarr datasets to HPC."""
    
    def __init__(self, db: DownloadDatabase, config: Optional[Config] = None, countries: Optional[List[str]] = None):
        """Initialize transfer worker.
        
        Args:
            db: Database instance
            config: Configuration instance
            countries: Optional list of ISO3 country codes to filter tasks
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.worker_name = "transfer"
        
        logger.info("Initialized transfer worker")
        
        # TODO: Add HPC configuration
        # self.hpc_host = config.HPC_HOST
        # self.hpc_user = config.HPC_USER
        # self.hpc_path = config.HPC_ZARR_PATH
    
    def transfer_zarr(self, task_data: dict) -> bool:
        """Transfer zarr dataset to HPC via SCP.
        
        Args:
            task_data: Task data from database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            geometry_hash = task_data['geometry_hash']
            year = task_data['year']
            zarr_path = task_data.get('zarr_region_path')
            
            if not zarr_path:
                logger.warning(f"No zarr path for {geometry_hash} {year}")
                return False
            
            zarr_path = Path(zarr_path)
            if not zarr_path.exists():
                logger.error(f"Zarr path not found: {zarr_path}")
                return False
            
            logger.info(f"Transferring {zarr_path.name} to HPC")
            
            # TODO: Implement SCP transfer
            # Example implementation:
            # import subprocess
            # scp_cmd = [
            #     'scp', '-r',
            #     str(zarr_path),
            #     f'{self.hpc_user}@{self.hpc_host}:{self.hpc_path}/'
            # ]
            # result = subprocess.run(scp_cmd, check=True, capture_output=True)
            
            # For now, just log
            logger.info(f"TODO: Transfer {zarr_path} to HPC")
            
            # Update database
            self.db.update_task_status(
                geometry_hash, year, self.config.STATUS_UPLOADED
            )
            
            self.db.increment_worker_counter(self.worker_name, "tasks_processed")
            return True
            
        except Exception as e:
            logger.error(f"Error transferring zarr: {e}", exc_info=True)
            self.db.increment_worker_counter(self.worker_name, "errors")
            return False
    
    def run(self, continuous: bool = True):
        """Run the transfer worker.
        
        Args:
            continuous: If True, run continuously; otherwise run once
        """
        logger.info(f"Starting {self.worker_name} worker")
        
        while True:
            self.db.update_worker_heartbeat(self.worker_name, "running")
            
            # Get reprojected tasks
            reprojected_tasks = self.db.get_tasks_by_status(
                self.config.STATUS_REPROJECTED,
                limit=self.config.BATCH_SIZE
            )
            
            # Filter by countries if specified
            if self.countries and reprojected_tasks:
                reprojected_tasks = [t for t in reprojected_tasks if t['country_code'] in self.countries]
            
            if reprojected_tasks:
                logger.info(f"Processing {len(reprojected_tasks)} reprojected tasks")
                for task in reprojected_tasks:
                    self.transfer_zarr(task)
            else:
                logger.debug("No reprojected tasks to transfer")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
