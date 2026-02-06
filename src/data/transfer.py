"""Transfer worker to upload zarr datasets to HPC via SCP."""

import time
import logging
import subprocess
import shutil
import tarfile
import hashlib
import json
from typing import Optional, List
from pathlib import Path
from datetime import datetime

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
        logger.info(f"HPC host: {self.config.HPC_HOST}")
        logger.info(f"HPC user: {self.config.HPC_USER}")
        logger.info(f"HPC path: {self.config.HPC_ZARR_PATH}")
        
        # Create temp directory for compression
        self.temp_dir = self.config.DATA_DIR / "temp_transfer"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Create manifests directory
        self.manifests_dir = self.config.DATA_DIR / "manifests"
        self.manifests_dir.mkdir(exist_ok=True)
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file.
        
        Args:
            filepath: Path to file
            
        Returns:
            Hex digest of hash
        """
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _compress_cluster(self, cluster_path: Path) -> Optional[Path]:
        """Compress a cluster directory to tar.gz.
        
        Args:
            cluster_path: Path to cluster directory
            
        Returns:
            Path to compressed archive or None on failure
        """
        try:
            archive_name = f"{cluster_path.name}.tar.gz"
            archive_path = self.temp_dir / archive_name
            
            logger.info(f"Compressing {cluster_path.name} to {archive_name}")
            
            with tarfile.open(archive_path, "w:gz", compresslevel=6) as tar:
                tar.add(cluster_path, arcname=cluster_path.name)
            
            original_size = sum(f.stat().st_size for f in cluster_path.rglob('*') if f.is_file())
            compressed_size = archive_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            logger.info(f"Compressed {original_size / 1e9:.2f} GB to {compressed_size / 1e9:.2f} GB ({ratio:.1f}% reduction)")
            
            return archive_path
            
        except Exception as e:
            logger.error(f"Error compressing cluster: {e}", exc_info=True)
            return None
    
    def _create_manifest(self, cluster_id: int) -> Optional[Path]:
        """Create a manifest file for a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Path to manifest file or None on failure
        """
        try:
            # Get cluster information from database
            cluster_info = self.db.get_cluster_info(cluster_id)
            if not cluster_info:
                logger.error(f"No cluster info found for cluster {cluster_id}")
                return None
            
            # Create manifest
            manifest = {
                "version": "1.0",
                "cluster_id": cluster_id,
                "country_code": cluster_info['country_code'],
                "created_at": cluster_info['created_at'],
                "latest_reprojected_at": cluster_info['latest_reprojected_at'],
                "generated_at": datetime.utcnow().isoformat(),
                "years": cluster_info['years'],
                "tile_count": cluster_info['tile_count'],
                "tiles": [
                    {
                        "tile_ix": tile['tile_ix'],
                        "tile_iy": tile['tile_iy'],
                        "year": tile['year'],
                        "geometry_hash": tile['geometry_hash'],
                        "mmap_written_at": tile['mmap_written_at']
                    }
                    for tile in cluster_info['tiles']
                ]
            }
            
            # Save manifest locally
            manifest_filename = f"cluster_{cluster_id}_manifest.json"
            manifest_path = self.manifests_dir / manifest_filename
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"Created manifest for cluster {cluster_id}: {cluster_info['tile_count']} tiles across {len(cluster_info['years'])} years")
            
            return manifest_path
            
        except Exception as e:
            logger.error(f"Error creating manifest: {e}", exc_info=True)
            return None
    
    def _transfer_file(self, local_path: Path, remote_path: str) -> bool:
        """Transfer file to HPC via SCP.
        
        Args:
            local_path: Local file path
            remote_path: Remote directory path on HPC (file will be placed there)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            remote_target = f"{self.config.HPC_USER}@{self.config.HPC_HOST}:{remote_path}/"
            
            logger.info(f"Transferring {local_path.name} to {remote_path}")
            
            # Create remote directory first
            remote_dir = remote_path
            mkdir_cmd = [
                'ssh',
                '-i', str(self.config.SSH_KEY),
                '-o', 'StrictHostKeyChecking=no',
                f"{self.config.HPC_USER}@{self.config.HPC_HOST}",
                f"mkdir -p {remote_dir} && ls -ld {remote_dir}"
            ]
            result = subprocess.run(mkdir_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to create remote directory {remote_dir}: {result.stderr}")
                return False
            
            logger.debug(f"Remote directory ready: {result.stdout.strip()}")
            
            # Transfer file with progress
            scp_cmd = [
                'scp',
                '-i', str(self.config.SSH_KEY),
                '-o', 'StrictHostKeyChecking=no',
                '-C',  # Enable compression during transfer
                str(local_path),
                remote_target
            ]
            
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"SCP transfer failed: {result.stderr}")
                return False
            
            logger.info(f"Transfer completed: {local_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error transferring file: {e}", exc_info=True)
            return False
    
    def _decompress_remote(self, remote_archive: str, extract_dir: str) -> bool:
        """Decompress archive on remote HPC.
        
        Args:
            remote_archive: Path to remote archive
            extract_dir: Directory to extract to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Decompressing {remote_archive} on HPC")
            
            decompress_cmd = [
                'ssh',
                '-i', str(self.config.SSH_KEY),
                '-o', 'StrictHostKeyChecking=no',
                f"{self.config.HPC_USER}@{self.config.HPC_HOST}",
                f"cd {extract_dir} && tar -xzf {remote_archive} && rm {remote_archive}"
            ]
            
            result = subprocess.run(decompress_cmd, check=True, capture_output=True, text=True)
            
            logger.info("Decompression completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Remote decompression failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error decompressing remote file: {e}", exc_info=True)
            return False
    
    def transfer_cluster(self, task_data: dict) -> bool:
        """Transfer cluster zarr dataset to HPC.
        
        Compresses locally, transfers, and decompresses on HPC.
        
        Args:
            task_data: Task data from database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            geometry_hash = task_data['geometry_hash']
            year = task_data['year']
            cluster_id = task_data.get('cluster_id')
            
            if not cluster_id:
                logger.warning(f"No cluster ID for {geometry_hash} {year}")
                # Mark as failed since cluster_id is required
                self.db.update_task_status(
                    geometry_hash, year, self.config.STATUS_FAILED,
                    error_message="No cluster ID found"
                )
                return False
            
            # Find cluster directory
            cluster_dir = self.config.DATA_DIR / "landsat_mmap" / str(cluster_id)
            if not cluster_dir.exists():
                logger.warning(f"Cluster directory not found (skipping): {cluster_dir}")
                logger.info(f"Task {geometry_hash} {year} waiting for cluster {cluster_id} to be ready")
                # Don't mark as failed - the cluster might be created later
                # Return False to skip this task but continue processing others
                self.db.increment_worker_counter(self.worker_name, "errors")
                return False
            
            # Check if cluster has any data
            cluster_files = list(cluster_dir.rglob('*'))
            if not cluster_files or all(f.is_dir() for f in cluster_files):
                logger.warning(f"Cluster directory empty or no data: {cluster_dir}")
                self.db.update_task_status(
                    geometry_hash, year, self.config.STATUS_FAILED,
                    error_message="Cluster directory empty"
                )
                return False
            
            # Compress cluster
            archive_path = self._compress_cluster(cluster_dir)
            if not archive_path:
                self.db.update_task_status(
                    geometry_hash, year, self.config.STATUS_FAILED,
                    error_message="Compression failed"
                )
                return False
            
            try:
                # Create manifest for this cluster
                manifest_path = self._create_manifest(cluster_id)
                if not manifest_path:
                    self.db.update_task_status(
                        geometry_hash, year, self.config.STATUS_FAILED,
                        error_message="Manifest creation failed"
                    )
                    return False
                
                # Transfer compressed archive
                remote_archive = f"{self.config.HPC_ZARR_PATH}/{archive_path.name}"
                if not self._transfer_file(archive_path, self.config.HPC_ZARR_PATH):
                    self.db.update_task_status(
                        geometry_hash, year, self.config.STATUS_FAILED,
                        error_message="Transfer failed"
                    )
                    return False
                
                # Decompress on remote
                if not self._decompress_remote(remote_archive, self.config.HPC_ZARR_PATH):
                    self.db.update_task_status(
                        geometry_hash, year, self.config.STATUS_FAILED,
                        error_message="Remote decompression failed"
                    )
                    return False
                
                # Transfer manifest to HPC
                remote_manifests_dir = f"{self.config.HPC_ZARR_PATH}/manifests"
                if not self._transfer_file(manifest_path, remote_manifests_dir):
                    logger.warning(f"Failed to transfer manifest for cluster {cluster_id}")
                    # Don't fail the whole task if manifest upload fails
                
                # Update database
                self.db.update_task_status(
                    geometry_hash, year, self.config.STATUS_UPLOADED
                )
                
                self.db.increment_worker_counter(self.worker_name, "tasks_processed")
                
                logger.info(f"Successfully transferred cluster {cluster_id} with manifest")
                return True
                
            finally:
                # Cleanup local archive (keep manifest)
                if archive_path.exists():
                    archive_path.unlink()
                    logger.debug(f"Cleaned up local archive: {archive_path.name}")
            
        except Exception as e:
            logger.error(f"Error transferring cluster: {e}", exc_info=True)
            geometry_hash = task_data.get('geometry_hash')
            year = task_data.get('year')
            if geometry_hash and year:
                self.db.update_task_status(
                    geometry_hash, year, self.config.STATUS_FAILED,
                    error_message=f"Exception: {str(e)}"
                )
            self.db.increment_worker_counter(self.worker_name, "errors")
            return False
    
    def run(self, continuous: bool = True):
        """Run the transfer worker.
        
        Args:
            continuous: If True, run continuously; otherwise run once
        """
        logger.info(f"Starting {self.worker_name} worker")
        logger.info("Using manifest-based tracking (database is local-only)")
        
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
                    self.transfer_cluster(task)
            else:
                logger.debug("No reprojected tasks to transfer")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
