"""Janitor worker to verify filesystem integrity and sync with database.

The Janitor worker performs periodic integrity checks to ensure the database
state matches the actual filesystem state. It can run in two modes:

1. CHECK mode (clean=False): Reports issues without making changes
2. CLEAN mode (clean=True): Fixes inconsistencies automatically

Workflow:
---------
1. Database Integrity Checks:
   - For UPLOADED tasks: Verify clusters exist on HPC
     * Check cluster/year directories exist on HPC via SSH
     * If not on HPC: downgrade to REPROJECTED (if in MMAP) -> DOWNLOADED -> PENDING
   
   - For REPROJECTED tasks: Verify MMAP directories exist and are complete
     * Check cluster/year directories exist
     * Compare actual tile coordinates with tiles table in database
     * Verify each tile has features.pt, labels.pt, metadata.json
     * If mismatch: remove MMAP in CLEAN mode, downgrade to DOWNLOADED or PENDING
   
   - For DOWNLOADED tasks: Verify local files exist in download directory
     * If file missing: check if on Google Drive -> downgrade to COMPLETED
     * If not on Drive: reset to PENDING
   
   - For COMPLETED tasks: Verify files exist on Google Drive
     * If not found: reset to PENDING for re-export

2. Orphan Detection:
   - MMAP Orphans: Find cluster/year directories not in database
     * Remove in CLEAN mode
   
   - Download Orphans: Find local .tif files not referenced in database
     * Remove in CLEAN mode

3. Tile Integrity:
   - Compare MMAP tile subdirectories with tiles table entries
   - Verify tile coordinates match exactly (tile_ix, tile_iy)
   - Check for missing or extra tiles
   - Remove incomplete tile directories in CLEAN mode

The janitor ensures smooth recovery from interrupted workers and prevents
disk space waste from orphaned files.
"""

import time
import logging
import pickle
import subprocess
from pathlib import Path
from typing import Optional, List, Set, Tuple, Dict
from collections import defaultdict

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


class JanitorWorker:
    """Worker to verify filesystem integrity and keep database synchronized."""
    
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
            clean: If True, remove orphaned files and fix inconsistencies
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.clean = clean
        self.worker_name = "janitor"
        self.drive_service = None
        
        if self.clean:
            logger.info("Janitor running in CLEAN mode - will fix inconsistencies")
        else:
            logger.info("Janitor running in CHECK mode - will only report issues")
        
        # Initialize Google Drive API for checking files
        try:
            self._authenticate()
            logger.info("Initialized Google Drive API")
        except Exception as e:
            logger.warning(f"Could not initialize Drive API: {e}")
            self.drive_service = None
    
    def _authenticate(self):
        """Authenticate with Google Drive."""
        creds = None
        
        if self.config.TOKEN_FILE.exists():
            with open(self.config.TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.config.CREDENTIALS_FILE.exists():
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.config.CREDENTIALS_FILE}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.config.CREDENTIALS_FILE), 
                    self.config.DRIVE_SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            with open(self.config.TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        
        self.drive_service = build('drive', 'v3', credentials=creds)
    
    def _get_drive_files(self) -> Set[str]:
        """Get set of files on Google Drive.
        
        Returns:
            Set of filenames (without .tif extension)
        """
        if not self.drive_service:
            return set()
        
        try:
            # Find folder
            folder_query = f"name='{self.config.DRIVE_FOLDER}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            results = self.drive_service.files().list(
                q=folder_query,
                spaces='drive',
                fields='files(id, name)',
                pageSize=10
            ).execute()
            
            folders = results.get('files', [])
            if not folders:
                logger.warning(f"Drive folder '{self.config.DRIVE_FOLDER}' not found")
                return set()
            
            folder_id = folders[0]['id']
            
            # List files
            file_query = f"'{folder_id}' in parents and trashed=false"
            results = self.drive_service.files().list(
                q=file_query,
                spaces='drive',
                fields='files(id, name)',
                pageSize=1000
            ).execute()
            
            files = results.get('files', [])
            
            # Extract task descriptions from filenames (remove .tif extension)
            drive_files = set()
            for file in files:
                name = file['name']
                if name.endswith('.tif'):
                    name = name[:-4]
                drive_files.add(name)
            
            logger.info(f"Found {len(drive_files)} files on Google Drive")
            return drive_files
            
        except Exception as e:
            logger.error(f"Error listing Drive files: {e}")
            return set()
    
    def _get_downloaded_files(self) -> Set[Tuple[str, int]]:
        """Get set of downloaded files in local directory.
        
        Returns:
            Set of (geometry_hash, year) tuples
        """
        downloaded = set()
        
        if not self.config.DOWNLOAD_DIR.exists():
            return downloaded
        
        for file in self.config.DOWNLOAD_DIR.glob("*.tif"):
            # Parse filename to extract geometry_hash and year
            # Format: {geometry_hash}_{year}.tif or similar
            try:
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    # Assume last part is year, rest is hash
                    year = int(parts[-1])
                    geometry_hash = '_'.join(parts[:-1])
                    downloaded.add((geometry_hash, year))
            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse filename: {file.name}")
        
        logger.info(f"Found {len(downloaded)} files in download directory")
        return downloaded
    
    def _get_mmap_clusters(self) -> Dict[int, Set[int]]:
        """Get clusters and years in MMAP directory.
        
        Returns:
            Dict mapping cluster_id -> set of years
        """
        mmap_path = self.config.DATA_DIR / "landsat_mmap"
        clusters = defaultdict(set)
        
        if not mmap_path.exists():
            return clusters
        
        for cluster_dir in mmap_path.iterdir():
            if cluster_dir.is_dir():
                try:
                    cluster_id = int(cluster_dir.name)
                    
                    # Check for year subdirectories
                    for year_dir in cluster_dir.iterdir():
                        if year_dir.is_dir():
                            try:
                                year = int(year_dir.name)
                                clusters[cluster_id].add(year)
                            except ValueError:
                                pass
                except ValueError:
                    pass
        
        total_entries = sum(len(years) for years in clusters.values())
        logger.info(f"Found {len(clusters)} clusters with {total_entries} year entries in MMAP")
        return clusters
    
    def _get_hpc_clusters(self) -> Dict[int, Set[int]]:
        """Get clusters and years on HPC.
        
        Returns:
            Dict mapping cluster_id -> set of years
        """
        clusters = defaultdict(set)
        
        try:
            # List cluster directories on HPC
            list_cmd = [
                'ssh',
                '-i', str(self.config.SSH_KEY),
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=10',
                f"{self.config.HPC_USER}@{self.config.HPC_HOST}",
                f"find {self.config.HPC_ZARR_PATH} -mindepth 2 -maxdepth 2 -type d"
            ]
            
            result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"Could not list HPC clusters: {result.stderr}")
                return clusters
            
            # Parse output: paths like /path/to/cluster_id/year
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    parts = line.strip().split('/')
                    if len(parts) >= 2:
                        cluster_id = int(parts[-2])
                        year = int(parts[-1])
                        clusters[cluster_id].add(year)
                except (ValueError, IndexError):
                    continue
            
            total_entries = sum(len(years) for years in clusters.values())
            logger.info(f"Found {len(clusters)} clusters with {total_entries} year entries on HPC")
            return clusters
            
        except subprocess.TimeoutExpired:
            logger.warning("Timeout while checking HPC clusters")
            return clusters
        except Exception as e:
            logger.warning(f"Error checking HPC clusters: {e}")
            return clusters
    
    def _count_tiles_in_mmap(self, cluster_id: int, year: int) -> int:
        """Count number of tile directories in MMAP for a cluster/year.
        
        Args:
            cluster_id: Cluster ID
            year: Year
            
        Returns:
            Number of tile directories found
        """
        mmap_path = self.config.DATA_DIR / "landsat_mmap" / str(cluster_id) / str(year)
        
        if not mmap_path.exists():
            return 0
        
        # Count directories that match tile naming pattern (ix_iy)
        tile_count = 0
        for tile_dir in mmap_path.iterdir():
            if tile_dir.is_dir() and '_' in tile_dir.name:
                # Verify it has required files
                has_features = (tile_dir / "features.pt").exists()
                has_labels = (tile_dir / "labels.pt").exists()
                has_metadata = (tile_dir / "metadata.json").exists()
                
                if has_features and has_labels and has_metadata:
                    tile_count += 1
        
        return tile_count
    
    def _get_mmap_tile_coords(self, cluster_id: int, year: int) -> Set[Tuple[int, int]]:
        """Get actual tile coordinates in MMAP directory.
        
        Args:
            cluster_id: Cluster ID
            year: Year
            
        Returns:
            Set of (tile_ix, tile_iy) tuples
        """
        mmap_path = self.config.DATA_DIR / "landsat_mmap" / str(cluster_id) / str(year)
        tiles = set()
        
        if not mmap_path.exists():
            return tiles
        
        for tile_dir in mmap_path.iterdir():
            if tile_dir.is_dir() and '_' in tile_dir.name:
                try:
                    parts = tile_dir.name.split('_')
                    if len(parts) == 2:
                        tile_ix = int(parts[0])
                        tile_iy = int(parts[1])
                        tiles.add((tile_ix, tile_iy))
                except (ValueError, IndexError):
                    pass
        
        return tiles
    
    def check_database_integrity(self):
        """Check database tasks against filesystem and fix inconsistencies."""
        logger.info("=== Checking Database Integrity ===")
        
        issues_found = 0
        issues_fixed = 0
        
        # Get filesystem state
        drive_files = self._get_drive_files()
        downloaded_files = self._get_downloaded_files()
        mmap_clusters = self._get_mmap_clusters()
        hpc_clusters = self._get_hpc_clusters()
        
        # Check each status level
        statuses_to_check = [
            self.config.STATUS_UPLOADED,
            self.config.STATUS_REPROJECTED,
            self.config.STATUS_DOWNLOADED,
            self.config.STATUS_COMPLETED
        ]
        
        for status in statuses_to_check:
            tasks = self.db.get_tasks_by_status(status, limit=None)
            
            # Filter by countries if specified
            if self.countries:
                tasks = [t for t in tasks if t['country_code'] in self.countries]
            
            logger.info(f"Checking {len(tasks)} tasks with status '{status}'")
            
            for task in tasks:
                geometry_hash = task['geometry_hash']
                year = task['year']
                cluster_id = task.get('cluster_id')
                country_code = task['country_code']
                
                issue = None
                new_status = None
                
                if status == self.config.STATUS_UPLOADED:
                    # Check HPC has the cluster/year
                    if cluster_id is None:
                        issue = "No cluster_id"
                        new_status = self.config.STATUS_PENDING
                    elif cluster_id not in hpc_clusters or year not in hpc_clusters[cluster_id]:
                        issue = "Cluster not on HPC"
                        # Check if we can downgrade to reprojected
                        if cluster_id in mmap_clusters and year in mmap_clusters[cluster_id]:
                            new_status = self.config.STATUS_REPROJECTED
                        elif task.get('local_filepath') and Path(task['local_filepath']).exists():
                            new_status = self.config.STATUS_DOWNLOADED
                        else:
                            new_status = self.config.STATUS_PENDING
                
                elif status == self.config.STATUS_REPROJECTED:
                    # Check MMAP exists and has correct tiles matching database
                    if cluster_id is None:
                        issue = "No cluster_id"
                        new_status = self.config.STATUS_PENDING
                    elif cluster_id not in mmap_clusters or year not in mmap_clusters[cluster_id]:
                        issue = "MMAP cluster/year missing"
                        # Check if we can downgrade to downloaded
                        if task.get('local_filepath') and Path(task['local_filepath']).exists():
                            new_status = self.config.STATUS_DOWNLOADED
                        else:
                            new_status = self.config.STATUS_PENDING
                    else:
                        # Get expected tiles from database
                        db_tiles = self.db.get_tiles_for_task(geometry_hash, year)
                        expected_coords = {(t['tile_ix'], t['tile_iy']) for t in db_tiles}
                        
                        # Get actual tiles from MMAP
                        actual_coords = self._get_mmap_tile_coords(cluster_id, year)
                        
                        if not expected_coords:
                            issue = "No tiles in database"
                            new_status = self.config.STATUS_PENDING
                        elif not actual_coords:
                            issue = "No tiles in MMAP directory"
                            new_status = self.config.STATUS_DOWNLOADED if task.get('local_filepath') else self.config.STATUS_PENDING
                        elif actual_coords != expected_coords:
                            missing = expected_coords - actual_coords
                            extra = actual_coords - expected_coords
                            issue = f"Tile mismatch (expected {len(expected_coords)}, found {len(actual_coords)})"
                            if missing:
                                issue += f", missing {len(missing)} tiles"
                            if extra:
                                issue += f", {len(extra)} extra tiles"
                            
                            # Remove incomplete MMAP directory
                            if self.clean:
                                mmap_dir = self.config.DATA_DIR / "landsat_mmap" / str(cluster_id) / str(year)
                                if mmap_dir.exists():
                                    import shutil
                                    shutil.rmtree(mmap_dir)
                                    logger.info(f"  Removed incomplete MMAP: {mmap_dir}")
                            new_status = self.config.STATUS_DOWNLOADED if task.get('local_filepath') else self.config.STATUS_PENDING
                
                elif status == self.config.STATUS_DOWNLOADED:
                    # Check local file exists
                    local_filepath = task.get('local_filepath')
                    if not local_filepath or not Path(local_filepath).exists():
                        issue = "Downloaded file missing"
                        # Check if on Drive
                        task_desc = task.get('gee_task_description')
                        if task_desc and task_desc in drive_files:
                            new_status = self.config.STATUS_COMPLETED
                        else:
                            new_status = self.config.STATUS_PENDING
                
                elif status == self.config.STATUS_COMPLETED:
                    # Check file is on Drive
                    task_desc = task.get('gee_task_description')
                    if not task_desc or task_desc not in drive_files:
                        issue = "File not on Drive"
                        new_status = self.config.STATUS_PENDING
                
                if issue:
                    issues_found += 1
                    logger.warning(
                        f"  Issue: {country_code} {year} (cluster {cluster_id}): "
                        f"{issue} - status '{status}'"
                    )
                    
                    if self.clean and new_status:
                        self.db.update_task_status(
                            geometry_hash, year, new_status,
                            error_message=f"Janitor: {issue}"
                        )
                        logger.info(f"    Fixed: Reset to '{new_status}'")
                        issues_fixed += 1
        
        logger.info(f"Database integrity check complete: {issues_found} issues found, {issues_fixed} fixed")
        return issues_found, issues_fixed
    
    def check_orphaned_mmap(self):
        """Check for MMAP directories not in database."""
        logger.info("=== Checking for Orphaned MMAP Files ===")
        
        orphans_found = 0
        orphans_removed = 0
        
        mmap_path = self.config.DATA_DIR / "landsat_mmap"
        
        if not mmap_path.exists():
            logger.info("MMAP directory does not exist")
            return 0, 0
        
        # Get all tasks with cluster IDs from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT cluster_id, year 
                FROM tasks 
                WHERE cluster_id IS NOT NULL
            """)
            db_clusters = {(row['cluster_id'], row['year']) for row in cursor.fetchall()}
        
        logger.info(f"Database has {len(db_clusters)} cluster/year combinations")
        
        # Check each MMAP cluster directory
        for cluster_dir in mmap_path.iterdir():
            if not cluster_dir.is_dir():
                continue
            
            try:
                cluster_id = int(cluster_dir.name)
            except ValueError:
                logger.warning(f"Invalid cluster directory name: {cluster_dir.name}")
                continue
            
            # Check year subdirectories
            for year_dir in cluster_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                
                try:
                    year = int(year_dir.name)
                except ValueError:
                    logger.warning(f"Invalid year directory name: {year_dir.name}")
                    continue
                
                # Check if this cluster/year is in database
                if (cluster_id, year) not in db_clusters:
                    orphans_found += 1
                    logger.warning(f"  Orphaned MMAP: cluster {cluster_id}, year {year}")
                    
                    if self.clean:
                        import shutil
                        shutil.rmtree(year_dir)
                        logger.info(f"    Removed: {year_dir}")
                        orphans_removed += 1
            
            # Remove empty cluster directories
            if self.clean and cluster_dir.exists():
                try:
                    if not any(cluster_dir.iterdir()):
                        cluster_dir.rmdir()
                        logger.info(f"  Removed empty cluster directory: {cluster_dir}")
                except (OSError, StopIteration):
                    pass
        
        logger.info(f"Orphaned MMAP check complete: {orphans_found} found, {orphans_removed} removed")
        return orphans_found, orphans_removed
    
    def check_orphaned_downloads(self):
        """Check for downloaded files not referenced in database."""
        logger.info("=== Checking for Orphaned Download Files ===")
        
        orphans_found = 0
        orphans_removed = 0
        
        if not self.config.DOWNLOAD_DIR.exists():
            logger.info("Download directory does not exist")
            return 0, 0
        
        # Get all local_filepath values from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT local_filepath FROM tasks WHERE local_filepath IS NOT NULL")
            db_files = {Path(row['local_filepath']) for row in cursor.fetchall()}
        
        logger.info(f"Database references {len(db_files)} downloaded files")
        
        # Check each file in download directory
        for file in self.config.DOWNLOAD_DIR.glob("*.tif"):
            if file not in db_files:
                orphans_found += 1
                logger.warning(f"  Orphaned download: {file.name}")
                
                if self.clean:
                    file.unlink()
                    logger.info(f"    Removed: {file}")
                    orphans_removed += 1
        
        logger.info(f"Orphaned downloads check complete: {orphans_found} found, {orphans_removed} removed")
        return orphans_found, orphans_removed
    
    def run_checks(self) -> Dict[str, Tuple[int, int]]:
        """Run all integrity checks.
        
        Returns:
            Dict mapping check name to (issues_found, issues_fixed) tuple
        """
        results = {}
        
        # Check database integrity (checks against filesystem)
        results['database'] = self.check_database_integrity()
        
        # Check for orphaned MMAP directories
        results['mmap_orphans'] = self.check_orphaned_mmap()
        
        # Check for orphaned downloads
        results['download_orphans'] = self.check_orphaned_downloads()
        
        return results
    
    def run(self, continuous: bool = True):
        """Run the janitor worker.
        
        Args:
            continuous: If True, run continuously; otherwise run once
        """
        logger.info(f"Starting {self.worker_name} worker")
        
        while True:
            self.db.update_worker_heartbeat(self.worker_name, "running")
            
            try:
                results = self.run_checks()
                
                # Log summary
                total_issues = sum(found for found, _ in results.values())
                total_fixed = sum(fixed for _, fixed in results.values())
                
                logger.info(
                    f"\n=== Janitor Summary ===\n"
                    f"Total issues found: {total_issues}\n"
                    f"Total issues fixed: {total_fixed}\n"
                    f"  Database integrity: {results['database'][0]} found, {results['database'][1]} fixed\n"
                    f"  MMAP orphans: {results['mmap_orphans'][0]} found, {results['mmap_orphans'][1]} removed\n"
                    f"  Download orphans: {results['download_orphans'][0]} found, {results['download_orphans'][1]} removed"
                )
                
                # Update worker counters
                if total_issues > 0:
                    self.db.increment_worker_counter(self.worker_name, "tasks_processed")
                
            except Exception as e:
                logger.error(f"Error during janitor checks: {e}", exc_info=True)
                self.db.increment_worker_counter(self.worker_name, "errors")
            
            if not continuous:
                break
            
            # Sleep longer for janitor (less frequent checks)
            time.sleep(self.config.WORKER_SLEEP_INTERVAL * 10)  # 10x normal interval
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
