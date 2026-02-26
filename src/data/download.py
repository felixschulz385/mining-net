"""Download worker to fetch completed files from Google Drive."""

import time
import logging
import os
import pickle
import io
import json
from typing import Optional, List
from pathlib import Path

import rioxarray as rxr  # used for optional post-download compression


from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


class DownloadWorker:
    """Worker to download completed files from Google Drive."""
    
    # Max attempts to find file on Drive before resetting task
    MAX_FILE_NOT_FOUND_ATTEMPTS = 3
    
    def __init__(self, db: DownloadDatabase, config: Optional[Config] = None, countries: Optional[List[str]] = None):
        """Initialize download worker.
        
        Args:
            db: Database instance
            config: Configuration instance
            countries: Optional list of ISO3 country codes to filter tasks
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.worker_name = "download"
        self.drive_service = None
        # Track how many times we've checked for each task
        self.file_not_found_count = {}
        
        self._authenticate()
        logger.info("Initialized Google Drive API")
    
    def _authenticate(self):
        """Authenticate with Google Drive."""
        creds = None
        
        # Load credentials from token file
        if self.config.TOKEN_FILE.exists():
            with open(self.config.TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        
        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.config.CREDENTIALS_FILE.exists():
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.config.CREDENTIALS_FILE}. "
                        "Download from Google Cloud Console."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.config.CREDENTIALS_FILE), 
                    self.config.DRIVE_SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save credentials
            with open(self.config.TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        
        self.drive_service = build('drive', 'v3', credentials=creds)
    
    def _find_drive_folder(self) -> Optional[str]:
        """Find the Drive folder ID.
        
        Returns:
            Folder ID or None
        """
        folder_query = f"name='{self.config.DRIVE_FOLDER}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = self.drive_service.files().list(
            q=folder_query,
            spaces='drive',
            fields='files(id, name)',
            pageSize=10
        ).execute()
        
        folders = results.get('files', [])
        if folders:
            return folders[0]['id']
        return None
    
    def _list_drive_files(self, folder_id: str) -> list:
        """List files in Drive folder.
        
        Args:
            folder_id: Drive folder ID
            
        Returns:
            List of file dicts
        """
        file_query = f"'{folder_id}' in parents and trashed=false"
        results = self.drive_service.files().list(
            q=file_query,
            spaces='drive',
            fields='files(id, name, mimeType)',
            pageSize=1000
        ).execute()
        
        return results.get('files', [])
    
    def download_file(self, task_data: dict) -> bool:
        """Download a completed file from Drive.
        
        Args:
            task_data: Task data from database
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cluster_id = task_data['cluster_id']
            year = task_data['year']
            country_code = task_data['country_code']
            drive_filename = task_data['drive_filename']
            local_filepath = Path(task_data['local_filepath'])
            
            # Find folder
            folder_id = self._find_drive_folder()
            if not folder_id:
                logger.error(f"Drive folder '{self.config.DRIVE_FOLDER}' not found")
                return False
            
            # List files
            files = self._list_drive_files(folder_id)
            
            # Find matching file
            matching_file = None
            for file in files:
                if file['name'].startswith(drive_filename):
                    matching_file = file
                    break
            
            if not matching_file:
                # File not found on Drive - demote to pending for status checker to verify
                logger.warning(
                    f"File not found on Drive: {drive_filename}. "
                    f"Demoting task from COMPLETED to PENDING to verify GEE status."
                )
                self.db.update_task_status(
                    task_data['cluster_id'],
                    task_data['year'],
                    self.config.STATUS_PENDING
                )
                return False
            
            file_id = matching_file['id']
            filename = matching_file['name']
            
            # Ensure filename has .tif extension
            if not filename.endswith('.tif'):
                filename += '.tif'
            
            # Use pre-computed filepath
            filepath = local_filepath
            
            # Download file
            logger.info(f"Downloading {filename}")
            request = self.drive_service.files().get_media(fileId=file_id)
            fh = io.FileIO(filepath, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    if progress % 20 == 0:  # Log every 20%
                        logger.debug(f"  {filename}: {progress}%")
            
            fh.close()
            logger.info(f"✓ Downloaded: {filepath}")

            # run compress helper which may rename / replace the file
            filepath = self._compress_file(filepath)

            # Write metadata JSON alongside the downloaded TIFF
            try:
                tiles = self.db.get_tiles_for_cluster(cluster_id)
                footprint = self.db.get_mining_footprint(cluster_id)

                metadata = {
                    "cluster_id": cluster_id,
                    "country_code": country_code,
                    "year": year,
                    "tiles": [
                        {"tile_ix": t["tile_ix"], "tile_iy": t["tile_iy"]} for t in tiles
                    ],
                    "footprint": footprint
                }

                metadata_path = filepath.with_suffix('.json')
                with open(metadata_path, 'w', encoding='utf-8') as mf:
                    json.dump(metadata, mf, indent=2)
                logger.info(f"  Wrote metadata: {metadata_path}")
            except Exception as e:
                logger.warning(f"  Could not write metadata for {filepath}: {e}")

            # Delete from Drive
            try:
                self.drive_service.files().delete(fileId=file_id).execute()
                logger.info(f"  Deleted from Drive: {filename}")
            except Exception as e:
                logger.warning(f"  Could not delete {filename}: {e}")

            # Update database
            self.db.update_task_status(
                task_data['cluster_id'],
                task_data['year'],
                self.config.STATUS_DOWNLOADED
            )

            # Clear not-found counter on success
            task_key = (cluster_id, year)
            if task_key in self.file_not_found_count:
                del self.file_not_found_count[task_key]

            self.db.increment_worker_counter(self.worker_name, "tasks_processed")
            return True
        except Exception as e:
            logger.error(f"Error downloading file: {e}", exc_info=True)
            self.db.increment_worker_counter(self.worker_name, "errors")
            return False

    def _compress_file(self, filepath: Path) -> Path:
        """Optionally compress a TIFF and return the new path.

        If ``Config.COMPRESS_DOWNLOADS`` is False the original filepath is
        returned unchanged.  When compression is enabled the TIFF is read and
        rewritten with the codec/level specified in the configuration.  The
        behavior when ``COMPRESS_KEEP_RAW`` is set controls whether the original
        image is replaced (default) or a separate ``.compressed.tif`` copy is
        left alongside it.
        """
        if not self.config.COMPRESS_DOWNLOADS:
            return filepath

        try:
            with rxr.open_rasterio(filepath, masked=True, decode_coords="all") as ds:
                arr = ds.squeeze().astype("float32")
                tmp_path = filepath.with_suffix(".tmp.tif")
                arr.rio.to_raster(
                    tmp_path,
                    compress=self.config.COMPRESS_CODEC,
                    ZSTD_LEVEL=self.config.COMPRESS_LEVEL,
                )

            if not self.config.COMPRESS_KEEP_RAW:
                os.replace(tmp_path, filepath)
                logger.info(f"  Compressed and replaced original TIFF: {filepath}")
                return filepath
            else:
                new_path = filepath.with_suffix(".compressed.tif")
                os.replace(tmp_path, new_path)
                logger.info(f"  Wrote compressed copy: {new_path}")
                return new_path
        except Exception as e:
            logger.warning(f"  Compression failed for {filepath}: {e}")
            return filepath
    
    def run(self, continuous: bool = True):
        """Run the download worker.
        
        Args:
            continuous: If True, run continuously; otherwise run once
        """
        logger.info(f"Starting {self.worker_name} worker")
        
        while True:
            self.db.update_worker_heartbeat(self.worker_name, "running")
            
            # Get completed tasks
            completed_tasks_data = self.db.get_tasks_by_status(
                self.config.STATUS_COMPLETED,
                limit=self.config.BATCH_SIZE,
                include_filenames=True
            )
            
            if completed_tasks_data:
                logger.info(f"Processing {len(completed_tasks_data)} completed tasks")
                for task in completed_tasks_data:
                    self.download_file(task)
                    time.sleep(1)  # Rate limiting
            else:
                logger.debug("No completed tasks")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
