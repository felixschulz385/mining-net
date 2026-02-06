"""Download worker to fetch completed files from Google Drive."""

import time
import logging
import os
import pickle
import io
from typing import Optional, List
from pathlib import Path


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
            # Find folder
            folder_id = self._find_drive_folder()
            if not folder_id:
                logger.error(f"Drive folder '{self.config.DRIVE_FOLDER}' not found")
                return False
            
            # List files
            files = self._list_drive_files(folder_id)
            
            # Find matching file
            expected_filename = task_data['gee_task_description']
            matching_file = None
            for file in files:
                if file['name'].startswith(expected_filename):
                    matching_file = file
                    break
            
            if not matching_file:
                # Track how many times this file wasn't found
                task_key = (task_data['geometry_hash'], task_data['year'])
                self.file_not_found_count[task_key] = self.file_not_found_count.get(task_key, 0) + 1
                
                attempts = self.file_not_found_count[task_key]
                
                if attempts >= self.MAX_FILE_NOT_FOUND_ATTEMPTS:
                    logger.warning(
                        f"File not found after {attempts} attempts: {expected_filename}. "
                        f"Resetting task to pending for re-export."
                    )
                    # Reset task to pending so it can be re-exported
                    self.db.update_task_status(
                        task_data['geometry_hash'],
                        task_data['year'],
                        self.config.STATUS_PENDING,
                        error_message=f"File not found on Drive after {attempts} attempts"
                    )
                    # Clear the counter
                    del self.file_not_found_count[task_key]
                    return False
                else:
                    logger.debug(f"File not yet available ({attempts}/{self.MAX_FILE_NOT_FOUND_ATTEMPTS}): {expected_filename}")
                    return False
            
            file_id = matching_file['id']
            filename = matching_file['name']
            
            # Ensure filename has .tif extension
            if not filename.endswith('.tif'):
                filename += '.tif'
            
            # Download path
            filepath = self.config.DOWNLOAD_DIR / filename
            
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
            
            # Delete from Drive
            try:
                self.drive_service.files().delete(fileId=file_id).execute()
                logger.info(f"  Deleted from Drive: {filename}")
            except Exception as e:
                logger.warning(f"  Could not delete {filename}: {e}")
            
            # Update database
            self.db.update_task_status(
                task_data['geometry_hash'],
                task_data['year'],
                self.config.STATUS_DOWNLOADED,
                drive_file_id=file_id,
                drive_filename=filename,
                local_filepath=str(filepath)
            )
            
            # Clear not-found counter on success
            task_key = (task_data['geometry_hash'], task_data['year'])
            if task_key in self.file_not_found_count:
                del self.file_not_found_count[task_key]
            
            self.db.increment_worker_counter(self.worker_name, "tasks_processed")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}", exc_info=True)
            self.db.increment_worker_counter(self.worker_name, "errors")
            return False
    
    def run(self, continuous: bool = True):
        """Run the download worker.
        
        Args:
            continuous: If True, run continuously; otherwise run once
        """
        logger.info(f"Starting {self.worker_name} worker")
        
        while True:
            self.db.update_worker_heartbeat(self.worker_name, "running")
            
            # Get completed tasks
            completed_tasks = self.db.get_tasks_by_status(
                self.config.STATUS_COMPLETED,
                limit=self.config.BATCH_SIZE
            )
            
            if completed_tasks:
                logger.info(f"Processing {len(completed_tasks)} completed tasks")
                for task in completed_tasks:
                    self.download_file(task)
                    time.sleep(1)  # Rate limiting
            else:
                logger.debug("No completed tasks")
            
            if not continuous:
                break
            
            time.sleep(self.config.WORKER_SLEEP_INTERVAL)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
