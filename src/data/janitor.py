"""Janitor worker to verify database and filesystem consistency."""

import time
import logging
from typing import Optional, List
from pathlib import Path

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


class JanitorWorker:
    """Worker to verify consistency between database and filesystem."""
    
    def __init__(self, db: DownloadDatabase, config: Optional[Config] = None, countries: Optional[List[str]] = None, clean: bool = False):
        """Initialize janitor worker.
        
        Args:
            db: Database instance
            config: Configuration instance
            countries: Optional list of ISO3 country codes to filter tasks
            clean: If True, automatically clean up stale references
        """
        self.db = db
        self.config = config or Config()
        self.countries = countries
        self.clean = clean
        self.worker_name = "janitor"
    
    def check_downloaded_files(self, clean: bool = False) -> dict:
        """Check that downloaded files referenced in database exist on disk.
        
        Args:
            clean: If True, clean up stale file references for reprojected tasks
        
        Returns:
            Dictionary with check results
        """
        logger.info("Checking downloaded files...")
        
        # Get all tasks marked as downloaded or later
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT geometry_hash, year, country_code, local_filepath, status
                FROM tasks
                WHERE local_filepath IS NOT NULL
            """
            
            if self.countries:
                placeholders = ','.join('?' * len(self.countries))
                query += f" AND country_code IN ({placeholders})"
                cursor.execute(query, self.countries)
            else:
                cursor.execute(query)
            
            tasks = cursor.fetchall()
        
        missing_files = []
        existing_files = 0
        cleaned_up = 0
        
        for task in tasks:
            filepath = Path(task['local_filepath'])
            if not filepath.exists():
                # File is missing
                if task['status'] == self.config.STATUS_REPROJECTED:
                    # This is expected - file was deleted after reprojection
                    if clean:
                        # Clean up the stale reference
                        with self.db.get_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute("""
                                UPDATE tasks
                                SET local_filepath = NULL
                                WHERE geometry_hash = ? AND year = ?
                            """, (task['geometry_hash'], task['year']))
                        cleaned_up += 1
                    else:
                        # Just note it needs cleanup
                        missing_files.append({
                            'geometry_hash': task['geometry_hash'],
                            'year': task['year'],
                            'country': task['country_code'],
                            'status': task['status'],
                            'path': str(filepath),
                            'can_clean': True
                        })
                else:
                    # This is a problem - file should exist
                    missing_files.append({
                        'geometry_hash': task['geometry_hash'],
                        'year': task['year'],
                        'country': task['country_code'],
                        'status': task['status'],
                        'path': str(filepath),
                        'can_clean': False
                    })
            else:
                existing_files += 1
        
        results = {
            'total_checked': len(tasks),
            'existing': existing_files,
            'missing': len(missing_files),
            'cleaned_up': cleaned_up,
            'missing_files': missing_files
        }
        
        if cleaned_up > 0:
            logger.info(f"Cleaned up {cleaned_up} stale file references for reprojected tasks")
        
        if missing_files:
            cleanable = sum(1 for f in missing_files if f.get('can_clean', False))
            problematic = len(missing_files) - cleanable
            
            if cleanable > 0:
                logger.info(f"Found {cleanable} stale references (reprojected tasks, can clean)")
                if not clean:
                    logger.info("  Run with --clean flag to remove these references")
            
            if problematic > 0:
                logger.warning(f"Found {problematic} missing files that should exist")
                for item in [f for f in missing_files if not f.get('can_clean', False)][:5]:
                    logger.warning(f"  Missing: {item['country']} {item['year']} ({item['status']}) - {item['path']}")
        else:
            logger.info(f"All {existing_files} downloaded files exist on disk")
        
        return results
    
    def check_mmap_tiles(self, clean: bool = False) -> dict:
        """Check that MMAP tiles on disk are indexed in database.
        
        Args:
            clean: If True, remove unindexed cluster directories from disk
        
        Returns:
            Dictionary with check results
        """
        logger.info("Checking MMAP tiles...")
        
        mmap_path = self.config.DATA_DIR / "landsat_mmap"
        
        if not mmap_path.exists():
            logger.warning(f"MMAP directory does not exist: {mmap_path}")
            return {
                'total_clusters': 0,
                'total_tiles': 0,
                'unindexed_count': 0,
                'cleaned_up': 0,
                'unindexed_clusters': []
            }
        
        # Get all cluster IDs from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT DISTINCT cluster_id FROM tiles WHERE cluster_id IS NOT NULL"
            if self.countries:
                query = """
                    SELECT DISTINCT t.cluster_id 
                    FROM tiles t
                    JOIN tasks ta ON t.geometry_hash = ta.geometry_hash AND t.year = ta.year
                    WHERE t.cluster_id IS NOT NULL AND ta.country_code IN ({})
                """.format(','.join('?' * len(self.countries)))
                cursor.execute(query, self.countries)
            else:
                cursor.execute(query)
            
            db_clusters = set(row['cluster_id'] for row in cursor.fetchall())
        
        # Scan disk for cluster directories
        disk_clusters = []
        unindexed_clusters = []
        total_tiles_on_disk = 0
        cleaned_up = 0
        
        for item in mmap_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                try:
                    cluster_id = int(item.name)
                    
                    # Count tiles in this cluster
                    tile_count = 0
                    for year_dir in item.iterdir():
                        if year_dir.is_dir() and year_dir.name.isdigit() and len(year_dir.name) == 4:
                            tile_count += sum(1 for _ in year_dir.iterdir() if _.is_dir())
                    
                    disk_clusters.append(cluster_id)
                    total_tiles_on_disk += tile_count
                    
                    if cluster_id not in db_clusters:
                        unindexed_clusters.append({
                            'cluster_id': cluster_id,
                            'path': str(item),
                            'tile_count': tile_count
                        })
                        
                        # Clean up if requested
                        if clean:
                            try:
                                import shutil
                                shutil.rmtree(item)
                                logger.info(f"  Removed unindexed cluster directory: {cluster_id} ({tile_count} tiles)")
                                cleaned_up += 1
                                # Don't count as disk_cluster since we removed it
                                disk_clusters.pop()
                                total_tiles_on_disk -= tile_count
                            except Exception as e:
                                logger.error(f"  Failed to remove cluster {cluster_id}: {e}")
                except ValueError:
                    logger.warning(f"Non-numeric cluster directory: {item.name}")
        
        results = {
            'total_clusters': len(disk_clusters),
            'total_tiles': total_tiles_on_disk,
            'unindexed_count': len(unindexed_clusters),
            'cleaned_up': cleaned_up,
            'unindexed_clusters': unindexed_clusters
        }
        
        if cleaned_up > 0:
            logger.info(f"✓ Removed {cleaned_up} unindexed cluster directories")
        
        if unindexed_clusters:
            logger.warning(f"Found {len(unindexed_clusters)} unindexed cluster directories on disk")
            for cluster in unindexed_clusters[:5]:  # Show first 5
                logger.warning(f"  Unindexed: cluster {cluster['cluster_id']} ({cluster['tile_count']} tiles)")
            if not clean:
                logger.info("  Run with --clean flag to remove these directories")
        else:
            logger.info(f"All {len(disk_clusters)} cluster directories are indexed")
        
        return results
    
    def check_orphaned_tiles(self) -> dict:
        """Check for tiles in database without corresponding tasks.
        
        Returns:
            Dictionary with check results
        """
        logger.info("Checking for orphaned tiles in database...")
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT COUNT(*) as count
                FROM tiles t
                WHERE NOT EXISTS (
                    SELECT 1 FROM tasks ta
                    WHERE ta.geometry_hash = t.geometry_hash
                    AND ta.year = t.year
                )
            """
            
            if self.countries:
                # Can't easily filter orphaned tiles by country since they don't have tasks
                # Just run the full check
                pass
            
            cursor.execute(query)
            orphan_count = cursor.fetchone()['count']
        
        results = {
            'orphaned_tiles': orphan_count
        }
        
        if orphan_count > 0:
            logger.warning(f"Found {orphan_count} orphaned tiles (no matching task)")
        else:
            logger.info("No orphaned tiles found")
        
        return results
    
    def check_mmap_written_tiles(self, clean: bool = False) -> dict:
        """Check that tiles marked as mmap_written=1 actually exist on disk.
        
        Args:
            clean: If True, reset mmap_written flag for missing tiles
        
        Returns:
            Dictionary with check results
        """
        logger.info("Checking tiles marked as written in database...")
        
        mmap_path = self.config.DATA_DIR / "landsat_mmap"
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT t.tile_ix, t.tile_iy, t.cluster_id, t.year, ta.country_code
                FROM tiles t
                JOIN tasks ta ON t.geometry_hash = ta.geometry_hash AND t.year = ta.year
                WHERE t.mmap_written = 1
            """
            
            if self.countries:
                placeholders = ','.join('?' * len(self.countries))
                query += f" AND ta.country_code IN ({placeholders})"
                cursor.execute(query, self.countries)
            else:
                cursor.execute(query)
            
            tiles = cursor.fetchall()
        
        missing_tiles = []
        existing_tiles = 0
        cleaned_up = 0
        
        for tile in tiles:
            tile_dir = mmap_path / str(tile['cluster_id']) / str(tile['year']) / f"{tile['tile_ix']}_{tile['tile_iy']}"
            features_file = tile_dir / "features.pt"
            labels_file = tile_dir / "labels.pt"
            
            if not features_file.exists() or not labels_file.exists():
                missing_tiles.append({
                    'cluster_id': tile['cluster_id'],
                    'year': tile['year'],
                    'tile_ix': tile['tile_ix'],
                    'tile_iy': tile['tile_iy'],
                    'country': tile['country_code'],
                    'path': str(tile_dir)
                })
                
                if clean:
                    # Reset mmap_written flag
                    with self.db.get_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            UPDATE tiles
                            SET mmap_written = 0
                            WHERE tile_ix = ? AND tile_iy = ? AND year = ? AND cluster_id = ?
                        """, (tile['tile_ix'], tile['tile_iy'], tile['year'], tile['cluster_id']))
                    cleaned_up += 1
            else:
                existing_tiles += 1
        
        results = {
            'total_checked': len(tiles),
            'existing': existing_tiles,
            'missing': len(missing_tiles),
            'cleaned_up': cleaned_up,
            'missing_tiles': missing_tiles
        }
        
        if cleaned_up > 0:
            logger.info(f"Reset mmap_written flag for {cleaned_up} missing tiles")
        
        if missing_tiles:
            logger.warning(f"Found {len(missing_tiles)} tiles marked as written but missing on disk")
            for item in missing_tiles[:5]:
                logger.warning(f"  Missing: {item['country']} {item['year']} cluster {item['cluster_id']} tile {item['tile_ix']}_{item['tile_iy']}")
            if not clean:
                logger.info("  Run with --clean flag to reset mmap_written flags")
        else:
            logger.info(f"All {existing_tiles} tiles marked as written exist on disk")
        
        return results
    
    def check_consistency(self) -> dict:
        """Run all consistency checks.
        
        Returns:
            Dictionary with all check results
        """
        logger.info("="*60)
        logger.info("Starting consistency checks")
        if self.countries:
            logger.info(f"Filtering by countries: {', '.join(self.countries)}")
        if self.clean:
            logger.info("Auto-cleanup enabled")
        logger.info("="*60)
        
        results = {
            'downloaded_files': self.check_downloaded_files(clean=self.clean),
            'mmap_tiles': self.check_mmap_tiles(clean=self.clean),
            'mmap_written_tiles': self.check_mmap_written_tiles(clean=self.clean),
            'orphaned_tiles': self.check_orphaned_tiles()
        }
        
        # Summary
        logger.info("="*60)
        logger.info("Consistency Check Summary")
        logger.info("="*60)
        
        logger.info(f"Downloaded files: {results['downloaded_files']['existing']}/{results['downloaded_files']['total_checked']} exist")
        if results['downloaded_files']['cleaned_up'] > 0:
            logger.info(f"  ✓ Cleaned up {results['downloaded_files']['cleaned_up']} stale references")
        if results['downloaded_files']['missing'] > 0:
            cleanable = sum(1 for f in results['downloaded_files']['missing_files'] if f.get('can_clean', False))
            problematic = results['downloaded_files']['missing'] - cleanable
            if cleanable > 0 and not self.clean:
                logger.info(f"  ℹ️  {cleanable} stale references (run with --clean to remove)")
            if problematic > 0:
                logger.warning(f"  ⚠️  {problematic} missing files")
        
        logger.info(f"MMAP clusters: {results['mmap_tiles']['total_clusters']} on disk")
        logger.info(f"MMAP tiles: {results['mmap_tiles']['total_tiles']} on disk")
        if results['mmap_tiles']['cleaned_up'] > 0:
            logger.info(f"  ✓ Removed {results['mmap_tiles']['cleaned_up']} unindexed clusters")
        if results['mmap_tiles']['unindexed_count'] > 0:
            logger.warning(f"  ⚠️  {results['mmap_tiles']['unindexed_count']} unindexed clusters")
        
        logger.info(f"MMAP written tiles: {results['mmap_written_tiles']['existing']}/{results['mmap_written_tiles']['total_checked']} exist")
        if results['mmap_written_tiles']['cleaned_up'] > 0:
            logger.info(f"  ✓ Reset {results['mmap_written_tiles']['cleaned_up']} missing tile flags")
        if results['mmap_written_tiles']['missing'] > 0:
            if not self.clean:
                logger.warning(f"  ⚠️  {results['mmap_written_tiles']['missing']} tiles marked as written but missing (run with --clean to reset)")
            else:
                logger.warning(f"  ⚠️  {results['mmap_written_tiles']['missing']} tiles missing")
        
        if results['orphaned_tiles']['orphaned_tiles'] > 0:
            logger.warning(f"  ⚠️  {results['orphaned_tiles']['orphaned_tiles']} orphaned tiles in database")
        
        # Overall status
        total_issues = (
            sum(1 for f in results['downloaded_files']['missing_files'] if not f.get('can_clean', False)) +
            results['mmap_tiles']['unindexed_count'] +
            results['mmap_written_tiles']['missing'] +
            results['orphaned_tiles']['orphaned_tiles']
        )
        
        if total_issues == 0:
            logger.info("✓ All consistency checks passed")
        else:
            logger.warning(f"⚠️  Found {total_issues} total issues")
        
        logger.info("="*60)
        
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
                self.check_consistency()
                self.db.increment_worker_counter(self.worker_name, "tasks_processed")
            except Exception as e:
                logger.error(f"Error during consistency check: {e}", exc_info=True)
                self.db.increment_worker_counter(self.worker_name, "errors")
            
            if not continuous:
                break
            
            # Run less frequently than other workers (every 5 minutes)
            time.sleep(300)
        
        self.db.update_worker_heartbeat(self.worker_name, "stopped")
        logger.info(f"Stopped {self.worker_name} worker")
