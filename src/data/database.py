"""Database management for tracking download tasks and status."""

import sqlite3
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class DownloadDatabase:
    """Manages SQLite database for tracking download tasks."""
    
    def __init__(self, db_path: str = "mining_segmentation.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
        self._migrate_schema()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Main tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    geometry_hash TEXT NOT NULL UNIQUE,
                    country_code TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    cluster_id INTEGER,
                    geometry_json TEXT NOT NULL,
                    mining_footprint_json TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    gee_task_id TEXT,
                    gee_task_description TEXT,
                    drive_file_id TEXT,
                    drive_filename TEXT,
                    local_filepath TEXT,
                    created_at TEXT NOT NULL,
                    submitted_at TEXT,
                    completed_at TEXT,
                    downloaded_at TEXT,
                    reprojected_at TEXT,
                    error_message TEXT,
                    UNIQUE(geometry_hash, year)
                )
            """)
            
            # Index for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON tasks(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_country_year 
                ON tasks(country_code, year)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cluster 
                ON tasks(cluster_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_geometry_hash 
                ON tasks(geometry_hash)
            """)
            
            # Tiles table - tracks 64x64 geobox tiles
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tiles (
                    tile_ix INTEGER NOT NULL,
                    tile_iy INTEGER NOT NULL,
                    geometry_hash TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    cluster_id INTEGER NOT NULL,
                    mmap_written BOOLEAN DEFAULT 0,
                    created_at TEXT NOT NULL,
                    mmap_written_at TEXT,
                    PRIMARY KEY (tile_ix, tile_iy, geometry_hash, year),
                    FOREIGN KEY (geometry_hash, year) REFERENCES tasks(geometry_hash, year)
                )
            """)
            
            # Index for tile queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tiles_cluster 
                ON tiles(cluster_id, year)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tiles_coords 
                ON tiles(tile_ix, tile_iy)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_tiles_mmap_written 
                ON tiles(mmap_written)
            """)
            
            # Worker status table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS worker_status (
                    worker_name TEXT PRIMARY KEY,
                    last_heartbeat TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tasks_processed INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0
                )
            """)
    
    def _migrate_schema(self):
        """Migrate database schema for backward compatibility.
        
        Adds new columns to existing databases without breaking them.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get current table info for tiles
            cursor.execute("PRAGMA table_info(tiles)")
            columns = {row[1] for row in cursor.fetchall()}
            
            # Add mmap columns if they don't exist (for migration from zarr-only schema)
            if 'mmap_written' not in columns:
                try:
                    cursor.execute("ALTER TABLE tiles ADD COLUMN mmap_written BOOLEAN DEFAULT 0")
                except sqlite3.OperationalError:
                    pass  # Column already exists
            
            if 'mmap_written_at' not in columns:
                try:
                    cursor.execute("ALTER TABLE tiles ADD COLUMN mmap_written_at TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
    
    @staticmethod
    def hash_geometry(geometry: Dict[str, Any]) -> str:
        """Generate a unique hash for a geometry.
        
        Args:
            geometry: GeoJSON geometry dict
            
        Returns:
            SHA256 hash of the geometry
        """
        geom_str = json.dumps(geometry, sort_keys=True)
        return hashlib.sha256(geom_str.encode()).hexdigest()
    
    def create_task(
        self,
        geometry: Dict[str, Any],
        country_code: str,
        year: int,
        cluster_id: Optional[int] = None,
        mining_footprint: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new download task.
        
        Args:
            geometry: GeoJSON geometry dict (used for hash and stored)
            country_code: ISO3 country code
            year: Year to download
            cluster_id: Optional cluster ID for grouped mines
            mining_footprint: Optional GeoJSON of combined mining polygons for cluster
            
        Returns:
            Geometry hash (task ID)
        """
        geom_hash = self.hash_geometry(geometry)
        geom_json = json.dumps(geometry)
        mining_footprint_json = json.dumps(mining_footprint) if mining_footprint else None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO tasks (
                        geometry_hash, country_code, year, 
                        cluster_id, geometry_json, mining_footprint_json, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """, (geom_hash, country_code, year, cluster_id, geom_json,
                      mining_footprint_json, datetime.utcnow().isoformat()))
            except sqlite3.IntegrityError:
                # Task already exists
                pass
        
        return geom_hash
    
    def update_task_status(
        self,
        geometry_hash: str,
        year: int,
        status: str,
        **kwargs
    ):
        """Update task status and metadata.
        
        Args:
            geometry_hash: Task ID
            year: Year
            status: New status (pending, submitted, processing, completed, 
                   downloaded, compressed, failed)
            **kwargs: Additional fields to update
        """
        fields = {"status": status}
        fields.update(kwargs)
        
        # Add timestamp based on status
        if status == "submitted" and "submitted_at" not in fields:
            fields["submitted_at"] = datetime.utcnow().isoformat()
        elif status == "completed" and "completed_at" not in fields:
            fields["completed_at"] = datetime.utcnow().isoformat()
        elif status == "downloaded" and "downloaded_at" not in fields:
            fields["downloaded_at"] = datetime.utcnow().isoformat()
        elif status in ("reprojected", "stored") and "reprojected_at" not in fields:
            # Use reprojected_at for both (backward compatibility)
            fields["reprojected_at"] = datetime.utcnow().isoformat()
        
        set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
        values = list(fields.values()) + [geometry_hash, year]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE tasks 
                SET {set_clause}
                WHERE geometry_hash = ? AND year = ?
            """, values)
    
    def get_tasks_by_status(
        self,
        status: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get tasks with a specific status.
        
        Args:
            status: Task status to filter by
            limit: Maximum number of tasks to return
            
        Returns:
            List of task dicts
        """
        query = "SELECT * FROM tasks WHERE status = ?"
        params = [status]
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_task(self, geometry_hash: str, year: int) -> Optional[Dict[str, Any]]:
        """Get a specific task.
        
        Args:
            geometry_hash: Task ID
            year: Year
            
        Returns:
            Task dict or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tasks 
                WHERE geometry_hash = ? AND year = ?
            """, (geometry_hash, year))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_task_geometry(self, geometry_hash: str, year: int) -> Optional[Dict[str, Any]]:
        """Get task geometry as GeoJSON.
        
        Args:
            geometry_hash: Task ID
            year: Year
            
        Returns:
            GeoJSON geometry dict or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT geometry_json FROM tasks 
                WHERE geometry_hash = ? AND year = ?
            """, (geometry_hash, year))
            row = cursor.fetchone()
            if row and row['geometry_json']:
                return json.loads(row['geometry_json'])
            return None
    
    def get_mining_footprint(self, geometry_hash: str, year: int) -> Optional[Dict[str, Any]]:
        """Get mining footprint geometry for a task.
        
        Args:
            geometry_hash: Task ID
            year: Year
            
        Returns:
            GeoJSON geometry dict or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT mining_footprint_json FROM tasks 
                WHERE geometry_hash = ? AND year = ?
            """, (geometry_hash, year))
            row = cursor.fetchone()
            if row and row['mining_footprint_json']:
                return json.loads(row['mining_footprint_json'])
            return None
    
    def update_worker_heartbeat(self, worker_name: str, status: str = "running"):
        """Update worker heartbeat.
        
        Args:
            worker_name: Name of the worker
            status: Worker status
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO worker_status (worker_name, last_heartbeat, status, tasks_processed, errors)
                VALUES (?, ?, ?, 0, 0)
                ON CONFLICT(worker_name) DO UPDATE SET
                    last_heartbeat = excluded.last_heartbeat,
                    status = excluded.status
            """, (worker_name, datetime.utcnow().isoformat(), status))
    
    def increment_worker_counter(self, worker_name: str, counter: str = "tasks_processed"):
        """Increment worker counter.
        
        Args:
            worker_name: Name of the worker
            counter: Counter to increment (tasks_processed or errors)
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE worker_status 
                SET {counter} = {counter} + 1
                WHERE worker_name = ?
            """, (worker_name,))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get download statistics.
        
        Returns:
            Dict with statistics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Status counts
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM tasks
                GROUP BY status
            """)
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Country counts
            cursor.execute("""
                SELECT country_code, COUNT(DISTINCT geometry_hash) as count
                FROM tasks
                GROUP BY country_code
            """)
            country_counts = {row['country_code']: row['count'] for row in cursor.fetchall()}
            
            # Year range
            cursor.execute("SELECT MIN(year) as min_year, MAX(year) as max_year FROM tasks")
            year_range = dict(cursor.fetchone())
            
            return {
                "status_counts": status_counts,
                "country_counts": country_counts,
                "year_range": year_range,
                "total_tasks": sum(status_counts.values())
            }
    
    def create_tile(
        self,
        tile_ix: int,
        tile_iy: int,
        geometry_hash: str,
        year: int,
        cluster_id: int
    ):
        """Register a tile that will be written.
        
        Args:
            tile_ix: Tile X index
            tile_iy: Tile Y index  
            geometry_hash: Geometry hash
            year: Year
            cluster_id: Cluster ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO tiles (
                        tile_ix, tile_iy, geometry_hash, year,
                        cluster_id, mmap_written, created_at
                    ) VALUES (?, ?, ?, ?, ?, 0, ?)
                """, (tile_ix, tile_iy, geometry_hash, year, 
                      cluster_id, datetime.utcnow().isoformat()))
            except sqlite3.IntegrityError:
                # Tile already exists
                pass
    
    def mark_tile_mmap_written(
        self,
        tile_ix: int,
        tile_iy: int,
        geometry_hash: str,
        year: int,
        cluster_id: int
    ):
        """Mark a tile as written to memory-mapped format (deprecated).
        
        Kept for backward compatibility. Use mark_tile_stored instead.
        
        Args:
            tile_ix: Tile X index
            tile_iy: Tile Y index
            geometry_hash: Geometry hash
            year: Year
            cluster_id: Cluster ID for the tile
        """
        self.mark_tile_stored(tile_ix, tile_iy, geometry_hash, year, cluster_id)
    
    def mark_tile_stored(
        self,
        tile_ix: int,
        tile_iy: int,
        geometry_hash: str,
        year: int,
        cluster_id: int
    ):
        """Mark a tile as stored (in Zarr format).
        
        Args:
            tile_ix: Tile X index
            tile_iy: Tile Y index
            geometry_hash: Geometry hash
            year: Year
            cluster_id: Cluster ID for the tile
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tiles 
                SET mmap_written = 1, cluster_id = ?, mmap_written_at = ?
                WHERE tile_ix = ? AND tile_iy = ? 
                  AND geometry_hash = ? AND year = ?
            """, (cluster_id, datetime.utcnow().isoformat(),
                  tile_ix, tile_iy, geometry_hash, year))
    
    def get_tiles_for_task(
        self,
        geometry_hash: str,
        year: int
    ) -> List[Dict[str, Any]]:
        """Get all tiles for a task.
        
        Args:
            geometry_hash: Geometry hash
            year: Year
            
        Returns:
            List of tile dicts
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tiles 
                WHERE geometry_hash = ? AND year = ?
            """, (geometry_hash, year))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive cluster information including all tiles and metadata.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Dict with cluster metadata and tile inventory, or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get cluster metadata from tasks
            cursor.execute("""
                SELECT DISTINCT 
                    cluster_id,
                    country_code,
                    MIN(created_at) as created_at,
                    MAX(reprojected_at) as latest_reprojected_at
                FROM tasks
                WHERE cluster_id = ?
                GROUP BY cluster_id, country_code
            """, (cluster_id,))
            
            metadata_row = cursor.fetchone()
            if not metadata_row:
                return None
            
            metadata = dict(metadata_row)
            
            # Get all tiles for this cluster with their years
            cursor.execute("""
                SELECT 
                    tile_ix,
                    tile_iy,
                    year,
                    geometry_hash,
                    mmap_written,
                    mmap_written_at
                FROM tiles
                WHERE cluster_id = ? AND mmap_written = 1
                ORDER BY year, tile_ix, tile_iy
            """, (cluster_id,))
            
            tiles = [dict(row) for row in cursor.fetchall()]
            
            # Get year range
            years = sorted(set(t['year'] for t in tiles))
            
            return {
                'cluster_id': metadata['cluster_id'],
                'country_code': metadata['country_code'],
                'created_at': metadata['created_at'],
                'latest_reprojected_at': metadata['latest_reprojected_at'],
                'years': years,
                'tile_count': len(tiles),
                'tiles': tiles
            }
    
    def get_all_cluster_years(self) -> List[Dict[str, Any]]:
        """Get all distinct cluster_id and year combinations from database.
        
        Returns:
            List of dicts with cluster_id and year
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT cluster_id, year 
                FROM tasks 
                WHERE cluster_id IS NOT NULL
                ORDER BY cluster_id, year
            """)
            return [dict(row) for row in cursor.fetchall()]
