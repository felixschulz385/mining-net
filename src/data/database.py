"""Database management for tracking mining clusters and tiles using DuckDB."""

import duckdb
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


# Global lock for database access (shared across all instances)
_db_locks = {}
_db_locks_lock = threading.Lock()


class DownloadDatabase:
    """Manages DuckDB database for tracking mining clusters, tiles, and status."""
    
    def __init__(self, db_path: str = "mining_segmentation.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get or create a lock for this specific database file
        db_key = str(self.db_path.resolve())
        with _db_locks_lock:
            if db_key not in _db_locks:
                _db_locks[db_key] = threading.RLock()
            self._lock = _db_locks[db_key]
        
        self._ensure_spatial_extension()
        self._create_tables()
    
    def _ensure_spatial_extension(self):
        """Install spatial extension once (persists across connections)."""
        conn = duckdb.connect(str(self.db_path))
        try:
            conn.execute("INSTALL spatial;")
            conn.execute("LOAD spatial;")
        finally:
            conn.close()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with retry logic and locking."""
        max_retries = 10
        initial_delay = 0.1
        max_delay = 5.0
        backoff = 2.0
        
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Acquire lock before attempting connection
                with self._lock:
                    conn = duckdb.connect(str(self.db_path))
                    try:
                        # Load spatial extension (installed once during init)
                        conn.execute("LOAD spatial;")
                        yield conn
                        conn.commit()
                        return  # Success, exit
                    except Exception:
                        try:
                            conn.rollback()
                        except:
                            pass  # DuckDB may not have active transaction
                        raise
                    finally:
                        conn.close()
            except (duckdb.IOException, OSError) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Check if it's a file access error
                    error_msg = str(e).lower()
                    if "cannot open file" in error_msg or "verwendet wird" in error_msg or "being used" in error_msg:
                        time.sleep(delay)
                        delay = min(delay * backoff, max_delay)
                        continue
                raise
        
        # If we exhausted all retries, raise the last exception
        if last_exception:
            raise last_exception
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        with self.get_connection() as conn:
            # Create sequence FIRST before referencing it in table
            conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_tasks START 1")
            
            # Tiles table - indexed by cluster_id, tile_ix, tile_iy
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tiles (
                    cluster_id BIGINT NOT NULL,
                    tile_ix INTEGER NOT NULL,
                    tile_iy INTEGER NOT NULL,
                    PRIMARY KEY (cluster_id, tile_ix, tile_iy)
                )
            """)
            
            # Create indices for efficient tile queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tiles_coords 
                ON tiles(tile_ix, tile_iy)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tiles_cluster 
                ON tiles(cluster_id)
            """)
            
            # Cluster metadata table (includes footprint)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cluster (
                    cluster_id BIGINT PRIMARY KEY,
                    country_code VARCHAR NOT NULL,
                    footprint GEOMETRY
                )
            """)
            
            # Tasks table - each task is a cluster/year realization
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id INTEGER PRIMARY KEY DEFAULT NEXTVAL('seq_tasks'),
                    cluster_id BIGINT NOT NULL,
                    year INTEGER NOT NULL,
                    status VARCHAR NOT NULL DEFAULT 'pending',
                    gee_task_id VARCHAR,
                    UNIQUE (cluster_id, year)
                )
            """)
            
            # Create index for efficient task queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_cluster_year
                ON tasks(cluster_id, year)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status)
            """)
            
            # Worker status table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS worker_status (
                    worker_name VARCHAR PRIMARY KEY,
                    last_heartbeat TIMESTAMP NOT NULL,
                    status VARCHAR NOT NULL,
                    tasks_processed INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0
                )
            """)
    
    def update_task_status(
        self,
        cluster_id: int,
        year: int,
        status: str,
        gee_task_id: Optional[str] = None
    ):
        """Update task status and optionally GEE task ID.
        
        Args:
            cluster_id: Cluster ID
            year: Year
            status: New status
            gee_task_id: Optional GEE task ID to store
        """
        with self.get_connection() as conn:
            if gee_task_id:
                conn.execute("""
                    UPDATE tasks
                    SET status = ?, gee_task_id = ?
                    WHERE cluster_id = ? AND year = ?
                """, (status, gee_task_id, cluster_id, year))
            else:
                conn.execute("""
                    UPDATE tasks
                    SET status = ?
                    WHERE cluster_id = ? AND year = ?
                """, (status, cluster_id, year))

    
    def get_current_status(self, cluster_id: int, year: int) -> Optional[Dict[str, Any]]:
        """Get the current status of a task.
        
        Args:
            cluster_id: Cluster ID
            year: Year
            
        Returns:
            Dict with status information or None
        """
        with self.get_connection() as conn:
            result = conn.execute("""
                SELECT task_id, status FROM tasks
                WHERE cluster_id = ? AND year = ?
            """, (cluster_id, year)).fetchone()
            
            if result:
                return {
                    'task_id': result[0],
                    'cluster_id': cluster_id,
                    'year': year,
                    'status': result[1]
                }
            return None
    
    def get_tasks_by_status(
        self,
        status: str,
        limit: Optional[int] = None,
        countries: Optional[List[str]] = None,
        include_geometry: bool = False,
        include_filenames: bool = False
    ) -> List[Dict[str, Any]]:
        """Get all tasks with a specific status.
        
        Args:
            status: Status to filter by
            limit: Maximum number of results
            countries: Optional list of ISO3 country codes to filter
            include_geometry: Include footprint geometry in results
            include_filenames: Include drive_filename and local_filepath in results
            
        Returns:
            List of dicts with task information
        """
        from .config import Config
        from pathlib import Path
        
        config = Config()
        
        with self.get_connection() as conn:
            # Build query based on what's needed
            if include_geometry:
                query = """
                    SELECT
                        t.task_id,
                        t.cluster_id,
                        t.year,
                        t.status,
                        c.country_code,
                        ST_AsGeoJSON(c.footprint) as footprint_json,
                        t.gee_task_id
                    FROM tasks t
                    LEFT JOIN cluster c ON t.cluster_id = c.cluster_id
                    WHERE t.status = ?
                """
            else:
                query = """
                    SELECT
                        t.task_id,
                        t.cluster_id,
                        t.year,
                        t.status,
                        c.country_code,
                        t.gee_task_id
                    FROM tasks t
                    LEFT JOIN cluster c ON t.cluster_id = c.cluster_id
                    WHERE t.status = ?
                """
            
            # Add country filter if specified
            if countries:
                placeholders = ','.join(['?' for _ in countries])
                query += f" AND c.country_code IN ({placeholders})"
                params = [status] + countries
            else:
                params = [status]
            
            if limit:
                query += f" LIMIT {limit}"
            
            results = conn.execute(query, params).fetchall()
            
            tasks = []
            for row in results:
                task = {
                    'task_id': row[0],
                    'cluster_id': row[1],
                    'year': row[2],
                    'status': row[3],
                    'country_code': row[4]
                }
                
                # Add gee_task_id
                if include_geometry:
                    task['gee_task_id'] = row[6] if len(row) > 6 else None
                    # Add geometry if requested
                    footprint_geojson = json.loads(row[5]) if row[5] else None
                    task['footprint'] = footprint_geojson
                    task['geometry_json'] = json.dumps(footprint_geojson) if footprint_geojson else None
                else:
                    task['gee_task_id'] = row[5] if len(row) > 5 else None
                
                # Add filenames if requested
                if include_filenames:
                    cluster_id_hex = format(task['cluster_id'], 'x')[:8]
                    drive_filename = f"LANDSAT_C02_T1_L2_{task['country_code']}_{cluster_id_hex}_{task['year']}"
                    task['drive_filename'] = drive_filename
                    task['local_filepath'] = str(config.DOWNLOAD_DIR / f"{drive_filename}.tif")
                    # Add geometry_hash for backward compatibility
                    task['geometry_hash'] = cluster_id_hex
                
                tasks.append(task)
            
            return tasks
    
    def get_tiles_for_cluster(
        self,
        cluster_id: int
    ) -> List[Dict[str, Any]]:
        """Get all tiles for a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            List of tile dicts
        """
        with self.get_connection() as conn:
            results = conn.execute("""
                SELECT cluster_id, tile_ix, tile_iy
                FROM tiles
                WHERE cluster_id = ?
                ORDER BY tile_ix, tile_iy
            """, (cluster_id,)).fetchall()
            
            return [
                {
                    'cluster_id': row[0],
                    'tile_ix': row[1],
                    'tile_iy': row[2]
                }
                for row in results
            ]
    
    def get_mining_footprint(
        self,
        cluster_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get mining footprint for a cluster as GeoJSON.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            GeoJSON dict or None
        """
        with self.get_connection() as conn:
            result = conn.execute("""
                SELECT ST_AsGeoJSON(footprint) as geojson
                FROM cluster
                WHERE cluster_id = ?
            """, (cluster_id,)).fetchone()
            
            if result and result[0]:
                return json.loads(result[0])
            return None
    
    def get_cluster_info(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get comprehensive cluster information.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Dict with cluster metadata and tile inventory
        """
        with self.get_connection() as conn:
            # Get cluster metadata
            metadata = conn.execute("""
                SELECT country_code
                FROM cluster
                WHERE cluster_id = ?
            """, (cluster_id,)).fetchone()
            
            if not metadata:
                return None
            
            # Get all years with stored status
            stored_years = conn.execute("""
                SELECT year
                FROM tasks
                WHERE cluster_id = ? AND status = 'stored'
                ORDER BY year
            """, (cluster_id,)).fetchall()
            
            years = [row[0] for row in stored_years]
            
            # Get tile count for this cluster
            tile_count = conn.execute("""
                SELECT COUNT(*)
                FROM tiles
                WHERE cluster_id = ?
            """, (cluster_id,)).fetchone()[0]
            
            return {
                'cluster_id': cluster_id,
                'country_code': metadata[0],
                'years': years,
                'tile_count': tile_count,
                'tiles': []  # No longer returning individual tiles
            }
    
    def get_all_cluster_years(self) -> List[Dict[str, Any]]:
        """Get all distinct cluster_id and year combinations.
        
        Returns:
            List of dicts with cluster_id and year
        """
        with self.get_connection() as conn:
            results = conn.execute("""
                SELECT DISTINCT cluster_id, year
                FROM tasks
                ORDER BY cluster_id, year
            """).fetchall()
            
            return [
                {'cluster_id': row[0], 'year': row[1]}
                for row in results
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dict with statistics
        """
        with self.get_connection() as conn:
            # Status counts from tasks
            status_counts = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM tasks
                GROUP BY status
            """).fetchall()
            
            # Country counts - distinct clusters per country
            country_counts = conn.execute("""
                SELECT country_code, COUNT(*) as count
                FROM cluster
                GROUP BY country_code
            """).fetchall()
            
            # Year range
            year_range = conn.execute("""
                SELECT MIN(year) as min_year, MAX(year) as max_year
                FROM tasks
            """).fetchone()
            
            # Total clusters
            total_clusters = conn.execute("""
                SELECT COUNT(*) FROM cluster
            """).fetchone()[0]
            
            # Total tasks
            total_tasks = conn.execute("""
                SELECT COUNT(*) FROM tasks
            """).fetchone()[0]
            
            # Total tiles
            total_tiles = conn.execute("""
                SELECT COUNT(*) FROM tiles
            """).fetchone()[0]
            
            # Stored tasks
            stored_tasks = conn.execute("""
                SELECT COUNT(*) FROM tasks WHERE status = 'stored'
            """).fetchone()[0]
            
            return {
                "status_counts": {row[0]: row[1] for row in status_counts},
                "country_counts": {row[0]: row[1] for row in country_counts},
                "year_range": {
                    "min_year": year_range[0] if year_range else None,
                    "max_year": year_range[1] if year_range else None
                },
                "total_clusters": total_clusters,
                "total_tasks": total_tasks,
                "total_tiles": total_tiles,
                "stored_tasks": stored_tasks,
                "storage_percentage": (stored_tasks / total_tasks * 100) if total_tasks > 0 else 0
            }
    
    def update_worker_heartbeat(self, worker_name: str, status: str = "running"):
        """Update worker heartbeat.
        
        Args:
            worker_name: Name of the worker
            status: Worker status
        """
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO worker_status (worker_name, last_heartbeat, status, tasks_processed, errors)
                VALUES (?, ?, ?, 0, 0)
                ON CONFLICT (worker_name) DO UPDATE SET
                    last_heartbeat = EXCLUDED.last_heartbeat,
                    status = EXCLUDED.status
            """, (worker_name, datetime.utcnow(), status))
    
    def increment_worker_counter(self, worker_name: str, counter: str = "tasks_processed"):
        """Increment worker counter.
        
        Args:
            worker_name: Name of the worker
            counter: Counter to increment (tasks_processed or errors)
        """
        with self.get_connection() as conn:
            conn.execute(f"""
                UPDATE worker_status 
                SET {counter} = {counter} + 1
                WHERE worker_name = ?
            """, (worker_name,))
    
    def create_clusters_and_tiles(
        self,
        clusters_data: List[Dict[str, Any]],
        tiles_data: List[tuple]
    ) -> int:
        """Batch insert clusters and tiles into the database.
        
        Args:
            clusters_data: List of cluster dicts with cluster_id, country_code, mining_footprint_geojson
            tiles_data: List of tuples (cluster_id, tile_ix, tile_iy)
        
        Returns:
            Number of unique clusters inserted
        """
        import pandas as pd
        
        with self.get_connection() as conn:
            # Deduplicate clusters by cluster_id
            cluster_map = {}
            for c in clusters_data:
                cluster_id = c['cluster_id']
                if cluster_id not in cluster_map:
                    cluster_map[cluster_id] = c
            
            unique_clusters = list(cluster_map.values())
            
            # Create DataFrame for cluster metadata with footprints
            clusters_data = []
            for c in unique_clusters:
                footprint = c.get('mining_footprint_geojson')
                clusters_data.append({
                    'cluster_id': c['cluster_id'],
                    'country_code': c['country_code'],
                    'footprint_json': json.dumps(footprint) if footprint else None
                })
            
            clusters_df = pd.DataFrame(clusters_data)
            
            # Insert clusters with footprints
            conn.execute("""
                INSERT INTO cluster (cluster_id, country_code, footprint)
                SELECT cluster_id, country_code, ST_GeomFromGeoJSON(footprint_json) FROM clusters_df
                ON CONFLICT (cluster_id) DO UPDATE SET
                    country_code = EXCLUDED.country_code,
                    footprint = EXCLUDED.footprint
            """)
            
            # Create DataFrame for tiles
            tiles_df = pd.DataFrame(tiles_data, columns=['cluster_id', 'tile_ix', 'tile_iy'])
            
            # Insert tiles
            conn.execute("""
                INSERT INTO tiles (cluster_id, tile_ix, tile_iy)
                SELECT cluster_id, tile_ix, tile_iy FROM tiles_df
                ON CONFLICT DO NOTHING
            """)
        
        return len(unique_clusters)
    
    def get_all_cluster_ids(self, countries: Optional[List[str]] = None) -> List[int]:
        """Get all cluster IDs, optionally filtered by country.
        
        Args:
            countries: Optional list of ISO3 country codes
            
        Returns:
            List of cluster IDs
        """
        with self.get_connection() as conn:
            if countries:
                placeholders = ','.join(['?' for _ in countries])
                query = f"""
                    SELECT cluster_id FROM cluster
                    WHERE country_code IN ({placeholders})
                """
                results = conn.execute(query, countries).fetchall()
            else:
                results = conn.execute("SELECT cluster_id FROM cluster").fetchall()
            
            return [row[0] for row in results]
    
    def create_tasks(
        self,
        tasks_data: List[tuple]
    ) -> int:
        """Batch create tasks for cluster/year combinations.
        
        Args:
            tasks_data: List of tuples (cluster_id, year, status) or (cluster_id, year)
                        If status is omitted, defaults to 'pending'
        
        Returns:
            Number of tasks actually created (not counting conflicts/duplicates)
        """
        import pandas as pd
        
        if not tasks_data:
            return 0
        
        with self.get_connection() as conn:
            # Convert tuples to DataFrame
            # Handle both 2-tuple (cluster_id, year) and 3-tuple (cluster_id, year, status)
            processed_tasks = []
            for task in tasks_data:
                if len(task) == 2:
                    cluster_id, year = task
                    status = 'pending'
                else:
                    cluster_id, year, status = task
                processed_tasks.append({
                    'cluster_id': cluster_id,
                    'year': year,
                    'status': status
                })
            
            tasks_df = pd.DataFrame(processed_tasks)
            
            # Count existing tasks before insert to determine how many were actually added
            # Build WHERE clause to check for existing cluster_id/year combinations
            existing_count = conn.execute("""
                SELECT COUNT(*)
                FROM tasks t
                WHERE EXISTS (
                    SELECT 1 FROM tasks_df
                    WHERE tasks_df.cluster_id = t.cluster_id
                    AND tasks_df.year = t.year
                )
            """).fetchone()[0]
            
            # Insert tasks with explicit NEXTVAL for task_id, skipping conflicts
            conn.execute("""
                INSERT INTO tasks (task_id, cluster_id, year, status)
                SELECT NEXTVAL('seq_tasks'), cluster_id, year, status FROM tasks_df
                ON CONFLICT (cluster_id, year) DO NOTHING
            """)
            
            # Return the number of new tasks created
            return len(processed_tasks) - existing_count
    
    def get_task_count_for_cluster(self, cluster_id: int) -> int:
        """Get number of tasks for a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Number of tasks
        """
        with self.get_connection() as conn:
            result = conn.execute("""
                SELECT COUNT(*) FROM tasks WHERE cluster_id = ?
            """, (cluster_id,)).fetchone()
            
            return result[0] if result else 0
    
    def get_task_summary(self, countries: Optional[List[str]] = None) -> Dict[str, int]:
        """Get count of tasks by status.
        
        Args:
            countries: Optional list of ISO3 country codes to filter
            
        Returns:
            Dict mapping status to count
        """
        with self.get_connection() as conn:
            if countries:
                placeholders = ','.join(['?' for _ in countries])
                query = f"""
                    SELECT t.status, COUNT(*) as count
                    FROM tasks t
                    JOIN cluster c ON t.cluster_id = c.cluster_id
                    WHERE c.country_code IN ({placeholders})
                    GROUP BY t.status
                    ORDER BY t.status
                """
                results = conn.execute(query, countries).fetchall()
            else:
                results = conn.execute("""
                    SELECT status, COUNT(*) as count
                    FROM tasks
                    GROUP BY status
                    ORDER BY status
                """).fetchall()
            
            return {row[0]: row[1] for row in results}
    
    def get_task_data(self, cluster_id: int, year: int) -> Optional[Dict[str, Any]]:
        """Get complete task data for processing.
        
        This standardized method retrieves all information needed by workers
        (gee_export, status_checker, download, store) to process a task.
        
        Args:
            cluster_id: Cluster ID
            year: Year
            
        Returns:
            Dict with complete task information or None if not found
        """
        with self.get_connection() as conn:
            result = conn.execute("""
                SELECT
                    t.task_id,
                    t.cluster_id,
                    t.year,
                    t.status,
                    c.country_code,
                    ST_AsGeoJSON(c.footprint) as footprint_json
                FROM tasks t
                LEFT JOIN cluster c ON t.cluster_id = c.cluster_id
                WHERE t.cluster_id = ? AND t.year = ?
            """, (cluster_id, year)).fetchone()
            
            if not result:
                return None
            
            footprint_geojson = json.loads(result[5]) if result[5] else None
            
            return {
                'task_id': result[0],
                'cluster_id': result[1],
                'year': result[2],
                'status': result[3],
                'country_code': result[4],
                'footprint': footprint_geojson,
                'geometry_json': json.dumps(footprint_geojson) if footprint_geojson else None
            }

