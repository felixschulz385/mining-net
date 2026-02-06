"""Minimal database interface for network module (read-only tile queries)."""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager


class TileDatabase:
    """Lightweight read-only database interface for querying tiles.
    
    This is a minimal version that only supports querying written tiles,
    without the heavy dependencies of the full data management system.
    """
    
    def __init__(self, db_path: str):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections.
        
        Yields:
            sqlite3.Connection with row factory enabled
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def get_written_tiles(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Get all written tiles matching filters.
        
        Args:
            countries: Filter by ISO3 country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            
        Returns:
            List of tile dicts with metadata
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query with filters
            query = """
                SELECT t.tile_ix, t.tile_iy, t.cluster_id, t.year, 
                       tasks.country_code, tasks.geometry_hash
                FROM tiles t
                JOIN tasks ON t.geometry_hash = tasks.geometry_hash 
                    AND t.year = tasks.year
                WHERE t.mmap_written = 1
            """
            params = []
            
            if countries:
                placeholders = ','.join('?' * len(countries))
                query += f" AND tasks.country_code IN ({placeholders})"
                params.extend(countries)
            
            if years:
                placeholders = ','.join('?' * len(years))
                query += f" AND t.year IN ({placeholders})"
                params.extend(years)
            
            if cluster_ids:
                placeholders = ','.join('?' * len(cluster_ids))
                query += f" AND t.cluster_id IN ({placeholders})"
                params.extend(cluster_ids)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
