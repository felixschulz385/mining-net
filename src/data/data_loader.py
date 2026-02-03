"""TensorFlow data provider for mining segmentation tiles."""

import logging
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


class MiningSegmentationDataLoader:
    """TensorFlow data provider for mining segmentation from zarr tiles.
    
    Provides access to completely processed Landsat tiles with mining footprints
    for training machine learning models.
    
    Example:
        >>> from gnt.data.download.mining_segmentation import MiningSegmentationDataLoader
        >>> loader = MiningSegmentationDataLoader()
        >>> dataset = loader.create_tf_dataset(countries=['ZAF'], years=[2020])
        >>> for features, labels in dataset.take(5):
        ...     print(features.shape, labels.shape)
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        zarr_path: Optional[str] = None,
        config: Optional[Config] = None
    ):
        """Initialize data loader.
        
        Args:
            db_path: Path to database (defaults to config)
            zarr_path: Path to zarr dataset (defaults to config)
            config: Configuration instance
        """
        self.config = config or Config()
        
        # Setup database
        db_path = db_path or str(self.config.DB_PATH)
        self.db = DownloadDatabase(db_path)
        
        # Setup zarr path
        zarr_path = zarr_path or str(self.config.DATA_DIR / "global_landsat.zarr")
        self.zarr_path = Path(zarr_path)
        
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr dataset not found: {self.zarr_path}")
        
        # Open zarr dataset
        self.zarr_ds = xr.open_zarr(str(self.zarr_path), consolidated=False)
        
        logger.info(f"Initialized data loader with zarr: {self.zarr_path}")
    
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
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query with filters
            query = """
                SELECT t.tile_ix, t.tile_iy, t.cluster_id, t.year, 
                       tasks.country_code, tasks.geometry_hash
                FROM tiles t
                JOIN tasks ON t.geometry_hash = tasks.geometry_hash 
                    AND t.year = tasks.year
                WHERE t.zarr_written = 1
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
    
    def load_tile_data(
        self,
        tile_ix: int,
        tile_iy: int,
        bands: Optional[List[str]] = None,
        include_footprint: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load data for a specific tile from zarr.
        
        Args:
            tile_ix: Tile X index
            tile_iy: Tile Y index
            bands: List of bands to load (default: all Landsat bands)
            include_footprint: Whether to load mining footprint
            
        Returns:
            Tuple of (features, labels) where:
                - features: numpy array of shape (H, W, n_bands)
                - labels: numpy array of shape (H, W, 1) or None
        """
        from odc.geo.geobox import GeoBox, GeoboxTiles
        
        # Reconstruct world geobox
        world_geobox = GeoBox.from_bbox(
            [-180, -90, 180, 90],
            resolution=self.config.WORLD_GEOBOX_RESOLUTION,
            crs=4326
        )
        world_geobox_tiles = GeoboxTiles(
            world_geobox,
            tile_shape=self.config.WORLD_GEOBOX_TILE_SIZE
        )
        
        # Get tile geobox
        tile_geobox = world_geobox_tiles[tile_ix, tile_iy]
        bounds = tile_geobox.boundingbox
        
        # Default bands
        if bands is None:
            bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
        
        # Load band data - bands are guaranteed to exist as variables
        band_arrays = [
            self.zarr_ds[band].sel(
                latitude=slice(bounds.top, bounds.bottom),
                longitude=slice(bounds.left, bounds.right)
            ).values
            for band in bands
        ]
        
        # Stack bands to (H, W, n_bands)
        features = np.stack(band_arrays, axis=-1).astype(np.float32)
        
        # Load mining footprint if requested
        labels = None
        if include_footprint:
            labels = self.zarr_ds['mining_footprint'].sel(
                latitude=slice(bounds.top, bounds.bottom),
                longitude=slice(bounds.left, bounds.right)
            ).values
            
            # Add channel dimension: (H, W) -> (H, W, 1)
            if labels.ndim == 2:
                labels = labels[..., np.newaxis]
            
            labels = labels.astype(np.float32)
        
        return features, labels
    
    def create_tf_dataset(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None,
        bands: Optional[List[str]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        shuffle_buffer_size: int = 1000,
        prefetch_size: int = 2,
        skip_invalid: bool = True
    ):
        """Create a TensorFlow dataset from written tiles.
        
        Args:
            countries: Filter by ISO3 country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            bands: List of bands to load (default: all Landsat bands)
            batch_size: Batch size
            shuffle: Whether to shuffle tiles
            shuffle_buffer_size: Buffer size for shuffling
            prefetch_size: Number of batches to prefetch (-1 for AUTOTUNE)
            skip_invalid: Whether to skip tiles with invalid data (all NaN)
            
        Returns:
            tf.data.Dataset yielding (features, labels) tuples with shapes:
                - features: (batch_size, H, W, n_bands)
                - labels: (batch_size, H, W, 1)
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for create_tf_dataset. "
                "Install with: pip install tensorflow"
            )
        
        # Get all written tiles
        tiles = self.get_written_tiles(countries, years, cluster_ids)
        
        if not tiles:
            raise ValueError("No written tiles found matching filters")
        
        logger.info(f"Creating TensorFlow dataset from {len(tiles)} tiles")
        
        # Create tile index list
        tile_indices = [(t['tile_ix'], t['tile_iy']) for t in tiles]
        
        # Define generator function
        def tile_generator():
            """Generator that yields validated tiles."""
            yielded = 0
            skipped = 0
            
            for tile_ix, tile_iy in tile_indices:
                try:
                    features, labels = self.load_tile_data(
                        tile_ix, tile_iy, bands=bands, include_footprint=True
                    )
                    
                    # Skip tiles with all NaN features
                    if skip_invalid and np.all(np.isnan(features)):
                        skipped += 1
                        continue
                    
                    # Replace NaN/inf with 0
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    yielded += 1
                    yield features, labels
                    
                except Exception as e:
                    skipped += 1
                    logger.warning(f"Error loading tile ({tile_ix}, {tile_iy}): {e}")
                    continue
            
            logger.info(f"Dataset: {yielded} tiles yielded, {skipped} skipped")
        
        # Determine output shapes
        n_bands = 7 if bands is None else len(bands)
        tile_size = self.config.WORLD_GEOBOX_TILE_SIZE[0]
        
        # Create dataset with explicit output signature
        dataset = tf.data.Dataset.from_generator(
            tile_generator,
            output_signature=(
                tf.TensorSpec(shape=(tile_size, tile_size, n_bands), dtype=tf.float32),
                tf.TensorSpec(shape=(tile_size, tile_size, 1), dtype=tf.float32)
            )
        )
        
        # Apply transformations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
        
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.prefetch(tf.data.AUTOTUNE if prefetch_size == -1 else prefetch_size)
        
        logger.info(f"Dataset created: batch_size={batch_size}, shuffle={shuffle}")
        
        return dataset
    
    def get_tile_statistics(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Get statistics about available tiles.
        
        Args:
            countries: Filter by ISO3 country codes
            years: Filter by years
            
        Returns:
            Dictionary with tile statistics
        """
        tiles = self.get_written_tiles(countries, years)
        
        if not tiles:
            return {
                'total_tiles': 0,
                'countries': {},
                'years': {},
                'clusters': {}
            }
        
        # Count by country
        countries_count = {}
        for tile in tiles:
            country = tile['country_code']
            countries_count[country] = countries_count.get(country, 0) + 1
        
        # Count by year
        years_count = {}
        for tile in tiles:
            year = tile['year']
            years_count[year] = years_count.get(year, 0) + 1
        
        # Count by cluster
        clusters_count = {}
        for tile in tiles:
            cluster = tile['cluster_id']
            clusters_count[cluster] = clusters_count.get(cluster, 0) + 1
        
        return {
            'total_tiles': len(tiles),
            'countries': countries_count,
            'years': years_count,
            'clusters': clusters_count
        }
