"""PyTorch data provider for mining segmentation tiles."""

import logging
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


class MiningSegmentationDataLoader(Dataset):
    """PyTorch Dataset for mining segmentation from memory-mapped tiles.
    
    Provides lazy-loading access to completely processed Landsat tiles with mining 
    footprints for training machine learning models. Data is loaded on-demand per tile,
    not preloaded into memory.
    
    Uses memory-mapped PyTorch tensors for fast, zero-copy data access.
    
    Example:
        >>> from data.data_loader import MiningSegmentationDataLoader
        >>> from torch.utils.data import DataLoader
        >>> dataset = MiningSegmentationDataLoader(
        ...     countries=['ZAF'], 
        ...     years=[2020],
        ...     bands=['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        >>> for features, labels in loader:
        ...     print(features.shape, labels.shape)
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        mmap_path: Optional[str] = None,
        config: Optional[Config] = None,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None,
        bands: Optional[List[str]] = None,
        skip_invalid: bool = True,
        normalize: bool = True,
        band_means: Optional[List[float]] = None,
        band_stds: Optional[List[float]] = None,
        auto_compute_stats: bool = True,
        stats_samples: int = 100
    ):
        """Initialize data loader.
        
        Args:
            db_path: Path to database (defaults to config)
            mmap_path: Path to memory-mapped dataset (defaults to config.DATA_DIR/landsat_mmap)
            config: Configuration instance
            countries: Filter by ISO3 country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            bands: List of bands to load (default: all Landsat bands)
            skip_invalid: Whether to skip tiles with invalid data (all NaN)
            normalize: Whether to normalize inputs (default: True)
            band_means: Precomputed band means for normalization (optional)
            band_stds: Precomputed band stds for normalization (optional)
            auto_compute_stats: If True and band_means is None, compute stats during init (default: True)
            stats_samples: Number of samples to use for auto stats computation (default: 100)
        """
        self.config = config or Config()
        
        # Setup database
        db_path = db_path or str(self.config.DB_PATH)
        self.db = DownloadDatabase(db_path)
        
        # Setup storage paths
        mmap_path = mmap_path or str(self.config.DATA_DIR / "landsat_mmap")
        self.mmap_path = Path(mmap_path)
        
        if not self.mmap_path.exists():
            raise FileNotFoundError(f"MMAP dataset not found: {self.mmap_path}")
        
        logger.info(f"Using memory-mapped backend: {self.mmap_path}")
        
        # Store filter parameters
        self.bands = bands or ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
        self.skip_invalid = skip_invalid
        
        # Get filtered tiles first
        self.tiles = self.get_written_tiles(countries, years, cluster_ids)
        
        if not self.tiles:
            raise ValueError("No tiles found matching filters")
        
        # Normalization parameters (critical for convergence)
        self.normalize = normalize
        
        # Compute statistics once if not provided and normalization is enabled
        if normalize and band_means is None and auto_compute_stats:
            logger.info(f"Computing normalization statistics from {min(stats_samples, len(self.tiles))} samples...")
            self.band_means, self.band_stds = self._compute_stats_fast(max_samples=stats_samples)
            logger.info(f"Computed: means={[f'{m:.4f}' for m in self.band_means]}, stds={[f'{s:.4f}' for s in self.band_stds]}")
        else:
            self.band_means = band_means
            self.band_stds = band_stds
        
        logger.info(f"Initialized dataset with {len(self.tiles)} tiles using memory-mapped backend")
        logger.info(f"Normalization enabled: {normalize} (precomputed: {band_means is not None})")
    
    @staticmethod
    def _get_tile_path(cluster_id: int, year: int, tile_ix: int, tile_iy: int) -> str:
        """Reconstruct mmap path from tile coordinates.
        
        Args:
            cluster_id: Cluster ID
            year: Year
            tile_ix: Tile X index
            tile_iy: Tile Y index
            
        Returns:
            Relative path to tile directory: cluster_id/year/tile_ix_tile_iy
        """
        return f"{cluster_id}/{year}/{tile_ix}_{tile_iy}"
    
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
    
    def load_tile_data(
        self,
        tile_ix: int,
        tile_iy: int,
        year: int,
        cluster_id: int,
        bands: Optional[List[str]] = None,
        include_footprint: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Load data for a specific tile from memory-mapped storage.
        
        Args:
            tile_ix: Tile X index
            tile_iy: Tile Y index
            year: Year
            cluster_id: Cluster ID (needed to reconstruct path)
            bands: List of bands to load (not used, kept for compatibility)
            include_footprint: Whether to load mining footprint
            
        Returns:
            Tuple of (features, labels) where:
                - features: torch tensor of shape (C, H, W)
                - labels: torch tensor of shape (1, H, W)
        """
        # Reconstruct path from coordinates
        mmap_path = self._get_tile_path(cluster_id, year, tile_ix, tile_iy)
        tile_dir = self.mmap_path / mmap_path
        
        features = torch.load(tile_dir / "features.pt")
        labels = torch.load(tile_dir / "labels.pt") if include_footprint else None
        
        return features, labels
    
    def __len__(self) -> int:
        """Return the number of tiles in the dataset.
        
        Returns:
            Number of tiles
        """
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single tile by index (lazy loading).
        
        Args:
            idx: Tile index
            
        Returns:
            Tuple of (features, labels) as torch tensors where:
                - features: torch.Tensor of shape (C, H, W)
                - labels: torch.Tensor of shape (1, H, W)
        """
        tile = self.tiles[idx]
        
        # Reconstruct path and load from memory-mapped storage
        mmap_path = self._get_tile_path(
            tile['cluster_id'], tile['year'], tile['tile_ix'], tile['tile_iy']
        )
        tile_dir = self.mmap_path / mmap_path
        features = torch.load(tile_dir / "features.pt").float()
        labels = torch.load(tile_dir / "labels.pt").float()
        
        # Normalize features (critical for convergence)
        # Per-channel standardization using precomputed statistics
        if self.normalize and self.band_means is not None:
            mean = torch.tensor(self.band_means, dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor(self.band_stds, dtype=torch.float32).view(-1, 1, 1)
            features = (features - mean) / (std + 1e-8)
        
        return features, labels
    
    def get_tile_by_index(self, tile_ix: int, tile_iy: int, year: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a specific tile by its coordinates.
        
        Args:
            tile_ix: Tile X index
            tile_iy: Tile Y index
            year: Year
            
        Returns:
            Tuple of (features, labels) as torch tensors
        """
        # Find tile in list
        tile = next((t for t in self.tiles if t['tile_ix'] == tile_ix and t['tile_iy'] == tile_iy and t['year'] == year), None)
        
        if not tile:
            raise ValueError(f"Tile not found: {tile_ix}, {tile_iy}, {year}")
        
        features, labels = self.load_tile_data(
            tile_ix,
            tile_iy,
            year,
            tile['cluster_id'],
            bands=self.bands,
            include_footprint=True
        )
        
        return features, labels
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.tiles:
            return {
                'total_tiles': 0,
                'countries': {},
                'years': {},
                'clusters': {}
            }
        
        # Count by country
        countries_count = {}
        for tile in self.tiles:
            country = tile['country_code']
            countries_count[country] = countries_count.get(country, 0) + 1
        
        # Count by year
        years_count = {}
        for tile in self.tiles:
            year = tile['year']
            years_count[year] = years_count.get(year, 0) + 1
        
        # Count by cluster
        clusters_count = {}
        for tile in self.tiles:
            cluster = tile['cluster_id']
            clusters_count[cluster] = clusters_count.get(cluster, 0) + 1
        
        return {
            'total_tiles': len(self.tiles),
            'num_bands': len(self.bands),
            'bands': self.bands,
            'countries': countries_count,
            'years': years_count,
            'clusters': clusters_count
        }
    
    def _compute_stats_fast(self, max_samples: int = 100) -> Tuple[List[float], List[float]]:
        """Fast computation of normalization statistics without going through __getitem__.
        
        Args:
            max_samples: Maximum number of samples to use
            
        Returns:
            Tuple of (band_means, band_stds) as lists
        """
        n_samples = min(max_samples, len(self.tiles))
        indices = np.random.choice(len(self.tiles), n_samples, replace=False)
        
        # Accumulate statistics
        band_sums = np.zeros(len(self.bands))
        band_sq_sums = np.zeros(len(self.bands))
        total_pixels = 0
        
        for idx in indices:
            tile = self.tiles[idx]
            # Load directly without normalization
            features, _ = self.load_tile_data(
                tile['tile_ix'],
                tile['tile_iy'],
                tile['year'],
                tile['cluster_id'],
                bands=self.bands,
                include_footprint=False
            )
            # Convert torch tensor to numpy and transpose from (C, H, W) to (H, W, C)
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            if features.shape[0] == len(self.bands):
                features = features.transpose(1, 2, 0)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # features shape: (H, W, C)
            for c in range(features.shape[2]):
                channel = features[:, :, c]
                band_sums[c] += channel.sum()
                band_sq_sums[c] += (channel ** 2).sum()
            
            total_pixels += features.shape[0] * features.shape[1]
        
        # Compute means and stds
        band_means = band_sums / total_pixels
        band_vars = (band_sq_sums / total_pixels) - (band_means ** 2)
        band_stds = np.sqrt(np.maximum(band_vars, 0))
        
        return band_means.tolist(), band_stds.tolist()
    
    def compute_normalization_stats(self, max_samples: int = 1000) -> Tuple[List[float], List[float]]:
        """Compute mean and std for each band across dataset (for normalization).
        
        This is a user-facing method with progress bar for explicit stats computation.
        For automatic computation during init, use _compute_stats_fast instead.
        
        Args:
            max_samples: Maximum number of samples to use for computation
            
        Returns:
            Tuple of (band_means, band_stds) as lists
        """
        logger.info(f"Computing normalization statistics from {min(max_samples, len(self.tiles))} samples...")
        
        n_samples = min(max_samples, len(self.tiles))
        indices = np.random.choice(len(self.tiles), n_samples, replace=False)
        
        # Accumulate statistics
        band_sums = np.zeros(len(self.bands))
        band_sq_sums = np.zeros(len(self.bands))
        total_pixels = 0
        
        for idx in tqdm(indices, desc="Computing stats"):
            tile = self.tiles[idx]
            # Load directly without normalization to avoid recursion
            features, _ = self.load_tile_data(
                tile['tile_ix'],
                tile['tile_iy'],
                tile['year'],
                tile['cluster_id'],
                bands=self.bands,
                include_footprint=False
            )
            # Convert torch tensor to numpy and transpose from (C, H, W) to (H, W, C)
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            if features.shape[0] == len(self.bands):
                features = features.transpose(1, 2, 0)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # features shape: (H, W, C)
            for c in range(features.shape[2]):
                channel = features[:, :, c]
                band_sums[c] += channel.sum()
                band_sq_sums[c] += (channel ** 2).sum()
            
            total_pixels += features.shape[0] * features.shape[1]
        
        # Compute means and stds
        band_means = band_sums / total_pixels
        band_vars = (band_sq_sums / total_pixels) - (band_means ** 2)
        band_stds = np.sqrt(np.maximum(band_vars, 0))
        
        logger.info(f"Computed statistics from {n_samples} samples:")
        for i, band in enumerate(self.bands):
            logger.info(f"  {band}: mean={band_means[i]:.4f}, std={band_stds[i]:.4f}")
        
        return band_means.tolist(), band_stds.tolist()
    
    @staticmethod
    def create_dataloader(
        dataset: 'MiningSegmentationDataLoader',
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """Create a PyTorch DataLoader from this dataset.
        
        Args:
            dataset: MiningSegmentationDataLoader instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            
        Returns:
            torch.utils.data.DataLoader
        """
        from torch.utils.data import DataLoader
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
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
