"""PyTorch data provider for mining segmentation tiles using Zarr backend."""

import json
import logging
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import sys

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config

logger = logging.getLogger(__name__)


class MiningSegmentationDataLoader(Dataset):
    """PyTorch Dataset for mining segmentation using Zarr backend.
    
    Provides efficient lazy-loading access to Landsat tiles with mining footprints.
    Uses chunked Zarr arrays for optimized batch I/O performance.
    
    The Zarr backend offers:
    - Efficient chunked storage for batch operations
    - Cloud-native storage support (S3, GCS)
    - Single array per data type (easier management)
    - Optional compression
    - Parallel write capabilities
    
    Example:
        >>> from network.data_loader import MiningSegmentationDataLoader
        >>> from torch.utils.data import DataLoader
        >>> dataset = MiningSegmentationDataLoader(
        ...     countries=['ZAF'], 
        ...     years=[2020]
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        >>> for features, labels in loader:
        ...     print(features.shape, labels.shape)
    """
    
    def __init__(
        self,
        zarr_path: Optional[str] = None,
        config = None,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None,
        normalize: bool = True,
        band_means: Optional[List[float]] = None,
        band_stds: Optional[List[float]] = None,
        auto_compute_stats: bool = True,
        stats_samples: int = 100
    ):
        """Initialize data loader with Zarr backend.
        
        Args:
            zarr_path: Path to Zarr group (defaults to config.ZARR_STORE_PATH)
            config: Configuration instance
            countries: Filter by ISO3 country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            normalize: Whether to normalize inputs (default: True)
            band_means: Precomputed band means for normalization (optional)
            band_stds: Precomputed band stds for normalization (optional)
            auto_compute_stats: If True and band_means is None, compute stats during init (default: True)
            stats_samples: Number of samples to use for auto stats computation (default: 100)
        """
        self.config = config or Config()
        
        # Setup Zarr path - use global group by default
        zarr_path = zarr_path or str(self.config.ZARR_STORE_PATH)
        self.zarr_path = Path(zarr_path)
        
        if not self.zarr_path.exists():
            raise FileNotFoundError(f"Zarr dataset not found: {self.zarr_path}")
        
        logger.info(f"Using Zarr backend: {self.zarr_path}")
        
        # Don't open Zarr here for multiprocessing safety - open lazily per worker
        self._zarr_group = None
        self._features_array = None
        self._labels_array = None
        
        # Open temporarily to check for index arrays and get metadata
        temp_group = zarr.open_group(store=str(self.zarr_path), mode='r')
        
        # Build tile list from index arrays if available, otherwise fallback to metadata
        if 'cluster_ids' in temp_group and 'tile_ix' in temp_group:
            logger.info("Using index arrays from Zarr group")
            all_tiles = self._tiles_from_index_arrays(temp_group)
        else:
            logger.info("Index arrays not found, loading from metadata.json")
            metadata_path = self.zarr_path.parent / "metadata" / "tiles.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                all_tiles = json.load(f)
        
        # Get array shapes for logging
        self._n_bands = temp_group['features'].shape[1]
        self._feature_shape = temp_group['features'].shape
        self._feature_chunks = temp_group['features'].chunks
        self._label_shape = temp_group['labels'].shape
        self._label_chunks = temp_group['labels'].chunks
        
        # Apply filters
        self.tiles = self._filter_tiles(all_tiles, countries, years, cluster_ids)
        
        if not self.tiles:
            raise ValueError("No tiles found matching filters")
        
        # Build lookup dict for O(1) tile access by coordinates
        self._tile_lookup = self._build_tile_lookup()
        
        # Normalization parameters
        self.normalize = normalize
        
        logger.info(f"Loaded {len(self.tiles)} tiles matching filters")
        
        # Compute statistics once if not provided and normalization is enabled
        if normalize and band_means is None and auto_compute_stats:
            logger.info(f"Computing normalization statistics from {min(stats_samples, len(self.tiles))} samples...")
            self.band_means, self.band_stds = self._compute_stats_fast(max_samples=stats_samples)
            logger.info(f"Computed: means={[f'{m:.4f}' for m in self.band_means]}, stds={[f'{s:.4f}' for s in self.band_stds]}")
        else:
            self.band_means = band_means
            self.band_stds = band_stds
        
        logger.info(f"Initialized Zarr dataset with {len(self.tiles)} tiles")
        logger.info(f"  Features: shape={self._feature_shape}, chunks={self._feature_chunks}")
        logger.info(f"  Labels: shape={self._label_shape}, chunks={self._label_chunks}")
        logger.info(f"  Normalization enabled: {normalize} (precomputed: {band_means is not None})")
    
    @property
    def zarr_group(self):
        """Lazy-load Zarr group (multiprocessing-safe)."""
        if self._zarr_group is None:
            self._zarr_group = zarr.open_group(store=str(self.zarr_path), mode='r')
        return self._zarr_group
    
    @property
    def features_array(self):
        """Lazy-load features array (multiprocessing-safe)."""
        if self._features_array is None:
            self._features_array = self.zarr_group['features']
        return self._features_array
    
    @property
    def labels_array(self):
        """Lazy-load labels array (multiprocessing-safe)."""
        if self._labels_array is None:
            self._labels_array = self.zarr_group['labels']
        return self._labels_array
    
    @staticmethod
    def _filter_tiles(
        all_tiles: List[Dict[str, Any]],
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Filter tiles by countries, years, or cluster IDs.
        
        Args:
            all_tiles: List of all tile metadata
            countries: Filter by ISO3 country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            
        Returns:
            Filtered list of tile dicts
        """
        filtered = all_tiles
        
        if countries:
            filtered = [t for t in filtered if t['country_code'] in countries]
            logger.info(f"Filtered to {len(filtered)} tiles by countries: {countries}")
        
        if years:
            filtered = [t for t in filtered if t['year'] in years]
            logger.info(f"Filtered to {len(filtered)} tiles by years: {years}")
        
        if cluster_ids:
            filtered = [t for t in filtered if t['cluster_id'] in cluster_ids]
            logger.info(f"Filtered to {len(filtered)} tiles by cluster_ids: {cluster_ids}")
        
        return filtered
    
    def _build_tile_lookup(self) -> Dict[Tuple[int, int, int], int]:
        """Build lookup dict for fast tile access by coordinates.
        
        Returns:
            Dict mapping (tile_ix, tile_iy, year) -> index in self.tiles
        """
        lookup = {}
        for idx, tile in enumerate(self.tiles):
            key = (tile['tile_ix'], tile['tile_iy'], tile['year'])
            lookup[key] = idx
        return lookup
    
    def _tiles_from_index_arrays(self, zarr_group) -> List[Dict[str, Any]]:
        """Build tile list from Zarr index arrays.
        
        Args:
            zarr_group: Temporary Zarr group for reading indices
        
        Returns:
            List of tile metadata dicts from index arrays
        """
        try:
            cluster_ids = np.array(zarr_group['cluster_ids'][:])
            tile_ix = np.array(zarr_group['tile_ix'][:])
            tile_iy = np.array(zarr_group['tile_iy'][:])
            years = np.array(zarr_group['years'][:])
            
            tiles = []
            for idx in range(len(cluster_ids)):
                tiles.append({
                    'cluster_id': int(cluster_ids[idx]),
                    'tile_ix': int(tile_ix[idx]),
                    'tile_iy': int(tile_iy[idx]),
                    'year': int(years[idx]),
                    'country_code': 'UNKNOWN',  # Not tracked in index arrays
                    'geometry_hash': ''  # Not tracked in index arrays
                })
            
            logger.info(f"Built tile list from {len(tiles)} index entries")
            return tiles
        except Exception as e:
            logger.warning(f"Could not build tile list from index arrays: {e}")
            return []
    
    def __len__(self) -> int:
        """Return the number of tiles in the dataset.
        
        Returns:
            Number of tiles
        """
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single tile by index (lazy loading from Zarr).
        
        Args:
            idx: Tile index
            
        Returns:
            Tuple of (features, labels) as torch tensors where:
                - features: torch.Tensor of shape (C, H, W)
                - labels: torch.Tensor of shape (1, H, W)
        """
        # Load from Zarr (efficient, uses chunks)
        features = torch.from_numpy(np.array(self.features_array[idx])).float()
        labels = torch.from_numpy(np.array(self.labels_array[idx])).float()
        
        # Normalize features
        if self.normalize and self.band_means is not None:
            mean = torch.tensor(self.band_means, dtype=torch.float32).view(-1, 1, 1)
            std = torch.tensor(self.band_stds, dtype=torch.float32).view(-1, 1, 1)
            features = (features - mean) / (std + 1e-8)
        
        return features, labels
    
    def get_tile_by_index(self, tile_ix: int, tile_iy: int, year: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a specific tile by its coordinates (O(1) lookup).
        
        Args:
            tile_ix: Tile X index
            tile_iy: Tile Y index
            year: Year
            
        Returns:
            Tuple of (features, labels) as torch tensors
        """
        # Fast O(1) lookup using dict
        key = (tile_ix, tile_iy, year)
        idx = self._tile_lookup.get(key)
        
        if idx is None:
            raise ValueError(f"Tile not found: {tile_ix}, {tile_iy}, {year}")
        
        return self[idx]
    
    def get_tile_metadata(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific tile by index.
        
        Args:
            idx: Tile index
            
        Returns:
            Tile metadata dict
        """
        if idx < 0 or idx >= len(self.tiles):
            raise ValueError(f"Index out of range: {idx}")
        return self.tiles[idx]
    
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
                'clusters': {},
                'shape': None,
                'chunks': None
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
            'num_bands': self.features_array.shape[1],
            'countries': countries_count,
            'years': years_count,
            'clusters': clusters_count,
            'shape': self.features_array.shape,
            'chunks': self.features_array.chunks
        }
    
    def _compute_stats_fast(self, max_samples: int = 100) -> Tuple[List[float], List[float]]:
        """Fast computation of normalization statistics from Zarr arrays.
        
        Args:
            max_samples: Maximum number of samples to use
            
        Returns:
            Tuple of (band_means, band_stds) as lists
        """
        n_samples = min(max_samples, len(self.tiles))
        indices = np.random.choice(len(self.tiles), n_samples, replace=False)
        
        # Get number of bands from array shape
        n_bands = self.features_array.shape[1]
        
        # Accumulate statistics
        band_sums = np.zeros(n_bands)
        band_sq_sums = np.zeros(n_bands)
        total_pixels = 0
        
        for idx in indices:
            # Load from Zarr
            features = np.array(self.features_array[idx])  # Shape: (C, H, W)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute stats per channel
            for c in range(n_bands):
                channel = features[c]
                band_sums[c] += channel.sum()
                band_sq_sums[c] += (channel ** 2).sum()
            
            total_pixels += features.shape[1] * features.shape[2]
        
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
        
        # Get number of bands from cached shape
        n_bands = self._n_bands
        
        # Accumulate statistics
        band_sums = np.zeros(n_bands)
        band_sq_sums = np.zeros(n_bands)
        total_pixels = 0
        
        for idx in tqdm(indices, desc="Computing stats"):
            # Load from Zarr
            features = np.array(self.features_array[idx])  # Shape: (C, H, W)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Compute stats per channel
            for c in range(n_bands):
                channel = features[c]
                band_sums[c] += channel.sum()
                band_sq_sums[c] += (channel ** 2).sum()
            
            total_pixels += features.shape[1] * features.shape[2]
        
        # Compute means and stds
        band_means = band_sums / total_pixels
        band_vars = (band_sq_sums / total_pixels) - (band_means ** 2)
        band_stds = np.sqrt(np.maximum(band_vars, 0))
        
        logger.info(f"Computed statistics from {n_samples} samples:")
        for i in range(n_bands):
            logger.info(f"  Band {i}: mean={band_means[i]:.4f}, std={band_stds[i]:.4f}")
        
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
    
    def list_available_tiles(self) -> List[Dict[str, Any]]:
        """List all tiles in the dataset with their metadata.
        
        Returns:
            List of tile metadata dicts
        """
        return self.tiles
    
    def get_tile_count_by_country(self) -> Dict[str, int]:
        """Get tile count grouped by country.
        
        Returns:
            Dict mapping country code to tile count
        """
        counts = {}
        for tile in self.tiles:
            country = tile['country_code']
            counts[country] = counts.get(country, 0) + 1
        return counts
    
    def get_tile_count_by_year(self) -> Dict[int, int]:
        """Get tile count grouped by year.
        
        Returns:
            Dict mapping year to tile count
        """
        counts = {}
        for tile in self.tiles:
            year = tile['year']
            counts[year] = counts.get(year, 0) + 1
        return counts
