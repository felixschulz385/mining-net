"""Utility to read and validate manifest files.

Lightweight manifest reader with no heavy dependencies.
Used by data_loader for efficient tile indexing on HPC.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ManifestReader:
    """Reader for cluster manifest files with caching."""
    
    def __init__(self, manifests_dir: Path, cache_manifests: bool = True):
        """Initialize manifest reader.
        
        Args:
            manifests_dir: Directory containing manifest files
            cache_manifests: Whether to cache manifests in memory (default: True)
        """
        self.manifests_dir = Path(manifests_dir)
        self.cache_manifests = cache_manifests
        self._manifest_cache: Dict[int, Dict[str, Any]] = {}
        if not self.manifests_dir.exists():
            logger.warning(f"Manifests directory does not exist: {manifests_dir}")
    
    def read_manifest(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Read a manifest file for a cluster (with caching).
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Manifest dict or None if not found
        """
        # Check cache first
        if self.cache_manifests and cluster_id in self._manifest_cache:
            return self._manifest_cache[cluster_id]
        
        manifest_path = self.manifests_dir / f"cluster_{cluster_id}_manifest.json"
        
        if not manifest_path.exists():
            logger.warning(f"Manifest not found: {manifest_path}")
            return None
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Cache if enabled
            if self.cache_manifests:
                self._manifest_cache[cluster_id] = manifest
            
            logger.debug(f"Read manifest for cluster {cluster_id}: {manifest['tile_count']} tiles")
            return manifest
            
        except Exception as e:
            logger.error(f"Error reading manifest {manifest_path}: {e}")
            return None
    
    def list_all_manifests(self) -> List[Dict[str, Any]]:
        """List all available manifests (lightweight metadata only).
        
        Returns:
            List of manifest metadata (cluster_id, country, tile_count, years)
        """
        if not self.manifests_dir.exists():
            return []
        
        manifests = []
        
        for manifest_path in self.manifests_dir.glob("cluster_*_manifest.json"):
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                manifests.append({
                    'cluster_id': manifest['cluster_id'],
                    'country_code': manifest['country_code'],
                    'tile_count': manifest['tile_count'],
                    'years': manifest['years'],
                    'generated_at': manifest['generated_at']
                })
            except Exception as e:
                logger.error(f"Error reading manifest {manifest_path}: {e}")
        
        return sorted(manifests, key=lambda x: x['cluster_id'])
    
    def read_manifests_batch(self, cluster_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Read multiple manifests efficiently.
        
        Args:
            cluster_ids: List of cluster IDs to read
            
        Returns:
            Dict mapping cluster_id to manifest data
        """
        results = {}
        for cluster_id in cluster_ids:
            manifest = self.read_manifest(cluster_id)
            if manifest:
                results[cluster_id] = manifest
        return results
    
    def validate_manifest(self, cluster_id: int, data_dir: Path) -> Dict[str, Any]:
        """Validate that a manifest matches the actual data on disk.
        
        Args:
            cluster_id: Cluster ID
            data_dir: Data directory containing landsat_mmap
            
        Returns:
            Validation results dict
        """
        manifest = self.read_manifest(cluster_id)
        if not manifest:
            return {
                'valid': False,
                'error': 'Manifest not found'
            }
        
        cluster_dir = data_dir / "landsat_mmap" / str(cluster_id)
        if not cluster_dir.exists():
            return {
                'valid': False,
                'error': f'Cluster directory not found: {cluster_dir}'
            }
        
        # Check each tile
        missing_tiles = []
        extra_tiles = []
        
        # Track tiles from manifest
        manifest_tiles = set()
        for tile in manifest['tiles']:
            tile_key = (tile['tile_ix'], tile['tile_iy'], tile['year'])
            manifest_tiles.add(tile_key)
            
            tile_dir = cluster_dir / str(tile['year']) / f"{tile['tile_ix']}_{tile['tile_iy']}"
            
            if not tile_dir.exists():
                missing_tiles.append(tile_key)
            elif not (tile_dir / "features.pt").exists() or not (tile_dir / "labels.pt").exists():
                missing_tiles.append(tile_key)
        
        # Check for tiles on disk not in manifest
        for year_dir in cluster_dir.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            
            year = int(year_dir.name)
            for tile_dir in year_dir.iterdir():
                if not tile_dir.is_dir():
                    continue
                
                try:
                    tile_ix, tile_iy = map(int, tile_dir.name.split('_'))
                    tile_key = (tile_ix, tile_iy, year)
                    
                    if tile_key not in manifest_tiles:
                        extra_tiles.append(tile_key)
                except ValueError:
                    logger.warning(f"Invalid tile directory name: {tile_dir.name}")
        
        valid = len(missing_tiles) == 0 and len(extra_tiles) == 0
        
        return {
            'valid': valid,
            'cluster_id': cluster_id,
            'manifest_tile_count': manifest['tile_count'],
            'missing_tiles': missing_tiles,
            'extra_tiles': extra_tiles,
            'error': None if valid else f"Missing: {len(missing_tiles)}, Extra: {len(extra_tiles)}"
        }
    
    def get_cluster_summary(self, cluster_id: int) -> Optional[str]:
        """Get a human-readable summary of a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Summary string or None
        """
        manifest = self.read_manifest(cluster_id)
        if not manifest:
            return None
        
        summary = f"""
Cluster {cluster_id} ({manifest['country_code']})
  Tiles: {manifest['tile_count']}
  Years: {', '.join(map(str, manifest['years']))}
  Generated: {manifest['generated_at']}
  Latest data: {manifest.get('latest_reprojected_at', 'N/A')}
"""
        
        # Group tiles by year
        tiles_by_year = {}
        for tile in manifest['tiles']:
            year = tile['year']
            if year not in tiles_by_year:
                tiles_by_year[year] = []
            tiles_by_year[year].append((tile['tile_ix'], tile['tile_iy']))
        
        summary += "\n  Tiles by year:\n"
        for year in sorted(tiles_by_year.keys()):
            summary += f"    {year}: {len(tiles_by_year[year])} tiles\n"
        
        return summary


if __name__ == '__main__':
    """Command-line interface for manifest reader."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Read and validate cluster manifests')
    parser.add_argument('--manifests-dir', type=str, required=True, help='Manifests directory')
    parser.add_argument('--cluster', type=int, help='Specific cluster ID to inspect')
    parser.add_argument('--list', action='store_true', help='List all manifests')
    parser.add_argument('--validate', type=int, help='Validate cluster ID against disk data')
    parser.add_argument('--data-dir', type=str, help='Data directory (required for --validate)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    reader = ManifestReader(Path(args.manifests_dir))
    
    if args.list:
        manifests = reader.list_all_manifests()
        print(f"\nFound {len(manifests)} manifests:\n")
        for m in manifests:
            print(f"  Cluster {m['cluster_id']:>10} ({m['country_code']}) - {m['tile_count']:>4} tiles, years {m['years']}")
    
    elif args.cluster:
        summary = reader.get_cluster_summary(args.cluster)
        if summary:
            print(summary)
        else:
            print(f"Manifest not found for cluster {args.cluster}")
            sys.exit(1)
    
    elif args.validate is not None:
        if not args.data_dir:
            print("Error: --data-dir required for validation")
            sys.exit(1)
        
        result = reader.validate_manifest(args.validate, Path(args.data_dir))
        
        if result['valid']:
            print(f"✓ Cluster {args.validate} is valid ({result['manifest_tile_count']} tiles)")
        else:
            print(f"✗ Cluster {args.validate} validation failed: {result['error']}")
            if result.get('missing_tiles'):
                print(f"  Missing tiles: {result['missing_tiles'][:10]}...")
            if result.get('extra_tiles'):
                print(f"  Extra tiles: {result['extra_tiles'][:10]}...")
            sys.exit(1)
    
    else:
        parser.print_help()
