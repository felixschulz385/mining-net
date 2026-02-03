"""Task creation and clustering logic for mining segmentation."""

import logging
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import List, Optional
from scipy.sparse.csgraph import connected_components
from odc.geo.geobox import GeoBox, GeoboxTiles, geobox_union_conservative
from odc.geo.geom import Geometry
from shapely.geometry import box

from .database import DownloadDatabase
from .config import Config

logger = logging.getLogger(__name__)


def load_and_filter_mining(
    mining_file: str,
    countries: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """Load and filter mining polygons.
    
    Args:
        mining_file: Path to mining polygons GeoPackage
        countries: List of ISO3 country codes (None for all)
    
    Returns:
        Filtered GeoDataFrame with mining polygons
    
    Raises:
        FileNotFoundError: If mining file not found
        ValueError: If no polygons match filters
    """
    if not Path(mining_file).exists():
        raise FileNotFoundError(f"Mining file not found: {mining_file}")
    
    logger.info(f"Loading mining polygons from {mining_file}")
    mining = gpd.read_file(mining_file)
    
    # Filter by countries if specified
    if countries:
        mining = mining[mining['ISO3_CODE'].isin(countries)]
        if len(mining) == 0:
            raise ValueError(f"No mining polygons found for countries: {countries}")
        logger.info(f"Filtered to {len(mining)} polygons for countries: {countries}")
    
    if len(mining) == 0:
        raise ValueError("No mining polygons to process")
    
    logger.info(f"Processing {len(mining)} mines")
    return mining


def align_to_geobox_tiles(
    mining_gdf: gpd.GeoDataFrame,
    world_geobox: GeoBox,
    world_geobox_tiles: GeoboxTiles,
    buffer_size: float,
    config: Config
) -> tuple[list, list]:
    """Align mining envelopes to world geobox tiles.
    
    Args:
        mining_gdf: GeoDataFrame with mining polygons
        world_geobox: Global GeoBox for alignment
        world_geobox_tiles: GeoboxTiles instance
        buffer_size: Buffer size in degrees
        config: Configuration instance
    
    Returns:
        Tuple of (aligned_geoboxes, mine_tiles)
        - aligned_geoboxes: List of GeoBox objects aligned to tiles
        - mine_tiles: List of tile indices for each mine
    """
    logger.info(f"Aligning {len(mining_gdf)} mine envelopes to geobox tiles")
    
    # Buffer and get envelopes
    buffered = mining_gdf.buffer(buffer_size, cap_style="square").envelope
    
    # Align each envelope to world geobox tiles and store tile indices
    aligned_geoboxes = []
    mine_tiles = []  # Store tile indices for each mine
    
    for idx, geom in enumerate(buffered):
        # Get tiles for this geometry
        tiles = list(world_geobox_tiles.tiles(Geometry(geom, crs=4326)))
        
        if tiles:
            # Store tile indices for this mine
            mine_tiles.append(tiles)
            
            # Get union of all tile geoboxes
            tile_geoboxes = [
                world_geobox_tiles[tile_ix, tile_iy]
                for tile_ix, tile_iy in tiles
            ]
            aligned_geobox = geobox_union_conservative(tile_geoboxes)
        else:
            mine_tiles.append([])
            # Create a minimal geobox for this geometry
            bounds = geom.bounds
            aligned_geobox = GeoBox.from_bbox(bounds, crs=4326, resolution=config.WORLD_GEOBOX_RESOLUTION)
        
        aligned_geoboxes.append(aligned_geobox)
    
    logger.info(f"Aligned to {len(set(t for tiles in mine_tiles for t in tiles))} unique tiles")
    return aligned_geoboxes, mine_tiles


def compute_mine_adjacency(
    mining_gdf: gpd.GeoDataFrame,
    aligned_geoboxes: list
) -> np.ndarray:
    """Compute adjacency matrix for mine clustering.
    
    Mines are connected if any mine geometry intersects with another's buffered envelope.
    
    Args:
        mining_gdf: GeoDataFrame with mining polygons
        aligned_geoboxes: List of aligned GeoBox objects
    
    Returns:
        Adjacency matrix as 2D numpy array
    """
    logger.info("Computing mine adjacency matrix...")
    
    n_mines = len(mining_gdf)
    adjacency = np.zeros((n_mines, n_mines), dtype=int)
    
    for i in range(n_mines):
        mine_geom = mining_gdf.geometry.iloc[i]
        for j in range(n_mines):
            if i == j:
                adjacency[i, j] = 1  # Mine is always connected to itself
            else:
                # Check if mine i intersects with the buffered envelope of mine j
                buffered_envelope_j = aligned_geoboxes[j].extent.geom
                if mine_geom.intersects(buffered_envelope_j):
                    adjacency[i, j] = 1
    
    logger.debug(f"Computed {n_mines}x{n_mines} adjacency matrix")
    return adjacency


def cluster_mines(
    mining_gdf: gpd.GeoDataFrame,
    adjacency: np.ndarray
) -> np.ndarray:
    """Cluster mines using connected components.
    
    Args:
        mining_gdf: GeoDataFrame with mining polygons
        adjacency: Adjacency matrix
    
    Returns:
        Cluster labels array
    """
    n_components, labels = connected_components(adjacency, directed=False)
    logger.info(f"Created {n_components} clusters")
    return labels


def create_cluster_tasks(
    cluster_id: int,
    cluster_mask: np.ndarray,
    mining_gdf: gpd.GeoDataFrame,
    aligned_geoboxes: list,
    mine_tiles: list,
    cluster_indices: np.ndarray,
    world_geobox_tiles: GeoboxTiles,
    years: List[int],
    db: DownloadDatabase
) -> int:
    """Create download tasks for a single cluster.
    
    Args:
        cluster_id: Cluster identifier
        cluster_mask: Boolean mask for cluster mines
        mining_gdf: GeoDataFrame with mining polygons
        aligned_geoboxes: List of aligned GeoBox objects
        mine_tiles: List of tile indices for each mine
        cluster_indices: Indices of mines in this cluster
        world_geobox_tiles: GeoboxTiles instance
        years: List of years to download
        db: Database instance
    
    Returns:
        Number of tasks created for this cluster
    """
    cluster_mines = mining_gdf[cluster_mask]
    
    # Get primary country for this cluster (most common)
    country_code = cluster_mines['ISO3_CODE'].mode()[0]
    
    # Get combined mining footprint for this cluster
    from shapely.geometry import MultiPolygon
    mine_geoms = [geom for geom in cluster_mines.geometry]
    if len(mine_geoms) > 1:
        mining_footprint = MultiPolygon(mine_geoms)
    elif len(mine_geoms) == 1:
        mining_footprint = mine_geoms[0]
    else:
        mining_footprint = None
    
    mining_footprint_json = mining_footprint.__geo_interface__ if mining_footprint else None
    
    # Get union of all geoboxes in cluster
    cluster_geoboxes = [aligned_geoboxes[i] for i in cluster_indices]
    cluster_geobox = geobox_union_conservative(cluster_geoboxes)
    
    # Get all tiles for this cluster
    cluster_tiles = set()
    for i in cluster_indices:
        cluster_tiles.update(mine_tiles[i])
    cluster_tiles = sorted(cluster_tiles)
    
    # Calculate total pixels for this query
    height, width = cluster_geobox.shape
    total_pixels = height * width
    
    logger.debug(f"Cluster {cluster_id} ({country_code}): {len(cluster_mines)} mines, {total_pixels:.0f} pixels, {len(cluster_tiles)} tiles")
    
    tasks_created = 0
    
    # If too large, split into blocks of tiles
    max_pixels = 1e8
    if total_pixels > max_pixels:
        logger.info(f"Cluster {cluster_id} too large ({total_pixels:.0f} pixels), splitting into tile blocks")
        
        # Calculate how many tiles per block
        avg_pixels_per_tile = total_pixels / len(cluster_tiles) if cluster_tiles else total_pixels
        tiles_per_block = max(1, int(max_pixels / avg_pixels_per_tile))
        
        # Split tiles into blocks
        for block_idx in range(0, len(cluster_tiles), tiles_per_block):
            block_tiles = cluster_tiles[block_idx:block_idx + tiles_per_block]
            
            # Get union of geoboxes for this block of tiles
            block_geoboxes = [world_geobox_tiles[tile_ix, tile_iy] for tile_ix, tile_iy in block_tiles]
            block_geobox = geobox_union_conservative(block_geoboxes)
            
            # Store tiles in database for this task
            geojson = block_geobox.extent.__geo_interface__
            
            # Create task for each year
            for year in years:
                geom_hash = db.create_task(
                    geojson, country_code, year, cluster_id,
                    mining_footprint=mining_footprint_json
                )
                # Register tiles for this task
                for tile_ix, tile_iy in block_tiles:
                    db.create_tile(tile_ix, tile_iy, geom_hash, year, cluster_id)
                tasks_created += 1
            
            logger.debug(f"  Block {block_idx // tiles_per_block}: {len(block_tiles)} tiles")
    else:
        # Cluster is small enough, create single task
        geojson = cluster_geobox.extent.__geo_interface__
        
        # Create task for each year
        for year in years:
            geom_hash = db.create_task(
                geojson, country_code, year, cluster_id,
                mining_footprint=mining_footprint_json
            )
            # Register tiles for this task
            for tile_ix, tile_iy in cluster_tiles:
                db.create_tile(tile_ix, tile_iy, geom_hash, year, cluster_id)
            tasks_created += 1
    
    return tasks_created


def create_tasks(
    mining_file: str,
    countries: Optional[List[str]],
    years: List[int],
    buffer_size: float,
    db: DownloadDatabase
) -> int:
    """Create download tasks from mining polygons.
    
    Orchestrates the full task creation pipeline:
    1. Load and filter mining polygons
    2. Align to geobox tiles
    3. Compute mine clustering
    4. Create tasks for each cluster
    
    Args:
        mining_file: Path to mining polygons GeoPackage
        countries: List of ISO3 country codes (None for all)
        years: List of years to download
        buffer_size: Buffer size in degrees
        db: Database instance
    
    Returns:
        Number of tasks created
    
    Raises:
        FileNotFoundError: If mining file not found
        ValueError: If no valid polygons to process
    """
    # Validate inputs
    if buffer_size <= 0:
        raise ValueError(f"Buffer size must be positive, got {buffer_size}")
    if not years:
        raise ValueError("At least one year must be specified")
    
    # Load and filter mining data
    mining = load_and_filter_mining(mining_file, countries)
    
    # Setup world geobox for alignment
    config = Config()
    world_geobox = GeoBox.from_bbox(
        [-180, -90, 180, 90],
        resolution=config.WORLD_GEOBOX_RESOLUTION,
        crs=4326
    )
    world_geobox_tiles = GeoboxTiles(world_geobox, tile_shape=config.WORLD_GEOBOX_TILE_SIZE)
    
    # Align to geobox tiles
    aligned_geoboxes, mine_tiles = align_to_geobox_tiles(
        mining, world_geobox, world_geobox_tiles, buffer_size, config
    )
    
    # Compute adjacency matrix
    adjacency = compute_mine_adjacency(mining, aligned_geoboxes)
    
    # Cluster mines
    labels = cluster_mines(mining, adjacency)
    mining['cluster'] = labels
    
    # Create tasks for each cluster
    tasks_created = 0
    n_components = len(np.unique(labels))
    
    for cluster_id in range(n_components):
        cluster_mask = mining['cluster'] == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        tasks = create_cluster_tasks(
            cluster_id,
            cluster_mask,
            mining,
            aligned_geoboxes,
            mine_tiles,
            cluster_indices,
            world_geobox_tiles,
            years,
            db
        )
        tasks_created += tasks
    
    logger.info(f"\nTotal tasks created: {tasks_created}")
    return tasks_created
