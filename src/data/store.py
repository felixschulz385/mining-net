"""Storage worker: converts downloaded GeoTIFFs into a Zarr dataset.

Architecture
------------
  ZarrStore       low-level Zarr I/O: create, resize, write, metadata bookkeeping
  TileExtractor   pure computation: reproject + extract per-tile numpy arrays
  StoreWorker     orchestration: validate  extract  write

Public API
----------
  store_task_to_zarr(task_data, config)        bool
  process_downloaded_tasks(config, ...)        int
"""

from __future__ import annotations

import functools
import gc
import json
import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Sequence

import numpy as np
import dask.array as da
import pandas as pd
import zarr
import xarray as xr
import rioxarray as rxr
from odc.geo.geobox import GeoBox, GeoboxTiles, geobox_union_conservative
from odc.geo.geom import Geometry
from odc.geo.xr import rasterize, ODCExtensionDa
from shapely.geometry import shape

from .config import Config

warnings.filterwarnings("ignore", message=".*CPLE_NotSupported.*warp options.*")
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANDS: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2", "thermal")
_BAND_INDEX: dict[int, str] = {i + 1: b for i, b in enumerate(BANDS)}
_DEFAULT_CHUNK_SIZE: int = 64
_FORMAT_VERSION: str = "2.0"


def _blosc_codec() -> zarr.codecs.BloscCodec:
    """Return a Blosc/zstd codec configured for fast I/O (clevel=0)."""
    return zarr.codecs.BloscCodec(
        cname="zstd", clevel=3, shuffle="bitshuffle", blocksize=0
    )


# ---------------------------------------------------------------------------
# Retry decorator for Windows file-lock errors
# ---------------------------------------------------------------------------

def _retry_on_permission_error(
    max_retries: int = 5,
    initial_delay: float = 0.5,
    backoff: float = 2.0,
):
    """Retry a function when a PermissionError is raised (Windows file-lock)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except PermissionError as exc:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(
                        "PermissionError (attempt %d/%d): %s  retrying in %.2fs",
                        attempt + 1, max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TileSpec:
    """Grid coordinates of a single world-geobox tile."""
    tile_ix: int
    tile_iy: int


@dataclass
class TileBatch:
    """In-memory tile data ready for a single Zarr slice-write."""
    features: list[np.ndarray]    # each (C, H, W) float32
    labels: list[np.ndarray]      # each (1, H, W) float32
    cluster_ids: list[int]
    tile_ixs: list[int]
    tile_iys: list[int]
    years: list[int]

    def __len__(self) -> int:
        return len(self.features)

    @classmethod
    def concat(cls, a: "TileBatch", b: "TileBatch") -> "TileBatch":
        """Return a new TileBatch that is `a` followed by `b`. Does not
        mutate the inputs.
        """
        return TileBatch(
            features=list(a.features) + list(b.features),
            labels=list(a.labels) + list(b.labels),
            cluster_ids=list(a.cluster_ids) + list(b.cluster_ids),
            tile_ixs=list(a.tile_ixs) + list(b.tile_ixs),
            tile_iys=list(a.tile_iys) + list(b.tile_iys),
            years=list(a.years) + list(b.years),
        )

    @classmethod
    def split(cls, batch: "TileBatch", n: int) -> tuple["TileBatch", "TileBatch"]:
        """Return (batch[:n], batch[n:]).  Either half may be empty.
        Does not mutate the input.
        """
        left = TileBatch(
            features=list(batch.features[:n]),
            labels=list(batch.labels[:n]),
            cluster_ids=list(batch.cluster_ids[:n]),
            tile_ixs=list(batch.tile_ixs[:n]),
            tile_iys=list(batch.tile_iys[:n]),
            years=list(batch.years[:n]),
        )
        right = TileBatch(
            features=list(batch.features[n:]),
            labels=list(batch.labels[n:]),
            cluster_ids=list(batch.cluster_ids[n:]),
            tile_ixs=list(batch.tile_ixs[n:]),
            tile_iys=list(batch.tile_iys[n:]),
            years=list(batch.years[n:]),
        )
        return left, right


@dataclass
class TaskData:
    """Validated, typed representation of a single store task."""
    local_filepath: Path
    cluster_id: int
    country_code: str
    year: int
    tiles: list[dict]
    geometry_hash: str = ""
    footprint: Optional[dict] = None

    @classmethod
    def from_dict(cls, d: dict) -> TaskData:
        """Parse and validate a raw task dict.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If year or cluster_id cannot be coerced.
        """
        return cls(
            local_filepath=Path(d["local_filepath"]),
            cluster_id=int(d.get("cluster_id") or 0),
            country_code=d["country_code"],
            year=int(d["year"]),
            tiles=list(d["tiles"]),
            geometry_hash=str(d.get("geometry_hash") or ""),
            footprint=d.get("footprint"),
        )


# ---------------------------------------------------------------------------
# ZarrStore: pure I/O layer
# ---------------------------------------------------------------------------

class ZarrStore:
    """Zarr array I/O: create, resize, write, and metadata bookkeeping.

    Separates all disk interactions from processing logic.  The underlying
    group is opened fresh per operation to avoid Windows file-lock buildup.

    Lifecycle::

        store = ZarrStore(zarr_dir, tile_shape)
        store.create(capacity)          # once: writes empty Zarr structure and allocates capacity
        store.write_batch(0, batch)
        store.advance_index(len(batch))

    To resume writing to an existing store, simply omit ``create``; the
    metadata JSON restores the write position.
    """

    _METADATA_FILENAME = "store_metadata.json"
    _STORE_DIRNAME = "data.zarr"

    def __init__(
        self,
        zarr_dir: Path,
        tile_shape: tuple[int, int],
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> None:
        self._zarr_dir = zarr_dir
        self._tile_shape = tile_shape
        self._chunk_size = chunk_size
        self._store_path = zarr_dir / self._STORE_DIRNAME
        self._metadata_path = zarr_dir / self._METADATA_FILENAME

        self._zarr_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = self._load_or_create_metadata()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def store_path(self) -> Path:
        return self._store_path

    @property
    def next_index(self) -> int:
        return self._metadata["next_index"]

    @property
    def capacity(self) -> int:
        return self._metadata.get("capacity", 0)

    @property
    def exists(self) -> bool:
        return self._store_path.exists()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create(self, capacity: int) -> None:
        """Create a new Zarr store and allocate `capacity` tiles.

        Writes dask-backed empty arrays sized to `capacity` so the store is
        immediately ready for writes.

        Args:
            capacity: number of tiles to allocate (must be > 0)

        Raises:
            FileExistsError: If a store already exists at this path.
            ValueError: If capacity <= 0.
        """
        if capacity <= 0:
            raise ValueError(f"Capacity must be > 0, got {capacity}.")
        if self._store_path.exists():
            raise FileExistsError(
                f"Zarr store already exists at {self._store_path}. "
                "Delete the directory to start fresh."
            )
        # create arrays sized to `capacity` and write skeleton metadata to disk
        self._write_empty_store(capacity)
        self._metadata["capacity"] = capacity
        self._metadata["next_index"] = 0
        self._save_metadata()
        logger.info("Created Zarr store at %s (capacity=%d)", self._store_path, capacity)

    # initialize_capacity removed — capacity is now allocated at create(capacity).

    def check_capacity(self, n_new_tiles: int) -> bool:
        """Return True when the store can accept n_new_tiles additional tiles."""
        required = self.next_index + n_new_tiles
        if required > self.capacity:
            logger.error(
                "Zarr capacity exceeded: capacity=%d, required=%d. "
                "Recreate the store with larger capacity.",
                self.capacity, required,
            )
            return False
        return True

    @_retry_on_permission_error(max_retries=5, initial_delay=1.0)
    def write_batch(self, start_idx: int, batch: TileBatch) -> None:
        """Write a TileBatch to Zarr starting at start_idx.

        Opens, writes, and closes the Zarr group in a single cycle to avoid
        file-lock buildup on Windows.
        """
        end_idx = start_idx + len(batch)
        features_arr = np.stack(batch.features, axis=0)          # (N, C, H, W)
        labels_arr = np.stack(batch.labels, axis=0)              # (N, 1, H, W)
        cluster_ids_arr = np.array(batch.cluster_ids, dtype=np.int64)
        tile_ixs_arr = np.array(batch.tile_ixs, dtype=np.int32)
        tile_iys_arr = np.array(batch.tile_iys, dtype=np.int32)
        years_arr = np.array(batch.years, dtype=np.int32)

        # Build the batch Dataset with the same dims/coords as the empty
        # store so `to_zarr(region='auto')` can align variables correctly.
        tile_h, tile_w = self._tile_shape
        n_bands = len(BANDS)

        ds_batch = (
            xr.Dataset(
                data_vars={
                    "features": (["tile", "channel", "y", "x"], features_arr),
                    "labels": (["tile", "label_channel", "y", "x"], labels_arr),
                },
                coords={
                    "tile": pd.RangeIndex(start_idx, end_idx),
                    "channel": np.arange(n_bands),
                    "y": np.arange(tile_h),
                    "x": np.arange(tile_w),
                    "label_channel": [0],
                },
            )
            .assign_coords(
                cluster_id=("tile", cluster_ids_arr),
                tile_ix=("tile", tile_ixs_arr),
                tile_iy=("tile", tile_iys_arr),
                year=("tile", years_arr),
            )
        )

        # write into existing store (mode='r+'), let xarray handle the region write
        ds_batch.to_zarr(str(self._store_path), mode="r+", consolidated=False, region="auto", compute=True)

        logger.debug("Wrote %d tiles to Zarr [%d:%d] via xarray.to_zarr", len(batch), start_idx, end_idx)

    def advance_index(self, n: int) -> int:
        """Advance the write pointer by n tiles, persist, and return the new index."""
        self._metadata["next_index"] += n
        self._save_metadata()
        return self._metadata["next_index"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _open(self) -> Generator[zarr.Group, None, None]:
        """Open and cleanly close the Zarr group as a context manager."""
        group = zarr.open_group(
            store=str(self._store_path), mode="r+", zarr_format=3
        )
        try:
            yield group
        finally:
            _close_zarr_group(group)

    def _write_empty_store(self, capacity: int) -> None:
        tile_h, tile_w = self._tile_shape
        n_bands = len(BANDS)
        chunk = self._chunk_size
        codec = _blosc_codec()

        # create dask-backed empty arrays (not computed) sized to `capacity`
        features = da.zeros(
            (capacity, n_bands, tile_h, tile_w), dtype=np.float32,
            chunks=(chunk, n_bands, tile_h, tile_w),
        )
        labels = da.zeros(
            (capacity, 1, tile_h, tile_w), dtype=np.float32,
            chunks=(chunk, 1, tile_h, tile_w),
        )

        ds = (
            xr.Dataset(
                data_vars={
                    "features": (
                        ["tile", "channel", "y", "x"],
                        features,
                        {"long_name": "Landsat features", "units": "reflectance"},
                    ),
                    "labels": (
                        ["tile", "label_channel", "y", "x"],
                        labels,
                        {"long_name": "Mining labels", "units": "binary"},
                    ),
                },
                coords={
                    # use a pandas RangeIndex for the tile coordinate (fast indexing/serialization)
                    "tile": pd.RangeIndex(0, capacity),
                    "channel": np.arange(n_bands),
                    "y": np.arange(tile_h),
                    "x": np.arange(tile_w),
                    "label_channel": [0],
                },
            )
            .assign_coords(
                cluster_id=("tile", da.full((capacity,), -1, dtype=np.int64, chunks=(chunk,))),
                tile_ix=("tile", da.full((capacity,), -1, dtype=np.int32, chunks=(chunk,))),
                tile_iy=("tile", da.full((capacity,), -1, dtype=np.int32, chunks=(chunk,))),
                year=("tile", da.full((capacity,), -1, dtype=np.int32, chunks=(chunk,))),
            )
        )
        ds.attrs.update(
            description="Mining segmentation dataset from Landsat imagery",
            tile_shape=f"{tile_h}x{tile_w}",
            n_channels=n_bands,
        )
        ds.to_zarr(
            str(self._store_path),
            mode="w",
            encoding={
                "features":   {"chunks": (chunk, n_bands, tile_h, tile_w), "compressors": codec, "fill_value": 0.0},
                "labels":     {"chunks": (chunk, 1, tile_h, tile_w),       "compressors": codec, "fill_value": 0.0},
                "cluster_id": {"chunks": (chunk,),                         "compressors": codec, "fill_value": -1},
                "tile_ix":    {"chunks": (chunk,),                         "compressors": codec, "fill_value": -1},
                "tile_iy":    {"chunks": (chunk,),                         "compressors": codec, "fill_value": -1},
                "year":       {"chunks": (chunk,),                         "compressors": codec, "fill_value": -1},
            },
            consolidated=False,
            zarr_format=3,
            compute=False,
            write_empty_chunks=False,
        )

    def _load_or_create_metadata(self) -> dict:
        if self._metadata_path.exists():
            with open(self._metadata_path, encoding="utf-8") as f:
                return json.load(f)
        metadata: dict = {
            "next_index": 0,
            "capacity": 0,
            "tile_size": list(self._tile_shape),
            "n_bands": len(BANDS),
            "chunk_size": self._chunk_size,
            "format_version": _FORMAT_VERSION,
        }
        self._save_metadata(metadata)
        return metadata

    def _save_metadata(self, data: Optional[dict] = None) -> None:
        payload = data if data is not None else self._metadata
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def _close_zarr_group(group: zarr.Group) -> None:
    """Sync and close a Zarr group, suppressing cleanup errors."""
    try:
        if hasattr(group.store, "sync"):
            group.store.sync()
        if hasattr(group.store, "close"):
            group.store.close()
    except Exception as exc:
        logger.warning("Error closing Zarr group: %s", exc)
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# TileExtractor: pure computation layer
# ---------------------------------------------------------------------------

class TileExtractor:
    """Reprojects a GeoTIFF onto the world grid and extracts per-tile arrays.

    Entirely stateless with respect to disk; operates only on in-memory
    xarray/numpy objects.
    """

    def __init__(self, config: Config) -> None:
        self._config = config
        self._world_geobox = GeoBox.from_bbox(
            [-180, -90, 180, 90],
            resolution=config.WORLD_GEOBOX_RESOLUTION,
            crs=4326,
        )
        self._world_tiles = GeoboxTiles(
            self._world_geobox, config.WORLD_GEOBOX_TILE_SIZE
        )

    def extract(self, task: TaskData) -> TileBatch:
        """Reproject task.local_filepath and extract all tiles.

        Returns:
            TileBatch ready for writing to the Zarr store.

        Raises:
            FileNotFoundError: If the TIFF does not exist on disk.
            ValueError: If the tile list is empty.
        """
        if not task.local_filepath.exists():
            raise FileNotFoundError(f"TIFF not found: {task.local_filepath}")
        if not task.tiles:
            raise ValueError(
                f"No tiles specified for task {task.geometry_hash!r} year={task.year}"
            )

        tile_specs = [TileSpec(t["tile_ix"], t["tile_iy"]) for t in task.tiles]
        union_geobox = geobox_union_conservative(
            [self._world_tiles[s.tile_ix, s.tile_iy] for s in tile_specs]
        )
        logger.info(
            "Reprojecting %s -> %s (%d tiles)",
            task.local_filepath.name, union_geobox.shape, len(tile_specs),
        )

        image = rxr.open_rasterio(task.local_filepath)
        try:
            reprojected = self._reproject(image, union_geobox)
        finally:
            image.close()

        mining_mask = self._rasterize_footprint(task, union_geobox)
        return self._build_batch(task, tile_specs, reprojected, mining_mask)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reproject(image: xr.DataArray, geobox: GeoBox) -> xr.Dataset:
        reprojected = image.odc.reproject(geobox)
        reprojected.coords["latitude"] = (
            reprojected.coords["latitude"].values.round(5)
        )
        reprojected.coords["longitude"] = (
            reprojected.coords["longitude"].values.round(5)
        )
        return reprojected.to_dataset(dim="band").rename(_BAND_INDEX)

    def _rasterize_footprint(
        self, task: TaskData, geobox: GeoBox
    ) -> Optional[xr.DataArray]:
        if task.year != self._config.GROUND_TRUTH_YEAR or task.footprint is None:
            return None
        geom = Geometry(shape(task.footprint), crs=4326)
        mask = rasterize(geom, geobox)
        logger.info("Rasterized footprint for ground-truth year %d", task.year)
        return mask

    def _build_batch(
        self,
        task: TaskData,
        tile_specs: list[TileSpec],
        reprojected: xr.Dataset,
        mining_mask: Optional[xr.DataArray],
    ) -> TileBatch:
        batch = TileBatch(
            features=[], labels=[],
            cluster_ids=[], tile_ixs=[], tile_iys=[], years=[],
        )
        for spec in tile_specs:
            tile_geobox = self._world_tiles[spec.tile_ix, spec.tile_iy]
            bounds = tile_geobox.boundingbox
            batch.features.append(_extract_band_data(reprojected, bounds))
            batch.labels.append(_extract_label_data(mining_mask, bounds, tile_geobox.shape))
            batch.cluster_ids.append(task.cluster_id)
            batch.tile_ixs.append(spec.tile_ix)
            batch.tile_iys.append(spec.tile_iy)
            batch.years.append(task.year)
        return batch


def _extract_band_data(dataset: xr.Dataset, bounds) -> np.ndarray:
    """Slice each band from dataset at bounds and stack into (C, H, W)."""
    arrays = [
        dataset[band].sel(
            latitude=slice(bounds.top, bounds.bottom),
            longitude=slice(bounds.left, bounds.right),
        ).values
        for band in BANDS
    ]
    result = np.stack(arrays, axis=0).astype(np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_label_data(
    mask: Optional[xr.DataArray],
    bounds,
    tile_shape: tuple[int, int],
) -> np.ndarray:
    """Slice the mining mask at bounds, returning a (1, H, W) float32 array."""
    if mask is not None:
        labels = mask.sel(
            latitude=slice(bounds.top, bounds.bottom),
            longitude=slice(bounds.left, bounds.right),
        ).values.astype(np.float32)
    else:
        labels = np.zeros(tile_shape, dtype=np.float32)

    if labels.ndim == 2:
        labels = labels[np.newaxis, ...]  # (H, W) -> (1, H, W)
    return np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)


def _empty_tile_batch() -> TileBatch:
    """Return an empty TileBatch (module-level helper)."""
    return TileBatch(features=[], labels=[], cluster_ids=[], tile_ixs=[], tile_iys=[], years=[])


# ---------------------------------------------------------------------------
# StoreWorker: orchestration layer
# ---------------------------------------------------------------------------

class StoreWorker:
    """Orchestrates the full pipeline: validate -> extract -> write.

    Owns a ZarrStore and a TileExtractor.  Does not interact with databases.

    Usage for a new store::

        worker = StoreWorker(config)
        worker.initialize_store(capacity=10_000)   # create + resize
        worker.process_task(task_dict)

    Usage to resume writing to an existing store::

        worker = StoreWorker(config)               # loads existing metadata
        worker.process_task(task_dict)
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()
        self._store = ZarrStore(
            zarr_dir=self._config.DATA_DIR / "landsat_zarr",
            tile_shape=self._config.WORLD_GEOBOX_TILE_SIZE,
        )
        self._extractor = TileExtractor(self._config)
        # in-memory buffer for partial (cross-task) chunks
        self._residual: TileBatch = _empty_tile_batch()

    @property
    def store(self) -> ZarrStore:
        return self._store

    def initialize_store(self, capacity: int) -> None:
        """Create a Zarr store and allocate `capacity` tiles.

        Raises:
            FileExistsError, ValueError, RuntimeError -- propagated from ZarrStore.
        """
        self._store.create(capacity)

    def process_task(self, task_data: dict) -> bool:
        """Extract and store a single task.

        Args:
            task_data: Raw task dictionary (see TaskData.from_dict).

        Returns:
            True on success, False on any handled failure.
        """
        try:
            task = TaskData.from_dict(task_data)
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Invalid task data: %s", exc)
            return False

        try:
            batch = self._extractor.extract(task)
        except FileNotFoundError as exc:
            logger.warning("Skipping missing file: %s", exc)
            return False
        except ValueError as exc:
            logger.warning("Skipping invalid task: %s", exc)
            return False
        except Exception as exc:
            logger.error(
                "Unexpected extraction error for %s year=%d: %s",
                task.geometry_hash, task.year, exc, exc_info=True,
            )
            return False

        # Combine any previously buffered residual with the newly-extracted
        # tiles from this task so we can write only full chunk multiples.
        combined = TileBatch.concat(self._residual, batch)

        # Ensure store has capacity for the combined set of tiles (residual + new).
        if not self._store.check_capacity(len(combined)):
            return False

        chunk = self._store._chunk_size
        n_complete = (len(combined) // chunk) * chunk

        if n_complete > 0:
            to_write, remainder = TileBatch.split(combined, n_complete)

            start_idx = self._store.next_index
            try:
                self._store.write_batch(start_idx, to_write)
            except Exception as exc:
                logger.error("Failed to write batch to Zarr: %s", exc, exc_info=True)
                return False

            # Advance metadata only for fully-written tiles
            self._store.advance_index(n_complete)
            self._residual = remainder

            logger.info(
                "Stored %d tile(s) [%d:%d] -- cluster=%d year=%d (residual=%d)",
                n_complete, start_idx, start_idx + n_complete,
                task.cluster_id, task.year, len(self._residual),
            )
        else:
            # Nothing to flush to disk yet — keep everything in the residual buffer.
            self._residual = combined
            logger.debug("Buffered %d residual tile(s) — deferring write", len(self._residual))

        return True

    def flush_residual(self) -> int:
        """Write any buffered residual tiles to Zarr. Returns number of tiles written."""
        n = len(self._residual)
        if n == 0:
            logger.debug("No residual tiles to flush")
            return 0

        # capacity check (preserves existing semantics)
        if not self._store.check_capacity(n):
            logger.error("Could not flush residual: insufficient capacity for %d tiles", n)
            return 0

        start_idx = self._store.next_index
        try:
            self._store.write_batch(start_idx, self._residual)
        except Exception as exc:
            logger.error("Failed to write residual tiles to Zarr: %s", exc, exc_info=True)
            return 0

        self._store.advance_index(n)
        self._residual = _empty_tile_batch()
        logger.info("Flushed %d residual tile(s) [%d:%d]", n, start_idx, start_idx + n)
        return n


# ---------------------------------------------------------------------------
# Module-level public API
# ---------------------------------------------------------------------------

def store_task_to_zarr(
    task_data: dict,
    config: Optional[Config] = None,
) -> bool:
    """Store a single downloaded GeoTIFF into an existing Zarr store.

    The Zarr store must already exist and have capacity initialized.  Use
    process_downloaded_tasks for bulk processing with automatic store creation.

    Args:
        task_data: Must contain ``local_filepath``, ``country_code``, ``year``,
            and ``tiles``.  Optional: ``cluster_id``, ``geometry_hash``,
            ``footprint``.
        config: Optional Config instance.

    Returns:
        True on success, False on failure.
    """
    return StoreWorker(config=config).process_task(task_data)


def process_downloaded_tasks(
    config: Optional[Config] = None,
    countries: Optional[List[str]] = None,
    input_dir: Optional[str] = None,
    limit: Optional[int] = None,
) -> int:
    """Bulk-process downloaded TIFFs into a new Zarr dataset.

    Scans input_dir for *.tif files with adjacent metadata JSONs produced by
    the download worker, pre-allocates the Zarr store to the exact required
    capacity, then processes each file sequentially.

    Args:
        config: Optional Config instance.
        countries: Optional ISO-3 country-code whitelist.
        input_dir: Source directory.  Defaults to Config.DOWNLOAD_DIR.
        limit: Optional cap on the number of TIFFs to process.

    Returns:
        Number of successfully stored tasks.
    """
    cfg = config or Config()
    input_path = Path(input_dir) if input_dir else Path(cfg.DOWNLOAD_DIR)
    if not input_path.exists():
        logger.error("Input directory not found: %s", input_path)
        return 0

    tiff_files = sorted(input_path.rglob("*.tif"))
    if not tiff_files:
        logger.info("No TIFF files found in %s", input_path)
        return 0

    if limit:
        tiff_files = tiff_files[:limit]

    total_tiles = _count_tiles_in_manifests(tiff_files)
    if total_tiles == 0:
        logger.warning("No tile metadata found in manifests -- aborting")
        return 0

    worker = StoreWorker(config=cfg)
    try:
        worker.initialize_store(capacity=total_tiles)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        logger.error("Could not initialize Zarr store: %s", exc)
        return 0

    logger.info(
        "Processing %d TIFF(s) from %s (capacity=%d)",
        len(tiff_files), input_path, total_tiles,
    )

    processed = 0
    for tiff in tiff_files:
        task_data = _load_task_from_tiff(tiff, countries)
        if task_data is None:
            continue
        if worker.process_task(task_data):
            processed += 1

    # write any buffered residual tiles (final partial chunk)
    flushed = worker.flush_residual()
    if flushed:
        logger.info("Flushed %d residual tile(s) to Zarr", flushed)

    logger.info("Finished: %d/%d files stored", processed, len(tiff_files))
    return processed


# ---------------------------------------------------------------------------
# Private module helpers
# ---------------------------------------------------------------------------

def _count_tiles_in_manifests(tiff_files: Sequence[Path]) -> int:
    """Sum tile counts from the metadata JSONs adjacent to tiff_files."""
    total = 0
    for tiff in tiff_files:
        meta_path = tiff.with_suffix(".json")
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                total += len(json.load(f).get("tiles") or [])
        except Exception:
            pass
    return total


def _load_task_from_tiff(
    tiff: Path,
    countries: Optional[List[str]],
) -> Optional[dict]:
    """Read and validate the metadata JSON for tiff.

    Returns a task dict if the file passes filtering, otherwise None.
    """
    meta_path = tiff.with_suffix(".json")
    if not meta_path.exists():
        logger.warning("Skipping %s: metadata JSON not found", tiff.name)
        return None

    try:
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        logger.warning("Skipping %s: could not read metadata: %s", tiff.name, exc)
        return None

    country = metadata.get("country_code")
    if countries and country not in countries:
        logger.debug("Skipping %s: country %r not in filter", tiff.name, country)
        return None

    return {
        "local_filepath": str(tiff),
        "cluster_id":     metadata.get("cluster_id"),
        "country_code":   country,
        "year":           metadata.get("year"),
        "geometry_hash":  metadata.get("geometry_hash"),
        "tiles":          metadata.get("tiles"),
        "footprint":      metadata.get("footprint"),
    }
