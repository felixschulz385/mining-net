# Mining Segmentation Downloader

Multi-worker system for downloading Landsat annual composites for mining regions from Google Earth Engine with automatic clustering, geobox alignment, and zarr storage.

## Features

- **Multi-worker architecture**: Separate workers for export, status checking, downloading, and compression
- **Mine clustering**: Automatically clusters overlapping mining regions to reduce redundant downloads
- **Geobox alignment**: Aligns all queries to a world grid (64x64 tiles at ~30m resolution)
- **Zarr storage**: Repro jects downloaded data onto grid and stores as zarr by region
- **SQL database tracking**: Tracks all tasks and tiles with geometry hashing
- **CLI interface**: Easy command-line control for task creation and worker management
- **Batch processing**: Compresses files into 100-file zip archives
- **Automatic cleanup**: Deletes files from Google Drive after download

## Architecture

### Workers

1. **Export Worker**: Submits batch export tasks to Google Earth Engine with geobox-aligned queries
2. **Status Checker**: Monitors GEE task completion status
3. **Download Worker**: Downloads completed files, reprojects to grid, saves as zarr, and deletes from Drive
4. **Reprojection Worker**: Converts downloaded GeoTIFFs to memory-mapped PyTorch format
5. **Transfer Worker**: Uploads processed data to HPC cluster
6. **Janitor Worker**: Verifies filesystem integrity and fixes inconsistencies
   - Checks database tasks match actual files
   - Detects and removes orphaned files
   - Validates tile completeness
   - Can run in CHECK (report only) or CLEAN (fix) mode

### Database Schema

**tasks table**: Tracks each geometry-year combination
- Unique hash from geometry for deduplication
- Status tracking: pending → submitted → completed → downloaded → reprojected → uploaded
- Metadata: GEE task IDs, Drive file IDs, zarr paths, timestamps
- Cluster ID for grouped mines

**tiles table**: Tracks 64x64 geobox tiles
- Tile indices (tile_ix, tile_iy) in world grid
- Links to geometry_hash and year
- Write status tracking for zarr regions

**worker_status table**: Worker heartbeats and statistics

## Installation

```bash
# Install required packages
pip install geopandas earthengine-api google-auth-oauthlib google-api-python-client \
    rioxarray odc-geo scipy xarray zarr

# Authenticate with Google Earth Engine
earthengine authenticate

# Setup Google Drive API credentials
# 1. Go to https://console.developers.google.com/apis/credentials
# 2. Create OAuth 2.0 credentials for "Desktop application"
# 3. Download JSON and save as credentials.json in the project root
```

## Usage

### 1. Create Download Tasks

```bash
# Download specific years for specific countries
python -m gnt.data.download.mining_segmentation create \
  --mining-file /path/to/global_mining_polygons.gpkg \
  --countries ZAF USA CHL \
  --years 2020 2021 2022

# Download year range for all countries
python -m gnt.data.download.mining_segmentation create \
  --mining-file /path/to/global_mining_polygons.gpkg \
  --year-range 1992 2021 \
  --buffer 0.05
```

### 2. Run Workers

```bash
# Run all workers continuously
python -m gnt.data.download.mining_segmentation run

# Run specific workers
python -m gnt.data.download.mining_segmentation run --workers export status download

# Run janitor in check mode (reports issues only)
python -m gnt.data.download.mining_segmentation run --workers janitor --once

# Run janitor in clean mode (fixes issues)
python -m gnt.data.download.mining_segmentation run --workers janitor --clean --once

# Run once (useful for testing)
python -m gnt.data.download.mining_segmentation run --once

# Verbose logging
python -m gnt.data.download.mining_segmentation run -v
```

### 3. Check Status

```bash
python -m gnt.data.download.mining_segmentation status
```

### 4. Compress Remaining Files

```bash
# Compress any downloaded files that haven't reached batch size
python -m gnt.data.download.mining_segmentation compress-remaining
```

## Configuration

Edit `config.py` to customize:

- Google Earth Engine project ID
- Landsat collection and parameters
- Google Drive folder name
- Local paths for downloads and archives
- Worker settings (sleep intervals, batch sizes)
- Buffer size for geometries

## File Organization

```
data/mining_segmentation/
├── mining_segmentation.db        # SQLite database
├── downloads/                     # Downloaded GeoTIFF files (temporary)
└── archives/                      # Compressed zip archives
```

## Example Workflow

```bash
# 1. Create tasks for South Africa, 2000-2020
python -m gnt.data.download.mining_segmentation create \
  --mining-file mining_polygons.gpkg \
  --countries ZAF \
  --year-range 2000 2020

# 2. Run all workers in background
nohup python -m gnt.data.download.mining_segmentation run &

# 3. Monitor progress
watch -n 60 'python -m gnt.data.download.mining_segmentation status'

# 4. Run janitor to check integrity
python -m gnt.data.download.mining_segmentation run --workers janitor --once -v

# 5. Fix any issues found
python -m gnt.data.download.mining_segmentation run --workers janitor --clean --once
```

## Janitor Worker

The janitor maintains filesystem integrity and database consistency:

**Check Mode** (reports issues only):
```bash
python -m gnt.data.download.mining_segmentation run --workers janitor --once
```

**Clean Mode** (fixes issues automatically):
```bash
python -m gnt.data.download.mining_segmentation run --workers janitor --clean --once
```

The janitor performs these checks:
1. **Database Integrity**: Verifies task status matches filesystem
   - UPLOADED tasks exist on HPC cluster
   - REPROJECTED tasks have complete MMAP directories with exact tile matches
   - DOWNLOADED tasks have local files
   - COMPLETED tasks are on Google Drive
2. **Orphan Detection**: Finds files not in database
   - MMAP directories without database entries
   - Downloaded files without task references
3. **Tile Validation**: Ensures MMAP tiles match database exactly
   - Compares actual tile coordinates (tile_ix, tile_iy) with tiles table
   - Checks for features.pt, labels.pt, metadata.json in each tile
   - Detects missing or extra tiles
   - Removes incomplete tiles in clean mode

**Sequential Demotion**: When issues are found, tasks are demoted safely:
- UPLOADED → REPROJECTED (if in MMAP) → DOWNLOADED (if file exists) → PENDING
- REPROJECTED → DOWNLOADED (if file exists) → PENDING
- DOWNLOADED → COMPLETED (if on Drive) → PENDING
- COMPLETED → PENDING (if not on Drive)

## Task Status Flow

```completed → downloaded → reprojected → (uploaded)
         (GEE)        (GEE)        (Drive)      (Zarr)        (HPC)
                                      ↓
                                   failed
```

## Clustering Algorithm

Mines within each country are clustered based on overlap:
1. Buffer each mine polygon by specified amount (default 0.05°)
2. Compute intersection areas between all buffered envelopes
3. Create overlap matrix where mines overlap >75% (configurable)
4. Perform transitive closure to find connected components
5. Create one query per cluster (union of all envelopes)

This reduces redundant downloads when mines are close together.

## Geobox Tiling

All queries and outputs are aligned to a world grid:
- Resolution: 0.000269495° (~30m at equator, matching Landsat)
- Tile size: 64×64 pixels
- Each downloaded image is reprojected to tiles and stored in zarr

This ensures consistent spatial alignment across all data.

## File Organization

```
data/mining_segmentation/
├── mining_segmentation.db        # SQLite database
├── downloads/                     # Downloaded GeoTIFF files (temporary)
├── regions/                       # Zarr regions by country/cluster/year
│   ├── ZAF_cluster0_2020.zarr/
│   ├── ZAF_cluster1_2020.zarr/
│   └── ...
└── archives/                      # Compressed zip archives (deprecated)
```ogress
- Compression happens in batches of 100 files by default
