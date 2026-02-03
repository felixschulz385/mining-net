# Mining Segmentation Network (mining-net)

Deep learning framework for detecting and segmenting mining areas from satellite imagery using Landsat data and the Google Earth Engine API.

## Overview

This project provides an end-to-end pipeline for:
1. **Data Collection**: Download and process Landsat satellite imagery via Google Earth Engine
2. **Data Storage**: Organize multi-spectral data in efficient Zarr format
3. **Model Development**: Train UNet models for mining footprint segmentation
4. **Evaluation**: Assess model performance with comprehensive metrics and visualizations

The system enables researchers and organizations to automatically detect mining activities globally at 30-meter resolution from freely available Landsat imagery.

## Project Structure

```
mining-net/
├── README.md                          # This file
├── LICENSE                            # Project license
├── data/
│   ├── global_mining_polygons_v2.gpkg # Mining polygon reference data (GeoPackage)
│   ├── global_landsat.zarr/           # Multi-spectral Landsat data (Zarr format)
│   ├── downloads/                     # Temporary storage for GEE downloads
│   └── mining_segmentation.db         # SQLite database (generated)
│
├── secrets/
│   └── client_secret_*.json           # Google Earth Engine credentials (gitignored)
│
├── src/
│   ├── data/                          # Data processing and loading
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── config.py                  # Data configuration
│   │   ├── cli.py                     # Command-line interface
│   │   ├── data_loader.py             # TensorFlow data provider
│   │   ├── database.py                # SQLite database manager
│   │   ├── download.py                # Download task management
│   │   ├── gee_export.py              # Google Earth Engine integration
│   │   ├── reproject.py               # Data reprojection utilities
│   │   ├── status_checker.py          # Task status monitoring
│   │   ├── transfer.py                # File transfer utilities
│   │   ├── tasks.py                   # Task definitions
│   │   ├── README.md                  # Data module documentation
│   │   └── tests/
│   │
│   ├── network/                       # Deep learning models
│   │   ├── __init__.py
│   │   ├── unet.py                    # UNet architecture
│   │   ├── train.py                   # Training pipeline
│   │   ├── evaluate.py                # Evaluation and inference
│   │   ├── config.py                  # Model configuration
│   │   ├── utils.py                   # Loss functions, metrics, utilities
│   │   ├── README.md                  # Network module documentation
│   │   └── requirements.txt           # Network dependencies
│   │
│   └── notebooks/
│       ├── mining_segmentation.ipynb  # Main segmentation notebook
│       └── unet_training_demo.ipynb   # UNet training tutorial
│
└── models/                            # Model checkpoints (generated)
    └── checkpoints/
```

## Key Components

### 1. Data Module (`src/data/`)

**Purpose**: Download, process, and manage Landsat satellite imagery

**Key Features**:
- Google Earth Engine integration for large-scale data download
- Multi-spectral band extraction (blue, green, red, NIR, SWIR1, SWIR2, thermal)
- Zarr-based efficient storage for fast access
- SQLite database for tracking download status
- Support for mining polygon reference data (GeoPackage format)

**Usage**:
```bash
# Download Landsat data
python -m src.data download --country ZAF --year 2020

# Check status
python -m src.data status

# Reproject and process
python -m src.data reproject
```

**See**: [Data Module README](src/data/README.md)

### 2. Network Module (`src/network/`)

**Purpose**: Train and evaluate deep learning models for mining segmentation

**Architecture**: UNet with encoder-decoder structure
- Input: 64×64 pixel tiles with 7 Landsat bands (~30m resolution)
- Output: Binary segmentation mask (mining vs. non-mining)
- Encoder: 4 levels with skip connections
- Bottleneck: 1024 filters
- Decoder: 4 levels with upsampling

**Key Features**:
- Multiple loss functions (BCE, Dice, Focal, Combined)
- Data augmentation (flips, rotations, brightness/contrast)
- Comprehensive metrics (Dice, IoU, precision, recall, F1, AUC)
- Mixed precision training support
- Early stopping and learning rate scheduling
- TensorBoard integration

**Usage**:
```bash
# Train model
python src/network/train.py \
    --countries ZAF USA \
    --years 2019 2020 \
    --epochs 50 \
    --batch-size 32

# Evaluate
python src/network/evaluate.py models/checkpoints/unet_best.h5 \
    --countries ZAF \
    --visualize

# Or use the Jupyter notebook
jupyter notebook src/notebooks/unet_training_demo.ipynb
```

**See**: [Network Module README](src/network/README.md)

### 3. Notebooks

- **mining_segmentation.ipynb**: Main workflow and exploration
- **unet_training_demo.ipynb**: Interactive training tutorial with examples

## Quick Start

### 1. Setup

#### Prerequisites
- Python 3.8+
- CUDA 11.x (for GPU training, optional)
- Google Earth Engine account

#### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mining-net.git
cd mining-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Google Earth Engine Setup

```bash
# Authenticate with Earth Engine
earthengine authenticate

# This creates credentials at ~/.config/earthengine/
```

### 2. Download Data

```bash
# Download Landsat data for a country and year
python -m src.data download \
    --country ZAF \
    --year 2020 \
    --bands blue green red nir swir1 swir2 thermal

# Monitor download progress
python -m src.data status

# Once downloads are complete, process and store as Zarr
python -m src.data process
```

### 3. Train Model

```bash
# Quick test training
python src/network/train.py \
    --epochs 10 \
    --batch-size 16 \
    --run-name test_run

# Full training
python src/network/train.py \
    --countries ZAF USA \
    --years 2019 2020 \
    --epochs 100 \
    --batch-size 32 \
    --run-name production_model
```

### 4. Evaluate & Predict

```bash
# Evaluate on test set
python src/network/evaluate.py models/checkpoints/unet_best.h5 \
    --countries ZAF \
    --countries USA

# Visualize predictions
python src/network/evaluate.py models/checkpoints/unet_best.h5 \
    --visualize \
    --num-samples 8 \
    --output-dir results/

# Export predictions
python src/network/evaluate.py models/checkpoints/unet_best.h5 \
    --export \
    --output-dir predictions/
```

## Data Format

### Input: Landsat Imagery

- **Source**: Google Earth Engine LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL
- **Resolution**: 30 meters
- **Bands**:
  - Blue (B2)
  - Green (B3)
  - Red (B4)
  - Near Infrared (B5)
  - Short-wave Infrared 1 (B6)
  - Short-wave Infrared 2 (B7)
  - Thermal Infrared (B10)
- **Tile Size**: 64×64 pixels (~1.9 km × 1.9 km)
- **Storage**: Zarr format for efficient multi-dimensional access

### Ground Truth: Mining Polygons

- **Source**: global_mining_polygons_v2.gpkg
- **Format**: GeoPackage (standardized geospatial vector format)
- **Geometry**: Polygon boundaries of known mining areas
- **Processing**: Rasterized to 30m binary masks

### Output: Predictions

- **Format**: Binary segmentation masks (64×64 float32)
- **Values**: Probability [0, 1] of mining activity
- **Threshold**: 0.5 for binary classification

## Model Configuration

Key hyperparameters can be adjusted in [src/network/config.py](src/network/config.py):

```python
# Architecture
INPUT_SHAPE = (64, 64, 7)
FILTERS_BASE = 64
DEPTH = 4
DROPOUT_RATE = 0.1

# Training
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4

# Loss function
LOSS_TYPE = 'combined'  # 'bce', 'dice', 'focal', 'combined'
POSITIVE_CLASS_WEIGHT = 10.0  # For imbalanced data

# Augmentation
USE_AUGMENTATION = True
FLIP_HORIZONTAL = True
FLIP_VERTICAL = True
ROTATE_90 = True
```

## Performance Metrics

The model is evaluated using:
- **Dice Coefficient**: Overlap between prediction and ground truth (0-1)
- **IoU (Jaccard Index)**: Intersection over union (0-1)
- **Precision**: Proportion of predicted mining pixels that are correct
- **Recall**: Proportion of actual mining pixels that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

## Advanced Features

### Class Weighting
For imbalanced datasets with few mining pixels:
```python
config.USE_CLASS_WEIGHTS = True
config.POSITIVE_CLASS_WEIGHT = 10.0
```

### Mixed Precision Training
For faster training on GPUs with Tensor Cores (V100, A100, etc.):
```python
config.USE_MIXED_PRECISION = True
```

### Multi-GPU Training
```python
config.USE_MULTI_GPU = True
```

### Custom Loss Functions
Select or implement custom loss functions in [src/network/utils.py](src/network/utils.py):
- Binary Cross-Entropy
- Dice Loss (good for imbalanced segmentation)
- Focal Loss (emphasizes hard examples)
- Combined Loss (weighted BCE + Dice)

## Requirements

### Core Dependencies
- Python 3.8+
- TensorFlow 2.10+
- NumPy
- Pandas
- Xarray
- Zarr
- Rasterio
- Geopandas

### Geospatial Tools
- GDAL
- Google Earth Engine Python API
- Fiona

### Visualization
- Matplotlib
- Cartopy

### Development
- Jupyter
- Pytest

See [requirements.txt](requirements.txt) for complete list with versions.

## Troubleshooting

### Google Earth Engine Issues

**Problem**: Authentication fails
```bash
# Re-authenticate
earthengine authenticate
```

**Problem**: Tasks stuck in pending state
```bash
# Check and clean up failed tasks
python -m src.data status --verbose
```

### Data Issues

**Problem**: Out of memory during training
- Reduce `BATCH_SIZE` in config
- Enable `USE_MIXED_PRECISION`
- Reduce `WORLD_GEOBOX_TILE_SIZE`

**Problem**: No tiles found after download
- Verify `MINING_FILE` path in config
- Check that tiles intersect with mining polygons
- Ensure download completed successfully

### Model Training Issues

**Problem**: Loss not decreasing
- Increase `LEARNING_RATE`
- Check data augmentation settings
- Verify loss function selection
- Inspect data normalization

**Problem**: Out of GPU memory
- Reduce `BATCH_SIZE`
- Enable `USE_MIXED_PRECISION`
- Reduce `WORLD_GEOBOX_TILE_SIZE` or `FILTERS_BASE`

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use this project in your research, please cite:

```bibtex
@software{mining_segmentation_2024,
  author = {Your Name},
  title = {Mining Segmentation Network: Deep Learning for Mining Detection},
  year = {2024},
  url = {https://github.com/yourusername/mining-net}
}
```

## License

This project is licensed under the [LICENSE](LICENSE) file included in the repository.

## References

### Satellite Imagery
- [Google Earth Engine Documentation](https://developers.google.com/earth-engine)
- [Landsat 8/9 Surface Reflectance Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2)

### Deep Learning
- [UNet: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide)

### Segmentation Metrics
- [Dice Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
- [Intersection over Union](https://en.wikipedia.org/wiki/Jaccard_index)

## Acknowledgments

- Google Earth Engine for satellite imagery
- USGS for Landsat data
- Global Mining Polygons dataset contributors

## Contact & Support

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: See individual module READMEs

## Status

- ✅ Data collection and processing
- ✅ UNet model architecture
- ✅ Training pipeline
- ✅ Evaluation framework
- 🔄 Multi-task learning (planned)
- 🔄 Temporal analysis (planned)
- 🔄 Real-time inference API (planned)

---

**Last Updated**: February 2026  
**Version**: 1.0.0
