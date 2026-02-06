# Mining Segmentation Network

Comprehensive UNet architecture for semantic segmentation of mining areas from multi-spectral Landsat imagery, implemented in **PyTorch**.

## Architecture

The network uses a standard UNet architecture with:
- **Input**: 64×64 tiles with 7 Landsat bands (blue, green, red, NIR, SWIR1, SWIR2, thermal)
- **Output**: Binary segmentation mask (mining vs. non-mining pixels)
- **Encoder**: 4 levels with filters [64, 128, 256, 512]
- **Bottleneck**: 1024 filters
- **Decoder**: 4 levels with skip connections from encoder

### Key Features

- **Multiple Loss Functions**: Binary cross-entropy, Dice loss, Focal loss, or combined
- **Data Augmentation**: Random flips, rotations, brightness, and contrast adjustments
- **Comprehensive Metrics**: Dice coefficient, IoU
- **GPU Support**: CUDA, MPS (Apple Silicon), and CPU
- **TensorBoard Logging**: Track training progress with TensorBoard
- **Checkpointing**: Save best model and training history

## Files

- **unet.py**: UNet model architecture with encoder-decoder blocks
- **train.py**: Training pipeline with data loading and augmentation
- **evaluate.py**: Model evaluation and inference
- **config.py**: Hyperparameters and training configuration
- **utils.py**: Loss functions, metrics, visualization utilities
- **__init__.py**: Package initialization

## Usage

### Training

Basic training:
```bash
python src/network/train.py
```

With filters:
```bash
python src/network/train.py \
    --countries ZAF USA \
    --years 2019 2020 \
    --epochs 50 \
    --batch-size 32 \
    --run-name my_experiment
```

### Evaluation

```bash
python src/network/evaluate.py path/to/checkpoint.h5 \
    --countries ZAF \
    --visualize \
    --num-samples 8
```

Export predictions:
```bash
python src/network/evaluate.py path/to/checkpoint.pth \
    --export \
    --output-dir predictions/
```

### Programmatic Usage

```python
from network.train import MiningSegmentationTrainer
from network.config import NetworkConfig

# Configure training
config = NetworkConfig()
config.EPOCHS = 50
config.BATCH_SIZE = 32
config.LEARNING_RATE = 1e-4
config.DEVICE = 'cuda'  # or 'cpu' or 'mps'

# Initialize trainer
trainer = MiningSegmentationTrainer(network_config=config)

# Train model
history = trainer.train(
    countries=['ZAF'],
    years=[2019, 2020],
    run_name='south_africa_experiment'
)
```

### Inference

```python
from network.evaluate import MiningSegmentationEvaluator

# Load model
evaluator = MiningSegmentationEvaluator('path/to/checkpoint.pth')

# Predict on a single tile
features, ground_truth, prediction = evaluator.predict_tile(
    tile_ix=187,
    tile_iy=1148
)

# Evaluate on test set
results = evaluator.evaluate(countries=['ZAF'])
print(f"Dice: {results['dice_coefficient']:.4f}")
print(f"IoU: {results['iou']:.4f}")
```

### Using the PyTorch Dataset

```python
from data.data_loader import MiningSegmentationDataLoader
from torch.utils.data import DataLoader

# Create dataset (lazy-loading from Zarr)
dataset = MiningSegmentationDataLoader(
    countries=['ZAF'],
    years=[2020],
    bands=['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
)

# Create PyTorch DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Iterate through batches
for features, labels in loader:
    # features: (batch_size, 7, 64, 64)
    # labels: (batch_size, 1, 64, 64)
    print(features.shape, labels.shape)
```

## Configuration

Key configuration options in [config.py](config.py):

```python
# Model architecture
IN_CHANNELS = 7
INPUT_SIZE = 64
FILTERS_BASE = 64
DEPTH = 4
DROPOUT_RATE = 0.1

# Training
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 0.95

# Device
DEVICE = 'cuda'  # 'cuda', 'cpu', or 'mps'
NUM_WORKERS = 4
PIN_MEMORY = True

# Loss function
LOSS_TYPE = 'combined'  # 'bce', 'dice', 'focal', 'combined'
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# Data augmentation
USE_AUGMENTATION = True
FLIP_HORIZONTAL = True
FLIP_VERTICAL = True
ROTATE_90 = True
```

## Model Performance

The model uses the following metrics to evaluate performance:

- **Dice Coefficient**: Measures overlap between prediction and ground truth (0-1, higher is better)
- **IoU (Jaccard Index)**: Intersection over union metric (0-1, higher is better)

## Data Requirements

The network expects data from [MiningSegmentationDataLoader](../data/data_loader.py):
- 64×64 pixel tiles at ~30m resolution
- 7 Landsat bands: blue, green, red, nir, swir1, swir2, thermal
- Binary mining footprint labels (0 = non-mining, 1 = mining)
- Data format: PyTorch tensors with shape (batch, channels, height, width)
- **Lazy Loading**: Tiles are loaded on-demand from Zarr, not preloaded into memory

## Advanced Features

### Device Configuration

The model supports multiple device types:
```python
config.DEVICE = 'cuda'  # NVIDIA GPU
config.DEVICE = 'mps'   # Apple Silicon GPU
config.DEVICE = 'cpu'   # CPU only
```

### Data Loading

The PyTorch Dataset implementation uses lazy loading:
- Tiles are only loaded when accessed during training
- Minimal memory footprint - only current batch in memory
- Efficient for large datasets
- Multi-worker support for parallel loading

### Custom Loss Functions

Modify the loss function in [utils.py](utils.py) or select in config:
- `bce`: Binary cross-entropy
- `dice`: Dice loss (good for imbalanced data)
- `focal`: Focal loss (emphasizes hard examples)
- `combined`: Weighted combination of BCE and Dice

### Multi-GPU Training

For multi-GPU training, wrap the model with `DataParallel` or `DistributedDataParallel`:
```python
import torch.nn as nn
model = nn.DataParallel(model)
```

## Output

Training outputs:
- **Checkpoints**: Saved in `models/checkpoints/<run_name>/`
  - `best_model.pth`: Best model based on validation loss
  - `epoch_XXX.pth`: Checkpoint at each epoch
- **TensorBoard logs**: Saved in `logs/<run_name>/`
- **Training history**: Plots saved in checkpoint directory

## Requirements

- **PyTorch** >= 2.0
- torch
- torchvision (optional)
- numpy
- xarray
- zarr
- matplotlib
- tqdm
- tensorboard
- zarr
- matplotlib
- tqdm
- tensorboard

## License

See project LICENSE file.
