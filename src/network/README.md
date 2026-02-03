# Mining Segmentation Network

Comprehensive UNet architecture for semantic segmentation of mining areas from multi-spectral Landsat imagery.

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
- **Comprehensive Metrics**: Dice coefficient, IoU, precision, recall, F1-score, AUC
- **Mixed Precision Training**: Optional for faster training on compatible GPUs
- **Callbacks**: Early stopping, learning rate reduction, checkpointing, TensorBoard logging

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
python src/network/evaluate.py path/to/checkpoint.h5 \
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
evaluator = MiningSegmentationEvaluator('path/to/checkpoint.h5')

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

## Configuration

Key configuration options in [config.py](src/network/config.py):

```python
# Model architecture
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
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# Data augmentation
USE_AUGMENTATION = True
FLIP_HORIZONTAL = True
FLIP_VERTICAL = True
ROTATE_90 = True
```

## Model Performance

The model uses several metrics to evaluate performance:

- **Dice Coefficient**: Measures overlap between prediction and ground truth (0-1, higher is better)
- **IoU (Jaccard Index)**: Intersection over union metric (0-1, higher is better)
- **Precision**: Proportion of predicted mining pixels that are correct
- **Recall**: Proportion of actual mining pixels that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

## Data Requirements

The network expects data from [MiningSegmentationDataLoader](../data/data_loader.py):
- 64×64 pixel tiles at ~30m resolution
- 7 Landsat bands: blue, green, red, nir, swir1, swir2, thermal
- Binary mining footprint labels (0 = non-mining, 1 = mining)

## Advanced Features

### Class Weighting

For imbalanced datasets (few mining pixels), enable class weighting:
```python
config.USE_CLASS_WEIGHTS = True
config.POSITIVE_CLASS_WEIGHT = 10.0  # Increase weight for mining class
```

### Mixed Precision Training

For faster training on GPUs with Tensor Cores (e.g., V100, A100):
```python
config.USE_MIXED_PRECISION = True
```

### Custom Loss Functions

Modify the loss function in [utils.py](src/network/utils.py) or select in config:
- `bce`: Binary cross-entropy
- `dice`: Dice loss (good for imbalanced data)
- `focal`: Focal loss (emphasizes hard examples)
- `combined`: Weighted combination of BCE and Dice

## Output

Training outputs:
- **Checkpoints**: Saved in `models/checkpoints/`
- **TensorBoard logs**: Saved in `logs/`
- **Training history**: CSV and plots in checkpoint directory

## Requirements

- TensorFlow >= 2.10
- numpy
- xarray
- zarr
- matplotlib
- tqdm

## License

See project LICENSE file.
