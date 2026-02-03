"""Training script for mining segmentation UNet model."""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.unet import build_unet
from network.config import NetworkConfig
from network.utils import (
    combined_loss,
    dice_loss,
    focal_loss,
    DiceCoefficient,
    IoU,
    augment_tile,
    prepare_for_training,
    plot_training_history
)
from data.data_loader import MiningSegmentationDataLoader
from data.config import Config as DataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MiningSegmentationTrainer:
    """Trainer for mining segmentation model."""
    
    def __init__(
        self,
        network_config: Optional[NetworkConfig] = None,
        data_config: Optional[DataConfig] = None
    ):
        """Initialize trainer.
        
        Args:
            network_config: Network configuration
            data_config: Data configuration
        """
        self.network_config = network_config or NetworkConfig()
        self.data_config = data_config or DataConfig()
        
        # Create necessary directories
        self.network_config.create_directories()
        
        # Initialize data loader
        self.data_loader = MiningSegmentationDataLoader(config=self.data_config)
        
        # Model will be initialized in setup_model
        self.model = None
        
        # Setup mixed precision if enabled
        if self.network_config.USE_MIXED_PRECISION:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled")
    
    def setup_model(self):
        """Build and compile the model."""
        logger.info("Building UNet model...")
        
        # Build model
        self.model = build_unet(
            input_shape=self.network_config.INPUT_SHAPE,
            num_classes=self.network_config.NUM_CLASSES,
            filters_base=self.network_config.FILTERS_BASE,
            depth=self.network_config.DEPTH,
            dropout_rate=self.network_config.DROPOUT_RATE
        )
        
        # Setup optimizer with learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.network_config.LEARNING_RATE,
            decay_steps=self.network_config.DECAY_STEPS,
            decay_rate=self.network_config.LEARNING_RATE_DECAY,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Use mixed precision optimizer if enabled
        if self.network_config.USE_MIXED_PRECISION:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # Select loss function
        if self.network_config.LOSS_TYPE == 'bce':
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        elif self.network_config.LOSS_TYPE == 'dice':
            loss_fn = dice_loss
        elif self.network_config.LOSS_TYPE == 'focal':
            loss_fn = lambda y_t, y_p: focal_loss(
                y_t, y_p,
                alpha=self.network_config.FOCAL_ALPHA,
                gamma=self.network_config.FOCAL_GAMMA
            )
        elif self.network_config.LOSS_TYPE == 'combined':
            loss_fn = lambda y_t, y_p: combined_loss(
                y_t, y_p,
                bce_weight=self.network_config.BCE_WEIGHT,
                dice_weight=self.network_config.DICE_WEIGHT
            )
        else:
            raise ValueError(f"Unknown loss type: {self.network_config.LOSS_TYPE}")
        
        # Compile model
        metrics = [
            'accuracy',
            DiceCoefficient(),
            IoU(),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        
        logger.info("Model compiled successfully")
        self.model.summary(print_fn=logger.info)
    
    def create_datasets(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None
    ):
        """Create training and validation datasets.
        
        Args:
            countries: Filter by country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        logger.info("Creating datasets...")
        
        # Get all available tiles
        tiles = self.data_loader.get_written_tiles(countries, years, cluster_ids)
        
        if not tiles:
            raise ValueError("No tiles found matching the specified filters")
        
        logger.info(f"Found {len(tiles)} tiles")
        
        # Split into train and validation
        np.random.shuffle(tiles)
        n_val = int(len(tiles) * self.network_config.VALIDATION_SPLIT)
        val_tiles = tiles[:n_val]
        train_tiles = tiles[n_val:]
        
        logger.info(f"Training tiles: {len(train_tiles)}, Validation tiles: {len(val_tiles)}")
        
        # Create datasets
        train_dataset = self._create_dataset_from_tiles(
            train_tiles,
            shuffle=True,
            augment=self.network_config.USE_AUGMENTATION
        )
        
        val_dataset = self._create_dataset_from_tiles(
            val_tiles,
            shuffle=False,
            augment=False
        )
        
        return train_dataset, val_dataset
    
    def _create_dataset_from_tiles(
        self,
        tiles: List[dict],
        shuffle: bool = True,
        augment: bool = False
    ):
        """Create TensorFlow dataset from tile list.
        
        Args:
            tiles: List of tile metadata dicts
            shuffle: Whether to shuffle
            augment: Whether to apply augmentation
            
        Returns:
            tf.data.Dataset
        """
        # Use data loader's create_tf_dataset method for consistency
        # This ensures proper shape handling
        logger.info(f"Creating dataset from {len(tiles)} tiles")
        
        tile_indices = [(t['tile_ix'], t['tile_iy']) for t in tiles]
        
        def tile_generator():
            for tile_ix, tile_iy in tile_indices:
                try:
                    features, labels = self.data_loader.load_tile_data(
                        tile_ix,
                        tile_iy,
                        bands=self.network_config.INPUT_BANDS,
                        include_footprint=True
                    )
                    
                    # Skip tiles with all NaN
                    if not np.all(np.isnan(features)):
                        # Replace NaN with 0
                        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                        labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        # Ensure labels have shape (H, W, 1)
                        if labels.ndim == 2:
                            labels = labels[..., np.newaxis]
                        
                        yield features, labels
                        
                except Exception as e:
                    logger.warning(f"Error loading tile ({tile_ix}, {tile_iy}): {e}")
                    continue
        
        # Create dataset with consistent shapes
        tile_size = self.network_config.INPUT_SHAPE[0]
        n_bands = self.network_config.INPUT_SHAPE[2]
        
        dataset = tf.data.Dataset.from_generator(
            tile_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=(tile_size, tile_size, n_bands),
                    dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=(tile_size, tile_size, 1),
                    dtype=tf.float32
                )
            )
        )
        
        # Apply augmentation if requested (data is already prepared with correct shapes)
        if augment:
            dataset = dataset.map(
                lambda f, l: augment_tile(f, l, self.network_config),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Shuffle and batch
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(self.network_config.BATCH_SIZE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def setup_callbacks(self, run_name: str):
        """Setup training callbacks.
        
        Args:
            run_name: Name for this training run
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = self.network_config.CHECKPOINT_DIR / f"{run_name}_best.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=self.network_config.SAVE_BEST_ONLY,
            save_weights_only=self.network_config.SAVE_WEIGHTS_ONLY,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.network_config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.network_config.REDUCE_LR_FACTOR,
            patience=self.network_config.REDUCE_LR_PATIENCE,
            min_lr=self.network_config.MIN_LR,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        log_dir = self.network_config.LOG_DIR / run_name
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            update_freq=self.network_config.UPDATE_FREQ,
            histogram_freq=1
        )
        callbacks.append(tensorboard)
        
        # CSV logger
        csv_path = self.network_config.CHECKPOINT_DIR / f"{run_name}_history.csv"
        csv_logger = tf.keras.callbacks.CSVLogger(str(csv_path))
        callbacks.append(csv_logger)
        
        logger.info(f"Callbacks configured. Checkpoint: {checkpoint_path}")
        logger.info(f"TensorBoard logs: {log_dir}")
        
        return callbacks
    
    def train(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None,
        run_name: Optional[str] = None
    ):
        """Train the model.
        
        Args:
            countries: Filter by country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            run_name: Name for this training run
            
        Returns:
            Training history
        """
        # Generate run name if not provided
        if run_name is None:
            run_name = f"unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting training run: {run_name}")
        
        # Setup model if not already done
        if self.model is None:
            self.setup_model()
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets(
            countries=countries,
            years=years,
            cluster_ids=cluster_ids
        )
        
        # Setup callbacks
        callbacks = self.setup_callbacks(run_name)
        
        # Train model
        logger.info(f"Training for {self.network_config.EPOCHS} epochs...")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.network_config.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        
        # Plot and save training history
        plot_path = self.network_config.CHECKPOINT_DIR / f"{run_name}_history.png"
        plot_training_history(history, save_path=plot_path)
        
        return history
    
    def load_weights(self, checkpoint_path: str):
        """Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.model is None:
            self.setup_model()
        
        logger.info(f"Loading weights from {checkpoint_path}")
        self.model.load_weights(checkpoint_path)
        logger.info("Weights loaded successfully")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train UNet for mining segmentation')
    parser.add_argument(
        '--countries',
        nargs='+',
        help='Country codes to filter (e.g., ZAF USA)'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        help='Years to filter (e.g., 2019 2020)'
    )
    parser.add_argument(
        '--cluster-ids',
        nargs='+',
        type=int,
        help='Cluster IDs to filter'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        help='Name for this training run'
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    config = NetworkConfig()
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    # Initialize trainer
    trainer = MiningSegmentationTrainer(network_config=config)
    
    # Train model
    trainer.train(
        countries=args.countries,
        years=args.years,
        cluster_ids=args.cluster_ids,
        run_name=args.run_name
    )


if __name__ == "__main__":
    main()
