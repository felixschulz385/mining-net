"""Evaluation and inference script for mining segmentation model."""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import tensorflow as tf
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from network.unet import build_unet
from network.config import NetworkConfig
from network.utils import (
    DiceCoefficient,
    IoU,
    prepare_for_training,
    visualize_predictions
)
from data.data_loader import MiningSegmentationDataLoader
from data.config import Config as DataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MiningSegmentationEvaluator:
    """Evaluator for mining segmentation model."""
    
    def __init__(
        self,
        checkpoint_path: str,
        network_config: Optional[NetworkConfig] = None,
        data_config: Optional[DataConfig] = None
    ):
        """Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            network_config: Network configuration
            data_config: Data configuration
        """
        self.network_config = network_config or NetworkConfig()
        self.data_config = data_config or DataConfig()
        self.checkpoint_path = checkpoint_path
        
        # Initialize data loader
        self.data_loader = MiningSegmentationDataLoader(config=self.data_config)
        
        # Build and load model
        self.model = self._load_model()
    
    def _load_model(self) -> tf.keras.Model:
        """Load model from checkpoint.
        
        Returns:
            Loaded Keras model
        """
        logger.info("Building model...")
        model = build_unet(
            input_shape=self.network_config.INPUT_SHAPE,
            num_classes=self.network_config.NUM_CLASSES,
            filters_base=self.network_config.FILTERS_BASE,
            depth=self.network_config.DEPTH,
            dropout_rate=self.network_config.DROPOUT_RATE
        )
        
        logger.info(f"Loading weights from {self.checkpoint_path}")
        model.load_weights(self.checkpoint_path)
        logger.info("Model loaded successfully")
        
        return model
    
    def evaluate(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None,
        batch_size: int = 32
    ) -> dict:
        """Evaluate model on test data.
        
        Args:
            countries: Filter by country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting evaluation...")
        
        # Get test tiles
        tiles = self.data_loader.get_written_tiles(countries, years, cluster_ids)
        
        if not tiles:
            raise ValueError("No tiles found matching the specified filters")
        
        logger.info(f"Evaluating on {len(tiles)} tiles")
        
        # Initialize metrics
        dice_metric = DiceCoefficient()
        iou_metric = IoU()
        precision_metric = tf.keras.metrics.Precision()
        recall_metric = tf.keras.metrics.Recall()
        auc_metric = tf.keras.metrics.AUC()
        
        # Process tiles in batches
        for i in tqdm(range(0, len(tiles), batch_size)):
            batch_tiles = tiles[i:i + batch_size]
            
            features_batch = []
            labels_batch = []
            
            for tile in batch_tiles:
                try:
                    features, labels = self.data_loader.load_tile_data(
                        tile['tile_ix'],
                        tile['tile_iy'],
                        bands=self.network_config.INPUT_BANDS,
                        include_footprint=True
                    )
                    
                    # Skip invalid tiles
                    if np.all(np.isnan(features)):
                        continue
                    
                    # Replace NaN with 0
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Ensure labels have shape (H, W, 1)
                    if labels.ndim == 2:
                        labels = labels[..., np.newaxis]
                    
                    features_batch.append(features)
                    labels_batch.append(labels)
                    
                except Exception as e:
                    logger.warning(f"Error loading tile: {e}")
                    continue
            
            if not features_batch:
                continue
            
            # Convert to arrays (labels already have shape (H, W, 1))
            features_batch = np.array(features_batch, dtype=np.float32)
            labels_batch = np.array(labels_batch, dtype=np.float32)
            
            # Predict
            predictions = self.model.predict(features_batch, verbose=0)
            
            # Update metrics
            dice_metric.update_state(labels_batch, predictions)
            iou_metric.update_state(labels_batch, predictions)
            precision_metric.update_state(labels_batch, predictions > 0.5)
            recall_metric.update_state(labels_batch, predictions > 0.5)
            auc_metric.update_state(labels_batch, predictions)
        
        # Compute results
        results = {
            'dice_coefficient': float(dice_metric.result().numpy()),
            'iou': float(iou_metric.result().numpy()),
            'precision': float(precision_metric.result().numpy()),
            'recall': float(recall_metric.result().numpy()),
            'auc': float(auc_metric.result().numpy())
        }
        
        # Compute F1 score
        if results['precision'] + results['recall'] > 0:
            results['f1_score'] = (
                2 * results['precision'] * results['recall'] /
                (results['precision'] + results['recall'])
            )
        else:
            results['f1_score'] = 0.0
        
        logger.info("Evaluation completed!")
        logger.info("Results:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def predict_tile(
        self,
        tile_ix: int,
        tile_iy: int,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict on a single tile.
        
        Args:
            tile_ix: Tile X index
            tile_iy: Tile Y index
            threshold: Classification threshold
            
        Returns:
            Tuple of (features, ground_truth, prediction)
        """
        # Load tile data
        features, labels = self.data_loader.load_tile_data(
            tile_ix,
            tile_iy,
            bands=self.network_config.INPUT_BANDS,
            include_footprint=True
        )
        
        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0)
        labels = np.nan_to_num(labels, nan=0)
        
        # Predict
        features_batch = features[np.newaxis, ...]
        prediction = self.model.predict(features_batch, verbose=0)[0]
        
        return features, labels, prediction
    
    def visualize_predictions(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None,
        num_samples: int = 8,
        save_dir: Optional[Path] = None
    ):
        """Visualize model predictions on sample tiles.
        
        Args:
            countries: Filter by country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            num_samples: Number of samples to visualize
            save_dir: Directory to save visualizations
        """
        logger.info(f"Generating visualizations for {num_samples} samples...")
        
        # Get tiles
        tiles = self.data_loader.get_written_tiles(countries, years, cluster_ids)
        
        if not tiles:
            raise ValueError("No tiles found")
        
        # Randomly sample tiles
        np.random.shuffle(tiles)
        sample_tiles = tiles[:num_samples]
        
        features_list = []
        labels_list = []
        predictions_list = []
        
        for tile in sample_tiles:
            try:
                features, labels, prediction = self.predict_tile(
                    tile['tile_ix'],
                    tile['tile_iy']
                )
                
                features_list.append(features)
                labels_list.append(labels)
                predictions_list.append(prediction)
                
            except Exception as e:
                logger.warning(f"Error processing tile: {e}")
                continue
        
        if not features_list:
            logger.error("No valid tiles to visualize")
            return
        
        # Convert to arrays
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        predictions_array = np.array(predictions_list)
        
        # Create save path
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "predictions_visualization.png"
        else:
            save_path = None
        
        # Visualize
        visualize_predictions(
            features_array,
            labels_array,
            predictions_array,
            num_samples=len(features_list),
            save_path=save_path
        )
        
        logger.info("Visualization completed!")
    
    def export_predictions(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        cluster_ids: Optional[List[int]] = None,
        output_dir: Optional[Path] = None,
        threshold: float = 0.5
    ):
        """Export predictions for all tiles to disk.
        
        Args:
            countries: Filter by country codes
            years: Filter by years
            cluster_ids: Filter by cluster IDs
            output_dir: Directory to save predictions
            threshold: Classification threshold
        """
        if output_dir is None:
            output_dir = Path("predictions")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting predictions to {output_dir}")
        
        # Get tiles
        tiles = self.data_loader.get_written_tiles(countries, years, cluster_ids)
        
        if not tiles:
            raise ValueError("No tiles found")
        
        logger.info(f"Processing {len(tiles)} tiles...")
        
        for tile in tqdm(tiles):
            try:
                features, labels, prediction = self.predict_tile(
                    tile['tile_ix'],
                    tile['tile_iy'],
                    threshold=threshold
                )
                
                # Save prediction
                filename = f"tile_{tile['tile_ix']}_{tile['tile_iy']}_pred.npy"
                np.save(output_dir / filename, prediction)
                
            except Exception as e:
                logger.warning(f"Error processing tile {tile['tile_ix']}, {tile['tile_iy']}: {e}")
                continue
        
        logger.info("Export completed!")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate UNet for mining segmentation')
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--countries',
        nargs='+',
        help='Country codes to filter'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        type=int,
        help='Years to filter'
    )
    parser.add_argument(
        '--cluster-ids',
        nargs='+',
        type=int,
        help='Cluster IDs to filter'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization of predictions'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=8,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export all predictions to disk'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save outputs'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = MiningSegmentationEvaluator(args.checkpoint)
    
    # Run evaluation
    results = evaluator.evaluate(
        countries=args.countries,
        years=args.years,
        cluster_ids=args.cluster_ids
    )
    
    # Visualize if requested
    if args.visualize:
        evaluator.visualize_predictions(
            countries=args.countries,
            years=args.years,
            cluster_ids=args.cluster_ids,
            num_samples=args.num_samples,
            save_dir=args.output_dir
        )
    
    # Export if requested
    if args.export:
        evaluator.export_predictions(
            countries=args.countries,
            years=args.years,
            cluster_ids=args.cluster_ids,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
