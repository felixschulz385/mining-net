"""Evaluation and inference script for mining segmentation model."""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import torch
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
from network.data_loader import MiningSegmentationDataLoader
from config import Config as DataConfig

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
        data_config = None
    ):
        """Initialize evaluator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            network_config: Network configuration
            data_config: Data configuration (from config.Config)
        """
        self.network_config = network_config or NetworkConfig()
        self.data_config = data_config or DataConfig()
        self.checkpoint_path = checkpoint_path
        
        # Setup device
        if self.network_config.DEVICE == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif self.network_config.DEVICE == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            logger.info("Using Apple Silicon GPU")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Initialize data loader (will be created per-need for tile access)
        self.data_config = data_config
        
        # Build and load model
        self.model = self._load_model()
    
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint.
        
        Returns:
            Loaded PyTorch model
        """
        logger.info("Building model...")
        model = build_unet(
            in_channels=self.network_config.IN_CHANNELS,
            num_classes=self.network_config.NUM_CLASSES,
            filters_base=self.network_config.FILTERS_BASE,
            depth=self.network_config.DEPTH,
            dropout_rate=self.network_config.DROPOUT_RATE
        )
        
        logger.info(f"Loading weights from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        # Optionally compile model (PyTorch 2.0+)
        if self.network_config.USE_COMPILE:
            try:
                logger.info(f"Compiling model with mode='{self.network_config.COMPILE_MODE}'...")
                model = torch.compile(
                    model,
                    mode=self.network_config.COMPILE_MODE
                )
                logger.info("Model compiled successfully")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}. Continuing without compilation.")
        
        model.eval()
        
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
        
        # Create dataset
        dataset = MiningSegmentationDataLoader(
            config=self.data_config,
            countries=countries,
            years=years,
            cluster_ids=cluster_ids,
            bands=self.network_config.INPUT_BANDS
        )
        
        if len(dataset) == 0:
            raise ValueError("No tiles found with specified filters")
        
        logger.info(f"Evaluating on {len(dataset)} tiles")
        
        # Create data loader
        from torch.utils.data import DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.network_config.NUM_WORKERS,
            pin_memory=self.network_config.PIN_MEMORY
        )
        
        # Initialize metrics
        dice_metric = DiceCoefficient()
        iou_metric = IoU()
        
        # Process batches
        with torch.no_grad():
            for batch_features, batch_labels in tqdm(data_loader, desc="Evaluating"):
                # Move to device
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Predict
                predictions = self.model(batch_features)
                
                # Update metrics
                dice_metric.update(predictions, batch_labels)
                iou_metric.update(predictions, batch_labels)
        
        # Compute results
        results = {
            'dice_coefficient': dice_metric.compute(),
            'iou': iou_metric.compute()
        }
        
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
        # Create a temporary dataset for this single tile
        temp_dataset = MiningSegmentationDataLoader(
            config=self.data_config,
            bands=self.network_config.INPUT_BANDS
        )
        
        # Use the get_tile_by_index method
        features_tensor, labels_tensor = temp_dataset.get_tile_by_index(tile_ix, tile_iy)
        
        # Add batch dimension and move to device
        features_batch = features_tensor.unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(features_batch)
        
        # Convert to numpy and remove batch dimension
        prediction = prediction.cpu().squeeze().numpy()
        
        # Convert features and labels to numpy (H, W, C) format for visualization
        features = features_tensor.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
        labels = labels_tensor.squeeze().numpy()  # (1, H, W) -> (H, W)
        
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
        
        # Create dataset
        dataset = MiningSegmentationDataLoader(
            config=self.data_config,
            countries=countries,
            years=years,
            cluster_ids=cluster_ids,
            bands=self.network_config.INPUT_BANDS
        )
        
        if len(dataset) == 0:
            raise ValueError("No tiles found with specified filters")
        
        # Randomly sample tiles
        num_samples = min(num_samples, len(dataset))
        sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        features_list = []
        labels_list = []
        predictions_list = []
        
        with torch.no_grad():
            for idx in sample_indices:
                # Get tile from dataset
                features_tensor, labels_tensor = dataset[idx]
                
                # Add batch dimension and predict
                features_batch = features_tensor.unsqueeze(0).to(self.device)
                prediction = self.model(features_batch)
                
                # Convert to numpy for visualization
                features = features_tensor.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
                labels = labels_tensor.squeeze().numpy()  # (1, H, W) -> (H, W)
                prediction = prediction.cpu().squeeze().numpy()  # -> (H, W)
                
                features_list.append(features)
                labels_list.append(labels)
                predictions_list.append(prediction)
        
        # Convert to arrays
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        predictions_array = np.array(predictions_list)
        
        # Create save path
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "predictions.png"
        else:
            save_path = None
        
        # Visualize (features in H,W,C format, predictions need channel dim for viz function)
        visualize_predictions(
            features_array,
            labels_array,
            predictions_array[:, np.newaxis, :, :],  # Add channel dimension
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
        
        # Create dataset
        dataset = MiningSegmentationDataLoader(
            config=self.data_config,
            countries=countries,
            years=years,
            cluster_ids=cluster_ids,
            bands=self.network_config.INPUT_BANDS
        )
        
        if len(dataset) == 0:
            raise ValueError("No tiles found with specified filters")
        
        logger.info(f"Processing {len(dataset)} tiles")
        
        with torch.no_grad():
            for idx in tqdm(range(len(dataset)), desc="Exporting"):
                tile = dataset.tiles[idx]
                
                # Get prediction
                features_tensor, labels_tensor = dataset[idx]
                features_batch = features_tensor.unsqueeze(0).to(self.device)
                prediction = self.model(features_batch)
                
                # Convert to numpy
                features = features_tensor.permute(1, 2, 0).numpy()
                labels = labels_tensor.squeeze().numpy()
                prediction = prediction.cpu().squeeze().numpy()
                
                # Save prediction as numpy array
                output_path = output_dir / f"tile_{tile['tile_ix']}_{tile['tile_iy']}.npz"
                np.savez_compressed(
                    output_path,
                    features=features,
                    labels=labels,
                    prediction=prediction,
                    tile_ix=tile['tile_ix'],
                    tile_iy=tile['tile_iy']
                )
        
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
