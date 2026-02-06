"""UNet architecture for mining segmentation.

This module implements a comprehensive UNet model for semantic segmentation
of mining areas from multi-spectral Landsat imagery.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Convolutional block with two conv layers, batch norm, and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.0
    ):
        """Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            dropout_rate: Dropout rate (0 = no dropout)
        """
        super().__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Optional dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
        
        return x


class EncoderBlock(nn.Module):
    """Encoder block with conv block and max pooling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0
    ):
        """Initialize encoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        """Forward pass.
        
        Returns:
            Tuple of (skip_connection, pooled_output)
        """
        skip = self.conv_block(x)
        pooled = self.pool(skip)
        return skip, pooled


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, concatenation, and conv block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: float = 0.0
    ):
        """Initialize decoder block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )
        self.conv_block = ConvBlock(
            in_channels,  # Concatenated channels (out_channels + out_channels from skip)
            out_channels,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x, skip_connection):
        """Forward pass.
        
        Args:
            x: Input tensor from previous layer
            skip_connection: Skip connection from encoder
        """
        x = self.upconv(x)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    """UNet model for mining segmentation.
    
    Architecture:
        - Encoder: 4 encoding blocks with increasing filters (64, 128, 256, 512)
        - Bottleneck: 1024 filters
        - Decoder: 4 decoding blocks with decreasing filters (512, 256, 128, 64)
        - Output: Single channel with sigmoid activation for binary segmentation
    
    Input:
        - Shape: (batch, 7, 64, 64) - 7 Landsat bands
        - Bands: [blue, green, red, nir, swir1, swir2, thermal]
    
    Output:
        - Shape: (batch, 1, 64, 64) - Binary mining footprint mask
        - Values: [0, 1] probability of mining pixel
    """
    
    def __init__(
        self,
        in_channels: int = 7,
        num_classes: int = 1,
        filters_base: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1
    ):
        """Initialize UNet model.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes (1 for binary)
            filters_base: Base number of filters (doubled at each level)
            depth: Depth of encoder/decoder (number of pooling operations)
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.filters_base = filters_base
        self.depth = depth
        self.dropout_rate = dropout_rate
        
        # Build encoder blocks
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for i in range(depth):
            out_channels = filters_base * (2 ** i)
            self.encoders.append(
                EncoderBlock(
                    current_channels,
                    out_channels,
                    dropout_rate=dropout_rate
                )
            )
            current_channels = out_channels
        
        # Bottleneck
        bottleneck_filters = filters_base * (2 ** depth)
        self.bottleneck = ConvBlock(
            current_channels,
            bottleneck_filters,
            dropout_rate=dropout_rate
        )
        
        # Build decoder blocks
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_filters
        for i in range(depth - 1, -1, -1):
            out_channels = filters_base * (2 ** i)
            self.decoders.append(
                DecoderBlock(
                    current_channels,
                    out_channels,
                    dropout_rate=dropout_rate
                )
            )
            current_channels = out_channels
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(current_channels, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, C, H, W)
            
        Returns:
            Output tensor of shape (batch, num_classes, H, W)
        """
        # Encoder path
        skip_connections = []
        
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for decoder, skip in zip(self.decoders, skip_connections):
            x = decoder(x, skip)
        
        # Output
        outputs = self.output_layer(x)
        
        return outputs


def build_unet(
    in_channels: int = 7,
    num_classes: int = 1,
    filters_base: int = 64,
    depth: int = 4,
    dropout_rate: float = 0.1
) -> UNet:
    """Build UNet model.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        filters_base: Base number of filters
        depth: Depth of encoder/decoder
        dropout_rate: Dropout rate
        
    Returns:
        UNet model
    """
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        filters_base=filters_base,
        depth=depth,
        dropout_rate=dropout_rate
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    model = build_unet()
    print(model)
    
    # Test with random input
    test_input = torch.randn(2, 7, 64, 64)
    test_output = model(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
