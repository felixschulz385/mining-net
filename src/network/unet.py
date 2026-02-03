"""UNet architecture for mining segmentation.

This module implements a comprehensive UNet model for semantic segmentation
of mining areas from multi-spectral Landsat imagery.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Tuple, Optional


class ConvBlock(layers.Layer):
    """Convolutional block with two conv layers, batch norm, and ReLU."""
    
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
        name: str = "conv_block"
    ):
        """Initialize convolutional block.
        
        Args:
            filters: Number of filters
            kernel_size: Size of convolutional kernel
            dropout_rate: Dropout rate (0 = no dropout)
            name: Block name
        """
        super().__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # First convolution
        self.conv1 = layers.Conv2D(
            filters,
            kernel_size,
            padding='same',
            kernel_initializer='he_normal'
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        # Second convolution
        self.conv2 = layers.Conv2D(
            filters,
            kernel_size,
            padding='same',
            kernel_initializer='he_normal'
        )
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        
        # Optional dropout
        if dropout_rate > 0:
            self.dropout = layers.Dropout(dropout_rate)
        else:
            self.dropout = None
    
    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        
        if self.dropout is not None:
            x = self.dropout(x, training=training)
        
        return x


class EncoderBlock(layers.Layer):
    """Encoder block with conv block and max pooling."""
    
    def __init__(
        self,
        filters: int,
        dropout_rate: float = 0.0,
        name: str = "encoder_block"
    ):
        """Initialize encoder block.
        
        Args:
            filters: Number of filters
            dropout_rate: Dropout rate
            name: Block name
        """
        super().__init__(name=name)
        self.conv_block = ConvBlock(filters, dropout_rate=dropout_rate)
        self.pool = layers.MaxPooling2D(pool_size=(2, 2))
    
    def call(self, inputs, training=False):
        """Forward pass."""
        x = self.conv_block(inputs, training=training)
        p = self.pool(x)
        return x, p


class DecoderBlock(layers.Layer):
    """Decoder block with upsampling, concatenation, and conv block."""
    
    def __init__(
        self,
        filters: int,
        dropout_rate: float = 0.0,
        name: str = "decoder_block"
    ):
        """Initialize decoder block.
        
        Args:
            filters: Number of filters
            dropout_rate: Dropout rate
            name: Block name
        """
        super().__init__(name=name)
        self.upconv = layers.Conv2DTranspose(
            filters,
            kernel_size=2,
            strides=2,
            padding='same'
        )
        self.concat = layers.Concatenate()
        self.conv_block = ConvBlock(filters, dropout_rate=dropout_rate)
    
    def call(self, inputs, skip_connection, training=False):
        """Forward pass.
        
        Args:
            inputs: Input tensor from previous layer
            skip_connection: Skip connection from encoder
            training: Training mode
        """
        x = self.upconv(inputs)
        x = self.concat([x, skip_connection])
        x = self.conv_block(x, training=training)
        return x


class UNet(Model):
    """UNet model for mining segmentation.
    
    Architecture:
        - Encoder: 4 encoding blocks with increasing filters (64, 128, 256, 512)
        - Bottleneck: 1024 filters
        - Decoder: 4 decoding blocks with decreasing filters (512, 256, 128, 64)
        - Output: Single channel with sigmoid activation for binary segmentation
    
    Input:
        - Shape: (batch, 64, 64, 7) - 7 Landsat bands
        - Bands: [blue, green, red, nir, swir1, swir2, thermal]
    
    Output:
        - Shape: (batch, 64, 64, 1) - Binary mining footprint mask
        - Values: [0, 1] probability of mining pixel
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (64, 64, 7),
        num_classes: int = 1,
        filters_base: int = 64,
        depth: int = 4,
        dropout_rate: float = 0.1,
        name: str = "unet"
    ):
        """Initialize UNet model.
        
        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes (1 for binary)
            filters_base: Base number of filters (doubled at each level)
            depth: Depth of encoder/decoder (number of pooling operations)
            dropout_rate: Dropout rate for regularization
            name: Model name
        """
        super().__init__(name=name)
        
        self.input_shape_ = input_shape
        self.num_classes = num_classes
        self.filters_base = filters_base
        self.depth = depth
        self.dropout_rate = dropout_rate
        
        # Build encoder blocks
        self.encoders = []
        for i in range(depth):
            filters = filters_base * (2 ** i)
            self.encoders.append(
                EncoderBlock(
                    filters,
                    dropout_rate=dropout_rate,
                    name=f"encoder_{i}"
                )
            )
        
        # Bottleneck
        bottleneck_filters = filters_base * (2 ** depth)
        self.bottleneck = ConvBlock(
            bottleneck_filters,
            dropout_rate=dropout_rate,
            name="bottleneck"
        )
        
        # Build decoder blocks
        self.decoders = []
        for i in range(depth - 1, -1, -1):
            filters = filters_base * (2 ** i)
            self.decoders.append(
                DecoderBlock(
                    filters,
                    dropout_rate=dropout_rate,
                    name=f"decoder_{i}"
                )
            )
        
        # Output layer
        self.output_layer = layers.Conv2D(
            num_classes,
            kernel_size=1,
            activation='sigmoid',
            name='output'
        )
    
    def call(self, inputs, training=False):
        """Forward pass.
        
        Args:
            inputs: Input tensor of shape (batch, H, W, C)
            training: Training mode
            
        Returns:
            Output tensor of shape (batch, H, W, num_classes)
        """
        # Encoder path
        skip_connections = []
        x = inputs
        
        for encoder in self.encoders:
            skip, x = encoder(x, training=training)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x, training=training)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for decoder, skip in zip(self.decoders, skip_connections):
            x = decoder(x, skip, training=training)
        
        # Output
        outputs = self.output_layer(x)
        
        return outputs
    
    def get_config(self):
        """Get configuration for serialization."""
        return {
            'input_shape': self.input_shape_,
            'num_classes': self.num_classes,
            'filters_base': self.filters_base,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'name': self.name
        }


def build_unet(
    input_shape: Tuple[int, int, int] = (64, 64, 7),
    num_classes: int = 1,
    filters_base: int = 64,
    depth: int = 4,
    dropout_rate: float = 0.1
) -> Model:
    """Build and compile UNet model.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        filters_base: Base number of filters
        depth: Depth of encoder/decoder
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    model = UNet(
        input_shape=input_shape,
        num_classes=num_classes,
        filters_base=filters_base,
        depth=depth,
        dropout_rate=dropout_rate
    )
    
    # Build model by calling it once
    dummy_input = tf.zeros((1, *input_shape))
    _ = model(dummy_input)
    
    return model


if __name__ == "__main__":
    # Example usage
    model = build_unet()
    model.summary()
    
    # Test with random input
    test_input = tf.random.normal((2, 64, 64, 7))
    test_output = model(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {test_output.shape}")
