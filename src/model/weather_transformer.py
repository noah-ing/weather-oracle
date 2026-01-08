"""Transformer-based neural network for weather forecasting (V3).

Implements a Transformer encoder architecture with positional encoding
for temporal weather prediction. Handles long-range dependencies better
than LSTM-based approaches.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences.

    Adds positional information to input embeddings using sine and cosine
    functions of different frequencies, allowing the model to learn
    temporal relationships.
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        """Initialize positional encoding.

        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length to support
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class WeatherTransformer(nn.Module):
    """Transformer encoder for weather forecasting.

    Architecture:
        1. Input projection: Maps input features to d_model dimensions
        2. Positional encoding: Adds temporal position information
        3. Transformer encoder: Self-attention layers
        4. Output projection: Maps to target sequence

    Input: (batch, 168 hours, N features) - 7 day context window
    Output: (batch, 24 hours, 3 targets) - temp, precip, wind predictions
    """

    def __init__(
        self,
        input_size: int = 12,
        output_size: int = 3,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        input_hours: int = 168,
        output_hours: int = 24,
        dim_feedforward: int = 512,
    ):
        """Initialize WeatherTransformer.

        Args:
            input_size: Number of input features (default 12 for V3)
            output_size: Number of output targets (default 3: temp, precip, wind)
            d_model: Dimension of transformer embeddings (default 128)
            nhead: Number of attention heads (default 8)
            num_layers: Number of transformer encoder layers (default 4)
            dropout: Dropout probability (default 0.1)
            input_hours: Length of input sequence (default 168 = 7 days)
            output_hours: Length of output sequence (default 24 = 1 day)
            dim_feedforward: Dimension of feedforward network (default 512)
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_hours = input_hours
        self.output_hours = output_hours

        # Input projection: map input features to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding for temporal sequence
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max(input_hours, output_hours) + 100,
            dropout=dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Temporal aggregation: compress input sequence to output sequence length
        # Use a learnable query for each output time step
        self.output_queries = nn.Parameter(
            torch.randn(1, output_hours, d_model) * 0.02
        )

        # Cross-attention to attend from output queries to encoded sequence
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm after cross-attention
        self.norm = nn.LayerNorm(d_model)

        # Output projection: map d_model to target features
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights with Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            x: Input sequence, shape (batch, input_hours, input_size)
            src_mask: Optional attention mask for encoder

        Returns:
            Output predictions, shape (batch, output_hours, output_size)
        """
        batch_size = x.size(0)

        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch, input_hours, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Encode through transformer
        encoded = self.transformer_encoder(x, src_mask)  # (batch, input_hours, d_model)

        # Expand output queries for batch
        queries = self.output_queries.expand(batch_size, -1, -1)

        # Cross-attention: queries attend to encoded sequence
        attended, _ = self.cross_attention(
            query=queries,
            key=encoded,
            value=encoded,
        )  # (batch, output_hours, d_model)

        # Residual connection and layer norm
        attended = self.norm(queries + attended)

        # Project to output size
        output = self.output_projection(attended)  # (batch, output_hours, output_size)

        return output

    def get_attention_weights(
        self,
        x: torch.Tensor,
    ) -> tuple:
        """Get predictions and cross-attention weights for visualization.

        Args:
            x: Input sequence, shape (batch, input_hours, input_size)

        Returns:
            outputs: Predictions, shape (batch, output_hours, output_size)
            attention_weights: Cross-attention weights, shape (batch, output_hours, input_hours)
        """
        batch_size = x.size(0)

        # Project and encode
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        encoded = self.transformer_encoder(x)

        # Get attention weights
        queries = self.output_queries.expand(batch_size, -1, -1)
        attended, attn_weights = self.cross_attention(
            query=queries,
            key=encoded,
            value=encoded,
            average_attn_weights=True,
        )

        # Finish forward pass
        attended = self.norm(queries + attended)
        output = self.output_projection(attended)

        return output, attn_weights


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    input_size: int = 12,
    output_size: int = 3,
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 4,
    dropout: float = 0.1,
    device: Optional[str] = None,
) -> WeatherTransformer:
    """Create a WeatherTransformer model with automatic device detection.

    Args:
        input_size: Number of input features
        output_size: Number of output targets
        d_model: Transformer embedding dimension
        nhead: Number of attention heads
        num_layers: Number of encoder layers
        dropout: Dropout probability
        device: Device to place model on (auto-detected if None)

    Returns:
        WeatherTransformer model on the specified device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = WeatherTransformer(
        input_size=input_size,
        output_size=output_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    )

    return model.to(device)


if __name__ == "__main__":
    # Test the model
    print("Testing WeatherTransformer architecture...")

    # Create model with default parameters
    model = WeatherTransformer()
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    print(f"Configuration: d_model={model.d_model}, nhead={model.nhead}, "
          f"num_layers={model.num_layers}, dropout={model.dropout}")

    # Test forward pass with V3 input size (12 features)
    batch_size = 8
    x = torch.randn(batch_size, 168, 12)
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Verify output shape matches expected
    expected_shape = (batch_size, 24, 3)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"\nForward pass successful! Output shape: {output.shape}")

    # Test with attention weights
    outputs, attn_weights = model.get_attention_weights(x)
    print(f"Attention weights shape: {attn_weights.shape}")

    # Print model architecture summary
    print("\nModel Architecture:")
    print(f"  Input projection: {model.input_size} -> {model.d_model}")
    print(f"  Positional encoding: max_len={model.positional_encoding.pe.size(1)}")
    print(f"  Transformer encoder: {model.num_layers} layers, {model.nhead} heads")
    print(f"  Cross-attention: {model.output_hours} queries -> {model.input_hours} keys")
    print(f"  Output projection: {model.d_model} -> {model.output_size}")
