"""Neural network architecture for weather forecasting.

Implements an LSTM encoder-decoder with attention mechanism for
sequence-to-sequence weather prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from src.config import (
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    SEQUENCE_INPUT_HOURS,
    SEQUENCE_OUTPUT_HOURS,
)


class Attention(nn.Module):
    """Bahdanau-style attention mechanism for sequence-to-sequence models."""

    def __init__(self, hidden_size: int):
        """Initialize attention layer.

        Args:
            hidden_size: Size of hidden states
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Attention weights
        self.W_encoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_decoder = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and context vector.

        Args:
            decoder_hidden: Current decoder hidden state, shape (batch, hidden_size)
            encoder_outputs: All encoder outputs, shape (batch, seq_len, hidden_size)

        Returns:
            context: Context vector, shape (batch, hidden_size)
            attention_weights: Attention weights, shape (batch, seq_len)
        """
        batch_size, seq_len, _ = encoder_outputs.shape

        # Expand decoder hidden for attention calculation
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)

        # Compute attention scores
        # score = v * tanh(W_encoder * encoder_output + W_decoder * decoder_hidden)
        energy = torch.tanh(
            self.W_encoder(encoder_outputs) + self.W_decoder(decoder_hidden_expanded)
        )
        attention_scores = self.v(energy).squeeze(-1)  # (batch, seq_len)

        # Normalize to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)

        # Compute context vector as weighted sum of encoder outputs
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights


class WeatherNet(nn.Module):
    """LSTM encoder-decoder with attention for weather forecasting.

    Architecture:
        1. LSTM Encoder: Processes input sequence (72 hours)
        2. Attention: Computes context from encoder outputs
        3. LSTM Decoder: Generates output sequence (24 hours)

    Input: (batch, 72 hours, N features) where N=6 weather features
    Output: (batch, 24 hours, M targets) where M=3 target features (temp, precip, wind)
    """

    def __init__(
        self,
        input_size: int = 6,
        output_size: int = 3,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
        input_hours: int = SEQUENCE_INPUT_HOURS,
        output_hours: int = SEQUENCE_OUTPUT_HOURS,
    ):
        """Initialize WeatherNet.

        Args:
            input_size: Number of input features (default 6)
            output_size: Number of output targets (default 3)
            hidden_size: Size of LSTM hidden states (default from config)
            num_layers: Number of LSTM layers (default from config)
            dropout: Dropout probability (default from config)
            input_hours: Length of input sequence (default 72)
            output_hours: Length of output sequence (default 24)
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_hours = input_hours
        self.output_hours = output_hours

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Attention mechanism
        self.attention = Attention(hidden_size)

        # Decoder LSTM
        # Input: previous output (output_size) + context (hidden_size)
        self.decoder = nn.LSTM(
            input_size=output_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_size, output_size)

        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input sequence, shape (batch, input_hours, input_size)
            teacher_forcing_ratio: Probability of using ground truth for decoder input
            target: Ground truth output for teacher forcing, shape (batch, output_hours, output_size)

        Returns:
            Output predictions, shape (batch, output_hours, output_size)
        """
        batch_size = x.size(0)
        device = x.device

        # Encode input sequence
        encoder_outputs, (hidden, cell) = self.encoder(x)
        # encoder_outputs: (batch, input_hours, hidden_size)
        # hidden, cell: (num_layers, batch, hidden_size)

        # Apply dropout to encoder outputs
        encoder_outputs = self.dropout_layer(encoder_outputs)

        # Initialize decoder input with zeros (start token equivalent)
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)

        # Store outputs
        outputs = []

        # Decode step by step
        for t in range(self.output_hours):
            # Get attention context
            # Use top layer hidden state for attention
            decoder_hidden_for_attn = hidden[-1]  # (batch, hidden_size)
            context, _ = self.attention(decoder_hidden_for_attn, encoder_outputs)

            # Concatenate decoder input with context
            decoder_input_with_context = torch.cat(
                [decoder_input, context.unsqueeze(1)], dim=-1
            )  # (batch, 1, output_size + hidden_size)

            # Decode one step
            decoder_output, (hidden, cell) = self.decoder(
                decoder_input_with_context, (hidden, cell)
            )
            # decoder_output: (batch, 1, hidden_size)

            # Project to output size
            output = self.output_projection(decoder_output)  # (batch, 1, output_size)
            outputs.append(output)

            # Determine next decoder input
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                decoder_input = target[:, t : t + 1, :]
            else:
                # Use model prediction
                decoder_input = output

        # Concatenate all outputs
        outputs = torch.cat(outputs, dim=1)  # (batch, output_hours, output_size)

        return outputs

    def get_attention_weights(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions and attention weights for visualization.

        Args:
            x: Input sequence, shape (batch, input_hours, input_size)

        Returns:
            outputs: Predictions, shape (batch, output_hours, output_size)
            attention_weights: Attention weights for each output step,
                             shape (batch, output_hours, input_hours)
        """
        batch_size = x.size(0)
        device = x.device

        # Encode
        encoder_outputs, (hidden, cell) = self.encoder(x)

        # Initialize decoder
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=device)

        outputs = []
        attention_weights_list = []

        for t in range(self.output_hours):
            decoder_hidden_for_attn = hidden[-1]
            context, attn_weights = self.attention(decoder_hidden_for_attn, encoder_outputs)
            attention_weights_list.append(attn_weights)

            decoder_input_with_context = torch.cat(
                [decoder_input, context.unsqueeze(1)], dim=-1
            )

            decoder_output, (hidden, cell) = self.decoder(
                decoder_input_with_context, (hidden, cell)
            )

            output = self.output_projection(decoder_output)
            outputs.append(output)
            decoder_input = output

        outputs = torch.cat(outputs, dim=1)
        attention_weights = torch.stack(attention_weights_list, dim=1)

        return outputs, attention_weights


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing WeatherNet architecture...")

    # Create model
    model = WeatherNet()
    print(f"Model created with {count_parameters(model):,} trainable parameters")

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 72, 6)
    output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Verify output shape matches expected
    expected_shape = (batch_size, 24, 3)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    print(f"\n✓ Forward pass successful! Output shape: {output.shape}")

    # Test with attention weights
    outputs, attn_weights = model.get_attention_weights(x)
    print(f"Attention weights shape: {attn_weights.shape}")
