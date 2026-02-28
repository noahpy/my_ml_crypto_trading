"""
Tests for my_ml_crypto_trading.machine_learning.transformer_model.embedding
"""

import pytest
import torch

from my_ml_crypto_trading.machine_learning.transformer_model.embedding import PositionalEncoding


class TestEmbeddingPositionalEncoding:

    def test_output_shape(self):
        pe = PositionalEncoding(d_model=16, max_len=20)
        x = torch.randn(2, 10, 16)
        out = pe(x)
        assert out.shape == (2, 10, 16)

    def test_adds_positional_info(self):
        pe = PositionalEncoding(d_model=16, max_len=20)
        x = torch.zeros(1, 10, 16)
        out = pe(x)
        # Output should differ from input (positional encodings added)
        assert out.abs().sum() > 0

    def test_different_positions_differ(self):
        pe = PositionalEncoding(d_model=16, max_len=20)
        x = torch.zeros(1, 10, 16)
        out = pe(x)
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_shorter_sequence_works(self):
        pe = PositionalEncoding(d_model=8, max_len=50)
        x = torch.randn(3, 5, 8)  # sequence shorter than max
        out = pe(x)
        assert out.shape == (3, 5, 8)
