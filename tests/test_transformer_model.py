"""
Tests for my_ml_crypto_trading.machine_learning.transformer_model.model
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from my_ml_crypto_trading.machine_learning.transformer_model.model import (
    SimpleOrderBookModel,
    PositionalEncoding,
    GaussianNLLLoss,
    BasicTransformerModel,
)


# ---------------------------------------------------------------------------
# SimpleOrderBookModel
# ---------------------------------------------------------------------------


class TestSimpleOrderBookModel:

    def test_output_shape_scalar(self):
        model = SimpleOrderBookModel(hidden_dim=32, output_dim=1, use_softmax=False)
        x = torch.randn(4, 10, 20)  # batch=4, timesteps=10, features=20
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_shape_multi_dim(self):
        model = SimpleOrderBookModel(hidden_dim=32, output_dim=3, use_softmax=False)
        x = torch.randn(4, 10, 20)
        out = model(x)
        assert out.shape == (4, 3)

    def test_softmax_output(self):
        model = SimpleOrderBookModel(hidden_dim=32, output_dim=3, use_softmax=True)
        x = torch.randn(4, 10, 20)
        out = model(x)
        # Softmax outputs should sum to ~1
        sums = out.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_accepts_numpy_input(self):
        model = SimpleOrderBookModel(hidden_dim=16, output_dim=1)
        x = np.random.randn(2, 5, 10).astype(np.float32)
        out = model(x)
        assert out.shape == (2, 1)

    def test_handles_4d_input(self):
        model = SimpleOrderBookModel(hidden_dim=16, output_dim=1)
        x = torch.randn(2, 5, 4, 3)  # batch, time, rows, cols
        out = model(x)
        assert out.shape == (2, 1)


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------


class TestPositionalEncoding:

    def test_output_shape_matches_input(self):
        pe = PositionalEncoding(d_model=32)
        x = torch.randn(2, 10, 32)
        out = pe(x)
        assert out.shape == x.shape

    def test_encoding_is_additive(self):
        pe = PositionalEncoding(d_model=16)
        x = torch.zeros(1, 5, 16)
        out = pe(x)
        # Output should be just the positional encoding since input is zeros
        assert out.abs().sum() > 0

    def test_different_positions_get_different_encodings(self):
        pe = PositionalEncoding(d_model=16)
        x = torch.zeros(1, 10, 16)
        out = pe(x)
        # Position 0 and position 1 should differ
        assert not torch.allclose(out[0, 0], out[0, 1])


# ---------------------------------------------------------------------------
# GaussianNLLLoss
# ---------------------------------------------------------------------------


class TestGaussianNLLLoss:

    def test_loss_is_scalar(self):
        loss_fn = GaussianNLLLoss()
        y_pred = torch.tensor([[1.0, 0.5], [2.0, 0.8]])  # [mean, var]
        y_true = torch.tensor([1.0, 2.0])
        loss = loss_fn(y_pred, y_true)
        assert loss.dim() == 0  # scalar

    def test_perfect_prediction_low_loss(self):
        loss_fn = GaussianNLLLoss()
        y_pred = torch.tensor([[5.0, 0.01]])  # mean=5, var=0.01
        y_true = torch.tensor([5.0])
        loss_perfect = loss_fn(y_pred, y_true)

        y_pred_bad = torch.tensor([[10.0, 0.01]])  # mean=10, var=0.01
        loss_bad = loss_fn(y_pred_bad, y_true)

        assert loss_perfect < loss_bad

    def test_variance_clamped(self):
        loss_fn = GaussianNLLLoss(eps=1e-6)
        y_pred = torch.tensor([[1.0, -10.0]])  # negative variance
        y_true = torch.tensor([1.0])
        loss = loss_fn(y_pred, y_true)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ---------------------------------------------------------------------------
# BasicTransformerModel
# ---------------------------------------------------------------------------


class TestBasicTransformerModel:

    def test_output_shape(self):
        model = BasicTransformerModel(d_model=32, nhead=4, num_layers=1, output_dim=1)
        x = torch.randn(4, 10, 20)
        out = model(x)
        assert out.shape == (4, 1)

    def test_output_shape_multi_dim(self):
        model = BasicTransformerModel(d_model=32, nhead=4, num_layers=1, output_dim=3)
        x = torch.randn(4, 10, 20)
        out = model(x)
        assert out.shape == (4, 3)

    def test_softmax_output(self):
        model = BasicTransformerModel(
            d_model=32, nhead=4, num_layers=1, output_dim=3, use_softmax=True
        )
        x = torch.randn(4, 10, 20)
        out = model(x)
        sums = out.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self):
        model = BasicTransformerModel(d_model=16, nhead=2, num_layers=1, output_dim=1)
        x = torch.randn(2, 5, 8, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that gradients exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_handles_4d_input(self):
        model = BasicTransformerModel(d_model=16, nhead=2, num_layers=1, output_dim=1)
        x = torch.randn(2, 5, 4, 3)  # will be reshaped to [2, 5, 12]
        out = model(x)
        assert out.shape == (2, 1)

    def test_accepts_non_tensor_input(self):
        model = BasicTransformerModel(d_model=16, nhead=2, num_layers=1, output_dim=1)
        x = np.random.randn(2, 5, 8).astype(np.float32)
        out = model(x)
        assert out.shape == (2, 1)
