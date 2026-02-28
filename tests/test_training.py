"""
Tests for my_ml_crypto_trading.machine_learning.training
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from my_ml_crypto_trading.machine_learning.training import (
    proper_init_weights,
    train,
)


# ---------------------------------------------------------------------------
# Minimal model stubs
# ---------------------------------------------------------------------------


class _SmallMSEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)
        self.use_softmax = False
        self.loss_function = "mse"

    def forward(self, x):
        return self.fc(x)


class _SmallCEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.use_softmax = True
        self.loss_function = "ce"

    def forward(self, x):
        return self.fc(x)


def _make_loader(n=100, in_features=4, out_features=1, batch_size=16, classification=False):
    X = torch.randn(n, in_features)
    if classification:
        Y = torch.randint(0, 3, (n,))
    else:
        Y = torch.randn(n, out_features)
    return DataLoader(TensorDataset(X, Y), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProperInitWeights:

    def test_initializes_without_error(self):
        model = _SmallMSEModel()
        loader = _make_loader()
        device = torch.device("cpu")
        result = proper_init_weights(model, loader, device)
        assert result is not None

    def test_biases_are_zero(self):
        model = _SmallMSEModel()
        loader = _make_loader()
        device = torch.device("cpu")
        proper_init_weights(model, loader, device)
        for name, param in model.named_parameters():
            if "bias" in name:
                assert torch.allclose(param, torch.zeros_like(param))


class TestTrain:

    def test_mse_training_runs(self):
        model = _SmallMSEModel()
        loader_train = _make_loader(n=50)
        loader_val = _make_loader(n=20)
        trained = train(model, loader_train, loader_val, epochs=2, lr=0.01)
        assert trained is not None

    def test_ce_training_runs(self):
        model = _SmallCEModel()
        loader_train = _make_loader(n=50, out_features=1, classification=True)
        loader_val = _make_loader(n=20, out_features=1, classification=True)
        trained = train(model, loader_train, loader_val, epochs=2, lr=0.01)
        assert trained is not None

    def test_training_reduces_loss(self):
        """Training for a few epochs should generally reduce loss."""
        model = _SmallMSEModel()
        # Create a learnable pattern
        X = torch.randn(200, 4)
        Y = X[:, 0:1] * 2  # simple linear relationship
        loader_train = DataLoader(TensorDataset(X[:150], Y[:150]), batch_size=32)
        loader_val = DataLoader(TensorDataset(X[150:], Y[150:]), batch_size=32)

        trained = train(model, loader_train, loader_val, epochs=20, lr=0.01)

        # Check the trained model produces reasonable outputs
        with torch.no_grad():
            preds = trained(X[:10])
        # Should be somewhat correlated with target
        correlation = np.corrcoef(
            preds.numpy().flatten(), Y[:10].numpy().flatten()
        )[0, 1]
        assert correlation > 0.3  # loose check

    def test_training_with_metrics(self):
        from my_ml_crypto_trading.machine_learning.evaluation import trend_accuracy_metric
        model = _SmallMSEModel()
        loader_train = _make_loader(n=50)
        loader_val = _make_loader(n=20)
        trained = train(
            model, loader_train, loader_val,
            epochs=2, lr=0.01,
            metrics=[trend_accuracy_metric],
        )
        assert trained is not None

    def test_training_no_init(self):
        model = _SmallMSEModel()
        loader_train = _make_loader(n=50)
        loader_val = _make_loader(n=20)
        trained = train(
            model, loader_train, loader_val,
            epochs=1, lr=0.01, init_params=False,
        )
        assert trained is not None
