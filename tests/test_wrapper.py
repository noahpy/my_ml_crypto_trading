"""
Tests for my_ml_crypto_trading.machine_learning.wrapper
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import pickle

from my_ml_crypto_trading.machine_learning.wrapper import PyTorchWrapper
from my_ml_crypto_trading.data_processing.FeatureCreation import FeatureCreator
from my_ml_crypto_trading.data_processing.ob_features import MidPriceFeature
from tests.conftest import make_snapshot, make_snapshot_sequence


# ---------------------------------------------------------------------------
# Helper: create a model + feature_creator and save to disk
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self, in_features=3):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)
        self.use_softmax = False
        self.loss_function = "mse"

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)
        # Use a lazy approach: just take the first `in_features` columns
        return self.fc(flat[:, : self.fc.in_features])


def _create_wrapper_files(tmpdir, input_length=3, feature_size=1):
    """Create model.pth and feature_creator.pkl in tmpdir."""
    fc = FeatureCreator(features=[MidPriceFeature(inc_mp=True)])

    model = _TinyModel(in_features=input_length * feature_size)
    model.eval()
    torch.save(model, os.path.join(tmpdir, "model.pth"))

    with open(os.path.join(tmpdir, "feature_creator.pkl"), "wb") as f:
        pickle.dump(fc, f)

    return fc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPyTorchWrapper:

    def test_init_loads_model_and_fc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_wrapper_files(tmpdir, input_length=3)
            wrapper = PyTorchWrapper(tmpdir, input_length=3)
            assert wrapper.model is not None
            assert wrapper.feature_creator is not None
            assert wrapper.input_length == 3

    def test_feed_snapshot_returns_false_until_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_wrapper_files(tmpdir, input_length=3)
            wrapper = PyTorchWrapper(tmpdir, input_length=3)

            snaps = make_snapshot_sequence(n=5)
            results = []
            for s in snaps[:2]:
                results.append(wrapper.feed_snapshot(s))
            # First 2 calls shouldn't reach input_length=3 yet
            # (depends on feature_creator buffer, but first call should produce a feature)
            # With MidPriceFeature(min_ts=1), each snapshot produces a feature
            # So after 2 snapshots, we have 2 features < 3 needed
            assert results[-1] is False

    def test_feed_snapshot_returns_true_when_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_wrapper_files(tmpdir, input_length=3)
            wrapper = PyTorchWrapper(tmpdir, input_length=3)

            snaps = make_snapshot_sequence(n=5)
            for s in snaps[:3]:
                ready = wrapper.feed_snapshot(s)
            assert ready is True

    def test_predict_returns_float(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_wrapper_files(tmpdir, input_length=3)
            wrapper = PyTorchWrapper(tmpdir, input_length=3)

            snaps = make_snapshot_sequence(n=10)
            # Feed enough snapshots to get ready
            for s in snaps[:3]:
                wrapper.feed_snapshot(s)

            # Now predict with the next snapshot
            result = wrapper.predict(snaps[3])
            assert isinstance(result, float)

    def test_predict_not_ready_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_wrapper_files(tmpdir, input_length=10)
            wrapper = PyTorchWrapper(tmpdir, input_length=10)

            snap = make_snapshot()
            result = wrapper.predict(snap)
            assert result is None

    def test_feature_data_truncated_to_input_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_wrapper_files(tmpdir, input_length=3)
            wrapper = PyTorchWrapper(tmpdir, input_length=3)

            snaps = make_snapshot_sequence(n=10)
            for s in snaps:
                wrapper.feed_snapshot(s)

            assert len(wrapper.feature_data) == 3
