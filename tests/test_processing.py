"""
Tests for my_ml_crypto_trading.machine_learning.processing
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import os
import tempfile
import pickle

from my_ml_crypto_trading.machine_learning.processing import (
    NormalizedModel,
    build_mp_change_prediction_data_set,
    store_model,
    load_model,
)
from my_ml_crypto_trading.data_processing.FeatureCreation import FeatureCreator
from my_ml_crypto_trading.data_processing.ob_features import MidPriceFeature
from tests.conftest import make_snapshot_sequence


# ---------------------------------------------------------------------------
# Minimal model for testing
# ---------------------------------------------------------------------------


class _DummyModel(nn.Module):
    def __init__(self, in_features=4, out_features=1):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.use_softmax = False
        self.loss_function = "mse"

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------------------------
# NormalizedModel
# ---------------------------------------------------------------------------


class TestNormalizedModel:

    def test_forward_normalizes_input(self):
        base = _DummyModel(in_features=4, out_features=1)
        x_train = torch.randn(100, 4)
        y_train = torch.randn(100, 1)

        nm = NormalizedModel(base, x_train, y_train)
        x = torch.randn(5, 4)
        out = nm(x)
        assert out.shape == (5, 1)

    def test_norm_flags(self):
        base = _DummyModel(4, 1)
        x_train = torch.randn(50, 4)
        y_train = torch.randn(50, 1)

        nm_no_norm = NormalizedModel(
            base, x_train, y_train,
            norm_input_std=False, norm_input_mean=False,
            norm_output_std=False, norm_output_mean=False,
        )
        # input_mean should be 0, input_std should be 1
        assert nm_no_norm.input_mean == 0
        assert nm_no_norm.input_std == 1

    def test_inherits_loss_function(self):
        base = _DummyModel()
        x_train = torch.randn(10, 4)
        y_train = torch.randn(10, 1)
        nm = NormalizedModel(base, x_train, y_train)
        assert nm.loss_function == "mse"
        assert nm.use_softmax is False


# ---------------------------------------------------------------------------
# build_mp_change_prediction_data_set
# ---------------------------------------------------------------------------


class TestBuildMPChangePredictionDataSet:

    def _make_feature_data(self, n=50):
        """Create fake feature data with a mid_price column."""
        f = MidPriceFeature(inc_mp=True, inc_mp_change=False)
        fc = FeatureCreator(features=[f])
        snaps = make_snapshot_sequence(n=n)
        rows = []
        for s in snaps:
            fc.feed_datapoint(s)
            if fc.is_ready():
                rows.append(fc.create_features())
        return np.array(rows), fc

    def test_basic_shape(self):
        features, fc = self._make_feature_data(n=30)
        X, Y = build_mp_change_prediction_data_set(
            features, fc, mp_f_index=0, horizon=2,
            input_length=5, steps_between=1,
        )
        assert X.ndim == 2 or X.ndim == 3  # depends on features shape
        assert len(X) == len(Y)
        assert len(X) > 0

    def test_input_length_correct(self):
        features, fc = self._make_feature_data(n=30)
        X, Y = build_mp_change_prediction_data_set(
            features, fc, mp_f_index=0, horizon=1,
            input_length=5, steps_between=1,
        )
        # Each X sample should have input_length time steps
        assert X.shape[1] == 5

    def test_y_is_price_diff(self):
        features, fc = self._make_feature_data(n=20)
        mp_data = fc.get_feature(features, 0)
        X, Y = build_mp_change_prediction_data_set(
            features, fc, mp_f_index=0, horizon=1,
            input_length=3, steps_between=1,
        )
        # Y[0] should be mp[3+1-1] - mp[3-1] = mp[3] - mp[2]
        expected = mp_data[3][0] - mp_data[2][0]
        np.testing.assert_allclose(Y[0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# store_model / load_model
# ---------------------------------------------------------------------------


class TestStoreLoadModel:

    def test_store_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                model = _DummyModel(4, 1)
                fc = FeatureCreator(features=[MidPriceFeature(inc_mp=True)])

                store_model(
                    coin="TEST",
                    time_delta_ms=1000,
                    window_length=10,
                    target="mp_change",
                    model_name="test_model",
                    model=model,
                    input_feature_creator=fc,
                    description="This is a test model description for roundtrip testing.",
                )

                loaded_model, loaded_fc = load_model(
                    coin="TEST",
                    time_delta_ms=1000,
                    window_length=10,
                    target="mp_change",
                    model_name="test_model",
                )

                assert loaded_model is not None
                assert loaded_fc is not None
                # Check model works
                x = torch.randn(1, 4)
                out = loaded_model(x)
                assert out.shape == (1, 1)
            finally:
                os.chdir(original_cwd)

    def test_store_rejects_short_description(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                model = _DummyModel(4, 1)
                fc = FeatureCreator(features=[MidPriceFeature(inc_mp=True)])

                store_model(
                    coin="TEST",
                    time_delta_ms=1000,
                    window_length=10,
                    target="mp_change",
                    model_name="test_model",
                    model=model,
                    input_feature_creator=fc,
                    description="too short",
                )
                captured = capsys.readouterr()
                assert "longer description" in captured.out
            finally:
                os.chdir(original_cwd)
