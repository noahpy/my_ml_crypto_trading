"""
Tests for my_ml_crypto_trading.data_processing.FeatureCreation
"""

import pytest
import numpy as np
from typing import List
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests
import matplotlib.pyplot as plt

from my_ml_crypto_trading.data_processing.FeatureCreation import Feature, FeatureCreator
from tests.conftest import make_snapshot, make_snapshot_sequence


# ---------------------------------------------------------------------------
# Concrete stub Feature for testing the FeatureCreator pipeline
# ---------------------------------------------------------------------------

class StubFeature(Feature):
    """Trivial feature that returns the mid_price from the last snapshot."""

    def __init__(self, min_ts=1):
        self._min_ts = min_ts

    def get_min_timesteps(self) -> int:
        return self._min_ts

    def create_feature(self, buffer: List[dict]) -> List[float]:
        return [buffer[-1]["mid_price"]]

    def get_feature_size(self) -> int:
        return 1

    def visualize_feature(self, features: np.ndarray, ax=None) -> None:
        pass

    def get_subfeature_names_and_toggle_function(self) -> List[List]:
        return []

    def turn_all_subfeatures_on(self):
        pass

    def turn_all_subfeatures_off(self):
        pass


class StubFeature2(Feature):
    """Returns [mid_price, mid_price*2] – size 2."""

    def get_min_timesteps(self) -> int:
        return 1

    def create_feature(self, buffer: List[dict]) -> List[float]:
        mp = buffer[-1]["mid_price"]
        return [mp, mp * 2]

    def get_feature_size(self) -> int:
        return 2

    def visualize_feature(self, features: np.ndarray, ax=None) -> None:
        pass

    def get_subfeature_names_and_toggle_function(self) -> List[List]:
        return []

    def turn_all_subfeatures_on(self):
        pass

    def turn_all_subfeatures_off(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatureCreator:

    def test_buffer_length_determined_by_features(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=3), StubFeature(min_ts=5)])
        assert fc.buffer_len == 5

    def test_buffer_length_minimum_one(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=1)])
        assert fc.buffer_len == 1

    def test_is_ready_false_before_enough_data(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=3)])
        fc.feed_datapoint(make_snapshot())
        assert not fc.is_ready()

    def test_is_ready_true_after_enough_data(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=2)])
        fc.feed_datapoint(make_snapshot())
        fc.feed_datapoint(make_snapshot())
        assert fc.is_ready()

    def test_create_features_returns_correct_vector(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=1)])
        snap = make_snapshot(mid_price=42.0)
        fc.feed_datapoint(snap)
        result = fc.create_features()
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, [42.0])

    def test_create_features_concatenates_multiple_features(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=1), StubFeature2()])
        snap = make_snapshot(mid_price=10.0)
        fc.feed_datapoint(snap)
        result = fc.create_features()
        np.testing.assert_allclose(result, [10.0, 10.0, 20.0])

    def test_create_features_raises_when_not_ready(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=5)])
        fc.feed_datapoint(make_snapshot())
        with pytest.raises(Exception, match="Not enough data"):
            fc.create_features()

    def test_buffer_is_truncated_to_max_needed(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=3)])
        for _ in range(10):
            fc.feed_datapoint(make_snapshot())
        assert len(fc.buffer) == 3

    def test_get_feature_extracts_correct_slice(self):
        f1 = StubFeature(min_ts=1)  # size 1
        f2 = StubFeature2()  # size 2
        fc = FeatureCreator(features=[f1, f2])

        # Build a small features array (2 time steps)
        rows = []
        for mp in [5.0, 10.0]:
            snap = make_snapshot(mid_price=mp)
            fc.feed_datapoint(snap)
            rows.append(fc.create_features())
        data = np.array(rows)

        # feature index 0 → StubFeature (col 0)
        f0_data = fc.get_feature(data, 0)
        np.testing.assert_allclose(f0_data, data[:, 0:1])

        # feature index 1 → StubFeature2 (cols 1-2)
        f1_data = fc.get_feature(data, 1)
        np.testing.assert_allclose(f1_data, data[:, 1:3])

    def test_visualize_does_not_crash(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=1)])
        rows = []
        for mp in [1.0, 2.0, 3.0]:
            snap = make_snapshot(mid_price=mp)
            fc.feed_datapoint(snap)
            rows.append(fc.create_features())
        data = np.array(rows)
        # Should not raise
        fc.visualize(data)
        plt.close("all")

    def test_visualize_at_does_not_crash(self):
        fc = FeatureCreator(features=[StubFeature(min_ts=1), StubFeature2()])
        rows = []
        for mp in [1.0, 2.0, 3.0]:
            snap = make_snapshot(mid_price=mp)
            fc.feed_datapoint(snap)
            rows.append(fc.create_features())
        data = np.array(rows)
        fc.visualize_at(data, 0)
        fc.visualize_at(data, 1)
        plt.close("all")
