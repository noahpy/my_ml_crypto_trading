"""
Tests for my_ml_crypto_trading.data_processing.ob_features
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from my_ml_crypto_trading.data_processing.ob_features import (
    LevelOBFeature,
    MidPriceFeature,
    TrendFeature,
)
from my_ml_crypto_trading.data_processing.FeatureCreation import FeatureCreator
from tests.conftest import make_snapshot, make_snapshot_sequence


# ---------------------------------------------------------------------------
# LevelOBFeature
# ---------------------------------------------------------------------------


class TestLevelOBFeature:

    def test_min_timesteps_no_change(self):
        f = LevelOBFeature(num_levels=5, change=False)
        assert f.get_min_timesteps() == 1

    def test_min_timesteps_with_change(self):
        f = LevelOBFeature(num_levels=5, change=True)
        assert f.get_min_timesteps() == 2

    def test_feature_size_without_prices(self):
        f = LevelOBFeature(num_levels=10, include_prices=False)
        assert f.get_feature_size() == 20

    def test_feature_size_with_prices(self):
        f = LevelOBFeature(num_levels=10, include_prices=True)
        assert f.get_feature_size() == 40

    def test_create_feature_no_change(self):
        f = LevelOBFeature(num_levels=5, change=False)
        snap = make_snapshot(mid_price=100.0, num_levels=10)
        result = f.create_feature([snap])
        assert len(result) == f.get_feature_size()
        # All values should be non-negative (sizes)
        assert all(v >= 0 for v in result)

    def test_create_feature_with_change(self):
        f = LevelOBFeature(num_levels=5, change=True)
        snaps = make_snapshot_sequence(n=2, base_mid=100.0, num_levels=10)
        result = f.create_feature(snaps)
        assert len(result) == f.get_feature_size()

    def test_create_feature_with_prices(self):
        f = LevelOBFeature(num_levels=3, include_prices=True)
        snap = make_snapshot(mid_price=50.0, num_levels=5)
        result = f.create_feature([snap])
        assert len(result) == 12  # 3*4

    def test_visualize_no_crash(self):
        f = LevelOBFeature(num_levels=5)
        data = np.random.rand(20, f.get_feature_size())
        f.visualize_feature(data)
        plt.close("all")

    def test_visualize_with_ax_no_crash(self):
        f = LevelOBFeature(num_levels=5)
        data = np.random.rand(20, f.get_feature_size())
        fig, ax = plt.subplots()
        f.visualize_feature(data, ax=ax)
        plt.close("all")

    def test_toggle_subfeatures(self):
        f = LevelOBFeature()
        f.turn_all_subfeatures_on()
        f.turn_all_subfeatures_off()
        # Should not raise


# ---------------------------------------------------------------------------
# MidPriceFeature
# ---------------------------------------------------------------------------


class TestMidPriceFeature:

    def test_feature_size_only_mp(self):
        f = MidPriceFeature(inc_mp=True, inc_mp_change=False)
        assert f.get_feature_size() == 1

    def test_feature_size_only_change(self):
        f = MidPriceFeature(inc_mp=False, inc_mp_change=True, timesteps_back=1)
        assert f.get_feature_size() == 1

    def test_feature_size_both(self):
        f = MidPriceFeature(inc_mp=True, inc_mp_change=True)
        assert f.get_feature_size() == 2

    def test_feature_size_none(self):
        f = MidPriceFeature(inc_mp=False, inc_mp_change=False)
        assert f.get_feature_size() == 0

    def test_min_timesteps_no_change(self):
        f = MidPriceFeature(inc_mp=True, inc_mp_change=False)
        assert f.get_min_timesteps() == 1

    def test_min_timesteps_with_change(self):
        f = MidPriceFeature(inc_mp_change=True, timesteps_back=3)
        assert f.get_min_timesteps() == 4

    def test_create_feature_mp_only(self):
        f = MidPriceFeature(inc_mp=True, inc_mp_change=False)
        snap = make_snapshot(mid_price=42.0)
        result = f.create_feature([snap])
        assert result == [42.0]

    def test_create_feature_mp_change(self):
        f = MidPriceFeature(inc_mp=False, inc_mp_change=True, timesteps_back=1)
        s1 = make_snapshot(mid_price=100.0)
        s2 = make_snapshot(mid_price=102.0)
        result = f.create_feature([s1, s2])
        assert len(result) == 1
        assert abs(result[0] - 2.0) < 1e-9

    def test_create_feature_both(self):
        f = MidPriceFeature(inc_mp=True, inc_mp_change=True, timesteps_back=1)
        s1 = make_snapshot(mid_price=100.0)
        s2 = make_snapshot(mid_price=105.0)
        result = f.create_feature([s1, s2])
        assert len(result) == 2
        assert result[0] == 105.0
        assert abs(result[1] - 5.0) < 1e-9

    def test_toggle_functions(self):
        f = MidPriceFeature(inc_mp=False, inc_mp_change=False)
        f.toggle_mp()
        assert f.inc_mp is True
        f.toggle_mp_change()
        assert f.inc_mp_change is True
        f.toggle_mp()
        assert f.inc_mp is False

    def test_subfeature_names(self):
        f = MidPriceFeature()
        names = f.get_subfeature_names_and_toggle_function()
        assert len(names) == 2
        assert names[0][0] == "mid_price"
        assert names[1][0] == "mid_price_change"

    def test_turn_all_on_off(self):
        f = MidPriceFeature()
        f.turn_all_subfeatures_on()
        assert f.inc_mp is True
        assert f.inc_mp_change is True
        f.turn_all_subfeatures_off()
        assert f.inc_mp is False
        assert f.inc_mp_change is False

    def test_visualize_no_crash(self):
        f = MidPriceFeature(inc_mp=True, inc_mp_change=True, timesteps_back=1)
        data = np.random.rand(20, 2)
        f.visualize_feature(data)
        plt.close("all")

    def test_visualize_single_feature_no_crash(self):
        f = MidPriceFeature(inc_mp=True, inc_mp_change=False)
        data = np.random.rand(20, 1)
        f.visualize_feature(data)
        plt.close("all")


# ---------------------------------------------------------------------------
# TrendFeature
# ---------------------------------------------------------------------------


class TestTrendFeature:

    def test_min_timesteps(self):
        f = TrendFeature(timesteps_back=3)
        assert f.get_min_timesteps() == 4

    def test_feature_size(self):
        f = TrendFeature()
        assert f.get_feature_size() == 3

    def test_trend_up(self):
        f = TrendFeature(timesteps_back=1)
        s1 = make_snapshot(mid_price=100.0)
        s2 = make_snapshot(mid_price=102.0)
        result = f.create_feature([s1, s2])
        assert result == [1, 0, 0]

    def test_trend_equal(self):
        f = TrendFeature(timesteps_back=1)
        s1 = make_snapshot(mid_price=100.0)
        s2 = make_snapshot(mid_price=100.0)
        result = f.create_feature([s1, s2])
        assert result == [0, 1, 0]

    def test_trend_down(self):
        f = TrendFeature(timesteps_back=1)
        s1 = make_snapshot(mid_price=102.0)
        s2 = make_snapshot(mid_price=100.0)
        result = f.create_feature([s1, s2])
        assert result == [0, 0, 1]

    def test_visualize_no_crash(self):
        f = TrendFeature()
        data = np.random.rand(20, 3)
        f.visualize_feature(data)
        plt.close("all")

    def test_visualize_with_ax(self):
        f = TrendFeature()
        data = np.random.rand(20, 3)
        fig, ax = plt.subplots()
        f.visualize_feature(data, ax=ax)
        plt.close("all")


# ---------------------------------------------------------------------------
# Integration: OB Features with FeatureCreator
# ---------------------------------------------------------------------------


class TestOBFeaturesIntegration:

    def test_level_ob_in_creator_pipeline(self):
        f = LevelOBFeature(num_levels=5, change=False)
        fc = FeatureCreator(features=[f])
        snaps = make_snapshot_sequence(n=3, num_levels=10)
        for s in snaps:
            fc.feed_datapoint(s)
        assert fc.is_ready()
        result = fc.create_features()
        assert len(result) == f.get_feature_size()

    def test_midprice_in_creator_pipeline(self):
        f = MidPriceFeature(inc_mp=True, inc_mp_change=True, timesteps_back=1)
        fc = FeatureCreator(features=[f])
        snaps = make_snapshot_sequence(n=5)
        for s in snaps:
            fc.feed_datapoint(s)
        assert fc.is_ready()
        result = fc.create_features()
        assert len(result) == 2

    def test_combined_features_in_pipeline(self):
        features = [
            LevelOBFeature(num_levels=5),
            MidPriceFeature(inc_mp=True, inc_mp_change=True, timesteps_back=1),
            TrendFeature(timesteps_back=1),
        ]
        fc = FeatureCreator(features=features)
        snaps = make_snapshot_sequence(n=5)
        for s in snaps:
            fc.feed_datapoint(s)
        assert fc.is_ready()
        result = fc.create_features()
        expected_len = sum(f.get_feature_size() for f in features)
        assert len(result) == expected_len
