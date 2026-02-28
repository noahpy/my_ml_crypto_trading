"""
Tests for my_ml_crypto_trading.data_processing.trade_features
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from my_ml_crypto_trading.data_processing.trade_features import (
    TradeFeature,
    VolatilityFeature,
    create_feature_subplot,
)
from my_ml_crypto_trading.data_processing.FeatureCreation import FeatureCreator
from tests.conftest import make_snapshot, make_snapshot_sequence, make_trade


# ---------------------------------------------------------------------------
# TradeFeature
# ---------------------------------------------------------------------------


class TestTradeFeature:

    def test_min_timesteps_no_change(self):
        f = TradeFeature(timesteps_back=3, inc_vwap=True)
        assert f.get_min_timesteps() == 3

    def test_min_timesteps_with_change(self):
        f = TradeFeature(timesteps_back=3, inc_vwap_change=True)
        assert f.get_min_timesteps() == 4

    def test_feature_size_vwap_only(self):
        f = TradeFeature(inc_vwap=True)
        assert f.get_feature_size() == 1

    def test_feature_size_all_on(self):
        f = TradeFeature(
            inc_vwap=True,
            inc_vwap_change=True,
            inc_vol=True,
            inc_vol_change=True,
            inc_taker_vol=True,
            inc_taker_vol_change=True,
        )
        # vwap(1) + vwap_change(1) + vol(1) + vol_change(1) + taker(2) + taker_change(2) = 8
        assert f.get_feature_size() == 8

    def test_feature_size_none(self):
        f = TradeFeature()
        assert f.get_feature_size() == 0

    def test_create_feature_vwap(self):
        f = TradeFeature(timesteps_back=1, inc_vwap=True)
        # TradeFeature iterates over entire buffer; needs at least 2 snapshots
        # for past/current calculations even without _change flags
        snap1 = make_snapshot(
            mid_price=100.0,
            trades=[make_trade("Buy", 1.0, 100.0)],
        )
        snap2 = make_snapshot(
            mid_price=100.0,
            trades=[
                make_trade("Buy", 2.0, 100.0),
                make_trade("Sell", 3.0, 101.0),
            ],
        )
        result = f.create_feature([snap1, snap2])
        assert len(result) == 1
        # VWAP for last timesteps_back=1 snap: (2*100 + 3*101) / 5 = 100.6
        assert abs(result[0] - 100.6) < 1e-6

    def test_create_feature_vol(self):
        f = TradeFeature(timesteps_back=1, inc_vol=True)
        snap1 = make_snapshot(
            mid_price=100.0,
            trades=[make_trade("Buy", 1.0, 99.0)],
        )
        snap2 = make_snapshot(
            mid_price=100.0,
            trades=[
                make_trade("Buy", 2.0, 100.0),
                make_trade("Sell", 3.0, 101.0),
            ],
        )
        result = f.create_feature([snap1, snap2])
        assert len(result) == 1
        assert abs(result[0] - 5.0) < 1e-6

    def test_create_feature_taker_vols(self):
        f = TradeFeature(timesteps_back=1, inc_taker_vol=True)
        snap1 = make_snapshot(
            mid_price=100.0,
            trades=[make_trade("Buy", 1.0, 99.0)],
        )
        snap2 = make_snapshot(
            mid_price=100.0,
            trades=[
                make_trade("Buy", 2.0, 100.0),
                make_trade("Sell", 3.0, 101.0),
            ],
        )
        result = f.create_feature([snap1, snap2])
        assert len(result) == 2
        assert abs(result[0] - 2.0) < 1e-6  # bid takers
        assert abs(result[1] - 3.0) < 1e-6  # ask takers

    def test_create_feature_no_trades_uses_midprice(self):
        f = TradeFeature(timesteps_back=1, inc_vwap=True)
        snap1 = make_snapshot(mid_price=50.0, trades=[])
        snap2 = make_snapshot(mid_price=50.0, trades=[])
        result = f.create_feature([snap1, snap2])
        assert len(result) == 1
        # When no trades, VWAP falls back to mid_price
        assert abs(result[0] - 50.0) < 0.5

    def test_create_feature_vwap_change(self):
        f = TradeFeature(timesteps_back=1, inc_vwap_change=True)
        s1 = make_snapshot(
            mid_price=100.0,
            trades=[make_trade("Buy", 1.0, 100.0)],
        )
        s2 = make_snapshot(
            mid_price=102.0,
            trades=[make_trade("Buy", 1.0, 102.0)],
        )
        result = f.create_feature([s1, s2])
        assert len(result) == 1
        assert abs(result[0] - 2.0) < 1e-6

    def test_toggle_functions(self):
        f = TradeFeature()
        f.toggle_vwap()
        assert f.inc_vwap is True
        f.toggle_vwap()
        assert f.inc_vwap is False

        f.toggle_vol()
        assert f.inc_vol is True

        f.toggle_taker()
        assert f.inc_taker is True

    def test_turn_all_on_off(self):
        f = TradeFeature()
        f.turn_all_subfeatures_on()
        assert f.inc_vwap is True
        assert f.inc_vol is True
        assert f.inc_taker is True
        f.turn_all_subfeatures_off()
        assert f.inc_vwap is False
        assert f.inc_vol is False
        assert f.inc_taker is False

    def test_subfeature_names(self):
        f = TradeFeature()
        names = f.get_subfeature_names_and_toggle_function()
        assert len(names) == 6

    def test_visualize_no_crash(self):
        f = TradeFeature(inc_vwap=True, inc_vol=True, inc_taker_vol=True)
        data = np.random.rand(20, f.get_feature_size())
        f.visualize_feature(data)
        plt.close("all")

    def test_visualize_no_features_selected(self):
        f = TradeFeature()
        data = np.random.rand(20, 0)
        f.visualize_feature(data)
        plt.close("all")


# ---------------------------------------------------------------------------
# VolatilityFeature
# ---------------------------------------------------------------------------


class TestVolatilityFeature:

    def test_min_timesteps(self):
        f = VolatilityFeature(timesteps_back=10)
        assert f.get_min_timesteps() == 10

    def test_feature_size(self):
        f = VolatilityFeature()
        assert f.get_feature_size() == 1

    def test_create_feature_returns_scalar(self):
        f = VolatilityFeature(timesteps_back=3)
        snaps = make_snapshot_sequence(n=3, include_trades=True)
        result = f.create_feature(snaps)
        assert len(result) == 1
        assert result[0] >= 0  # volatility is non-negative

    def test_create_feature_constant_price_zero_vol(self):
        f = VolatilityFeature(timesteps_back=3)
        # All snapshots with same price and no trades → uses mid_price → std = 0
        snaps = [make_snapshot(mid_price=100.0, trades=[]) for _ in range(3)]
        result = f.create_feature(snaps)
        assert abs(result[0]) < 1e-9

    def test_visualize_no_crash(self):
        f = VolatilityFeature()
        data = np.random.rand(20, 1)
        f.visualize_feature(data)
        plt.close("all")

    def test_visualize_with_ax(self):
        f = VolatilityFeature()
        data = np.random.rand(20, 1)
        fig, ax = plt.subplots()
        f.visualize_feature(data, ax=ax)
        plt.close("all")


# ---------------------------------------------------------------------------
# create_feature_subplot helper
# ---------------------------------------------------------------------------


class TestCreateFeatureSubplot:

    def test_basic_subplot(self):
        fig, ax = plt.subplots()
        data = np.random.rand(50)
        create_feature_subplot(ax, data, "Test", "blue")
        plt.close("all")

    def test_subplot_with_second_data(self):
        fig, ax = plt.subplots()
        data1 = np.random.rand(50)
        data2 = np.random.rand(50)
        create_feature_subplot(ax, data1, "Test", "blue", "Line1", data2, "red", "Line2")
        plt.close("all")


# ---------------------------------------------------------------------------
# Integration with FeatureCreator
# ---------------------------------------------------------------------------


class TestTradeFeatureIntegration:

    def test_trade_feature_in_pipeline(self):
        # TradeFeature needs change=True to get min_timesteps > 1 and avoid
        # index errors. When using only non-change features, the internal
        # create_feature still accesses buffer[-2] for past values, so we
        # need to use inc_vwap_change to ensure proper buffer sizing.
        f = TradeFeature(timesteps_back=1, inc_vwap=True, inc_vwap_change=True)
        fc = FeatureCreator(features=[f])
        snaps = make_snapshot_sequence(n=5, include_trades=True)
        for s in snaps:
            fc.feed_datapoint(s)
        assert fc.is_ready()
        result = fc.create_features()
        assert len(result) == 2

    def test_volatility_in_pipeline(self):
        f = VolatilityFeature(timesteps_back=5)
        fc = FeatureCreator(features=[f])
        snaps = make_snapshot_sequence(n=5)
        for s in snaps:
            fc.feed_datapoint(s)
        assert fc.is_ready()
        result = fc.create_features()
        assert len(result) == 1
