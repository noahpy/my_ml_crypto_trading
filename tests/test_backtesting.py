"""
Tests for my_ml_crypto_trading.backtesting.HistoricMarketSimulator
and my_ml_crypto_trading.backtesting.Strategy
"""

import pytest
import json
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pytz

from my_ml_crypto_trading.backtesting.HistoricMarketSimulator import (
    HistoricMarketSimulator,
    PeriodicDataLoader,
)
from my_ml_crypto_trading.backtesting.Strategy import Strategy, SimpleStrategy


# ---------------------------------------------------------------------------
# HistoricMarketSimulator utility methods
# ---------------------------------------------------------------------------


class TestHistoricMarketSimulatorUtils:

    def test_zero_pad(self):
        hms = HistoricMarketSimulator()
        assert hms.zero_pad(5) == "05"
        assert hms.zero_pad(10) == 10

    def test_update_order_book_snapshot(self):
        hms = HistoricMarketSimulator()
        bids, asks = hms.update_order_book(
            current_bids={"old": 1},
            current_asks={"old": 1},
            new_bids={100.0: 5.0},
            new_asks={101.0: 5.0},
            ob_type="snapshot",
        )
        assert bids == {100.0: 5.0}
        assert asks == {101.0: 5.0}

    def test_update_order_book_delta_add(self):
        hms = HistoricMarketSimulator()
        bids, asks = hms.update_order_book(
            current_bids={100.0: 5.0},
            current_asks={101.0: 5.0},
            new_bids={99.0: 3.0},
            new_asks={102.0: 3.0},
            ob_type="delta",
        )
        assert 99.0 in bids
        assert 102.0 in asks

    def test_update_order_book_delta_remove(self):
        hms = HistoricMarketSimulator()
        bids, asks = hms.update_order_book(
            current_bids={100.0: 5.0, 99.0: 3.0},
            current_asks={101.0: 5.0},
            new_bids={100.0: 0},
            new_asks={},
            ob_type="delta",
        )
        assert 100.0 not in bids
        assert 99.0 in bids


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------


class TestPositionManagement:

    def test_initialize_positions(self):
        hms = HistoricMarketSimulator()
        tradable_assets = [
            {"category": "linear", "symbol": "CAKEUSDT", "base": "USDT"}
        ]
        hms.initialize_positions(tradable_assets)
        assert "CAKEUSDT" in hms.positions
        assert "USDT" in hms.positions
        assert hms.positions["CAKEUSDT"]["balance"] == 0
        assert hms.positions["USDT"]["balance"] == 0

    def test_initialize_multiple_assets(self):
        hms = HistoricMarketSimulator()
        tradable_assets = [
            {"category": "linear", "symbol": "CAKEUSDT", "base": "USDT"},
            {"category": "linear", "symbol": "BTCUSDT", "base": "USDT"},
        ]
        hms.initialize_positions(tradable_assets)
        assert "CAKEUSDT" in hms.positions
        assert "BTCUSDT" in hms.positions
        assert "USDT" in hms.positions

    def test_update_positions_buy_order(self):
        hms = HistoricMarketSimulator()
        tradable_assets = [
            {"category": "linear", "symbol": "CAKEUSDT", "base": "USDT"}
        ]
        hms.initialize_positions(tradable_assets)

        orders = [{
            "symbol": "CAKEUSDT",
            "side": "buy",
            "price": 5.0,
            "qty": 10,
            "type": "FOK",
        }]

        data = {
            "linear": {
                "CAKEUSDT": {
                    "bids": {4.99: 100.0},
                    "asks": {5.00: 100.0},
                }
            }
        }

        hms.update_positions(orders, data)
        assert hms.positions["CAKEUSDT"]["balance"] == 10
        assert hms.positions["USDT"]["balance"] == -50.0

    def test_update_positions_sell_order(self):
        hms = HistoricMarketSimulator()
        tradable_assets = [
            {"category": "linear", "symbol": "CAKEUSDT", "base": "USDT"}
        ]
        hms.initialize_positions(tradable_assets)
        hms.positions["CAKEUSDT"]["balance"] = 10  # already have some

        orders = [{
            "symbol": "CAKEUSDT",
            "side": "sell",
            "price": 4.0,
            "qty": 5,
            "type": "FOK",
        }]

        data = {
            "linear": {
                "CAKEUSDT": {
                    "bids": {4.99: 100.0},
                    "asks": {5.00: 100.0},
                }
            }
        }

        hms.update_positions(orders, data)
        assert hms.positions["CAKEUSDT"]["balance"] == 5
        assert hms.positions["USDT"]["balance"] == pytest.approx(4.99 * 5)

    def test_update_positions_unfilled_buy(self):
        """Buy order with price below best ask should not fill."""
        hms = HistoricMarketSimulator()
        tradable_assets = [
            {"category": "linear", "symbol": "CAKEUSDT", "base": "USDT"}
        ]
        hms.initialize_positions(tradable_assets)

        orders = [{
            "symbol": "CAKEUSDT",
            "side": "buy",
            "price": 3.0,  # below best ask of 5.0
            "qty": 10,
            "type": "FOK",
        }]

        data = {
            "linear": {
                "CAKEUSDT": {
                    "bids": {4.99: 100.0},
                    "asks": {5.00: 100.0},
                }
            }
        }

        hms.update_positions(orders, data)
        assert hms.positions["CAKEUSDT"]["balance"] == 0
        assert hms.positions["USDT"]["balance"] == 0


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class TestStrategyABC:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Strategy()

    def test_concrete_strategy_implements_interface(self):
        """Just verify the SimpleStrategy can be imported without error."""
        # SimpleStrategy requires model files, so just verify it's a Strategy subclass
        assert issubclass(SimpleStrategy, Strategy)


# ---------------------------------------------------------------------------
# PeriodicDataLoader (mocked retriever)
# ---------------------------------------------------------------------------


class TestPeriodicDataLoader:

    @patch("my_ml_crypto_trading.backtesting.HistoricMarketSimulator.HistoricalDataRetriever")
    def test_init_buffer(self, MockRetriever):
        mock_instance = MockRetriever.return_value
        mock_instance.get_historical_orderbook_day_data.return_value = [
            json.dumps({
                "type": "snapshot",
                "ts": 1000000,
                "data": {"b": [["100.0", "5.0"]], "a": [["101.0", "5.0"]]},
            })
        ]
        import pandas as pd
        mock_instance.get_historical_trading_day_data.return_value = pd.DataFrame({
            "timestamp": [1000.0],
            "side": ["Buy"],
            "size": [1.0],
            "price": [100.0],
        })

        data_req = {"linear": {"TESTUSDT": ["bids", "asks", "trades"]}}
        start = datetime(2024, 1, 1, tzinfo=pytz.UTC)

        pdl = PeriodicDataLoader(data_req, start)
        assert "linear" in pdl.buffered_data
        assert "TESTUSDT" in pdl.buffered_data["linear"]

    @patch("my_ml_crypto_trading.backtesting.HistoricMarketSimulator.HistoricalDataRetriever")
    def test_update_order_book(self, MockRetriever):
        mock_instance = MockRetriever.return_value
        mock_instance.get_historical_orderbook_day_data.return_value = []
        import pandas as pd
        mock_instance.get_historical_trading_day_data.return_value = pd.DataFrame()

        data_req = {"linear": {"TESTUSDT": ["bids"]}}
        start = datetime(2024, 1, 1, tzinfo=pytz.UTC)
        pdl = PeriodicDataLoader(data_req, start)

        bids, asks = pdl.update_order_book(
            {100.0: 5.0}, {101.0: 5.0},
            {100.0: 8.0}, {101.0: 0},
            "delta",
        )
        assert bids[100.0] == 8.0
        assert 101.0 not in asks
