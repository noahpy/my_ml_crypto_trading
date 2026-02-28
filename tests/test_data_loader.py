"""
Tests for my_ml_crypto_trading.data_processing.data_loader
"""

import pytest
import json
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import pytz

from my_ml_crypto_trading.data_processing.data_loader import DataLoader, convert_bybit_ob_to_snapshot
from my_ml_crypto_trading.data_processing.FeatureCreation import FeatureCreator
from tests.conftest import make_snapshot


# ---------------------------------------------------------------------------
# DataLoader utility methods
# ---------------------------------------------------------------------------


class TestDataLoaderUtils:

    def setup_method(self):
        # Patch HistoricalDataRetriever so it doesn't make real HTTP requests
        with patch(
            "my_ml_crypto_trading.data_processing.data_loader.HistoricalDataRetriever"
        ):
            self.loader = DataLoader()

    def test_zero_pad_single_digit(self):
        assert self.loader.zero_pad(1) == "01"
        assert self.loader.zero_pad(9) == "09"

    def test_zero_pad_double_digit(self):
        assert self.loader.zero_pad(10) == 10
        assert self.loader.zero_pad(12) == 12

    def test_update_order_book_snapshot(self):
        bids, asks = self.loader.update_order_book(
            current_bids={"old": 1},
            current_asks={"old": 1},
            new_bids={100.0: 5.0, 99.0: 10.0},
            new_asks={101.0: 5.0, 102.0: 10.0},
            ob_type="snapshot",
        )
        assert bids == {100.0: 5.0, 99.0: 10.0}
        assert asks == {101.0: 5.0, 102.0: 10.0}

    def test_update_order_book_delta_add(self):
        bids, asks = self.loader.update_order_book(
            current_bids={100.0: 5.0},
            current_asks={101.0: 5.0},
            new_bids={99.0: 3.0},
            new_asks={102.0: 3.0},
            ob_type="delta",
        )
        assert 99.0 in bids
        assert 102.0 in asks
        assert bids[100.0] == 5.0  # existing unchanged

    def test_update_order_book_delta_update(self):
        bids, asks = self.loader.update_order_book(
            current_bids={100.0: 5.0},
            current_asks={101.0: 5.0},
            new_bids={100.0: 8.0},
            new_asks={101.0: 12.0},
            ob_type="delta",
        )
        assert bids[100.0] == 8.0
        assert asks[101.0] == 12.0

    def test_update_order_book_delta_remove(self):
        bids, asks = self.loader.update_order_book(
            current_bids={100.0: 5.0, 99.0: 3.0},
            current_asks={101.0: 5.0, 102.0: 3.0},
            new_bids={100.0: 0},
            new_asks={101.0: 0},
            ob_type="delta",
        )
        assert 100.0 not in bids
        assert 101.0 not in asks
        assert 99.0 in bids
        assert 102.0 in asks

    def test_update_order_book_delta_remove_nonexistent(self):
        """Removing a level that doesn't exist should not crash."""
        bids, asks = self.loader.update_order_book(
            current_bids={100.0: 5.0},
            current_asks={101.0: 5.0},
            new_bids={999.0: 0},  # doesn't exist
            new_asks={999.0: 0},
            ob_type="delta",
        )
        assert bids == {100.0: 5.0}
        assert asks == {101.0: 5.0}


# ---------------------------------------------------------------------------
# convert_bybit_ob_to_snapshot (module-level function)
# ---------------------------------------------------------------------------


class TestConvertBybitOBToSnapshot:

    def test_basic_conversion(self, bybit_orderbook_raw):
        result = convert_bybit_ob_to_snapshot(bybit_orderbook_raw)
        assert "bids" in result
        assert "asks" in result
        assert "mid_price" in result
        assert "ts" in result

    def test_bids_asks_are_float_dicts(self, bybit_orderbook_raw):
        result = convert_bybit_ob_to_snapshot(bybit_orderbook_raw)
        for price, size in result["bids"].items():
            assert isinstance(price, float)
            assert isinstance(size, float)
        for price, size in result["asks"].items():
            assert isinstance(price, float)
            assert isinstance(size, float)

    def test_mid_price_calculation(self, bybit_orderbook_raw):
        result = convert_bybit_ob_to_snapshot(bybit_orderbook_raw)
        best_bid = max(result["bids"].keys())
        best_ask = min(result["asks"].keys())
        expected_mid = (best_bid + best_ask) / 2
        assert abs(result["mid_price"] - expected_mid) < 1e-9

    def test_timestamp_is_datetime(self, bybit_orderbook_raw):
        result = convert_bybit_ob_to_snapshot(bybit_orderbook_raw)
        assert isinstance(result["ts"], datetime)


# ---------------------------------------------------------------------------
# DataLoader.load_features_from_data (mocked retriever)
# ---------------------------------------------------------------------------


class TestLoadFeaturesFromData:

    def _make_ob_lines(self, n_lines=5, base_ts_ms=None):
        """Create fake orderbook JSON lines."""
        if base_ts_ms is None:
            base_ts_ms = int(datetime(2024, 3, 14, 0, 0, 0, tzinfo=pytz.UTC).timestamp() * 1000)

        lines = []
        for i in range(n_lines):
            ts = base_ts_ms + i * 500  # 500ms apart
            ob_type = "snapshot" if i == 0 else "delta"
            line = json.dumps({
                "type": ob_type,
                "ts": ts,
                "data": {
                    "b": [["100.00", "5.0"], ["99.99", "10.0"]],
                    "a": [["100.01", "5.0"], ["100.02", "10.0"]],
                },
            })
            lines.append(line)
        return lines

    @patch("my_ml_crypto_trading.data_processing.data_loader.HistoricalDataRetriever")
    def test_load_features_returns_array(self, MockRetriever):
        import pandas as pd

        mock_retriever_instance = MockRetriever.return_value
        base_ts = datetime(2024, 3, 14, 0, 0, 0, tzinfo=pytz.UTC)
        base_ts_ms = int(base_ts.timestamp() * 1000)

        ob_lines = self._make_ob_lines(n_lines=10, base_ts_ms=base_ts_ms)
        mock_retriever_instance.get_historical_orderbook_day_data.return_value = ob_lines

        # Create matching trade data
        trade_data = pd.DataFrame({
            "timestamp": [base_ts.timestamp() + i * 0.5 for i in range(10)],
            "side": ["Buy"] * 10,
            "size": [1.0] * 10,
            "price": [100.0] * 10,
        })
        mock_retriever_instance.get_historical_trading_day_data.return_value = trade_data

        loader = DataLoader()
        loader.retriever = mock_retriever_instance

        from my_ml_crypto_trading.data_processing.ob_features import MidPriceFeature

        fc = FeatureCreator(features=[MidPriceFeature(inc_mp=True)])

        result = loader.load_features_from_data(
            contract="TESTUSDT",
            start_date=base_ts,
            end_date=base_ts,
            time_delta=timedelta(milliseconds=500),
            feature_creator=fc,
            category="linear",
        )

        assert isinstance(result, np.ndarray)
        # Should have some features extracted
        assert result.shape[0] > 0
