"""
Tests for my_ml_crypto_trading.data_processing.live_snapshot_download
(LiveSnapshotDownloader)
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open
import queue

from my_ml_crypto_trading.data_processing.live_snapshot_download import (
    convert_bybit_ob_to_snapshot,
    LiveSnapshotDownloader,
)


class TestConvertBybitOBToSnapshotDownload:

    def test_basic_conversion(self):
        ob = {
            "ts": int(datetime(2024, 1, 1).timestamp() * 1000),
            "b": [["100.0", "5.0"], ["99.0", "10.0"]],
            "a": [["101.0", "5.0"], ["102.0", "10.0"]],
        }
        result = convert_bybit_ob_to_snapshot(ob)
        assert "mid_price" in result
        assert "bids" in result
        assert "asks" in result
        assert "ts" in result

    def test_mid_price_correct(self):
        ob = {
            "ts": 1000000,
            "b": [["100.0", "5.0"]],
            "a": [["101.0", "5.0"]],
        }
        result = convert_bybit_ob_to_snapshot(ob)
        assert result["mid_price"] == 100.5

    def test_no_timestamp_field(self):
        """ts field is an int timestamp in ms, not a datetime."""
        ob = {
            "ts": 1000000,
            "b": [["50.0", "1.0"]],
            "a": [["51.0", "1.0"]],
        }
        result = convert_bybit_ob_to_snapshot(ob)
        assert isinstance(result["ts"], int)


class TestLiveSnapshotDownloaderTradeCalculation:

    @patch("my_ml_crypto_trading.data_processing.live_snapshot_download.PeriodicLiveRetriever")
    def test_calculate_trades_first_snapshot(self, MockRetriever):
        mock_instance = MockRetriever.return_value
        mock_instance.data_queue = queue.Queue()

        downloader = LiveSnapshotDownloader.__new__(LiveSnapshotDownloader)
        downloader.last_snapshot = None
        downloader.pld = mock_instance

        trades = [
            {"price": "100.0", "size": "1.0", "side": "Buy", "time": "1000"},
            {"price": "101.0", "size": "2.0", "side": "Sell", "time": "2000"},
        ]

        result = downloader.calculate_trades_since_last_timestep(trades)
        assert len(result) == 2
        assert result[0]["price"] == 100.0
        assert result[1]["size"] == 2.0

    @patch("my_ml_crypto_trading.data_processing.live_snapshot_download.PeriodicLiveRetriever")
    def test_calculate_trades_filters_by_timestamp(self, MockRetriever):
        mock_instance = MockRetriever.return_value
        mock_instance.data_queue = queue.Queue()

        downloader = LiveSnapshotDownloader.__new__(LiveSnapshotDownloader)
        downloader.last_snapshot = {"ts": 1500}
        downloader.pld = mock_instance

        trades = [
            {"price": "100.0", "size": "1.0", "side": "Buy", "time": "1000"},
            {"price": "101.0", "size": "2.0", "side": "Sell", "time": "2000"},
        ]

        result = downloader.calculate_trades_since_last_timestep(trades)
        assert len(result) == 1
        assert result[0]["price"] == 101.0
