"""
Tests for my_ml_crypto_trading.data_retrieving.HistoricalDataRetriever
"""

import pytest
import io
import zipfile
import gzip
from datetime import datetime
from unittest.mock import patch, MagicMock

from my_ml_crypto_trading.data_retrieving.HistoricalDataRetriever import HistoricalDataRetriever


class TestHistoricalDataRetriever:

    def setup_method(self):
        self.retriever = HistoricalDataRetriever(donwload_path="/tmp/test_data")

    def test_init_default_path(self):
        r = HistoricalDataRetriever()
        assert r.download_path == "."

    def test_init_strips_trailing_slash(self):
        r = HistoricalDataRetriever(donwload_path="/some/path/")
        assert r.download_path == "/some/path"

    def test_init_empty_path_defaults_to_dot(self):
        r = HistoricalDataRetriever(donwload_path="")
        assert r.download_path == "."

    def test_zero_pad_single_digit(self):
        assert self.retriever.zero_pad_time(5) == "05"
        assert self.retriever.zero_pad_time(9) == "09"

    def test_zero_pad_double_digit(self):
        assert self.retriever.zero_pad_time(10) == 10
        assert self.retriever.zero_pad_time(12) == 12

    @patch("my_ml_crypto_trading.data_retrieving.HistoricalDataRetriever.requests.get")
    def test_check_category_symbol_exists_true(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        result = HistoricalDataRetriever.check_category_symbol_exists(
            "orderbook", "BTCUSDT", "linear"
        )
        assert result is True

    @patch("my_ml_crypto_trading.data_retrieving.HistoricalDataRetriever.requests.get")
    def test_check_category_symbol_exists_false(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        result = HistoricalDataRetriever.check_category_symbol_exists(
            "orderbook", "FAKEUSDT", "linear"
        )
        assert result is False

    @patch("my_ml_crypto_trading.data_retrieving.HistoricalDataRetriever.requests.get")
    def test_get_historical_orderbook_day_data_404(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        result = self.retriever.get_historical_orderbook_day_data(
            datetime(2024, 3, 14), "TESTUSDT", "linear"
        )
        assert result is None

    @patch("my_ml_crypto_trading.data_retrieving.HistoricalDataRetriever.requests.get")
    def test_get_historical_orderbook_day_data_success(self, mock_get):
        """Test successful download and extraction of orderbook data."""
        # Create a mock ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("test_data.data", '{"type":"snapshot","ts":1000}\n{"type":"delta","ts":2000}')
        zip_buffer.seek(0)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = zip_buffer.getvalue()
        mock_get.return_value = mock_response

        result = self.retriever.get_historical_orderbook_day_data(
            datetime(2024, 3, 14), "TESTUSDT", "linear"
        )
        assert result is not None
        assert len(result) == 2

    @patch("my_ml_crypto_trading.data_retrieving.HistoricalDataRetriever.requests.get")
    def test_get_historical_trading_day_data_success(self, mock_get):
        """Test successful download and decompression of trade data."""
        csv_content = b"timestamp,side,size,price\n1000.0,Buy,1.0,100.0\n2000.0,Sell,2.0,101.0"
        compressed = gzip.compress(csv_content)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = compressed
        mock_get.return_value = mock_response

        result = self.retriever.get_historical_trading_day_data(
            datetime(2024, 3, 14), "TESTUSDT", "linear"
        )
        assert result is not None
        assert len(result) == 2
        assert "timestamp" in result.columns
        assert "side" in result.columns

    @patch("my_ml_crypto_trading.data_retrieving.HistoricalDataRetriever.requests.get")
    def test_get_historical_orderbook_request_exception(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("fail")
        result = self.retriever.get_historical_orderbook_day_data(
            datetime(2024, 3, 14), "TESTUSDT", "linear"
        )
        assert result is None
