"""
Tests for my_ml_crypto_trading.data_retrieving.LiveDataRetriever
"""

import pytest
import json
import time
from unittest.mock import patch, MagicMock, mock_open

from my_ml_crypto_trading.data_retrieving.LiveDataRetriever import LiveDataRetriever


class TestLiveDataRetriever:
    """Tests for the LiveDataRetriever singleton (mocked Bybit session)."""

    def setup_method(self):
        # Reset the singleton between tests
        LiveDataRetriever._LiveDataRetriever__instance = None

    def teardown_method(self):
        # Cleanup singleton
        try:
            if LiveDataRetriever._LiveDataRetriever__instance is not None:
                LiveDataRetriever._LiveDataRetriever__instance.loop_refresh = False
                time.sleep(0.15)
                LiveDataRetriever._LiveDataRetriever__instance = None
        except Exception:
            pass

    @patch("my_ml_crypto_trading.data_retrieving.LiveDataRetriever.BybitSession")
    @patch("builtins.open", mock_open(read_data='{"key": "test_key", "secret": "test_secret"}'))
    def test_singleton_pattern(self, mock_session):
        ld1 = LiveDataRetriever("fake_key.json")
        ld2 = LiveDataRetriever("fake_key.json")
        # Both should reference the same implementation
        assert ld1._LiveDataRetriever__instance is ld2._LiveDataRetriever__instance

    @patch("my_ml_crypto_trading.data_retrieving.LiveDataRetriever.BybitSession")
    @patch("builtins.open", mock_open(read_data='{"key": "test_key", "secret": "test_secret"}'))
    def test_create_session(self, mock_session_cls):
        ld = LiveDataRetriever("fake_key.json")
        mock_session_cls.assert_called_once_with(
            api_key="test_key", api_secret="test_secret"
        )

    @patch("my_ml_crypto_trading.data_retrieving.LiveDataRetriever.BybitSession")
    @patch("builtins.open", mock_open(read_data='{"key": "k", "secret": "s"}'))
    def test_fetch_current_orderbook(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.get_orderbook.return_value = {
            "result": {"b": [["100", "5"]], "a": [["101", "5"]], "ts": 1000}
        }
        mock_session_cls.return_value = mock_session

        ld = LiveDataRetriever("fake.json")
        result = ld.fetch_current_orderbook("BTCUSDT", "linear")
        assert result is not None
        assert "b" in result

    @patch("my_ml_crypto_trading.data_retrieving.LiveDataRetriever.BybitSession")
    @patch("builtins.open", mock_open(read_data='{"key": "k", "secret": "s"}'))
    def test_fetch_recent_trading_history(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.get_public_trade_history.return_value = {
            "result": {"list": [{"price": "100", "size": "1"}]}
        }
        mock_session_cls.return_value = mock_session

        ld = LiveDataRetriever("fake.json")
        result = ld.fetch_recent_trading_history("BTCUSDT", "linear")
        assert result is not None

    @patch("my_ml_crypto_trading.data_retrieving.LiveDataRetriever.BybitSession")
    @patch("builtins.open", mock_open(read_data='{"key": "k", "secret": "s"}'))
    def test_fetch_orderbook_limit_clamped(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session.get_orderbook.return_value = {"result": {}}
        mock_session_cls.return_value = mock_session

        ld = LiveDataRetriever("fake.json")
        # Passing limit > 200 should get clamped to 200
        ld.fetch_current_orderbook("BTCUSDT", "linear", limit=500)
        call_args = mock_session.get_orderbook.call_args
        assert call_args.kwargs.get("limit", call_args[1].get("limit", None)) == 200
