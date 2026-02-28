"""
Tests for my_ml_crypto_trading.data_processing.live_prediction
"""

import pytest
import math
from my_ml_crypto_trading.data_processing.live_prediction import (
    probability_random_achieves_accuracy,
    probability_table_for_accuracy,
    convert_bybit_ob_to_snapshot,
)
from datetime import datetime


class TestProbabilityRandomAchievesAccuracy:

    def test_coinflip_at_50_percent(self):
        """At 50% accuracy with p=0.5, probability should be close to 0.5."""
        prob = probability_random_achieves_accuracy(0.5, 1000, p=0.5)
        assert 0.4 < prob < 0.6

    def test_high_accuracy_low_probability(self):
        """Very high accuracy with many attempts should be extremely unlikely for p=0.5."""
        prob = probability_random_achieves_accuracy(0.7, 1000, p=0.5)
        assert prob < 0.001

    def test_perfect_accuracy_very_unlikely(self):
        prob = probability_random_achieves_accuracy(1.0, 100, p=0.5)
        assert prob < 1e-20

    def test_returns_float(self):
        result = probability_random_achieves_accuracy(0.6, 100)
        assert isinstance(result, float)

    def test_biased_coin(self):
        """With p=0.7, achieving 70% should be near 50%."""
        prob = probability_random_achieves_accuracy(0.7, 10000, p=0.7)
        assert 0.3 < prob < 0.7


class TestProbabilityTableForAccuracy:

    def test_prints_table(self, capsys):
        probability_table_for_accuracy(0.6, 100, granularity=0.1)
        captured = capsys.readouterr()
        assert "p (random success rate)" in captured.out


class TestConvertBybitOBToSnapshotLivePrediction:

    def test_basic_conversion(self):
        ob = {
            "ts": int(datetime(2024, 6, 1).timestamp() * 1000),
            "b": [["100.0", "5.0"]],
            "a": [["101.0", "5.0"]],
        }
        result = convert_bybit_ob_to_snapshot(ob)
        assert "mid_price" in result
        assert "bids" in result
        assert "asks" in result
        assert "original_ts" in result
        assert result["mid_price"] == 100.5

    def test_preserves_original_ts(self):
        ts = 1717200000000
        ob = {
            "ts": ts,
            "b": [["50.0", "1.0"]],
            "a": [["51.0", "1.0"]],
        }
        result = convert_bybit_ob_to_snapshot(ob)
        assert result["original_ts"] == ts
