"""
Tests for my_ml_crypto_trading.data_processing.live_snapshot_decompression
"""

import pytest
import os
import tempfile
import gzip
import msgpack
from datetime import date

from my_ml_crypto_trading.data_processing.live_snapshot_decompression import (
    retrieve_and_decompress_data,
)


class TestRetrieveAndDecompressData:

    def _create_test_file(self, tmpdir, category, symbol, date_str, snapshots, compression="gzip"):
        """Helper to create a test data file."""
        symbol_path = os.path.join(tmpdir, category, symbol)
        os.makedirs(symbol_path, exist_ok=True)

        if compression == "gzip":
            file_path = os.path.join(symbol_path, f"{date_str}_snapshots.msgpack.gz")
            with gzip.open(file_path, "wb") as f:
                for snap in snapshots:
                    f.write(msgpack.packb(snap, use_bin_type=True))
        elif compression == "none":
            file_path = os.path.join(symbol_path, f"{date_str}_snapshots.msgpack")
            with open(file_path, "wb") as f:
                for snap in snapshots:
                    f.write(msgpack.packb(snap, use_bin_type=True))
        return file_path

    def test_single_day_gzip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshots = [
                {"mid_price": 100.0, "ts": 1000, "bids": {}, "asks": {}},
                {"mid_price": 101.0, "ts": 2000, "bids": {}, "asks": {}},
            ]
            self._create_test_file(tmpdir, "spot", "CAKEUSDT", "2024-01-01", snapshots)

            result = retrieve_and_decompress_data(
                tmpdir, "CAKEUSDT", "spot",
                "2024-01-01", "2024-01-01",
                compression_format="gzip",
            )
            assert result is not None
            assert len(result) == 2
            assert result[0]["mid_price"] == 100.0

    def test_date_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snap1 = [{"mid_price": 100.0, "ts": 1000}]
            snap2 = [{"mid_price": 200.0, "ts": 2000}]
            self._create_test_file(tmpdir, "spot", "CAKEUSDT", "2024-01-01", snap1)
            self._create_test_file(tmpdir, "spot", "CAKEUSDT", "2024-01-02", snap2)

            result = retrieve_and_decompress_data(
                tmpdir, "CAKEUSDT", "spot",
                "2024-01-01", "2024-01-02",
                compression_format="gzip",
            )
            assert result is not None
            assert len(result) == 2

    def test_no_compression(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshots = [{"mid_price": 42.0, "ts": 3000}]
            self._create_test_file(tmpdir, "spot", "TEST", "2024-06-01", snapshots, compression="none")

            result = retrieve_and_decompress_data(
                tmpdir, "TEST", "spot",
                "2024-06-01", "2024-06-01",
                compression_format="none",
            )
            assert result is not None
            assert len(result) == 1

    def test_missing_directory(self):
        result = retrieve_and_decompress_data(
            "/nonexistent/path", "FAKE", "spot",
            "2024-01-01", "2024-01-01",
        )
        assert result is None

    def test_start_after_end_date(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "spot", "TEST"))
            result = retrieve_and_decompress_data(
                tmpdir, "TEST", "spot",
                "2024-02-01", "2024-01-01",
            )
            assert result is None

    def test_invalid_compression_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "spot", "TEST"))
            result = retrieve_and_decompress_data(
                tmpdir, "TEST", "spot",
                "2024-01-01", "2024-01-01",
                compression_format="invalid",
            )
            assert result is None

    def test_no_files_for_date_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "spot", "TEST"))
            result = retrieve_and_decompress_data(
                tmpdir, "TEST", "spot",
                "2024-01-01", "2024-01-01",
            )
            assert result is None

    def test_date_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshots = [{"mid_price": 50.0, "ts": 4000}]
            self._create_test_file(tmpdir, "spot", "TEST", "2024-03-15", snapshots)

            result = retrieve_and_decompress_data(
                tmpdir, "TEST", "spot",
                date(2024, 3, 15), date(2024, 3, 15),
                compression_format="gzip",
            )
            assert result is not None
            assert len(result) == 1

    def test_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            symbol_path = os.path.join(tmpdir, "spot", "TEST")
            os.makedirs(symbol_path)
            # Create empty gzip file
            file_path = os.path.join(symbol_path, "2024-01-01_snapshots.msgpack.gz")
            with gzip.open(file_path, "wb") as f:
                pass  # empty file

            result = retrieve_and_decompress_data(
                tmpdir, "TEST", "spot",
                "2024-01-01", "2024-01-01",
                compression_format="gzip",
            )
            # Should return empty list (file exists but no data)
            assert result is not None or result == []
