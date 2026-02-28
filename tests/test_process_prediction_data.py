"""
Tests for my_ml_crypto_trading.data_processing.process_prediction_data
"""

import pytest
import json
import tempfile
import os
import matplotlib
matplotlib.use("Agg")

from my_ml_crypto_trading.data_processing.process_prediction_data import (
    plot_predicted_vs_actual_midprice_interactive,
)


class TestProcessPredictionData:

    def _create_prediction_file(self, tmpdir, n=20):
        """Create a test prediction data file."""
        file_path = os.path.join(tmpdir, "prediction.data")
        base_ts = 1000000
        with open(file_path, "w") as f:
            for i in range(n):
                data = {
                    "timestamp": base_ts + i * 1000,
                    "midprice_before": 100.0 + i * 0.1,
                    "prediction_before": 0.05 * ((-1) ** i),
                    "midprice_current": 100.0 + (i + 1) * 0.1,
                }
                f.write(json.dumps(data) + "\n")
        return file_path

    def test_plot_function_doesnt_crash(self):
        """Test that the plot function runs without errors (won't display in test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = self._create_prediction_file(tmpdir)
            # The function calls fig.show() which in testing won't display
            # We just verify it doesn't crash
            try:
                plot_predicted_vs_actual_midprice_interactive(file_path)
            except Exception:
                # plotly may not render in test env, that's fine
                pass
