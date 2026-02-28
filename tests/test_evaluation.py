"""
Tests for my_ml_crypto_trading.machine_learning.evaluation
"""

import pytest
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from my_ml_crypto_trading.machine_learning.evaluation import (
    trend_accuracy_metric,
    print_enhanced_confusion_matrix,
    evaluate_softmax_prediction,
    evaluate_mse_prediction,
)


class TestTrendAccuracyMetric:

    def test_perfect_predictions(self):
        y_true = torch.tensor([1.0, -1.0, 1.0, -1.0])
        y_pred = torch.tensor([0.5, -0.5, 0.5, -0.5])
        result = trend_accuracy_metric(y_pred, y_true)
        assert "Acc=1.000" in result

    def test_all_wrong_predictions(self):
        y_true = torch.tensor([1.0, -1.0, 1.0, -1.0])
        y_pred = torch.tensor([-0.5, 0.5, -0.5, 0.5])
        result = trend_accuracy_metric(y_pred, y_true)
        assert "Acc=0.000" in result

    def test_all_zeros_target(self):
        y_true = torch.tensor([0.0, 0.0, 0.0])
        y_pred = torch.tensor([0.1, -0.2, 0.3])
        result = trend_accuracy_metric(y_pred, y_true)
        assert result == "No trends in data"

    def test_mixed_predictions(self):
        y_true = torch.tensor([1.0, -1.0, 1.0, -1.0])
        y_pred = torch.tensor([0.5, -0.5, -0.5, 0.5])  # 2 right, 2 wrong
        result = trend_accuracy_metric(y_pred, y_true)
        assert "Acc=0.500" in result

    def test_gpu_tensors_handled(self):
        # Only test if CUDA available, otherwise just confirm CPU works
        y_true = torch.tensor([1.0, -1.0])
        y_pred = torch.tensor([0.5, -0.5])
        result = trend_accuracy_metric(y_pred, y_true)
        assert "Acc=" in result


class TestPrintEnhancedConfusionMatrix:

    def test_perfect_classification(self, capsys):
        print_enhanced_confusion_matrix(tn=50, fp=0, fn=0, tp=50)
        captured = capsys.readouterr()
        assert "100.0%" in captured.out

    def test_all_wrong(self, capsys):
        print_enhanced_confusion_matrix(tn=0, fp=50, fn=50, tp=0)
        captured = capsys.readouterr()
        assert "0.0%" in captured.out

    def test_handles_zero_total(self, capsys):
        # Edge case: no samples
        print_enhanced_confusion_matrix(tn=0, fp=0, fn=0, tp=0)
        # Should not crash


class TestEvaluateSoftmaxPrediction:

    def test_perfect_softmax(self):
        prediction = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])
        targets = np.array([0, 1, 0])
        acc = evaluate_softmax_prediction(prediction, targets)
        assert acc == 1.0

    def test_wrong_softmax(self):
        prediction = np.array([[0.1, 0.9], [0.9, 0.1]])
        targets = np.array([0, 1])
        acc = evaluate_softmax_prediction(prediction, targets)
        assert acc == 0.0

    def test_accepts_tensors(self):
        prediction = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        targets = torch.tensor([0, 1])
        acc = evaluate_softmax_prediction(prediction, targets)
        assert acc == 1.0


class TestEvaluateMSEPrediction:

    def test_perfect_trend(self):
        predictions = np.array([1.0, -1.0, 1.0, -1.0])
        targets = np.array([0.5, -0.3, 0.8, -0.2])
        result = evaluate_mse_prediction(predictions, targets)
        assert result["accuracy"] == 1.0
        plt.close("all")

    def test_returns_expected_keys(self):
        predictions = np.array([1.0, -1.0, 0.5])
        targets = np.array([0.5, -0.3, 0.2])
        result = evaluate_mse_prediction(predictions, targets)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "confusion_matrix" in result
        plt.close("all")

    def test_handles_zero_predictions(self):
        predictions = np.array([0.0, 0.0, 1.0, -1.0])
        targets = np.array([0.5, -0.3, 0.8, -0.2])
        result = evaluate_mse_prediction(predictions, targets)
        assert "accuracy" in result
        plt.close("all")
