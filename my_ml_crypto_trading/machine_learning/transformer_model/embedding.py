"""
Embedding utilities for transformer models.

The canonical PositionalEncoding implementation lives in
``my_ml_crypto_trading.machine_learning.transformer_model.model``.
This module re-exports it for backward compatibility.
"""

from my_ml_crypto_trading.machine_learning.transformer_model.model import PositionalEncoding

__all__ = ["PositionalEncoding"]
