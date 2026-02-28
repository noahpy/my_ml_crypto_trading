"""
This submodule consists of code dedicated to transforming raw market data (orderbooks, tick trades) 
into structured, numerical data that can be ingested by machine learning algorithms or quantitative mechanisms.

Highlights:

*   **Feature Engineering**: Pipelines defining specific indicator derivations (`trade_features`, `ob_features`).
*   **Data Pipelines**: Tools to transform, window, normalize, or batch data streams (`FeatureCreation`).
*   **Live Inference Architecture**: Components designed to feed Bybit streams directly into 
    live models without latency (`InteractiveLiveProcessor`, `live_prediction`).
"""
