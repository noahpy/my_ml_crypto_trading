"""
This submodule contains classes for fetching, storing, and orchestrating historical 
and live cryptocurrency data from the Bybit API and public data providers.

It differentiates between:

*   **Historical Retrieval**: Fetching large blocks of ZIP archives containing tick-level data 
    from previous days/months to build datasets for ML training and model backtesting.
*   **Live Retrieval**: Streaming current market orderbooks and trade history 
    via the Bybit REST/WebSocket API to feed live prediction engines.
"""
