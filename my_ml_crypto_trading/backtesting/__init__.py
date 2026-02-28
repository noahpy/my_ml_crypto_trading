"""
The backtesting submodule provides an isolated, local market simulation environment. 

It allows defined Trading Strategies to plug in historical datasets (fetched by `data_retrieving`) 
to accurately mock their success across varying scenarios.

### Overview

*   **HistoricMarketSimulator**: Uses historical DataRetrievers to simulate orderbook matches, spreads, and fee slippage.
*   **Strategy**: Base interface from which users define specific AI/ML trading agents. 
"""
