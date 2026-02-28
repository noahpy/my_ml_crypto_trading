# Crypto Trading Project Documentation

This document provides a comprehensive overview of the project structure, differentiating between implementation code and scripting/analysis files. It also details the workflow for generating browsable API documentation.

## Project Structure

The project is organized into a main python package `my_ml_crypto_trading` containing the core logic, and various notebooks and scripts for analysis and execution.

### Implementation Files (Library)

These files contain the reusable logic, classes, and functions used across the project. They are located in the `my_ml_crypto_trading` directory.

#### `my_ml_crypto_trading/`
*   **`backtesting/`**: Modules related to simulating trading strategies on historical data.
    *   `HistoricMarketSimulator.py`: Simulates market conditions for backtesting.
    *   `Strategy.py`: Base classes and definitions for trading strategies.
*   **`data_processing/`**: Functions for cleaning, transforming, and feature engineering.
    *   `data_loader.py`: Utilities for loading raw data.
    *   `FeatureCreation.py`: Logic for creating features from raw data.
    *   `InteractiveLiveProcessor.py`: Real-time data processing for live trading.
    *   `live_prediction.py`: Modules for making predictions in a live environment.
    *   `live_snapshot_decompression.py` & `live_snapshot_download.py`: Handling of live market snapshots.
    *   `ob_features.py`: Order book feature engineering.
    *   `process_prediction_data.py`: Preparing data for model inference.
    *   `trade_features.py`: Trade-based feature engineering.
*   **`data_retrieving/`**: Modules for fetching data from exchanges or storage.
    *   `HistoricalDataRetriever.py`: Fetching historical market data.
    *   `InteractiveHistoricalRetriever.py`: Interactive tools for history retrieval.
    *   `InteractiveLiveRetriever.py`: Real-time data fetching.
    *   `LiveDataRetriever.py`: Abstract/Base classes for live data.
    *   `PeriodicLiveRetriever.py`: Fetching data at set intervals.
*   **`machine_learning/`**: Core machine learning definitions and training loops.
    *   `evaluation.py`: Metrics and evaluation logic for models.
    *   `processing.py`: specific data processing for ML pipelines.
    *   `training.py`: Model training loops and utilities.
    *   `wrapper.py`: Wrappers for model interaction.
    *   `transformer_model/`: Specific implementation of transformer architectures.

### Scripting & Analysis Files

These files are used for running experiments, training models, or performing analysis. They utilize the code from the implementation files.

#### Root Directory
*   **`main.ipynb`**: The primary entry point for running the main pipeline or analysis.
*   **`learning.ipynb`**: Notebook focused on training and experimenting with ML models.
*   **`data_loader.ipynb`**: specialized notebook for testing and visualizing data loading.
*   **`generate_docs.py`**: A utility script to generate the HTML API documentation.
*   **`pyproject.toml`**: Project configuration and dependencies.

#### `notebooks/`
*   **`backtesting.ipynb`**: Notebook for running and visualizing backtests.
*   **`data_loader.ipynb`**, **`learning.ipynb`**, **`main.ipynb`**: Copies or specific versions of the root notebooks.

## API Documentation

We use `pdoc` to generate browsable API documentation (HTML) for the implementation files. This provides a Doxygen-like experience for exploring modules, classes, and functions.

### How to Generate Documentation

1.  Ensure you have the project dependencies installed (including `pdoc`).
2.  Run the generation script from the project root:

    ```bash
    python generate_docs.py
    ```

    Or manually:

    ```bash
    pdoc my_ml_crypto_trading -o docs
    ```

3.  The documentation will be generated in the `docs/` folder.
4.  Open `docs/index.html` in your web browser to browse the API reference.

### Browsing the Documentation

*   **Index**: The entry point `docs/index.html` lists all submodules.
*   **Search**: The generated documentation includes a search bar to find specific functions or classes.
*   **Source**: You can view the source code for any function by clicking the "Source" button next to its definition in the documentation.
