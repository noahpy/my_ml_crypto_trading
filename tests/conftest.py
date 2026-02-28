"""
Shared test fixtures and helpers for the crypto_trading test suite.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import pytz


def make_snapshot(
    mid_price=100.0,
    num_levels=10,
    spread=0.1,
    trades=None,
    timestamp=None,
):
    """
    Create a synthetic market data snapshot for testing.

    Args:
        mid_price: The mid price around which bids/asks are centered.
        num_levels: Number of price levels on each side.
        spread: The bid-ask spread.
        trades: List of trade dicts. Defaults to empty list.
        timestamp: Snapshot timestamp. Defaults to now (UTC).

    Returns:
        A dict matching the snapshot schema used throughout the codebase.
    """
    if timestamp is None:
        timestamp = datetime.now(tz=pytz.UTC)
    if trades is None:
        trades = []

    best_bid = mid_price - spread / 2
    best_ask = mid_price + spread / 2

    bids = {}
    asks = {}
    for i in range(num_levels):
        bid_price = round(best_bid - i * spread, 6)
        ask_price = round(best_ask + i * spread, 6)
        bids[bid_price] = float(10 + i)  # increasing size away from mid
        asks[ask_price] = float(10 + i)

    return {
        "timestamp": timestamp,
        "mid_price": mid_price,
        "bids": bids,
        "asks": asks,
        "trades": trades,
    }


def make_trade(side="Buy", size=1.0, price=100.0):
    """Create a single trade dict."""
    return {"side": side, "size": size, "price": price}


def make_snapshot_sequence(
    n=20,
    base_mid=100.0,
    spread=0.1,
    num_levels=10,
    include_trades=True,
    time_step=timedelta(seconds=1),
):
    """
    Generate a list of n sequential snapshots with slightly varying mid prices.
    Useful for testing features that require a buffer of historical data.
    """
    snapshots = []
    ts = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    for i in range(n):
        mp = base_mid + 0.01 * (i % 5) - 0.02  # small oscillation
        trades = []
        if include_trades:
            trades = [
                make_trade("Buy", size=1.0 + 0.1 * i, price=mp + 0.01),
                make_trade("Sell", size=0.5 + 0.05 * i, price=mp - 0.01),
            ]
        snapshots.append(
            make_snapshot(
                mid_price=mp,
                num_levels=num_levels,
                spread=spread,
                trades=trades,
                timestamp=ts + i * time_step,
            )
        )
    return snapshots


@pytest.fixture
def single_snapshot():
    """A single snapshot with trades."""
    return make_snapshot(
        mid_price=100.0,
        trades=[
            make_trade("Buy", 2.0, 100.05),
            make_trade("Sell", 1.5, 99.95),
        ],
    )


@pytest.fixture
def snapshot_sequence():
    """A sequence of 20 snapshots for buffer-based feature tests."""
    return make_snapshot_sequence(n=20)


@pytest.fixture
def short_snapshot_sequence():
    """A short sequence of 5 snapshots."""
    return make_snapshot_sequence(n=5)


@pytest.fixture
def bybit_orderbook_raw():
    """A raw Bybit orderbook response for convert_bybit_ob_to_snapshot tests."""
    return {
        "ts": int(datetime(2024, 6, 1, 12, 0, 0).timestamp() * 1000),
        "b": [["100.00", "5.0"], ["99.99", "10.0"], ["99.98", "15.0"]],
        "a": [["100.01", "5.0"], ["100.02", "10.0"], ["100.03", "15.0"]],
    }


@pytest.fixture
def sample_ob_json_lines():
    """
    Simulated orderbook JSON lines as returned by HistoricalDataRetriever.
    Includes a snapshot followed by delta updates.
    """
    import json

    ts_base = int(datetime(2024, 3, 14, 0, 0, 0, tzinfo=pytz.UTC).timestamp() * 1000)

    snapshot_line = json.dumps(
        {
            "type": "snapshot",
            "ts": ts_base,
            "data": {
                "b": [["100.00", "5.0"], ["99.99", "10.0"]],
                "a": [["100.01", "5.0"], ["100.02", "10.0"]],
            },
        }
    )

    delta_line = json.dumps(
        {
            "type": "delta",
            "ts": ts_base + 1000,
            "data": {
                "b": [["100.00", "7.0"]],  # update bid
                "a": [["100.01", "0"]],  # remove ask level
            },
        }
    )

    return [snapshot_line, delta_line]
