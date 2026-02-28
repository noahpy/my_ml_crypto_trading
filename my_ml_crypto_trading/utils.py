"""
Shared utility functions used across the my_ml_crypto_trading package.

This module consolidates common helper functions that were previously
duplicated across multiple modules.
"""

from datetime import datetime


def convert_bybit_ob_to_snapshot(order_book, *, include_original_ts=False):
    """
    Convert a raw Bybit REST/WS orderbook response into a standardised snapshot dict.

    Parameters
    ----------
    order_book : dict
        Raw Bybit orderbook with keys 'ts', 'b' (bids) and 'a' (asks).
    include_original_ts : bool, optional
        If True, the raw millisecond timestamp is kept under the
        'original_ts' key (used by live-prediction code).

    Returns
    -------
    dict
        {'ts': datetime, 'mid_price': float, 'bids': dict, 'asks': dict}
        and optionally 'original_ts': int.
    """
    ts = datetime.fromtimestamp(order_book['ts'] / 1000)

    bids = {float(price): float(size) for price, size in order_book['b']}
    asks = {float(price): float(size) for price, size in order_book['a']}

    mid_price = (min(asks.keys()) + max(bids.keys())) / 2

    snapshot = {"ts": ts, "mid_price": mid_price, "bids": bids, "asks": asks}
    if include_original_ts:
        snapshot["original_ts"] = order_book['ts']
    return snapshot


def convert_bybit_ob_to_snapshot_raw(order_book):
    """
    Lightweight variant that keeps the raw integer timestamp and omits
    the datetime conversion.  Used by LiveSnapshotDownloader.

    Returns
    -------
    dict
        {'mid_price': float, 'bids': dict, 'asks': dict, 'ts': int}
    """
    bids = {float(price): float(size) for price, size in order_book['b']}
    asks = {float(price): float(size) for price, size in order_book['a']}
    mid_price = (min(asks.keys()) + max(bids.keys())) / 2
    return {"mid_price": mid_price, "bids": bids, "asks": asks, "ts": order_book['ts']}


def update_order_book(current_bids, current_asks, new_bids, new_asks, ob_type):
    """
    Apply an incremental update (or full snapshot) to bid/ask dictionaries.

    Parameters
    ----------
    current_bids, current_asks : dict
        Existing price->size mappings.
    new_bids, new_asks : dict
        Incoming delta or snapshot data.
    ob_type : str
        "snapshot" replaces entirely; anything else applies deltas.

    Returns
    -------
    tuple[dict, dict]
        Updated (bids, asks).
    """
    if ob_type == "snapshot":
        return new_bids, new_asks

    for price in new_bids:
        if new_bids[price] == 0:
            if price in current_bids:
                del current_bids[price]
        else:
            current_bids[price] = new_bids[price]

    for price in new_asks:
        if new_asks[price] == 0:
            if price in current_asks:
                del current_asks[price]
        else:
            current_asks[price] = new_asks[price]

    return current_bids, current_asks


def zero_pad_time(x):
    """Pad a single-digit integer with a leading zero (for date formatting).

    >>> zero_pad_time(5)
    '05'
    >>> zero_pad_time(12)
    12
    """
    if x < 10:
        return f"0{x}"
    return x
