import os
import time
import requests
import pandas as pd
import datetime as dt

BASE = "https://api.polygon.io"

def _api_key() -> str:
    key = os.getenv("POLYGON_API_KEY", "")
    if not key:
        raise RuntimeError("POLYGON_API_KEY not set. Provide via .env or sidebar.")
    return key

def fetch_historic_bars(ticker: str, multiplier: int, timespan: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """
    Pull v2 aggregates bars: /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}
    """
    key = _api_key()
    # Polygon expects ISO8601 dates for day/hour/minute ranges
    frm = start.strftime("%Y-%m-%d")
    to = end.strftime("%Y-%m-%d")

    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{frm}/{to}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": "50000",
        "apiKey": key,
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    results = data.get("results", [])
    if not results:
        return None

    df = pd.DataFrame(results)
    # Columns: t (ms), o, h, l, c, v, vw, n
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df

def fetch_latest_trade(ticker: str) -> dict:
    """
    Latest trade: /v2/last/trade/{ticker}
    """
    key = _api_key()
    url = f"{BASE}/v2/last/trade/{ticker}"
    params = {"apiKey": key}
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        j = r.json()
        trade = j.get("results", {})
        # Normalize fields
        return {
            "price": trade.get("p"),
            "size": trade.get("s"),
            "exchange": trade.get("x"),
            "sip_timestamp": trade.get("t"),
        }
    except Exception:
        return None
