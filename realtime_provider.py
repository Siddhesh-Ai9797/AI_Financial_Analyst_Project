from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


def get_near_realtime_quote(ticker: str):
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        return {
            "last": float(fi.last_price),
            "open": float(fi.open) if fi.open is not None else None,
            "day_high": float(fi.day_high) if fi.day_high is not None else None,
            "day_low": float(fi.day_low) if fi.day_low is not None else None,
            "volume": int(fi.last_volume) if fi.last_volume is not None else None,
            "currency": fi.currency or "USD",
            "asof": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


def _normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure simple OHLCV columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[-1]).title() for col in df.columns]
    else:
        df.columns = [str(c).title() for c in df.columns]
    return df


def get_intraday_frame(ticker: str):
    combos = [
        ("5d", "1h"),
        ("1mo", "1h"),
        ("3mo", "1h"),
        ("5d", "30m"),
    ]

    for period, interval in combos:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                group_by="column",
            )
            if df is None or df.empty:
                continue

            df = _normalize_ohlc(df)
            if "Close" not in df.columns:
                continue

            df.attrs["yf_period"] = period
            df.attrs["yf_interval"] = interval
            return df

        except Exception:
            continue
    return None
