
"""Robust yfinance loader for US + India markets.

Usage examples (run from terminal):

  # Indian equities (NSE/BSE)
  python data_loader.py --tickers RELIANCE.NS TCS.NS SBIN.BO --start 2016-01-01 --end 2025-01-01 --interval 1d

  # Same as above, but let the script auto-append .NS for Indian names
  python data_loader.py --tickers RELIANCE TCS SBIN --india --start 2016-01-01 --end 2025-01-01

  # US tech sector basket (used in the dashboard for sector analysis)
  python data_loader.py --sector Tech --start 2016-01-01 --end 2025-01-01

  # Mix a sector basket with a custom ticker
  python data_loader.py --sector Tech --tickers NFLX --start 2016-01-01 --end 2025-01-01
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange


# Canonical OHLCV column order we expect / export
CANON = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# Simple US sector groups so it's easy to download baskets that the dashboard will use.
# You can extend this dictionary later if you want more sectors or tickers.
SECTOR_GROUPS = {
    "Tech": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "ADBE", "ORCL",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
    ],
    "Crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
    ],
}


def _extract(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw yfinance output into a single-level OHLCV frame."""
    if df.empty:
        raise RuntimeError("Empty data.")

    # Case 1: Simple single-index columns, e.g. from yfinance with group_by="column"
    if not isinstance(df.columns, pd.MultiIndex):
        df = df.rename(columns=str.title)
        cols = [c for c in CANON if c in df.columns]
        if not cols:
            raise RuntimeError(f"No OHLCV found: {df.columns}")
        return df[cols]

    # Case 2: MultiIndex columns (ticker, field). Try to pull out standard fields.
    parts: Dict[str, pd.Series] = {}
    for c in CANON:
        for lvl in range(df.columns.nlevels):
            for col in df.columns:
                if str(col[lvl]).strip().lower() == c.lower():
                    parts[c] = df[col].squeeze()
                    break
            if c in parts:
                break

    if not parts:
        raise RuntimeError("Failed extracting columns from MultiIndex frame.")

    out = pd.concat(parts.values(), axis=1)
    out.columns = list(parts.keys())
    return out


def download_one(ticker: str, start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    """Download one ticker from yfinance and normalize columns."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    return _extract(df)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and supervised-learning targets used by the models."""
    df = df.copy()
    c, h, l = df["Close"], df["High"], df["Low"]

    # Trend / moving averages
    df["SMA_50"] = SMAIndicator(c, 50).sma_indicator()
    df["SMA_200"] = SMAIndicator(c, 200).sma_indicator()
    df["EMA_12"] = EMAIndicator(c, 12).ema_indicator()
    df["EMA_26"] = EMAIndicator(c, 26).ema_indicator()

    # MACD
    macd = MACD(c)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    # Momentum
    df["RSI_14"] = RSIIndicator(c, 14).rsi()

    # Volatility (Bollinger + ATR)
    bb = BollingerBands(c, window=20, window_dev=2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    df["BB_width"] = df["BB_high"] - df["BB_low"]

    atr = AverageTrueRange(h, l, c, window=14)
    df["ATR_14"] = atr.average_true_range()

    # Next-day returns and binary classification target
    df["close_t+1"] = c.shift(-1)
    df["next_ret_1d"] = (df["close_t+1"] - c) / c
    df["target_up"] = (df["next_ret_1d"] > 0).astype(int)

    # Drop rows with NaNs from indicator warm-up periods / final shift
    return df.dropna()


def process_and_save(ticker: str, start: str, end: Optional[str], interval: str, outdir: Path) -> Path:
    """Download, enrich with indicators, and write CSV for a single ticker."""
    raw = download_one(ticker, start, end, interval)
    feat = add_indicators(raw)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{ticker}.csv"
    feat.to_csv(p)
    return p


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tickers",
        nargs="+",
        default=[],
        help="List of tickers (e.g. AAPL RELIANCE.NS SBIN.BO). Can be combined with --sector.",
    )
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument("--interval", default="1d", choices=["1d", "1wk", "1mo"])
    p.add_argument("--out", default="data")
    p.add_argument(
        "--india",
        action="store_true",
        help="Auto-append .NS to tickers that don't already end with .NS/.BO (for Indian stocks)",
    )
    p.add_argument(
        "--sector",
        choices=sorted(SECTOR_GROUPS.keys()),
        help=(
            "Optional US sector basket to download (e.g. Tech, Financials, Energy).\n"
            "Tickers from this sector are merged with any provided via --tickers."
        ),
    )
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)

    # Start from explicit tickers, then optionally extend with a sector basket
    tickers: List[str] = list(a.tickers) if a.tickers else []
    if a.sector:
        tickers.extend(SECTOR_GROUPS[a.sector])

    # Deduplicate while preserving order
    seen = set()
    tickers = [t for t in tickers if not (t in seen or seen.add(t))]

    if not tickers:
        raise SystemExit("No tickers provided. Use --tickers and/or --sector.")

    # If --india is set, automatically append .NS to plain Indian symbols
    tickers = [
        (t + ".NS") if a.india and not t.endswith((".NS", ".BO")) else t
        for t in tickers
    ]

    print(f"Downloading {tickers}")
    for t in tickers:
        try:
            p = process_and_save(t, a.start, a.end, a.interval, out)
            print("✓", t, "→", p)
        except Exception as e:
            print("✗", t, ":", e)
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
