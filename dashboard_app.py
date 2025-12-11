import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from realtime_provider import get_near_realtime_quote

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

# ---------------- Feature rebuild (must match model_trainer) ---------------- #

def add_lags(df: pd.DataFrame, cols, lags=(1, 2, 3, 5)) -> pd.DataFrame:
    """Create lag features for selected columns."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            for L in lags:
                out[f"{c}_lag{L}"] = out[c].shift(L)
    return out


def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Recreate rolling / derived features used during training."""
    out = df.copy()

    # 1-day returns and rolling volatility
    out["ret_1d"] = out["Close"].pct_change()
    out["vol_10"] = out["ret_1d"].rolling(10).std()
    out["vol_20"] = out["ret_1d"].rolling(20).std()

    # Distance from moving averages
    for w in (50, 200):
        sma = out.get(f"SMA_{w}", out["Close"].rolling(w).mean())
        out[f"dist_SMA_{w}"] = (out["Close"] - sma) / sma

    # SMA 50/200 regime flag
    out["sma_cross_up"] = (out.get("SMA_50", 0) > out.get("SMA_200", 0)).astype(int)

    # Bollinger band mid + %b, same as in model_trainer
    if "BB_high" in out.columns and "BB_low" in out.columns:
        out["BB_mavg"] = (out["BB_high"] + out["BB_low"]) / 2.0
        width = out["BB_high"] - out["BB_low"]
        width = width.replace(0, np.nan)
        out["BB_percent_b"] = (out["Close"] - out["BB_low"]) / width

    return out


def build_feat(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rolling + lag transforms (actual feature selection is in the model blob)."""
    df = add_rolling(df)
    df = add_lags(
        df,
        cols=["RSI_14", "MACD", "MACD_signal", "MACD_hist", "BB_percent_b", "ATR_14", "ret_1d"],
        lags=(1, 2, 3, 5),
    )
    return df


def load_model(ticker: str):
    """Load the saved model blob for a ticker (xgb preferred, else rf)."""
    for prefix in ["xgb", "rf"]:
        f = MODELS_DIR / f"{prefix}_model_{ticker}.joblib"
        if f.exists():
            blob = joblib.load(f)
            model = blob["model"]
            feats = blob["features"]
            thr = blob.get("threshold", 0.5)
            return model, feats, thr, f
    raise FileNotFoundError(f"No model found for {ticker}")


# ---------------- Chart helpers ---------------- #

def price_chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure()
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
            )
        )
    else:
        for c in [c for c in ["Close", "Adj Close"] if c in df.columns]:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], name=c))
    fig.update_layout(
        title=f"{ticker} Price",
        xaxis_rangeslider_visible=False,
        height=420,
    )
    return fig


def rsi_macd(df: pd.DataFrame):
    fig = go.Figure()
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI"))
    if "MACD" in df.columns and "MACD_signal" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="MACD_signal"))
    fig.update_layout(height=300)
    return fig


# ---------------- Option strategy logic ---------------- #

def strategy(prob: float, atr: float):
    """Simple rules to map probability + volatility to an option idea."""
    note = "High volatility" if atr >= 0.02 else None

    if prob >= 0.6:
        return "Bull Call Spread", "Strong upward bias", note
    if prob <= 0.4:
        return "Bear Put / Protective Put", "Downward bias", note
    if atr >= 0.02:
        return "Long Straddle", "High vol + uncertain", note
    if 0.5 <= prob < 0.6:
        return "Covered Call", "Mildly bullish", note
    if 0.4 < prob < 0.5:
        return "Cash-Secured Put", "Mildly bearish", note
    return "Hold / No Trade", "Weak signal", note


# ---------------- Sector helpers (for professor's group analysis) ---------------- #

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


def get_available_sectors(all_tickers):
    """Return only those sector groups for which we actually have trained models."""
    sectors = {}
    for name, members in SECTOR_GROUPS.items():
        usable = [t for t in members if t in all_tickers]
        if usable:
            sectors[name] = usable
    return sectors


def compute_sector_signals(sector_name, sector_map, years=3):
    """Aggregate model signals for all tickers in a sector.

    Returns:
        detail_df: per-ticker table with P(up), direction, ATR%
        summary: dict with mean_prob, bullish_frac, n_names
        basket_price: equal-weighted index series (normalized to 100)
    """
    basket = sector_map.get(sector_name, [])
    rows = []
    price_frames = []

    for t in basket:
        # Load model artifact
        try:
            model, feats, thr, _ = load_model(t)
        except Exception:
            continue

        csv_path = DATA_DIR / f"{t}.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
        if df.empty:
            continue

        # Rebuild features and align to what the model expects
        df_feat = build_feat(df)
        use_feats = [f for f in feats if f in df_feat.columns]
        df_feat = df_feat.dropna(subset=use_feats)
        if df_feat.empty:
            continue

        x = df_feat[use_feats].iloc[[-1]]
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x)[:, 1][0])
        else:
            prob = 0.5

        pred = int(prob >= thr)
        if "ATR_14" in df.columns:
            atr = df["ATR_14"].iloc[-1] / df["Close"].iloc[-1]
        else:
            atr = 0.0

        rows.append(
            {
                "Ticker": t,
                "P(up)": prob,
                "Threshold": thr,
                "Direction": "UP" if pred else "DOWN",
                "ATR%": atr * 100.0,
            }
        )

        # Build equal-weight price index
        if "Close" in df.columns:
            price_frames.append(df[["Close"]].rename(columns={"Close": t}))

    if not rows:
        return None, None, None

    detail = pd.DataFrame(rows).set_index("Ticker").sort_values("P(up)", ascending=False)
    mean_prob = detail["P(up)"].mean()
    bullish_frac = (detail["Direction"] == "UP").mean()

    basket_price = None
    if price_frames:
        merged = pd.concat(price_frames, axis=1).dropna()
        if not merged.empty:
            if len(merged) > 250 * years:
                merged = merged.iloc[-250 * years:]
            norm = merged / merged.iloc[0] * 100.0
            basket_price = norm.mean(axis=1)

    summary = {
        "mean_prob": mean_prob,
        "bullish_frac": bullish_frac,
        "n_names": len(detail),
    }
    return detail, summary, basket_price


# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="AI Financial Analyst", layout="wide")
st.title("AI Financial Analyst Dashboard")

st.caption("Machine Learning–Powered Stock Prediction Dashboard")

# Load available models / tickers
# Deduplicate tickers even if both rf_model_ and xgb_model_ exist
tickers = sorted(list({
    p.stem.split("_")[-1] for p in MODELS_DIR.glob("*_model_*.joblib")
}))
if not tickers:
    st.error("No trained models found in 'models/'.")
    st.stop()

mode = st.sidebar.radio("Analysis Mode", ["Single Stock", "Sector Basket"], index=0)

# Figure out which sectors are usable, given trained tickers
sector_map = get_available_sectors(tickers)

# ---------------- Single Stock Mode ---------------- #

if mode == "Single Stock":
    ticker = st.sidebar.selectbox("Ticker", tickers)
    years = st.sidebar.select_slider("Range (yrs)", options=[1, 2, 3, 5, 7, 10], value=3)
    live = st.sidebar.toggle("Live (refresh 60s)", value=False)

    if live:
        st_autorefresh(interval=60_000, key="refresh")

    # Load model and data
    model, feats, thr, _ = load_model(ticker)
    df = pd.read_csv(DATA_DIR / f"{ticker}.csv", index_col=0, parse_dates=True).sort_index()
    disp = df.iloc[-250 * years:] if len(df) > 250 * years else df

    # Rebuild features for prediction
    df_feat = build_feat(df)
    use_feats = [f for f in feats if f in df_feat.columns]
    df_feat = df_feat.dropna(subset=use_feats)

    if df_feat.empty:
        st.error("Not enough data after feature alignment for this ticker.")
        st.stop()

    x = df_feat[use_feats].iloc[[-1]]
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x)[:, 1][0])
    else:
        prob = 0.5

    pred = int(prob >= thr)
    atr = 0.0
    if "ATR_14" in df.columns and "Close" in df.columns:
        atr = df["ATR_14"].iloc[-1] / df["Close"].iloc[-1]

    strat, why, note = strategy(prob, atr)

    # Layout
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.plotly_chart(price_chart(disp, ticker), use_container_width=True)
        st.plotly_chart(rsi_macd(disp), use_container_width=True)

    with right_col:
        st.subheader("Model Signal")
        st.metric("Direction", "UP" if pred else "DOWN")
        st.metric("P(up)", f"{prob:.2%}")
        st.metric("Threshold", f"{thr:.2f}")
        st.metric("ATR%", f"{atr * 100:.2f}%")
        st.divider()

        st.subheader("Suggested Strategy")
        st.markdown(f"**{strat}** — {why}")
        if note:
            st.caption(note)

        st.divider()
        st.subheader("Live Quote (best-effort)")
        quote = get_near_realtime_quote(ticker)
        if quote:
            st.metric("Last", f"{quote['last']:.2f} {quote['currency']}")
            st.caption(f"Updated: {quote['asof']}")
        else:
            st.caption("Live quote unavailable.")


# ---------------- Sector Basket Mode ---------------- #

else:
    if not sector_map:
        st.warning("No sector groups available for the currently trained models.")
    else:
        sector_name = st.sidebar.selectbox("Sector", list(sector_map.keys()))
        years = st.sidebar.select_slider("Range (yrs)", options=[1, 2, 3, 5, 7, 10], value=3)

        detail, summary, basket_price = compute_sector_signals(sector_name, sector_map, years=years)

        if detail is None:
            st.warning("No valid predictions available for this sector.")
        else:
            st.subheader(f"{sector_name} Sector — Aggregated Model Signals")

            c1, c2 = st.columns([2, 1])

            with c1:
                st.dataframe(
                    detail.style.format({"P(up)": "{:.2%}", "ATR%": "{:.2f}"})
                )
                if basket_price is not None:
                    st.line_chart(basket_price, use_container_width=True)

            with c2:
                st.metric("Mean P(up)", f"{summary['mean_prob']:.2%}")
                st.metric(
                    "% Stocks Bullish",
                    f"{summary['bullish_frac'] * 100:.1f}%"
                )
                st.metric("# Names", summary["n_names"])

                # Sector-level option idea using mean P(up) + median ATR
                median_atr = (detail["ATR%"] / 100.0).median()
                s_strat, s_why, s_note = strategy(summary["mean_prob"], median_atr)
                st.subheader("Sector Strategy")
                st.markdown(f"**{s_strat}** — {s_why}")
                if s_note:
                    st.caption(s_note)
