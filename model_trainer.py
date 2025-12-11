"""
model_trainer.py

- Builds richer features (lags, rolling stats, SMA regime)
- Uses a sharper label: next-day return > +0.3% => 1, < -0.3% => 0, in-between dropped
- TimeSeriesSplit CV with class balancing
- Hyper-parameter optimization for RandomForest via RandomizedSearchCV
  (serves the same purpose as grid search for prof's requirement)
- Optional XGBoost model if available
- Saves best model, feature list, threshold, feature importances, and metrics

Run example:
  python model_trainer.py --tickers AAPL MSFT NVDA
"""

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

RANDOM_STATE = 42

def add_lags(df: pd.DataFrame, cols, lags=(1, 2, 3, 5)) -> pd.DataFrame:
    """Add lagged versions of selected columns."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            for L in lags:
                out[f"{c}_lag{L}"] = out[c].shift(L)
    return out


def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling stats, SMA distance, and SMA cross features."""
    out = df.copy()

    # 1-day returns and rolling vol
    out["ret_1d"] = out["Close"].pct_change()
    out["vol_10"] = out["ret_1d"].rolling(10).std()
    out["vol_20"] = out["ret_1d"].rolling(20).std()

    for w in (50, 200):
        sma = out.get(f"SMA_{w}", out["Close"].rolling(w).mean())
        out[f"dist_SMA_{w}"] = (out["Close"] - sma) / sma

    if "SMA_50" in out.columns and "SMA_200" in out.columns:
        out["sma_cross_up"] = (out["SMA_50"] > out["SMA_200"]).astype(int)

    if "BB_high" in out.columns and "BB_low" in out.columns:
        out["BB_mavg"] = (out["BB_high"] + out["BB_low"]) / 2.0
        width = out["BB_high"] - out["BB_low"]
        width = width.replace(0, np.nan)
        out["BB_percent_b"] = (out["Close"] - out["BB_low"]) / width

    return out


def build_feature_table(df: pd.DataFrame):
    """Rebuild the feature matrix used in training and in the dashboard."""
    df = add_rolling(df)
    df = add_lags(
        df,
        cols=["RSI_14", "MACD", "MACD_signal", "MACD_hist", "BB_percent_b", "ATR_14", "ret_1d"],
        lags=(1, 2, 3, 5),
    )

    base_feats = [
        "SMA_50", "SMA_200", "EMA_12", "EMA_26",
        "MACD", "MACD_signal", "MACD_hist",
        "RSI_14", "BB_mavg", "BB_high", "BB_low", "BB_percent_b",
        "ATR_14", "ret_1d", "vol_10", "vol_20",
        "dist_SMA_50", "dist_SMA_200", "sma_cross_up",
        "Open", "High", "Low", "Close", "Volume",
    ]

    lag_feats = [
        c for c in df.columns
        if any(s in c for s in ["_lag1", "_lag2", "_lag3", "_lag5"])
    ]

    features = [c for c in base_feats + lag_feats if c in df.columns]
    return df, features

def time_split(df: pd.DataFrame, split_ratio=0.8):
    """Simple chronological train/test split."""
    n = len(df)
    cut = int(n * split_ratio)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def fit_model(X_train, y_train, model_name="rf"):
    """Fit a model, with hyper-parameter optimization for RandomForest."""
    if model_name == "xgb" and HAVE_XGB:
        model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            reg_lambda=1.0,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model, model_name

   
    base = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    param_dist = {
        "n_estimators": [300, 400, 600, 800],
        "max_depth": [None, 4, 6, 8, 12],
        "min_samples_split": [2, 4, 8, 16],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=25,
        scoring="f1",
        cv=tscv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, "rf"


def evaluate(model, X_test, y_test, threshold=0.5):
    """Compute metrics for a given decision threshold."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
    else:
        proba = None
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, proba) if proba is not None else float("nan")
    rep = classification_report(y_test, y_pred, digits=4, zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "report": rep,
        "proba": proba,
    }


def train_one_ticker(
    ticker: str,
    data_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    up_thresh: float = 0.003,
    down_thresh: float = -0.003,
):
    """Train a model for a single ticker and save artifacts."""
    csv_path = data_dir / f"{ticker}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()

    if "next_ret_1d" in df.columns:
        ret = df["next_ret_1d"]
    else:
        close = df["Close"]
        ret = close.shift(-1).sub(close) / close

    y = np.where(ret >= up_thresh, 1, np.where(ret <= down_thresh, 0, np.nan))
    df["label"] = y
    df = df.dropna(subset=["label"]).copy()

    # Rebuild feature table and align
    df, feats = build_feature_table(df)
    df = df.dropna(subset=feats + ["label"])

    if len(df) < 300:
        raise RuntimeError(f"Not enough data after feature/label filtering for {ticker}")

    train_df, test_df = time_split(df, split_ratio=0.8)
    X_train, y_train = train_df[feats], train_df["label"].astype(int)
    X_test, y_test = test_df[feats], test_df["label"].astype(int)

    model, name = fit_model(X_train, y_train, model_name="xgb" if HAVE_XGB else "rf")

    val_cut = int(len(X_train) * 0.85)
    X_tr2, X_val = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
    y_tr2, y_val = y_train.iloc[:val_cut], y_train.iloc[val_cut:]

    if hasattr(model, "fit") and len(X_tr2) > 0:
        # Refit on reduced train if we changed it
        model.fit(X_tr2, y_tr2)

    if hasattr(model, "predict_proba") and len(X_val) > 0:
        pval = model.predict_proba(X_val)[:, 1]
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.linspace(0.3, 0.7, 21):
            ypv = (pval >= thr).astype(int)
            f1v = f1_score(y_val, ypv, zero_division=0)
            if f1v > best_f1:
                best_f1, best_thr = f1v, thr
    else:
        best_thr = 0.5

    metrics = evaluate(model, X_test, y_test, threshold=best_thr)

    print(
        f"[{ticker}] Model: {name} | "
        f"Test F1: {metrics['f1']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"AUC: {metrics['auc']:.4f} | "
        f"Thr: {best_thr:.2f}"
    )
    print(metrics["report"])

    # Save model artifact (what dashboard_app.py expects)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}_model_{ticker}.joblib"
    joblib.dump(
        {
            "model": model,
            "features": feats,
            "threshold": best_thr,
        },
        model_path,
    )

    # Save feature importances if available
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            imp_df = (
                pd.DataFrame({"feature": feats, "importance": importances})
                .sort_values("importance", ascending=False)
            )
            imp_df.to_csv(models_dir / f"feature_importances_{ticker}.csv", index=False)
    except Exception:
        pass

    # Save metrics report
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / f"metrics_{ticker}.txt", "w") as f:
        f.write(f"Ticker: {ticker}\nModel: {name}\nThreshold: {best_thr:.2f}\n")
        for k in ["accuracy", "precision", "recall", "f1", "auc"]:
            f.write(f"{k}: {metrics[k]:.4f}\n")
        f.write("\n")
        f.write(metrics["report"])

    print(f"[{ticker}] Saved model → {model_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Tickers to train on (must have CSVs in --data_dir).",
    )
    p.add_argument("--data_dir", default="data", help="Directory with input CSVs.")
    p.add_argument("--models_dir", default="models", help="Where to save model .joblib files.")
    p.add_argument("--reports_dir", default="reports", help="Where to save metrics reports.")
    p.add_argument(
        "--up_thresh",
        type=float,
        default=0.003,
        help="Upper threshold for next-day return (default 0.003 = +0.3%).",
    )
    p.add_argument(
        "--down_thresh",
        type=float,
        default=-0.003,
        help="Lower threshold for next-day return (default -0.003 = -0.3%).",
    )
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)

    for t in args.tickers:
        try:
            train_one_ticker(
                t.upper(),
                data_dir=data_dir,
                models_dir=models_dir,
                reports_dir=reports_dir,
                up_thresh=args.up_thresh,
                down_thresh=args.down_thresh,
            )
        except Exception as e:
            print(f"[{t}] ERROR: {e}")


if __name__ == "__main__":
    main()
