"""
backend/features.py
===================
Single source of truth for ALL feature engineering and target construction
used by Market Signal Lab.

This module is imported by:
  - notebooks/01_data_and_features.py  (data pipeline / EDA)
  - tests/test_features.py             (unit tests)
  - backend/                           (FastAPI prediction endpoints, Phase 5)

LEAKAGE POLICY
--------------
Every function here operates exclusively on past data.  The only exception
is compute_target(), which intentionally uses future prices to build labels —
but it is structurally isolated so it cannot accidentally leak into features.
Any new feature added here MUST be reviewed against this policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

# How many trading days ahead we forecast direction (~1 calendar week)
FORECAST_HORIZON: int = 5

FEATURE_COLS: list[str] = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d",
    "vol_5d", "vol_21d",
    "MA_ratio_5_21", "MA_ratio_10_50",
    "RSI_14",
    "MACD_line", "MACD_signal", "MACD_hist",
    "vol_change_5d",
]


# ──────────────────────────────────────────────────────────────
# INDICATOR HELPERS
# ──────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI) using Wilder's EMA smoothing.

    Measures momentum on a 0-100 scale:
      > 70 → overbought  |  < 30 → oversold

    Parameters
    ----------
    series : pd.Series
        Closing price series (must be sorted oldest→newest).
    window : int
        Look-back period (default 14 trading days per Wilder's original).

    Returns
    -------
    pd.Series named "RSI_{window}" with the same index as `series`.
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)

    # Wilder's smoothing: alpha = 1/window  ↔  com = window-1
    avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename(f"RSI_{window}")


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence/Divergence (MACD).

    MACD line   = EMA(fast) - EMA(slow)
    Signal line = EMA(MACD line, signal)
    Histogram   = MACD line - Signal line

    A positive histogram means short-term momentum is stronger than
    long-term momentum (bullish), and vice versa.

    Returns
    -------
    pd.DataFrame with columns: MACD_line, MACD_signal, MACD_hist
    """
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line

    return pd.DataFrame(
        {"MACD_line": macd_line, "MACD_signal": signal_line, "MACD_hist": histogram},
        index=series.index,
    )


# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (all backward-looking)
# ──────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix for a single ticker's OHLCV DataFrame.

    All features are computed using only data available at time t
    (right-aligned rolling windows, no center=True, no shift(-n)).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Close, Volume, and optionally Ticker.
        Index must be a DatetimeIndex sorted oldest→newest.

    Returns
    -------
    pd.DataFrame with columns listed in FEATURE_COLS plus "Ticker".

    Feature Catalogue
    -----------------
    Returns  (log returns — additive and approximately normal)
      ret_1d        ln(close_t / close_{t-1})      — daily momentum
      ret_5d        ln(close_t / close_{t-5})      — weekly momentum
      ret_10d       ln(close_t / close_{t-10})     — bi-weekly momentum
      ret_21d       ln(close_t / close_{t-21})     — monthly momentum

    Volatility  (annualised by × sqrt(252))
      vol_5d        rolling 5-day std of log returns
      vol_21d       rolling 21-day std of log returns

    Moving-average ratios  (> 1 = price above average = bullish)
      MA_ratio_5_21   SMA(5)  / SMA(21)
      MA_ratio_10_50  SMA(10) / SMA(50)

    Momentum indicators
      RSI_14        14-day Wilder RSI
      MACD_line     MACD line (12/26 EMA diff)
      MACD_signal   9-day EMA of MACD line
      MACD_hist     MACD line − signal

    Volume
      vol_change_5d  (today_vol − 5d_avg_vol) / 5d_avg_vol
                     Unusual volume often precedes directional moves.
    """
    close  = df["Close"].astype(float)
    volume = df["Volume"].astype(float)

    feats   = pd.DataFrame(index=df.index)
    log_ret = np.log(close / close.shift(1))

    # ── Returns ───────────────────────────────────────────────
    feats["ret_1d"]  = log_ret
    feats["ret_5d"]  = np.log(close / close.shift(5))
    feats["ret_10d"] = np.log(close / close.shift(10))
    feats["ret_21d"] = np.log(close / close.shift(21))

    # ── Volatility ────────────────────────────────────────────
    feats["vol_5d"]  = log_ret.rolling(5).std()  * np.sqrt(252)
    feats["vol_21d"] = log_ret.rolling(21).std() * np.sqrt(252)

    # ── Moving-average ratios ─────────────────────────────────
    sma5  = close.rolling(5).mean()
    sma10 = close.rolling(10).mean()
    sma21 = close.rolling(21).mean()
    sma50 = close.rolling(50).mean()

    feats["MA_ratio_5_21"]  = sma5  / sma21
    feats["MA_ratio_10_50"] = sma10 / sma50

    # ── RSI ───────────────────────────────────────────────────
    feats["RSI_14"] = compute_rsi(close, 14)

    # ── MACD ──────────────────────────────────────────────────
    macd_df = compute_macd(close)
    feats   = pd.concat([feats, macd_df], axis=1)

    # ── Volume change ─────────────────────────────────────────
    avg_vol_5 = volume.rolling(5).mean()
    feats["vol_change_5d"] = (volume - avg_vol_5) / avg_vol_5.replace(0, np.nan)

    # ── Ticker passthrough ────────────────────────────────────
    feats["Ticker"] = df["Ticker"].iloc[0] if "Ticker" in df.columns else "UNKNOWN"

    return feats


# ──────────────────────────────────────────────────────────────
# TARGET CONSTRUCTION  (uses future prices — labels only)
# ──────────────────────────────────────────────────────────────

def compute_target(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.Series:
    """
    Construct the binary prediction target.

    target[t] = 1  if close[t + horizon] > close[t]
                0  otherwise
                NaN for the last `horizon` rows (no realised outcome yet)

    LEAKAGE WARNING
    ---------------
    This function intentionally uses future prices — that is correct,
    because this is the LABEL.  The leakage guard is structural:
      1. This Series must NEVER be used as a feature input.
      2. The last `horizon` rows (NaN targets) are always dropped before
         any training or validation step (see build_dataset()).

    Why np.where instead of .astype(int)
    --------------------------------------
    (NaN > close) evaluates to False in pandas, so a naive .astype(int)
    would silently turn unlabelled rows into 0 ("down") instead of NaN.
    np.where() explicitly preserves NaN, making missing labels visible.
    """
    close  = df["Close"].astype(float)
    future = close.shift(-horizon)   # future close — labels only, never a feature

    values = np.where(future.isna(), np.nan, (future > close).astype(float))
    return pd.Series(values, index=df.index, name="target")


# ──────────────────────────────────────────────────────────────
# DATASET ASSEMBLY
# ──────────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> pd.DataFrame:
    """
    Combine features and target into a clean, NaN-free DataFrame.

    Steps:
      1. Compute features (backward-looking only).
      2. Compute target (forward-looking label).
      3. Drop the last `horizon` rows (no valid label yet).
      4. Drop any remaining NaN rows (rolling-window warm-up period).

    Returns
    -------
    pd.DataFrame with FEATURE_COLS + ["Ticker", "target"], no NaNs.
    """
    feats  = compute_features(df)
    target = compute_target(df, horizon)

    combined = pd.concat([feats, target], axis=1)

    # Remove unlabelled tail rows before dropna so they don't
    # masquerade as "down" signals if NaN slipped through.
    combined = combined.iloc[:-horizon]

    n_before = len(combined)
    combined.dropna(inplace=True)
    n_dropped = n_before - len(combined)

    return combined, n_dropped   # return drop count for logging
