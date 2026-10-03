"""
tests/test_features.py
======================
Unit tests for the feature engineering and target logic.

Imports directly from backend/features.py — the single source of truth
shared by the notebook, the tests, and the FastAPI backend.

Run from the repo root:
    python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on sys.path so `backend` package resolves
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.features import (  # noqa: E402
    FORECAST_HORIZON,
    build_dataset,
    compute_features,
    compute_macd,
    compute_rsi,
    compute_target,
)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def make_price_df(n: int = 120, start_price: float = 100.0,
                  drift: float = 0.001) -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame with a mild upward drift.
    Prices follow a geometric random walk so all indicators behave
    realistically (not flat, not monotone).
    """
    rng     = np.random.default_rng(seed=42)
    returns = rng.normal(loc=drift, scale=0.015, size=n)
    closes  = start_price * np.exp(np.cumsum(returns))

    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "Open":   closes * rng.uniform(0.99,  1.005, n),
        "High":   closes * rng.uniform(1.00,  1.015, n),
        "Low":    closes * rng.uniform(0.985, 1.00,  n),
        "Close":  closes,
        "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        "Ticker": "TEST",
    }, index=dates)
    df.index.name = "Date"
    return df


# ──────────────────────────────────────────────────────────────
# TARGET TESTS
# ──────────────────────────────────────────────────────────────

class TestTarget:
    def test_target_is_binary(self):
        """Valid (non-NaN) target values must be only 0 or 1."""
        df     = make_price_df(60)
        target = compute_target(df)
        valid  = target.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0}), \
            "Target must be 0 or 1 only"

    def test_target_length_matches_input(self):
        df     = make_price_df(60)
        target = compute_target(df)
        assert len(target) == len(df)

    def test_target_last_horizon_rows_are_nan(self):
        """
        The last FORECAST_HORIZON rows have no realised outcome yet.
        compute_target() must return NaN for them (not silently 0).

        Background: (NaN > x) evaluates to False in pandas, so a naive
        .astype(int) would turn unlabelled rows into 0, hiding the missing
        label.  We use np.where() in compute_target() to prevent this.
        """
        df     = make_price_df(60)
        target = compute_target(df)
        assert target.iloc[-FORECAST_HORIZON:].isna().all(), \
            f"Last {FORECAST_HORIZON} target values must be NaN"

    def test_target_value_is_correct(self):
        """Spot-check: target[0] == 1 iff close[5] > close[0]."""
        df    = make_price_df(60)
        close = df["Close"]
        target = compute_target(df)
        expected = int(close.iloc[FORECAST_HORIZON] > close.iloc[0])
        assert int(target.iloc[0]) == expected

    def test_target_trending_up_gives_all_ones(self):
        """Perfectly increasing price → every labelled row must be 1."""
        n      = 100
        dates  = pd.date_range("2020-01-02", periods=n, freq="B")
        closes = np.arange(100, 100 + n, dtype=float)
        df = pd.DataFrame({
            "Close": closes, "Open": closes, "High": closes,
            "Low": closes, "Volume": np.ones(n), "Ticker": "UP",
        }, index=dates)
        df.index.name = "Date"
        target = compute_target(df)
        valid  = target.dropna()
        assert valid.mean() == 1.0, \
            "All targets should be 1 for a perfectly increasing price series"

    def test_target_trending_down_gives_all_zeros(self):
        """Perfectly decreasing price → every labelled row must be 0."""
        n      = 100
        dates  = pd.date_range("2020-01-02", periods=n, freq="B")
        closes = np.arange(n, 0, -1, dtype=float)
        df = pd.DataFrame({
            "Close": closes, "Open": closes, "High": closes,
            "Low": closes, "Volume": np.ones(n), "Ticker": "DOWN",
        }, index=dates)
        df.index.name = "Date"
        target = compute_target(df)
        valid  = target.dropna()
        assert valid.mean() == 0.0, \
            "All targets should be 0 for a perfectly decreasing price series"


# ──────────────────────────────────────────────────────────────
# FEATURE TESTS
# ──────────────────────────────────────────────────────────────

class TestFeatures:
    def test_feature_columns_present(self):
        """All 13 expected feature columns must exist in the output."""
        df    = make_price_df(120)
        feats = compute_features(df)
        expected = [
            "ret_1d", "ret_5d", "ret_10d", "ret_21d",
            "vol_5d", "vol_21d",
            "MA_ratio_5_21", "MA_ratio_10_50",
            "RSI_14",
            "MACD_line", "MACD_signal", "MACD_hist",
            "vol_change_5d",
        ]
        for col in expected:
            assert col in feats.columns, f"Missing feature column: {col}"

    def test_no_leakage_ret_1d(self):
        """
        Leakage check: ret_1d[t] must equal ln(close[t] / close[t-1]).
        It must NOT reference close[t+1] or any future value.
        """
        df    = make_price_df(120)
        feats = compute_features(df)
        close = df["Close"]
        idx   = 50
        expected = np.log(close.iloc[idx] / close.iloc[idx - 1])
        assert abs(feats["ret_1d"].iloc[idx] - expected) < 1e-10

    def test_no_leakage_ret_5d(self):
        """ret_5d[t] must use close[t] and close[t-5] only."""
        df    = make_price_df(120)
        feats = compute_features(df)
        close = df["Close"]
        idx   = 50
        expected = np.log(close.iloc[idx] / close.iloc[idx - 5])
        assert abs(feats["ret_5d"].iloc[idx] - expected) < 1e-10

    def test_rsi_range(self):
        """RSI must stay within [0, 100]."""
        df  = make_price_df(120)
        rsi = compute_features(df)["RSI_14"].dropna()
        assert rsi.between(0, 100).all(), \
            f"RSI out of [0,100]: min={rsi.min():.2f}, max={rsi.max():.2f}"

    def test_ma_ratio_above_one_when_trending_up(self):
        """SMA(5) > SMA(21) for a perfectly increasing series after warm-up."""
        n      = 120
        dates  = pd.date_range("2020-01-02", periods=n, freq="B")
        closes = np.arange(100, 100 + n, dtype=float)
        df = pd.DataFrame({
            "Close": closes, "Open": closes, "High": closes,
            "Low": closes, "Volume": np.ones(n) * 1e6, "Ticker": "UP",
        }, index=dates)
        df.index.name = "Date"
        ratio = compute_features(df)["MA_ratio_5_21"].dropna()
        assert (ratio > 1).all()

    def test_vol_5d_is_nonnegative(self):
        """Volatility (std-dev) must always be >= 0."""
        vol = compute_features(make_price_df(120))["vol_5d"].dropna()
        assert (vol >= 0).all()

    def test_macd_hist_equals_line_minus_signal(self):
        """MACD_hist must exactly equal MACD_line - MACD_signal."""
        feats = compute_features(make_price_df(120))
        diff  = (feats["MACD_line"] - feats["MACD_signal"] - feats["MACD_hist"]).abs()
        assert diff.max() < 1e-10


# ──────────────────────────────────────────────────────────────
# BUILD DATASET TESTS
# ──────────────────────────────────────────────────────────────

class TestBuildDataset:
    def test_no_nan_in_output(self):
        ds, _ = build_dataset(make_price_df(120))
        assert ds.isnull().sum().sum() == 0

    def test_target_column_present(self):
        ds, _ = build_dataset(make_price_df(120))
        assert "target" in ds.columns

    def test_dataset_shorter_than_raw(self):
        df    = make_price_df(120)
        ds, _ = build_dataset(df)
        assert len(ds) < len(df)

    def test_target_is_binary_in_final_dataset(self):
        ds, _  = build_dataset(make_price_df(120))
        unique = set(ds["target"].unique())
        assert unique.issubset({0.0, 1.0}), \
            f"Unexpected target values: {unique}"

    def test_no_future_target_in_features(self):
        """
        LEAKAGE CHECK: no feature column name should contain 'future'.
        This is a structural guard to catch accidental label leakage.
        """
        ds, _       = build_dataset(make_price_df(120))
        feature_cols = [c for c in ds.columns if c not in ("target", "Ticker")]
        future_cols  = [c for c in feature_cols if "future" in c.lower()]
        assert len(future_cols) == 0, \
            f"Potentially leaked future columns: {future_cols}"


# ──────────────────────────────────────────────────────────────
# RSI UNIT TESTS
# ──────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_flat_price_is_nan_or_50(self):
        """A flat price series has no gains/losses → RSI is NaN or 50."""
        series = pd.Series([100.0] * 30)
        rsi    = compute_rsi(series, 14)
        valid  = rsi.dropna()
        assert valid.empty or valid.between(49.9, 50.1).all()

    def test_rsi_warmup_period(self):
        """First value should be NaN (insufficient history)."""
        series = pd.Series(np.random.randn(50).cumsum() + 100)
        rsi    = compute_rsi(series, 14)
        assert pd.isna(rsi.iloc[0])


# ──────────────────────────────────────────────────────────────
# MACD UNIT TESTS
# ──────────────────────────────────────────────────────────────

class TestMACD:
    def test_macd_columns(self):
        series = pd.Series(np.random.randn(100).cumsum() + 100)
        result = compute_macd(series)
        assert set(result.columns) == {"MACD_line", "MACD_signal", "MACD_hist"}

    def test_macd_hist_is_difference(self):
        series = pd.Series(np.random.randn(100).cumsum() + 100)
        result = compute_macd(series)
        diff   = (result["MACD_line"] - result["MACD_signal"] -
                  result["MACD_hist"]).abs()
        assert diff.max() < 1e-10
