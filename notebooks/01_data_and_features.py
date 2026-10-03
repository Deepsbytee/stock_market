"""
notebooks/01_data_and_features.py
==================================
Phase 1 orchestrator: downloads raw OHLCV data from Yahoo Finance,
delegates all feature/target logic to backend/features.py, and saves
processed CSVs to data/processed/.

Feature and target functions live in backend/features.py — that is the
single source of truth shared by this script, the tests, and the API.

Run from the repo root:
    python notebooks/01_data_and_features.py
"""

import logging
import sys
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Make the repo root importable so `backend` package resolves ──
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.features import (  # noqa: E402
    FEATURE_COLS,
    build_dataset,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN",
    "JPM",  "XOM",  "JNJ",  "WMT",
    "TSLA", "NVDA",
]

START_DATE = "2013-01-01"

# Dynamic end date: always downloads up to the latest available trading day.
# Do NOT hardcode this — yfinance will return data up to (but not including)
# the supplied end date, so passing tomorrow ensures we always get today's data.
def _today_str() -> str:
    return date.today().isoformat()

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")


# ──────────────────────────────────────────────────────────────
# DOWNLOAD
# ──────────────────────────────────────────────────────────────

def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV from Yahoo Finance.

    auto_adjust=True uses split/dividend-adjusted close prices so that
    stock splits don't create artificial price jumps in the data.
    """
    log.info(f"Downloading {ticker} ...")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns (some yfinance versions return them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df["Ticker"] = ticker
    return df[["Open", "High", "Low", "Close", "Volume", "Ticker"]]


def download_all(tickers=TICKERS, start=START_DATE, cache=True) -> dict:
    """
    Download all tickers; cache CSVs in data/raw/ to avoid hitting Yahoo
    Finance on every run.  Delete data/raw/*.csv to force a fresh download.
    """
    end = _today_str()   # always dynamic — never hardcoded
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = {}

    for ticker in tickers:
        cache_path = RAW_DIR / f"{ticker}.csv"
        if cache and cache_path.exists():
            log.info(f"Loading {ticker} from cache ...")
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        else:
            df = download_ticker(ticker, start, end)
            df.to_csv(cache_path)

        data[ticker] = df
        log.info(f"  {ticker}: {len(df)} rows  "
                 f"({df.index.min().date()} -> {df.index.max().date()})")

    return data


# ──────────────────────────────────────────────────────────────
# BUILD PROCESSED DATASETS
# ──────────────────────────────────────────────────────────────

def build_all_datasets(raw_data: dict, save: bool = True) -> dict:
    """
    Call build_dataset() from backend/features.py for every ticker and
    optionally persist the results to data/processed/.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {}

    for ticker, df in raw_data.items():
        log.info(f"\nProcessing {ticker} ...")
        ds, n_dropped = build_dataset(df)
        datasets[ticker] = ds

        log.info(f"  Dropped {n_dropped} NaN rows ({len(ds)} rows remain)")

        if save:
            out = PROC_DIR / f"{ticker}_features.csv"
            ds.to_csv(out)
            log.info(f"  Saved -> {out}")

        up_pct = ds["target"].mean() * 100
        log.info(f"  {ticker}: {len(ds)} rows | "
                 f"target=1 (up): {up_pct:.1f}%  "
                 f"target=0 (down): {100 - up_pct:.1f}%")

    return datasets


# ──────────────────────────────────────────────────────────────
# EDA SUMMARY
# ──────────────────────────────────────────────────────────────

def summarise(datasets: dict) -> None:
    """Print a concise summary table to stdout."""
    print("\n" + "=" * 65)
    print(f"{'TICKER':<8} {'ROWS':>6} {'FEATURES':>9} "
          f"{'TARGET=1%':>10} {'DATE RANGE'}")
    print("=" * 65)

    for ticker, ds in datasets.items():
        date_range = (f"{ds.index.min().date()} -> "
                      f"{ds.index.max().date()}")
        print(f"{ticker:<8} {len(ds):>6} {len(FEATURE_COLS):>9} "
              f"{ds['target'].mean()*100:>9.1f}%  {date_range}")

    print("=" * 65)
    print(f"\nFeatures ({len(FEATURE_COLS)}):")
    for c in FEATURE_COLS:
        print(f"  - {c}")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Market Signal Lab - Phase 1: Data & Feature Engineering")
    print(f"  Downloading up to: {_today_str()}")
    print("=" * 65)

    raw_data = download_all()
    datasets = build_all_datasets(raw_data)
    summarise(datasets)

    print("\n  Phase 1 complete. Processed CSVs saved to data/processed/")
    print("    Next: Phase 2 - Baselines & Model Training")
