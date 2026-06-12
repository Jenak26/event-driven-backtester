# tests/test_universe.py
"""Point-in-time universe loader: parsing, as-of selection, and graceful
fallback when the historical-constituents file is unavailable (offline-safe —
no network is touched; histories are passed in or written to tmp paths)."""
import pandas as pd
import pytest

from backtester.universe import (
    SECTOR_UNIVERSES,
    constituents_as_of,
    get_point_in_time_universe,
    load_sp500_constituents,
)


def _history():
    return pd.DataFrame({
        "date": pd.to_datetime(["2018-01-01", "2020-06-01"]),
        "tickers": ["AAPL,MSFT,XOM", "AAPL,MSFT,NVDA,XOM,CVX"],
    })


def test_sector_universes_has_four_sectors():
    assert set(SECTOR_UNIVERSES) == {"tech", "financials", "energy", "utilities"}
    # New sectors carry the requested names.
    assert "XOM" in SECTOR_UNIVERSES["energy"]
    assert "NEE" in SECTOR_UNIVERSES["utilities"]


def test_constituents_as_of_picks_latest_prior_snapshot():
    hist = _history()
    early = constituents_as_of(hist, "2019-01-01")
    assert early == {"AAPL", "MSFT", "XOM"}
    later = constituents_as_of(hist, "2021-01-01")
    assert "NVDA" in later and "CVX" in later


def test_constituents_before_first_snapshot_uses_earliest():
    hist = _history()
    assert constituents_as_of(hist, "2010-01-01") == {"AAPL", "MSFT", "XOM"}


def test_pit_universe_drops_names_not_yet_in_index():
    hist = _history()
    base = {"tech": ["AAPL", "MSFT", "NVDA"]}
    # NVDA only joins in the 2020 snapshot, so it is excluded as of 2019.
    pit = get_point_in_time_universe("tech", "2019-01-01", history=hist, base_universe=base)
    assert pit == ["AAPL", "MSFT"]
    # As of 2021 NVDA is in the index again.
    pit_later = get_point_in_time_universe("tech", "2021-01-01", history=hist, base_universe=base)
    assert "NVDA" in pit_later


def test_pit_universe_falls_back_when_history_missing():
    base = {"tech": ["AAPL", "MSFT", "NVDA"]}
    # history=None and no cache file -> load returns None -> fall back to full list.
    pit = get_point_in_time_universe(
        "tech", "2019-01-01",
        history=load_sp500_constituents(path="does_not_exist.csv", auto_download=False),
        base_universe=base,
    )
    assert pit == ["AAPL", "MSFT", "NVDA"]


def test_pit_universe_unknown_sector_raises():
    with pytest.raises(KeyError):
        get_point_in_time_universe("biotech", "2019-01-01", history=_history())


def test_load_parses_written_file(tmp_path):
    p = tmp_path / "sp500.csv"
    p.write_text("date,tickers\n2018-01-01,\"AAPL,MSFT\"\n2020-06-01,\"AAPL,MSFT,NVDA\"\n")
    df = load_sp500_constituents(path=str(p), auto_download=False)
    assert df is not None
    assert list(df.columns) == ["date", "tickers"]
    assert len(df) == 2
    # Sorted ascending by date.
    assert df["date"].is_monotonic_increasing


def test_load_missing_file_returns_none():
    assert load_sp500_constituents(path="nope_not_here.csv", auto_download=False) is None
