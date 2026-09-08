"""Data quality checks from FR-1.6."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from tradingbot.data.feed import rows_to_frame
from tradingbot.data.validators import find_gaps, mark_gaps, validate_ohlcv

from .test_data_feed import STEP, synthetic_rows

START = datetime(2024, 1, 1, tzinfo=UTC)


def clean_frame(count: int = 20) -> pd.DataFrame:
    return rows_to_frame(synthetic_rows(count))


def with_hole(count: int = 20, drop: slice = slice(5, 8)) -> pd.DataFrame:
    frame = clean_frame(count)
    return frame.drop(frame.index[drop])


class TestCleanData:
    def test_no_issues_reported(self) -> None:
        report = validate_ohlcv(clean_frame(), "BTC/USDT", "4h")
        assert report.is_valid
        assert report.errors == []
        assert report.warnings == []
        assert report.rows == 20
        assert report.start == START
        assert "OK" in report.render()

    def test_empty_dataset_is_invalid(self) -> None:
        report = validate_ohlcv(clean_frame(0), "BTC/USDT", "4h")
        assert not report.is_valid
        assert "empty" in report.errors[0]


class TestIntegrityErrors:
    def test_duplicate_timestamps(self) -> None:
        frame = clean_frame(5)
        frame = pd.concat([frame, frame.iloc[[2]]]).sort_index()
        report = validate_ohlcv(frame, "BTC/USDT", "4h")
        assert report.duplicate_timestamps == 1
        assert not report.is_valid

    def test_unsorted_index(self) -> None:
        frame = clean_frame(5)
        frame = frame.iloc[[0, 2, 1, 3, 4]]
        report = validate_ohlcv(frame, "BTC/USDT", "4h")
        assert report.unsorted is True
        assert not report.is_valid

    def test_nan_values(self) -> None:
        frame = clean_frame(5)
        frame.iloc[2, frame.columns.get_loc("close")] = np.nan
        report = validate_ohlcv(frame, "BTC/USDT", "4h")
        assert report.nan_rows == 1
        assert not report.is_valid

    def test_high_below_close_is_flagged(self) -> None:
        frame = clean_frame(5)
        frame.iloc[1, frame.columns.get_loc("high")] = frame.iloc[1]["close"] - 1.0
        report = validate_ohlcv(frame, "BTC/USDT", "4h")
        assert report.invalid_ohlc == 1
        assert not report.is_valid

    def test_low_above_open_is_flagged(self) -> None:
        frame = clean_frame(5)
        frame.iloc[3, frame.columns.get_loc("low")] = frame.iloc[3]["open"] + 1.0
        assert validate_ohlcv(frame, "BTC/USDT", "4h").invalid_ohlc == 1

    def test_non_positive_price(self) -> None:
        frame = clean_frame(5)
        frame.iloc[0, frame.columns.get_loc("low")] = -1.0
        report = validate_ohlcv(frame, "BTC/USDT", "4h")
        assert report.non_positive == 1
        assert not report.is_valid


class TestGaps:
    def test_gap_is_found_and_measured(self) -> None:
        report = validate_ohlcv(with_hole(), "BTC/USDT", "4h", max_gap_bars=3)
        assert len(report.gaps) == 1
        assert report.gaps[0].missing_bars == 3
        assert report.missing_bars == 3
        assert report.is_valid  # gaps are a warning, not an error
        assert report.warnings

    def test_gap_above_the_limit_is_called_out(self) -> None:
        report = validate_ohlcv(with_hole(drop=slice(5, 12)), "BTC/USDT", "4h", max_gap_bars=3)
        assert report.max_gap_bars == 7
        assert "limit 3" in report.warnings[0]

    def test_find_gaps_on_contiguous_series(self) -> None:
        assert find_gaps(clean_frame().index, "4h") == []

    def test_find_gaps_needs_two_points(self) -> None:
        assert find_gaps(clean_frame(1).index, "4h") == []

    def test_render_truncates_long_gap_lists(self) -> None:
        frame = clean_frame(60)
        frame = frame.drop(frame.index[list(range(1, 50, 2))])
        report = validate_ohlcv(frame, "BTC/USDT", "4h")
        rendered = report.render()
        assert "more gaps" in rendered


class TestMarkGaps:
    def test_bar_after_a_gap_is_flagged(self) -> None:
        frame = mark_gaps(with_hole(), "4h")
        flagged = frame.index[frame["is_gap"]]
        assert len(flagged) == 1
        assert flagged[0] == START + 8 * STEP

    def test_no_synthetic_bars_are_inserted(self) -> None:
        original = with_hole()
        assert len(mark_gaps(original, "4h")) == len(original)

    def test_contiguous_series_has_no_flags(self) -> None:
        assert not mark_gaps(clean_frame(), "4h")["is_gap"].any()

    def test_short_series_is_returned_untouched(self) -> None:
        frame = mark_gaps(clean_frame(1), "4h")
        assert len(frame) == 1
        assert frame["is_gap"].tolist() == [False]


class TestReportRendering:
    def test_errors_and_warnings_are_visible(self) -> None:
        frame = with_hole()
        frame.iloc[0, frame.columns.get_loc("close")] = np.nan
        rendered = validate_ohlcv(frame, "BTC/USDT", "4h").render()
        assert "ERROR" in rendered
        assert "WARNING" in rendered
        assert "BTC/USDT 4h" in rendered

    def test_header_contains_the_window(self) -> None:
        rendered = validate_ohlcv(clean_frame(), "ETH/USDT", "1h").render()
        assert "2024-01-01 00:00" in rendered
        assert pytest.approx(20) == validate_ohlcv(clean_frame(), "ETH/USDT", "1h").rows
