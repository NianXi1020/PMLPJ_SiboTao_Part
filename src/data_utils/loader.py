"""Load daily ETF datasets with unified preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data_utils.preprocess import preprocess_daily_prices


CHINESE_TO_ENGLISH = {
    "日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量(股)": "volume",
    "涨跌额": "price_change",
    "涨跌幅(%)": "pct_change",
}

COMMON_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "gbk", "gb18030", "big5")


def read_csv_with_fallback_encodings(csv_path: Path) -> pd.DataFrame:
    """Read CSV with common UTF/Chinese encodings.

    Raises:
        UnicodeDecodeError: If no encoding can decode the file.
    """
    last_error: UnicodeDecodeError | None = None

    for encoding in COMMON_CSV_ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as err:
            last_error = err

    if last_error is None:
        # Defensive fallback for non-decoding errors.
        return pd.read_csv(csv_path)

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        (
            f"Unable to decode {csv_path} with tried encodings: "
            f"{', '.join(COMMON_CSV_ENCODINGS)}. "
            f"Original error: {last_error.reason}"
        ),
    )


def load_daily_csv(
    csv_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load one daily CSV file and return cleaned data with log returns."""
    raw_df = read_csv_with_fallback_encodings(csv_path)
    return preprocess_daily_prices(
        raw_df,
        rename_map=CHINESE_TO_ENGLISH,
        start_date=start_date,
        end_date=end_date,
    )
