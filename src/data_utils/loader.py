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



def load_daily_csv(
    csv_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load one daily CSV file and return cleaned data with log returns."""
    raw_df = pd.read_csv(csv_path)
    return preprocess_daily_prices(
        raw_df,
        rename_map=CHINESE_TO_ENGLISH,
        start_date=start_date,
        end_date=end_date,
    )
