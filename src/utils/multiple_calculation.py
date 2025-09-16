"""
Fund Multiple Calculation Module

This module provides both dynamic and static multiple calculation functions for TVPI, DPI, and RVPI
from cashflow data with comprehensive validation and error handling.
"""

import pandas as pd
import numpy as np
from typing import Iterable

REQUIRED_COLS = {
    "fund_id",
    "transaction_date",
    "transaction_type",
    "transaction_amount",
    "cumulative_contribution",
    "cumulative_distribution",
}

def _check_required(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

def compute_multiple_dynamic(
    df: pd.DataFrame,
    value_flag: str = "Value",
    fund_col: str = "fund_id",
    date_col: str = "transaction_date",
    type_col: str = "transaction_type",
    nav_col: str = "transaction_amount",
    cum_contrib_col: str = "cumulative_contribution",
    cum_dist_col: str = "cumulative_distribution",
) -> pd.DataFrame:
    """
    Return TVPI/DPI/RVPI for **every Value row** (dynamic multiples).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input cashflow data
    value_flag : str, default "Value"
        Transaction type indicating NAV valuation
    fund_col : str, default "fund_id"
        Column name for fund identifier
    date_col : str, default "transaction_date"
        Column name for transaction date
    type_col : str, default "transaction_type"
        Column name for transaction type
    nav_col : str, default "transaction_amount"
        Column name for NAV amount (on Value rows)
    cum_contrib_col : str, default "cumulative_contribution"
        Column name for cumulative contributions
    cum_dist_col : str, default "cumulative_distribution"
        Column name for cumulative distributions
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: fund_id, transaction_date, tvpi, dpi, rvpi, multiple
        One row per Value transaction showing rolling multiples
    """
    _check_required(df, REQUIRED_COLS)

    val = (
        df.loc[df[type_col] == value_flag, [fund_col, date_col, nav_col, cum_contrib_col, cum_dist_col]]
          .sort_values([fund_col, date_col])
          .copy()
    )
    val["nav"] = val[nav_col]

    denom = val[cum_contrib_col].abs().replace(0, np.nan)
    val["dpi"]  = val[cum_dist_col] / denom
    val["rvpi"] = val["nav"] / denom
    val["tvpi"] = val["dpi"] + val["rvpi"]
    val["multiple"] = val["tvpi"]

    return val[[fund_col, date_col, "tvpi", "dpi", "rvpi", "multiple"]]

def compute_multiple_static(
    df: pd.DataFrame,
    **kwargs
) -> pd.DataFrame:
    """
    Return **one row per fund** – the last Value row's TVPI/DPI/RVPI (static multiples).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input cashflow data
    **kwargs
        Additional arguments passed to compute_multiple_dynamic
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: fund_id, tvpi, dpi, rvpi, multiple
        One row per fund showing final multiples
    """
    dyn = compute_multiple_dynamic(df, **kwargs)
    if dyn.empty:
        # Return empty DataFrame with correct columns if no Value transactions
        return pd.DataFrame(columns=["fund_id", "tvpi", "dpi", "rvpi", "multiple"])
    
    last = (
        dyn.sort_values(["fund_id", "transaction_date"])
           .groupby("fund_id", as_index=False)
           .tail(1)
           .drop(columns=["transaction_date"])
           .reset_index(drop=True)
    )
    return last

