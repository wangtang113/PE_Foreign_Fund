"""
IRR Calculation Module

This module provides functions for calculating Internal Rate of Return (IRR) from cashflow data.
Includes both static IRR (single value per fund) and dynamic IRR (per Value date).
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
import warnings
from typing import Optional, List, Dict

def compute_irr_dynamic(
    df: pd.DataFrame,
    periods_per_year: int = 4,
    value_flag: str = "Value",
    fund_col: str = "fund_id",
    date_col: str = "transaction_date",
    type_col: str = "transaction_type",
    nav_col: str = "transaction_amount",
    net_cf_col: str = "net_cashflow",
) -> pd.DataFrame:
    """
    Compute *dynamic* IRR per fund: you get an IRR for EVERY 'Value' date,
    using only the cash flows observed up to that date, plus the NAV at that date.

    For each fund:
      - Keep only 'Value' rows, sorted by date.
      - Build period cash flows as the difference in the cumulative net cash flow
        between consecutive 'Value' rows (first row uses its own net_cashflow).
      - For each date t, compute IRR on cash flows up to t, adding NAV at t to the
        last cash flow.
      - Convert quarterly IRR to annual by (1 + r_q) ** periods_per_year - 1.

    Returns:
        One row per 'Value' transaction with:
          irr_quarterly_dynamic, irr_annual_dynamic, n_periods_used, error_reason
    """

    # ---- Defensive checks
    required = {fund_col, date_col, type_col, nav_col, net_cf_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out_cols = [
        fund_col, date_col, nav_col, net_cf_col,
        "irr_quarterly_dynamic", "irr_annual_dynamic",
        "n_periods_used", "error_reason"
    ]

    # ---- Prep
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([fund_col, date_col]).reset_index(drop=True)

    results: List[pd.DataFrame] = []
    fund_warnings: List[str] = []

    # ---- Per fund
    for fid, g in df.groupby(fund_col, sort=False):
        # Only Value rows (where we want to display IRR on each date)
        val = (
            g.loc[g[type_col] == value_flag, [fund_col, date_col, nav_col, net_cf_col]]
             .sort_values(date_col)
             .copy()
        )

        if val.empty:
            # nothing to compute – skip
            continue

        # Period cash flows = diff in cumulative net cash flow between Value dates
        val["delta_cf"] = val[net_cf_col].diff()
        val.loc[val.index[0], "delta_cf"] = val.iloc[0][net_cf_col]

        # Now compute a rolling IRR for each row
        irr_q_list, irr_a_list, n_list, err_list = [], [], [], []

        # Build an expanding list of CFs
        expanding_cfs: List[float] = []

        for i, row in val.iterrows():
            # append this period's CF
            expanding_cfs.append(row["delta_cf"])

            # copy to avoid mutation when we add NAV
            cfs = expanding_cfs.copy()
            cfs[-1] += row[nav_col]  # NAV at *this* date

            # Validation
            if len(cfs) < 2:
                irr_q = np.nan
                irr_a = np.nan
                err = "Insufficient cash flows (need at least 2)"
            else:
                has_pos = any(cf > 0 for cf in cfs)
                has_neg = any(cf < 0 for cf in cfs)

                if not (has_pos and has_neg):
                    irr_q = np.nan
                    irr_a = np.nan
                    err = "No sign change in cash flows"
                else:
                    try:
                        irr_q = npf.irr(cfs)
                        irr_a = (1 + irr_q) ** periods_per_year - 1 if not np.isnan(irr_q) else np.nan
                        err = None if not np.isnan(irr_q) else "IRR is NaN"
                    except Exception as e:
                        irr_q = np.nan
                        irr_a = np.nan
                        err = f"IRR failed: {e}"

            irr_q_list.append(irr_q)
            irr_a_list.append(irr_a)
            n_list.append(len(cfs))
            err_list.append(err)

        val["irr_quarterly_dynamic"] = irr_q_list * 100 # convert to percentage
        val["irr_annual_dynamic"]    = irr_a_list * 100 # convert to percentage
        val["n_periods_used"]        = n_list
        val["error_reason"]          = err_list

        # capture warnings for the fund
        bad = [e for e in err_list if e is not None]
        if bad:
            fund_warnings.append(f"Fund {fid}: {bad[-1]}")

        results.append(val[out_cols])

    if fund_warnings:
        warnings.warn(f"Dynamic IRR issues for {len(fund_warnings)} funds. "
                      f"First few: {fund_warnings[:5]}")

    if not results:
        # Return empty DataFrame with correct columns if no results
        return pd.DataFrame(columns=out_cols)
    
    return pd.concat(results, axis=0).sort_values([fund_col, date_col]).reset_index(drop=True)

def compute_irr_static(df: pd.DataFrame, periods_per_year: int = 4) -> pd.DataFrame:
    """
    Compute per-fund IRR with robust error handling and validation.
    
    This function calculates the Internal Rate of Return for each fund based on 
    cash flow data, using quarterly periods by default and converting to annual IRR.
    
    Args:
        df: DataFrame with columns: fund_id, firm_id, transaction_date, transaction_type, 
            transaction_amount, net_cashflow
        periods_per_year: Number of periods per year for annualization (default: 4 for quarterly)
    
    Returns:
        DataFrame with columns: fund_id, firm_id, irr_quarterly, irr_annual
    """
    
    # Defensive checks for required columns
    required_columns = ['fund_id', 'firm_id', 'transaction_date', 'transaction_type', 
                       'transaction_amount', 'net_cashflow']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    def compute_fund_irr(fund_df):
        """Calculate IRR for a single fund"""
        try:
            # Sort by transaction date to ensure proper chronological order
            fund_df = fund_df.sort_values('transaction_date').copy()
            
            # Get the last NAV (transaction_amount for Value transactions)
            nav_rows = fund_df[fund_df['transaction_type'] == 'Value']
            if not nav_rows.empty:
                final_nav = nav_rows.iloc[-1]['transaction_amount']
            else:
                final_nav = 0
            
            # Calculate cash flows as differences in net_cashflow
            cash_flows = fund_df['net_cashflow'].diff().fillna(fund_df['net_cashflow'].iloc[0])
            
            # Add final NAV to the last cash flow
            if len(cash_flows) > 0:
                cash_flows.iloc[-1] += final_nav
            
            # Convert to list for numpy_financial
            cash_flows_list = cash_flows.tolist()
            
            # Check for sufficient data points
            if len(cash_flows_list) < 2:
                return np.nan, f"Insufficient cash flows (only {len(cash_flows_list)} periods)"
            
            # Check for sign change (necessary for IRR calculation)
            has_positive = any(cf > 0 for cf in cash_flows_list)
            has_negative = any(cf < 0 for cf in cash_flows_list)
            
            if not (has_positive and has_negative):
                return np.nan, "No sign change in cash flows"
            
            # Calculate IRR using numpy_financial
            quarterly_irr = npf.irr(cash_flows_list)
            
            if np.isnan(quarterly_irr):
                return np.nan, "IRR calculation returned NaN"
            
            # Convert to annual IRR
            annual_irr = (1 + quarterly_irr) ** periods_per_year - 1
            
            return quarterly_irr, annual_irr, None
            
        except Exception as e:
            return np.nan, np.nan, f"IRR calculation failed: {str(e)}"
    
    # Group by fund_id and calculate IRR for each
    results = []
    fund_warnings = []
    
    for fund_id, fund_group in df.groupby('fund_id'):
        # Get firm_id (should be constant within fund)
        firm_id = fund_group['firm_id'].iloc[0]
        
        # Calculate IRR
        irr_result = compute_fund_irr(fund_group)
        
        if len(irr_result) == 3:
            quarterly_irr, annual_irr, error = irr_result
            if error:
                fund_warnings.append(f"Fund {fund_id}: {error}")
        else:
            quarterly_irr, error = irr_result
            annual_irr = np.nan
            fund_warnings.append(f"Fund {fund_id}: {error}")
        
        results.append({
            'fund_id': fund_id,
            'firm_id': firm_id,
            'irr_quarterly': quarterly_irr * 100,
            'irr_annual': annual_irr * 100
        })
    
    # Issue warnings for problematic funds
    if fund_warnings:
        warnings.warn(f"IRR calculation issues for {len(fund_warnings)} funds: " + 
                     "; ".join(fund_warnings[:5]))
    
    return pd.DataFrame(results)

