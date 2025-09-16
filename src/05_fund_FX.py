"""
04_fund_FX.py - Merge FX forwards into fund buyout deals panel

This script merges FX forward data with fund buyout deals data to calculate
FX spot and forward rates with proper time lag to avoid look-ahead bias.

## Key Assumptions:
1. **Base Currency**: All FX rates in FX_forward.xlsx are quoted vs USD
2. **Time Lag**: FX data is lagged by 1 quarter relative to deal quarter to avoid look-ahead
3. **Cross Rates**: Calculated using USD as the common denominator

## Methodology:

### FX Data Processing:
- Parse quarterly periods from FX forward dates
- Clean column names using regex: `[A-Z]{3} (SP|5Y)` patterns
- Remove duplicate .1, .2 suffixes from column names
- Pivot to separate SP (spot) and 5Y (forward) rate tables

### Deal Data Processing:
- Extract Deal Quarter from DEAL DATE
- Create FX Quarter = Deal Quarter - 1 (lag by 1 quarter)
- Clean and standardize currency codes (uppercase, strip whitespace)

### Cross Rate Calculation:
Given fund_currency and deal_currency, calculate exchange rates:

```
if deal_currency == "USD":
    rate = 1.0 / fund_rate_vs_usd
elif fund_currency == "USD":
    rate = deal_rate_vs_usd
else:
    rate = deal_rate_vs_usd / fund_rate_vs_usd
```

### Output:
- `SP rate`: Spot exchange rate (deal_currency per fund_currency)
- `Forward 5Y rate`: 5-year forward exchange rate
- `Currency Pair`: String format "{deal_currency} {fund_currency}"

## Usage:

### Command Line:
```bash
python 04_fund_FX.py
python 04_fund_FX.py --fx-path data/FX_forward.xlsx --deals-path data/deals.csv --out-path output.csv
```

### Programmatic:
```python
from src.fund_FX import main
result_df = main(fx_path="Input_data/FX_forward.xlsx", 
                 deals_path="Output_data/fund_buyout_deals.csv",
                 out_path="Output_data/fund_buyout_deals_fx.csv")
```

## Data Quality:
- Validates no forward-looking bias (FX Quarter < Deal Quarter)
- Reports match rates for FX data availability
- Handles missing currency data gracefully (NaN values)
- Processes cross-rates for 400+ unique currency pairs

"""

from __future__ import annotations
from multiprocessing import allow_connection_pickling

import pandas as pd
import numpy as np
import re
import argparse
from pathlib import Path
from pandas import DataFrame, Series
from typing import Final
from utils.dataclean_utils import group_weighted_mean, group_equal_mean
from utils.winsorize_utils import winsorize



def load_and_clean_rer_data(rer_path: str | Path = "Output_data/RER_transformed.csv") -> DataFrame:
    """
    Load and clean RER data, returning DataFrame indexed by Quarter with currency columns.
    RER data is already in the correct format (quarters as rows, currencies as columns).
    
    Args:
        rer_path: Path to RER transformed CSV file
        
    Returns:
        DataFrame indexed by Quarter with currency columns containing RER values
    """
    # Load RER data
    rer_raw = pd.read_csv(rer_path)
    
    # Parse Quarter to quarterly periods
    rer_raw['Quarter'] = pd.to_datetime(rer_raw['Date']).dt.to_period('Q')
    
    # Set Quarter as index and drop Date column
    rer_clean = rer_raw.set_index('Quarter').drop(columns=['Date'])
    
    # Add USD column with value 1.0 (RER is relative to USD)
    rer_clean['USD'] = 1.0
    
    return rer_clean


def load_and_clean_fx_data(fx_path: str | Path = "Output_data/FX_forward.csv") -> tuple[DataFrame, DataFrame]:
    """
    Load and clean FX forward data, returning separate SP and 5Y DataFrames.
    
    Args:
        fx_path: Path to FX forward Excel file
        
    Returns:
        Tuple of (fx_sp, fx_5y) DataFrames indexed by Quarter with currency columns
    """
    # Load FX data
    fx_raw = pd.read_csv(fx_path)
    
    # Parse Date to quarterly periods
    fx_raw['Quarter'] = pd.to_datetime(fx_raw['Date']).dt.to_period('Q')
    
    # Clean column names: keep only [A-Z]{3} (SP|5Y) columns, strip .1, .2 suffixes
    fx_cols = fx_raw.columns.tolist()
    clean_cols = {}
    cols_to_keep = ['Date', 'Quarter']
    
    for col in fx_cols:
        if col in ['Date', 'Quarter']:
            clean_cols[col] = col
            continue
            
        # Match pattern: CCY_SP or CCY_5Y (underscore format)
        match = re.match(r'^([A-Z]{3})_(SP|5Y)$', col)
        if match:
            currency, tenor = match.groups()
            clean_name = f"{currency}_{tenor}"
            
            # Only keep the first occurrence of each clean name to avoid duplicates
            if clean_name not in clean_cols.values():
                clean_cols[col] = clean_name
                cols_to_keep.append(col)
    
    # Select only the first occurrence of each column type and rename
    fx_selected = fx_raw[cols_to_keep].copy()
    fx_clean = fx_selected.rename(columns=clean_cols)
    
    # Convert to long format then pivot to separate SP and 5Y tables
    fx_long_data = []
    
    for col in fx_clean.columns:
        if col in ['Date', 'Quarter']:
            continue
            
        parts = col.split('_')
        if len(parts) == 2:
            currency, tenor = parts
            for _, row in fx_clean.iterrows():
                fx_long_data.append({
                    'Quarter': row['Quarter'],
                    'Currency': currency,
                    'Tenor': tenor,
                    'Rate': row[col]
                })
    
    fx_long = pd.DataFrame(fx_long_data)
    
    # Remove duplicates before pivoting (take last value if duplicates exist)
    fx_long = fx_long.drop_duplicates(subset=['Quarter', 'Currency', 'Tenor'], keep='last')
    
    # Split into SP and 5Y tables
    fx_sp = (fx_long[fx_long['Tenor'] == 'SP']
             .pivot(index='Quarter', columns='Currency', values='Rate'))
    
    fx_5y = (fx_long[fx_long['Tenor'] == '5Y']
             .pivot(index='Quarter', columns='Currency', values='Rate'))
    
    return fx_sp, fx_5y





def load_fund_buyout_deals(deals_path: str | Path = "Output_data/fund_buyout_deals.csv") -> DataFrame:
    """
    Load fund buyout deals data and prepare for FX merge.
    
    Args:
        deals_path: Path to fund buyout deals CSV file
        
    Returns:
        DataFrame with Deal Quarter, FX Quarter, and cleaned currency codes
    """
    deals = pd.read_csv(deals_path)
    
    # Convert DEAL DATE to datetime and extract quarterly periods
    deals['DEAL DATE'] = pd.to_datetime(deals['DEAL DATE'])
    deals['Deal Quarter'] = deals['DEAL DATE'].dt.to_period('Q')
    
    # Create FX Quarter (lagged by 1 quarter to avoid look-ahead)
    deals['FX Quarter'] = deals['Deal Quarter'] - 1
    
    # Clean and uppercase currency codes
    deals['Deal Currency'] = deals['DEAL CURRENCY'].str.strip().str.upper()
    deals['Fund Currency'] = deals['FUND CURRENCY'].str.strip().str.upper()

    return deals


def merge_fx_rates(deals: DataFrame, fx_sp: DataFrame, fx_5y: DataFrame) -> DataFrame:
    """
    Merge FX spot and forward rates into deals data using vectorized operations.
    Removes deals where currencies are not covered in FX data.
    Adds:
      - sp_deal_5y, sp_fund_5y: spot rates looked up at FX Quarter + 5 years (20 quarters).
    
    Args:
        deals: Fund buyout deals DataFrame
        fx_sp: FX spot rates DataFrame (Quarter x Currency)
        fx_5y: FX 5Y forward rates DataFrame (Quarter x Currency)
        
    Returns:
        DataFrame with deals filtered to only supported currencies and FX rates added
    """
    # Get currencies available in FX data
    fx_currencies = set(fx_sp.columns.tolist())
    fx_currencies.add('USD')  # USD is the base currency (implicit in all rates)
    
    print(f"FX data covers {len(fx_currencies)} currencies: {sorted(fx_currencies)}")
    
    # Apply currency standardization for common mismatches
    deals = deals.copy()
    deals['Deal Currency'] = deals['Deal Currency'].replace({'RMB': 'CNY'})
    deals['Fund Currency'] = deals['Fund Currency'].replace({'RMB': 'CNY'})
    
    # Check coverage before filtering
    original_count = len(deals)
    deal_currency_covered = deals['Deal Currency'].isin(fx_currencies)
    fund_currency_covered = deals['Fund Currency'].isin(fx_currencies)
    both_covered = deal_currency_covered & fund_currency_covered
    
    print(f"Currency coverage analysis:")
    print(f"  Original deals: {original_count:,}")
    print(f"  Deal currency covered: {deal_currency_covered.sum():,} ({deal_currency_covered.mean():.1%})")
    print(f"  Fund currency covered: {fund_currency_covered.sum():,} ({fund_currency_covered.mean():.1%})")
    print(f"  Both currencies covered: {both_covered.sum():,} ({both_covered.mean():.1%})")
    
    # Show uncovered currencies
    uncovered_deal_currencies = deals[~deal_currency_covered]['Deal Currency'].value_counts()
    uncovered_fund_currencies = deals[~fund_currency_covered]['Fund Currency'].value_counts()
    
    if len(uncovered_deal_currencies) > 0:
        print(f"  Uncovered deal currencies: {dict(uncovered_deal_currencies.head(5))}")
    if len(uncovered_fund_currencies) > 0:
        print(f"  Uncovered fund currencies: {dict(uncovered_fund_currencies.head(5))}")
    
    # Filter deals to only those with both currencies covered in FX data
    deals = deals[both_covered].copy()
    print(f"  Deals after filtering: {len(deals):,} (removed {original_count - len(deals):,})")
    
    # Create lookup tables by stacking currency columns
    sp_lookup = (fx_sp.stack()
                 .rename("sp")
                 .reset_index())
    sp_lookup.columns = ["FX Quarter", "Currency", "sp"]
    
    fwd5_lookup = (fx_5y.stack()
                   .rename("fwd5")
                   .reset_index())
    fwd5_lookup.columns = ["FX Quarter", "Currency", "fwd5"]

    # Merge spot rates for deal and fund currencies at current FX Quarter
    deals = deals.merge(
        sp_lookup.rename(columns={"Currency": "Deal Currency", "sp": "sp_deal"}),
        on=["FX Quarter", "Deal Currency"], 
        how="left"
    )
    
    deals = deals.merge(
        sp_lookup.rename(columns={"Currency": "Fund Currency", "sp": "sp_fund"}),
        on=["FX Quarter", "Fund Currency"], 
        how="left"
    )
    
    # Merge 5Y forward rates for deal and fund currencies (forward_fx usage)
    deals = deals.merge(
        fwd5_lookup.rename(columns={"Currency": "Deal Currency", "fwd5": "fwd5_deal"}),
        on=["FX Quarter", "Deal Currency"], 
        how="left"
    )
    
    deals = deals.merge(
        fwd5_lookup.rename(columns={"Currency": "Fund Currency", "fwd5": "fwd5_fund"}),
        on=["FX Quarter", "Fund Currency"], 
        how="left"
    )
    
    # ---------- NEW: future spot at FX Quarter + 5Y (20 quarters)
    # Ensure FX Quarter is a quarterly Period dtype
    if not isinstance(deals["FX Quarter"].dtype, pd.PeriodDtype):
        deals["FX Quarter"] = deals["FX Quarter"].astype("period[Q]")
    
    deals["FX Quarter 5Y"] = deals["FX Quarter"] + 20  # five years ahead (quarterly)
    
    # Merge future spot for deal and fund currencies
    deals = deals.merge(
        sp_lookup.rename(columns={"Currency": "Deal Currency", "sp": "sp_deal_5y"}),
        left_on=["FX Quarter 5Y", "Deal Currency"],
        right_on=["FX Quarter", "Deal Currency"],
        how="left",
        suffixes=("", "_drop")
    )
    deals = deals.merge(
        sp_lookup.rename(columns={"Currency": "Fund Currency", "sp": "sp_fund_5y"}),
        left_on=["FX Quarter 5Y", "Fund Currency"],
        right_on=["FX Quarter", "Fund Currency"],
        how="left",
        suffixes=("", "_drop2")
    )
    
    # Clean up any duplicated merge keys (if created)
    drop_cols = [c for c in deals.columns if c.endswith("_drop") or c.endswith("_drop2")]
    if drop_cols:
        deals = deals.drop(columns=drop_cols)



    return deals


def merge_rer_rates(deals: DataFrame, rer: DataFrame) -> DataFrame:
    """
    Merge RER rates into deals data using vectorized operations.
    RER is defined relative to USD, so USD always has RER = 1.0.
    
    Args:
        deals: Fund buyout deals DataFrame (must have FX Quarter column)
        rer: RER DataFrame indexed by Quarter with currency columns
        
    Returns:
        DataFrame with RER rates added for deal and fund currencies
    """
    # Get currencies available in RER data
    rer_currencies = set(rer.columns.tolist())
    
    print(f"RER data covers {len(rer_currencies)} currencies")
    
    # Create lookup table by stacking currency columns
    rer_lookup = (rer.stack()
                  .rename("rer")
                  .reset_index())
    rer_lookup.columns = ["FX Quarter", "Currency", "rer"]
    
    # Merge RER rates for deal and fund currencies at current FX Quarter
    deals = deals.merge(
        rer_lookup.rename(columns={"Currency": "Deal Currency", "rer": "rer_deal"}),
        on=["FX Quarter", "Deal Currency"], 
        how="left"
    )
    
    deals = deals.merge(
        rer_lookup.rename(columns={"Currency": "Fund Currency", "rer": "rer_fund"}),
        on=["FX Quarter", "Fund Currency"], 
        how="left"
    )
    
    # Fill missing RER values with 1.0 for USD (since RER is relative to USD)
    usd_deal_mask = (deals['Deal Currency'] == 'USD') & deals['rer_deal'].isna()
    usd_fund_mask = (deals['Fund Currency'] == 'USD') & deals['rer_fund'].isna()
    
    deals.loc[usd_deal_mask, 'rer_deal'] = 1.0
    deals.loc[usd_fund_mask, 'rer_fund'] = 1.0
    
    return deals


def calculate_fx_rates(deals: DataFrame) -> DataFrame:
    """
    Calculate:
      - SP rate (current cross spot)
      - Forward 5Y rate (cross forward)
      - USD SP (deal vs USD)
      - Currency Pair
      - Deal forward_fx (expected FX measure from forward/spot)
      - SP rate 5Y (cross spot 5 years later)
      - Deal realized_fx (actual FX measure from future spot / current spot)
    
    Conversion rule (all FX rates are vs USD):
    - If both currencies are USD: rate = 1.0 (no conversion)
    - If deal currency is USD and fund currency is not: rate = 1 / fund_rate
    - If fund currency is USD and deal currency is not: rate = deal_rate  
    - Otherwise (both non-USD): rate = deal_rate / fund_rate
    
    Args:
        deals: DataFrame with sp_deal, sp_fund, fwd5_deal, fwd5_fund, sp_deal_5y, sp_fund_5y columns
        
    Returns:
        DataFrame with all FX rates and realized_fx/forward_fx columns added
    """
    deals = deals.copy()
    
    # Boolean masks for USD currencies
    deal_is_usd = deals["Deal Currency"].eq("USD")
    fund_is_usd = deals["Fund Currency"].eq("USD")
    
    # ---------- SP rate (current)
    deals["SP rate"] = np.where(
        deal_is_usd & fund_is_usd,
        1.0,  # Both USD: no conversion needed
        np.where(
            deal_is_usd & ~fund_is_usd,
            1.0 / deals["sp_fund"],  # Deal=USD, Fund≠USD: invert fund rate
            np.where(
                ~deal_is_usd & fund_is_usd,
                deals["sp_deal"],  # Deal≠USD, Fund=USD: use deal rate
                deals["sp_deal"] / deals["sp_fund"]  # Both≠USD: cross rate
            )
        )
    )
    
    # ---------- Forward 5Y rate
    deals["Forward 5Y rate"] = np.where(
        deal_is_usd & fund_is_usd,
        1.0,  # Both USD: no conversion needed
        np.where(
            deal_is_usd & ~fund_is_usd,
            1.0 / deals["fwd5_fund"],  # Deal=USD, Fund≠USD: invert fund rate
            np.where(
                ~deal_is_usd & fund_is_usd,
                deals["fwd5_deal"],  # Deal≠USD, Fund=USD: use deal rate
                deals["fwd5_deal"] / deals["fwd5_fund"]  # Both≠USD: cross rate
            )
        )
    )
    
    # ---------- USD SP (deal vs USD)
    deals["USD SP"] = np.where(deal_is_usd, 1.0, deals["sp_deal"])
    
    # ---------- Currency Pair
    deals["Currency Pair"] = deals["Deal Currency"] + " " + deals["Fund Currency"]
    
    # ---------- Ensure numeric
    for c in ["SP rate", "Forward 5Y rate", "USD SP", "sp_deal_5y", "sp_fund_5y"]:
        if c in deals.columns:
            deals[c] = pd.to_numeric(deals[c], errors="coerce")
    
    # ---------- Deal forward_fx (forward / spot, expected)
    deals["Deal forward_fx"] = np.where(
        deals["SP rate"].notna() & deals["Forward 5Y rate"].notna() & (deals["SP rate"] != 0),
        (((deals["Forward 5Y rate"] / deals["SP rate"]) ** (1/5) - 1)*100),
        np.nan
    )
    
    # ---------- NEW: SP rate 5Y using *future* spots
    # If future spots are missing, SP rate 5Y becomes NaN
    if {"sp_deal_5y", "sp_fund_5y"}.issubset(deals.columns):
        deal_is_usd = deals["Deal Currency"].eq("USD")
        fund_is_usd = deals["Fund Currency"].eq("USD")
        
        deals["SP rate 5Y"] = np.where(
            deal_is_usd & fund_is_usd,
            1.0,  # Both USD: no conversion needed
            np.where(
                deal_is_usd & ~fund_is_usd,
                1.0 / deals["sp_fund_5y"],  # Deal=USD, Fund≠USD: invert fund rate
                np.where(
                    ~deal_is_usd & fund_is_usd,
                    deals["sp_deal_5y"],  # Deal≠USD, Fund=USD: use deal rate
                    deals["sp_deal_5y"] / deals["sp_fund_5y"]  # Both≠USD: cross rate
                )
            )
        )
        deals["SP rate 5Y"] = pd.to_numeric(deals["SP rate 5Y"], errors="coerce")
    else:
        # Column not available → leave as NaN for clarity
        deals["SP rate 5Y"] = np.nan
    
    # ---------- NEW: Deal realized_fx (actual, from realized future spot vs current spot)
    deals["Deal realized_fx"] = np.where(
        deals["SP rate"].notna() & (deals["SP rate"] != 0) & deals["SP rate 5Y"].notna(),
        (((deals["SP rate 5Y"] / deals["SP rate"]) ** (1/5) - 1)*100),
        np.nan
    )

    # ---------- NEW: Deal RER (Real Exchange Rate)
    # Deal RER = Deal Currency RER / Fund Currency RER
    # Handle edge cases where RER data might be missing
    if {"rer_deal", "rer_fund"}.issubset(deals.columns):
        deals["Deal RER"] = np.where(
            deals["rer_deal"].notna() & deals["rer_fund"].notna() & (deals["rer_fund"] != 0),
            deals["rer_deal"] / deals["rer_fund"],
            np.nan
        )
        deals["Deal RER"] = pd.to_numeric(deals["Deal RER"], errors="coerce")
    else:
        # RER columns not available
        deals["Deal RER"] = np.nan

    # Add a column for the deal size in USD
    deals["Deal Size USD (MN)"] = deals["DEAL SIZE (CURR. MN)"] * deals["SP rate"]
    deals['Deal Year'] = deals['Deal Quarter'].dt.year
    # rename columns to avoid spaces and special characters in formulas
    deals = deals.rename(columns={
    'Deal Size USD (MN)': 'deal_size_usd_mn',
    'Deal Year': 'deal_year',
    'Deal realized_fx': 'deal_realized_fx',
    'Deal forward_fx': 'deal_forward_fx',
    'Deal RER': 'deal_rer',
    })
    
    # Ensure deal_year is preserved through all subsequent operations
    # add the logarithm variables to the dta_deal
    ln_vars = ['deal_size_usd_mn']
    for var in ln_vars:
        deals[f'ln_{var}'] = np.log(deals[var] + 1)

    # drop all deals with missing deal_forward_fx before winsorizing
    deals = deals[deals['deal_forward_fx'].notna()]
    # winsorize the deal_forward_fx, deal_realized_fx, and deal_rer in the dta_deal at 1% and 99%
    winsorize_cols = ['deal_forward_fx', 'deal_realized_fx']
    # Only winsorize deal_rer if it has non-null values
    if deals['deal_rer'].notna().any():
        winsorize_cols.append('deal_rer')
    deals = winsorize(deals, winsorize_cols)
    # sort the deals by deal_id
    deals = deals.sort_values(by='DEAL ID')
    # create deal_currency_simplified and fund_currency_simplified
    major_currencies = ['EUR', 'GBP', 'USD', 'CHF', 'INR', 'RUB', 'CAD', 'CNY', 'KRW', 'CPY']
    deals['deal_currency_simplified'] = np.where(
        deals['Deal Currency'].isin(major_currencies), 
        deals['Deal Currency'], 
        'Other'
    )
    deals['fund_currency_simplified'] = np.where(
        deals['Fund Currency'].isin(major_currencies), 
        deals['Fund Currency'], 
        'Other'
    )
    # read the macro_controls.csv
    macro_controls = pd.read_csv("Output_data/macro_controls.csv")
    # add macro_controls to the deals, ensuring deal_year is preserved
    deals = deals.merge(
        macro_controls, 
        left_on=['deal_year','TARGET COMPANY COUNTRY'], 
        right_on=['year','country_name'], 
        how='left',
        suffixes=('', '_macro')
    ).drop(columns=['year', 'country_name'])

   
    return deals



def calculate_deal_weight(
    deals: DataFrame,
    fund_col: str = "fund_id",
    deal_col: str = "DEAL ID",
    num_funds_in_deal_col: str = "num_funds_in_deal",
    size_col: str = "DEAL SIZE (CURR. MN)",
    currency_col: str = "Deal Currency",
    usd_sp_col: str = "USD SP",
    ecr_col: str = "Deal forward_fx",   # kept for validation / downstream uses
    usd_size_col: str = "Deal Size USD"
) -> DataFrame:
    """
    For each deal:
      1) Build Deal Size USD:
         - If Deal Currency == 'USD' -> Deal Size USD = DEAL SIZE (CURR. MN)
         - Else -> Deal Size USD = DEAL SIZE (CURR. MN) * USD SP   (USD SP = <deal_ccy>→USD)
      2) Deal size is equally divided among investors: size_per_fund = Deal Size USD / number_funds_in_deal
      3) For deals with missing size, assign the fund's average size_per_fund.
      4) If an entire fund has all deal sizes missing, apply equal weighting.
      5) Weight = assigned size_per_fund / total assigned size_per_fund (per fund).
    """

    df = deals.copy()

    # 1) Deal Size USD using the pre-computed USD SP
    size_vals = pd.to_numeric(df[size_col], errors="coerce")
    usd_sp_vals = pd.to_numeric(df[usd_sp_col], errors="coerce")

    is_usd = df[currency_col].astype(str).str.upper().eq("USD")
    df[usd_size_col] = np.where(
        size_vals.notna(),
        np.where(is_usd, size_vals, size_vals * usd_sp_vals),
        np.nan
    )

    # 2) Split among investors
    fund_counts = pd.to_numeric(df[num_funds_in_deal_col], errors="coerce")
    df["deal_size_per_fund_usd"] = df[usd_size_col] / fund_counts

    # 3) Funds with at least one observed size
    has_obs_size = df.groupby(fund_col)["deal_size_per_fund_usd"].transform(lambda s: s.notna().any()).astype(bool)
    avg_assigned = df.groupby(fund_col)["deal_size_per_fund_usd"].transform("mean")

    fill_mask = has_obs_size & df["deal_size_per_fund_usd"].isna()
    df.loc[fill_mask, "deal_size_per_fund_usd"] = avg_assigned[fill_mask]

    # 4) Funds with all sizes missing -> equal weighting (use 1 so post-normalization is equal)
    no_obs_size = ~has_obs_size
    df.loc[no_obs_size, "deal_size_per_fund_usd"] = 1.0

    # 5) Normalize to weights
    total_assigned = df.groupby(fund_col)["deal_size_per_fund_usd"].transform("sum")
    df["deal_weight"] = df["deal_size_per_fund_usd"] / total_assigned

    # Diagnostics
    df["total_assigned_size_usd_fund"] = total_assigned
    df["fund_n_deals"] = df.groupby(fund_col)[deal_col].transform("nunique")
    df["all_sizes_missing_in_fund"] = no_obs_size

    return df


def calculate_fund_fx_measure(
    deals: DataFrame,
    fund_col: str = "fund_id",
    deal_col: str = "DEAL ID",
    ecr_col: str = "deal_forward_fx",
    acr_col: str = "deal_realized_fx",
    rer_col: str = "deal_rer",
    deal_ccy_col: str = "Deal Currency",
    fund_ccy_col: str = "Fund Currency",
) -> DataFrame:
    """
    Return fund-level currency metrics with BOTH weighting variants for forward_fx, realized_fx, rer,
    and the share of foreign investment (deal ccy != fund ccy).

    Outputs (one row per fund):
      - forward_fx_weighted,  forward_fx
      - realized_fx_weighted,  realized_fx
      - rer_weighted, rer
      - foreign_investment_pct_weighted, foreign_investment_pct
      - n_deals, total_assigned_size_usd_fund
      - fund_forward_fx, fund_realized_fx (compat aliases to weighted forward_fx/realized_fx)
    """
    # Required base columns
    required = {fund_col, deal_col, deal_ccy_col, fund_ccy_col}
    missing = required - set(deals.columns)
    if missing:
        raise KeyError(f"Missing required columns in deals: {missing}")

    df = deals.copy()

    # Ensure weights exist
    if "deal_weight" not in df.columns:
        df = calculate_deal_weight(df)  # must create normalized 'deal_weight'

    # Coerce forward_fx/realized_fx/rer if present
    if ecr_col in df.columns:
        df[ecr_col] = pd.to_numeric(df[ecr_col], errors="coerce")
    if acr_col in df.columns:
        df[acr_col] = pd.to_numeric(df[acr_col], errors="coerce")
    if rer_col in df.columns:
        df[rer_col] = pd.to_numeric(df[rer_col], errors="coerce")

    # ---- forward_fx weighted & equal
    if ecr_col in df.columns:
        size_w_ear = (group_weighted_mean(df, fund_col, ecr_col, "deal_weight")).rename("forward_fx_weighted")
        eq_w_ear   = (group_equal_mean(df, fund_col, ecr_col)).rename("forward_fx")
    else:
        size_w_ear = pd.Series(dtype="float64", name="forward_fx_weighted")
        eq_w_ear   = pd.Series(dtype="float64", name="forward_fx")

    # ---- realized_fx weighted & equal
    if acr_col in df.columns:
        size_w_acr = (group_weighted_mean(df, fund_col, acr_col, "deal_weight")).rename("realized_fx_weighted")
        eq_w_acr   = (group_equal_mean(df, fund_col, acr_col)).rename("realized_fx")
    else:
        size_w_acr = pd.Series(dtype="float64", name="realized_fx_weighted")
        eq_w_acr   = pd.Series(dtype="float64", name="realized_fx")

    # ---- rer weighted & equal
    if rer_col in df.columns:
        size_w_rer = (group_weighted_mean(df, fund_col, rer_col, "deal_weight")).rename("rer_weighted")
        eq_w_rer   = (group_equal_mean(df, fund_col, rer_col)).rename("rer")
    else:
        size_w_rer = pd.Series(dtype="float64", name="rer_weighted")
        eq_w_rer   = pd.Series(dtype="float64", name="rer")

    # ---- Foreign investment indicator (1 if deal ccy != fund ccy, else 0; NaN if either missing)
    deal_ccy = df[deal_ccy_col].astype(str).str.upper()
    fund_ccy = df[fund_ccy_col].astype(str).str.upper()
    valid_ccy = df[deal_ccy_col].notna() & df[fund_ccy_col].notna()  # Use original columns for notna check
    df["_is_foreign_num"] = np.where(valid_ccy, (deal_ccy != fund_ccy).astype(float), np.nan)

    size_w_foreign = group_weighted_mean(df, fund_col, "_is_foreign_num", "deal_weight") \
        .rename("foreign_investment_pct_weighted")
    eq_w_foreign   = group_equal_mean(df, fund_col, "_is_foreign_num") \
        .rename("foreign_investment_pct")

    # ---- Number of foreign currencies per fund
    # n_foreign_currencies: number of unique foreign currencies (excluding fund currency) per fund, for valid_ccy
    mask_foreign = valid_ccy & (deal_ccy != fund_ccy)
    n_foreign_currencies = (
        df[mask_foreign].groupby(fund_col)[deal_ccy_col].nunique().rename("n_foreign_currencies")
    )
    
    # if valid_ccy and deal_ccy == fund_ccy, then n_foreign_currencies is 0
    # Get funds that have valid currencies but no foreign deals (should be 0, not NaN)
    funds_with_valid_ccy = df[valid_ccy].groupby(fund_col).size().index
    funds_with_foreign_deals = n_foreign_currencies.index
    funds_with_only_domestic = funds_with_valid_ccy.difference(funds_with_foreign_deals)
    
    # Add zeros for funds that have valid currencies but no foreign deals
    domestic_zeros = pd.Series(0, index=funds_with_only_domestic, name="n_foreign_currencies")
    n_foreign_currencies = pd.concat([n_foreign_currencies, domestic_zeros]).sort_index()

    
    # ---- Other summaries
    # 1) Collapse to one row per (fund, deal) with a single ECR (first non-null)
    df = (df
      .sort_values([fund_col, deal_col])  # or add a date column if you want earliest
      .drop_duplicates([fund_col, deal_col], keep='first')  # one row per fund–deal
    )
    res = df.groupby(fund_col).agg(
    fund_n_deals=(deal_col, 'nunique'),
    fund_n_deals_with_negative_forward_fx=(ecr_col, lambda s: (s < 0).sum()),
    fund_n_deals_with_non_negative_forward_fx=(ecr_col, lambda s: (s >= 0).sum())
    )
    fund_n_deals = res['fund_n_deals']
    fund_n_deals_with_negative_forward_fx = res['fund_n_deals_with_negative_forward_fx']
    fund_n_deals_with_non_negative_forward_fx = res['fund_n_deals_with_non_negative_forward_fx']
    fund_negative_forward_fx_ratio = (fund_n_deals_with_negative_forward_fx / fund_n_deals).rename("fund_negative_forward_fx_ratio")
    fund_non_negative_forward_fx_ratio = (fund_n_deals_with_non_negative_forward_fx / fund_n_deals).rename("fund_non_negative_forward_fx_ratio")
    if "total_assigned_size_usd_fund" in df.columns:
        # aggregate to one value per fund as The fund_total_assigned_size_usd_fund column contains the same value for all deals within a fund
        size_summary = df.groupby(fund_col)["total_assigned_size_usd_fund"].max()
    else:
        size_summary = pd.Series(index=fund_n_deals.index, dtype="float64", name="total_assigned_size_usd_fund")

    # ---- Assemble final DataFrame
    fund_fx = (
        pd.concat(
            [
                size_w_ear, eq_w_ear,
                size_w_acr, eq_w_acr,
                size_w_rer, eq_w_rer,
                size_w_foreign, eq_w_foreign,
                fund_n_deals, 
                fund_n_deals_with_negative_forward_fx, fund_n_deals_with_non_negative_forward_fx, 
                fund_negative_forward_fx_ratio, fund_non_negative_forward_fx_ratio, size_summary,
                n_foreign_currencies
            ],
            axis=1
        )
        .reset_index()
        .rename(columns={fund_col: "fund_id"})
    )

    # Clean temp columns in local df
    if "_is_foreign_num" in df.columns:
        df.drop(columns=["_is_foreign_num"], inplace=True, errors="ignore")

    # align the fund_id format
    fund_fx['fund_id'] = pd.to_numeric(fund_fx['fund_id'], errors='coerce').astype(int).astype(str)
    return fund_fx




def main(fx_path: str | Path = "Output_data/FX_forward.csv",
         deals_path: str | Path = "Output_data/fund_buyout_deals.csv", 
         rer_path: str | Path = "Output_data/RER_transformed.csv",
         out_path: str | Path = "Output_data/dta_deal.csv") -> DataFrame:
    """
    Main processing function to merge FX forwards and RER data into fund buyout deals.
    
    Args:
        fx_path: Path to FX forward csv file
        deals_path: Path to fund buyout deals CSV file
        rer_path: Path to RER transformed CSV file
        out_path: Path for output CSV file
        
    Returns:
        DataFrame with merged FX and RER data
    """
    print("Loading and cleaning FX forward data...")
    fx_sp, fx_5y = load_and_clean_fx_data(fx_path)
    print(f"✓ Loaded FX data: {fx_sp.shape[0]} quarters, {fx_sp.shape[1]} currencies")

    # save the fx_sp and fx_5y
    fx_sp.to_csv("Output_data/fx_sp.csv", index=True)
    fx_5y.to_csv("Output_data/fx_5y.csv", index=True)
    
    print("Loading and cleaning RER data...")
    rer = load_and_clean_rer_data(rer_path)
    print(f"✓ Loaded RER data: {rer.shape[0]} quarters, {rer.shape[1]} currencies")
    
    print("Loading fund buyout deals data...")
    deals = load_fund_buyout_deals(deals_path)
    print(f"✓ Loaded deals data: {len(deals):,} records")
    
    print("Merging FX rates...")
    deals_with_fx = merge_fx_rates(deals, fx_sp, fx_5y)
    
    print("Merging RER rates...")
    deals_with_fx_rer = merge_rer_rates(deals_with_fx, rer)
    
    print("Calculating cross rates and RER...")
    final_deals = calculate_fx_rates(deals_with_fx_rer)
    
    # Select and order final columns
    output_cols = [
        'DEAL ID', 'DEAL FUND', 'FUND NUMBER', 'fund_id', 'firm_id', 'vintage', 'deal_size_usd_mn',
        'Deal Quarter', 'deal_year', 'FX Quarter', 'FX Quarter 5Y', 'Currency Pair', 'Deal Currency', 'Fund Currency',
        'SP rate', 'Forward 5Y rate', 'SP rate 5Y', 'USD SP', 'geographic_focus', 'PRIMARY INDUSTRY',
        'deal_forward_fx', 'deal_realized_fx', 'deal_rer', 'rer_deal', 'rer_fund', 'deal_weight', 'deal_size_per_fund_usd',
        'TARGET COMPANY', 'TARGET COMPANY COUNTRY', 'DEAL DATE', 'firmcountry','fund_size_usd_mn','fund_number_overall','fund_number_series', 
        'buyout_fund_size','carried_interest_pct','hurdle_rate_pct',
        'DEAL STATUS', 'PRIMARY INDUSTRY', 'ln_deal_size_usd_mn',
        'deal_currency_simplified', 'fund_currency_simplified', 'fund_country'
    ]
    
    # Keep only columns that exist
    output_cols = [col for col in output_cols if col in final_deals.columns]
    final_output = final_deals[output_cols]

    # Save to output path
    print(f"Saving results to {out_path}...")
    final_output.to_csv(out_path, index=False)
    print(f"✓ Saved {len(final_output):,} records")
    
    # Compute weights & fund-level forward_fx
    print("Computing deal weights and fund-level forward_fx...")
    deals_w = calculate_deal_weight(final_deals)
    
    fund_fx_measure = calculate_fund_fx_measure(deals_w)
    
    fund_fx_measure.to_csv("Output_data/fund_fx_measure.csv", index=False)
    print(f"✓ Saved fund-level forward_fx measures: {len(fund_fx_measure):,} funds")

    
    
    return final_output


if __name__ == "__main__":
    # CLI arguments
    parser = argparse.ArgumentParser(description="Merge FX forwards into fund buyout deals")
    parser.add_argument("--fx-path", default="Output_data/FX_forward.csv", 
                       help="Path to FX forward Excel file")
    parser.add_argument("--deals-path", default="Output_data/fund_buyout_deals.csv",
                       help="Path to fund buyout deals CSV file")
    parser.add_argument("--rer-path", default="Output_data/RER_transformed.csv",
                       help="Path to RER transformed CSV file") 
    parser.add_argument("--out-path", default="Output_data/dta_deal.csv",
                       help="Path for output CSV file")
    
    args = parser.parse_args()
    
    # Run main processing
    result = main(args.fx_path, args.deals_path, args.rer_path, args.out_path)
    
    # Validations
    print("\n" + "="*60)
    print("VALIDATION CHECKS:")
    
    # 1. Assert no forward-looking merge
    valid_lag = result["FX Quarter"] < result["Deal Quarter"]
    print(f"✓ No forward-looking bias: {valid_lag.sum():,}/{len(result):,} records have proper lag")
    
    if not valid_lag.all():
        invalid_count = (~valid_lag).sum()
        print(f"⚠ Warning: {invalid_count:,} records have invalid time lag")
    
    # 2. Sample currency pair examples
    print("\nSample currency pairs:")
    print(f"  Most common pairs: {result['Currency Pair'].value_counts().head(5).to_dict()}")
    
    # 3. Data quality checks
    total_records = len(result)
    valid_sp_rates = result["SP rate"].notna().sum()
    valid_fwd_rates = result["Forward 5Y rate"].notna().sum()
    unique_quarters = result["Deal Quarter"].nunique()
    unique_pairs = result["Currency Pair"].nunique()
    
    print(f"\nData quality summary:")
    print(f"  Total records: {total_records:,}")
    print(f"  Valid SP rates: {valid_sp_rates:,} ({valid_sp_rates/total_records:.1f}%)")
    print(f"  Valid Forward rates: {valid_fwd_rates:,} ({valid_fwd_rates/total_records:.1f}%)")
    print(f"  Unique quarters: {unique_quarters}")
    print(f"  Unique currency pairs: {unique_pairs}")
    
    print("="*60)
    print("FX MERGE COMPLETE")
    
    # Show sample of final data
    print(f"\nSample of final data:")
    sample_cols = ['Currency Pair', 'Deal Quarter', 'FX Quarter', 'SP rate', 'Forward 5Y rate', 'USD SP']
    print(result[sample_cols].head(10).to_string())

