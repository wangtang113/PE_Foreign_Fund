import pandas as pd
import numpy as np


def group_weighted_mean(df: pd.DataFrame, group_col: str, value_col: str, weight_col: str) -> pd.Series:
    """
    Weight-normalized mean per group over non-null values:
      sum(value * weight) / sum(weight), excluding rows where value is NaN.
    NaN if a group's denominator is 0 (i.e., all values were NaN).
    """
    vals = pd.to_numeric(df[value_col], errors="coerce")
    wts  = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)

    eff_w = np.where(vals.notna(), wts, 0.0)
    contrib = np.where(vals.notna(), vals * wts, 0.0)

    num = pd.Series(contrib, index=df.index).groupby(df[group_col]).sum()
    den = pd.Series(eff_w,   index=df.index).groupby(df[group_col]).sum().replace(0.0, np.nan)
    return (num / den)

def group_equal_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Simple mean over non-null values per group."""
    vals = pd.to_numeric(df[value_col], errors="coerce")
    return vals.groupby(df[group_col]).mean()


def legal_tender(country: str, period: pd.Period, changes: pd.DataFrame, currencies: pd.DataFrame) -> str:
    """
    Return ISO code valid for <country> in <period> (monthly).
    If the month is before the first recorded change, return that first row's OLD code.
    If there are no changes for the country, fall back to the currencies master.
    
    Args:
        country: Country name
        period: Period (monthly) to check
        changes: DataFrame with currency change history
        currencies: DataFrame with current currency information
        
    Returns:
        ISO currency code for the given country and period
    """
    sub = changes.loc[changes["COUNTRY"] == country].sort_values("EFFECTIVE")

    if sub.empty:
        # no changes at all → use the current currency
        row = currencies.loc[currencies["COUNTRY"] == country, "ISO CODE"]
        return row.iat[0] if not row.empty else np.nan

    first_eff = sub["EFFECTIVE"].iat[0]

    # BEFORE the first change: use the first OLD code
    if period < first_eff:
        return sub["OLD"].iat[0]

    # ON/AFTER: take the latest NEW whose EFFECTIVE <= period
    idx = sub["EFFECTIVE"].searchsorted(period, side="right") - 1
    return sub["NEW"].iat[idx]


def build_country_currency_lookup(ccm: pd.DataFrame) -> pd.DataFrame:
    """
    Build a mapping of (FIRMCOUNTRY, vintage_year) -> currency_code.

    Accepts common shapes for country_currency_month.csv:
      - A date-like column (e.g., 'date', 'month', 'period', 'as_of', 'yyyymm')
      - OR a numeric 'year' column
      - A country column: one of {'country','firmcountry','country_name'}
      - A currency column: one of {'currency','local_currency','iso_code','iso','ccy'}

    Returns a DataFrame with columns: ['firmcountry','vintage','currency_lookup']
    """
    df = ccm.copy()

    # --- detect columns (case-insensitive)
    cols_lower = {c.lower(): c for c in df.columns}

    # country
    country_col = None
    for cand in ('country', 'firmcountry', 'country_name'):
        if cand in cols_lower:
            country_col = cols_lower[cand]
            break
    if country_col is None:
        raise KeyError("country_currency_month.csv needs a country column (country/firmcountry/country_name).")

    # currency
    currency_col = None
    for cand in ('currency', 'local_currency', 'iso_code', 'iso', 'ccy'):
        if cand in cols_lower:
            currency_col = cols_lower[cand]
            break
    if currency_col is None:
        raise KeyError("country_currency_month.csv needs a currency column (currency/local_currency/iso_code/iso/ccy).")

    # year
    date_col = None
    for cand in ('date', 'month', 'period', 'as_of', 'yyyymm'):
        if cand in cols_lower:
            date_col = cols_lower[cand]
            break
    year_col = cols_lower.get('year')

    if date_col:
        # try robust parse of date → year
        try:
            year_series = pd.to_datetime(df[date_col], errors='coerce').dt.year
        except Exception:
            # try yyyymm style
            year_series = pd.to_datetime(df[date_col].astype(str), format="%Y%m", errors='coerce').dt.year
        df['_vintage'] = year_series
    elif year_col:
        df['_vintage'] = pd.to_numeric(df[year_col], errors='coerce')
    else:
        raise KeyError("country_currency_month.csv must have a date/month or year column to derive vintage.")

    # normalize keys
    df['_firmcountry'] = df[country_col].astype(str).str.strip().str.upper()
    df['_currency']    = df[currency_col].astype(str).str.strip().str.upper()

    # pick a single currency per (country, year). Prefer the last observation in that year.
    sort_key = date_col if date_col is not None else year_col
    if sort_key is not None and sort_key in df.columns:
        df = df.sort_values(sort_key)

    look = (df.dropna(subset=['_vintage', '_currency'])
              .groupby(['_firmcountry', '_vintage'])['_currency']
              .last()
              .reset_index()
              .rename(columns={'_firmcountry': 'firmcountry', '_vintage': 'vintage', '_currency': 'currency_lookup'}))

    # ensure types
    look['vintage'] = pd.to_numeric(look['vintage'], errors='coerce').astype('Int64')
    return look

def z_standardize(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Z-standardize specified columns in a DataFrame to have mean 0 and standard deviation 1.
    
    This function standardizes variables by subtracting the mean and dividing by the standard
    deviation, creating z-scores. This is useful for making variables comparable and for
    regression analysis when variables have different scales.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to standardize
    columns : list
        List of column names to standardize
        
    Returns:
    --------
    pd.DataFrame
        A copy of the input DataFrame with standardized columns
        
    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
    >>> standardized_df = z_standardize(df, ['A', 'B'])
    """
    
    # Create a copy to avoid modifying the original DataFrame
    df_standardized = df.copy()
    
    # Validate inputs
    if not isinstance(columns, list):
        raise TypeError("columns must be a list of column names")
    
    # Check if all columns exist in the DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Standardize each specified column
    for col in columns:
        if df[col].dtype in ['object', 'category', 'bool']:
            print(f"Warning: Skipping non-numeric column '{col}'")
            continue
            
        # Convert to numeric, handling any non-numeric values
        series = pd.to_numeric(df[col], errors='coerce')
        
        # Skip if all values are NaN
        if series.isna().all():
            print(f"Warning: Column '{col}' contains only NaN values, skipping")
            continue
        
        # Calculate mean and standard deviation, ignoring NaN values
        mean_val = series.mean()
        std_val = series.std()
        
        # Skip if standard deviation is 0 (constant variable)
        if std_val == 0 or pd.isna(std_val):
            print(f"Warning: Column '{col}' has zero variance or missing std, skipping standardization")
            continue
        
        # Apply z-standardization: (x - mean) / std
        df_standardized[col] = (series - mean_val) / std_val
        
        # Print information about the standardization
        print(f"Standardized column '{col}': mean={mean_val:.4f}, std={std_val:.4f}")
        
        # Verify standardization (should be approximately 0 mean, 1 std)
        new_mean = df_standardized[col].mean()
        new_std = df_standardized[col].std()
        print(f"  After standardization: mean={new_mean:.6f}, std={new_std:.6f}")
    
    return df_standardized

def merge_benchmarks_by_date(dta_fund, benchmarks):
    # Prepare left date key
    if not np.issubdtype(dta_fund['date_reported'].dtype, np.datetime64):
        dta_fund['date_reported'] = pd.to_datetime(dta_fund['date_reported'], errors='coerce')
    dta_fund['date_reported_key'] = dta_fund['date_reported'].dt.strftime('%Y%m%d')

    # Prepare right keys / columns
    bm = benchmarks.rename(columns={
        'net_multiple_____median': 'net_multiple_median',
        'net_irr_____median': 'net_irr_median'
    }).copy()

    bm['constituent_as_at'] = bm['constituent_as_at'].astype(str).str.strip()

    keep = ['benchmark_id','benchmark_name','benchmark_vintage','constituent_as_at',
            'net_multiple_median','net_irr_median']
    bm = bm[keep]

    # Merge with indicator
    out = dta_fund.merge(
        bm,
        left_on=['benchmark_id','vintage','date_reported_key'],
        right_on=['benchmark_id','benchmark_vintage','constituent_as_at'],
        how='left',
        validate='many_to_one',
        indicator=True
    )

    # Report unmatched due to date
    pairs_in_bench = set(bm[['benchmark_id','benchmark_vintage']].dropna().apply(tuple, axis=1))
    is_left_only = out['_merge'].eq('left_only')
    has_series   = out[is_left_only].apply(lambda r: (r['benchmark_id'], r['vintage']) in pairs_in_bench, axis=1)

    report = out.loc[is_left_only & has_series, ['fund_id','benchmark_id','vintage','date_reported','date_reported_key']].copy()
    if not report.empty:
        avail = (bm.groupby(['benchmark_id','benchmark_vintage'])['constituent_as_at']
                 .apply(lambda s: ','.join(sorted(set(s.astype(str)))))
                 .rename('available_constituent_as_at')
                 .reset_index())
        report = report.merge(
            avail, left_on=['benchmark_id','vintage'],
            right_on=['benchmark_id','benchmark_vintage'], how='left'
        ).drop(columns=['benchmark_vintage'])
        report.to_csv(OUTPUT_UNMATCHED, index=False)
        print(f"[INFO] Wrote {len(report)} unmatched benchmark-by-date rows to {OUTPUT_UNMATCHED}")

    # Clean up
    out.drop(columns=['_merge','benchmark_vintage','constituent_as_at'], inplace=True, errors='ignore')

    # Convert all numeric columns to float (handle string/mixed type columns)
    out['net_multiple_median'] = pd.to_numeric(out['net_multiple_median'], errors='coerce')
    out['net_irr_median'] = pd.to_numeric(out['net_irr_median'], errors='coerce')
    out['net_irr_pcent'] = pd.to_numeric(out['net_irr_pcent'], errors='coerce')
    out['multiple'] = pd.to_numeric(out['multiple'], errors='coerce')

    # Excess returns
    out['excess_return_multiple'] = out['multiple']      - out['net_multiple_median']
    out['excess_return_irr']      = out['net_irr_pcent'] - out['net_irr_median']

    return out