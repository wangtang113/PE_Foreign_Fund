"""
Data type coercion and validation utilities for robust data pipeline handling.

This module provides utilities to handle common data type issues in financial datasets,
including string representations of numbers with currency symbols, percentages,
thousands separators, and various missing value encodings.
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Union, Any, Optional
import warnings


def _strip_currency(value: str) -> str:
    """Strip currency symbols from string values."""
    if pd.isna(value) or not isinstance(value, str):
        return value
    
    # Common currency symbols and patterns
    currency_patterns = [
        r'^\$',        # Dollar sign at start
        r'^USD\s*',    # USD prefix
        r'^EUR\s*',    # EUR prefix
        r'^GBP\s*',    # GBP prefix
        r'^CHF\s*',    # CHF prefix
        r'€',          # Euro symbol
        r'£',          # Pound symbol
        r'¥',          # Yen symbol
        r'\$',         # Dollar symbol anywhere
    ]
    
    for pattern in currency_patterns:
        value = re.sub(pattern, '', value, flags=re.IGNORECASE)
    
    return value.strip()


def _strip_thousands(value: str, separators: List[str] = [",", " ", "'"]) -> str:
    """Strip thousands separators from string values."""
    if pd.isna(value) or not isinstance(value, str):
        return value
    
    for sep in separators:
        # Only remove separators that are followed by exactly 3 digits
        # to avoid removing decimal separators in European notation
        value = re.sub(f'\\{sep}(?=\\d{{3}})', '', value)
    
    return value


def _handle_parentheses_negative(value: str) -> str:
    """Convert parentheses notation for negative numbers to minus sign."""
    if pd.isna(value) or not isinstance(value, str):
        return value
    
    # Match (123.45) or ( 123.45 ) patterns
    match = re.match(r'^\s*\(\s*([\d.,]+)\s*\)\s*$', value.strip())
    if match:
        return f"-{match.group(1)}"
    
    return value


def _strip_percentage(value: str) -> str:
    """Strip percentage signs and convert to decimal if pct=True in coerce_numeric."""
    if pd.isna(value) or not isinstance(value, str):
        return value
    
    return value.replace('%', '').strip()


def coerce_numeric(
    df: pd.DataFrame, 
    cols: Union[str, List[str]], 
    *,
    strip: bool = True,
    pct: bool = False,
    currency: bool = False,
    thousands: List[str] = [",", " "],
    na_values: Tuple[str, ...] = ("", "NA", "N/A", "-", "—", "n/a", "NULL", "null", "NaN", "nan"),
    errors: str = "coerce"
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Union[int, float]]]]:
    """
    Coerce columns to numeric with robust handling of common data issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    cols : str or list of str
        Column name(s) to coerce
    strip : bool, default True
        Whether to strip whitespace
    pct : bool, default False
        Whether to handle percentage values (convert from % to decimal)
    currency : bool, default False
        Whether to strip currency symbols
    thousands : list of str, default [",", " "]
        Thousands separators to remove
    na_values : tuple of str
        Values to treat as NaN
    errors : str, default "coerce"
        How to handle conversion errors ('coerce', 'raise')
        
    Returns
    -------
    tuple
        (DataFrame with coerced columns, report dict with conversion statistics)
    """
    if isinstance(cols, str):
        cols = [cols]
    
    df_result = df.copy()
    report = {}
    
    for col in cols:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame, skipping")
            continue
            
        original_series = df[col].copy()
        working_series = original_series.copy()
        
        # Track original state
        original_count = len(working_series)
        original_na_count = working_series.isna().sum()
        original_numeric_count = pd.to_numeric(working_series, errors='coerce').notna().sum()
        
        # Convert to string for processing, but preserve original NAs
        string_series = working_series.astype(str)
        
        # Replace custom NA values with pandas NA
        for na_val in na_values:
            string_series = string_series.replace(na_val, pd.NA)
        
        # Apply transformations only to non-NA values
        mask_not_na = string_series.notna()
        
        if strip:
            string_series.loc[mask_not_na] = string_series.loc[mask_not_na].str.strip()
        
        if currency:
            string_series.loc[mask_not_na] = string_series.loc[mask_not_na].apply(_strip_currency)
        
        # Handle thousands separators
        for sep in thousands:
            string_series.loc[mask_not_na] = string_series.loc[mask_not_na].apply(
                lambda x: _strip_thousands(x, [sep]) if pd.notna(x) else x
            )
        
        # Handle parentheses for negatives
        string_series.loc[mask_not_na] = string_series.loc[mask_not_na].apply(_handle_parentheses_negative)
        
        # Handle percentage
        if pct:
            string_series.loc[mask_not_na] = string_series.loc[mask_not_na].apply(_strip_percentage)
        
        # Convert to numeric
        numeric_series = pd.to_numeric(string_series, errors=errors)
        
        # If percentage handling was requested, divide by 100
        if pct:
            # Only divide non-NA values
            mask_numeric = numeric_series.notna()
            numeric_series.loc[mask_numeric] = numeric_series.loc[mask_numeric] / 100
        
        # Store result
        df_result[col] = numeric_series
        
        # Calculate report statistics
        final_na_count = numeric_series.isna().sum()
        final_numeric_count = numeric_series.notna().sum()
        n_converted = final_numeric_count - original_numeric_count
        
        report[col] = {
            'original_count': original_count,
            'original_na_count': original_na_count,
            'original_numeric_count': original_numeric_count,
            'final_na_count': final_na_count,
            'final_numeric_count': final_numeric_count,
            'n_converted': n_converted,
            'conversion_rate': n_converted / original_count if original_count > 0 else 0,
            'ok_ratio': final_numeric_count / original_count if original_count > 0 else 0,
        }
    
    return df_result, report


def coerce_datetime(
    df: pd.DataFrame,
    cols: Union[str, List[str]],
    *,
    dayfirst: bool = False,
    yearfirst: bool = False,
    errors: str = "coerce"
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Union[int, float]]]]:
    """
    Coerce columns to datetime with robust handling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    cols : str or list of str
        Column name(s) to coerce
    dayfirst : bool, default False
        Whether to interpret the first value as day
    yearfirst : bool, default False  
        Whether to interpret the first value as year
    errors : str, default "coerce"
        How to handle conversion errors
        
    Returns
    -------
    tuple
        (DataFrame with coerced columns, report dict with conversion statistics)
    """
    if isinstance(cols, str):
        cols = [cols]
    
    df_result = df.copy()
    report = {}
    
    for col in cols:
        if col not in df.columns:
            warnings.warn(f"Column '{col}' not found in DataFrame, skipping")
            continue
            
        original_series = df[col].copy()
        original_count = len(original_series)
        original_na_count = original_series.isna().sum()
        
        # Try to convert to datetime
        try:
            datetime_series = pd.to_datetime(
                original_series, 
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                errors=errors
            )
        except Exception as e:
            if errors == "raise":
                raise
            else:
                warnings.warn(f"Failed to convert column '{col}' to datetime: {e}")
                datetime_series = original_series
        
        df_result[col] = datetime_series
        
        # Calculate statistics
        final_na_count = datetime_series.isna().sum()
        final_valid_count = datetime_series.notna().sum()
        
        report[col] = {
            'original_count': original_count,
            'original_na_count': original_na_count,
            'final_na_count': final_na_count,
            'final_valid_count': final_valid_count,
            'conversion_rate': final_valid_count / original_count if original_count > 0 else 0,
        }
    
    return df_result, report


def require_numeric(df: pd.DataFrame, cols: Union[str, List[str]], min_ok_ratio: float = 0.95) -> None:
    """
    Assert that columns are sufficiently numeric after coercion.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check
    cols : str or list of str
        Column name(s) to check
    min_ok_ratio : float, default 0.95
        Minimum ratio of non-NA numeric values required
        
    Raises
    ------
    ValueError
        If any column fails the numeric requirement
    """
    if isinstance(cols, str):
        cols = [cols]
    
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        
        series = df[col]
        total_count = len(series)
        valid_count = pd.to_numeric(series, errors='coerce').notna().sum()
        ok_ratio = valid_count / total_count if total_count > 0 else 0
        
        if ok_ratio < min_ok_ratio:
            # Get sample of problematic values
            numeric_series = pd.to_numeric(series, errors='coerce')
            bad_mask = numeric_series.isna() & series.notna()
            bad_values = series[bad_mask].head(10).tolist()
            
            raise ValueError(
                f"Column '{col}' fails numeric requirement: "
                f"only {ok_ratio:.2%} of values are numeric (need {min_ok_ratio:.2%}). "
                f"Sample bad values: {bad_values}"
            )


def get_dtype_info(df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """
    Get comprehensive dtype information for all columns in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze
    sample_size : int, default 100
        Number of non-null values to sample for analysis
        
    Returns
    -------
    pd.DataFrame
        DataFrame with dtype information for each column
    """
    info_records = []
    
    for col in df.columns:
        series = df[col]
        
        # Basic info
        total_count = len(series)
        na_count = series.isna().sum()
        non_na_count = total_count - na_count
        
        # Sample non-null values for analysis
        non_na_series = series.dropna()
        if len(non_na_series) > sample_size:
            sample_values = non_na_series.sample(sample_size, random_state=42)
        else:
            sample_values = non_na_series
        
        # Try numeric conversion
        numeric_series = pd.to_numeric(series, errors='coerce')
        numeric_count = numeric_series.notna().sum()
        
        # Try datetime conversion
        try:
            datetime_series = pd.to_datetime(series, errors='coerce')
            datetime_count = datetime_series.notna().sum()
        except:
            datetime_count = 0
        
        # Find problematic values (non-numeric among non-null)
        if non_na_count > 0:
            bad_mask = series.notna() & numeric_series.isna()
            bad_values = series[bad_mask].head(5).tolist()
        else:
            bad_values = []
        
        # Detect patterns
        has_currency = False
        has_percentage = False
        has_thousands_sep = False
        has_parentheses = False
        
        if len(sample_values) > 0:
            sample_str = sample_values.astype(str)
            has_currency = sample_str.str.contains(r'[$€£¥]|USD|EUR|GBP|CHF', case=False).any()
            has_percentage = sample_str.str.contains(r'%').any()
            has_thousands_sep = sample_str.str.contains(r'\d+[,\s]\d+').any()
            has_parentheses = sample_str.str.contains(r'\(\s*\d+.*\)').any()
        
        info_records.append({
            'column': col,
            'dtype': str(series.dtype),
            'total_count': total_count,
            'na_count': na_count,
            'non_na_count': non_na_count,
            'numeric_count': numeric_count,
            'datetime_count': datetime_count,
            'numeric_ratio': numeric_count / total_count if total_count > 0 else 0,
            'datetime_ratio': datetime_count / total_count if total_count > 0 else 0,
            'has_currency': has_currency,
            'has_percentage': has_percentage,
            'has_thousands_sep': has_thousands_sep,
            'has_parentheses': has_parentheses,
            'sample_bad_values': str(bad_values) if bad_values else '',
        })
    
    return pd.DataFrame(info_records)


def audit_numeric_columns(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Audit columns that should be numeric and identify issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to audit
    threshold : float, default 0.8
        Minimum ratio of numeric values to consider a column as "should be numeric"
        
    Returns
    -------
    pd.DataFrame
        Audit results for potentially numeric columns
    """
    dtype_info = get_dtype_info(df)
    
    # Filter to columns that are likely meant to be numeric
    potentially_numeric = dtype_info[
        (dtype_info['numeric_ratio'] >= threshold) |
        (dtype_info['has_currency']) |
        (dtype_info['has_percentage']) |
        (dtype_info['has_thousands_sep']) |
        (dtype_info['dtype'].str.contains('object|string'))
    ]
    
    return potentially_numeric.sort_values('numeric_ratio')
