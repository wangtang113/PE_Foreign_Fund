import pandas as pd
import numpy as np





def winsorize(df: pd.DataFrame, columns: list, limits: tuple = (0.01, 0.01)) -> pd.DataFrame:
    """
    Winsorize specified columns in a DataFrame by clipping extreme values to percentiles.
    
    This function replaces extreme values in the specified columns with less extreme values
    to reduce the impact of outliers. Values below the lower percentile are replaced with
    the lower percentile value, and values above the upper percentile are replaced with
    the upper percentile value.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to winsorize
    columns : list
        List of column names to winsorize
    limits : tuple, optional
        Tuple of (lower_limit, upper_limit) as percentiles between 0 and 1.
        Default is (0.01, 0.01) for 1st and 99th percentiles.
        
    Returns:
    --------
    pd.DataFrame
        A copy of the input DataFrame with winsorized columns
        
    Examples:
    ---------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 100], 'B': [10, 20, 30, 40]})
    >>> winsorized_df = winsorize(df, ['A'], limits=(0.25, 0.25))
    """
    
    # Create a copy to avoid modifying the original DataFrame
    df_winsorized = df.copy()
    
    # Validate inputs
    if not isinstance(columns, list):
        raise TypeError("columns must be a list of column names")
    
    if not isinstance(limits, tuple) or len(limits) != 2:
        raise TypeError("limits must be a tuple of (lower_limit, upper_limit)")
    
    lower_limit, upper_limit = limits
    
    if not (0 <= lower_limit <= 1 and 0 <= upper_limit <= 1):
        raise ValueError("limits must be between 0 and 1")
    
    if lower_limit + upper_limit >= 1:
        raise ValueError("sum of lower_limit and upper_limit must be less than 1")
    
    # Check if all columns exist in the DataFrame
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
    
    # Winsorize each specified column
    for col in columns:
        if df[col].dtype in ['object', 'category', 'bool']:
            print(f"Warning: Skipping non-numeric column '{col}'")
            continue
            
        # Convert to numeric, handling any non-numeric values
        series = pd.to_numeric(df[col], errors='coerce')
        
        # Remove missing values before winsorizing
        non_na_series = series.dropna()
        
        # Skip if all values are NaN
        if non_na_series.empty:
            print(f"Warning: Column '{col}' contains only NaN values, skipping")
            continue
        
        # Calculate percentiles, ignoring NaN values
        lower_percentile = non_na_series.quantile(lower_limit)
        upper_percentile = non_na_series.quantile(1 - upper_limit)
        
        # Apply winsorization using numpy's clip function, only to non-NaN values
        clipped = np.clip(non_na_series, lower_percentile, upper_percentile)
        
        # Assign clipped values back, preserving NaNs in their original positions
        series_winsorized = series.copy()
        series_winsorized.loc[non_na_series.index] = clipped
        df_winsorized[col] = series_winsorized
        
        # Print information about the winsorization
        lower_clipped = (non_na_series < lower_percentile).sum()
        upper_clipped = (non_na_series > upper_percentile).sum()
        total_clipped = lower_clipped + upper_clipped
        
        if total_clipped > 0:
            print(f"Winsorized column '{col}': {lower_clipped} values clipped at lower bound "
                  f"({lower_percentile:.4f}), {upper_clipped} values clipped at upper bound "
                  f"({upper_percentile:.4f})")
    
    return df_winsorized 