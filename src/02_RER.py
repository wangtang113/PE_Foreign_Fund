# This script is to calculate the RER of every country in every year
# Transform RER data from wide format (quarters as columns) to long format matching FX_forward.csv structure

import pandas as pd
import numpy as np
import re
from typing import Dict, Optional
from utils.country_name_standardizer import standardize_country_name



def build_country_currency_lookup(ccm_df: pd.DataFrame) -> Dict[tuple, str]:
    """
    Build optimized lookup dictionary for country-currency mapping.
    
    Args:
        ccm_df: Country-currency mapping DataFrame
        
    Returns:
        Dictionary with (country, year, quarter) -> currency mapping
    """
    print("Building country-currency lookup...")
    
    # Convert PERIOD to datetime and extract year/month
    ccm_df = ccm_df.copy()
    ccm_df['period_dt'] = pd.to_datetime(ccm_df['PERIOD'])
    ccm_df['year'] = ccm_df['period_dt'].dt.year
    ccm_df['month'] = ccm_df['period_dt'].dt.month
    ccm_df['quarter'] = ((ccm_df['month'] - 1) // 3) + 1
    
    # Create lookup dictionary for fast access
    lookup = {}
    
    for _, row in ccm_df.iterrows():
        key = (row['COUNTRY'], row['year'], row['quarter'])
        lookup[key] = row['ISO_CODE']
    
    print(f"Built lookup with {len(lookup)} entries")
    return lookup

def transform_rer_to_fx_format():
    """
    Optimized transformation of RER data from wide format to long format matching FX_forward.csv structure.
    """
    
    print("Loading RER data...")
    rer = pd.read_excel("Input_data/RER_Quarterly.xlsx")
    
    print("Loading country-currency mapping...")
    country_currency_month = pd.read_csv("Output_data/country_currency_month.csv")
    
    # Build optimized lookup dictionary
    currency_lookup = build_country_currency_lookup(country_currency_month)
    available_countries = set(country_currency_month['COUNTRY'].unique())
    
    print("Processing country names...")
    # Get and clean country names
    rer_countries = rer.iloc[:, 0].copy()
    country_name_map = {}
    failed_mappings = []
    
    for original_name in rer_countries:
        if pd.isna(original_name):
            continue
        cleaned_name = standardize_country_name(original_name)
        country_name_map[original_name] = cleaned_name
        if cleaned_name and cleaned_name not in available_countries:
            failed_mappings.append((original_name, cleaned_name))
    
    if failed_mappings:
        print(f"Found {len(failed_mappings)} countries that failed to map to currencies:")
        for orig, cleaned in failed_mappings[:10]:
            print(f"  '{orig}' -> '{cleaned}'")
        if len(failed_mappings) > 10:
            print(f"  ... and {len(failed_mappings) - 10} more")
    
    print("Parsing quarter columns...")
    # Parse quarter columns and build time periods
    quarter_columns = rer.columns[1:]
    time_periods = []
    valid_col_indices = []
    
    for col_idx, quarter_col in enumerate(quarter_columns):
        if quarter_col.startswith('Q') and ' ' in quarter_col:
            parts = quarter_col.split()
            if len(parts) == 2 and parts[0][1:].isdigit() and parts[1].isdigit():
                quarter = int(parts[0][1])
                year = int(parts[1])
                
                # Create date in FX_forward.csv format (last day of quarter)
                if quarter == 1:
                    date_str = f"{year}-03-31"
                elif quarter == 2:
                    date_str = f"{year}-06-30"  
                elif quarter == 3:
                    date_str = f"{year}-09-30"
                else:  # quarter == 4
                    date_str = f"{year}-12-31"
                
                quarter_str = f"{year}Q{quarter}"
                
                time_periods.append({
                    'Date': date_str,
                    'Quarter': quarter_str,
                    'year': year,
                    'quarter': quarter,
                    'col_index': col_idx
                })
                valid_col_indices.append(col_idx)
    
    print(f"Found {len(time_periods)} valid time periods")
    
    # Create result DataFrame structure
    result_df = pd.DataFrame(time_periods)
    
    print("Building currency columns...")
    # Pre-compute all currencies we'll need
    all_currencies = set()
    
    for original_name, cleaned_name in country_name_map.items():
        if cleaned_name:
            for period in time_periods:
                key = (cleaned_name, period['year'], period['quarter'])
                if key in currency_lookup:
                    all_currencies.add(currency_lookup[key])
    
    sorted_currencies = sorted(all_currencies)
    print(f"Found {len(sorted_currencies)} currencies")
    
    # Initialize currency columns efficiently to avoid DataFrame fragmentation
    currency_data = {currency: np.nan for currency in sorted_currencies}
    currency_df = pd.DataFrame(currency_data, index=result_df.index)
    result_df = pd.concat([result_df, currency_df], axis=1)

    print("Filling RER values...")
    # Vectorized approach: process all countries and time periods
    rer_values = rer.iloc[:, 1:].values  # All RER values (excluding country column)
    
    for country_idx, original_name in enumerate(rer_countries):
        if pd.isna(original_name):
            continue
            
        cleaned_name = country_name_map.get(original_name)
        if not cleaned_name:
            continue
        
        # Process all time periods for this country
        for period_idx, period_info in enumerate(time_periods):
            col_idx = period_info['col_index']
            
            # Get currency for this country/time
            key = (cleaned_name, period_info['year'], period_info['quarter'])
            currency = currency_lookup.get(key)
            
            if currency and currency in result_df.columns:
                # Get RER value
                rer_value = rer_values[country_idx, col_idx]
                
                if pd.notna(rer_value) and np.isfinite(rer_value):
                    result_df.loc[period_idx, currency] = rer_value
    
    # Final sorting and column selection
    result_df = result_df.sort_values('Date').reset_index(drop=True)
    final_columns = ['Date', 'Quarter'] + sorted_currencies
    result_df = result_df[final_columns]
    
    print(f"Transformation complete. Result shape: {result_df.shape}")
    print(f"Date range: {result_df['Date'].min()} to {result_df['Date'].max()}")
    print(f"Sample non-null counts:")
    for currency in sorted_currencies[:5]:
        non_null_count = result_df[currency].notna().sum()
        print(f"  {currency}: {non_null_count}")
    
    return result_df, failed_mappings

if __name__ == "__main__":
    # Transform RER data
    print("Starting RER transformation...")
    rer_transformed, failed_mappings = transform_rer_to_fx_format()
    
    # Save to Output_data
    output_path = "Output_data/RER_transformed.csv"
    rer_transformed.to_csv(output_path, index=False)
    
    print(f"\nTransformed RER data saved to: {output_path}")
    
    # Save failed mappings for review
    if failed_mappings:
        failed_df = pd.DataFrame(failed_mappings, columns=['Original_Name', 'Cleaned_Name'])
        failed_path = "Output_data/RER_failed_mappings.csv"
        failed_df.to_csv(failed_path, index=False)
        print(f"Failed country mappings saved to: {failed_path}")
    
    print("\nSample of transformed data:")
    print(rer_transformed.head())
    
    # Final validation
    print("\nValidation:")
    total_values = rer_transformed.select_dtypes(include=[np.number]).notna().sum().sum()
    print(f"Total non-null RER values: {total_values}")
    print(f"Date range: {rer_transformed['Date'].min()} to {rer_transformed['Date'].max()}")
    print(f"Number of currencies: {len([c for c in rer_transformed.columns if c not in ['Date', 'Quarter']])}")