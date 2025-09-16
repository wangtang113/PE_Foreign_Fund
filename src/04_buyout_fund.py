# This script is to calculate the FX holdings of each fund
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from utils.dataclean_utils import build_country_currency_lookup
from utils.country_name_standardizer import standardize_country_name


def get_fund_details_config():
    """
    Configuration for fund details column mapping and requirements.
    All column mappings verified to exist in Preqin_fund_details.xlsx.
    Makes it easy to add new variables without modifying the main function.
    
    Returns:
        dict: Configuration with column_mapping, required_cols, and optional_cols
    """
    config = {
        'column_mapping': {
            # Required columns (core fund identification)
            'FUND ID': 'fund_id',
            'FIRM ID': 'firm_id',
            'NAME': 'fund_name',
            'VINTAGE / INCEPTION YEAR': 'vintage',
            'FUND CURRENCY': 'fund_currency',
            'ASSET CLASS': 'asset_class',
            'STRATEGY': 'strategy',
            
            # Optional columns (fund characteristics) - verified to exist
            'GEOGRAPHIC FOCUS': 'geographic_focus',
            'PRIMARY REGION FOCUS': 'primary_region_focus',
            'FUND SIZE (USD MN)': 'fund_size_usd_mn',
            'TARGET SIZE (USD MN)': 'target_size_usd_mn',
            'DOMICILE': 'domicile',
            'STATUS': 'status',
            'FINAL CLOSE DATE': 'final_close_date',
            'REGION': 'region',
            'COUNTRY': 'fund_country',
            'FUND STRUCTURE': 'fund_structure',
            'FUND NUMBER (OVERALL)': 'fund_number_overall',
            'FUND NUMBER (SERIES)': 'fund_number_series',
            'CARRIED INTEREST (%)': 'carried_interest_pct',
            'HURDLE RATE (%)': 'hurdle_rate_pct',
            'PE: BUYOUT FUND SIZE': 'buyout_fund_size',
            # Add more verified mappings here as needed
        },
        'required_cols': ['fund_id', 'fund_name', 'fund_currency', 'firm_id', 'vintage', 'asset_class', 'strategy'],
        'optional_cols': [
            # Fund characteristics
            'geographic_focus', 'primary_region_focus', 'region', 'fund_country', 'domicile',
            # Fund size and financial details
            'fund_size_usd_mn', 'target_size_usd_mn', 'carried_interest_pct', 'buyout_fund_size', 'hurdle_rate_pct',
            # Fund structure and status
            'status', 'fund_structure', 'fund_number_overall', 'fund_number_series', 'final_close_date'
            # Add new verified optional columns here
        ]
    }
    return config


def load_and_process_buyout_deals():
    """Load and process buyout deals data with FX currency mapping."""
    # read the columns in buyout_deals.
    # Now using Lukas' dataset as buyout deals

    
    # Define selected columns upfront for efficient SQL reading
    selected_columns = ['DEAL ID', 'TARGET COMPANY ID', 'TARGET COMPANY',
           'TARGET COMPANY COUNTRY', 'DEAL DATE', 'DEAL STATUS', 'DEAL TYPES', 'INVESTORS', 'FUNDS',
           'DEAL CURRENCY', 'DEAL SIZE (CURR. MN)', 'PRIMARY INDUSTRY']
    # Read CSV files with proper handling for mixed data types
    print("Loading Preqin deals data from CSV files...")
    buyout_deals_1 = pd.read_csv("Input_data/Preqin_deals_1.csv", low_memory=False)
    buyout_deals_2 = pd.read_csv("Input_data/Preqin_deals_2.csv", low_memory=False)
    buyout_deals = pd.concat([buyout_deals_1, buyout_deals_2], ignore_index=True)
    buyout_deals = buyout_deals[selected_columns]
   
    # We allow non-buyout deals in buyout funds
    # buyout_deals = buyout_deals[buyout_deals['STRATEGY'] == 'Buyout']

    country_currency_month = pd.read_csv("Output_data/country_currency_month.csv")


    # Include only buyout deals are completed. Drop it temporarily to align sample with Lukas
    #buyout_deals = buyout_deals[buyout_deals['DEAL STATUS'] == 'Completed']

    # Standardize country names in 'TARGET COMPANY COUNTRY' using centralized standardizer
    buyout_deals['TARGET COMPANY COUNTRY'] = buyout_deals['TARGET COMPANY COUNTRY'].apply(
        lambda x: standardize_country_name(x, source="deals")
    )
 
    # Extract deal month as period (YYYY-MM) and convert to string for merging
    # Handle mixed date formats in the CSV data
    buyout_deals['DEAL DATE'] = pd.to_datetime(buyout_deals['DEAL DATE'], format='mixed', errors='coerce')
    buyout_deals['DEAL MONTH'] = buyout_deals['DEAL DATE'].dt.to_period('M').astype(str)

    # Fill missing deal currency using country and month from country_currency_month, drop column COUNTRY
    buyout_deals = buyout_deals.merge(
        country_currency_month,
        left_on=['TARGET COMPANY COUNTRY', 'DEAL MONTH'],
        right_on=['COUNTRY', 'PERIOD'],
        how='left',
        validate='many_to_one'
    )
    # Now I try dropping the deal currency filling part to see if it works (already reverted)
    mask = buyout_deals['DEAL CURRENCY'].isna()
    buyout_deals.loc[mask, 'DEAL CURRENCY'] = buyout_deals.loc[mask, 'ISO_CODE']

    # Remove extra columns from merge
    buyout_deals = buyout_deals.drop(columns=['ISO_CODE', 'COUNTRY'])

    
    return buyout_deals


def transform_buyout_deals(
    fund_details_path: Union[str, Path] = "Input_data/Preqin_fund_details.xlsx",
    manager_details_path: Union[str, Path] = "Input_data/Preqin_fund_managers.xlsx",
    country_currency_month_path: Union[str, Path] = "Output_data/country_currency_month.csv",
    fund_col: str = "FUNDS",
    deal_fund_col: str = "DEAL FUND",
    num_funds_in_deal_col: str = "num_funds_in_deal",
    fund_name_col: str = "fund_name",
    fund_id_col: str = "fund_id",
) -> pd.DataFrame:
    """
    Transform buyout deals by exploding comma-separated fund names and enriching with fund details.

    Args:
        fund_details_path: Path to fund details Excel file
        manager_details_path: Path to manager details Excel file  
        country_currency_month_path: Path to country currency mapping CSV
        fund_col: Column name for funds in deals data
        deal_fund_col: Output column name for individual fund per deal
        num_funds_in_deal_col: Output column name for fund count per deal
        fund_name_col: Column name for fund names in fund details
        fund_id_col: Column name for fund IDs in fund details

    Returns:
        buyout_details DataFrame with DEAL FUND, num_funds_in_deal, fund_id, and FUND CURRENCY.
    """
    # 1) Processed buyout deals
    buyout = load_and_process_buyout_deals()

    # 2) Load fund_details with necessary columns using configuration
    fund_details = pd.read_excel(fund_details_path)
    
    # Get configuration for column mapping and requirements
    config = get_fund_details_config()
    column_mapping = config['column_mapping']
    base_required_cols = config['required_cols']
    optional_cols = config['optional_cols']
    
    # Adjust required columns based on function parameters
    required_cols = [c if c != 'fund_id' else fund_id_col for c in base_required_cols]
    required_cols = [c if c != 'fund_name' else fund_name_col for c in required_cols]
    
    # Rename available columns only
    available_mappings = {k: v for k, v in column_mapping.items() if k in fund_details.columns}
    fund_details = fund_details.rename(columns=available_mappings)
    
    # Select available columns (required + available optional)
    available_optional_cols = [c for c in optional_cols if c in fund_details.columns]
    available_cols = [c for c in required_cols if c in fund_details.columns] + available_optional_cols
    fund_details = fund_details[available_cols].copy()

    # 3) Merge manager_details (normalize firmcountry)
    # Specify dtypes to avoid mixed type warnings for problematic columns
    dtype_spec = {
        'totalfundsraised10yearsmn': str,  # Column 21 - mixed numeric/text
        'iswomenowned': str                # Column 33 - mixed text values
    }
    mgr = pd.read_excel(manager_details_path, dtype=dtype_spec)[['FIRM ID', 'COUNTRY']].copy()
    # rename the columns
    mgr = mgr.rename(columns={
        'FIRM ID': 'firm_id',
        'COUNTRY': 'firmcountry'
    })
    # Standardize firm country names using centralized standardizer
    mgr['firmcountry'] = mgr['firmcountry'].apply(
        lambda x: standardize_country_name(x, source="manager")
    )
    # align the format of firm_id
    mgr['firm_id'] = pd.to_numeric(mgr['firm_id'], errors='coerce').astype(int).astype(str)
    fund_details['firm_id'] = pd.to_numeric(fund_details['firm_id'], errors='coerce').astype(int).astype(str)
    fund_details = fund_details.merge(mgr[['firm_id', 'firmcountry']], on='firm_id', how='left')

    fund_details['vintage'] = pd.to_numeric(fund_details['vintage'], errors='coerce').astype('Int64')
    fund_details['firmcountry'] = fund_details['firmcountry'].astype(str).str.strip().str.upper()

    # Rename to FUND CURRENCY for downstream consistency
    fund_details.rename(columns={'fund_currency': 'FUND CURRENCY'}, inplace=True)

    # --- From here, original explode/merge logic ---
    # Drop rows with missing FUNDS
    buyout = buyout[buyout[fund_col].notna() & (buyout[fund_col].str.strip() != "")].copy()

    deal_id_col = "DEAL ID"

    # Split fund list
    buyout["_fund_list"] = (
        buyout[fund_col]
        .astype(str)
        .str.split(",")
        .apply(lambda lst: [s.strip() for s in lst if s is not None and s.strip() != ""])
    )

    # Count funds per deal
    buyout[num_funds_in_deal_col] = buyout["_fund_list"].apply(len)

    # Explode
    exploded = buyout.explode("_fund_list", ignore_index=True)
    exploded.rename(columns={"_fund_list": deal_fund_col}, inplace=True)

    # Drop empty after trim
    exploded = exploded[exploded[deal_fund_col].notna() & (exploded[deal_fund_col] != "")].copy()

    # Merge fund_id + FUND CURRENCY by normalized fund_name (do not lower/strip for join)
    fund_details["_fund_name_key"] = fund_details[fund_name_col].astype(str).str.strip()
    exploded["_deal_fund_key"] = exploded[deal_fund_col].astype(str).str.strip()

    # Select merge columns dynamically based on what's available
    base_merge_cols = [fund_id_col, "FUND CURRENCY", "_fund_name_key", "asset_class", 'firm_id', 'fund_country',
    'vintage', 'strategy','firmcountry','fund_size_usd_mn','fund_number_overall','fund_number_series',
    'buyout_fund_size','carried_interest_pct','hurdle_rate_pct']
    
    # Add all available optional columns
    available_optional_cols = [c for c in config['optional_cols'] if c in fund_details.columns]
    merge_cols = base_merge_cols + available_optional_cols
    
    buyout_details = exploded.merge(
        fund_details[merge_cols],
        left_on="_deal_fund_key",
        right_on="_fund_name_key",
        how="left"
    )

    # Cleanup
    buyout_details.drop(columns=["_fund_name_key", "_deal_fund_key"], inplace=True)

    # Reorder columns - include optional columns only if they exist
    preferred_front = [deal_id_col, deal_fund_col, num_funds_in_deal_col, fund_id_col, "asset_class", 'firm_id', 'vintage', 'firmcountry']
    
    # Add all available optional columns to front
    available_optional_cols_in_result = [c for c in config['optional_cols'] if c in buyout_details.columns]
    preferred_front.extend(available_optional_cols_in_result)
    
    # Filter to only include columns that actually exist
    preferred_front = [c for c in preferred_front if c in buyout_details.columns]
    other_cols = [c for c in buyout_details.columns if c not in preferred_front]
    buyout_details = buyout_details[preferred_front + other_cols]

    # keep only asset_class == 'Private Equity'
    buyout_details = buyout_details[buyout_details['asset_class'] == 'Private Equity']
    # keep only strategy == 'Buyout'
    buyout_details = buyout_details[buyout_details['strategy'] == 'Buyout']
    # drop the missing fund currency
    buyout_details = buyout_details[buyout_details['FUND CURRENCY'].notna()]

    return buyout_details

if __name__ == "__main__":
    exploded_deals = transform_buyout_deals()
    exploded_deals.to_csv("Output_data/fund_buyout_deals.csv", index=False)

        








