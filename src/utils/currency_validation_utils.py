"""
Utility functions for validating currency availability periods.

This module provides functions to ensure that currencies in the panel data
only appear within their valid time windows as defined in country_currency_month.csv.
"""

import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple, Optional

def load_currency_validity_periods(country_currency_path: str = "Output_data/country_currency_month.csv") -> pd.DataFrame:
    """
    Load and process currency validity periods from country_currency_month.csv.
    
    Returns:
        DataFrame with columns: currency, min_year, max_year
    """
    try:
        country_currency_df = pd.read_csv(country_currency_path)
        
        # Extract year from period (format: YYYY-MM)
        country_currency_df['year'] = country_currency_df['PERIOD'].str[:4].astype(int)
        
        # Get validity periods for each currency
        currency_periods = country_currency_df.groupby('ISO_CODE')['year'].agg(['min', 'max']).reset_index()
        currency_periods.columns = ['currency', 'min_year', 'max_year']
        
        return currency_periods
    
    except FileNotFoundError:
        print(f"Warning: Currency validity file not found at {country_currency_path}")
        return pd.DataFrame(columns=['currency', 'min_year', 'max_year'])
    except Exception as e:
        print(f"Error loading currency validity periods: {e}")
        return pd.DataFrame(columns=['currency', 'min_year', 'max_year'])

def get_currency_validity_dict(currency_periods: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
    """
    Convert currency periods DataFrame to a dictionary for fast lookup.
    
    Args:
        currency_periods: DataFrame with currency, min_year, max_year columns
        
    Returns:
        Dictionary mapping currency -> (min_year, max_year)
    """
    return {
        row['currency']: (row['min_year'], row['max_year']) 
        for _, row in currency_periods.iterrows()
    }

def is_currency_valid_for_year(currency: str, year: int, validity_dict: Dict[str, Tuple[int, int]]) -> bool:
    """
    Check if a currency is valid for a given year.
    
    Args:
        currency: Currency code (e.g., 'USD', 'EUR')
        year: Year to check
        validity_dict: Dictionary mapping currency -> (min_year, max_year)
        
    Returns:
        True if currency is valid for the year, False otherwise
    """
    if currency not in validity_dict:
        # If currency not in reference data, assume invalid
        return False
    
    min_year, max_year = validity_dict[currency]
    return min_year <= year <= max_year

def filter_currencies_by_validity(currencies: list, year: int, validity_dict: Dict[str, Tuple[int, int]]) -> list:
    """
    Filter a list of currencies to only include those valid for a given year.
    
    Args:
        currencies: List of currency codes
        year: Year to check validity for
        validity_dict: Dictionary mapping currency -> (min_year, max_year)
        
    Returns:
        List of currencies valid for the given year
    """
    return [
        currency for currency in currencies 
        if is_currency_valid_for_year(currency, year, validity_dict)
    ]

def validate_currency_combination(fund_currency: str, deal_currency: str, year: int, 
                                validity_dict: Dict[str, Tuple[int, int]]) -> bool:
    """
    Check if a fund-deal currency combination is valid for a given year.
    
    Args:
        fund_currency: Fund currency code
        deal_currency: Deal currency code  
        year: Year to check
        validity_dict: Dictionary mapping currency -> (min_year, max_year)
        
    Returns:
        True if both currencies are valid for the year, False otherwise
    """
    return (is_currency_valid_for_year(fund_currency, year, validity_dict) and 
            is_currency_valid_for_year(deal_currency, year, validity_dict))

def get_currency_transitions() -> Dict[str, Dict[str, int]]:
    """
    Get known currency transitions (for Euro adoption and other changes).
    
    Returns:
        Dictionary mapping old_currency -> {'new_currency': str, 'transition_year': int}
    """
    return {
        # Euro adoptions (official dates when national currencies were replaced)
        'ATS': {'new_currency': 'EUR', 'transition_year': 1999},  # Austrian Schilling
        'BEF': {'new_currency': 'EUR', 'transition_year': 1999},  # Belgian Franc
        'DEM': {'new_currency': 'EUR', 'transition_year': 1999},  # German Deutsche Mark
        'ESP': {'new_currency': 'EUR', 'transition_year': 1999},  # Spanish Peseta
        'FIM': {'new_currency': 'EUR', 'transition_year': 1999},  # Finnish Markka
        'FRF': {'new_currency': 'EUR', 'transition_year': 1999},  # French Franc
        'IEP': {'new_currency': 'EUR', 'transition_year': 1999},  # Irish Pound
        'ITL': {'new_currency': 'EUR', 'transition_year': 1999},  # Italian Lira
        'LUF': {'new_currency': 'EUR', 'transition_year': 1999},  # Luxembourg Franc
        'NLG': {'new_currency': 'EUR', 'transition_year': 1999},  # Dutch Guilder
        'PTE': {'new_currency': 'EUR', 'transition_year': 1999},  # Portuguese Escudo
        
        # Later Euro adoptions
        'GRD': {'new_currency': 'EUR', 'transition_year': 2001},  # Greek Drachma
        'SIT': {'new_currency': 'EUR', 'transition_year': 2007},  # Slovenian Tolar
        'CYP': {'new_currency': 'EUR', 'transition_year': 2008},  # Cypriot Pound
        'MTL': {'new_currency': 'EUR', 'transition_year': 2008},  # Maltese Lira
        'SKK': {'new_currency': 'EUR', 'transition_year': 2009},  # Slovak Koruna
        'EEK': {'new_currency': 'EUR', 'transition_year': 2011},  # Estonian Kroon
        'LVL': {'new_currency': 'EUR', 'transition_year': 2014},  # Latvian Lats
        'LTL': {'new_currency': 'EUR', 'transition_year': 2015},  # Lithuanian Litas
        'HRK': {'new_currency': 'EUR', 'transition_year': 2023},  # Croatian Kuna
    }

def apply_currency_transition(currency: str, year: int) -> str:
    """
    Apply currency transition if applicable.
    
    Args:
        currency: Original currency code
        year: Year of the transaction
        
    Returns:
        Updated currency code (original or transitioned currency)
    """
    transitions = get_currency_transitions()
    
    if currency in transitions:
        transition_info = transitions[currency]
        if year >= transition_info['transition_year']:
            return transition_info['new_currency']
    
    return currency

def validate_and_fix_panel_currencies(panel_df: pd.DataFrame, 
                                     country_currency_path: str = "Output_data/country_currency_month.csv",
                                     apply_transitions: bool = True,
                                     remove_invalid: bool = True) -> pd.DataFrame:
    """
    Validate and optionally fix currency issues in panel data.
    
    Args:
        panel_df: Panel DataFrame with fund_currency, deal_currency, deal_year columns
        country_currency_path: Path to country_currency_month.csv
        apply_transitions: Whether to apply currency transitions (e.g., NLG->EUR)
        remove_invalid: Whether to remove rows with invalid currencies
        
    Returns:
        Updated panel DataFrame
    """
    print("🔍 Validating panel currencies...")
    
    # Load currency validity periods
    currency_periods = load_currency_validity_periods(country_currency_path)
    validity_dict = get_currency_validity_dict(currency_periods)
    
    print(f"✓ Loaded validity periods for {len(validity_dict)} currencies")
    
    # Make a copy to avoid modifying original
    panel_fixed = panel_df.copy()
    original_size = len(panel_fixed)
    
    # Apply currency transitions if requested
    if apply_transitions:
        print("🔄 Applying currency transitions...")
        
        transitions_applied = 0
        for _, row in panel_fixed.iterrows():
            year = row['deal_year']
            
            # Apply transition for deal currency
            original_deal = row['deal_currency']
            new_deal = apply_currency_transition(original_deal, year)
            if new_deal != original_deal:
                panel_fixed.loc[row.name, 'deal_currency'] = new_deal
                transitions_applied += 1
            
            # Apply transition for fund currency
            original_fund = row['fund_currency']
            new_fund = apply_currency_transition(original_fund, year)
            if new_fund != original_fund:
                panel_fixed.loc[row.name, 'fund_currency'] = new_fund
                transitions_applied += 1
        
        print(f"✓ Applied {transitions_applied} currency transitions")
    
    # Remove invalid currencies if requested
    if remove_invalid:
        print("🗑️  Removing invalid currency combinations...")
        
        # Create mask for valid combinations
        valid_mask = panel_fixed.apply(
            lambda row: validate_currency_combination(
                row['fund_currency'], row['deal_currency'], row['deal_year'], validity_dict
            ), axis=1
        )
        
        removed_count = len(panel_fixed) - valid_mask.sum()
        panel_fixed = panel_fixed[valid_mask].copy()
        
        print(f"✓ Removed {removed_count} invalid currency combinations")
    
    final_size = len(panel_fixed)
    print(f"✓ Panel size: {original_size:,} → {final_size:,} observations")
    
    return panel_fixed

def generate_currency_validation_report(panel_df: pd.DataFrame,
                                       country_currency_path: str = "Output_data/country_currency_month.csv") -> Dict:
    """
    Generate a detailed report on currency validity issues in panel data.
    
    Args:
        panel_df: Panel DataFrame to analyze
        country_currency_path: Path to country_currency_month.csv
        
    Returns:
        Dictionary with validation results
    """
    # Load currency validity periods
    currency_periods = load_currency_validity_periods(country_currency_path)
    validity_dict = get_currency_validity_dict(currency_periods)
    
    # Check each row for validity
    violations = []
    
    for _, row in panel_df.iterrows():
        fund_currency = row['fund_currency']
        deal_currency = row['deal_currency']
        year = row['deal_year']
        
        fund_valid = is_currency_valid_for_year(fund_currency, year, validity_dict)
        deal_valid = is_currency_valid_for_year(deal_currency, year, validity_dict)
        
        if not fund_valid or not deal_valid:
            violations.append({
                'fund_id': row.get('fund_id'),
                'deal_year': year,
                'fund_currency': fund_currency,
                'deal_currency': deal_currency,
                'fund_currency_valid': fund_valid,
                'deal_currency_valid': deal_valid,
                'fund_validity_period': validity_dict.get(fund_currency, 'Unknown'),
                'deal_validity_period': validity_dict.get(deal_currency, 'Unknown')
            })
    
    return {
        'total_violations': len(violations),
        'violation_details': violations,
        'currencies_checked': len(set(panel_df['fund_currency'].tolist() + panel_df['deal_currency'].tolist())),
        'validity_periods_available': len(validity_dict)
    }
