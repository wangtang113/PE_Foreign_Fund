from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union

def apply_eur_nan_rule(panel: pd.DataFrame,
                       currencies: pd.DataFrame,
                       currency_history: pd.DataFrame,
                       country_col: str = "COUNTRY",
                       currency_col: str = "ISO CODE",
                       iso_col: str = "ISO_CODE") -> pd.DataFrame:
    """
    For countries whose currency is EUR in `currencies` but are NOT present
    in `currency_history`, set ISO_CODE to NaN in `panel`.
    
    Args:
        panel: DataFrame with country-period data and ISO_CODE column
        currencies: DataFrame with country and currency information
        currency_history: DataFrame with currency change history
        country_col: Name of country column (default: "COUNTRY")
        currency_col: Name of currency column in currencies (default: "ISO CODE")
        iso_col: Name of ISO code column in panel (default: "ISO_CODE")
    
    Returns:
        Updated panel DataFrame with NaN values for target countries
    """
    # Find EUR countries in currencies
    eur_countries = set(
        currencies.loc[currencies[currency_col].eq("EUR"), country_col].unique()
    )
    
    # Find countries present in currency_history
    # Handle case where currency_history might use "country" instead of "COUNTRY"
    history_country_col = "country" if "country" in currency_history.columns else country_col
    countries_in_history = set(currency_history[history_country_col].unique())
    
    # Identify target countries: EUR but not in history
    target_countries = eur_countries - countries_in_history
    
    print(f"EUR countries missing from currency_history: {sorted(target_countries)}")
    
    # Apply the rule: set ISO_CODE to NaN for target countries
    mask = panel[country_col].isin(target_countries)
    panel.loc[mask, iso_col] = pd.NA
    
    return panel
