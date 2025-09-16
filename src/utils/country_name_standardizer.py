"""
Centralized country name standardization utility.

This module provides a consistent approach to standardizing country names across all scripts
in the project. It uses the clean_country_name.csv as the master reference and includes
additional mappings for various data sources.

The standardization follows this principle:
1. All country names are converted to uppercase
2. Standard mappings from clean_country_name.csv are applied
3. Additional source-specific mappings are applied
4. The result should match the country names used in country_currency_month.csv

Usage:
    from utils.country_name_standardizer import standardize_country_name
    
    clean_name = standardize_country_name("United States", source="gdp")
"""

import pandas as pd
import re
from typing import Optional, Dict
from pathlib import Path


class CountryNameStandardizer:
    """Centralized country name standardization class."""
    
    def __init__(self, base_path: str = "."):
        """
        Initialize the standardizer with the master mapping file.
        
        Args:
            base_path: Base path for locating the clean_country_name.csv file
        """
        self.base_path = Path(base_path)
        self._load_master_mappings()
        self._load_additional_mappings()
    
    def _load_master_mappings(self):
        """Load master mappings from clean_country_name.csv."""
        try:
            csv_path = self.base_path / "Input_data" / "clean_country_name.csv"
            if not csv_path.exists():
                # Try relative to current working directory
                csv_path = Path("Input_data/clean_country_name.csv")
            
            country_mapping = pd.read_csv(csv_path)
            self.master_mappings = dict(zip(
                country_mapping["original_name"].str.upper(),
                country_mapping["standardized_name"].str.upper()
            ))
            print(f"Loaded {len(self.master_mappings)} master country mappings")
        except FileNotFoundError:
            print("Warning: clean_country_name.csv not found. Using built-in mappings only.")
            self.master_mappings = {}
    
    def _load_additional_mappings(self):
        """Load additional mappings for various data sources."""
        
        # Common variations that appear across multiple sources
        self.common_mappings = {
            # USA variations
            'USA': 'UNITED STATES',
            'US': 'UNITED STATES', 
            'UNITED STATES OF AMERICA': 'UNITED STATES',
            'AMERICA': 'UNITED STATES',
            
            # UK variations  
            'UK': 'UNITED KINGDOM',
            'GREAT BRITAIN': 'UNITED KINGDOM',
            'BRITAIN': 'UNITED KINGDOM',
            
            # Korea variations
            'KOREA, REP.': 'SOUTH KOREA',
            'KOREA, REP. OF': 'SOUTH KOREA',
            'KOREA, REPUBLIC OF': 'SOUTH KOREA',
            'REPUBLIC OF KOREA': 'SOUTH KOREA',
            'KOREA (REPUBLIC OF)': 'SOUTH KOREA',
            'KOREA, DEM. PEOPLE\'S REP.': 'NORTH KOREA',
            'KOREA, DEMOCRATIC PEOPLE\'S REPUBLIC OF': 'NORTH KOREA',
            
            # Russia variations
            'RUSSIAN FEDERATION': 'RUSSIA',
            'RUSSIA': 'RUSSIA',
            
            # China territories
            'CHINA, HONG KONG SAR': 'HONG KONG',
            'HONG KONG SAR, CHINA': 'HONG KONG',
            'HONG KONG SAR - CHINA': 'HONG KONG',
            'CHINA, MACAO SAR': 'MACAU',
            'MACAU SAR, CHINA': 'MACAO',
            'MACAO SAR - CHINA': 'MACAO',
            'TAIWAN PROVINCE OF CHINA': 'TAIWAN',
            'TAIWAN, CHINA': 'TAIWAN',
            'CHINA, PEOPLE\'S REPUBLIC OF': 'CHINA',
            'CHINA, P.R.: MAINLAND': 'CHINA',
            'CHINA, P.R.: HONG KONG': 'HONG KONG',
            'CHINA, P.R.: MACAO': 'MACAU',
            
            # Iran variations
            'IRAN, ISLAMIC REP.': 'IRAN',
            'IRAN, ISLAMIC REPUBLIC OF': 'IRAN',
            'IRAN (ISLAMIC REP. OF)': 'IRAN',
            'IRAN, ISLAMIC REP. OF': 'IRAN',
            
            # Venezuela variations
            'VENEZUELA, RB': 'VENEZUELA',
            'VENEZUELA, BOLIVARIAN REPUBLIC OF': 'VENEZUELA',
            'VENEZUELA, REP. BOLIVARIANA DE': 'VENEZUELA',
            'VENEZUELA, BOLIV REP OF': 'VENEZUELA',
            
            # Other common variations
            'EGYPT, ARAB REP.': 'EGYPT',
            'EGYPT, ARAB REP. OF': 'EGYPT',
            'SYRIAN ARAB REPUBLIC': 'SYRIA',
            'SYRIAN ARAB REP.': 'SYRIA',
            'SLOVAK REPUBLIC': 'SLOVAKIA',
            'SLOVAK REP.': 'SLOVAKIA',
            'CZECH REPUBLIC': 'CZECH REPUBLIC',  # Keep full name
            'CZECHIA': 'CZECH REPUBLIC',
            'CZECH REP.': 'CZECH REPUBLIC',
            'BRUNEI DARUSSALAM': 'BRUNEI',
            'VIETNAM': 'VIETNAM',
            'VIET NAM': 'VIETNAM',
            'LAO PDR': 'LAOS',
            'LAO PEOPLE\'S DEM. REP.': 'LAOS',
            'LAO PEOPLE\'S DEMOCRATIC REPUBLIC': 'LAOS',
            'LAO P.D.R.': 'LAOS',
            'KYRGYZ REPUBLIC': 'KYRGYZSTAN',
            'KYRGYZ REP.': 'KYRGYZSTAN',
            
            # "THE" variations
            'GAMBIA, THE': 'GAMBIA',
            'BAHAMAS, THE': 'BAHAMAS',
            'NETHERLANDS, THE': 'NETHERLANDS',
            
            # Congo variations
            'CONGO, REP.': 'CONGO',
            'CONGO, REPUBLIC OF': 'CONGO (REPUBLIC)',
            'CONGO, DEM. REP.': 'CONGO, THE DEMOCRATIC REPUBLIC OF THE',
            'CONGO, DEM. REP. OF THE': 'CONGO, THE DEMOCRATIC REPUBLIC OF THE',
            
            # Other standardizations
            'CABO VERDE': 'CAPE VERDE',
            'TIMOR-LESTE': 'TIMOR-LESTE',
            'EAST TIMOR': 'TIMOR-LESTE',
            'WEST BANK AND GAZA': 'PALESTINIAN TERRITORY, OCCUPIED',
            'PALESTINE': 'PALESTINIAN TERRITORY, OCCUPIED',
            'YEMEN, REP.': 'YEMEN',
            'YEMEN, REP. OF': 'YEMEN',
            
            # Variations with formal names
            'AFGHANISTAN, ISLAMIC REP. OF': 'AFGHANISTAN',
            'ARMENIA, REP. OF': 'ARMENIA',
            'AZERBAIJAN, REP. OF': 'AZERBAIJAN',
            'BAHRAIN, KINGDOM OF': 'BAHRAIN',
            'BELARUS, REP. OF': 'BELARUS',
            'BOLIVIA, PLURINATIONAL STATE OF': 'BOLIVIA',
            'BOLIVIA (PLURINAT.STATE)': 'BOLIVIA',
            'BOSNIA AND HERZEGOVINA': 'BOSNIA HERZEGOVINA',
            'COMOROS, UNION OF THE': 'COMOROS',
            'DOMINICAN REP.': 'DOMINICAN REPUBLIC',
            'ETHIOPIA, THE FEDERAL DEM. REP. OF': 'ETHIOPIA',
            'MACEDONIA, FYR': 'MACEDONIA',
            'MACEDONIA, FMR YUG RP OF': 'MACEDONIA',
            'NORTH MACEDONIA, REPUBLIC OF': 'MACEDONIA',
            'MOLDOVA, REP. OF': 'MOLDOVA',
            'MOLDOVA, REPUBLIC OF': 'MOLDOVA',
            'MOLDOVA': 'MOLDOVA',
            'MONTENEGRO, REP. OF': 'MONTENEGRO',
            'SERBIA, REP. OF': 'SERBIA',
            'TANZANIA, UNITED REP. OF': 'TANZANIA',
            'TÜRKIYE, REP OF': 'TURKEY',
            'ESWATINI, KINGDOM OF': 'SWAZILAND',
            
            # Caribbean and Pacific islands
            'TRINIDAD AND TOBAGO': 'TRINIDAD',
            'ST. KITTS AND NEVIS': 'SAINT KITTS AND NEVIS',
            'ST. LUCIA': 'SAINT LUCIA',
            'ST. VINCENT AND THE GRENADINES': 'SAINT VINCENT/GRENADINES',
            'MICRONESIA, FEDERATED STATES OF': 'MICRONESIA,FED.STATES OF',
            'SÃO TOMÉ AND PRÍNCIPE, DEM. REP. OF': 'SAO TOME AND PRINCIPE',
            'ARUBA, KINGDOM OF THE NETHERLANDS': 'ARUBA',
            'FIJI, REPUBLIC OF': 'FIJI',
            
            # African countries
            'CÔTE D\'IVOIRE': 'IVORY COAST',
            'IVORY COAST': 'IVORY COAST',
            'CENTRAL AFRICAN REP.': 'CENTRAL AFRICAN REPUBLIC',
            'TIMOR-LESTE, DEM. REP. OF': 'TIMOR-LESTE',
        }
        
        # RER-specific mappings (for use with RER data)
        self.rer_specific_mappings = {
            'KOSOVO, REP. OF': None,  # Kosovo not in currency data
            'SAN MARINO, REP. OF': None,  # San Marino not in currency data  
            'WEST BANK AND GAZA': None,  # No currency data
        }
    
    def standardize(self, country_name: str, source: str = "default") -> Optional[str]:
        """
        Standardize a country name using the centralized mapping approach.
        
        Args:
            country_name: Original country name to standardize
            source: Data source context ("default", "rer", "gdp", "deals", etc.)
            
        Returns:
            Standardized country name or None if not mappable
        """
        if pd.isna(country_name) or country_name is None:
            return None
        
        # Convert to uppercase and strip whitespace
        cleaned = str(country_name).strip().upper()
        
        if not cleaned:
            return None
        
        # Apply source-specific mappings first (for RER data that may exclude certain countries)
        if source == "rer" and cleaned in self.rer_specific_mappings:
            return self.rer_specific_mappings[cleaned]
        
        # Apply master mappings from CSV
        if cleaned in self.master_mappings:
            cleaned = self.master_mappings[cleaned]
        
        # Apply common mappings
        if cleaned in self.common_mappings:
            cleaned = self.common_mappings[cleaned]
        
        # Apply regex-based cleaning for common patterns
        cleaned = self._apply_regex_cleaning(cleaned)
        
        # Always strip trailing/leading whitespace as final step
        if cleaned:
            cleaned = cleaned.strip()
        
        return cleaned
    
    def _apply_regex_cleaning(self, country: str) -> str:
        """Apply regex-based cleaning for common patterns."""
        
        # Remove ", THE" at the end
        country = re.sub(r',\s+THE$', '', country)
        
        # Remove ", REP. OF" and similar patterns
        country = re.sub(r',\s+(REP\.|REPUBLIC)\s+(OF|STATE).*$', '', country)
        country = re.sub(r',\s+ISLAMIC\s+REP\.\s+OF$', '', country)
        country = re.sub(r',\s+PEOPLE\'S\s+REPUBLIC\s+OF$', '', country)
        country = re.sub(r',\s+KINGDOM\s+OF.*$', '', country)
        
        # Handle "PLURINATIONAL STATE OF" pattern
        country = re.sub(r',\s+PLURINATIONAL\s+STATE\s+OF$', '', country)
        
        return country

    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about available mappings."""
        return {
            "master_mappings": len(self.master_mappings),
            "common_mappings": len(self.common_mappings),
            "rer_specific_mappings": len(self.rer_specific_mappings)
        }


# Global instance for easy import
_standardizer = None

def get_standardizer() -> CountryNameStandardizer:
    """Get the global standardizer instance."""
    global _standardizer
    if _standardizer is None:
        _standardizer = CountryNameStandardizer()
    return _standardizer

def standardize_country_name(country_name: str, source: str = "default") -> Optional[str]:
    """
    Convenience function to standardize a country name.
    
    Args:
        country_name: Original country name to standardize
        source: Data source context ("default", "rer", "gdp", "deals", etc.)
        
    Returns:
        Standardized country name or None if not mappable
    """
    return get_standardizer().standardize(country_name, source)

def validate_standardization(test_cases: Dict[str, str], source: str = "default") -> Dict[str, bool]:
    """
    Validate that a set of test cases produces expected results.
    
    Args:
        test_cases: Dictionary of {input_name: expected_output}
        source: Data source context
        
    Returns:
        Dictionary of {input_name: is_correct}
    """
    standardizer = get_standardizer()
    results = {}
    
    for input_name, expected in test_cases.items():
        actual = standardizer.standardize(input_name, source)
        results[input_name] = (actual == expected)
        if actual != expected:
            print(f"MISMATCH: '{input_name}' -> expected '{expected}', got '{actual}'")
    
    return results
