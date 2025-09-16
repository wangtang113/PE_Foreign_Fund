"""
Region-Country Mapping Utility

This module creates comprehensive mappings between geographic regions and countries,
ensuring consistency with the country name standardization used throughout the project.
It combines region definitions with historical currency data to create a complete
region-country-year-currency dataset.

The mapping follows these principles:
1. Use standardized country names from country_name_standardizer.py
2. Apply geographic and economic region definitions based on international standards
3. Combine with historical currency data from country_currency_month.csv
4. Follow currency transitions as observed in the source data

Usage:
    from utils.region_country_mapper import create_region_country_currency_mapping
    
    result_df = create_region_country_currency_mapping()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.country_name_standardizer import standardize_country_name


class RegionCountryMapper:
    """Maps regions to countries using comprehensive geographic and economic definitions."""
    
    def __init__(self):
        """Initialize the mapper with comprehensive region-country mappings."""
        self._load_region_mappings()
    
    def _load_region_mappings(self):
        """Load comprehensive region-to-country mappings."""
        
        # Geographic regions based on UN geoscheme and other international standards
        self.region_country_mapping = {
            # Major Geographic Regions
            'Europe': [
                'ALBANIA', 'ANDORRA', 'ARMENIA', 'AUSTRIA', 'AZERBAIJAN', 'BELARUS', 'BELGIUM',
                'BOSNIA HERZEGOVINA', 'BULGARIA', 'CROATIA', 'CYPRUS', 'CZECH REPUBLIC', 
                'DENMARK', 'ESTONIA', 'FINLAND', 'FRANCE', 'GEORGIA', 'GERMANY', 'GREECE',
                'HUNGARY', 'ICELAND', 'IRELAND', 'ITALY', 'KAZAKHSTAN', 'LATVIA', 'LIECHTENSTEIN',
                'LITHUANIA', 'LUXEMBOURG', 'MACEDONIA', 'MALTA', 'MOLDOVA', 'MONACO', 'MONTENEGRO',
                'NETHERLANDS', 'NORWAY', 'POLAND', 'PORTUGAL', 'ROMANIA', 'RUSSIA', 'SAN MARINO',
                'SERBIA', 'SLOVAKIA', 'SLOVENIA', 'SPAIN', 'SWEDEN', 'SWITZERLAND', 'TURKEY',
                'UKRAINE', 'UNITED KINGDOM', 'VATICAN CITY', 'GIBRALTAR', 'CHANNEL ISLANDS'
            ],
            
            'West Europe': [
                'AUSTRIA', 'BELGIUM', 'FRANCE', 'GERMANY', 'IRELAND', 'ITALY', 'LIECHTENSTEIN',
                'LUXEMBOURG', 'MONACO', 'NETHERLANDS', 'PORTUGAL', 'SPAIN', 'SWITZERLAND',
                'UNITED KINGDOM'
            ],
            
            'Asia': [
                'AFGHANISTAN', 'ARMENIA', 'AZERBAIJAN', 'BAHRAIN', 'BANGLADESH', 'BHUTAN', 'BRUNEI',
                'CAMBODIA', 'CHINA', 'CYPRUS', 'GEORGIA', 'HONG KONG', 'INDIA', 'INDONESIA', 'IRAN',
                'IRAQ', 'ISRAEL', 'JAPAN', 'JORDAN', 'KAZAKHSTAN', 'KUWAIT', 'KYRGYZSTAN', 'LAOS',
                'LEBANON', 'MACAU', 'MALAYSIA', 'MALDIVES', 'MONGOLIA', 'MYANMAR', 'NEPAL',
                'NORTH KOREA', 'KOREA, DEM. PEOPLE\'S REP', 'OMAN', 'PAKISTAN', 'PALESTINIAN TERRITORY, OCCUPIED', 'PHILIPPINES',
                'QATAR', 'RUSSIA', 'SAUDI ARABIA', 'SINGAPORE', 'SOUTH KOREA', 'SRI LANKA', 'SYRIA',
                'TAIWAN', 'TAJIKISTAN', 'THAILAND', 'TIMOR-LESTE', 'TURKEY', 'TURKMENISTAN',
                'UNITED ARAB EMIRATES', 'UZBEKISTAN', 'VIETNAM', 'YEMEN'
            ],
            
            'North America': [
                'CANADA', 'MEXICO', 'UNITED STATES'
            ],
            
            'Americas': [
                'ANTIGUA AND BARBUDA', 'ARGENTINA', 'BAHAMAS', 'BARBADOS', 'BELIZE', 'BOLIVIA',
                'BRAZIL', 'CANADA', 'CHILE', 'COLOMBIA', 'COSTA RICA', 'CUBA', 'DOMINICA',
                'DOMINICAN REPUBLIC', 'ECUADOR', 'EL SALVADOR', 'GRENADA', 'GUATEMALA', 'GUYANA',
                'HAITI', 'HONDURAS', 'JAMAICA', 'MEXICO', 'NICARAGUA', 'PANAMA', 'PARAGUAY',
                'PERU', 'SAINT KITTS AND NEVIS', 'SAINT LUCIA', 'SAINT VINCENT/GRENADINES',
                'SURINAME', 'TRINIDAD', 'UNITED STATES', 'URUGUAY', 'VENEZUELA', 'FRENCH GUIANA'
            ],
            
            'Africa': [
                'ALGERIA', 'ANGOLA', 'BENIN', 'BOTSWANA', 'BURKINA FASO', 'BURUNDI', 'CAMEROON',
                'CAPE VERDE', 'CENTRAL AFRICAN REPUBLIC', 'CHAD', 'COMOROS', 'CONGO', 
                'CONGO, THE DEMOCRATIC REPUBLIC OF THE', 'DJIBOUTI', 'EGYPT', 'EQUATORIAL GUINEA',
                'ERITREA', 'ETHIOPIA', 'GABON', 'GAMBIA', 'GHANA', 'GUINEA', 'GUINEA-BISSAU',
                'IVORY COAST', 'KENYA', 'LESOTHO', 'LIBERIA', 'LIBYA', 'MADAGASCAR', 'MALAWI',
                'MALI', 'MAURITANIA', 'MAURITIUS', 'MOROCCO', 'MOZAMBIQUE', 'NAMIBIA', 'NIGER',
                'NIGERIA', 'RWANDA', 'SAO TOME AND PRINCIPE', 'SENEGAL', 'SEYCHELLES', 'SIERRA LEONE',
                'SOMALIA', 'SOUTH AFRICA', 'SOUTH SUDAN', 'SUDAN', 'SWAZILAND', 'TANZANIA',
                'TOGO', 'TUNISIA', 'UGANDA', 'ZAMBIA', 'ZIMBABWE', 'MAYOTTE', 'RÉUNION', 'SAINT HELENA',
                'WESTERN SAHARA'
            ],
            
            # Economic and Regional Groupings
            'ASEAN': [
                'BRUNEI', 'CAMBODIA', 'INDONESIA', 'LAOS', 'MALAYSIA', 'MYANMAR', 'PHILIPPINES',
                'SINGAPORE', 'THAILAND', 'VIETNAM'
            ],
            
            'GCC': [
                'BAHRAIN', 'KUWAIT', 'OMAN', 'QATAR', 'SAUDI ARABIA', 'UNITED ARAB EMIRATES'
            ],
            
            'EU': [
                'AUSTRIA', 'BELGIUM', 'BULGARIA', 'CROATIA', 'CYPRUS', 'CZECH REPUBLIC', 'DENMARK',
                'ESTONIA', 'FINLAND', 'FRANCE', 'GERMANY', 'GREECE', 'HUNGARY', 'IRELAND', 'ITALY',
                'LATVIA', 'LITHUANIA', 'LUXEMBOURG', 'MALTA', 'NETHERLANDS', 'POLAND', 'PORTUGAL',
                'ROMANIA', 'SLOVAKIA', 'SLOVENIA', 'SPAIN', 'SWEDEN'
            ],
            
            'OECD': [
                'AUSTRALIA', 'AUSTRIA', 'BELGIUM', 'CANADA', 'CHILE', 'COLOMBIA', 'CZECH REPUBLIC',
                'DENMARK', 'ESTONIA', 'FINLAND', 'FRANCE', 'GERMANY', 'GREECE', 'HUNGARY', 'ICELAND',
                'IRELAND', 'ISRAEL', 'ITALY', 'JAPAN', 'SOUTH KOREA', 'LATVIA', 'LITHUANIA',
                'LUXEMBOURG', 'MEXICO', 'NETHERLANDS', 'NEW ZEALAND', 'NORWAY', 'POLAND', 'PORTUGAL',
                'SLOVAKIA', 'SLOVENIA', 'SPAIN', 'SWEDEN', 'SWITZERLAND', 'TURKEY', 'UNITED KINGDOM',
                'UNITED STATES'
            ],
            
            # Sub-regional groupings
            'Central and East Europe': [
                'ALBANIA', 'BELARUS', 'BOSNIA HERZEGOVINA', 'BULGARIA', 'CROATIA', 'CZECH REPUBLIC',
                'ESTONIA', 'HUNGARY', 'LATVIA', 'LITHUANIA', 'MACEDONIA', 'MOLDOVA', 'MONTENEGRO',
                'POLAND', 'ROMANIA', 'RUSSIA', 'SERBIA', 'SLOVAKIA', 'SLOVENIA', 'UKRAINE'
            ],
            
            'Nordic': [
                'DENMARK', 'ESTONIA', 'FINLAND', 'ICELAND', 'LATVIA', 'LITHUANIA', 'NORWAY', 'SWEDEN'
            ],
            
            'Middle East': [
                'BAHRAIN', 'IRAN', 'IRAQ', 'ISRAEL', 'JORDAN', 'KUWAIT', 'LEBANON', 'OMAN',
                'PALESTINIAN TERRITORY, OCCUPIED', 'QATAR', 'SAUDI ARABIA', 'SYRIA', 'TURKEY',
                'UNITED ARAB EMIRATES', 'YEMEN'
            ],
            
            'MENA': [
                'ALGERIA', 'BAHRAIN', 'DJIBOUTI', 'EGYPT', 'IRAN', 'IRAQ', 'ISRAEL', 'JORDAN',
                'KUWAIT', 'LEBANON', 'LIBYA', 'MOROCCO', 'OMAN', 'PALESTINIAN TERRITORY, OCCUPIED',
                'QATAR', 'SAUDI ARABIA', 'SUDAN', 'SYRIA', 'TUNISIA', 'TURKEY', 'UNITED ARAB EMIRATES',
                'YEMEN'
            ],
            
            'East and Southeast Asia': [
                'BRUNEI', 'CAMBODIA', 'CHINA', 'HONG KONG', 'INDONESIA', 'JAPAN', 'LAOS', 'MACAU',
                'MALAYSIA', 'MONGOLIA', 'MYANMAR', 'NORTH KOREA', 'PHILIPPINES', 'SINGAPORE',
                'SOUTH KOREA', 'TAIWAN', 'THAILAND', 'TIMOR-LESTE', 'VIETNAM'
            ],
            
            'South Asia': [
                'AFGHANISTAN', 'BANGLADESH', 'BHUTAN', 'INDIA', 'MALDIVES', 'NEPAL', 'PAKISTAN',
                'SRI LANKA'
            ],
            
            'Sub-Saharan Africa': [
                'ANGOLA', 'BENIN', 'BOTSWANA', 'BURKINA FASO', 'BURUNDI', 'CAMEROON', 'CAPE VERDE',
                'CENTRAL AFRICAN REPUBLIC', 'CHAD', 'COMOROS', 'CONGO', 'CONGO, THE DEMOCRATIC REPUBLIC OF THE',
                'DJIBOUTI', 'EQUATORIAL GUINEA', 'ERITREA', 'ETHIOPIA', 'GABON', 'GAMBIA', 'GHANA',
                'GUINEA', 'GUINEA-BISSAU', 'IVORY COAST', 'KENYA', 'LESOTHO', 'LIBERIA', 'MADAGASCAR',
                'MALAWI', 'MALI', 'MAURITANIA', 'MAURITIUS', 'MOZAMBIQUE', 'NAMIBIA', 'NIGER', 'NIGERIA',
                'RWANDA', 'SAO TOME AND PRINCIPE', 'SENEGAL', 'SEYCHELLES', 'SIERRA LEONE', 'SOMALIA',
                'SOUTH AFRICA', 'SOUTH SUDAN', 'SWAZILAND', 'TANZANIA', 'TOGO', 'UGANDA', 'ZAMBIA', 
                'ZIMBABWE'
            ],
            
            'North Africa': [
                'ALGERIA', 'EGYPT', 'LIBYA', 'MOROCCO', 'SUDAN', 'TUNISIA'
            ],
            
            'South America': [
                'ARGENTINA', 'BOLIVIA', 'BRAZIL', 'CHILE', 'COLOMBIA', 'ECUADOR', 'GUYANA',
                'PARAGUAY', 'PERU', 'SURINAME', 'URUGUAY', 'VENEZUELA'
            ],
            
            'Central America': [
                'BELIZE', 'COSTA RICA', 'EL SALVADOR', 'GUATEMALA', 'HONDURAS', 'MEXICO',
                'NICARAGUA', 'PANAMA'
            ],
            
            'Caribbean': [
                'ANTIGUA AND BARBUDA', 'BAHAMAS', 'BARBADOS', 'CUBA', 'DOMINICA', 'DOMINICAN REPUBLIC',
                'GRENADA', 'HAITI', 'JAMAICA', 'SAINT KITTS AND NEVIS', 'SAINT LUCIA',
                'SAINT VINCENT/GRENADINES', 'TRINIDAD', 'ANGUILLA', 'MONTSERRAT', 'TURKS AND CAICOS IS.',
                'US VIRGIN ISLANDS', 'GUADELOUPE', 'MARTINIQUE', 'SAINT BARTHÉLEMY', 'SAINT-MARTIN',
                'SINT MAARTEN', 'CURAÇAO', 'BONAIRE/S.EUSTATIUS/SABA'
            ],
            
            'Australasia': [
                'AUSTRALIA', 'NEW ZEALAND'
            ],
            
            'Oceania': [
                'AUSTRALIA', 'NEW ZEALAND', 'FIJI', 'PAPUA NEW GUINEA', 'SOLOMON ISLANDS', 'VANUATU',
                'SAMOA', 'TONGA', 'KIRIBATI', 'TUVALU', 'NAURU', 'PALAU', 'MARSHALL ISLANDS',
                'MICRONESIA,FED.STATES OF', 'COOK ISLANDS', 'NIUE', 'TOKELAU', 'AMERICAN SAMOA',
                'GUAM', 'NORTHERN MARIANA IS.', 'FRENCH POLYNESIA', 'NEW CALEDONIA', 'WALLIS AND FUTUNA IS.',
                'PITCAIRN ISLANDS'
            ],
            
            'Greater China': [
                'CHINA', 'HONG KONG', 'MACAU', 'TAIWAN'
            ],
            
            'Central Asia': [
                'KAZAKHSTAN', 'KYRGYZSTAN', 'TAJIKISTAN', 'TURKMENISTAN', 'UZBEKISTAN'
            ],
            
            # Global and broad categories
            'Global': [],  # Will be populated with all countries
            'Emerging Markets': [],  # Will be populated with emerging market countries
            'Asia and Rest of World': [],  # Will be populated with all countries
            'Other': [],  # Catch-all for unspecified regions
            
            # Overseas territories and dependencies
            'French Territories': [
                'FRENCH GUIANA', 'GUADELOUPE', 'MARTINIQUE', 'RÉUNION', 'MAYOTTE', 'SAINT BARTHÉLEMY',
                'SAINT-MARTIN', 'ST. PIERRE AND MIQUELON', 'FRENCH POLYNESIA', 'NEW CALEDONIA',
                'WALLIS AND FUTUNA IS.', 'FRENCH SOUTHERN TERR'
            ],
            
            'British Territories': [
                'ANGUILLA', 'BERMUDA', 'BRITISH VIRGIN ISLANDS', 'CAYMAN ISLANDS', 'FALKLAND IS.(MALVINAS)',
                'GIBRALTAR', 'MONTSERRAT', 'PITCAIRN ISLANDS', 'SAINT HELENA', 'TURKS AND CAICOS IS.',
                'BRITISH INDIAN OCEAN TER', 'CHANNEL ISLANDS'
            ],
            
            'US Territories': [
                'AMERICAN SAMOA', 'GUAM', 'NORTHERN MARIANA IS.', 'US VIRGIN ISLANDS'
            ],
            
            'Australian Territories': [
                'CHRISTMAS ISLAND', 'COCOS (KEELING) ISLANDS'
            ],
            
            'Netherlands Territories': [
                'CURAÇAO', 'SINT MAARTEN', 'BONAIRE/S.EUSTATIUS/SABA'
            ]
        }
        
        # Individual country mappings (for countries that appear as regions)
        self.individual_countries = {
            'United States': ['UNITED STATES'],
            'US': ['UNITED STATES'],
            'United Kingdom': ['UNITED KINGDOM'],
            'UK': ['UNITED KINGDOM'],
            'Germany': ['GERMANY'],
            'France': ['FRANCE'],
            'Italy': ['ITALY'],
            'Spain': ['SPAIN'],
            'Japan': ['JAPAN'],
            'China': ['CHINA'],
            'India': ['INDIA'],
            'Brazil': ['BRAZIL'],
            'Canada': ['CANADA'],
            'Australia': ['AUSTRALIA'],
            'Mexico': ['MEXICO'],
            'Russia': ['RUSSIA'],
            'South Korea': ['SOUTH KOREA'],
            'Netherlands': ['NETHERLANDS'],
            'Switzerland': ['SWITZERLAND'],
            'Belgium': ['BELGIUM'],
            'Austria': ['AUSTRIA'],
            'Sweden': ['SWEDEN'],
            'Denmark': ['DENMARK'],
            'Norway': ['NORWAY'],
            'Finland': ['FINLAND'],
            'Poland': ['POLAND'],
            'Czech Republic': ['CZECH REPUBLIC'],
            'Hungary': ['HUNGARY'],
            'Israel': ['ISRAEL'],
            'Turkey': ['TURKEY'],
            'South Africa': ['SOUTH AFRICA'],
            'New Zealand': ['NEW ZEALAND'],
            'Ireland': ['IRELAND'],
            'Portugal': ['PORTUGAL'],
            'Greece': ['GREECE'],
            'Chile': ['CHILE'],
            'Colombia': ['COLOMBIA'],
            'Peru': ['PERU'],
            'Argentina': ['ARGENTINA'],
            'Thailand': ['THAILAND'],
            'Malaysia': ['MALAYSIA'],
            'Singapore': ['SINGAPORE'],
            'Philippines': ['PHILIPPINES'],
            'Indonesia': ['INDONESIA'],
            'Vietnam': ['VIETNAM'],
            'Taiwan': ['TAIWAN'],
            'Hong Kong SAR - China': ['HONG KONG'],
            'Macao SAR - China': ['MACAU'],
            'Taiwan - China': ['TAIWAN'],
            'Saudi Arabia': ['SAUDI ARABIA'],
            'United Arab Emirates': ['UNITED ARAB EMIRATES'],
            'Kuwait': ['KUWAIT'],
            'Qatar': ['QATAR'],
            'Bahrain': ['BAHRAIN'],
            'Oman': ['OMAN'],
            'Egypt': ['EGYPT'],
            'Morocco': ['MOROCCO'],
            'Nigeria': ['NIGERIA'],
            'Kenya': ['KENYA'],
            'Ghana': ['GHANA'],
            'Ethiopia': ['ETHIOPIA'],
            'Angola': ['ANGOLA'],
            'Algeria': ['ALGERIA'],
            'Tunisia': ['TUNISIA'],
            'Ivory Coast': ['IVORY COAST'],
            'Senegal': ['SENEGAL'],
            'Cameroon': ['CAMEROON'],
            'Zambia': ['ZAMBIA'],
            'Botswana': ['BOTSWANA'],
            'Mauritius': ['MAURITIUS'],
            'Madagascar': ['MADAGASCAR'],
            'Mozambique': ['MOZAMBIQUE'],
            'Tanzania': ['TANZANIA'],
            'Uganda': ['UGANDA'],
            'Rwanda': ['RWANDA'],
            'Gambia': ['GAMBIA'],
            'Sierra Leone': ['SIERRA LEONE'],
            'Liberia': ['LIBERIA'],
            'Benin': ['BENIN'],
            'Croatia': ['CROATIA'],
            'Slovenia': ['SLOVENIA'],
            'Slovakia': ['SLOVAKIA'],
            'Estonia': ['ESTONIA'],
            'Latvia': ['LATVIA'],
            'Lithuania': ['LITHUANIA'],
            'Bulgaria': ['BULGARIA'],
            'Romania': ['ROMANIA'],
            'Serbia': ['SERBIA'],
            'Montenegro': ['MONTENEGRO'],
            'Macedonia': ['MACEDONIA'],
            'Albania': ['ALBANIA'],
            'Bosnia & Herzegovina': ['BOSNIA HERZEGOVINA'],
            'Belarus': ['BELARUS'],
            'Ukraine': ['UKRAINE'],
            'Moldova': ['MOLDOVA'],
            'Georgia': ['GEORGIA'],
            'Armenia': ['ARMENIA'],
            'Azerbaijan': ['AZERBAIJAN'],
            'Kazakhstan': ['KAZAKHSTAN'],
            'Uzbekistan': ['UZBEKISTAN'],
            'Kyrgyzstan': ['KYRGYZSTAN'],
            'Tajikistan': ['TAJIKISTAN'],
            'Turkmenistan': ['TURKMENISTAN'],
            'Mongolia': ['MONGOLIA'],
            'North Korea': ['NORTH KOREA'],
            'Afghanistan': ['AFGHANISTAN'],
            'Pakistan': ['PAKISTAN'],
            'Bangladesh': ['BANGLADESH'],
            'Sri Lanka': ['SRI LANKA'],
            'Myanmar': ['MYANMAR'],
            'Cambodia': ['CAMBODIA'],
            'Laos': ['LAOS'],
            'Brunei': ['BRUNEI'],
            'Nepal': ['NEPAL'],
            'Bhutan': ['BHUTAN'],
            'Maldives': ['MALDIVES'],
            'Iran': ['IRAN'],
            'Iraq': ['IRAQ'],
            'Syria': ['SYRIA'],
            'Lebanon': ['LEBANON'],
            'Jordan': ['JORDAN'],
            'Yemen': ['YEMEN'],
            'Iceland': ['ICELAND'],
            'Luxembourg': ['LUXEMBOURG'],
            'Malta': ['MALTA'],
            'Cyprus': ['CYPRUS'],
            'Liechtenstein': ['LIECHTENSTEIN'],
            'Monaco': ['MONACO'],
            'Andorra': ['ANDORRA'],
            'San Marino': ['SAN MARINO'],
            'Vatican City': ['VATICAN CITY'],
            'Uruguay': ['URUGUAY'],
            'Paraguay': ['PARAGUAY'],
            'Bolivia': ['BOLIVIA'],
            'Ecuador': ['ECUADOR'],
            'Guyana': ['GUYANA'],
            'Suriname': ['SURINAME'],
            'Venezuela': ['VENEZUELA'],
            'Panama': ['PANAMA'],
            'Costa Rica': ['COSTA RICA'],
            'Nicaragua': ['NICARAGUA'],
            'Honduras': ['HONDURAS'],
            'El Salvador': ['EL SALVADOR'],
            'Guatemala': ['GUATEMALA'],
            'Belize': ['BELIZE'],
            'Cuba': ['CUBA'],
            'Haiti': ['HAITI'],
            'Dominican Republic': ['DOMINICAN REPUBLIC'],
            'Jamaica': ['JAMAICA'],
            'Trinidad and Tobago': ['TRINIDAD'],
            'Barbados': ['BARBADOS'],
            'Bahamas': ['BAHAMAS'],
            'Puerto Rico': ['PUERTO RICO'],
            'Bermuda': ['BERMUDA'],
            'British Virgin Islands': ['BRITISH VIRGIN ISLANDS'],
            'Cayman Islands': ['CAYMAN ISLANDS'],
            'Greenland': ['GREENLAND'],
            'Faroe Islands': ['FAROE ISLANDS'],
            'Jersey': ['JERSEY'],
            'Guernsey': ['GUERNSEY'],
            'Isle of Man': ['ISLE OF MAN'],
            'Aruba': ['ARUBA'],
        }
        
        # Combine all mappings
        self.region_country_mapping.update(self.individual_countries)
        
        # US regional subdivisions
        us_regions = {
            'Northeast': ['UNITED STATES'],
            'Midwest': ['UNITED STATES'],
            'Southeast': ['UNITED STATES'],
            'Southwest': ['UNITED STATES'],
            'West': ['UNITED STATES']
        }
        self.region_country_mapping.update(us_regions)
    
    def get_countries_for_region(self, region: str) -> List[str]:
        """Get standardized country names for a given region."""
        # Handle special cases
        if region in ['Global', 'Emerging Markets', 'Asia and Rest of World', 'Other']:
            return self._get_special_region_countries(region)
        
        # Direct mapping lookup
        if region in self.region_country_mapping:
            countries = self.region_country_mapping[region]
            # Standardize all country names
            standardized = []
            for country in countries:
                std_name = standardize_country_name(country)
                if std_name:
                    standardized.append(std_name)
            return standardized
        
        # Try to match region name as a country
        std_region = standardize_country_name(region)
        if std_region:
            return [std_region]
        
        return []
    
    def _get_special_region_countries(self, region: str) -> List[str]:
        """Handle special region categories."""
        if region == 'Global' or region == 'Asia and Rest of World' or region == 'Other':
            # Return all unique countries from all other regions
            all_countries = set()
            for countries in self.region_country_mapping.values():
                for country in countries:
                    std_name = standardize_country_name(country)
                    if std_name:
                        all_countries.add(std_name)
            return sorted(list(all_countries))
        
        elif region == 'Emerging Markets':
            # Define emerging markets based on common classifications
            emerging = [
                'ARGENTINA', 'BRAZIL', 'CHILE', 'CHINA', 'COLOMBIA', 'CZECH REPUBLIC', 'EGYPT',
                'GREECE', 'HUNGARY', 'INDIA', 'INDONESIA', 'KUWAIT', 'MALAYSIA', 'MEXICO',
                'PERU', 'PHILIPPINES', 'POLAND', 'QATAR', 'RUSSIA', 'SAUDI ARABIA', 'SOUTH AFRICA',
                'SOUTH KOREA', 'TAIWAN', 'THAILAND', 'TURKEY', 'UNITED ARAB EMIRATES'
            ]
            return [standardize_country_name(c) for c in emerging if standardize_country_name(c)]
        
        return []


def create_region_country_currency_mapping(
    currency_history_path: str = "Output_data/country_currency_month.csv",
    output_path: str = "Output_data/region_country_currency_mapping.csv"
) -> pd.DataFrame:
    """
    Create comprehensive region-country-year-currency mapping.
    
    This function directly follows the currency transitions as observed in 
    country_currency_month.csv without any additional processing.
    
    Args:
        currency_history_path: Path to country-currency history file
        output_path: Path for output file
        
    Returns:
        DataFrame with columns: region, country, year, currency
    """
    
    print("🌍 CREATING REGION-COUNTRY-CURRENCY MAPPING")
    print("=" * 60)
    
    # Initialize mapper
    mapper = RegionCountryMapper()
    
    # Generate all unique regions from the mapper's internal definitions
    print("📊 Generating region mappings internally...")
    unique_regions = list(mapper.region_country_mapping.keys())
    print(f"✓ Generated {len(unique_regions)} unique regions from internal definitions")
    
    # Load currency history
    print("💱 Loading currency history...")
    currency_df = pd.read_csv(currency_history_path)
    
    # Extract year from period and standardize country names
    currency_df['year'] = currency_df['PERIOD'].str[:4].astype(int)
    currency_df['country_std'] = currency_df['COUNTRY'].apply(
        lambda x: standardize_country_name(x)
    )
    
    # Remove rows where country standardization failed
    currency_df = currency_df[currency_df['country_std'].notna()].copy()
    print(f"✓ Loaded currency data: {len(currency_df):,} records for {currency_df['country_std'].nunique()} countries")
    print(f"✓ Year range: {currency_df['year'].min()}-{currency_df['year'].max()}")
    
    # Create region-country-year-currency mappings
    print("🔗 Creating region-country mappings...")
    
    results = []
    regions_processed = 0
    
    for region in sorted(unique_regions):
        regions_processed += 1
        if regions_processed % 20 == 0:
            print(f"  Processed {regions_processed}/{len(unique_regions)} regions...")
        
        countries = mapper.get_countries_for_region(region)
        
        if not countries:
            # For regions without country mappings, try direct lookup
            std_region = standardize_country_name(region)
            if std_region:
                countries = [std_region]
        
        # For each country in this region, get currency history
        for country in countries:
            country_currency_data = currency_df[currency_df['country_std'] == country]
            
            if len(country_currency_data) > 0:
                # Add records for each year-currency combination as observed in source data
                for _, row in country_currency_data.iterrows():
                    results.append({
                        'region': region,
                        'country': country,
                        'year': row['year'],
                        'currency': row['ISO_CODE']
                    })
            else:
                # If no currency data found, create a placeholder entry
                results.append({
                    'region': region,
                    'country': country,
                    'year': None,
                    'currency': None
                })
    
    print(f"✓ Processed all {len(unique_regions)} regions")
    
    # Create final DataFrame
    result_df = pd.DataFrame(results)
    
    # Remove duplicates and sort
    result_df = result_df.drop_duplicates().sort_values(['region', 'country', 'year'])
    
    print(f"📈 Final dataset statistics:")
    print(f"  Total records: {len(result_df):,}")
    print(f"  Unique regions: {result_df['region'].nunique()}")
    print(f"  Unique countries: {result_df['country'].nunique()}")
    print(f"  Year range: {result_df['year'].min()}-{result_df['year'].max()}")
    print(f"  Unique currencies: {result_df['currency'].nunique()}")
    
    # Show sample data
    print(f"\n📋 Sample data:")
    sample = result_df.head(10)
    print(sample.to_string(index=False))
    
    # Save result
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved complete mapping to {output_path}")
    
    return result_df


if __name__ == "__main__":
    print("🚀 RUNNING SIMPLIFIED REGION-COUNTRY MAPPER")
    print("=" * 60)
    
    # Create comprehensive region-country-currency mapping
    result = create_region_country_currency_mapping()
    
    # Show some interesting statistics
    print(f"\n📊 FINAL STATISTICS")
    print(f"=" * 40)
    
    # Countries with currency changes
    currency_changes = result.groupby('country')['currency'].nunique()
    countries_with_changes = currency_changes[currency_changes > 1]
    print(f"Countries with currency changes: {len(countries_with_changes)}")
    
    if len(countries_with_changes) > 0:
        print(f"Top 5 countries with most currency changes:")
        for country, count in countries_with_changes.nlargest(5).items():
            print(f"  {country}: {count} different currencies")
    
    # Regional coverage
    print(f"\nRegional coverage:")
    region_stats = result.groupby('region').agg({
        'country': 'nunique',
        'currency': 'nunique'
    }).sort_values('country', ascending=False)
    
    print(f"Top 10 regions by number of countries:")
    for region, row in region_stats.head(10).iterrows():
        print(f"  {region}: {row['country']} countries, {row['currency']} currencies")
    
    print(f"\n✅ SIMPLIFIED REGION-COUNTRY MAPPER COMPLETED!")
    print(f"Generated file: Output_data/region_country_currency_mapping.csv ({len(result):,} records)")