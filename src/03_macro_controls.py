# This script is to clean the interest rate and GDP data and then merge them as macro_control

import pandas as pd
import numpy as np
# Import utilities
from utils.dataclean_utils import build_country_currency_lookup
from utils.country_name_standardizer import standardize_country_name


# read the interest rate data xlsx file from Input_data folder
interest_rate = pd.read_excel("Input_data/bond_yield_10y.xlsx")

# read the GDP data csv file from Input_data folder
gdp = pd.read_csv("Input_data/annual_gdp_growth.csv")

# Load reference country-currency mapping
country_currency = pd.read_csv("Output_data/country_currency_month.csv")

# clean interest rate data, keep December value for the next year (avoid forward looking bias)
# pivot the data to have currency and year as columns, the value is the interest rate of the December of the last year

print("Cleaning interest rate data...")

# Extract year and month from Date column
interest_rate['year'] = interest_rate['Date'].dt.year
interest_rate['month'] = interest_rate['Date'].dt.month

# Keep only December values to avoid forward-looking bias
interest_rate_dec = interest_rate[interest_rate['month'] == 12].copy()

# Use same year for simultaneous approach (December 1999 rates for 1999 performance)
# No year adjustment needed - use same year

# Get currency columns (exclude Date, year, month)
currency_cols = [col for col in interest_rate_dec.columns if col not in ['Date', 'year', 'month']]

# Melt to long format: currency, year, interest_rate
interest_rate_long = pd.melt(
    interest_rate_dec, 
    id_vars=['year'], 
    value_vars=currency_cols,
    var_name='currency', 
    value_name='interest_rate'
)


# No renaming needed - use year column directly



# clean GDP data, keep the annual growth rate of the last year
# pivot the data to have country name and year as columns, the value is the annual growth rate of the last year
# Clean the all country name to match the country_currency_month.csv and then add a column currency

print("\nCleaning GDP data...")

# Get year columns (columns that contain 'YR' and year numbers)
year_columns = [col for col in gdp.columns if 'YR' in col]
id_columns = ['Country Name', 'Country Code', 'Series Name', 'Series Code']

# Melt GDP data to long format
gdp_long = pd.melt(
    gdp,
    id_vars=id_columns,
    value_vars=year_columns,
    var_name='year_col',
    value_name='gdp_growth'
)

# Extract year from column names like '1999 [YR1999]'
gdp_long['year'] = gdp_long['year_col'].str.extract(r'(\d{4})').astype(int)

# Use same year for simultaneous approach (1999 GDP growth for 1999 performance)
# No year adjustment needed

# Remove missing values and convert GDP growth to numeric
gdp_long['gdp_growth'] = pd.to_numeric(gdp_long['gdp_growth'], errors='coerce')


# Keep only relevant columns - use same year
gdp_clean = gdp_long[['Country Name', 'year', 'gdp_growth']].copy()



# Build time-aware country-currency mapping

country_currency_lookup = build_country_currency_lookup(country_currency)

# save the country_currency_lookup to a csv file
country_currency_lookup.to_csv("Output_data/country_currency_lookup.csv", index=False)


# Apply country name cleaning
gdp_clean['country_name'] = gdp_clean['Country Name'].apply(standardize_country_name)

# Merge with time-aware currency mapping using both country and year

gdp_with_currency = gdp_clean.merge(
    country_currency_lookup.rename(columns={'firmcountry': 'country_name', 'vintage': 'year', 'currency_lookup': 'currency'}),
    on=['country_name', 'year'], 
    how='left'
)



# Keep only matched countries
gdp_final = gdp_with_currency.dropna(subset=['currency']).copy()
gdp_final = gdp_final[['country_name', 'currency', 'year', 'gdp_growth']].copy()


# Create macro controls dataset by merging interest rates and GDP data
print("\nCreating macro controls dataset...")

# Merge interest rates and GDP data on currency and year
macro_controls = interest_rate_long.merge(
    gdp_final,
    on=['currency', 'year'],
    how='outer'  # Keep all combinations
)

# Apply currency transition logic using country_currency_lookup
print("Applying currency transition logic from lookup table...")

# Load the country currency lookup to get actual transitions
lookup = pd.read_csv("Output_data/country_currency_lookup.csv")

# Find countries with currency transitions where target currency has interest rate data
available_currencies = set(interest_rate_long['currency'].unique())

# Identify currency transitions that matter for interest rates
currency_transitions = {}
for country in lookup['firmcountry'].unique():
    country_data = lookup[lookup['firmcountry'] == country].sort_values('vintage')
    
    # Check for currency changes
    if len(country_data['currency_lookup'].unique()) > 1:
        transitions = {}
        prev_currency = None
        
        for _, row in country_data.iterrows():
            current_currency = row['currency_lookup']
            year = row['vintage']
            
            # If currency changed and both old and new currencies have interest rate data
            if (prev_currency is not None and 
                current_currency != prev_currency and 
                current_currency in available_currencies and
                prev_currency in available_currencies):
                
                transitions[year] = current_currency
            
            prev_currency = current_currency
        
        if transitions:
            currency_transitions[country] = transitions

print(f"Found {len(currency_transitions)} countries with relevant currency transitions")

# Apply transitions to interest rates
for country, transitions in currency_transitions.items():
    for transition_year, target_currency in transitions.items():
        # Get the country mask
        country_mask = macro_controls['country_name'] == country
        
        # For the transition year and onwards, use the target currency's rates
        for year in macro_controls[country_mask]['year'].unique():
            if year >= transition_year:
                year_mask = country_mask & (macro_controls['year'] == year)
                
                # Find the target currency rate for this year
                target_rate = interest_rate_long[
                    (interest_rate_long['currency'] == target_currency) & 
                    (interest_rate_long['year'] == year)
                ]['interest_rate']
                
                if len(target_rate) > 0:
                    macro_controls.loc[year_mask, 'interest_rate'] = target_rate.iloc[0]
                    print(f"  Updated {country} {year}: {target_currency} rate {target_rate.iloc[0]:.3f}%")

# get the gdp_growth_tm1 and interest_rate_tm1 from the interest_rate and gdp_growth
# group by country to get proper country-specific lagged values
# This ensures each country's lagged values come from its own previous year data
macro_controls = macro_controls.sort_values(['country_name', 'year'])
macro_controls['gdp_growth_tm1'] = macro_controls.groupby(['country_name'])['gdp_growth'].shift(1)
macro_controls['interest_rate_tm1'] = macro_controls.groupby(['country_name'])['interest_rate'].shift(1)

# Apply currency transition logic to lagged values as well
print("Applying currency transition logic to lagged values...")

for country, transitions in currency_transitions.items():
    for transition_year, target_currency in transitions.items():
        # For the transition year, the tm1 value should be from the target currency's previous year
        country_mask = (macro_controls['country_name'] == country) & (macro_controls['year'] == transition_year)
        
        if country_mask.any():
            # Find the target currency rate for the previous year (which becomes tm1)
            target_rate_tm1 = interest_rate_long[
                (interest_rate_long['currency'] == target_currency) & 
                (interest_rate_long['year'] == transition_year - 1)
            ]['interest_rate']
            
            if len(target_rate_tm1) > 0:
                macro_controls.loc[country_mask, 'interest_rate_tm1'] = target_rate_tm1.iloc[0]
                print(f"  Updated {country} {transition_year} tm1: {target_currency} rate {target_rate_tm1.iloc[0]:.3f}%")

#-------------add the country-level all_deals_by_country_year--------------------------------
# read the deals data from full deals 1 and 2 using efficient SQL approach
# Read CSV files with proper handling for mixed data types
print("Loading deals data from CSV files...")
full_deals_1 = pd.read_csv("Input_data/full_deals_1.csv", low_memory=False)
full_deals_2 = pd.read_csv("Input_data/full_deals_2.csv", low_memory=False)
full_deals = pd.concat([full_deals_1, full_deals_2], ignore_index=True)


# standardize the country name
full_deals['target_company_country'] = full_deals['TARGET COMPANY COUNTRY'].apply(standardize_country_name)
# get the year from the deal date and convert to datetime
# Handle mixed date formats in the CSV data
full_deals['DEAL DATE'] = pd.to_datetime(full_deals['DEAL DATE'], format='mixed', errors='coerce')
full_deals['deal_year'] = full_deals['DEAL DATE'].dt.year
# get the country-level all_deals_by_country_year each year, each year has a different country-level all_deals_by_country_year
country_year_deals = full_deals.groupby(['target_company_country', 'deal_year']).size().reset_index(name='all_deals_by_country_year')

# Create all possible country-year combinations to handle gaps
countries = country_year_deals['target_company_country'].unique()
years = range(int(country_year_deals['deal_year'].min()), int(country_year_deals['deal_year'].max()) + 1)

# Create complete country-year grid
country_year_grid = []
for country in countries:
    for year in years:
        country_year_grid.append({'target_company_country': country, 'deal_year': year})

country_year_complete = pd.DataFrame(country_year_grid)

# Merge with actual deal counts, filling missing years with 0
country_year_complete = country_year_complete.merge(
    country_year_deals, 
    on=['target_company_country', 'deal_year'], 
    how='left'
)
country_year_complete['all_deals_by_country_year'] = country_year_complete['all_deals_by_country_year'].fillna(0)

# Sort and calculate proper t-1 values (actual previous year, not last appearance)
country_year_complete = country_year_complete.sort_values(['target_company_country', 'deal_year'])
country_year_complete['all_deals_by_country_year_tm1'] = country_year_complete.groupby('target_company_country')['all_deals_by_country_year'].shift(1)

# Keep only original years with actual data for merging (but now with correct tm1 values)
country_year_final = country_year_complete[
    country_year_complete['deal_year'].isin(country_year_deals['deal_year'])
].copy()

# merge the country-level all_deals_by_country_year to the macro_controls
macro_controls = macro_controls.merge(
    country_year_final[['target_company_country', 'deal_year', 'all_deals_by_country_year', 'all_deals_by_country_year_tm1']], 
    left_on=['country_name', 'year'], 
    right_on=['target_company_country', 'deal_year'], 
    how='left'
)

# drop duplicate rows in target_company_country and deal_year
macro_controls = macro_controls.drop_duplicates(subset=['country_name', 'year'])

# Save cleaned datasets
macro_controls.to_csv("Output_data/macro_controls.csv", index=False)
interest_rate_long.to_csv("Output_data/interest_rate_clean.csv", index=False)
gdp_final.to_csv("Output_data/gdp_clean.csv", index=False)


