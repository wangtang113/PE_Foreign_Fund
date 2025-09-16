# import the necessary libraries
import pandas as pd
import numpy as np
from utils.winsorize_utils import winsorize
from utils.currency_validation_utils import (
    load_currency_validity_periods, 
    get_currency_validity_dict,
    filter_currencies_by_validity,
    validate_and_fix_panel_currencies
)
from utils.region_country_mapper import create_region_country_currency_mapping
from utils.country_name_standardizer import standardize_country_name
# Load the datasets
dta_deal = pd.read_csv("Output_data/dta_deal.csv")
fx_forward = pd.read_csv("Output_data/FX_forward.csv")
rer_data = pd.read_csv("Output_data/RER_transformed.csv")

# Remove rows with missing key identifiers
dta_deal_clean = dta_deal.dropna(subset=['fund_id', 'Deal Currency', 'deal_year']).copy()

# Group by fund_id × deal_currency × deal_year and aggregate
dta_panel_currency = dta_deal_clean.groupby(['fund_id', 'Deal Currency', 'deal_year']).agg({
    'Fund Currency': 'first',           # Use first occurrence (should be consistent)
    'Currency Pair': 'first',           # Use first occurrence (should be consistent)
    'vintage': 'first',
    'firm_id': 'first',
    'deal_size_usd_mn': lambda x: x.sum() if x.notna().any() else np.nan,  # Sum, but remain NA if all are NA
    'DEAL ID': 'count'                  # Count number of deals (useful for validation)
}).reset_index()

# Rename columns for clarity
dta_panel_currency = dta_panel_currency.rename(columns={
    'fund_id': 'fund_id',
    'Deal Currency': 'deal_currency', 
    'deal_year': 'deal_year',
    'Fund Currency': 'fund_currency',
    'Currency Pair': 'currency_pair',
    'vintage': 'vintage',
    'firm_id': 'firm_id',
    'deal_size_usd_mn': 'sum_deal_size_usd_mn',
    'DEAL ID': 'n_deals'
})


# Convert FX data from wide to long format and create yearly data
# Extract year and quarter from Quarter format like "1998Q4"
fx_forward['fx_year'] = fx_forward['Quarter'].str.extract('(\d{4})').astype(int)
fx_forward['quarter_num'] = fx_forward['Quarter'].str.extract('Q(\d)').astype(int)

# Get yearly FX data (Q4 only - last quarter of each year)
fx_forward_q4 = fx_forward[fx_forward['quarter_num'] == 4].copy()
fx_forward_yearly = fx_forward_q4[['fx_year'] + [col for col in fx_forward_q4.columns if col not in ['fx_year', 'quarter_num', 'Quarter']]].copy()

print(f"✓ Using Q4 data only: {len(fx_forward_yearly)} years of FX data")

# Process RER data similar to FX data
# Extract year and quarter from Quarter format and create yearly RER data
rer_data['rer_year'] = rer_data['Quarter'].str.extract('(\d{4})').astype(int)
rer_data['quarter_num'] = rer_data['Quarter'].str.extract('Q(\d)').astype(int)

# Get yearly RER data (Q4 only - last quarter of each year)
rer_data_q4 = rer_data[rer_data['quarter_num'] == 4].copy()
rer_yearly = rer_data_q4[['rer_year'] + [col for col in rer_data_q4.columns if col not in ['rer_year', 'quarter_num', 'Quarter']]].copy()

print(f"✓ Using Q4 data only: {len(rer_yearly)} years of RER data")

# Add USD column to RER data (USD always = 1.0 as it's the baseline)
if 'USD' not in rer_yearly.columns:
    rer_yearly['USD'] = 1.0

# Extract currency columns (SP and 5Y rates)
currency_cols = [col for col in fx_forward_yearly.columns if '_SP' in col or '_5Y' in col]
sp_cols = [col for col in currency_cols if '_SP' in col]
fwd5_cols = [col for col in currency_cols if '_5Y' in col]

# Get unique currencies
currencies = sorted(list(set([col.replace('_SP', '').replace('_5Y', '') for col in currency_cols])))

def calculate_cross_rate(deal_rate, fund_rate, deal_currency, fund_currency):
    """Calculate cross rate using the same logic as calculate_fx_rates()"""
    if deal_currency == "USD" and fund_currency == "USD":
        return 1.0
    elif deal_currency == "USD" and fund_currency != "USD":
        return 1.0 / fund_rate if fund_rate != 0 else np.nan
    elif deal_currency != "USD" and fund_currency == "USD":
        return deal_rate
    else:  # Both non-USD
        return deal_rate / fund_rate if fund_rate != 0 else np.nan

def calculate_rer_cross_rate(deal_rer, fund_rer, deal_currency, fund_currency):
    """Calculate RER cross rate: deal_currency_rer / fund_currency_rer"""
    if deal_currency == "USD" and fund_currency == "USD":
        return 1.0
    elif deal_currency == "USD" and fund_currency != "USD":
        return 1.0 / fund_rer if pd.notna(fund_rer) and fund_rer != 0 else np.nan
    elif deal_currency != "USD" and fund_currency == "USD":
        return deal_rer if pd.notna(deal_rer) else np.nan
    else:  # Both non-USD
        return deal_rer / fund_rer if pd.notna(deal_rer) and pd.notna(fund_rer) and fund_rer != 0 else np.nan

# Create all combinations of fund_currency × deal_currency × year
unique_fund_currencies = dta_panel_currency['fund_currency'].unique()
unique_deal_currencies = dta_panel_currency['deal_currency'].unique()
unique_years = sorted(dta_panel_currency['deal_year'].unique())



# Build forward_fx/realized_fx/rer dataset
ecr_acr_data = []

for fund_curr in unique_fund_currencies:
    for deal_curr in unique_deal_currencies:
        for year in unique_years:
            # Skip if currencies are not in FX data
            if (fund_curr != "USD" and f"{fund_curr}_SP" not in fx_forward_yearly.columns) or \
               (deal_curr != "USD" and f"{deal_curr}_SP" not in fx_forward_yearly.columns):
                continue
                
            # For forward_fx calculation, use FX data from beginning of year (year-1)
            fx_year_current = year - 1
            # For realized_fx calculation, use FX data 5 years later
            fx_year_future = year + 4  # 5 years from start of investment year
            
            # For RER calculation, use RER data from beginning of year (year-1) same as forward_fx
            rer_year_current = year - 1
            
            # Check if required FX data exists
            fx_current = fx_forward_yearly[fx_forward_yearly['fx_year'] == fx_year_current]
            fx_future = fx_forward_yearly[fx_forward_yearly['fx_year'] == fx_year_future]
            
            # Check if required RER data exists
            rer_current = rer_yearly[rer_yearly['rer_year'] == rer_year_current]
            
            if len(fx_current) == 0:
                continue
                
            fx_current = fx_current.iloc[0]
            
            # Get spot and forward rates for current year
            if fund_curr == "USD":
                fund_sp_current = 1.0
                fund_fwd5_current = 1.0
            else:
                fund_sp_current = fx_current.get(f"{fund_curr}_SP", np.nan)
                fund_fwd5_current = fx_current.get(f"{fund_curr}_5Y", np.nan)
                
            if deal_curr == "USD":
                deal_sp_current = 1.0
                deal_fwd5_current = 1.0
            else:
                deal_sp_current = fx_current.get(f"{deal_curr}_SP", np.nan)
                deal_fwd5_current = fx_current.get(f"{deal_curr}_5Y", np.nan)
            
            # Get RER rates for current year
            panel_rer = np.nan
            if len(rer_current) > 0:
                rer_current = rer_current.iloc[0]
                
                # Get RER values for deal and fund currencies
                if fund_curr == "USD":
                    fund_rer_current = 1.0
                else:
                    fund_rer_current = rer_current.get(fund_curr, np.nan)
                    
                if deal_curr == "USD":
                    deal_rer_current = 1.0
                else:
                    deal_rer_current = rer_current.get(deal_curr, np.nan)
                
                # Calculate RER cross rate (deal_rer / fund_rer)
                panel_rer = calculate_rer_cross_rate(deal_rer_current, fund_rer_current, deal_curr, fund_curr)
            
            # Calculate cross rates for current year
            sp_rate_current = calculate_cross_rate(deal_sp_current, fund_sp_current, deal_curr, fund_curr)
            fwd5_rate_current = calculate_cross_rate(deal_fwd5_current, fund_fwd5_current, deal_curr, fund_curr)
            
            # Calculate forward_fx
            if pd.notna(sp_rate_current) and pd.notna(fwd5_rate_current) and sp_rate_current != 0 and sp_rate_current > 0 and fwd5_rate_current > 0:
                try:
                    ecr = ((fwd5_rate_current / sp_rate_current) ** (1/5) - 1)*100
                except (ValueError, OverflowError):
                    ecr = np.nan
            else:
                ecr = np.nan
            
            # Calculate realized_fx (if future data exists)
            acr = np.nan
            if len(fx_future) > 0:
                fx_future = fx_future.iloc[0]
                
                if fund_curr == "USD":
                    fund_sp_future = 1.0
                else:
                    fund_sp_future = fx_future.get(f"{fund_curr}_SP", np.nan)
                    
                if deal_curr == "USD":
                    deal_sp_future = 1.0
                else:
                    deal_sp_future = fx_future.get(f"{deal_curr}_SP", np.nan)
                
                sp_rate_future = calculate_cross_rate(deal_sp_future, fund_sp_future, deal_curr, fund_curr)
                
                if pd.notna(sp_rate_current) and pd.notna(sp_rate_future) and sp_rate_current != 0 and sp_rate_current > 0 and sp_rate_future > 0:
                    try:
                        acr = ((sp_rate_future / sp_rate_current) ** (1/5) - 1)*100
                    except (ValueError, OverflowError):
                        acr = np.nan
            
            ecr_acr_data.append({
                'fund_currency': fund_curr,
                'deal_currency': deal_curr,
                'year': year,
                'forward_fx': ecr,
                'realized_fx': acr,
                'rer': panel_rer,
                'sp_rate_current': sp_rate_current,
                'fwd5_rate_current': fwd5_rate_current
            })

# Convert to DataFrame
currency_ecr_acr = pd.DataFrame(ecr_acr_data)

# save the currency_ecr_acr to a csv
currency_ecr_acr.to_csv("Output_data/currency_forward_measures.csv", index=False)

# Merge forward_fx/realized_fx/rer data into panel
dta_panel_currency = dta_panel_currency.merge(
    currency_ecr_acr[['fund_currency', 'deal_currency', 'year', 'forward_fx', 'realized_fx', 'rer']],
    left_on=['fund_currency', 'deal_currency', 'deal_year'],
    right_on=['fund_currency', 'deal_currency', 'year'],
    how='left'
).drop('year', axis=1)

# create the simplified currency variables
major_currencies = ['EUR', 'GBP', 'USD', 'CHF', 'INR', 'RUB', 'CAD', 'CNY', 'KRW', 'CPY']
dta_panel_currency['deal_currency_simplified'] = np.where(
    dta_panel_currency['deal_currency'].isin(major_currencies), 
    dta_panel_currency['deal_currency'], 
    'Other'
)
dta_panel_currency['fund_currency_simplified'] = np.where(
    dta_panel_currency['fund_currency'].isin(major_currencies), 
    dta_panel_currency['fund_currency'], 
    'Other'
)

# Create complete fund-year-currency panel structure
print("Creating complete fund-year-currency panel...")

# Load currency validity periods for validation
print("Loading currency validity periods...")
currency_periods = load_currency_validity_periods()
currency_validity_dict = get_currency_validity_dict(currency_periods)
print(f"✓ Loaded validity periods for {len(currency_validity_dict)} currencies")

# Get fund characteristics from original data (drop NaN vintage years)
fund_chars = dta_deal.dropna(subset=['vintage', 'fund_id']).groupby('fund_id').agg({
    'firm_id': 'first',
    'vintage': 'first', 
    'Fund Currency': 'first',
    'firmcountry': 'first',
    'fund_size_usd_mn': 'first',
    'fund_number_overall': 'first',
    'fund_number_series': 'first',
    'buyout_fund_size': 'first',
    'DEAL ID': 'count',
    'fund_country': 'first',
    'carried_interest_pct': 'first',
    'hurdle_rate_pct': 'first'
}).rename(columns={
    'DEAL ID': 'fund_n_deals'
}).reset_index()


# Analysis of fund currency space construction:
# I've created a mapping from geographic regions (based on 'geographic focus') to their corresponding currencies.
# For broad regions like 'global', 'international', and 'other', I include all available deal currencies 
# to capture their global investment scope. The final fund currency space for the fund-currency-year panel 
# is constructed by taking the intersection of (1) geographic currencies based on fund's regional focus, 
# (2) currencies that the firm has actually invested in, and then plus the currencies that the specific fund has invested in.
# Region-currency mapping is loaded from pre-constructed CSV file

# Create region-currency mapping
create_region_country_currency_mapping()

region_country_currency_df = pd.read_csv("Output_data/region_country_currency_mapping.csv")

# CONSTRAINT: Limit currency space to actual deal currencies only
# The currency space should not be larger than currencies that actually appear in deals
actual_deal_currencies = set(dta_deal['Deal Currency'].dropna().unique())
print(f"  Constraining currency space to actual deal currencies: {len(actual_deal_currencies)} currencies")
print(f"   Sample deal currencies: {sorted(list(actual_deal_currencies))[:10]}")

# Filter region-country-currency mapping to only include actual deal currencies
original_currencies = region_country_currency_df['currency'].nunique()
region_country_currency_df = region_country_currency_df[
    region_country_currency_df['currency'].isin(actual_deal_currencies)
].copy()

filtered_currencies = region_country_currency_df['currency'].nunique()
print(f"✓ Filtered currency mapping: {original_currencies} → {filtered_currencies} currencies")
print(f"   Excluded {original_currencies - filtered_currencies} currencies not in actual deals")

########################################
# Create time-aware region-country-currency mapping
# This will be used to build the comprehensive country-currency spaces per fund-year

# First, let's create a country-currency mapping by year for time-aware lookups
country_currency_map_year = (
    region_country_currency_df[['country', 'year', 'currency']]
    .dropna()
    .drop_duplicates()
)

# STEP 1: Build helper functions for time-aware country spaces
def build_geo_inferred_countries(fund_geographic_info, region_country_currency_df):
    """
    Build geo-inferred countries per fund-year from geographic focus.
    Returns [fund_id, year, country_name]
    """
    geo_countries = []
    
    for _, fund_row in fund_geographic_info.iterrows():
        fund_id = fund_row['fund_id']
        geographic_focus = fund_row['geographic_focus']
        
        if pd.isna(geographic_focus):
            continue
            
        # Split regions by comma
        regions = [region.strip() for region in str(geographic_focus).split(',') if region.strip()]
        
        # For each region, get countries across all years
        fund_region_data = region_country_currency_df[
            region_country_currency_df['region'].isin(regions)
        ][['country', 'year']].drop_duplicates()
        
        for _, row in fund_region_data.iterrows():
            geo_countries.append({
                'fund_id': fund_id,
                'year': row['year'], 
                'country_name': row['country']
            })
    
    return pd.DataFrame(geo_countries).drop_duplicates()

def explode_actual_countries(fund_geographic_info):
    """
    Explode actual countries per fund across all years.
    Returns [fund_id, year, country_name]
    """
    actual_countries = []
    
    for _, fund_row in fund_geographic_info.iterrows():
        fund_id = fund_row['fund_id']
        countries = fund_row['fund_actual_countries']
        
        if countries is None or len(countries) == 0:
            continue
            
        # Get all years from country_currency_map for these countries
        country_years = country_currency_map_year[
            country_currency_map_year['country'].isin(countries)
        ][['country', 'year']].drop_duplicates()
        
        for _, row in country_years.iterrows():
            actual_countries.append({
                'fund_id': fund_id,
                'year': row['year'],
                'country_name': row['country']
            })
    
    return pd.DataFrame(actual_countries).drop_duplicates()

def compute_fund_windows(dta_deal, fund_chars):
    """
    Compute fund time windows using corrected logic:
    Default window: vintage to vintage + 11
    Expand only when deals occur outside this default window
    Only include funds with at least one deal
    
    Logic:
    - min_year = min(vintage, min_deal_year) if deals exist before vintage
    - max_year = max(vintage + 11, max_deal_year) if deals exist after vintage + 11
    - Only include funds that appear in deal data
    """
    # Get actual year ranges for each fund from deal data
    fund_year_ranges = dta_deal.groupby('fund_id')['deal_year'].agg(['min', 'max']).reset_index()
    fund_year_ranges.columns = ['fund_id', 'min_deal_year', 'max_deal_year']
    # set a global upper bound to be the max year of the deal data
    max_year = dta_deal['deal_year'].max()
    # Merge with fund characteristics to get vintage year
    # Use inner join to only include funds with deals
    fund_windows = fund_year_ranges.merge(
        fund_chars[['fund_id', 'vintage']], 
        on='fund_id', 
        how='inner'  # Only include funds with at least one deal
    )
    
    print(f"✓ Fund window computation: {len(fund_windows):,} funds with deals (excluded funds without deals)")
    
    # Apply corrected time window logic
    # Default window: vintage to vintage + 11
    fund_windows['default_min'] = fund_windows['vintage']
    fund_windows['default_max'] = fund_windows['vintage'] + 11
    
    # Expand window only when deals occur outside default range
    fund_windows['min_year'] = fund_windows[['default_min', 'min_deal_year']].min(axis=1, skipna=True)
    fund_windows['max_year'] = fund_windows[['default_max', 'max_deal_year']].max(axis=1, skipna=True)
    # set the max year to be the global upper bound
    fund_windows['max_year'] = fund_windows['max_year'].apply(lambda x: min(x, max_year))
    
    return fund_windows[['fund_id', 'min_year', 'max_year']]

# STEP 2: Get fund-level geographic focus and actual countries invested
fund_geographic_info = dta_deal.dropna(subset=['fund_id', 'geographic_focus']).groupby('fund_id').agg({
    'geographic_focus': 'first',  # Assuming consistent within fund
    'Deal Currency': lambda x: list(x.unique()),  # All currencies this fund actually invested in
    'TARGET COMPANY COUNTRY': lambda x: list(x.unique()),  # All countries this fund actually invested in
    'firm_id': 'first'
}).reset_index()

fund_geographic_info = fund_geographic_info.rename(columns={
    'Deal Currency': 'fund_actual_currencies',
    'TARGET COMPANY COUNTRY': 'fund_actual_countries',
    'geographic_focus': 'geographic_focus'
})


# STEP 3: Build time-aware country spaces using the comprehensive approach

print("Building time-aware country spaces...")

# 3.1) Geo-inferred countries per fund-year (G_f,y)
geo_inferred_countries = build_geo_inferred_countries(fund_geographic_info, region_country_currency_df)
print(f"✓ Built geo-inferred countries: {len(geo_inferred_countries):,} fund-year-country combinations")

# construct the fund_geographic_countries
fund_geographic_countries = geo_inferred_countries.copy().drop_duplicates(['fund_id', 'country_name']).drop(columns=['year'])


# merge with fund_chars
fund_geographic_countries = fund_geographic_countries.merge(
    fund_chars[['fund_id', 'fund_country']],
    on='fund_id',
    how='left'
)
# Clean the fund country name using centralized standardizer
fund_geographic_countries['fund_country'] = fund_geographic_countries['fund_country'].apply(
    lambda x: standardize_country_name(x, source="fund")
)
# save the fund_geographic_countries to a csv
fund_geographic_countries.to_csv("Output_data/fund_geographic_countries.csv", index=False)


# 3.2) Actual countries per fund-year (A_f,y) 
actual_countries_year = explode_actual_countries(fund_geographic_info)
print(f"✓ Built actual countries by year: {len(actual_countries_year):,} fund-year-country combinations")

# 3.3) Firm-invested countries per fund-year (I_f,y)
# Note: For firm-invested countries, we include ALL countries the firm has ever invested in
# across all years, then expand to all fund-years within the time window

# First, get all countries each firm has ever invested in
firm_all_countries = (
    dta_deal[['firm_id', 'TARGET COMPANY COUNTRY']]
    .rename(columns={'TARGET COMPANY COUNTRY': 'country_name'})
    .dropna(subset=['country_name'])
    .drop_duplicates()
)

print(f"✓ Built firm investment universe: {len(firm_all_countries):,} firm-country combinations")

# Now expand this to all fund-years for each firm
firm_invested_countries_year = []

# Get fund-firm mapping
fund_firm_mapping = dta_deal[['fund_id', 'firm_id']].drop_duplicates()

for _, row in fund_firm_mapping.iterrows():
    fund_id = row['fund_id']
    firm_id = row['firm_id']
    
    # Get all countries this firm has ever invested in
    firm_countries = firm_all_countries[
        firm_all_countries['firm_id'] == firm_id
    ]['country_name'].tolist()
    
    # Get all years from the region mapping (to expand across all possible years)
    all_years = region_country_currency_df['year'].unique()
    
    # Create combinations for all years
    for year in all_years:
        for country in firm_countries:
            firm_invested_countries_year.append({
                'fund_id': fund_id,
                'year': year,
                'country_name': country
            })

# Convert to DataFrame and remove duplicates
firm_invested_countries_year = pd.DataFrame(firm_invested_countries_year).drop_duplicates()
print(f"✓ Built firm-invested countries (all years): {len(firm_invested_countries_year):,} fund-year-country combinations")

# 3.4) Apply fund time windows
fund_windows = compute_fund_windows(dta_deal, fund_chars)
print(f"✓ Computed fund time windows for {len(fund_windows):,} funds")

# Filter all country spaces by fund time windows
geo_inferred_countries = geo_inferred_countries.merge(
    fund_windows, on='fund_id', how='left'
).query('min_year <= year <= max_year')[['fund_id', 'year', 'country_name']]

actual_countries_year = actual_countries_year.merge(
    fund_windows, on='fund_id', how='left'  
).query('min_year <= year <= max_year')[['fund_id', 'year', 'country_name']]

firm_invested_countries_year = firm_invested_countries_year.merge(
    fund_windows, on='fund_id', how='left'
).query('min_year <= year <= max_year')[['fund_id', 'year', 'country_name']]

# 3.5) Build time-aware country space: C_f,y = (G_f,y ∩ I_f,y) ∪ A_f,y
print("Building comprehensive country space: C_f,y = (G ∩ I) ∪ A...")

# Intersection: G ∩ I  
G_cap_I = geo_inferred_countries.merge(
    firm_invested_countries_year, 
    on=['fund_id', 'year', 'country_name'], 
    how='inner'
)

# Union: (G ∩ I) ∪ A
time_aware_country_space = pd.concat([G_cap_I, actual_countries_year], ignore_index=True).drop_duplicates()
print(f"✓ Built time-aware country space: {len(time_aware_country_space):,} fund-year-country combinations")


# STEP 4: Build deal-currency/country space and fund-currency-year aggregations

print("Building deal-currency/country space...")

# 4.1) Create continuous deal-currency/country space across years
# This creates all possible fund-currency-country-year combinations within each fund's time window
# to enable measurement of deal appearance variation across years

print("Building continuous deal-currency/country space...")

# Step 1: Get all fund-currency combinations that could potentially exist
# Start with the time-aware country space and get valid currencies for each country-year
continuous_space = time_aware_country_space.merge(
    country_currency_map_year, 
    left_on=['country_name', 'year'], 
    right_on=['country', 'year'], 
    how='left',
    suffixes=('', '_map')
).dropna(subset=['currency'])  # Only keep combinations where currency mapping exists

# Select only the columns we need and rename for consistency
continuous_space = continuous_space[['fund_id', 'year', 'country_name', 'currency']].rename(
    columns={'country_name': 'country'}
)

print(f"✓ Base continuous space: {len(continuous_space):,} fund-year-country-currency combinations")

# Step 2: Filter by fund time windows to ensure we only include valid years for each fund
continuous_space = continuous_space.merge(
    fund_windows, on='fund_id', how='left'
)

# Apply time window filter
continuous_space = continuous_space[
    (continuous_space['year'] >= continuous_space['min_year']) & 
    (continuous_space['year'] <= continuous_space['max_year'])
][['fund_id', 'year', 'currency', 'country']]

print(f"✓ Filtered by fund time windows: {len(continuous_space):,} combinations")

# Step 3: Create the continuous deal-currency/country space
# This represents ALL possible investment opportunities for each fund across time
deal_currency_country_space = continuous_space.copy()

# Add a synthetic deal_id for tracking (since this is now continuous, not tied to actual deals)
deal_currency_country_space['deal_id'] = (
    deal_currency_country_space['fund_id'].astype(str) + '_' +
    deal_currency_country_space['year'].astype(str) + '_' +  
    deal_currency_country_space['currency'] + '_' +
    deal_currency_country_space['country']
)

print(f"✓ Built continuous deal-currency/country space: {len(deal_currency_country_space):,} combinations")
print(f"  This enables measuring deal appearance variation across {deal_currency_country_space['year'].nunique()} years")

# merge the deal_currency_country_space with the fund_chars
fund_country_space = deal_currency_country_space.merge(
    fund_chars, on='fund_id', how='left'
).rename(columns={
    'currency': 'deal_currency',
    'country': 'deal_country'
})

################# save the fund_country_space to a csv for constrcuting the fund-country-year panel
fund_country_space.to_csv("Output_data/fund_country_space.csv", index=False)
########################################################

# 4.2) Fund-currency-year equal-weight indicators (union over deals in that fund/currency/year)
fc_year_countries = deal_currency_country_space[['fund_id', 'year', 'currency', 'country']].drop_duplicates()

fc_counts = fc_year_countries.groupby(['fund_id', 'year', 'currency']).size().rename('n').reset_index()
fund_currency_year_country_weights = fc_year_countries.merge(
    fc_counts, 
    on=['fund_id', 'year', 'currency'], 
    how='left'
)
fund_currency_year_country_weights['weight'] = 1.0 / fund_currency_year_country_weights['n']
fund_currency_year_country_weights = fund_currency_year_country_weights.drop(columns='n')

print(f"✓ Built fund-currency-year country weights: {len(fund_currency_year_country_weights):,} combinations")

# STEP 5: Macro controls integration

print("Loading and processing macro controls...")

# 5.1) Load macro_controls.csv
macro_controls = pd.read_csv("Output_data/macro_controls.csv")
print(f"✓ Loaded macro controls: {len(macro_controls):,} country-year observations")



# 5.3) Merge macro controls into fund-currency-year-country weights (keep NAs)
print("Merging macro controls with country weights...")

# Prepare macro controls for merge (drop currency column to avoid conflict)
macro_for_merge = macro_controls.rename(columns={'country_name': 'country'}).drop(columns=['currency'])

w_macro = fund_currency_year_country_weights.merge(
    macro_for_merge, 
    on=['country', 'year'], 
    how='left'
)
print(f"✓ Merged macro data: {len(w_macro):,} fund-currency-year-country observations")

# 5.4) Aggregate to fund-currency-year level
print("Aggregating macro controls to fund-currency-year level...")

# STEP 1: Calculate global currency-year deal totals (same for all funds)
print("Calculating global currency-year deal totals...")
global_deals = macro_controls.groupby(['currency', 'year']).agg({
    'all_deals_by_country_year': 'sum',
    'all_deals_by_country_year_tm1': 'sum'
}).reset_index()
global_deals = global_deals.rename(columns={
    'all_deals_by_country_year': 'all_deals_by_currency_year', 
    'all_deals_by_country_year_tm1': 'all_deals_by_currency_year_tm1'
})
print(f"✓ Global deal totals: {len(global_deals):,} currency-year combinations")

# STEP 2: Aggregate fund-specific weighted macro variables (GDP, interest rates)
print("Aggregating fund-specific weighted macro variables...")
fund_macro_agg_funcs = {
    'gdp_growth': lambda s: s.dropna().mean() if len(s.dropna()) > 0 else np.nan,
    'gdp_growth_tm1': lambda s: s.dropna().mean() if len(s.dropna()) > 0 else np.nan,
    'interest_rate': lambda s: s.dropna().mean() if len(s.dropna()) > 0 else np.nan,
    'interest_rate_tm1': lambda s: s.dropna().mean() if len(s.dropna()) > 0 else np.nan
}

fund_macro_agg = w_macro.groupby(['fund_id', 'year', 'currency']).agg(fund_macro_agg_funcs).reset_index()
fund_macro_agg = fund_macro_agg.rename(columns={
    'gdp_growth': 'avg_gdp_growth',
    'gdp_growth_tm1': 'avg_gdp_growth_tm1', 
    'interest_rate': 'interest_rate',
    'interest_rate_tm1': 'interest_rate_tm1'
})
print(f"✓ Fund-specific macro aggregation: {len(fund_macro_agg):,} fund-currency-year observations")

# STEP 3: Merge global deal totals with fund-specific macro data
print("Merging global deal totals with fund macro data...")
macro_agg = fund_macro_agg.merge(
    global_deals,
    on=['currency', 'year'],
    how='left'
)

print(f"✓ Aggregated macro controls: {len(macro_agg):,} fund-currency-year observations")

# Show sample of aggregated macro data
print("Sample of aggregated macro controls:")
print(macro_agg.head())


# STEP 6: Create updated complete panel with time-aware country spaces

print("Creating complete panel with time-aware logic...")

# Get all currencies that each FIRM has invested in across ALL years and deals
# This represents the firm's complete currency investment universe
firm_currencies_invested = dta_deal.dropna(subset=['firm_id']).groupby('firm_id')['Deal Currency'].unique().reset_index()
firm_currencies_invested = firm_currencies_invested.rename(columns={'Deal Currency': 'firm_deal_currencies_list'})

# 6.1) Build fund characteristics with firm currencies (updated approach)
fund_full_info = fund_chars.merge(firm_currencies_invested, on='firm_id', how='inner')
print(f"✓ Built fund characteristics with firm currencies: {len(fund_full_info):,} fund-firm combinations")

# 6.2) Merge with fund windows (already computed)
fund_full_info = fund_full_info.merge(fund_windows, on='fund_id', how='left')
print(f"✓ Merged with fund windows: {len(fund_full_info):,} fund-year combinations")

# 6.3) Use time-aware currency spaces from deal-currency/country space
print("Building panel from time-aware deal-currency spaces...")



########################################################
# Get all fund-currency-year combinations from our time-aware analysis
fund_currency_year_combinations = deal_currency_country_space[
    ['fund_id', 'year', 'currency']
].drop_duplicates()
# fund_currency_year_panel: Merge with fund characteristics to get the fund characteristics
fund_currency_year_panel = fund_currency_year_combinations.merge(
    fund_full_info[['fund_id', 'firm_id', 'vintage', 'Fund Currency','fund_size_usd_mn','fund_n_deals','fund_number_overall',
    'fund_number_series','buyout_fund_size','carried_interest_pct','hurdle_rate_pct']], 
    on='fund_id', 
    how='left'
).rename(columns={
    'year': 'deal_year',
    'currency': 'deal_currency',
    'Fund Currency': 'fund_currency'
})

# Add currency pair identifier
fund_currency_year_panel['currency_pair'] = fund_currency_year_panel['deal_currency'] + ' ' + fund_currency_year_panel['fund_currency']


# STEP 7: Merge FX data and macro controls into the panel

print("Merging FX data and macro controls...")

# 7.1) fund_currency_year_panel: Merge with forward_fx/realized_fx/rer data
fund_currency_year_panel = fund_currency_year_panel.merge(
    currency_ecr_acr[['fund_currency', 'deal_currency', 'year', 'forward_fx', 'realized_fx', 'rer']],
    left_on=['fund_currency', 'deal_currency', 'deal_year'],
    right_on=['fund_currency', 'deal_currency', 'year'],
    how='left'
).drop('year', axis=1)



# 7.3) Merge macro controls into currency panel (preserve rows & NaNs per specifications)
print("Merging macro controls into currency panel...")

# Currency panel: Use aggregated macro controls (already at fund-currency-year level)
initial_currency_panel_count = len(fund_currency_year_panel)
fund_currency_year_panel = fund_currency_year_panel.merge(
    macro_agg, 
    left_on=['fund_id', 'deal_year', 'deal_currency'],
    right_on=['fund_id', 'year', 'currency'], 
    how='left'
).drop(['year', 'currency'], axis=1)

print(f"✓ Merged macro controls into currency panel: {len(fund_currency_year_panel):,} observations")
assert len(fund_currency_year_panel) == initial_currency_panel_count, "Currency panel size should not change after macro merge"


# STEP 8: Merge with actual deal activity (sum deal sizes and count deals)
print("Merging actual deal activity into currency panel...")

# Use the original aggregated deal activity data (constructed at lines 21-30)
# This contains the aggregated deal sizes and counts per fund-currency-year
# NOTE: Since we now have a CONTINUOUS space, many combinations will have zero deals

original_activity_currency = dta_panel_currency[['fund_id', 'deal_currency', 'deal_year', 'sum_deal_size_usd_mn', 'n_deals']].copy()

# Left join: keep all fund-currency-year combinations from continuous space
# Missing activity will be filled with 0 (representing no deals in that fund-currency-year)
fund_currency_year_panel = fund_currency_year_panel.merge(
    original_activity_currency,
    on=['fund_id', 'deal_currency', 'deal_year'],
    how='left'
)

print(f"✓ Merged actual activity into currency panel: {len(fund_currency_year_panel):,} observations")
print(f"  Continuous space enables measuring deal appearance variation across years")

# STEP 9: Final panel preparation and validation
print("Finalizing currency panel...")

# Fill missing deal activity with 0 (fund didn't invest in that currency that year)
fund_currency_year_panel['n_deals'] = fund_currency_year_panel['n_deals'].fillna(0)

fund_currency_year_panel['all_deals_by_currency_year'] = fund_currency_year_panel['all_deals_by_currency_year'].fillna(0)
fund_currency_year_panel['all_deals_by_currency_year_tm1'] = fund_currency_year_panel['all_deals_by_currency_year_tm1'].fillna(0)

# Note: Macro variables (avg_gdp_growth, avg_gdp_growth_tm1, avg_interest_rate, avg_interest_rate_tm1) are kept as NaN when no data available, per specifications

print(f"✓ Final currency panel size: {len(fund_currency_year_panel):,} observations")

# Use currency panel as the main dta_panel for backward compatibility
dta_panel = fund_currency_year_panel.copy()
print(f"✓ Set main dta_panel to currency panel: {len(dta_panel):,} observations")

# Add simplified currency variables
major_currencies = ['EUR', 'GBP', 'USD', 'CHF', 'INR', 'RUB', 'CAD', 'CNY', 'KRW', 'JPY']
dta_panel['deal_currency_simplified'] = np.where(
    dta_panel['deal_currency'].isin(major_currencies), 
    dta_panel['deal_currency'], 
    'Other'
)
dta_panel['fund_currency_simplified'] = np.where(
    dta_panel['fund_currency'].isin(major_currencies), 
    dta_panel['fund_currency'], 
    'Other'
)
dta_panel['fund_age'] = dta_panel['deal_year'] - dta_panel['vintage']
# build a dummy variable is_fund_currency taking value 1 if the deal currency is the same as the fund currency
dta_panel['is_fund_currency'] = np.where(dta_panel['deal_currency'] == dta_panel['fund_currency'], 1, 0)

# build a variable: deal_ratio taking value of n_deals / fund_n_deals
dta_panel['deal_ratio'] = dta_panel['n_deals'] / dta_panel['fund_n_deals']
# Only create logs for variables that actually exist
available_ln_vars = []
potential_ln_vars = ['n_deals', 'all_deals_by_currency_year', 'all_deals_by_currency_year_tm1','fund_size_usd_mn','fund_n_deals','fund_number_overall','fund_number_series']

for var in potential_ln_vars:
    if var in dta_panel.columns:
        available_ln_vars.append(var)
        dta_panel[f'ln_{var}'] = np.log(dta_panel[var] + 1)
        print(f"✓ Created ln_{var}")
    else:
        print(f"⚠️  Variable {var} not found in panel")

print(f"✓ Created logarithm variables for: {available_ln_vars}")

# Calculate cumulative currency experience
print("Calculating cumulative currency experience...")

def calculate_cumulative_currency_experience(df):
    """
    Calculate cumulative currency experience for each firm.
    For each fund-currency-year observation, count how many previous deals 
    (cumulative sum of n_deals) the firm has invested in that currency up to current year.
    """
    # Create a copy to avoid modifying original data
    df_work = df.copy()
    
    # Initialize the experience variable
    df_work['prev_currency_experience'] = 0
    
    # Sort by firm_id, deal_currency, and deal_year
    df_work = df_work.sort_values(['firm_id', 'deal_currency', 'deal_year'])
    
    # For each firm-currency combination, calculate cumulative experience
    for (firm_id, deal_currency), group in df_work.groupby(['firm_id', 'deal_currency']):
        # Create a temporary DataFrame for this firm-currency combination
        temp_group = group.copy()
        
        # For each observation in this firm-currency group
        for idx in temp_group.index:
            current_year = df_work.loc[idx, 'deal_year']
            
            # Get all previous observations for this firm-currency up to (but not including) current year
            previous_obs = temp_group[temp_group['deal_year'] < current_year]
            
            # Sum all previous n_deals for this firm-currency combination
            prev_experience = previous_obs['n_deals'].sum()
            df_work.loc[idx, 'prev_currency_experience'] = prev_experience
    
    return df_work

def calculate_time_to_latest_deal(df):
    """
    Calculate time to latest deal for each fund.
    For each fund-currency-year observation, count how many years 
    the fund hasn't invested in that currency up to current year.
    No previous deals in this currency - set to 999.
    """
    # Create a copy to avoid modifying original data
    df_work = df.copy()
    
    # Initialize the time to latest deal variable as 999
    df_work['time_to_latest_deal'] = 999
    
    # Sort by firm_id, deal_currency, and deal_year
    df_work = df_work.sort_values(['fund_id', 'deal_currency', 'deal_year'])
    
    # For each firm-currency combination, calculate time since latest deal
    for (fund_id, deal_currency), group in df_work.groupby(['fund_id', 'deal_currency']):
        # Create a temporary DataFrame for this firm-currency combination
        temp_group = group.copy()
        
        # Find all years where this firm had deals in this currency
        deal_years = temp_group[temp_group['n_deals'] > 0]['deal_year'].tolist()
        
        # For each observation in this firm-currency group
        for idx in temp_group.index:
            current_year = df_work.loc[idx, 'deal_year']
            
            # Find the most recent deal year before current year
            previous_deal_years = [year for year in deal_years if year < current_year]
            
            if previous_deal_years:
                # Calculate years since the most recent deal
                latest_deal_year = max(previous_deal_years)
                time_since_latest = current_year - latest_deal_year
                df_work.loc[idx, 'time_to_latest_deal'] = time_since_latest
            else:
                # No previous deals in this currency - set to a high value or NaN
                df_work.loc[idx, 'time_to_latest_deal'] = 999
    
    return df_work

def calculate_previous_invested_years(df):
    """
    Calculate previous invested years for each fund.
    For each fund-currency-year observation, count how many years 
    the fund has invested in that currency up to current year.
    """
    df_work = df.copy()
    
    # Initialize the previous invested years variable
    df_work['previous_invested_years'] = 0
    
    # Sort by fund_id, deal_currency, and deal_year
    df_work = df_work.sort_values(['fund_id', 'deal_currency', 'deal_year'])
    
    # For each fund-currency combination, calculate previous invested years
    for (fund_id, deal_currency), group in df_work.groupby(['fund_id', 'deal_currency']):
        # Create a temporary DataFrame for this fund-currency combination
        temp_group = group.copy()
        
        # For each observation in this fund-currency group
        for idx in temp_group.index:
            current_year = df_work.loc[idx, 'deal_year']
            
            # Get all previous observations for this fund-currency up to (but not including) current year
            previous_obs = temp_group[(temp_group['deal_year'] < current_year) & (temp_group['n_deals'] > 0)]
            
            # Sum all previous years with deals for this fund-currency combination
            previous_invested_years = previous_obs['deal_year'].nunique()
            df_work.loc[idx, 'previous_invested_years'] = previous_invested_years
    
    return df_work

# Apply currency experience calculation
dta_panel = calculate_cumulative_currency_experience(dta_panel)
print(f"✓ Added currency experience variable to currency panel dataset")

# Apply time to latest deal calculation
#dta_panel = calculate_time_to_latest_deal(dta_panel)
#print(f"✓ Added time to latest deal variable to currency panel dataset")

# Apply previous invested years calculation
#dta_panel = calculate_previous_invested_years(dta_panel)
#print(f"✓ Added previous invested years variable to currency panel dataset")

# add the firm_currency from the fund_full_info to the dta_panel based on firm_id
dta_panel = dta_panel.merge(
    fund_full_info[['fund_id', 'firmcountry']].drop_duplicates(),
    on='fund_id',
    how='left'
)
country_currency_lookup = pd.read_csv("Output_data/country_currency_lookup.csv")
country_currency_lookup.rename(columns={
    'firmcountry': 'firmcountry',
    'currency_lookup': 'firm_currency',
    'vintage': 'deal_year'
}, inplace=True)
dta_panel = dta_panel.merge(
    country_currency_lookup[['firmcountry', 'firm_currency', 'deal_year']],
    left_on=['firmcountry','deal_year'],
    right_on=['firmcountry','deal_year'],
    how='left'
)
print(f"✓ Added firm currency information to panel: {len(dta_panel):,} observations")

# construct a dummy variable is_firm_currency taking value 1 if the deal currency is the same as the firm currency
dta_panel['is_firm_currency'] = np.where(dta_panel['deal_currency'] == dta_panel['firm_currency'], 1, 0)

# Final validation and cleanup for currency panel
print("Performing final currency validation...")

dta_panel_validated = validate_and_fix_panel_currencies(
    dta_panel, 
    apply_transitions=True,
    remove_invalid=True
)

# Update the currency panel with validated data
dta_panel = dta_panel_validated.copy()
print(f"✓ Currency panel validation completed")

# STEP 10: Save currency panel and generate country panel
print("Saving currency panel and generating country panel...")

# Save currency panel (main panel for backward compatibility)
dta_panel.to_csv("Output_data/dta_panel.csv", index=False)
print(f"✓ Saved currency panel (main): {len(dta_panel):,} observations")

# Save fund characteristics for reference
fund_full_info.to_csv("Output_data/fund_characteristics.csv", index=False)
print(f"✓ Saved fund characteristics: {len(fund_full_info):,} funds")






