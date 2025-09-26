# This script creates the final fund deals dataset with all necessary variables
import pandas as pd
import numpy as np
from utils.winsorize_utils import winsorize
# Use the comprehensive country name standardizer
import sys
sys.path.append('src')
from utils.country_name_standardizer import standardize_country_name

fund_fx_measure = pd.read_csv("Output_data/fund_fx_measure.csv")
fund_buyout_deals = pd.read_csv("Output_data/fund_buyout_deals.csv")
dta_fund_performance = pd.read_csv("Output_data/dta_fund_performance.csv")

# Select specific columns to avoid conflicts
# From fund_fx_measure: all columns (no conflicts except fund_id)
fx_cols = list(fund_fx_measure.columns)

# From dta_fund_performance: select only performance-specific columns and fund_name
# Avoid overlapping fund characteristics that are already in fund_buyout_deals
perf_cols = ['fund_id', 'fund_name', 'fund_status', 'net_irr_pcent', 'multiple', 
             'benchmark_name', 'net_irr_median', 'net_multiple_median', 
             'excess_return_multiple', 'excess_return_irr']

# Select performance columns that exist in the dataset
existing_perf_cols = [col for col in perf_cols if col in dta_fund_performance.columns]
performance_data = dta_fund_performance[existing_perf_cols]

print(f"Selected {len(fx_cols)} columns from fund_fx_measure")
print(f"Selected {len(existing_perf_cols)} columns from dta_fund_performance: {existing_perf_cols}")

# Now merge these three tables together
dta_fund_deals = fund_buyout_deals.merge(fund_fx_measure, on="fund_id", how="left")
dta_fund_deals = dta_fund_deals.merge(performance_data, on="fund_id", how="left")

print(f"Final merged dataset shape: {dta_fund_deals.shape}")
print(f"Columns in final dataset: {len(dta_fund_deals.columns)}")

# Add necessary column renaming and new variables
dta_fund_deals = dta_fund_deals.rename(columns={"TARGET COMPANY COUNTRY": "deal_country", "FUND COUNTRY": "fund_country"})


# only include DEAL STATUS is "Completed"
#dta_fund_deals = dta_fund_deals[dta_fund_deals["DEAL STATUS"] == "Completed"]


# Apply standardization to both deal_country and fund_country using the standardizer
dta_fund_deals["deal_country"] = dta_fund_deals["deal_country"].apply(
    lambda x: standardize_country_name(x, source="deals") if pd.notna(x) else x
)

dta_fund_deals["fund_country"] = dta_fund_deals["fund_country"].apply(
    lambda x: standardize_country_name(x, source="deals") if pd.notna(x) else x
)

# Standardize firmcountry as well for consistent matching
dta_fund_deals["firmcountry"] = dta_fund_deals["firmcountry"].apply(
    lambda x: standardize_country_name(x, source="deals") if pd.notna(x) else x
)

# construct the is_firm_country column (using standardized countries)
dta_fund_deals["is_firm_country"] = np.where(dta_fund_deals["deal_country"] == dta_fund_deals["firmcountry"], 1, 0)
# construct the is_fund_country column (using standardized countries)
dta_fund_deals["is_fund_country"] = np.where(dta_fund_deals["deal_country"] == dta_fund_deals["fund_country"], 1, 0)
# rename the deal currency and fund currency
dta_fund_deals = dta_fund_deals.rename(columns={"DEAL CURRENCY": "deal_currency", "FUND CURRENCY": "fund_currency"})
# construct the is_firm_currency column (using standardized countries)
dta_fund_deals["is_firm_currency"] = np.where(dta_fund_deals["deal_currency"] == dta_fund_deals["firm_currency"], 1, 0)
# construct the is_fund_currency column (using standardized countries)
dta_fund_deals["is_fund_currency"] = np.where(dta_fund_deals["deal_currency"] == dta_fund_deals["fund_currency"], 1, 0)

# for each fund, calculate firm_country_ratio and fund_country_ratio
dta_fund_deals["firm_country_ratio"] = dta_fund_deals.groupby("fund_id")["is_firm_country"].transform("mean")
dta_fund_deals["fund_country_ratio"] = dta_fund_deals.groupby("fund_id")["is_fund_country"].transform("mean")
dta_fund_deals["firm_currency_ratio"] = dta_fund_deals.groupby("fund_id")["is_firm_currency"].transform("mean")
dta_fund_deals["fund_currency_ratio"] = dta_fund_deals.groupby("fund_id")["is_fund_currency"].transform("mean")

# construct the fund_n_deals_firm column
dta_fund_deals["fund_n_deals_firm"] = dta_fund_deals.groupby("fund_id")["is_firm_country"].transform("sum")

# save the dta_fund_deals table
dta_fund_deals.to_csv("Output_data/dta_fund_deals.csv", index=False)
print("Saved dta_fund_deals.csv")

dta_fund = dta_fund_deals.copy().drop_duplicates(subset=["fund_id"])

# Use fund_currency_ratio for consistent classification instead of foreign_investment_pct
# This avoids issues with filtered FX data affecting classification
cross_currency_funds = dta_fund[dta_fund['fund_currency_ratio'] < 1.0]['fund_id'].unique().tolist()
pd.DataFrame(cross_currency_funds, columns=['fund_id']).to_csv("Output_data/cross_currency_funds.csv", index=False)
# define cross_country_funds
cross_country_funds = dta_fund[dta_fund['fund_country_ratio'] < 1.0]['fund_id'].unique().tolist()
pd.DataFrame(cross_country_funds, columns=['fund_id']).to_csv("Output_data/cross_country_funds.csv", index=False)

# define the non-domestic funds where fund_currency != firm_currency and fund_country != firmcountry
non_domestic_funds = dta_fund[(dta_fund['fund_currency_ratio'] < 1.0) | (dta_fund['fund_country_ratio'] < 1.0)]['fund_id'].unique().tolist()
pd.DataFrame(non_domestic_funds, columns=['fund_id']).to_csv("Output_data/non_domestic_funds.csv", index=False)

# Keep fund status is liquidated or fund vintage is earlier than 2015
dta_fund = dta_fund[(dta_fund["fund_status"]== "Liquidated") | (dta_fund["vintage"] < 2014)]


# winsorize the net_irr_pcent and multiple
winsorize_cols = ["net_irr_pcent", "multiple", "fund_size_usd_mn"]
dta_fund = winsorize(dta_fund, winsorize_cols)

# Excess returns (recalculate with winsorized values)
dta_fund['excess_irr'] = dta_fund['net_irr_pcent'] - dta_fund['net_irr_median']
dta_fund['excess_multiple'] = dta_fund['multiple'] - dta_fund['net_multiple_median']
# winsorize the excess_irr and excess_multiple
winsorize_cols = ["excess_irr", "excess_multiple"]
dta_fund = winsorize(dta_fund, winsorize_cols)

dta_fund['fund_n_currencies'] = dta_fund['n_foreign_currencies'] + 1
# add the logarithm variables to the dta_fund monetary variables
ln_vars = ['total_assigned_size_usd_fund', 'fund_n_deals', 'fund_n_currencies','fund_number_overall','fund_number_series','fund_size_usd_mn']
for var in ln_vars:
    dta_fund[f'ln_{var}'] = np.log(dta_fund[var] + 1)

# Clean up the columns in dta_fund (keep both currency and country variables for comparison)
columns_to_drop = ['DEAL ID', 'DEAL FUND', 'num_funds_in_deal', 'TARGET COMPANY ID', 'TARGET COMPANY',
    'DEAL DATE','DEAL STATUS', 'DEAL TYPES', 'INVESTORS', 'FUNDS', 'DEAL SIZE (CURR. MN)',
    'DEAL MONTH', 'PERIOD', 'deal_year', 'year']
# Only drop columns that actually exist
columns_to_drop = [col for col in columns_to_drop if col in dta_fund.columns]
dta_fund = dta_fund.drop(columns=columns_to_drop)


# COUNTRY-BASED ANALYSIS VARIABLES (equivalent to currency variables)
# construct the diff_firm_country if fund_country != firm_country (as numeric for R compatibility)
dta_fund["diff_firm_country"] = (dta_fund['fund_country'] != dta_fund['firmcountry']).astype(int)
# construct the diff_firm_currency if fund_currency != firm_currency
dta_fund["diff_firm_currency"] = (dta_fund['fund_currency'] != dta_fund['firm_currency']).astype(int)
# Cross-country investment indicator
dta_fund['is_domestic_investment'] = np.where((dta_fund['fund_currency_ratio'] == 1.0) & (dta_fund['fund_country_ratio'] == 1.0), 1, 0)

# create distance_forward_fx and distance_fx_forward_firm
dta_fund["distance_forward_fx"] = np.abs(dta_fund['forward_fx'])
dta_fund["distance_forward_fx_firm"] = np.abs(dta_fund['forward_fx_firm'])


# save the dta_fund table
dta_fund.to_csv("Output_data/dta_fund.csv", index=False)
print("Saved dta_fund.csv")

