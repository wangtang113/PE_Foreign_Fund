# This script creates the final fund deals dataset with all necessary variables
import pandas as pd
import numpy as np
from utils.winsorize_utils import winsorize

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
dta_fund_deals = dta_fund_deals.rename(columns={"DEAL CURRENCY": "deal_currency", "FUND CURRENCY": "fund_currency"})


# only include DEAL STATUS is "Completed"
dta_fund_deals = dta_fund_deals[dta_fund_deals["DEAL STATUS"] == "Completed"]

# Keep fund status is liquidated or fund vintage is earlier than 2015
dta_fund_deals = dta_fund_deals[(dta_fund_deals["fund_status"]== "Liquidated") | (dta_fund_deals["vintage"] < 2015)]


# construct the is_firm_currency column
dta_fund_deals["is_firm_currency"] = np.where(dta_fund_deals["deal_currency"] == dta_fund_deals["firm_currency"], 1, 0)
# construct the is_fund_currency column
dta_fund_deals["is_fund_currency"] = np.where(dta_fund_deals["deal_currency"] == dta_fund_deals["fund_currency"], 1, 0)

# for each fund, calculate firm_currency_ratio and fund_currency_ratio
dta_fund_deals["firm_currency_ratio"] = dta_fund_deals.groupby("fund_id")["is_firm_currency"].transform("mean")
dta_fund_deals["fund_currency_ratio"] = dta_fund_deals.groupby("fund_id")["is_fund_currency"].transform("mean")


# save the dta_fund_deals table
dta_fund_deals.to_csv("Output_data/dta_fund_deals.csv", index=False)
print("Saved dta_fund_deals.csv")

dta_fund = dta_fund_deals.copy().drop_duplicates(subset=["fund_id"])


# winsorize the net_irr_pcent and multiple
winsorize_cols = ["net_irr_pcent", "multiple", "fund_size_usd_mn"]
dta_fund = winsorize(dta_fund, winsorize_cols)

# Excess returns (recalculate with winsorized values)
dta_fund['excess_irr'] = dta_fund['net_irr_pcent'] - dta_fund['net_irr_median']
dta_fund['excess_multiple'] = dta_fund['multiple'] - dta_fund['net_multiple_median']
# winsorize the excess_irr and excess_multiple
winsorize_cols = ["excess_irr", "excess_multiple"]
dta_fund = winsorize(dta_fund, winsorize_cols)

# create a new column called is_foreign_investment
dta_fund['is_foreign_investment'] = np.where(dta_fund['foreign_investment_pct'] > 0, 1, 0)
major_currencies = ['EUR', 'GBP', 'USD', 'CHF', 'INR', 'RUB', 'CAD', 'CNY', 'KRW', 'CPY']
dta_fund['fund_currency_simplified'] = np.where(
    dta_fund['fund_currency'].isin(major_currencies), 
    dta_fund['fund_currency'], 
    'Other'
)
dta_fund['fund_n_currencies'] = dta_fund['n_foreign_currencies'] + 1
# add the logarithm variables to the dta_fund monetary variables
ln_vars = ['total_assigned_size_usd_fund', 'fund_n_deals', 'fund_n_currencies','fund_number_overall','fund_number_series','fund_size_usd_mn']
for var in ln_vars:
    dta_fund[f'ln_{var}'] = np.log(dta_fund[var] + 1)

# Clean up the columns in dta_fund
dta_fund = dta_fund.drop(columns=[
    'DEAL ID', 'DEAL FUND', 'num_funds_in_deal', 'TARGET COMPANY ID', 'TARGET COMPANY','TARGET COMPANY COUNTRY',
    'DEAL DATE','DEAL STATUS', 'DEAL TYPES', 'INVESTORS', 'FUNDS', 'deal_currency', 'DEAL SIZE (CURR. MN)',
    'DEAL MONTH', 'PERIOD', 'deal_year', 'year', 'is_firm_currency', 'is_fund_currency'])

# construct the firm_currency_ratio^{+} and firm_currency_ratio^{-}
dta_fund["firm_currency_ratio_plus"] = dta_fund["firm_currency_ratio"]*(dta_fund['forward_fx_firm'] > 0)
dta_fund["firm_currency_ratio_minus"] = dta_fund["firm_currency_ratio"]*(dta_fund['forward_fx_firm'] < 0)

# construct the equal_firm_currency if fund_currency == firm_currency (as numeric for R compatibility)
dta_fund["equal_firm_currency"] = (dta_fund['fund_currency'] == dta_fund['firm_currency']).astype(int)

# save the dta_fund table
dta_fund.to_csv("Output_data/dta_fund.csv", index=False)
print("Saved dta_fund.csv")

# get the list fund_cross_currency for funds with foreign_investment_pct > 0
fund_cross_currency = dta_fund[dta_fund['is_foreign_investment'] == 1]['fund_id'].unique().tolist()
pd.DataFrame(fund_cross_currency, columns=['fund_id']).to_csv("Output_data/fund_cross_currency.csv", index=False)