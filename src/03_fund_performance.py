# @02_fund_performance.py

import pandas as pd
import numpy as np
from utils.country_name_standardizer import standardize_country_name
#from utils.irr_calculation import compute_irr_static  # keep dynamic import if you use it elsewhere
#from utils.multiple_calculation import compute_multiple_static

INPUT_DETAILS     = "Input_data/Preqin_fund_details.xlsx"
INPUT_PERF        = "Input_data/Preqin_fund_performance.xlsx"
#INPUT_CASHFLOW    = "Input_data/cashflow.csv"
OUTPUT_DTA        = "Output_data/dta_fund_performance.csv"
#OUTPUT_UNMATCHED  = "Output_data/unmatched_benchmark_by_date.csv"

def load_inputs():
    # Only load the columns we actually need to speed up reading
    fund_details_cols = [
        'FUND ID', 'FIRM ID', 'NAME', 'VINTAGE / INCEPTION YEAR', 'ASSET CLASS', 
        'STRATEGY', 'PRIMARY REGION FOCUS', 'FUND SIZE (USD MN)', 'FUND CURRENCY', 
        'COUNTRY', 'FINAL CLOSE SIZE (USD MN)', 'STATUS', 'FUND NUMBER (OVERALL)',
        'FUND NUMBER (SERIES)', 'PE: BUYOUT FUND SIZE', 'CARRIED INTEREST (%)', 'HURDLE RATE (%)'
    ]
    fund_details   = pd.read_excel(INPUT_DETAILS, usecols=fund_details_cols)
    fund_perf_raw  = pd.read_excel(INPUT_PERF)
    
    #cashflow       = pd.read_csv(INPUT_CASHFLOW)
    return fund_details, fund_perf_raw

def build_fund_performance_table(fund_details, fund_perf_raw):
    # Reduce perf table to last observation per fund
    perf = (fund_perf_raw[['FUND ID', 'VINTAGE / INCEPTION YEAR', 'NET IRR (%)','NET MULTIPLE (X)','FUND SIZE (USD MN)','GEOGRAPHIC FOCUS', 'BENCHMARK NAME',
    'MEDIAN BENCHMARK NET IRR (%)', 'MEDIAN BENCHMARK NET MULTIPLE (X)','ASSET CLASS','STRATEGY']]
    .copy())
    # filter out the asset class is not private equity
    perf = perf[perf['ASSET CLASS'] .isin (['Private Equity'])]
    # filter out strategy is not buyout
    perf = perf[perf['STRATEGY'] .isin (['Buyout'])]

    # rename the columns
    perf = perf.rename(columns={
        'FUND ID': 'fund_id',
        'VINTAGE / INCEPTION YEAR': 'vintage',
        'NET IRR (%)': 'net_irr_pcent',
        'NET MULTIPLE (X)': 'multiple',
        'GEOGRAPHIC FOCUS': 'geographic_focus',
        'BENCHMARK NAME': 'benchmark_name',
        'MEDIAN BENCHMARK NET IRR (%)': 'net_irr_median',
        'MEDIAN BENCHMARK NET MULTIPLE (X)': 'net_multiple_median',
    })
    # convert to numeric
    perf['net_irr_pcent'] = pd.to_numeric(perf['net_irr_pcent'], errors='coerce')
    perf['multiple'] = pd.to_numeric(perf['multiple'], errors='coerce')
    perf['net_irr_median'] = pd.to_numeric(perf['net_irr_median'], errors='coerce')
    perf['net_multiple_median'] = pd.to_numeric(perf['net_multiple_median'], errors='coerce')
    # calculate the excess return
    perf['excess_return_multiple'] = perf['multiple'] - perf['net_multiple_median']
    perf['excess_return_irr'] = perf['net_irr_pcent'] - perf['net_irr_median']
    # convert to datetime
    #perf['date_reported'] = pd.to_datetime(perf['date_reported'], errors='coerce')
    #perf = perf.sort_values(['fund_id', 'date_reported'])
    # Use groupby().apply to get the last row per group, including if it's all NA
    #perf = perf.groupby('fund_id', as_index=False, group_keys=False).apply(lambda g: g.tail(1))

    # IRR from cashflow (static) -> backfill net_irr_pcent if missing
    #irr_cashflow = compute_irr_static(cashflow)
    #irr_tab = (irr_cashflow[['fund_id','irr_annual']]
    #           .drop_duplicates('fund_id')
    #           .rename(columns={'irr_annual':'net_irr_pcent_cf'}))
    #perf = perf.merge(irr_tab, on='fund_id', how='left', validate='one_to_one')
    #perf['net_irr_pcent'] = perf['net_irr_pcent'].fillna(perf['net_irr_pcent_cf'])
    #perf.drop(columns=['net_irr_pcent_cf'], inplace=True)

    # Multiples from cashflow (static) -> backfill multiple if missing
    #mult = compute_multiple_static(cashflow)
    #mult = mult[['fund_id','multiple']].drop_duplicates('fund_id').rename(columns={'multiple':'multiple_cf'})
    #perf = perf.merge(mult, on='fund_id', how='left', validate='one_to_one')
    #perf['multiple'] = perf['multiple'].fillna(perf['multiple_cf'])
    #perf.drop(columns=['multiple_cf'], inplace=True)

    # Filter eligible funds
    # rename fund_details
    fund_details = fund_details.rename(columns={
        'FUND ID': 'fund_id',
        'FIRM ID': 'firm_id',
        'NAME': 'fund_name',
        'VINTAGE / INCEPTION YEAR': 'vintage',
        'ASSET CLASS': 'asset_class',
        'STRATEGY': 'strategy',
        'PRIMARY REGION FOCUS': 'primary_region_focus',
        'FUND SIZE (USD MN)': 'fund_size_usd_mn',
        'FUND CURRENCY': 'fund_currency',
        'COUNTRY': 'fund_country',
        'FUND SIZE (USD MN)': 'fund_size_usd_mn',
        'STATUS': 'fund_status',
        'FUND NUMBER (OVERALL)': 'fund_number_overall',
        'FUND NUMBER (SERIES)': 'fund_number_series',
        'PE: BUYOUT FUND SIZE': 'buyout_fund_size',
        'CARRIED INTEREST (%)': 'carried_interest_pct',
        'HURDLE RATE (%)': 'hurdle_rate_pct'
    })

    details = fund_details[[
        'fund_id','firm_id','fund_name','vintage','asset_class','strategy','primary_region_focus',
        'fund_currency','fund_country','fund_size_usd_mn','fund_status','fund_number_overall','fund_number_series',
        'buyout_fund_size','carried_interest_pct','hurdle_rate_pct'
    ]].copy()
    # filter out the missing fund_currency
    details = details[details['fund_currency'].notna()]
    # filter out the asset class is not private equity
    details = details[details['asset_class'].isin(['Private Equity'])]
    # filter out strategy is not buyout
    details = details[details['strategy'].isin(['Buyout'])]
    # Replace "RMB" with "CNY" in the "fund_currency" column
    details['fund_currency'] = details['fund_currency'].replace('RMB', 'CNY')

    # Clean the fund country name using centralized standardizer
    details['fund_country'] = details['fund_country'].apply(
        lambda x: standardize_country_name(x, source="fund")
    )

    
    # Identify duplicate fund names and log them (case-sensitive)
    duplicate_funds = details[details.duplicated(subset=['fund_name'], keep=False)].copy() 
    if not duplicate_funds.empty:
        print("Duplicate fund names found (showing all occurrences):")
        print(duplicate_funds[['fund_id', 'fund_name']].sort_values(by=['fund_name', 'fund_id']))
        duplicate_funds.to_csv("Output_data/duplicate_fund_ids.csv", index=False)
    else:
        print("No duplicate fund names found.")

    # For each fund_name, keep the row with the lowest fund_id
    details = details.sort_values(['fund_name', 'fund_id'])
    details = details.drop_duplicates(subset=['fund_name'], keep='first')


    # filter out the missing net_irr_pcent and multiple
    perf    = perf[perf['net_irr_pcent'].notna() & perf['multiple'].notna()]


    # Merge details + perf (on fund_id & vintage)
    # align the format of fund_id
    details['fund_id'] = pd.to_numeric(details['fund_id'], errors='coerce').astype(int).astype(str)
    perf['fund_id'] = pd.to_numeric(perf['fund_id'], errors='coerce').astype(int).astype(str)
    dta = details.merge(perf, on=['fund_id','vintage'], how='left', validate='one_to_one')

    return dta



def main():
    fund_details, fund_perf_raw = load_inputs()
    dta = build_fund_performance_table(fund_details, fund_perf_raw)
    dta.to_csv(OUTPUT_DTA, index=False)
    print(f"[OK] Wrote {OUTPUT_DTA} with {len(dta)} rows")

if __name__ == "__main__":
    main()




