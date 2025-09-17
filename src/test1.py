import pandas as pd
# load data
dta_fund = pd.read_csv("Output_data/dta_fund.csv")
dta_fund_ccs = dta_fund[dta_fund["is_foreign_investment"] == 1]
# check how many pair of vintage*fund_currency in dta_fund_ccs
print(f"Number of unique vintage years: {len(dta_fund_ccs['vintage'].unique())}")
print(f"Number of unique fund currencies: {len(dta_fund_ccs['fund_currency'].unique())}")

# Get unique combinations of vintage and fund_currency
unique_combinations = dta_fund_ccs[['vintage', 'fund_currency']].drop_duplicates()
print(f"Number of unique vintage-fund_currency combinations: {len(unique_combinations)}")

# Show the actual combinations
print("\nUnique vintage-fund_currency combinations:")
print(unique_combinations.sort_values(['vintage', 'fund_currency']))

# Now I want to know how many observations are estimated in the regression with vintage*fund_currency as the FE and other control variables
# see the regressions in the R script src/R_regression/fund_level.R

print("\n" + "="*80)
print("REGRESSION OBSERVATION ANALYSIS")
print("="*80)

# Load cross-currency fund IDs
fund_cross_currency = pd.read_csv("Output_data/fund_cross_currency.csv")
print(f"Cross-currency funds: {len(fund_cross_currency)}")

# Create the cross-currency sample (same as R script)
dta_fund_ccs = dta_fund[dta_fund["fund_id"].isin(fund_cross_currency["fund_id"])]
print(f"Cross-currency observations: {len(dta_fund_ccs)}")

# Define the required variables for each regression specification
base_vars = ['net_irr_pcent', 'multiple', 'fund_currency_ratio', 'firm_currency_ratio', 'forward_fx']
control_vars = ['ln_fund_size_usd_mn', 'ln_fund_n_deals', 'ln_n_currencies', 'ln_fund_number_overall']
fe_vars = ['fund_currency', 'vintage']

# Check missing values for each specification
print("\n" + "-"*60)
print("FULL SAMPLE ANALYSIS")
print("-"*60)

# Spec 1: Base model (no FE, no controls)
spec1_vars = base_vars
complete_spec1 = dta_fund.dropna(subset=spec1_vars)
print(f"Specification 1 (base model): {len(complete_spec1):,} observations")

# Spec 2: Base + FE + controls
spec2_vars = base_vars + control_vars + fe_vars
complete_spec2 = dta_fund.dropna(subset=spec2_vars)
print(f"Specification 2 (separate FE): {len(complete_spec2):,} observations")

# For spec 3 (vintage*fund_currency FE), check how many vintage-currency combinations have enough observations
spec3_data = complete_spec2.copy()
fe_combinations = spec3_data.groupby(['vintage', 'fund_currency']).size().reset_index(name='count')
print(f"Specification 3 (vintage×fund_currency FE): {len(complete_spec2):,} observations")
print(f"  - Number of vintage×fund_currency groups: {len(fe_combinations)}")
print(f"  - Groups with 1 observation: {sum(fe_combinations['count'] == 1)}")
print(f"  - Groups with 8+ observations: {sum(fe_combinations['count'] >= 8)}")
print(f"  - Average observations per group: {fe_combinations['count'].mean():.2f}")

print("\n" + "-"*60)
print("CROSS-CURRENCY SAMPLE ANALYSIS")
print("-"*60)

# Same analysis for cross-currency sample
ccs_spec1 = dta_fund_ccs.dropna(subset=spec1_vars)
print(f"Specification 1 (base model): {len(ccs_spec1):,} observations")

ccs_spec2 = dta_fund_ccs.dropna(subset=spec2_vars)
print(f"Specification 2 (separate FE): {len(ccs_spec2):,} observations")

ccs_spec3_data = ccs_spec2.copy()
ccs_fe_combinations = ccs_spec3_data.groupby(['vintage', 'fund_currency']).size().reset_index(name='count')
print(f"Specification 3 (vintage×fund_currency FE): {len(ccs_spec2):,} observations")
print(f"  - Number of vintage×fund_currency groups: {len(ccs_fe_combinations)}")
print(f"  - Groups with 1 observation: {sum(ccs_fe_combinations['count'] == 1)}")
print(f"  - Groups with 8+ observations: {sum(ccs_fe_combinations['count'] >= 8)}")
print(f"  - Average observations per group: {ccs_fe_combinations['count'].mean():.2f}")

# Show the distribution of group sizes
print(f"\nGroup size distribution (cross-currency sample):")
group_dist = ccs_fe_combinations['count'].value_counts().sort_index()
for size, count in group_dist.head(10).items():
    print(f"  {count} groups with {size} observation(s)")
if len(group_dist) > 10:
    print(f"  ... and {len(group_dist) - 10} more group sizes")

# Calculate effective degrees of freedom absorbed by FE
print(f"\nFixed Effects Analysis:")
print(f"  - Degrees of freedom absorbed by vintage×fund_currency FE: {len(ccs_fe_combinations) - 1}")
print(f"  - Remaining effective observations: {len(ccs_spec2) - (len(ccs_fe_combinations) - 1)}")

# Show some examples of the fixed effect groups
print(f"\nExample vintage×fund_currency groups (showing first 10):")
sample_groups = ccs_fe_combinations.head(10)
for _, row in sample_groups.iterrows():
    print(f"  {row['vintage']}-{row['fund_currency']}: {row['count']} observations")

print("\n" + "="*80)
print("IDENTIFICATION CONCERNS ANALYSIS")
print("="*80)

# Check what happens to the main variables of interest within FE groups
print("Analysis of key variable variation within vintage×fund_currency groups:")

# Focus on the main interaction term: forward_fx × firm_currency_ratio
key_vars = ['fund_currency_ratio', 'firm_currency_ratio', 'forward_fx']

total_variation = 0
within_variation = 0
groups_analyzed = 0

for _, group_data in ccs_spec3_data.groupby(['vintage', 'fund_currency']):
    if len(group_data) >= 8:  # Only analyze groups with 8+ observations for reliable identification
        groups_analyzed += 1
        for var in key_vars:
            if var in group_data.columns:
                total_var = group_data[var].var()
                if pd.notna(total_var) and total_var > 0:
                    total_variation += total_var
                    within_variation += total_var  # Within group, this IS the total variation
                    
print(f"\nGroups with 8+ observations for identification: {groups_analyzed}")
print(f"Percentage of groups that can contribute to identification: {groups_analyzed/len(ccs_fe_combinations)*100:.1f}%")

# Calculate effective sample size contributing to identification
effective_sample_8plus = sum(ccs_fe_combinations[ccs_fe_combinations['count'] >= 8]['count'])
print(f"Effective sample size from groups with 8+ observations: {effective_sample_8plus}")
print(f"Percentage of total sample contributing to identification: {effective_sample_8plus/len(ccs_spec2)*100:.1f}%")

# Show detailed breakdown by different thresholds
print(f"\nDetailed identification analysis for 8 independent variables:")
thresholds = [2, 5, 8, 10, 15, 20]
for threshold in thresholds:
    groups_above_threshold = sum(ccs_fe_combinations['count'] >= threshold)
    obs_above_threshold = sum(ccs_fe_combinations[ccs_fe_combinations['count'] >= threshold]['count'])
    print(f"  Groups with {threshold}+ obs: {groups_above_threshold:2d} groups, {obs_above_threshold:3d} observations ({obs_above_threshold/len(ccs_spec2)*100:4.1f}%)")

# Show the largest groups (most identification power)
largest_groups = ccs_fe_combinations.nlargest(10, 'count')
print(f"\nLargest vintage×fund_currency groups (most identification power):")
for _, row in largest_groups.iterrows():
    vintage, currency, count = row['vintage'], row['fund_currency'], row['count']
    group_data = ccs_spec3_data[(ccs_spec3_data['vintage'] == vintage) & 
                                (ccs_spec3_data['fund_currency'] == currency)]
    
    # Check variation in key variables within this group
    firm_ratio_var = group_data['firm_currency_ratio'].var() if 'firm_currency_ratio' in group_data.columns else 0
    forward_fx_var = group_data['forward_fx'].var() if 'forward_fx' in group_data.columns else 0
    
    print(f"  {vintage}-{currency}: {count} obs, firm_ratio_var={firm_ratio_var:.4f}, forward_fx_var={forward_fx_var:.4f}")

print(f"\n" + "="*80)
print("REGRESSION ROBUSTNESS COMPARISON")
print("="*80)

# Compare the actual regression results from the LaTeX table
print("From the regression results (Tables/perf_firm_currency_ratio_ccs.tex):")
print("Cross-currency sample results:")
print("  Specification 2 (separate FE):     782 obs, coefficient = 1.0051**")
print("  Specification 3 (vintage×currency FE): 782 obs, coefficient = 0.6114 (not significant)")
print("\nKey concern: The interaction effect becomes non-significant when using")
print("vintage×fund_currency FE, suggesting potential over-identification.")

print(f"\nStatistical power analysis:")
print(f"  - Total observations: {len(ccs_spec2)}")
print(f"  - Fixed effects (vintage×fund_currency): {len(ccs_fe_combinations) - 1}")
print(f"  - Independent variables: 8")
print(f"  - Total degrees of freedom absorbed: {(len(ccs_fe_combinations) - 1) + 8}")
print(f"  - Effective sample size: {len(ccs_spec2) - (len(ccs_fe_combinations) - 1) - 8}")
print(f"  - Proportion of degrees of freedom absorbed: {((len(ccs_fe_combinations) - 1) + 8)/len(ccs_spec2)*100:.1f}%")
print(f"  - Groups with 8+ obs contributing to identification: {sum(ccs_fe_combinations['count'] >= 8)} out of {len(ccs_fe_combinations)}")
print(f"  - Observations from groups with 8+ obs: {effective_sample_8plus} ({effective_sample_8plus/len(ccs_spec2)*100:.1f}%)")

# USD IDENTIFICATION ANALYSIS
print(f"\n" + "="*80)
print("USD DOMINANCE IN IDENTIFICATION")
print("="*80)

# Analyze USD vs non-USD identification
usd_groups = ccs_fe_combinations[ccs_fe_combinations['fund_currency'] == 'USD']
non_usd_groups = ccs_fe_combinations[ccs_fe_combinations['fund_currency'] != 'USD']

print(f"USD groups analysis:")
print(f"  - Total USD groups: {len(usd_groups)}")
print(f"  - USD groups with 8+ obs: {sum(usd_groups['count'] >= 8)}")
print(f"  - USD observations: {usd_groups['count'].sum()}")
print(f"  - USD observations from groups with 8+ obs: {sum(usd_groups[usd_groups['count'] >= 8]['count'])}")

print(f"\nNon-USD groups analysis:")
print(f"  - Total non-USD groups: {len(non_usd_groups)}")
print(f"  - Non-USD groups with 8+ obs: {sum(non_usd_groups['count'] >= 8)}")
print(f"  - Non-USD observations: {non_usd_groups['count'].sum()}")
print(f"  - Non-USD observations from groups with 8+ obs: {sum(non_usd_groups[non_usd_groups['count'] >= 8]['count'])}")

# Calculate percentages
total_identifiable_obs = sum(ccs_fe_combinations[ccs_fe_combinations['count'] >= 8]['count'])
usd_identifiable_obs = sum(usd_groups[usd_groups['count'] >= 8]['count'])
non_usd_identifiable_obs = sum(non_usd_groups[non_usd_groups['count'] >= 8]['count'])

print(f"\nIdentification dependency on USD:")
print(f"  - USD share of identifiable observations: {usd_identifiable_obs/total_identifiable_obs*100:.1f}%")
print(f"  - Non-USD share of identifiable observations: {non_usd_identifiable_obs/total_identifiable_obs*100:.1f}%")

# Show top USD groups
print(f"\nTop USD vintage×fund_currency groups:")
top_usd_groups = usd_groups.nlargest(10, 'count')
for _, row in top_usd_groups.iterrows():
    print(f"  {row['vintage']}-USD: {row['count']} observations")

# Show top non-USD groups
print(f"\nTop non-USD vintage×fund_currency groups:")
top_non_usd_groups = non_usd_groups.nlargest(5, 'count')
for _, row in top_non_usd_groups.iterrows():
    print(f"  {row['vintage']}-{row['fund_currency']}: {row['count']} observations")

# Currency breakdown for groups with 8+ observations
print(f"\nCurrency breakdown for groups with 8+ observations:")
identifiable_groups = ccs_fe_combinations[ccs_fe_combinations['count'] >= 8]
currency_breakdown = identifiable_groups.groupby('fund_currency').agg({
    'count': ['count', 'sum']
}).round(1)
currency_breakdown.columns = ['num_groups', 'total_obs']
currency_breakdown = currency_breakdown.sort_values('total_obs', ascending=False)
for currency, row in currency_breakdown.iterrows():
    print(f"  {currency}: {int(row['num_groups'])} groups, {int(row['total_obs'])} observations ({row['total_obs']/total_identifiable_obs*100:.1f}%)")

print(f"\nRecommendation:")
print(f"  The vintage×fund_currency FE specification may be over-controlling.")
print(f"  Consider using separate vintage and fund_currency FE (Specification 2) as the main specification,")
print(f"  and vintage×fund_currency FE as a robustness check.")
if usd_identifiable_obs/total_identifiable_obs > 0.7:
    print(f"  WARNING: {usd_identifiable_obs/total_identifiable_obs*100:.1f}% of identification comes from USD funds.")
    print(f"  Results may not generalize to other currencies. Consider currency-specific analysis.")
