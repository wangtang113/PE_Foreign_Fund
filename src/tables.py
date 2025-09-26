# This script is to create the tables for the research
# install the necessary libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
from utils.tables_utils import generate_publication_latex, save_latex_table, generate_summary_latex, fit_model_with_multiple_way_clustering, create_currency_crosstab_latex
from scipy.stats import fit, pearsonr
import subprocess




#--------------------------------- Fund Distribution Table ------------------
# Generate fund distribution table by currency and country diversification

def generate_fund_distribution_table():
    """Generate a 2x2 table showing fund distribution by currency and country diversification."""
    
    # Load the fund data
    dta_fund = pd.read_csv("Output_data/dta_fund.csv")
    
    # Create the 2x2 classification
    domestic_currency_domestic_country = ((dta_fund['fund_currency_ratio'] == 1.0) & (dta_fund['fund_country_ratio'] == 1.0)).sum()
    domestic_currency_cross_country = ((dta_fund['fund_currency_ratio'] == 1.0) & (dta_fund['fund_country_ratio'] < 1.0)).sum()
    cross_currency_domestic_country = ((dta_fund['fund_currency_ratio'] < 1.0) & (dta_fund['fund_country_ratio'] == 1.0)).sum()
    cross_currency_cross_country = ((dta_fund['fund_currency_ratio'] < 1.0) & (dta_fund['fund_country_ratio'] < 1.0)).sum()
    
    # Create the distribution table
    distribution_data = {
        'Classification': ['Domestic Country', 'Cross-Country', 'Total'],
        'Domestic Currency': [
            f"{domestic_currency_domestic_country:,}",
            f"{domestic_currency_cross_country:,}",
            f"{domestic_currency_domestic_country + domestic_currency_cross_country:,}"
        ],
        'Cross-Currency': [
            f"{cross_currency_domestic_country:,}",
            f"{cross_currency_cross_country:,}",
            f"{cross_currency_domestic_country + cross_currency_cross_country:,}"
        ],
        'Total': [
            f"{domestic_currency_domestic_country + cross_currency_domestic_country:,}",
            f"{domestic_currency_cross_country + cross_currency_cross_country:,}",
            f"{len(dta_fund):,}"
        ]
    }
    
    distribution_df = pd.DataFrame(distribution_data)
    
    # Create a more detailed LaTeX table with better formatting
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Fund Distribution by Currency and Country Diversification}}
\\label{{tab:fund_distribution}}
\\begin{{tabular}}{{lccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{Currency Diversification}} & \\\\
\\cmidrule(lr){{2-3}}
Country Diversification & Domestic Currency & Cross-Currency & Total \\\\
\\midrule
Domestic Country & {domestic_currency_domestic_country:,} & {cross_currency_domestic_country:,} & {domestic_currency_domestic_country + cross_currency_domestic_country:,} \\\\
Cross-Country & {domestic_currency_cross_country:,}$^*$ & {cross_currency_cross_country:,} & {domestic_currency_cross_country + cross_currency_cross_country:,} \\\\
\\midrule
Total & {domestic_currency_domestic_country + domestic_currency_cross_country:,} & {cross_currency_domestic_country + cross_currency_cross_country:,} & {len(dta_fund):,} \\\\
\\bottomrule
\\multicolumn{{4}}{{l}}{{\\footnotesize $^*$ Key group: Same currency, different countries}} \\\\
\\multicolumn{{4}}{{l}}{{\\footnotesize This explains why cross-country > cross-currency funds}} \\\\
\\end{{tabular}}
\\end{{table}}"""
    
    # Save to file
    with open("Tables/distribution_funds.tex", "w") as f:
        f.write(latex_table)
    
    
    # Print percentage analysis
    total_funds = len(dta_fund)
    print(f"\nPercentage Analysis:")
    print(f"Domestic Currency & Domestic Country: {domestic_currency_domestic_country/total_funds*100:.1f}%")
    print(f"Domestic Currency & Cross-Country: {domestic_currency_cross_country/total_funds*100:.1f}% ⭐")
    print(f"Cross-Currency & Domestic Country: {cross_currency_domestic_country/total_funds*100:.1f}%")
    print(f"Cross-Currency & Cross-Country: {cross_currency_cross_country/total_funds*100:.1f}%")
    
    return distribution_df

# Generate the fund distribution table
fund_distribution = generate_fund_distribution_table()

#--------------------------------- R regressions: panel-level and fund-level tables ------------------
# 1) Panel-level deal activity (R script)
result = subprocess.run([
    'Rscript', 
    'src/R_regression/fund_level.R'
], 
cwd='/Users/wangsmac/Desktop/PE_Foreign_Fund/',
capture_output=True, 
text=True, 
check=True
)


