import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.winsorize_utils import winsorize
from utils.plots_utils import create_scatter_regression
from utils.plots_utils import save_figure

dta_fund_deals = pd.read_csv("Output_data/dta_fund_deals.csv")
# Now we need to filter the dta_fund_deals table to only include the funds that have a foreign currency
# the foreign investment percentage is a fund level variable
dta_fund_deals_ccs = dta_fund_deals[dta_fund_deals["foreign_investment_pct"] > 0].rename(columns={"DEAL CURRENCY": "deal_currency", "FUND CURRENCY": "fund_currency"})

fund_deals_ccs = dta_fund_deals_ccs.copy().drop_duplicates(subset=["fund_id"])
# winsorize the net_irr_pcent and multiple
winsorize_cols = ["net_irr_pcent", "multiple"]
fund_deals_ccs = winsorize(fund_deals_ccs, winsorize_cols)

# draw a plot showing the distribution of firm_currency_ratio and fund_currency_ratio in the fund_deals_ccs table
fig1 = plt.figure(figsize=(10, 5))
plt.hist(fund_deals_ccs["firm_currency_ratio"], bins=100, alpha=0.5)
plt.hist(fund_deals_ccs["fund_currency_ratio"], bins=100, alpha=0.5)
plt.legend(["Firm currency ratio", "Fund currency ratio"])
plt.xlabel("Currency ratio")
plt.title("Firm and fund currency ratios in the cross currency sample")
save_figure(fig1, "firm_and_fund_currency_ratios.png", "Figures")

# draw a plot showing the relationship between net_irr_pcent/multiple and firm_currency_ratio/fund_currency_ratio
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
create_scatter_regression(
    x=fund_deals_ccs["firm_currency_ratio"], 
    y=fund_deals_ccs["net_irr_pcent"],
    xlabel="Firm currency ratio",
    ylabel="Net IRR (%)",
    ax=ax[0], 
    title="Net IRR (%) and firm currency ratio",
    binned=True,
    num_bins=100
    )
create_scatter_regression(
    x=fund_deals_ccs["fund_currency_ratio"], 
    y=fund_deals_ccs["net_irr_pcent"],
    xlabel="Fund currency ratio",
    ylabel="Net IRR (%)",
    ax=ax[1], 
    title="Net IRR (%) and fund currency ratio",
    binned=True,
    num_bins=100
    )
save_figure(fig, "net_irr_firm_fund_currency_ratios.png", "Figures")
