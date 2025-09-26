import pandas as pd
import matplotlib.pyplot as plt
from utils.plots_utils import create_scatter_regression
from utils.plots_utils import save_figure

dta_fund = pd.read_csv("Output_data/dta_fund.csv")
cross_currency_funds = pd.read_csv("Output_data/fund_cross_currency.csv")
dta_fund_ccs = dta_fund[dta_fund["fund_id"].isin(cross_currency_funds["fund_id"])]
dta_fund_domestic = pd.read_csv("Output_data/dta_fund_domestic.csv")
#-------------------- draw a plot showing the distribution of firm_currency_ratio and fund_currency_ratio in the dta_fund_ccs table
fig1 = plt.figure(figsize=(12, 9))
plt.hist(dta_fund_ccs["firm_currency_ratio"], bins=100, alpha=0.5)
plt.hist(dta_fund_ccs["fund_currency_ratio"], bins=100, alpha=0.5)
plt.legend(["Firm currency ratio", "Fund currency ratio"])
plt.xlabel("Currency ratio")
plt.title("Firm and fund currency ratios in the cross currency sample")
save_figure(fig1, "firm_and_fund_currency_ratios.png", "Figures")

#-------------------- draw a plot showing the relationship between net_irr_pcent/multiple and firm_currency_ratio/fund_currency_ratio
fig, ax = plt.subplots(1, 2, figsize=(16, 12))
create_scatter_regression(
    x=dta_fund_ccs["firm_currency_ratio"], 
    y=dta_fund_ccs["net_irr_pcent"],
    xlabel="Firm currency ratio",
    ylabel="Net IRR (%)",
    ax=ax[0], 
    title="Net IRR (%) and firm currency ratio",
    binned=True,
    num_bins=100
    )
create_scatter_regression(
    x=dta_fund_ccs["fund_currency_ratio"], 
    y=dta_fund_ccs["net_irr_pcent"],
    xlabel="Fund currency ratio",
    ylabel="Net IRR (%)",
    ax=ax[1], 
    title="Net IRR (%) and fund currency ratio",
    binned=True,
    num_bins=100
    )
save_figure(fig, "net_irr_firm_fund_currency_ratios.png", "Figures")

#-------------------- draw  a plot showing the firm_currency_ratio and forward_fx_firm
fig, ax = plt.subplots(1, 2, figsize=(16, 12))
create_scatter_regression(
    x=dta_fund_ccs["forward_fx_firm"], 
    y=dta_fund_ccs["firm_currency_ratio"],
    ax=ax[0],
    xlabel="Forward FX firm(%)",
    ylabel="Firm currency ratio",
    title="Firm currency ratio and forward FX firm (%) in the cross currency sample",
    binned=True,
    num_bins=20
    )
create_scatter_regression(
    x=dta_fund["forward_fx_firm"], 
    y=dta_fund["firm_currency_ratio"],
    ax=ax[1],
    xlabel="Forward FX firm(%)",
    ylabel="Firm currency ratio",
    title="Firm currency ratio and forward FX firm (%) in the full sample",
    binned=True,
    num_bins=20
    )
save_figure(fig, "firm_currency_ratio_forward_fx_firm.png", "Figures")

#-------------------- draw two plots showing the distance_fx_forward and fund_n_deals and distance_fx_forward_firm and fund_n_deals_firm, using dta_fund_ccs
fig, ax = plt.subplots(1, 2, figsize=(16, 12))
create_scatter_regression(
    x=dta_fund_ccs["distance_forward_fx"], 
    y=dta_fund_ccs["fund_n_deals"],
    ax=ax[0],
    xlabel="Distance forward FX",
    ylabel="Number of deals",
    title="Distance forward FX and number of deals in the cross currency sample",
    binned=True,
    num_bins=20
    )
create_scatter_regression(
    x=dta_fund_ccs["distance_forward_fx_firm"], 
    y=dta_fund_ccs["fund_n_deals_firm"],
    ax=ax[1],
    xlabel="Distance forward FX firm",
    ylabel="Number of deals in firm currency",
    title="Distance forward FX firm and number of deals in firm currency in the cross currency sample",
    binned=True,
    num_bins=20
    )
save_figure(fig, "distance_forward_fx_fund_n_deals.png", "Figures")

#-------------------- draw two plots showing the forward_fx with net_irr_pcent and multiple using dta_fund_ccs 
fig, ax = plt.subplots(1, 2, figsize=(16, 12))
create_scatter_regression(
    x=dta_fund_ccs["forward_fx"], 
    y=dta_fund_ccs["net_irr_pcent"],
    ax=ax[0],
    xlabel="Forward FX",
    ylabel="Net IRR (%)",
    title="Forward FX and net IRR (%) in the cross currency sample",
    binned=True,
    num_bins=100,
    split_regression_at=0
    )
create_scatter_regression(
    x=dta_fund_ccs["forward_fx"], 
    y=dta_fund_ccs["multiple"],
    ax=ax[1],
    xlabel="Forward FX",
    ylabel="Multiple",
    title="Forward FX and multiple in the cross currency sample",
    binned=True,
    num_bins=100,
    split_regression_at=0

    )
save_figure(fig, "forward_fx_irr_multiple.png", "Figures")

#-------------------- draw the scatter plot showing the multiple with net_irr_pcent using dta_fund and dta_fund_ccs
fig, ax = plt.subplots(1, 2, figsize=(16, 12))
create_scatter_regression(
    x=dta_fund["multiple"], 
    y=dta_fund["net_irr_pcent"],
    ax=ax[0],
    xlabel="Multiple",
    ylabel="Net IRR (%)",
    title="Multiple and net IRR (%) in the full sample"
    )
create_scatter_regression(
    x=dta_fund_ccs["multiple"], 
    y=dta_fund_ccs["net_irr_pcent"],
    ax=ax[1],
    xlabel="Multiple",
    ylabel="Net IRR (%)",
    title="Multiple and net IRR (%) in the cross currency sample"
    )
save_figure(fig, "multiple_irr.png", "Figures")

#-------------------- draw the histogram showing the distribution of net irr pcent and multiple in the dta_fund_domestic table and dta_fund_ccs table
fig, ax = plt.subplots(2, 1, figsize=(16, 12))
ax[0].hist(dta_fund_domestic["net_irr_pcent"], bins=100, alpha=0.5, label="Domestic funds")
ax[0].hist(dta_fund_ccs["net_irr_pcent"], bins=100, alpha=0.5, label="Cross-currency funds")
ax[0].axvline(dta_fund_domestic["net_irr_pcent"].mean(), color='blue', linestyle='--', alpha=0.8, label=f"Domestic mean: {dta_fund_domestic['net_irr_pcent'].mean():.2f}")
ax[0].axvline(dta_fund_ccs["net_irr_pcent"].mean(), color='orange', linestyle='--', alpha=0.8, label=f"Cross-currency mean: {dta_fund_ccs['net_irr_pcent'].mean():.2f}")
ax[0].legend()
ax[0].set_xlabel("Net IRR (%)")
ax[0].set_ylabel("Frequency")
ax[0].set_title("Net IRR (%) Distribution: Domestic vs Cross-Currency Samples")

ax[1].hist(dta_fund_domestic["multiple"], bins=100, alpha=0.5, label="Domestic funds")
ax[1].hist(dta_fund_ccs["multiple"], bins=100, alpha=0.5, label="Cross-currency funds")
ax[1].axvline(dta_fund_domestic["multiple"].mean(), color='blue', linestyle='--', alpha=0.8, label=f"Domestic mean: {dta_fund_domestic['multiple'].mean():.2f}")
ax[1].axvline(dta_fund_ccs["multiple"].mean(), color='orange', linestyle='--', alpha=0.8, label=f"Cross-currency mean: {dta_fund_ccs['multiple'].mean():.2f}")
ax[1].legend()
ax[1].set_xlabel("Multiple")
ax[1].set_ylabel("Frequency")
ax[1].set_title("Multiple Distribution: Domestic vs Cross-Currency Samples")

plt.tight_layout()
save_figure(fig, "net_irr_multiple_domestic_ccs.png", "Figures")

#-------------------- draw the histogram of vintage in the dta_fund and dta_fund_ccs
fig, ax = plt.subplots(2, 1, figsize=(16, 12))
ax[0].hist(dta_fund["vintage"], bins=30, alpha=0.7, label="Full sample", color='blue')
ax[0].legend()
ax[0].set_xlabel("Vintage")
ax[0].set_ylabel("Frequency")
ax[0].set_title("Vintage Distribution: Full Sample")

ax[1].hist(dta_fund_ccs["vintage"], bins=30, alpha=0.7, label="Cross-currency sample", color='orange')
ax[1].legend()
ax[1].set_xlabel("Vintage")
ax[1].set_ylabel("Frequency")
ax[1].set_title("Vintage Distribution: Cross-Currency Sample")

plt.tight_layout()
save_figure(fig, "vintage_distribution_full_ccs.png", "Figures")

#-------------------- draw the histogram of firm_currency_ratio and fund_currency_ratio in dta_fund_special
dta_fund_special = pd.read_csv("Output_data/dta_fund_special.csv")
fig, ax = plt.subplots(1, 2, figsize=(16, 12))
ax[0].hist(dta_fund_special['firm_currency_ratio'], bins=100, alpha=0.5)
ax[0].set_xlabel('Firm currency ratio')
ax[0].set_ylabel('Frequency')
ax[0].set_title('Histogram of firm currency ratio in dta_fund_special')
ax[1].hist(dta_fund_special['fund_currency_ratio'], bins=100, alpha=0.5)
ax[1].set_xlabel('Fund currency ratio')
ax[1].set_ylabel('Frequency')
ax[1].set_title('Histogram of fund currency ratio in dta_fund_special')
plt.savefig('Figures/currency_ratio_dta_fund_special.png')