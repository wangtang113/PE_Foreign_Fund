import pandas as pd
import matplotlib.pyplot as plt

dta_fund = pd.read_csv("Output_data/dta_fund.csv")

# Check the mean of fund_n_deals
print(f"Mean of fund_n_deals: {dta_fund['fund_n_deals'].mean()}")