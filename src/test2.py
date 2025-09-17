import pandas as pd

dta_fund = pd.read_csv("Output_data/dta_fund.csv")

table = pd.crosstab(dta_fund["vintage"], dta_fund["fund_currency"])
print(table)
