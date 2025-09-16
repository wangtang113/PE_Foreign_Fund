# This script is to define the currency type of the country in every year

import pandas as pd
import numpy as np
from utils.dataclean_utils import legal_tender
from utils.update_iso_codes import apply_eur_nan_rule
from utils.country_name_standardizer import standardize_country_name

currencies = (
    pd.read_excel(
        "Input_data/Currencies.xlsx",
        sheet_name="Sheet1",
        dtype=str,
    )
    .rename(columns=str.upper)
) 

# rename the column "COUNTRY NAME" to "COUNTRY"
currencies = currencies.rename(columns={"COUNTRY NAME": "COUNTRY"})


# Standardize country names using centralized standardizer
currencies["COUNTRY"] = currencies["COUNTRY"].apply(
    lambda x: standardize_country_name(x, source="currency")
)


# Load currency change history from CSV
currency_history = pd.read_csv("Input_data/currency_history.csv")

# Build unified table from the loaded data
change_rows = []
for _, row in currency_history.iterrows():
    change_rows.append({
        "COUNTRY": row["country"],
        "EFFECTIVE": pd.Period(row["effective_date"], "M"),
        "OLD": row["old_currency"],
        "NEW": row["new_currency"]
    })

changes = (
    pd.DataFrame(change_rows)
    .sort_values(["COUNTRY", "EFFECTIVE"])
    .reset_index(drop=True)
)


START = "1980-01"
END   = "2025-07"
all_countries = (
    set(currencies["COUNTRY"])
    | set(changes["COUNTRY"])
)

panel = (
    pd.MultiIndex.from_product(
        [sorted(all_countries), pd.period_range(START, END, freq="M")],
        names=["COUNTRY", "PERIOD"],
    )
    .to_frame(index=False)
)

panel["ISO_CODE"] = [
    legal_tender(c, p, changes, currencies) for c, p in zip(panel["COUNTRY"], panel["PERIOD"])
]

# Apply EUR/absent-history rule and save the panel to the Output_data folder
panel = apply_eur_nan_rule(panel, currencies, currency_history)
panel.to_csv("Output_data/country_currency_month.csv", index=False)

# Read the FX Forward curves_NEW.xlsx
fx_forward_curves = pd.read_excel("Input_data/FX Forward curves_NEW.xlsx")

# Rename the DATE column to FX Quarter
fx_forward_curves = fx_forward_curves.rename(columns={"DATE": "Date"})

# Extract unique currency codes from column names
currency_codes = set()
for col in fx_forward_curves.columns:
    if col != "Date" and ("_SP_" in col or "_5Y_" in col):
        currency = col.split("_")[0]
        currency_codes.add(currency)

# Calculate averages for each currency using pd.concat for better performance
averaged_columns = []
for currency in sorted(currency_codes):
    sp_bid_col = f"{currency}_SP_BID"
    sp_ask_col = f"{currency}_SP_ASK"
    y5_bid_col = f"{currency}_5Y_BID"
    y5_ask_col = f"{currency}_5Y_ASK"
    
    # Calculate spot rate average (SP)
    if sp_bid_col in fx_forward_curves.columns and sp_ask_col in fx_forward_curves.columns:
        sp_avg = pd.Series(
            (fx_forward_curves[sp_bid_col] + fx_forward_curves[sp_ask_col]) / 2,
            name=f"{currency}_SP"
        )
        averaged_columns.append(sp_avg)
    
    # Calculate 5-year forward rate average (5Y)
    if y5_bid_col in fx_forward_curves.columns and y5_ask_col in fx_forward_curves.columns:
        y5_avg = pd.Series(
            (fx_forward_curves[y5_bid_col] + fx_forward_curves[y5_ask_col]) / 2,
            name=f"{currency}_5Y"
        )
        averaged_columns.append(y5_avg)

# Combine all averaged columns with the date column
fx_forward_processed = pd.concat(
    [fx_forward_curves[["Date"]]] + averaged_columns,
    axis=1
)

# Calculate forward_fx and realized_fx for each currency against USD (optimized for performance)
print("Calculating forward_fx and realized_fx for each currency against USD...")

# Add quarterly periods for easier manipulation
fx_forward_processed = fx_forward_processed.copy()  # Avoid fragmentation
fx_forward_processed['Quarter'] = pd.to_datetime(fx_forward_processed['Date']).dt.to_period('Q')

# Prepare forward_fx and realized_fx calculations using vectorized operations
forward_fx_data = {}
realized_fx_data = {}
forward_fx_columns = []
realized_fx_columns = []

for currency in sorted(currency_codes):
    sp_col = f"{currency}_SP"
    fwd_col = f"{currency}_5Y"
    
    if sp_col in fx_forward_processed.columns and fwd_col in fx_forward_processed.columns:
        # forward_fx calculation: ((forward/spot)^(1/5) - 1)*100
        forward_fx_col_name = f"{currency}_forward_fx"
        forward_fx_data[forward_fx_col_name] = np.where(
            fx_forward_processed[sp_col].notna() & 
            fx_forward_processed[fwd_col].notna() & 
            (fx_forward_processed[sp_col] != 0),
            (((fx_forward_processed[fwd_col] / fx_forward_processed[sp_col]) ** (1/5) - 1)*100),
            np.nan
        )
        forward_fx_columns.append(forward_fx_col_name)
        
        # realized_fx calculation: ((future_spot/current_spot)^(1/5) - 1)
        realized_fx_col_name = f"{currency}_realized_fx"
        realized_fx_values = np.full(len(fx_forward_processed), np.nan)
        
        # Vectorized realized_fx calculation using future spot rates (5 years ahead)
        current_spots = fx_forward_processed[sp_col].values
        for i in range(len(fx_forward_processed) - 20):  # -20 because we need 20 quarters ahead
            current_spot = current_spots[i]
            future_spot = current_spots[i + 20]
            
            if pd.notna(current_spot) and pd.notna(future_spot) and current_spot != 0:
                realized_fx_values[i] = (((future_spot / current_spot) ** (1/5) - 1)*100)
        
        realized_fx_data[realized_fx_col_name] = realized_fx_values
        realized_fx_columns.append(realized_fx_col_name)

# Add all forward_fx and realized_fx columns at once to avoid fragmentation
forward_fx_df = pd.DataFrame(forward_fx_data, index=fx_forward_processed.index)
realized_fx_df = pd.DataFrame(realized_fx_data, index=fx_forward_processed.index)
fx_forward_processed = pd.concat([fx_forward_processed, forward_fx_df, realized_fx_df], axis=1)

print(f"✓ Calculated forward_fx for {len(forward_fx_columns)} currencies")
print(f"✓ Calculated realized_fx for {len(realized_fx_columns)} currencies")
print(f"Sample forward_fx columns: {forward_fx_columns[:5]}")
print(f"Sample realized_fx columns: {realized_fx_columns[:5]}")

# Show sample calculations
if len(forward_fx_columns) > 0:
    sample_currency = forward_fx_columns[0].replace('_forward_fx', '')
    print(f"\\nSample calculations for {sample_currency}:")
    sample_data = fx_forward_processed[['Date', f'{sample_currency}_SP', f'{sample_currency}_5Y', 
                                       f'{sample_currency}_forward_fx', f'{sample_currency}_realized_fx']].head(3)
    print(sample_data)

# Save the processed data to csv
fx_forward_processed.to_csv("Output_data/FX_forward.csv", index=False)

print(f"Processed FX forward data saved to Output_data/FX_forward.csv")
print(f"Shape: {fx_forward_processed.shape}")
print(f"Currencies processed: {len(currency_codes)}")
print(f"Sample columns: {fx_forward_processed.columns[:10].tolist()}")



