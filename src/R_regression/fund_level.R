# CLEAN REGRESSION SPECIFICATIONS - FOCUSED RESEARCH DESIGN
# Research Question: How does currency mismatch between PE funds and firms affect performance?

suppressWarnings(suppressMessages({
  library(readr)
  library(dplyr)
  library(fixest)
}))

source("src/utils/utils_latex.R")
source("src/utils/clustering_utils.R")

# Set working directory to project root
setwd("/Users/wangsmac/Desktop/PE_Foreign_Fund")
dta_fund <- read_csv("Output_data/dta_fund.csv")

# ============================================================================
# SAMPLE DEFINITIONS - FOCUSED ON CURRENCY ANALYSIS
# ============================================================================

# 3. Currency pair for clustering (fund_currency * firm_currency) - CREATE FIRST
dta_fund$currency_pair <- paste(dta_fund$fund_currency, dta_fund$firm_currency, sep = "_")

# 1. Cross-currency funds: invest in multiple currencies (main treatment group)
cross_currency_funds <- read_csv("Output_data/cross_currency_funds.csv")
dta_fund_cross_currency <- dta_fund %>% filter(fund_id %in% cross_currency_funds$fund_id)

# Define clustering function
fit_with_currency_clustering <- function(formula_str, data) {
  fit_with_two_way_clustering(formula_str, data, "vintage", "currency_pair")
}

# ============================================================================
# MAIN RESEARCH SPECIFICATIONS 
# ============================================================================

cat("RUNNING CURRENCY MISMATCH ANALYSIS\n")
cat("===================================\n")

# HYPOTHESIS: firm_currency_ratio has NEGATIVE effect when fund_currency != firm_currency
# This captures the cost of currency mismatch between fund and firm

# Specification 1: Cross-Currency Sample (Treatment Group)
cat("1. Cross-Currency Sample Analysis...\n")
models_cross_currency <- list(
    # Basic relationship
    fit_with_currency_clustering("net_irr_pcent ~ firm_currency_ratio + diff_firm_currency + firm_currency_ratio:diff_firm_currency + fund_currency_ratio", 
                                  dta_fund_cross_currency),
    
    # Add controls and fund currency FE
    fit_with_currency_clustering("net_irr_pcent ~ firm_currency_ratio + diff_firm_currency + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency + vintage", 
                                  dta_fund_cross_currency),
    
    # Add firm currency FE (identifies off remaining variation)
    fit_with_currency_clustering("net_irr_pcent ~ firm_currency_ratio + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency + vintage + firm_currency", 
                                  dta_fund_cross_currency),
    # Add fund currency ^ vintage FE
    fit_with_currency_clustering("net_irr_pcent ~ firm_currency_ratio + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency^vintage + firm_currency", 
                                  dta_fund_cross_currency),
    
    # Same for Multiple
    fit_with_currency_clustering("multiple ~ firm_currency_ratio + diff_firm_currency + firm_currency_ratio:diff_firm_currency + fund_currency_ratio", 
                                  dta_fund_cross_currency),
    
    fit_with_currency_clustering("multiple ~ firm_currency_ratio + diff_firm_currency + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency + vintage", 
                                  dta_fund_cross_currency),
    
    fit_with_currency_clustering("multiple ~ firm_currency_ratio + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency + vintage + firm_currency", 
                                  dta_fund_cross_currency),
    
    # Add fund currency ^ vintage FE
    fit_with_currency_clustering("multiple ~ firm_currency_ratio + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency^vintage + firm_currency", 
                                  dta_fund_cross_currency)
)

create_custom_latex_generic(models_cross_currency, 
                           coef_name = "firm_currency_ratio:diff_firm_currency", 
                           coef_label = "Firm currency ratio \\times Different firm currency", 
                           filename = "Tables/perf_firm_cross_currency.tex")

# Specification 2: Full Sample Analysis  
cat("2. Full Sample Analysis...\n")
models_full_sample <- list(
    # Basic relationship with domestic control
    fit_with_currency_clustering("net_irr_pcent ~ firm_currency_ratio + diff_firm_currency + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + is_domestic_investment", 
                                  dta_fund),
    
    # Add controls 
    fit_with_currency_clustering("net_irr_pcent ~ firm_currency_ratio + diff_firm_currency + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + is_domestic_investment + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency + vintage", 
                                  dta_fund),
    
    # Full specification
    fit_with_currency_clustering("net_irr_pcent ~ firm_currency_ratio + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + is_domestic_investment + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency + vintage + firm_currency", 
                                  dta_fund),
    
    # Add fund currency ^ vintage FE
    fit_with_currency_clustering("net_irr_pcent ~ firm_currency_ratio + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + is_domestic_investment + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency^vintage + firm_currency", 
                                  dta_fund),
    
    # Same for Multiple
    fit_with_currency_clustering("multiple ~ firm_currency_ratio + diff_firm_currency + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + is_domestic_investment", 
                                  dta_fund),
    
    fit_with_currency_clustering("multiple ~ firm_currency_ratio + diff_firm_currency + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + is_domestic_investment + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency + vintage", 
                                  dta_fund),
    
    fit_with_currency_clustering("multiple ~ firm_currency_ratio + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + is_domestic_investment + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency + vintage + firm_currency", 
                                  dta_fund),
    
    # Add fund currency ^ vintage FE
    fit_with_currency_clustering("multiple ~ firm_currency_ratio + firm_currency_ratio:diff_firm_currency + fund_currency_ratio + is_domestic_investment + ln_fund_size_usd_mn + ln_fund_number_series | fund_currency^vintage + firm_currency", 
                                  dta_fund)
)

create_custom_latex_generic(models_full_sample, 
                           coef_name = "firm_currency_ratio:diff_firm_currency", 
                           coef_label = "Firm currency ratio \\times Different firm currency", 
                           filename = "Tables/perf_firm_full.tex")

