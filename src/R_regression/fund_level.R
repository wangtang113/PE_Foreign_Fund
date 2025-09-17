# Modular R regressions for fund-level results and LaTeX export

suppressWarnings(suppressMessages({
  library(readr)
  library(dplyr)
  library(fixest)
}))

source("src/utils/utils_latex.R")
source("src/utils/clustering_utils.R")


# Set working directory to project root to mirror Python
setwd("/Users/wangsmac/Desktop/PE_Foreign_Fund")
dta_fund <- read_csv("Output_data/dta_fund.csv")
fund_cross_currency <- read_csv("Output_data/fund_cross_currency.csv")
# define pair 1 as pair of fund_currency*firm_currency
dta_fund$pair1 <- paste(dta_fund$fund_currency, dta_fund$firm_currency, sep = "_")
dta_fund_ccs <- dta_fund %>% filter(fund_id %in% fund_cross_currency$fund_id)

fit_with_two_way <- function(formula_str, data) {
  fit_with_two_way_clustering(formula_str, data, "vintage", "fund_currency")
}
# Run regressions
# Basic
# Full sample
models_basic_full <- list(
    fit_with_two_way("net_irr_pcent ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency", dta_fund),
    fit_with_two_way("net_irr_pcent ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency + ln_fund_size_usd_mn + ln_fund_number_overall | fund_currency + vintage", dta_fund),
    fit_with_two_way("net_irr_pcent ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency + ln_fund_size_usd_mn + ln_fund_number_overall | firm_currency + vintage", dta_fund),

    fit_with_two_way("multiple ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency", dta_fund),
    fit_with_two_way("multiple ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency + ln_fund_size_usd_mn + ln_fund_number_overall | fund_currency + vintage", dta_fund),
    fit_with_two_way("multiple ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency + ln_fund_size_usd_mn + ln_fund_number_overall | firm_currency + vintage", dta_fund)

)

create_custom_latex_generic(models_basic_full, coef_name = "", coef_label = "", filename = "Tables/perf_firm_basic_full.tex")
# Cross-currency sample
models_basic_ccs <- list(
    fit_with_two_way("net_irr_pcent ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency", dta_fund_ccs),
    fit_with_two_way("net_irr_pcent ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency + ln_fund_size_usd_mn + ln_fund_number_overall | fund_currency + vintage", dta_fund_ccs),
    fit_with_two_way("net_irr_pcent ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency + ln_fund_size_usd_mn + ln_fund_number_overall | firm_currency + vintage", dta_fund_ccs),


    fit_with_two_way("multiple ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency", dta_fund_ccs),
    fit_with_two_way("multiple ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency + ln_fund_size_usd_mn + ln_fund_number_overall | fund_currency + vintage", dta_fund_ccs),
    fit_with_two_way("multiple ~ fund_currency_ratio + firm_currency_ratio + equal_firm_currency + ln_fund_size_usd_mn + ln_fund_number_overall | firm_currency + vintage", dta_fund_ccs)

)
create_custom_latex_generic(models_basic_ccs, coef_name = "", coef_label = "", filename = "Tables/perf_firm_basic_ccs.tex")



