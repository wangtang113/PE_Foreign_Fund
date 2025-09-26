# Utility functions for LaTeX table generation from fixest models

suppressWarnings(suppressMessages({
  library(fixest)
}))

# Create a LaTeX body matching Python table style: all coefficients + separate FE rows
create_custom_latex_generic <- function(models, coef_name = "", coef_label = "", filename) {
  output <- c()

  # Header row (column numbers) - dynamically handle any number of models
  n_models <- length(models)
  header <- paste(c("", paste0("\\multicolumn{1}{c}{(", 1:n_models, ")}")), collapse = " & ")
  output <- c(output, paste(header, "\\\\"))
  output <- c(output, "\\midrule")

  # Helper function to get coefficient and SE with stars
  get_coef_and_se <- function(var_name, var_label) {
    coefs <- sapply(models, function(m) {
      nm <- names(coef(m))
      if (!is.null(nm) && var_name %in% nm) {
        coef_val <- as.numeric(coef(m)[[var_name]])
        # Use custom vcov if available, otherwise default vcov
        se_val <- tryCatch({
          if (!is.null(m$vcov_custom)) {
            sqrt(diag(m$vcov_custom))[[var_name]]
          } else {
            sqrt(diag(vcov(m)))[[var_name]]
          }
        }, error = function(e) NA_real_)
        # Significance stars
        p_val <- tryCatch({
          t_val <- coef_val / se_val
          2 * pt(abs(t_val), df = m$nobs - length(coef(m)), lower.tail = FALSE)
        }, error = function(e) NA_real_)
        stars <- ""
        if (!is.na(p_val)) {
          if (p_val < 0.01) stars <- "***" else if (p_val < 0.05) stars <- "**" else if (p_val < 0.1) stars <- "*"
        }
        paste0(sprintf("%.4f", coef_val), stars)
      } else {
        ""
      }
    })

    ses <- sapply(models, function(m) {
      nm <- names(coef(m))
      if (!is.null(nm) && var_name %in% nm) {
        # Use custom vcov if available, otherwise default vcov
        se_val <- tryCatch({
          if (!is.null(m$vcov_custom)) {
            sqrt(diag(m$vcov_custom))[[var_name]]
          } else {
            sqrt(diag(vcov(m)))[[var_name]]
          }
        }, error = function(e) NA_real_)
        paste0("(", sprintf("%.4f", se_val), ")")
      } else {
        ""
      }
    })
    
    # Add coefficient and SE rows
    output <<- c(output, paste(paste(c(var_label, coefs), collapse = " & "), "\\\\"))
    output <<- c(output, paste(paste(c("", ses), collapse = " & "), "\\\\"))
  }

  # Main FX variable (only show if specified)
  if (coef_name != "" && coef_label != "") {
    get_coef_and_se(coef_name, coef_label)
  }

  # Control variables (show coefficients if present)
  control_vars <- list(
    # Currency-based variables
    "firm_currency_ratio" = "Firm currency ratio",
    "diff_firm_currency" = "Different firm currency",
    "fund_currency_ratio" = "Fund currency ratio",
    "is_foreign_investment" = "Cross-currency investment",
    "is_domestic_investment" = "Domestic investment",
    # Country-based variables (NEW)
    "firm_country_ratio" = "Firm country ratio",
    "diff_firm_country" = "Different firm country",
    "fund_country_ratio" = "Fund country ratio",
    "is_cross_country_investment" = "Cross-country investment",
    
    # Other controls
    "forward_fx" = "Forward FX",
    "forward_fx_firm" = "Forward FX firm",
    "ln_fund_size_usd_mn" = "ln(Fund size)",
    "ln_fund_n_deals" = "ln(Number of fund deals)",
    "ln_fund_number_overall" = "ln(Fund number overall)",
    "ln_fund_number_series" = "ln(Fund number series)",
    "ln_fund_n_currencies" = "ln(Number of currencies)",
    "distance_forward_fx" = "Distance forward FX",
    "distance_forward_fx_firm" = "Distance forward FX firm"
  )
  
  for (var_name in names(control_vars)) {
    # Check if any model has this control variable and it's not the main coefficient already shown
    has_var <- any(sapply(models, function(m) var_name %in% names(coef(m))))
    if (has_var && var_name != coef_name) {
      get_coef_and_se(var_name, control_vars[[var_name]])
    }
  }

  # Fixed Effects Detection - Individual FE rows like Python
  # Note: fixest uses "^" for interactions, not ":" or other separators
  # If a variable appears in an interaction, don't mark the individual FE as "Yes"
  fe_checks <- list(
    "Year FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "deal_year" %in% fe_vars && !any(grepl("deal_year\\^", fe_vars))
    },
    "Fund age FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "fund_age" %in% fe_vars && !any(grepl("fund_age\\^", fe_vars))
    },
    "Fund currency FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "fund_currency" %in% fe_vars && !any(grepl("fund_currency\\^", fe_vars))
    },
    "Firm FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "firm_id" %in% fe_vars && !any(grepl("firm_id\\^", fe_vars))
    },
    "Fund FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "fund_id" %in% fe_vars && !any(grepl("fund_id\\^", fe_vars))
    },
    "Deal currency FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "deal_currency" %in% fe_vars && !any(grepl("deal_currency\\^", fe_vars))
    },
    "Deal country FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "deal_country" %in% fe_vars && !any(grepl("deal_country\\^", fe_vars))
    },
    "Buyout fund size FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "buyout_fund_size" %in% fe_vars && !any(grepl("buyout_fund_size\\^", fe_vars))
    },
    "Firm currency FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "firm_currency" %in% fe_vars && !any(grepl("firm_currency\\^", fe_vars))
    },
    "Firm country FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "firmcountry" %in% fe_vars && !any(grepl("firmcountry\\^", fe_vars))
    },
    "Year $\\times$ Deal country FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("deal_country\\^deal_year", fe_vars)) || any(grepl("deal_year\\^deal_country", fe_vars))
    },
    "Year $\\times$ Fund currency FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("deal_year\\^fund_currency", fe_vars)) || any(grepl("fund_currency\\^deal_year", fe_vars))
    },
    "Year $\\times$ Deal currency FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("deal_currency\\^deal_year", fe_vars)) || any(grepl("deal_year\\^deal_currency", fe_vars))
    },
    "Fund currency $\\times$ Deal currency FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("fund_currency\\^deal_currency", fe_vars)) || any(grepl("deal_currency\\^fund_currency", fe_vars))
    },
    "Fund currency $\\times$ Deal country FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("fund_currency\\^deal_country", fe_vars)) || any(grepl("deal_country\\^fund_currency", fe_vars))
    },
    "Firm $\\times$ Deal currency FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("firm_id\\^deal_currency", fe_vars)) || any(grepl("deal_currency\\^firm_id", fe_vars))
    },
    "Deal currency experience FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "prev_currency_experience" %in% fe_vars && !any(grepl("prev_currency_experience\\^", fe_vars))
    },
    "Deal country experience FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "prev_country_experience" %in% fe_vars && !any(grepl("prev_country_experience\\^", fe_vars))
    },
    "Deal currency history FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "prev_currency_investment" %in% fe_vars && !any(grepl("prev_currency_investment\\^", fe_vars))
    },
    "Time to latest deal FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "time_to_latest_deal" %in% fe_vars && !any(grepl("time_to_latest_deal\\^", fe_vars))
    },
    "Previous invested years FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "previous_invested_years" %in% fe_vars && !any(grepl("previous_invested_years\\^", fe_vars))
    },
    "Vintage FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "vintage" %in% fe_vars && !any(grepl("vintage\\^", fe_vars))
    },
    "Firm $\\times$ Year FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("firm_id\\^deal_year", fe_vars)) || any(grepl("deal_year\\^firm_id", fe_vars))
    },    
    "Fund $\\times$ Deal Currency FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("fund_id\\^deal_currency", fe_vars)) || any(grepl("deal_currency\\^fund_id", fe_vars))
    },
    "Fund currency $\\times$ Vintage FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("fund_currency\\^vintage", fe_vars)) || any(grepl("vintage\\^fund_currency", fe_vars))
    },
    "Fund currency $\\times$ Firm currency FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("fund_currency\\^firm_currency", fe_vars)) || any(grepl("firm_currency\\^fund_currency", fe_vars))
    },
    "Firm currency $\\times$ Vintage FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      any(grepl("firm_currency\\^vintage", fe_vars)) || any(grepl("vintage\\^firm_currency", fe_vars))
    },
    "Fund number overall FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "fund_number_overall" %in% fe_vars && !any(grepl("fund_number_overall\\^", fe_vars))
    },
    "Fund number series FE" = function(m) {
      fe_vars <- tryCatch(m$fixef_vars, error = function(e) character(0))
      # Only exact match, not in interactions
      "fund_number_series" %in% fe_vars && !any(grepl("fund_number_series\\^", fe_vars))
    }
  )
  
  for (fe_name in names(fe_checks)) {
    fe_check_func <- fe_checks[[fe_name]]
    has_fe <- sapply(models, fe_check_func)
    fe_row <- ifelse(has_fe, "Yes", "No")
    
    # Only add FE row if at least one model has it
    if (any(has_fe)) {
      output <- c(output, paste(paste(c(fe_name, fe_row), collapse = " & "), "\\\\"))
    }
  }

  # Separator
  output <- c(output, "\\midrule")

  # N and R2 rows
  n_obs <- sapply(models, function(m) m$nobs)
  output <- c(output, paste(paste(c("N", format(n_obs, big.mark = ",")), collapse = " & "), "\\\\"))

  r2_vals <- sapply(models, function(m) {
    # Use fitstat when available; fallback to summary
    tryCatch({
      sprintf("%.4f", fitstat(m, "r2"))
    }, error = function(e) {
      tryCatch({
        sprintf("%.4f", summary(m)$r.squared)
      }, error = function(e2) {
        "0.0000"
      })
    })
  })
  output <- c(output, paste(paste(c("\\(R^{2}\\)", r2_vals), collapse = " & "), "\\\\"))

  # Write LaTeX body
  dir.create(dirname(filename), recursive = TRUE, showWarnings = FALSE)
  writeLines(output, filename)
  
  cat(sprintf("Generated %s with %d models\n", filename, n_models))
}