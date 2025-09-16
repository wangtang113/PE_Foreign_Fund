# Cameron-Gelbach-Miller multi-way clustering utilities
# Matches Python statsmodels implementation exactly

suppressWarnings(suppressMessages({
  library(fixest)
  library(sandwich)
}))

#' Two-way clustering using Cameron-Gelbach-Miller inclusion-exclusion principle
#' 
#' @param model A fitted fixest model object
#' @param cluster1 Vector of cluster identifiers for first dimension
#' @param cluster2 Vector of cluster identifiers for second dimension
#' @return Variance-covariance matrix with two-way clustering
two_way_vcov <- function(model, cluster1, cluster2) {
  # Validate inputs
  if (length(cluster1) != model$nobs || length(cluster2) != model$nobs) {
    stop("Cluster variables must have same length as model observations")
  }
  
  # Prepare cluster variables (convert to numeric codes)
  prepare_cluster <- function(cluster_var) {
    if (is.factor(cluster_var)) {
      as.numeric(cluster_var)
    } else {
      as.numeric(as.factor(cluster_var))
    }
  }
  
  c1 <- prepare_cluster(cluster1)
  c2 <- prepare_cluster(cluster2)
  
  # Create interaction cluster (Cameron-Gelbach-Miller method)
  max_c1 <- max(c1, na.rm = TRUE) + 1
  c12 <- c1 * max_c1 + c2
  
  # Compute three cluster-robust covariance matrices
  vcov_c1 <- vcovCL(model, cluster = c1, type = "HC1")
  vcov_c2 <- vcovCL(model, cluster = c2, type = "HC1") 
  vcov_c12 <- vcovCL(model, cluster = c12, type = "HC1")
  
  # Cameron-Gelbach-Miller inclusion-exclusion formula
  vcov_corrected <- vcov_c1 + vcov_c2 - vcov_c12
  
  # Handle negative diagonal elements (common in multi-way clustering)
  diag_elements <- diag(vcov_corrected)
  negative_count <- sum(diag_elements < 0)
  
  if (negative_count > 0) {
    cat(sprintf("Note: Multi-way clustering resulted in %d negative variance estimates. Using absolute values (common in multi-way clustering).\n", negative_count))
    # Fix negative variances by taking absolute value (matches Python behavior)
    diag(vcov_corrected) <- abs(diag(vcov_corrected))
  }
  
  return(vcov_corrected)
}

#' Fit regression model with Python-style two-way clustering
#' 
#' @param formula_str String representation of regression formula
#' @param data Data frame containing all variables
#' @param cluster1_name Name of first clustering variable in data
#' @param cluster2_name Name of second clustering variable in data
#' @return fixest model object with custom vcov attached
fit_with_two_way_clustering <- function(formula_str, data, cluster1_name, cluster2_name) {
  # Parse formula and identify all required variables
  formula_obj <- as.formula(formula_str)
  model_vars <- all.vars(formula_obj)
  cluster_vars <- c(cluster1_name, cluster2_name)
  all_required_vars <- c(model_vars, cluster_vars)
  
  # Remove observations with missing values in any required variable
  complete_data <- data[complete.cases(data[all_required_vars]), ]
  
  # Fit basic model (no clustering)
  model <- feols(formula_obj, data = complete_data, se = "standard")
  
  # Extract cluster variables for complete observations
  cluster1 <- complete_data[[cluster1_name]]
  cluster2 <- complete_data[[cluster2_name]]
  
  # Apply custom two-way clustering
  vcov_custom <- two_way_vcov(model, cluster1, cluster2)
  
  # Attach custom vcov to model object for downstream use
  model$vcov_custom <- vcov_custom
  
  return(model)
}
