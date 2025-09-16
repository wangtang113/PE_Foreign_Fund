import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os

def run_regression_models(data, dep_vars, specs, fixed_effects=True, cluster_var='system_id'):
    """
    Run multiple regression models with different dependent variables and specifications.
    
    Args:
        data (pd.DataFrame): Input data for regression
        dep_vars (dict): Dictionary of dependent variables {label: variable_name}
        specs (dict): Dictionary of specifications {label: formula}
        fixed_effects (bool): Whether to include fixed effects
        cluster_var (str): Variable to cluster standard errors on
    
    Returns:
        tuple: List of fitted models and their names
    """
    models = []
    model_names = []
    
    # Convert cluster_var to categorical
    data = data.copy()
    data[cluster_var] = data[cluster_var].astype('category')
    
    for dep_label, dep_var in dep_vars.items():
        for spec_label, spec in specs.items():
            formula = f"{dep_var} ~ {spec}"
            if fixed_effects:
                formula += f" + C({cluster_var})"
                
            model = smf.ols(formula, data=data).fit(
                cov_type='cluster',
                cov_kwds={'groups': data[cluster_var]}
            )
            models.append(model)
            model_names.append(f"{dep_label} ~ {spec_label} overvaluation")
            
    return models, model_names



def _get_stars_for_p_value(p_value):
    if p_value is None:
        return ''
    if p_value < 0.01:
        return '***'
    elif p_value < 0.05:
        return '**'
    elif p_value < 0.1:
        return '*'
    return ''

def generate_publication_latex(models, regressor_order, rename_dict, fe_list, float_format="%.4f"):
    """
    Generates a publication-ready LaTeX regression table body.
    """
    out = []
    k = len(models)

    # Header
    header = [""] + [f"\\multicolumn{{1}}{{c}}{{({i})}}" for i in range(1, k + 1)]
    out.append(" & ".join(header) + r" \\")
    out.append(r"\midrule") 

    # Coefficients
    for var in regressor_order:
        if not any(var in m.params for m in models):
            continue

        row_name = rename_dict.get(var, var).replace('"', '')

        betas = []
        ses = []
        for model in models:
            if var in model.params:
                param = model.params[var]
                p_val = model.pvalues[var]
                stars = _get_stars_for_p_value(p_val)
                betas.append((float_format % param) + stars)

                se = model.bse[var]
                ses.append("(" + (float_format % se) + ")")
            else:
                betas.append("")
                ses.append("")

        out.append(f"{row_name} & " + " & ".join(betas) + r" \\")
        out.append(" & " + " & ".join(ses) + r" \\")
    
    # Fixed Effects
    for fe in fe_list:
        def has_fe(model, fe_name):
            if fe_name == "Year FE":
                formula = model.model.formula
                return "Yes" if ("C(year)" in formula or "C(deal_year)" in formula) else "No"
            if fe_name == "Firm FE":
                return "Yes" if "C(firm_id)" in model.model.formula else "No"
            if fe_name == "Intra-fund FE":
                return "Yes" if "C(years_since_last)" in model.model.formula else "No"
            if fe_name == "System FE":
                return "Yes" if "C(system_id)" in model.model.formula else "No"
            if fe_name == "Quarter FE":
                return "Yes" if "C(quarter_number)" in model.model.formula else "No"
            if fe_name == "Vintage FE":
                return "Yes" if "C(vintage)" in model.model.formula else "No"
            if fe_name == "Currency FE":
                return "Yes" if "C(fund_currency_simplified)" in model.model.formula else "No"
            if fe_name == "Fund FE":
                return "Yes" if "C(fund_id)" in model.model.formula else "No"
            if fe_name == "Currency × Vintage FE":
                formula = model.model.formula
                return "Yes" if ("C(fund_currency_simplified)*C(vintage)" in formula or 
                               "C(vintage)*C(fund_currency_simplified)" in formula) else "No"
            if fe_name == "Firm × Vintage FE":
                formula = model.model.formula
                return "Yes" if ("C(firm_id)*C(vintage)" in formula or 
                               "C(vintage)*C(firm_id)" in formula) else "No"
            if fe_name == "Fund × Vintage FE":
                formula = model.model.formula
                return "Yes" if ("C(fund_id)*C(vintage)" in formula or 
                               "C(vintage)*C(fund_id)" in formula) else "No"
            if fe_name == "Firm × Year FE":
                formula = model.model.formula
                return "Yes" if ("C(firm_id)*C(deal_year)" in formula or 
                               "C(deal_year)*C(firm_id)" in formula) else "No"
            return ""

        yesno = [has_fe(m, fe) for m in models]
        if "Yes" in yesno:
            out.append(f"{fe} & " + " & ".join(yesno) + r" \\")

    # Midrule
    out.append("\\midrule")

    # Stats
    n_obs = ["N"] + [f"{int(m.nobs)}" for m in models]
    out.append(" & ".join(n_obs) + r" \\")

    r_squared_vals = [float_format % m.rsquared for m in models]
    r_squared = ["\\(R^{2}\\)"] + r_squared_vals
    out.append(" & ".join(r_squared) + r" \\")

    out.append("")
    return "\n".join(out)


def save_latex_table(latex_content, filename, folder="Tables"):
    """
    Save LaTeX table content to a file.
    
    Args:
        latex_content (str): LaTeX table content
        filename (str): Name of the file (without extension)
        folder (str): Folder to save the file in
    """
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Full path for the file
    filepath = os.path.join(folder, f"{filename}.tex")
    
    # Save the content
    with open(filepath, "w") as f:
        f.write(latex_content)

def generate_summary_latex(df: pd.DataFrame, float_format="%.4f") -> str:
    """
    Return a LaTeX fragment with the descriptive-statistics table
    but **without** the surrounding tabular environment, and with all
    percent signs properly escaped, including in the header.
    """
    summary = df.copy()

    # 1) Escape the quartile column headers
    summary = summary.rename(
        columns={"25%": r"25\%", "50%": r"50\%", "75%": r"75\%"}
    )

    # 2) Escape any % and # in the row labels (index)
    summary.index = summary.index.str.replace("%", r"\%", regex=False)
    summary.index = summary.index.str.replace("#", r"\#", regex=False)

    # 3) Let pandas render LaTeX (no auto-escape → we keep our backslashes)
    latex = summary.to_latex(float_format=float_format, escape=False)

    # 4) Strip the tabular wrapper and the top/bottom rules
    filtered_lines = [
        line
        for line in latex.splitlines()
        if not line.startswith(r"\begin{tabular")
        and not line.startswith(r"\toprule")
        and not line.startswith(r"\bottomrule")
        and not line.startswith(r"\end{tabular")
    ]

    # 5) Escape % and # in the header line (column names)
    if filtered_lines:
        # The first line is the header
        filtered_lines[0] = filtered_lines[0].replace("%", r"\%")
        filtered_lines[0] = filtered_lines[0].replace("#", r"\#")

    # Ensure midrule is present after the header
    if len(filtered_lines) > 1 and r'\midrule' not in filtered_lines[1]:
        filtered_lines.insert(1, r'\midrule')

    return "\n".join(filtered_lines)

# Define function for three-way clustering using Cameron-Gelbach-Miller approach
def get_clustered_vcov(model, clusters):
    """Return a three-way clustered variance-covariance matrix using the C-G-M formula."""
    from statsmodels.stats.sandwich_covariance import cov_cluster
    import numpy as np


    # Convert cluster variables to numeric codes
    def prepare_cluster(cluster_var):
        """Convert cluster variable to numeric codes."""
        if hasattr(cluster_var, 'values'):
            cluster_var = cluster_var.values
        
        # Convert to pandas Series and use factorize for encoding
        cluster_series = pd.Series([str(x) for x in cluster_var])
        codes, _ = pd.factorize(cluster_series)
        return codes
    
    # Prepare individual clusters
    cluster1_encoded = prepare_cluster(clusters[0])
    cluster2_encoded = prepare_cluster(clusters[1]) 
    cluster3_encoded = prepare_cluster(clusters[2])

    # Calculate each individual cluster
    cov_1 = cov_cluster(model, cluster1_encoded)
    cov_2 = cov_cluster(model, cluster2_encoded)
    cov_3 = cov_cluster(model, cluster3_encoded)

    # Two-way clusters - create interaction groups
    def create_interaction_cluster(c1, c2):
        """Create interaction cluster from two cluster variables."""
        # Create unique combinations
        max_c1 = max(c1) + 1
        return c1 * max_c1 + c2
    
    cluster_12 = create_interaction_cluster(cluster1_encoded, cluster2_encoded)
    cluster_13 = create_interaction_cluster(cluster1_encoded, cluster3_encoded)
    cluster_23 = create_interaction_cluster(cluster2_encoded, cluster3_encoded)
    
    cov_12 = cov_cluster(model, cluster_12)
    cov_13 = cov_cluster(model, cluster_13)
    cov_23 = cov_cluster(model, cluster_23)

    # Three-way cluster
    cluster_123 = create_interaction_cluster(cluster_12, cluster3_encoded)
    cov_123 = cov_cluster(model, cluster_123)

    # Apply inclusion-exclusion principle
    vcov = (cov_1 + cov_2 + cov_3
            - cov_12 - cov_13 - cov_23
            + cov_123)
    
    return vcov

def fit_model_with_multiple_way_clustering(formula, data, cluster_vars):
    """
    Fit OLS model and apply three-way clustering.
    
    Args:
        formula (str): Regression formula
        data (pd.DataFrame): Dataset for regression
        cluster_vars (list): List of 3 cluster variables (can be column names or pandas Series)
                            If only 1 or 2 variables provided, will fall back to appropriate clustering
    
    Returns:
        fitted model with corrected standard errors
    """
    import statsmodels.formula.api as smf
    import numpy as np
    
    # Fit basic OLS model
    model = smf.ols(formula, data=data).fit()
    
    # Extract cluster variables from data if they are column names
    cluster_series = []
    for i, var in enumerate(cluster_vars):
        if isinstance(var, str):
            if var in data.columns:
                cluster_series.append(data[var])
            else:
                raise ValueError(f"Cluster variable '{var}' not found in data columns")
        else:
            cluster_series.append(var)
    
    # Determine clustering approach based on number of cluster variables
    if len(cluster_series) == 3:
        # Three-way clustering
        vcov_corrected = get_clustered_vcov(model, cluster_series)
    elif len(cluster_series) == 2:
        # Two-way clustering using inclusion-exclusion
        from statsmodels.stats.sandwich_covariance import cov_cluster
        
        # Prepare clusters
        def prepare_cluster(cluster_var):
            if hasattr(cluster_var, 'values'):
                cluster_var = cluster_var.values
            cluster_series = pd.Series([str(x) for x in cluster_var])
            codes, _ = pd.factorize(cluster_series)
            return codes
        
        c1 = prepare_cluster(cluster_series[0])
        c2 = prepare_cluster(cluster_series[1])
        
        # Two-way clustering: V = V1 + V2 - V12
        cov_1 = cov_cluster(model, c1)
        cov_2 = cov_cluster(model, c2)
        
        # Create interaction cluster
        max_c1 = max(c1) + 1
        c12 = c1 * max_c1 + c2
        cov_12 = cov_cluster(model, c12)
        
        vcov_corrected = cov_1 + cov_2 - cov_12
        
    elif len(cluster_series) == 1:
        # Standard one-way clustering
        from statsmodels.stats.sandwich_covariance import cov_cluster
        
        def prepare_cluster(cluster_var):
            if hasattr(cluster_var, 'values'):
                cluster_var = cluster_var.values
            cluster_series = pd.Series([str(x) for x in cluster_var])
            codes, _ = pd.factorize(cluster_series)
            return codes
        
        c1 = prepare_cluster(cluster_series[0])
        vcov_corrected = cov_cluster(model, c1)
    else:
        raise ValueError("Must provide 1, 2, or 3 cluster variables")
    
    # Update model with clustered standard errors
    model.cov_params_default = model.cov_params()
    model.cov_params = lambda: vcov_corrected
    
    # Update standard errors - ensure proper indexing and handle negative variances
    diag_elements = np.diag(vcov_corrected)
    
    # Check for negative diagonal elements (can happen with multi-way clustering)
    negative_count = np.sum(diag_elements < 0)
    if negative_count > 0:
        # For negative diagonal elements, use absolute value
        diag_elements = np.abs(diag_elements)
        # Only print warning once per model, not for every negative element
        if not hasattr(fit_model_with_multiple_way_clustering, '_negative_warning_shown'):
            print(f"Note: Multi-way clustering resulted in {negative_count} negative variance estimates. Using absolute values (common in multi-way clustering).")
            fit_model_with_multiple_way_clustering._negative_warning_shown = True
    
    se_corrected = np.sqrt(diag_elements)
    model.bse = pd.Series(se_corrected, index=model.params.index)
    
    # Update t-statistics and p-values - ensure same indexing
    model.tvalues = pd.Series(model.params.values / model.bse.values, index=model.params.index)
    from scipy.stats import t
    model.pvalues = pd.Series(2 * (1 - t.cdf(np.abs(model.tvalues.values), model.df_resid)), index=model.params.index)
    
    # Ensure all series have consistent indexing
    assert len(model.params) == len(model.bse) == len(model.tvalues) == len(model.pvalues)
    assert all(model.params.index == model.bse.index)
    assert all(model.params.index == model.tvalues.index) 
    assert all(model.params.index == model.pvalues.index)
    
    return model


def create_currency_crosstab_latex(crosstab_df, caption="Currency distribution of deals"):
    """
    Convert a currency crosstab to LaTeX table format matching the style of perf_forward_fx.tex
    """
    latex_content = []
    
    # Header row with deal currencies (columns) - clearly indicate these are deal currencies
    deal_currency_headers = " & ".join([f"\\multicolumn{{1}}{{c}}{{{col}}}" for col in crosstab_df.columns])
    latex_content.append(f" & {deal_currency_headers} \\\\")
    latex_content.append("\\midrule")
    
    # Add a subtitle row to clarify dimensions
    latex_content.append(f"\\multicolumn{{1}}{{l}}{{\\textit{{Fund Currency}}}} & \\multicolumn{{{len(crosstab_df.columns)}}}{{c}}{{\\textit{{Deal Currency}}}} \\\\")
    latex_content.append("\\midrule")
    
    # Data rows - fund currencies (rows)
    for idx in crosstab_df.index[:-1]:  # Exclude Total row for now
        row_data = [f"{idx}"] + [f"{val:,}" for val in crosstab_df.loc[idx]]
        latex_content.append(" & ".join(row_data) + " \\\\")
    
    latex_content.append("\\midrule")
    
    # Total row
    total_row = [f"\\textbf{{{crosstab_df.index[-1]}}}"] + [f"\\textbf{{{val:,}}}" for val in crosstab_df.loc['Total']]
    latex_content.append(" & ".join(total_row) + " \\\\")
    
    return "\n".join(latex_content)
