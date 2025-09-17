# This script is to create the tables for the research
# install the necessary libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import os
from utils.tables_utils import generate_publication_latex, save_latex_table, generate_summary_latex, fit_model_with_multiple_way_clustering, create_currency_crosstab_latex
from scipy.stats import fit, pearsonr
import subprocess


#--------------------------------- R regressions: panel-level and fund-level tables ------------------
# 1) Panel-level deal activity (R script)
result = subprocess.run([
    'Rscript', 
    'src/R_regression/fund_level.R'
], 
cwd='/Users/wangsmac/Desktop/PE_Foreign_Fund/',
capture_output=True, 
text=True, 
check=True
)


