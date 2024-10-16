# ================================
#           WORK PLAN
# ================================

# 1. Data Preparation
-------------------

### Python
1.1 **Load NETCDF files** (factual and counterfactual datasets) into R  
   - **Input**: NETCDF files for precipitation sum (74 years) and simulated stationary data.
   
1.2 **Convert NETCDF files to DataFrame**  
   - **Output**: DataFrames for each dataset.

1.3 **Preprocess data**: Remove 6 winter months to focus on the summer period  
   - **Output**: Cleaned DataFrames (only summer months).

1.4 **Block data into yearly (or other meaningful) block sizes**, e.g., monthly maxima  
   - **Output**: Blocked datasets for factual and counterfactual worlds.

# 2. Modeling
-----------

### R
2.1 **Fit different GEV models** to both datasets:  
   - **Model A**: Linear trend in location, constant scale and shape.  
   - **Model B**: Linear trend in location and scale, constant shape.  
   - **Model C**: Linear trends in location, scale, and shape.  
   - **Model D**: Tradowsky-style model (constant mu/sigma assumption).  
   - **Model E**: GAM-based model for non-stationary parameters.  
   - **Output**: Fitted GEV models for both datasets.

# 3. Model Diagnostics & Validation
---------------------------------

### R
3.1 **Verify Tradowsky assumptions** (constant mu/sigma):  
   - Create plots to check stationarity and ratios for the assumption.

3.2 **Compare model fits** across factual and counterfactual datasets:  
   - Check confidence intervals on shape parameter (xi) for each model.  
   - **Output**: Comparison plots for model parameters (factual vs. counterfactual).

3.3 **Assess correlations between parameter estimates**:  
   - Explore correlations between location (mu), scale (sigma), and shape (xi).  
   - **Output**: Correlation plots between parameter estimates.

# 4. Post-Processing & Interpretation
-----------------------------------

### R
4.1 **Extract return levels and confidence intervals** for all models  
   - **Output**: Return level estimates for both datasets.

4.2 **Compare return levels** between factual and counterfactual worlds.

4.3 **Summarize findings** on differences in precipitation extremes  
   - **Output**: Comparative analysis report.
