# =====================================
#               🌟 WORK PLAN 🌟
# =====================================

## 1. Data Preparation
----------------------

### 🐍 **Python**

#### 1.1. **Load NETCDF Files**
- **Objective**: Load both factual and counterfactual datasets (precipitation sums over 74 years and simulated stationary data).
- **Input**: NETCDF files.

#### 1.2. **Convert NETCDF to DataFrame**
- **Objective**: Convert the loaded datasets into DataFrames.
- **Output**: DataFrames for both factual and counterfactual datasets.

#### 1.3. **Data Preprocessing**
- **Objective**: Focus on the summer period by removing the 6 winter months (October to March).
- **Output**: Cleaned DataFrames containing only data from April to September.

#### 1.4. **Block Data into Meaningful Periods**
- **Objective**: Block the data into yearly or other relevant periods (e.g., monthly maxima or 10-day blocks).
- **Output**: Blocked datasets for both factual and counterfactual worlds.

---

## 2. Modeling
--------------

### 📊 **R**

#### 2.1. **Fit GEV Models to Both Datasets**
- **Objective**: Fit different Generalized Extreme Value (GEV) models to the data.
  
  - **Model A**: Linear trend in location, constant scale, and shape.
  - **Model B**: Linear trend in location and scale, constant shape.
  - **Model C**: Linear trends in location, scale, and shape.
  - **Model D**: Tradowsky-style model (constant \(\mu\) and \(\sigma\)).
  - **Model E**: GAM-based model for non-stationary parameters.

- **Output**: Fitted GEV models for both factual and counterfactual datasets.

---

## 3. Model Diagnostics & Validation
------------------------------------

### 📈 **R**

#### 3.1. **Verify Tradowsky Assumptions**
- **Objective**: Create diagnostic plots to verify the constant \(\mu\) and \(\sigma\) assumptions from the Tradowsky model.
- **Output**: Stationarity check plots.

#### 3.2. **Compare Model Fits**
- **Objective**: Compare the fits of the GEV models across factual and counterfactual datasets.
- **Focus**: Confidence intervals on the shape parameter (\(\xi\)) for each model.
- **Output**: Comparison plots of model parameters (factual vs. counterfactual).

#### 3.3. **Assess Correlations Between Parameters**
- **Objective**: Investigate correlations between location (\(\mu\)), scale (\(\sigma\)), and shape (\(\xi\)) parameters.
- **Output**: Correlation plots between the parameter estimates.

---

## 4. Post-Processing & Interpretation
--------------------------------------

### 🔍 **R**

#### 4.1. **Extract Return Levels and Confidence Intervals**
- **Objective**: Extract return levels and confidence intervals for all fitted models.
- **Output**: Return level estimates for both datasets.

#### 4.2. **Compare Return Levels**
- **Objective**: Compare return levels between the factual and counterfactual worlds.

#### 4.3. **Summarize Findings**
- **Objective**: Summarize the findings on differences in precipitation extremes between the factual and counterfactual worlds.
- **Output**: Comprehensive comparative analysis report.
