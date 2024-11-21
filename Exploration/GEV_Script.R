# -----------------------------------------------------------
# LOAD NECESSARY LIBRARIES
# -----------------------------------------------------------
library(ismev)
library(terra)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lubridate) 

# ================================
#           WORK PLAN
# ================================

# 1. Data Preparation
# -------------------
## Python ##
# 1.1 Load NETCDF files (factual and counterfactual datasets) into R
#     - Input: NETCDF files for precipitation sum (74 years) and simulated stationary data.
# 1.2 Convert NETCDF files to DataFrame.
#     - Output: DataFrames for each dataset.
# 1.3 Preprocess data: Remove 6 winter months to focus on the summer period.
#     - Output: Cleaned DataFrames (only summer months).
# 1.4 Block data into yearly (or other meaningful) block sizes, e.g., monthly maxima.
#     - Output: Blocked datasets for factual and counterfactual worlds.

# 2. Modeling
# -----------
##R## 
# 2.1 Fit different GEV models to both datasets:
#     - Model A: Linear trend in location, constant scale and shape.
#     - Model B: Linear trend in location and scale, constant shape.
#     - Model C: Linear trends in location, scale, and shape.
#     - Model D: Tradowsky-style model (constant mu/sigma assumption).
#     - Model E: GAM-based model for non-stationary parameters.
#     - Output: Fitted GEV models for both datasets.

# 3. Model Diagnostics & Validation
# ---------------------------------
##R##
# 3.1 Verify Tradowsky assumptions (constant mu/sigma):
#     - Create plots to check stationarity and ratios for the assumption.
# 3.2 Compare model fits across factual and counterfactual datasets:
#     - Check confidence intervals on shape parameter (xi) for each model.
#     - Output: Comparison plots for model parameters (factual vs. counterfactual).
# 3.3 Assess correlations between parameter estimates:
#     - Explore correlations between location (mu), scale (sigma), and shape (xi).
#     - Output: Correlation plots between parameter estimates.

# 4. Post-Processing & Interpretation
# -----------------------------------
##R##
# 4.1 Extract return levels and confidence intervals for all models.
#     - Output: Return level estimates for both datasets.
# 4.2 Compare return levels between factual and counterfactual worlds.
# 4.3 Summarize findings on differences in precipitation extremes.
#     - Output: Comparative analysis report.

# ================================
# Function: 1] load_nc_to_rast
# Purpose: Load a NETCDF file and return it as a SpatRaster object
# ================================
load_nc_to_rast <- function(nc_file_path) {
  # Load the NETCDF file as a SpatRaster object using the terra package
  nc_data <- rast(nc_file_path)
  
  # Print a message indicating successful loading
  message("NETCDF file loaded successfully.")
  
  # Return the SpatRaster object
  return(nc_data)
}

# ================================
# Function: 2] process_rast
# Purpose: Crop a SpatRaster object to a specific extent, convert to DataFrame, and reshape data
# ================================
process_nc_raster <- function(nc_data, xmin, xmax, ymin, ymax) {
  # 2.1 Define the extent for cropping
  extent_art <- ext(xmin, xmax, ymin, ymax)
  
  # 2.2 Crop the raster to the specified extent
  nc_subset <- crop(nc_data, extent_art)
  message("Raster cropped to the specified extent.")
  
  # 2.3 Convert the raster to a data frame
  flood_df <- as.data.frame(nc_subset, cells = FALSE, na.rm = NA)
  
  # Rename columns to represent time steps
  colnames(flood_df) <- 1:ncol(flood_df)
  
  # 2.4 Reshape the data to long format and filter for the last x time steps
  time_series_long_filtered <- flood_df %>%
    pivot_longer(cols = everything(), names_to = "day", values_to = "value") %>%
    mutate(
      day = as.numeric(day),
      date = as.Date(day - 1, origin = "1950-01-01"),  # Convert to date format
      month = month(date)  # Extract the month
    ) %>%
    filter(month >= 4 & month <= 9)
  
  # Return the reshaped and filtered data frame
  return(time_series_long_filtered)
}


# -----------------------------------------------------------
# LOAD AND EXPLORE NETCDF DATA
# -----------------------------------------------------------
nc_file <- "C:/Users/bobel/OneDrive - Université de Namur/Data/E_OBS/rr_ens_mean_0.25deg_reg_v30.0e.nc"
nc_data = load_nc_to_rast(nc_file)

has_na <- global(nc_data, fun = function(x) any(is.na(x)))
# Output result
print(has_na)
# -----------------------------------------------------------
# VISUALIZE THE LAST TIME LAYER (FINAL DAY) 
# -----------------------------------------------------------

# Extract and visualize the last day layer
last_day_layer <- nc_data[[nlyr(nc_data)]]
plot(last_day_layer, main = "Grid for the Last Day")

# -----------------------------------------------------------
# CROP DATA TO FOCUS ON A SPECIFIC REGION (BELGIUM)
# -----------------------------------------------------------

xmin <- 3.0   # Minimum longitude
xmax <- 6.0   # Maximum longitude
ymin <- 48    # Minimum latitude
ymax <- 52    # Maximum latitude

# Process the data (crop, convert, and reshape)
time_series_long_filtered <- process_nc_raster(nc_data, xmin, xmax, ymin, ymax)


# -----------------------------------------------------------
# PLOT TIME SERIES FOR THE LAST X TIME STEPS
# -----------------------------------------------------------
time_series_long_filtered %>% filter(day > (max(day) - 1000)) %>% ggplot(aes(x = day, y = value)) +
  geom_point(alpha = 0.6, color = "blue", size = 0.2) +
  labs(title = "Time Series Values for the Last 100 Time Steps",
       x = "time",
       y = "value") +
  theme_minimal(base_size = 15) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text = element_text(color = "black"),
        panel.grid.major = element_line(color = "grey80"))


# -----------------------------------------------------------
# FIT GEV MODEL USING ISMEV FOR ALL SPATIAL POINTS TOGETHER
# -----------------------------------------------------------
## First We use synthetic non stationary data with constant xi and scale but different shape varying through time.

generate_synthetic_data <- function(n_days = 365 * 10, start_value = 100, growth_rate = 1.05, noise_sd = 5) {
  # Create time variable (number of days)
  time <- 1:n_days
  
  # Generate synthetic data with a linear growth over time
  synthetic_data <- start_value * growth_rate^(time / 365) + rnorm(n_days, mean = 0, sd = noise_sd)
  
  # Create a data frame
  data_frame <- data.frame(day = time, value = synthetic_data)
  
  # Return the data frame
  return(data_frame)
}
synthetic_data <- generate_synthetic_data()

block_maxima <- synthetic_data %>%
  mutate(block = ceiling(day / 10)) %>%  # Create 10-day blocks
  group_by(block) %>%
  summarise(max_value = max(value), block_time = mean(day))

# Fit a basic stationary GEV model
gev_fit_basic <- gev.fit(xdat = block_maxima$max_value, show = TRUE)

# Print the results
print(gev_fit_basic$mle)  # MLEs for location, scale, and shape

log_link <- function(x) {
  exp(x)
}

gev_fit_custom <- gev.fit(
  xdat = block_maxima$max_value, 
  ydat = data.frame(block_time = block_maxima$block_time),  # Covariate for location
  mul = 1,   # Use block_time as a covariate for the location parameter
  sigl = NULL,   # Keep the scale parameter stationary
  shl = NULL,    # Keep the shape parameter stationary
  mulink = identity,  # No transformation for location
  siglink = identity,  # Apply log link to the scale
  shlink = identity,  # Apply logistic link to the shape
  show = TRUE,   # Show fitting process
)

gev.diag(gev_fit_custom)
xi_mle <- gev_fit_custom$mle[3]  # MLE for shape parameter (xi)
xi_se <- gev_fit_custom$se[3]    # SE for shape parameter (xi)

# 95% confidence interval using normal approximation
z_95 <- 1.96  # Z-value for 95% confidence interval
# Calculate lower and upper bounds of the confidence interval
ci_lower <- xi_mle - z_95 * xi_se
ci_upper <- xi_mle + z_95 * xi_se

cat("95% Confidence Interval for xi:", ci_lower, "to", ci_upper, "\n")


## ---- Now we test with our dataset.

# Function to test different block sizes and parameter forms
test_gev_models <- function(time_series_long_filtered) {
  
  # Define different block sizes to test
  block_sizes <- c(0.3,0.5)  # 1 year, 5 years
  
  # Define parameter functions to test for location, scale, and shape
  param_forms <- list(
    list(location.fun = ~ 1, scale.fun = ~ 1, shape.fun = ~ 1, name = "Constant Parameters"),
    list(location.fun = ~ block_time, scale.fun = ~ 1, shape.fun = ~ 1, name = "Non-stationary Location"),
    list(location.fun = ~ block_time, scale.fun = ~ block_time, shape.fun = ~ 1, name = "Non-stationary Location & Scale"),
    list(location.fun = ~ block_time, scale.fun = ~ block_time, shape.fun = ~ block_time, name = "Non-stationary Location, Scale & Shape")
  )
  
  # Store models and AIC values
  model_list <- list()
  aic_list <- data.frame(model = character(), block_size = integer(), aic = numeric(), stringsAsFactors = FALSE)
  
  # Loop through block sizes and parameter forms
  for (block_size in block_sizes) {
    
    # Create block maxima
    block_maxima <- time_series_long_filtered %>%
      mutate(block = ceiling(day / (365 * block_size))) %>%
      group_by(block) %>%
      summarise(max_value = max(value), block_time = mean(day))
    
    # Loop through different parameter functions
    for (param in param_forms) {
      
      # Fit the GEV model
      fit <- fevd(
        x = block_maxima$max_value,
        data = block_maxima,
        location.fun = param$location.fun,
        scale.fun = param$scale.fun,
        shape.fun = param$shape.fun,
        type = "GEV"
      )
      
      # Get the AIC value and store the model
      aic_value <- summary(fit)$AIC
      model_name <- paste0(param$name, " - Block Size: ", block_size, " years")
      model_list[[model_name]] <- fit
      aic_list <- rbind(aic_list, data.frame(model = model_name, block_size = block_size, aic = aic_value))
    }
  }
  
  # Sort by AIC and return the top 3 models
  aic_list <- aic_list[order(aic_list$aic), ]
  top_3_models <- aic_list[1:3, ]
  
  # Pretty print the top 3 models
  cat("\n========================================\n")
  cat(" Top 3 GEV Models Based on AIC\n")
  cat("========================================\n")
  for (i in 1:3) {
    cat(paste0("Top ", i, ": ", top_3_models$model[i], "\n"))
    cat(paste0("   AIC: ", round(top_3_models$aic[i], 2), "\n"))
    cat("--------------------------------------------------\n")
  }
  
  return(list(top_3_models = top_3_models, models = model_list))
}

# Example usage with your time series data
result <- test_gev_models(time_series_long_filtered)

plot(result$models[[6]])
