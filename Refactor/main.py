import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme

# Import your package (assuming the folder is named 'gev_package' or similar)
# If running inside the folder, use local imports, but here is the standard way:
from gevPackage import GEVModel
from gevPackage import GEVLinkage
from gevPackage import ReturnLevel

def generate_synthetic_data(n_obs=20, n_samples=10):
    """
    Generates realistic GEV data.
    Automatically drops spatial covariates if n_samples=1 to avoid multicollinearity.
    """
    np.random.seed(42)
    
    # --- 1. Create Covariates ---
    time_trend = np.linspace(0, 5, n_obs)
    global_temp = time_trend + np.random.normal(0, 0.1, n_obs)
    
    # Logic switch for Single Sample case
    is_single_sample = (n_samples == 1)
    
    if is_single_sample:
        print("(!) S=1 detected: Dropping Elevation covariate to avoid Multicollinearity.")
        # Dummy elevation (not used in model, just for shape consistency if needed)
        X_elev = np.zeros((n_obs, n_samples)) 
    else:
        station_elevation = np.linspace(0, 2, n_samples)
        X_elev = np.tile(station_elevation[None, :], (n_obs, 1))
    
    # Expand Temp: (N, S)
    X_temp = np.tile(global_temp[:, None], (1, n_samples))

    # --- 2. Define Ground Truth Coefficients ---
    
    # Location: Mu = 100 + 10 * Temp (+ 5 * Elev IF S > 1)
    mu_true = 100.0 + 1 * X_temp 
    if not is_single_sample:
        mu_true += 5.0 * X_elev
    
    # Scale: Sigma = Softplus( 2.0 + 0.05 * Temp )
    linker = GEVLinkage()
    lin_scale = 2.0 + 0.5 * X_temp
    sigma_true = linker.np_transform_scale(lin_scale)
    
    # Shape: Xi = 0.15
    xi_true = np.full((n_obs, n_samples), 0.1)
    
    # --- 3. Simulate Endog Data ---
    endog = genextreme.rvs(c=-xi_true, loc=mu_true, scale=sigma_true, size=(n_obs, n_samples))
    
    # --- 4. Pack Exog Dictionary ---
    # Temp is always used
    cov1 = np.tile(global_temp[:, None], (1, n_samples))
    
    if is_single_sample:
        # S=1: Only use Temperature for Location
        exog_loc_final = cov1[:, :, np.newaxis] # Shape (N, S, 1)
    else:
        # S>1: Use [Temp, Elevation]
        cov2 = np.tile(np.linspace(0, 2, n_samples)[None, :], (n_obs, 1))
        exog_loc_final = np.stack([cov1, cov2], axis=2) # Shape (N, S, 2)
    
    final_exog = {
        'location': exog_loc_final,
        'scale': cov1,       # 1 Covariate (Temp)
        'shape': None        # Intercept only
    }
    
    return endog, final_exog, (mu_true, sigma_true, xi_true)

def main():
    print("==========================================")
    print("      JAX GEV REFACTOR - INTEGRATION TEST ")
    print("==========================================")
    
    # 1. Generate Data
    N, S = 100,9
    print(f"Generating data: {N} obs, {S} samples...")
    endog, exog, truths = generate_synthetic_data(N, S)
    
    # 2. Initialize Model
    model = GEVModel(max_iter=5000)
    
    # 3. Fit
    print("\nFitting Model...")
    # Note: We expect:
    # Loc Params: 3 (Intercept, Temp, Elev) -> [100, 10, 5]
    # Scale Params: 2 (Intercept, Temp)     -> [2.0, 0.5]
    # Shape Params: 1 (Intercept)           -> [0.1]
    result = model.fit(endog, exog=exog)
    
    # 4. Summary
    print(result)
    
    # 5. Check Values against Truth
    print("\n--- Ground Truth Check ---")
    est = result.params
    
    # Use the dimensions from the fit result to know what to print
    # dims is tuple (d_loc, d_scale, d_shape)
    d_l, d_s, d_x = result.dims
    
    # --- Location ---
    print(f"Loc Intercept (True 100.0): {est[0]:.4f}")
    print(f"Loc Temp      (True 10.0):  {est[1]:.4f}")
    
    # Only print Elevation if we actually fitted it (d_l > 2 means Intercept + Temp + Elev)
    if d_l > 2:
        print(f"Loc Elev      (True 5.0):   {est[2]:.4f}")
        offset = d_l # index where Scale parameters start
    else:
        print(f"Loc Elev      (SKIPPED):    [Not in Model]")
        offset = d_l
        
    # --- Scale ---
    print(f"Scale Intercept(True 2.0):  {est[offset]:.4f}")
    print(f"Scale Temp    (True 0.5):   {est[offset+1]:.4f}")
    
    # --- Shape ---
    print(f"Shape Intercept(True 0.15): {est[offset+d_s]:.4f}")
    
    z,se = result.return_level(t=[1,10],s=0).compute([10,100])
    print(z.shape)
    
    ax = result.plot.return_levels(
        T=[10,50,100,200,500,1000,5000], 
        t=[0, N-1], 
        s=0
    )
    plt.show()
    
    np.random.seed(999)
    random_x = np.random.uniform(0, 100, S)
    random_y = np.random.uniform(0, 100, S)

    # This works perfectly:
    ax = result.plot.spatial_map(
        T=100, 
        coordinates=(random_x, random_y), # <--- Irregular inputs
        cmap='viridis'
    )
    plt.show()
    
    # --- C. SPATIAL MAP: HEATMAP (Grid Shape) ---
    print("Plotting 3: Spatial Grid Heatmap...")
    
    # Plot the same data but as a continuous field
    result.plot.spatial_map(
        T=100,
        t=0,
        grid_shape=(3, 3), 
        cmap='magma',
        interpolation='nearest', # or 'bicubic' for smooth look
        figsize=(7, 6)
    )
    plt.title("Spatial Heatmap: 100-Year RL (Grid)")
        
if __name__ == "__main__":
    main()