# 0. Imports
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial # For jax.jit

# Matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define the Spatial Weighted Neural Network ---
class SpatialWeightNN(nn.Module):
    num_outputs: int
    hidden_sizes: list[int]

    @nn.compact
    def __call__(self, spatial_distances: jnp.ndarray) -> jnp.ndarray:
        x = spatial_distances
        for i, size in enumerate(self.hidden_sizes):
            x = nn.Dense(features=size, name=f'hidden_{i}')(x)
            x = nn.relu(x)
        weights_w = nn.Dense(features=self.num_outputs, name='output_weights')(x)
        return weights_w

# --- 2. Define the Full Model with SEPARATE WEIGHTS for Mu and Sigma ---
#    AND MODIFIED TO OUTPUT WEIGHTS
class SpatioTemporalWeightedModelWithSeparateWeights(nn.Module):
    num_independent_vars: int
    spatial_nn_hidden_sizes: list[int]

    def setup(self):
        self.num_coeffs = self.num_independent_vars + 1
        self.spatial_nn = SpatialWeightNN(
            num_outputs=2 * self.num_coeffs, # Double outputs for mu and sigma weights
            hidden_sizes=self.spatial_nn_hidden_sizes
        )
        self.beta_coeffs = self.param('beta_coeffs', nn.initializers.normal(stddev=0.01), (self.num_coeffs,))
        self.gamma_coeffs = self.param('gamma_coeffs', nn.initializers.normal(stddev=0.01), (self.num_coeffs,))

    def __call__(self, spatial_distances: jnp.ndarray, independent_vars: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        combined_weights = self.spatial_nn(spatial_distances)
        weights_w_mu = combined_weights[:, :self.num_coeffs]
        weights_w_sigma = combined_weights[:, self.num_coeffs:]

        X_prime = jnp.concatenate([
            jnp.ones((independent_vars.shape[0], 1), dtype=independent_vars.dtype),
            independent_vars
        ], axis=-1)

        mu = jnp.sum(weights_w_mu * self.beta_coeffs * X_prime, axis=-1)
        log_sigma_sq = jnp.sum(weights_w_sigma * self.gamma_coeffs * X_prime, axis=-1)
        sigma_sq = jnp.exp(log_sigma_sq) + 1e-6
        
        return mu, sigma_sq, weights_w_mu, weights_w_sigma # Now returns weights too

# --- 3. Define the Loss Function (Negative Log-Likelihood) ---
def nll_loss(mu: jnp.ndarray, sigma_sq: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
    y_true = jnp.squeeze(y_true)
    loss_terms = 0.5 * (jnp.log(2 * jnp.pi) + jnp.log(sigma_sq) + ((y_true - mu)**2 / sigma_sq))
    return jnp.mean(loss_terms)

# --- 4. Training Step and Utilities ---
@partial(jax.jit, static_argnames=("model_apply_fn", "optimizer_update_fn"))
def train_step(params: optax.Params, opt_state: optax.OptState,
               model_apply_fn, optimizer_update_fn,
               spatial_distances: jnp.ndarray, independent_vars: jnp.ndarray, y_true: jnp.ndarray
               ) -> tuple[optax.Params, optax.OptState, jnp.ndarray]:
    def loss_fn(current_params):
        # model.apply now returns mu, sigma_sq, weights_mu, weights_sigma
        # We only need mu and sigma_sq for the loss.
        mu, sigma_sq, _, _ = model_apply_fn({'params': current_params}, spatial_distances, independent_vars)
        loss = nll_loss(mu, sigma_sq, y_true)
        return loss

    loss_value, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer_update_fn(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_value

# --- 5. Plotting Function for General Evaluation ---
def generate_evaluation_plots(y_true, mu_pred, sigma_sq_pred, data_name="Data"):
    y_true_np = np.array(y_true).flatten()
    mu_pred_np = np.array(mu_pred).flatten()
    sigma_pred_np = np.sqrt(np.array(sigma_sq_pred).flatten())
    residuals_np = y_true_np - mu_pred_np
    abs_error_np = np.abs(residuals_np)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except IOError:
        print("Seaborn style not found, using default Matplotlib style.")

    # Plot 1: True Y vs. Predicted Mu
    plt.figure(figsize=(8, 8))
    min_val = min(np.min(y_true_np), np.min(mu_pred_np))
    max_val = max(np.max(y_true_np), np.max(mu_pred_np))
    plt.scatter(y_true_np, mu_pred_np, alpha=0.6, edgecolors='k', s=50, label='Predictions')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal y=x line')
    plt.xlabel("True Values (y_true)", fontsize=12)
    plt.ylabel("Predicted Mean (mu_pred)", fontsize=12)
    plt.title(f"{data_name}: True Values vs. Predicted Mean", fontsize=14)
    plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7); plt.axis('square'); plt.tight_layout(); plt.show()

    # Plot 2: Predicted Mu vs. Residuals
    plt.figure(figsize=(8, 6))
    plt.scatter(mu_pred_np, residuals_np, alpha=0.6, edgecolors='k', s=50, label='Residuals')
    plt.axhline(0, color='r', linestyle='--', lw=2, label='Zero Residual line')
    plt.xlabel("Predicted Mean (mu_pred)", fontsize=12); plt.ylabel("Residuals (y_true - mu_pred)", fontsize=12)
    plt.title(f"{data_name}: Predicted Mean vs. Residuals", fontsize=14)
    plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout(); plt.show()

    # Plot 3: Absolute Error vs. Predicted Sigma
    plt.figure(figsize=(8, 6))
    plt.scatter(sigma_pred_np, abs_error_np, alpha=0.6, edgecolors='k', s=50, label='Error vs. Sigma')
    min_diag, max_diag = 0, max(np.max(sigma_pred_np), np.max(abs_error_np))
    plt.plot([min_diag, max_diag], [min_diag, max_diag], 'r--', lw=2, label='Ideal (Error = Sigma)')
    plt.xlabel("Predicted Standard Deviation (sigma_pred)", fontsize=12); plt.ylabel("Absolute Error (|y_true - mu_pred|)", fontsize=12)
    plt.title(f"{data_name}: Absolute Error vs. Predicted Sigma", fontsize=14)
    plt.legend(fontsize=10); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout(); plt.show()

    # Plot 4: True Y (sorted) vs. Predicted Mu with Uncertainty Bands
    sorted_indices = np.argsort(y_true_np)
    y_true_sorted, mu_pred_sorted, sigma_pred_sorted = y_true_np[sorted_indices], mu_pred_np[sorted_indices], sigma_pred_np[sorted_indices]
    plt.figure(figsize=(12, 7))
    plt.scatter(np.arange(len(y_true_sorted)), y_true_sorted, color='black', s=15, alpha=0.7, label='True Values (Sorted)')
    plt.plot(np.arange(len(y_true_sorted)), mu_pred_sorted, color='blue', lw=2, label='Predicted Mean (mu_pred)')
    plt.fill_between(np.arange(len(y_true_sorted)), mu_pred_sorted - sigma_pred_sorted, mu_pred_sorted + sigma_pred_sorted, color='skyblue', alpha=0.4, label='mu +/- 1 sigma')
    plt.fill_between(np.arange(len(y_true_sorted)), mu_pred_sorted - 1.96 * sigma_pred_sorted, mu_pred_sorted + 1.96 * sigma_pred_sorted, color='lightcyan', alpha=0.3, label='mu +/- 1.96 sigma (95% CI)')
    plt.xlabel("Sample Index (Sorted by True y)", fontsize=12); plt.ylabel("Values", fontsize=12)
    plt.title(f"{data_name}: Predictions with Uncertainty Bands", fontsize=14)
    plt.legend(fontsize=10, loc='upper left'); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout(); plt.show()

# --- 6. NEW Plotting Function for Visualizing Spatial Weights ---
def generate_weight_visualizations(weights_mu, weights_sigma, spatial_distances_data, num_independent_vars, data_name="Data"):
    weights_mu_np = np.array(weights_mu)
    weights_sigma_np = np.array(weights_sigma)
    spatial_distances_np = np.array(spatial_distances_data)
    
    num_coeffs = num_independent_vars + 1

    # Ensure we don't try to plot more weights than available coefficients
    plot_indices = [0] # Weight for the intercept term (beta_0, gamma_0)
    if num_coeffs > 1:
        plot_indices.append(1) # Weight for the first independent variable (beta_1, gamma_1)
    
    # Determine number of rows for subplot based on how many indices we are plotting
    n_plot_indices = len(plot_indices)
    if n_plot_indices == 0:
        print("No weight indices to plot.")
        return

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except IOError:
        print("Seaborn style not found, using default Matplotlib style.")

    # --- Plot A: Histograms of selected spatial weights ---
    fig_hist, axes_hist = plt.subplots(n_plot_indices, 2, figsize=(12, 5 * n_plot_indices), squeeze=False)
    fig_hist.suptitle(f'{data_name}: Distribution of Selected Spatial Weights', fontsize=16, y=1.02)

    for i, coeff_idx in enumerate(plot_indices):
        coeff_label = f"w_({coeff_idx})" # e.g. w_0 for intercept, w_1 for first var

        # Histogram for mu_weights
        axes_hist[i, 0].hist(weights_mu_np[:, coeff_idx], bins=30, alpha=0.7, color='dodgerblue', edgecolor='k')
        axes_hist[i, 0].set_title(f'Spatial Weights for Mean: {coeff_label} (for beta_{coeff_idx})', fontsize=12)
        axes_hist[i, 0].set_xlabel('Weight Value', fontsize=10)
        axes_hist[i, 0].set_ylabel('Frequency', fontsize=10)
        axes_hist[i, 0].grid(True, linestyle=':', alpha=0.7)

        # Histogram for sigma_weights
        axes_hist[i, 1].hist(weights_sigma_np[:, coeff_idx], bins=30, alpha=0.7, color='tomato', edgecolor='k')
        axes_hist[i, 1].set_title(f'Spatial Weights for Sigma: {coeff_label} (for gamma_{coeff_idx})', fontsize=12)
        axes_hist[i, 1].set_xlabel('Weight Value', fontsize=10)
        axes_hist[i, 1].set_ylabel('Frequency', fontsize=10)
        axes_hist[i, 1].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle
    plt.show()

    # --- Plot B: Scatter plot of a primary spatial feature vs. selected weights ---
    if spatial_distances_np.shape[1] > 0: # Check if there are any spatial features
        primary_spatial_feature = spatial_distances_np[:, 0] # Using the first spatial distance feature
        
        fig_scatter, axes_scatter = plt.subplots(n_plot_indices, 2, figsize=(14, 5 * n_plot_indices), squeeze=False)
        fig_scatter.suptitle(f'{data_name}: Primary Spatial Feature vs. Selected Weights', fontsize=16, y=1.02)

        for i, coeff_idx in enumerate(plot_indices):
            coeff_label = f"w_({coeff_idx})"

            # Scatter for mu_weights
            axes_scatter[i, 0].scatter(primary_spatial_feature, weights_mu_np[:, coeff_idx], alpha=0.5, color='dodgerblue', s=30, edgecolors='k')
            axes_scatter[i, 0].set_title(f'Mean Weight {coeff_label} vs. Spatial Feature 0', fontsize=12)
            axes_scatter[i, 0].set_xlabel('Value of Primary Spatial Feature (e.g., Distance 0)', fontsize=10)
            axes_scatter[i, 0].set_ylabel(f'Weight Value {coeff_label} (for beta_{coeff_idx})', fontsize=10)
            axes_scatter[i, 0].grid(True, linestyle=':', alpha=0.7)

            # Scatter for sigma_weights
            axes_scatter[i, 1].scatter(primary_spatial_feature, weights_sigma_np[:, coeff_idx], alpha=0.5, color='tomato', s=30, edgecolors='k')
            axes_scatter[i, 1].set_title(f'Sigma Weight {coeff_label} vs. Spatial Feature 0', fontsize=12)
            axes_scatter[i, 1].set_xlabel('Value of Primary Spatial Feature (e.g., Distance 0)', fontsize=10)
            axes_scatter[i, 1].set_ylabel(f'Weight Value {coeff_label} (for gamma_{coeff_idx})', fontsize=10)
            axes_scatter[i, 1].grid(True, linestyle=':', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
    else:
        print("No spatial distance features to plot against weights.")


# --- 7. Example Usage / Mock Data and Training ---
def run_example():
    key = jax.random.PRNGKey(43) # Changed seed for variety
    key_data, key_init, key_train_loop = jax.random.split(key, 3)

    num_samples = 1000# Slightly more samples
    num_spatial_features = 1000
    num_ind_vars = 2 # p=2, so num_coeffs = 3 (w0, w1, w2)
    batch_size = 32

    spatial_distances_data = jax.random.normal(key_data, (num_samples, num_spatial_features))
    independent_vars_data = jax.random.normal(key_data, (num_samples, num_ind_vars))

    key_y_gen, _ = jax.random.split(key_data)
    true_beta_coeffs_gen = jax.random.normal(key_y_gen, (num_ind_vars + 1,)) * 0.8
    true_gamma_coeffs_gen = jax.random.normal(key_y_gen, (num_ind_vars + 1,)) * 0.3
    true_w_gen_matrix = jax.random.normal(key_y_gen, (num_spatial_features, num_ind_vars + 1)) * 0.2
    simulated_true_weights_w = jnp.tanh(spatial_distances_data @ true_w_gen_matrix)
    X_prime_data_gen = jnp.concatenate([jnp.ones((num_samples, 1)), independent_vars_data], axis=-1)
    true_mu_gen = jnp.sum(simulated_true_weights_w * true_beta_coeffs_gen * X_prime_data_gen, axis=-1)
    true_log_sigma_sq_gen = jnp.sum(simulated_true_weights_w * true_gamma_coeffs_gen * X_prime_data_gen, axis=-1) - 1.0
    true_sigma_sq_gen = jnp.exp(true_log_sigma_sq_gen) + 1e-6
    y_true_data = true_mu_gen + jax.random.normal(key_y_gen, (num_samples,)) * jnp.sqrt(true_sigma_sq_gen)

    spatial_nn_hidden_sizes_config = [64, 32]
    learning_rate = 1e-3
    num_epochs = 150 # Adjusted epochs

    # USE THE UPDATED MODEL
    model = SpatioTemporalWeightedModelWithSeparateWeights(
        num_independent_vars=num_ind_vars,
        spatial_nn_hidden_sizes=spatial_nn_hidden_sizes_config
    )
    mock_spatial_distances = jnp.ones((1, num_spatial_features))
    mock_independent_vars = jnp.ones((1, num_ind_vars))
    params = model.init(key_init, mock_spatial_distances, mock_independent_vars)['params']

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    jitted_train_step = partial(train_step, model_apply_fn=model.apply, optimizer_update_fn=optimizer.update)

    print("Starting training...")
    for epoch in range(num_epochs):
        key_train_loop, key_epoch_shuffle = jax.random.split(key_train_loop)
        perm = jax.random.permutation(key_epoch_shuffle, num_samples)
        # ... (rest of batching and training loop as before) ...
        epoch_loss = 0.0
        num_batches = num_samples // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_spatial_dist = spatial_distances_data[perm][start_idx:end_idx]
            batch_ind_vars = independent_vars_data[perm][start_idx:end_idx]
            batch_y_true = y_true_data[perm][start_idx:end_idx]

            params, opt_state, loss_val = jitted_train_step(
                params=params, opt_state=opt_state,
                spatial_distances=batch_spatial_dist, independent_vars=batch_ind_vars, y_true=batch_y_true
            )
            epoch_loss += loss_val
        
        avg_epoch_loss = epoch_loss / num_batches
        if (epoch + 1) % 25 == 0 or epoch == 0: # Adjusted print frequency
            print(f"Epoch {epoch+1}/{num_epochs}, Avg NLL Loss: {avg_epoch_loss:.4f}")
    print("Training finished.")

    # Get final predictions AND WEIGHTS using the trained parameters
    final_mu_preds, final_sigma_sq_preds, final_weights_mu, final_weights_sigma = model.apply(
        {'params': params}, spatial_distances_data, independent_vars_data
    )

    print("\nSample true vs. predicted (first 5 from training data for demo):")
    for i in range(min(5, num_samples)):
        print(f"  True y: {jnp.squeeze(y_true_data[i]):.2f}, "
              f"Predicted mu: {final_mu_preds[i]:.2f}, "
              f"Predicted sigma: {jnp.sqrt(final_sigma_sq_preds[i]):.2f}")

    # Call the general evaluation plotting function
    print("\nGenerating general evaluation plots...")
    generate_evaluation_plots(y_true_data,
                              final_mu_preds,
                              final_sigma_sq_preds,
                              data_name="Training Demo Data")
    
    # Call the NEW weight visualization plotting function
    print("\nGenerating spatial weight visualizations...")
    generate_weight_visualizations(final_weights_mu,
                                   final_weights_sigma,
                                   spatial_distances_data,
                                   num_ind_vars, # Pass num_independent_vars
                                   data_name="Training Demo Data")
    print("All plotting complete.")

# --- 8. Main execution block ---
if __name__ == '__main__':
    run_example()