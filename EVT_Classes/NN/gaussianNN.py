import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from functools import partial
import matplotlib.pyplot as plt
import os

# --- JAX Setup ---
key = jax.random.PRNGKey(0)
# Set JAX traceback filtering to 'off' to get more detailed error messages.
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
DIST_EPSILON = 1e-8 # A small constant to prevent division by zero or log(0).

# ==============================================================================
# SECTION 1: DATA PREPARATION & UTILITIES
# ==============================================================================

# --- K-Nearest Neighbors Hyperparameter ---
K_NEIGHBORS = 5000 # The number of nearest neighbors to consider for each point.

def calculate_distances(point_locs, reference_locs):
    """Calculates the Euclidean distance matrix between two sets of points."""
    # Broadcasting to compute pairwise distances: (n, 1, d) - (1, m, d) -> (n, m, d)
    return jnp.sqrt(jnp.sum((point_locs[:, jnp.newaxis, :] - reference_locs[jnp.newaxis, :, :])**2, axis=-1))

# --- K-Nearest Neighbors Function ---
@partial(jax.jit, static_argnums=(2,))
def find_knn(query_locs, reference_locs, k):
    """
    Finds the k-nearest neighbors for each query point from a set of reference points.
    Uses jax.lax.top_k for efficiency, which is much faster than a full sort.
    """
    all_dists = calculate_distances(query_locs, reference_locs)
    # We use top_k on the *negative* distances. The largest negative distances
    # correspond to the smallest positive distances.
    neg_dists = -all_dists
    top_k_neg_dists, top_k_indices = jax.lax.top_k(neg_dists, k=k)
    # Return the positive distances and the indices of the neighbors.
    return -top_k_neg_dists, top_k_indices

def calculate_ols_coefficients(x_train, y_train):
    """Calculates global OLS coefficients to serve as a baseline prior."""
    x_train_intercept = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    ols_mu_model = LinearRegression(fit_intercept=False).fit(x_train_intercept, y_train)
    mu_coeffs = ols_mu_model.coef_
    mu_predictions = ols_mu_model.predict(x_train_intercept)
    residuals = y_train - mu_predictions
    log_variance_targets = np.log(residuals**2 + 1e-6)
    ols_sigma_model = LinearRegression(fit_intercept=False).fit(x_train_intercept, log_variance_targets)
    sigma_coeffs = ols_sigma_model.coef_
    global_coeffs = np.concatenate([mu_coeffs, sigma_coeffs])
    return jnp.array(mu_coeffs), jnp.array(sigma_coeffs), jnp.array(global_coeffs)

def get_simulated_data(n_samples=5000, key_data=jax.random.PRNGKey(123), **kwargs):
    """Generates simulated data and returns the ground truth mu and sigma."""
    config = {'x_minval': -2 * jnp.pi, 'x_maxval': 2 * jnp.pi, 'amplitude': 1.5, 'frequency': 1.0, 'phase': 0.0, 'vertical_offset': 0.5, 'x_slope_coeff': 1.0, 'noise_y_std': 0.3, 'noise_beta0_std': 0.5, 'noise_beta1_std': 0.05}
    config.update(kwargs)
    key_beta0, key_beta1, key_y_noise = jax.random.split(key_data, 3)
    X_features = jnp.linspace(config['x_minval'], config['x_maxval'], n_samples).reshape(-1, 1)
    locs = X_features # Using features as locations
    
    # --- Latent Beta Coefficient Generation ---
    beta0_main_curve = config['amplitude'] * jnp.sin(config['frequency'] * locs.flatten() + config['phase'])
    beta0_noise = jax.random.normal(key_beta0, (n_samples,)) * config['noise_beta0_std']
    beta0_values = config['vertical_offset'] + beta0_main_curve + beta0_noise
    
    beta1_main_curve = 0.7 * config['amplitude'] * jnp.cos(1.5 * config['frequency'] * locs.flatten() + config['phase'])
    beta1_noise = jax.random.normal(key_beta1, (n_samples,)) * config['noise_beta1_std']
    beta1_values = config['x_slope_coeff'] + beta1_main_curve + beta1_noise
    
    # --- Ground Truth mu and sigma Generation ---
    # This is the TRUE mean of the data generating process
    true_mu = beta0_values + beta1_values * locs.flatten()
    
    # This is the TRUE standard deviation (aleatoric uncertainty) of the data generating process
    # UPDATED to be a slow sinusoidal wave
    sigma_frequency = 0.8 # A lower frequency for a "slow" wave
    sigma_amplitude = config['noise_y_std'] * 0.7 # Make it vary by 70% of the base noise
    true_sigma = config['noise_y_std'] + sigma_amplitude * jnp.sin(sigma_frequency * locs.flatten())
    
    # --- Final Y Generation ---
    # Add noise to the true mean to get the final observed y
    y_noise = jax.random.normal(key_y_noise, (n_samples,)) * true_sigma
    y = true_mu + y_noise
    
    return (np.array(locs), np.array(X_features), np.array(y), 
            np.array(true_mu), np.array(true_sigma))

# --- Data Preparation ---
locs, X, y, true_mu, true_sigma = get_simulated_data(n_samples=10000, noise_y_std=1.5, x_slope_coeff=0.9)
locs_train_val, locs_test, X_train_val, X_test, y_train_val, y_test = train_test_split(locs, X, y, test_size=0.20, random_state=42)
locs_train, locs_val, X_train, X_val, y_train, y_val = train_test_split(locs_train_val, X_train_val, y_train_val, test_size=0.25, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_train_jax, y_train_jax = jnp.array(X_train_scaled), jnp.array(y_train)
X_val_jax, y_val_jax = jnp.array(X_val_scaled), jnp.array(y_val)
locs_train_jax, locs_val_jax = jnp.array(locs_train), jnp.array(locs_val)

# --- Pre-compute the (N, K) nearest neighbors matrix before training ---
print(f"Pre-computing the (N, K) nearest neighbors matrix for the entire training set...")
precomputed_knn_dists, precomputed_knn_indices = find_knn(locs_train_jax, locs_train_jax, K_NEIGHBORS)
print("KNN distance and index matrices computed.")

# --- Calculate distance stats from the pre-computed matrix ---
swnn_input_dist_mean = jnp.mean(precomputed_knn_dists)
swnn_input_dist_std = jnp.std(precomputed_knn_dists)
print(f"Distance stats for k-NN: Mean={swnn_input_dist_mean:.4f}, Std={swnn_input_dist_std:.4f}")

_, _, global_ols_coeffs = calculate_ols_coefficients(X_train_scaled, y_train)

# ==============================================================================
# SECTION 2: DEFINE AND TRAIN THE PROBABILISTIC MODEL
# ==============================================================================
print("\n--- Training Probabilistic Model ---")

class ProbabilisticCoefficientAdapterNet(nn.Module):
    """
    A network that outputs parameters for a distribution over coefficients.
    For each coefficient, it outputs a mean and a log standard deviation.
    """
    num_coeffs: int
    hidden_dims: tuple = (64, 32)

    @nn.compact
    def __call__(self, x, train: bool):
        for i, h_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=h_dim, name=f'hidden_dense_{i}')(x)
            x = nn.PReLU()(x)
        
        combined_output = nn.Dense(features=self.num_coeffs * 2, name='output_dist_params')(x)
        mean_factors, log_std_factors = jnp.split(combined_output, 2, axis=-1)
        log_stds = nn.tanh(log_std_factors) * 5.0
        
        return mean_factors, log_stds

class GNNWRTrainState(train_state.TrainState):
    batch_stats: dict

def nll_loss(mu_pred, sigma_pred, targets):
    """Negative Log-Likelihood loss for a Gaussian distribution."""
    safe_sigma = sigma_pred + DIST_EPSILON
    return jnp.mean(0.5 * jnp.log(2 * jnp.pi * safe_sigma**2) + ((targets - mu_pred)**2 / (2 * safe_sigma**2)))

def kl_divergence_loss(mu, log_std, prior_mu, prior_log_std):
    """Calculates the KL divergence between the predicted distribution (q) and a prior (p)."""
    prior_var = jnp.exp(2 * prior_log_std)
    q_var = jnp.exp(2 * log_std)
    term1 = prior_log_std - log_std
    term2 = (q_var + (mu - prior_mu)**2) / (2 * prior_var + DIST_EPSILON)
    kl_div = term1 + term2 - 0.5
    return jnp.mean(jnp.sum(kl_div, axis=-1))

@partial(jax.jit, static_argnums=(3, 6, 10))
def gnnwr_get_beta_dists(model_params, model_batch_stats, query_locs, apply_fn,
                         x_batch, global_ols_coeffs, is_training: bool,
                         reference_locs, swnn_input_dist_mean, swnn_input_dist_std, k: int):
    """Forward pass to get the parameters of the beta distributions."""
    knn_dists, _ = find_knn(query_locs, reference_locs, k)
    standardized_dists = (knn_dists - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)
    
    model_vars = {'params': model_params, 'batch_stats': model_batch_stats}
    
    mean_factors, log_std_factors = apply_fn(model_vars, standardized_dists, train=is_training)
    if isinstance(mean_factors, tuple):
        (mean_factors, log_std_factors), _ = (mean_factors, log_std_factors)

    local_beta_means = mean_factors * global_ols_coeffs
    local_beta_log_stds = log_std_factors
    
    return local_beta_means, local_beta_log_stds

@partial(jax.jit, static_argnames=('adapter_net_apply_fn',))
def train_step_dist(state: GNNWRTrainState, x_batch, knn_dists_batch, y_batch,
                    global_ols_coeffs, adapter_net_apply_fn,
                    dropout_key, swnn_input_dist_mean, swnn_input_dist_std, kl_weight: float):
    """Performs a single training step using the variational loss."""
    def loss_fn(model_params):
        model_vars = {'params': model_params, 'batch_stats': state.batch_stats}
        standardized_dists = (knn_dists_batch - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)
        x_intercept = jnp.concatenate([jnp.ones((x_batch.shape[0], 1)), x_batch], axis=1)

        (mean_factors, log_std_factors), new_model_state = adapter_net_apply_fn(
            model_vars, standardized_dists, train=True, mutable=['batch_stats'], rngs={'dropout': dropout_key}
        )
        
        local_beta_means = mean_factors * global_ols_coeffs
        local_beta_log_stds = log_std_factors

        key_sample, _ = jax.random.split(dropout_key)
        epsilon = jax.random.normal(key_sample, shape=local_beta_means.shape)
        local_betas_sample = local_beta_means + jnp.exp(local_beta_log_stds) * epsilon

        mu_coeffs_sample = local_betas_sample[:, :2]
        log_sigma2_coeffs_sample = local_betas_sample[:, 2:]
        
        mu_pred = jnp.sum(x_intercept * mu_coeffs_sample, axis=1)
        log_sigma2_pred = jnp.sum(x_intercept * log_sigma2_coeffs_sample, axis=1)
        log_sigma2_pred = jnp.clip(log_sigma2_pred, a_min=-15.0, a_max=15.0)
        sigma_pred = jnp.exp(0.5 * log_sigma2_pred)
        
        reconstruction_loss = nll_loss(mu_pred, sigma_pred, y_batch)
        
        prior_mu = global_ols_coeffs
        prior_log_std = jnp.log(1.0) 
        kl_loss = kl_divergence_loss(local_beta_means, local_beta_log_stds, prior_mu, prior_log_std)
        
        total_loss = reconstruction_loss + kl_weight * kl_loss
        
        return total_loss, (new_model_state['batch_stats'], reconstruction_loss, kl_loss)

    (loss_val, (new_batch_stats, recon_loss, kl_loss_val)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats)
    return new_state, loss_val, recon_loss, kl_loss_val

# --- Model and Optimizer Setup ---
adapter_net = ProbabilisticCoefficientAdapterNet(num_coeffs=len(global_ols_coeffs))
key, key_init, key_dropout = jax.random.split(key, 3)
dummy_input = jax.random.normal(key_init, (1, K_NEIGHBORS))
model_variables = adapter_net.init(key_init, dummy_input, train=False)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.0005)
)
prob_state = GNNWRTrainState.create(
    apply_fn=adapter_net.apply, params=model_variables['params'],
    tx=optimizer, batch_stats=model_variables.get('batch_stats', {})
)

# --- Training Loop ---
epochs, batch_size, patience, best_val_loss, patience_counter = 5000, 5000, 1000, float('inf'), 0
kl_weight = 0.05

for epoch in range(epochs):
    key, key_shuffle = jax.random.split(key)
    perm = jax.random.permutation(key_shuffle, X_train_jax.shape[0])
    X_train_s, y_train_s, knn_dists_s = X_train_jax[perm], y_train_jax[perm], precomputed_knn_dists[perm]
    
    epoch_total_loss, epoch_recon_loss, epoch_kl_loss = 0.0, 0.0, 0.0
    num_batches = int(np.ceil(X_train_s.shape[0] / batch_size))
    
    for i in range(num_batches):
        start_idx, end_idx = i * batch_size, (i + 1) * batch_size
        x_b, y_b, knn_dists_b = X_train_s[start_idx:end_idx], y_train_s[start_idx:end_idx], knn_dists_s[start_idx:end_idx]
        key, step_key = jax.random.split(key)
        
        prob_state, total_loss_item, recon_loss_item, kl_loss_item = train_step_dist(
            prob_state, x_b, knn_dists_b, y_b, global_ols_coeffs,
            adapter_net.apply, step_key, swnn_input_dist_mean, swnn_input_dist_std, kl_weight
        )
        epoch_total_loss += total_loss_item.item()
        epoch_recon_loss += recon_loss_item.item()
        epoch_kl_loss += kl_loss_item.item()

        if jnp.isnan(total_loss_item) or jnp.isinf(total_loss_item):
            print(f"\n!!! Training stopped at epoch {epoch+1}, batch {i+1} due to NaN/inf loss. !!!")
            break
    if jnp.isnan(total_loss_item) or jnp.isinf(total_loss_item): break

    avg_epoch_total_loss = epoch_total_loss / num_batches
    avg_epoch_recon_loss = epoch_recon_loss / num_batches
    avg_epoch_kl_loss = epoch_kl_loss / num_batches
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:03d} | Total Loss: {avg_epoch_total_loss:.5f} "
              f"[Recon: {avg_epoch_recon_loss:.5f}, KL: {avg_epoch_kl_loss:.5f}]")

    if avg_epoch_total_loss < best_val_loss:
        best_val_loss, best_prob_state, patience_counter = avg_epoch_total_loss, prob_state, 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Training stopped early at epoch {epoch+1}.")
        break

# ==============================================================================
# SECTION 3: DIAGNOSTIC PLOTS FOR THE PROBABILISTIC MODEL
# ==============================================================================
print("\n--- Generating Diagnostic Plots for Trained Probabilistic Model ---")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
plt.style.use('seaborn-v0_8-whitegrid')
X_all_scaled = scaler_X.transform(X)
X_all_jax, locs_all_jax = jnp.array(X_all_scaled), jnp.array(locs)

# --- Monte Carlo Sampling to get Confidence Intervals ---
print("Performing Monte Carlo sampling for confidence intervals...")
num_samples = 500
# 1. Get the parameters of the beta distributions once
local_beta_means, local_beta_log_stds = gnnwr_get_beta_dists(
    best_prob_state.params, best_prob_state.batch_stats,
    query_locs=locs_all_jax, apply_fn=adapter_net.apply, x_batch=X_all_jax,
    global_ols_coeffs=global_ols_coeffs, is_training=False, reference_locs=locs_train_jax,
    swnn_input_dist_mean=swnn_input_dist_mean, swnn_input_dist_std=swnn_input_dist_std, k=K_NEIGHBORS
)

# 2. Sample from these distributions many times
key, sample_key = jax.random.split(key)
# Shape: (num_samples, num_data_points, num_coeffs)
epsilon_samples = jax.random.normal(sample_key, (num_samples,) + local_beta_means.shape)
beta_samples = local_beta_means + jnp.exp(local_beta_log_stds) * epsilon_samples

# 3. Calculate mu and sigma for each sample
x_intercept_all = jnp.concatenate([jnp.ones((X_all_jax.shape[0], 1)), X_all_jax], axis=1)
mu_coeff_samples = beta_samples[:, :, :2]
sigma_coeff_samples = beta_samples[:, :, 2:]

# Broadcasting to get mu and sigma samples
# (num_samples, num_data_points, 2) * (1, num_data_points, 2) -> (num_samples, num_data_points)
mu_samples = jnp.sum(mu_coeff_samples * x_intercept_all, axis=2)
log_sigma2_samples = jnp.sum(sigma_coeff_samples * x_intercept_all, axis=2)
sigma_samples = jnp.exp(0.5 * jnp.clip(log_sigma2_samples, a_min=-15.0, a_max=15.0))

# 4. Compute statistics (mean and percentiles) over the samples
mu_pred_all = jnp.mean(mu_samples, axis=0)
sigma_pred_all = jnp.mean(sigma_samples, axis=0)

mu_ci_lower, mu_ci_upper = jnp.percentile(mu_samples, jnp.array([5.0, 95.0]), axis=0)
sigma_ci_lower, sigma_ci_upper = jnp.percentile(sigma_samples, jnp.array([5.0, 95.0]), axis=0)
print("Sampling complete.")

# Plot 1: Overall Model Fit
# -------------------------------------------------
plt.figure(figsize=(12, 7))
sorted_indices_all = np.argsort(X[:, 0])
plt.scatter(X[:, 0], y, label='Data Points', alpha=0.3, s=20, color='gray')
plt.plot(X[sorted_indices_all, 0], mu_pred_all[sorted_indices_all], label=r'Predicted Mean ($\mu_{pred}$)', color='firebrick', linewidth=2.5)
# This shows the mean of the predicted sigmas, representing total uncertainty
plt.fill_between(X[sorted_indices_all, 0], 
                 (mu_pred_all - 2 * sigma_pred_all)[sorted_indices_all], 
                 (mu_pred_all + 2 * sigma_pred_all)[sorted_indices_all],
                 color='firebrick', alpha=0.2, label=r'Total Predicted Uncertainty ($\pm 2\sigma_{pred}$)')
plt.title(r'Overall Model Fit and Predictive Uncertainty', fontsize=16)
plt.xlabel('Feature X (Location)'), plt.ylabel('Target y'), plt.legend()
plt.tight_layout(), plt.show()


# Plot 2: Predicted vs. True Distribution Parameters with Confidence Intervals
# -------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
sorted_indices = np.argsort(locs[:, 0])
x_sorted = locs[sorted_indices, 0]

# --- Mean (mu) Comparison ---
axes[0].plot(x_sorted, true_mu[sorted_indices], color='black', linestyle='--', label=r'True $\mu$')
axes[0].plot(x_sorted, mu_pred_all[sorted_indices], color='navy', label=r'Predicted $\mu$ (Mean)')
axes[0].fill_between(x_sorted, 
                     mu_ci_lower[sorted_indices], 
                     mu_ci_upper[sorted_indices], 
                     color='cornflowerblue', alpha=0.4, label=r'90% CI on $\mu$ (Epistemic)')
axes[0].set_title(r'Predicted Mean vs. True Mean', fontsize=14)
axes[0].set_ylabel('Value')
axes[0].legend()

# --- Standard Deviation (sigma) Comparison ---
axes[1].plot(x_sorted, true_sigma[sorted_indices], color='black', linestyle='--', label=r'True $\sigma$')
axes[1].plot(x_sorted, sigma_pred_all[sorted_indices], color='darkgreen', label=r'Predicted $\sigma$ (Mean)')
axes[1].fill_between(x_sorted, 
                     sigma_ci_lower[sorted_indices], 
                     sigma_ci_upper[sorted_indices], 
                     color='mediumseagreen', alpha=0.4, label=r'90% CI on $\sigma$ (Epistemic)')
axes[1].set_title(r'Predicted Std. Dev. vs. True Std. Dev. (Aleatoric Uncertainty)', fontsize=14)
axes[1].set_xlabel('Feature X (Location)')
axes[1].set_ylabel('Value')
axes[1].legend()

plt.tight_layout()
plt.show()
