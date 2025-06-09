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

# For reproducibility
key = jax.random.PRNGKey(0)
import os
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

# Epsilon for standardization to prevent division by zero
DIST_EPSILON = 1e-7

# --- Utility Functions ---
def calculate_distances(point_locs, reference_locs):
    """Calculates 1D distances.
    point_locs: (n_points,) array of locations for the points of interest.
    reference_locs: (n_refs,) array of locations for the reference points (e.g., all training locs).
    Returns: (n_points, n_refs) array of absolute distances.
    """
    return jnp.sqrt((point_locs[:, jnp.newaxis] - reference_locs[jnp.newaxis, :])**2)

def calculate_ols_coefficients(X_train, y_train):
    """
    Returns:
    - mu_coeffs: Coefficients for mean prediction (shape: (2,))
    - sigma_coeffs: Coefficients for log-variance prediction (shape: (2,))
    - full_coeffs: Concatenated array of both (shape: (4,))
    """
    # Add intercept term to X
    X_train_intercept = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)

    # --- Fit OLS for the mean (mu) ---
    ols_mu = LinearRegression(fit_intercept=False)
    ols_mu.fit(X_train_intercept, y_train)
    mu_coeffs = ols_mu.coef_  # shape (2,)
    mu_preds = ols_mu.predict(X_train_intercept)

    # --- Fit OLS for log-variance ---
    residuals = y_train - mu_preds
    log_variance_targets = np.log(residuals**2 + 1e-6)  # prevent log(0)

    ols_sigma = LinearRegression(fit_intercept=False)
    ols_sigma.fit(X_train_intercept, log_variance_targets)
    sigma_coeffs = ols_sigma.coef_  # shape (2,)

    full_coeffs = np.concatenate([mu_coeffs, sigma_coeffs])  # shape (4,)
    return jnp.array(mu_coeffs), jnp.array(sigma_coeffs), jnp.array(full_coeffs)


# --- Data Generation Function (True Weights based on Conceptual Reference Points) ---
def get_simulated_data(
    n_samples=300,
    key_data=jax.random.PRNGKey(123),
    x_minval=-2 * jnp.pi,
    x_maxval=2 * jnp.pi,
    curve_type='sin',
    amplitude=1.5,
    frequency=1.0,
    phase=0.0,
    vertical_offset=0.5,
    x_slope_coeff=1.0,
    noise_y_std=0.5,
    noise_beta0_std=0.05,
    noise_beta1_std=0.05,
    noise_type='constant'  # string or callable
):
    key_beta0_noise, key_beta1_noise, key_y_noise = jax.random.split(key_data, 3)

    X = jnp.linspace(x_minval, x_maxval, n_samples).reshape(-1, 1)
    locs = X[:, 0]

    if curve_type == 'sin':
        main_curve = amplitude * jnp.sin(frequency * locs + phase)
    elif curve_type == 'cos':
        main_curve = amplitude * jnp.cos(frequency * locs + phase)
    else:
        raise ValueError("curve_type must be 'sin' or 'cos'")

    beta0_noise_values = jax.random.normal(key_beta0_noise, (n_samples,)) * noise_beta0_std
    beta0_values = vertical_offset + main_curve + beta0_noise_values

    beta1_noise_values = jax.random.normal(key_beta1_noise, (n_samples,)) * noise_beta1_std
    beta1_values = x_slope_coeff + beta1_noise_values

    y_deterministic = beta0_values + beta1_values * locs

    # Determine noise std pattern
    if isinstance(noise_type, str):
        if noise_type == 'constant':
            dynamic_std = jnp.ones(n_samples) * noise_y_std
        elif noise_type == 'wavy':
            wave1 = 1.5 * jnp.sin(2.5 * locs + 0.3)
            wave2 = 3.0 * jnp.sin(6.3 * locs + 1.8)
            wave3 = 5.0 * jnp.cos(9.7 * locs + 0.7)
            pattern = jnp.abs(wave1 + wave2 + wave3) + 0.2  # Ensure nonzero
            pattern /= jnp.mean(pattern)  # Normalize to mean 1
            dynamic_std = noise_y_std * pattern
        elif noise_type == 'chaotic':
            # Irregular spikes + some smooth base noise
            key_pattern = jax.random.PRNGKey(999)
            random_mask = jax.random.bernoulli(key_pattern, p=0.15, shape=(n_samples,))
            spike_noise = jnp.where(random_mask, jnp.abs(jax.random.normal(key_pattern, (n_samples,))) * 5.0, 0.0)

            base_wave = 0.3 * jnp.sin(3 * locs + 1.0) + 0.3 * jnp.cos(7 * locs + 0.5)
            base_wave = jnp.abs(base_wave) + 0.1  # ensure nonzero

            dynamic_std = noise_y_std * (base_wave + spike_noise)
        else:
            raise ValueError("noise_type must be 'constant', 'wavy', or a callable")
    elif callable(noise_type):
        dynamic_std = noise_y_std * noise_type(locs)
    else:
        raise ValueError("noise_type must be 'constant', 'wavy', or a callable")

    y_noise = jax.random.normal(key_y_noise, (n_samples,)) * dynamic_std
    y = y_deterministic + y_noise

    local_coeffs = jnp.zeros((n_samples, 2))
    local_coeffs = local_coeffs.at[:, 0].set(beta0_values)
    local_coeffs = local_coeffs.at[:, 1].set(beta1_values)

    return (np.array(locs), np.array(X), np.array(y), np.array(local_coeffs))

# --- SWNN Flax Module (Input dimension is num_training_locations) ---
class SWNN(nn.Module):
    num_outputs: int
    hidden_dims: tuple = (64, 32) # Corresponds to hidden_layer_neurals
    dropout_rate: float = 0.1        # if > 0, dropout is applied. keep_prop = 1.0 - dropout_rate
    use_batch_size_norm: bool = True      # Corresponds to batch_size_norm flag
    # New attributes for more flexibility, mirroring TF S_NETWORK/GW_NETWORK if needed:
    # activation_fn: callable = nn.relu # Corresponds to activate_fun
    # kernel_init_fn: callable = nn.initializers.he_normal() # Corresponds to weight_init

    @nn.compact
    def __call__(self, x, train: bool): # x is (batch_size, num_training_locations)
                                        # train flag corresponds to bn_is_training

        # For clarity, you could use self.activation_fn and self.kernel_init_fn if you add them as attributes
        activation_fn = nn.relu # Or self.activation_fn
        kernel_init_fn = nn.initializers.he_normal() # Or self.kernel_init_fn

        for i, h_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=h_dim,
                         kernel_init=kernel_init_fn,
                         name=f'hidden_dense_{i}')(x)
            x = nn.PReLU()(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate,
                               deterministic=not train,
                               name=f'dropout_{i}')(x)

        # Output layer:
        # Matches GW_NETWORK's output_W initialization and linear output for gtweight
        weights_output = nn.Dense(features=self.num_outputs,
                                  kernel_init=kernel_init_fn,
                                  name='output_weights')(x)

        # !!! Key change: Removed nn.softmax to match the TF GW_NETWORK's linear output layer for gtweight !!!
        # If the paper implies a softmax or other activation on the GNNWR weights *after* this network,
        # it should be applied outside this module or explicitly added if it's part of the "spatial network" itself.

        return weights_output


# --- TrainState ---
class GNNWRTrainState(train_state.TrainState):
    batch_size_stats: dict

# --- Loss Function ---
def nll_loss(mu_pred, sigma_pred, targets):
    return jnp.sum((targets - mu_pred)**2 / (2 * sigma_pred**2) + jnp.log(sigma_pred) + 0.5 * jnp.log(2 * jnp.pi))

# --- GNNWR Prediction Logic (SWNN input: distances to training locs) ---
@partial(jax.jit, static_argnums=(3, 6))
def gnnwr_predict(swnn_params, swnn_batch_size_stats,
                  dists_to_all_train_locs,
                  apply_fn,
                  X_independent_vars_batch_size,
                  ols_coeffs,
                  is_training: bool,
                  sw_dist_mean: jnp.ndarray,
                  sw_dist_std: jnp.ndarray,
                  dropout_key_predict=None):

    X_intercept = jnp.concatenate([jnp.ones((X_independent_vars_batch_size.shape[0], 1)),
                                   X_independent_vars_batch_size], axis=1)

    # Standardize distance input
    standardized_dists = (dists_to_all_train_locs - sw_dist_mean) / (sw_dist_std + DIST_EPSILON)

    swnn_vars = {'params': swnn_params}
    if swnn_batch_size_stats:
        swnn_vars['batch_size_stats'] = swnn_batch_size_stats

    dropout_rngs = {'dropout': dropout_key_predict} if dropout_key_predict else {}

    if is_training:
        learned_weights, _ = apply_fn(
            swnn_vars, standardized_dists, train=True, mutable=['batch_size_stats'], rngs=dropout_rngs
        )
    else:
        learned_weights = apply_fn(
            swnn_vars, standardized_dists, train=False, rngs=dropout_rngs
        )

    # Split weights
    weights_mu = learned_weights[:, :2]
    weights_log_sigma2 = learned_weights[:, 2:]

    ols_mu = ols_coeffs[:2]
    ols_sigma = ols_coeffs[2:]

    mu_coeffs = weights_mu * ols_mu
    log_sigma2_coeffs = weights_log_sigma2 * ols_sigma

    mu_pred = jnp.sum(X_intercept * mu_coeffs, axis=1)
    log_sigma2_pred = jnp.sum(X_intercept * log_sigma2_coeffs, axis=1)

    sigma_pred = jnp.exp(0.5 * log_sigma2_pred)  # convert log(σ²) to σ

    return mu_pred, sigma_pred, learned_weights

# --- Training and Evaluation Step Functions ---
@partial(jax.jit, static_argnames=('swnn_apply_fn',))
def train_step(state: GNNWRTrainState,
               batch_X, batch_locs, batch_y,
               all_train_locs,
               ols_coeffs_glob,
               swnn_apply_fn, dropout_key,
               sw_dist_mean: jnp.ndarray,
               sw_dist_std: jnp.ndarray):

    dists = calculate_distances(batch_locs, all_train_locs)
    standardized_dists = (dists - sw_dist_mean) / (sw_dist_std + DIST_EPSILON)

    def loss_fn(swnn_params_inner):
        swnn_vars = {'params': swnn_params_inner, 'batch_size_stats': state.batch_size_stats}
        learned_weights, new_stats = swnn_apply_fn(
            swnn_vars, standardized_dists, train=True, mutable=['batch_size_stats'], rngs={'dropout': dropout_key}
        )

        weights_mu = learned_weights[:, :2]
        weights_sigma = learned_weights[:, 2:]
        ols_mu = ols_coeffs_glob[:2]
        ols_sigma = ols_coeffs_glob[2:]

        X_intercept = jnp.concatenate([jnp.ones((batch_X.shape[0], 1)), batch_X], axis=1)
        mu_pred = jnp.sum(X_intercept * (weights_mu * ols_mu), axis=1)
        log_sigma2_pred = jnp.sum(X_intercept * (weights_sigma * ols_sigma), axis=1)
        sigma_pred = jnp.exp(0.5 * log_sigma2_pred)

        loss = nll_loss(mu_pred, sigma_pred, batch_y)
        return loss, new_stats['batch_size_stats']

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, new_stats), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state.replace(batch_size_stats=new_stats), loss_val

@partial(jax.jit, static_argnames=('swnn_apply_fn_eval',))
def eval_step(state_params_eval, state_batch_size_stats_eval,
              X_eval, locs_eval, y_eval,
              all_train_locs,
              ols_coeffs_glob_eval,
              swnn_apply_fn_eval,
              sw_dist_mean: jnp.ndarray,
              sw_dist_std: jnp.ndarray):

    dists = calculate_distances(locs_eval, all_train_locs)

    mu_pred, sigma_pred, learned_weights = gnnwr_predict(
        state_params_eval,
        state_batch_size_stats_eval,
        dists,
        swnn_apply_fn_eval,
        X_eval,
        ols_coeffs_glob_eval,
        is_training=False,
        sw_dist_mean=sw_dist_mean,
        sw_dist_std=sw_dist_std
    )

    loss_val = nll_loss(mu_pred, sigma_pred, y_eval)
    return loss_val, mu_pred, sigma_pred, learned_weights

# --- Data Preparation ---
# ------ GENERATION ------------#
locs, X, y, local_coeffs = \
    get_simulated_data(n_samples=1000, key_data=jax.random.PRNGKey(789),x_slope_coeff=0.5,noise_type="wavy")
original_indices_np = np.arange(X.shape[0])

locs_train_val, locs_test, \
X_train_val, X_test, \
y_train_val, y_test, \
indices_train_val, indices_test= train_test_split(
    locs, X, y, original_indices_np,
    test_size=0.20, random_state=42)

locs_train, locs_val, \
X_train, X_val, \
y_train, y_val, \
indices_train, indices_val = train_test_split(
    locs_train_val, X_train_val, y_train_val, indices_train_val,
    test_size=0.25, random_state=42) # 0.25 of (1-0.2) = 0.2 -> 60% train, 20% val, 20% test

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, alpha=0.6, edgecolors='w', linewidth=0.5)
plt.title('Simulated Data: y vs. X')
plt.xlabel('Feature X1')
plt.ylabel('Target y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
local_coeffs_test = local_coeffs[indices_test]

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

X_train_jax, y_train_jax = jnp.array(X_train_scaled), jnp.array(y_train)
X_val_jax, y_val_jax = jnp.array(X_val_scaled), jnp.array(y_val)
X_test_jax, y_test_jax = jnp.array(X_test_scaled), jnp.array(y_test)
X_train_jax, y_train_jax = jnp.array(X_train_scaled), jnp.array(y_train)
locs_train_jax = jnp.array(locs_train)
locs_val_jax = jnp.array(locs_val)

# Calculate mean and std of distances FOR SWNN input standardization
# Based on all pairwise distances between training locations themselves for SWNN input vectors
print("Calculating distance statistics for SWNN input standardization...")
all_dists_matrix_train_for_stats = calculate_distances(locs_train_jax, locs_train_jax)

dist_mean_for_sw = jnp.mean(all_dists_matrix_train_for_stats)
dist_std_for_sw = jnp.std(all_dists_matrix_train_for_stats)
print(f"SWNN Input Distance Mean for standardization: {dist_mean_for_sw}")
print(f"SWNN Input Distance Std for standardization: {dist_std_for_sw}")

print(f"1D Data. SWNN input: distances to ALL {locs_train_jax.shape[0]} training locations (now standardized).")

mu_coeffs, sigma_coeffs, ols_coeffs = calculate_ols_coefficients(X_train_scaled, y_train)
print("Global OLS Coefficients:", ols_coeffs)

# --- SWNN Initialization and Training ---
num_training_locations = locs_train_jax.shape[0]
SWNN_model = SWNN(num_outputs=len(ols_coeffs), hidden_dims=(32, 16), dropout_rate=0.1)

key_init, key_dropout, key_dummy_input = jax.random.split(key, num=3)
# Dummy input: (1 sample, distances to all training locations)
# Generate raw-scale dummy distances, e.g., based on typical range of locations [0,10]
raw_dummy_dists = jax.random.uniform(key_dummy_input, (1, num_training_locations), minval=0.0, maxval=10.0)
# Standardize the dummy input
dummy_swnn_input = (raw_dummy_dists - dist_mean_for_sw) / (dist_std_for_sw + DIST_EPSILON)
swnn_vars = SWNN_model.init(key_init, dummy_swnn_input, train=False)

state = GNNWRTrainState.create(
    apply_fn=SWNN_model.apply,
    params=swnn_vars['params'],
    tx=optax.adam(learning_rate=0.0001),
    batch_size_stats=swnn_vars.get('batch_size_stats', {})
)

epochs = 100; batch_size = 132; patience = 50
best_val_loss = float('inf'); patience_counter = 10
best_state_params = state.params; best_state_batch_size_stats = state.batch_size_stats
current_dropout_key = key_dropout

for epoch in range(epochs):
    num_train_samples = X_train_jax.shape[0]
    key_shuffle, current_dropout_key = jax.random.split(current_dropout_key)
    perm = jax.random.permutation(key_shuffle, num_train_samples)

    X_train_s, y_train_s, locs_train_s = X_train_jax[perm], y_train_jax[perm], locs_train_jax[perm]

    epoch_train_loss = 0.0
    num_batch_size = int(np.ceil(num_train_samples / batch_size))
    for batch in range(num_batch_size):
        start_idx, end_idx = batch*batch_size, min((batch+1)*batch_size, num_train_samples)
        bX, bL, bY = X_train_s[start_idx:end_idx], locs_train_s[start_idx:end_idx], y_train_s[start_idx:end_idx]

        step_dropout_key = jax.random.fold_in(current_dropout_key, batch)
        state, loss_item = train_step(state, bX, bL, bY,
                                            locs_train_jax, # ALL training locations
                                            ols_coeffs,
                                            SWNN_model.apply, step_dropout_key,
                                            dist_mean_for_sw, dist_std_for_sw) # Pass dist stats                     
        epoch_train_loss += loss_item.item()
    avg_epoch_train_loss = epoch_train_loss / num_batch_size

    epoch_val_loss = 0.0
    num_val_batch_size = int(np.ceil(X_val_jax.shape[0] / batch_size))
    for val_batch in range(num_val_batch_size):
        start_idx, end_idx = val_batch*batch_size, min((val_batch+1)*batch_size, X_val_jax.shape[0])
        vbX, vbL, vbY = X_val_jax[start_idx:end_idx], locs_val_jax[start_idx:end_idx], y_val_jax[start_idx:end_idx]

        val_loss_item, *_  = eval_step(state.params, state.batch_size_stats,
                                        vbX, vbL, vbY,
                                        locs_train_jax, # ALL training locations
                                        ols_coeffs, SWNN_model.apply,
                                        dist_mean_for_sw, dist_std_for_sw) # Pass dist stats
        epoch_val_loss += val_loss_item.item()
    avg_epoch_val_loss = epoch_val_loss / num_val_batch_size

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_epoch_train_loss:.6f} - Val Loss: {avg_epoch_val_loss:.6f}")

    if avg_epoch_val_loss < best_val_loss:
        best_val_loss = avg_epoch_val_loss; best_state_params = state.params
        best_state_batch_size_stats = state.batch_size_stats; patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}."); break
    
# --- Prepare full dataset for prediction (including unseen test and val data) ---
X_all_scaled = scaler_X.transform(X)  # Scale entire feature set
X_all_jax = jnp.array(X_all_scaled)
locs_all_jax = jnp.array(locs)
y_all = jnp.array(y)

# Calculate distances to training points
dists_all_to_train = calculate_distances(locs_all_jax, locs_train_jax)

# --- Plot predictions over the original data ---
# Unpack predictions properly
mu_all, sigma_all, weights_all = gnnwr_predict(
    best_state_params,
    best_state_batch_size_stats,
    dists_all_to_train,
    SWNN_model.apply,
    X_all_jax,
    ols_coeffs,
    is_training=False,
    sw_dist_mean=dist_mean_for_sw,
    sw_dist_std=dist_std_for_sw,
    dropout_key_predict=None
)

# Convert to NumPy for plotting
mu_np = np.array(mu_all)
sigma_np = np.array(sigma_all)
X_np = np.array(X[:, 0])
y_np = np.array(y)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_np, y_np, label='Ground Truth', alpha=0.5, edgecolors='w', linewidth=0.5)

# Plot mean prediction
plt.plot(X_np, mu_np, label='Predicted μ', color='red', linewidth=2)

# Uncertainty bands: μ ± 2σ
plt.fill_between(X_np,
                 mu_np - 2 * sigma_np,
                 mu_np + 2 * sigma_np,
                 color='red',
                 alpha=0.2,
                 label='Uncertainty (±2σ)')

plt.title('SWNN Predictions with Uncertainty Bands')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# Choose 20 points evenly spaced across the data to visualize their local linear models
n_local_lines = 100
indices_to_plot = np.linspace(0, X_np.shape[0] - 1, n_local_lines, dtype=int)

plt.figure(figsize=(10, 6))
plt.scatter(X_np, y_np, label='Ground Truth', alpha=0.5, edgecolors='w', linewidth=0.5)
plt.plot(X_np, mu_np, label='Predicted μ', color='red', linewidth=2)
plt.fill_between(X_np,
                 mu_np - 2 * sigma_np,
                 mu_np + 2 * sigma_np,
                 color='red',
                 alpha=0.2,
                 label='Uncertainty (±2σ)')

# Plot local linear models
for idx in indices_to_plot:
    x_center = X_np[idx]
    beta0, beta1 = weights_all[idx, 0], weights_all[idx, 1]

    x_range = np.linspace(x_center - 0.5, x_center + 0.5, 50)
    
    # ⚠️ Apply same scaling as during training
    x_range_scaled = (x_range - scaler_X.mean_[0]) / scaler_X.scale_[0]
    
    y_line = beta0 + beta1 * x_range_scaled
    plt.plot(x_range, y_line, color='blue', alpha=0.7, linestyle='--', linewidth=1)

plt.title('SWNN Predictions with Local Linear Models')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# --- Extract and print coefficients for a single mid-point location ---
mid_index = 500 # Index in the middle of the dataset

# Extract learned spatial weights at that location
weights_mid = weights_all[mid_index]  # shape (4,)

# Separate weights for mean and log-variance
weights_mu_mid = weights_mid[:2]
weights_sigma_mid = weights_mid[2:]

# Separate OLS coefficients
ols_mu = ols_coeffs[:2]
ols_sigma = ols_coeffs[2:]

# Compute local coefficients
local_mu_coeffs = weights_mu_mid * ols_mu
local_sigma_coeffs = weights_sigma_mid * ols_sigma

# --- Print Results ---
print("\n--- Learned Local Coefficients at Middle Location ---")
print(f"Index: {mid_index}, X value: {X[mid_index, 0]:.4f}")
print(f"μ(x) coefficients:     Intercept = {local_mu_coeffs[0]:.4f},  Slope = {local_mu_coeffs[1]:.4f}")
print(f"log σ²(x) coefficients: Intercept = {local_sigma_coeffs[0]:.4f},  Slope = {local_sigma_coeffs[1]:.4f}")

# --- Plot local linear prediction and its uncertainty at midpoint ---
x_range = jnp.linspace(X[mid_index, 0] - 1, X[mid_index, 0] + 1, 100).reshape(-1, 1)
x_range_scaled = (x_range - scaler_X.mean_[0]) / scaler_X.scale_[0]
x_range_intercept = jnp.concatenate([jnp.ones_like(x_range_scaled), x_range_scaled], axis=1)

# Compute predicted mean and std using local coefficients
mu_local = jnp.sum(x_range_intercept * local_mu_coeffs, axis=1)
log_sigma2_local = jnp.sum(x_range_intercept * local_sigma_coeffs, axis=1)
sigma_local = jnp.exp(0.5 * log_sigma2_local)

# Convert to numpy
x_range_np = np.array(x_range).flatten()
mu_np = np.array(mu_local)
sigma_np = np.array(sigma_local)

# Plot the original data
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, label='Data', alpha=0.5, edgecolors='w', linewidth=0.5)

# Plot the local regression line
plt.plot(x_range_np, mu_np, linestyle='--', color='green', linewidth=2, label='Local Mean (μ)')

# Confidence band: μ ± 2σ
plt.fill_between(x_range_np, mu_np - 2 * sigma_np, mu_np + 2 * sigma_np,
                 color='green', alpha=0.2, label='±2σ Confidence Interval')

plt.axvline(X[mid_index, 0], color='gray', linestyle=':', label='Midpoint Location')
plt.title('Local Linear Prediction and Uncertainty at Midpoint')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------
# 1. WeightNet Definition
# ------------------------------------------------------
# --- New Flax Network: WeightNet (outputs 1 weight per point) ---
class WeightNet(nn.Module):
    hidden_dims: tuple = (32, 16)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool):
        kernel_init_fn = nn.initializers.he_normal()
        for i, h_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=h_dim, kernel_init=kernel_init_fn)(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=1, kernel_init=kernel_init_fn)(x)
        #x = nn.softplus(x)  # Ensure output in (0, 1)
        return jnp.squeeze(x, axis=-1)


# --- Weighted NLL Loss (using fixed mu_pred and sigma_pred) ---
def weighted_nll_loss(mu_pred, sigma_pred, y_true, weights):
    nll = (y_true - 1)**2 / (2 * 0.8**2) + jnp.log(0.8) + 0.5 * jnp.log(2 * jnp.pi)
    normalized_weights = weights / (jnp.sum(weights) + 1e-8)
    return jnp.sum(normalized_weights * nll)


# --- Training State for WeightNet ---
class WeightNetTrainState(train_state.TrainState):
    pass


# --- Training Step ---
@partial(jax.jit, static_argnames=('apply_fn',))
def train_step_weightnet(state, dists_input, mu_pred, sigma_pred, y_true, apply_fn, dropout_key):
    def loss_fn(params):
        weights = apply_fn({'params': params}, dists_input, train=True, rngs={'dropout': dropout_key})
        weights = jnp.maximum(weights, 1e-6)
        loss = weighted_nll_loss(mu_pred, sigma_pred, y_true, weights)
        return loss

    loss_val, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss_val

# --- Training Loop for WeightNet ---
def train_weightnet(weightnet_model, dists_train, mu_pred_train, sigma_pred_train, y_train,
                    learning_rate=0.0001, batch_size=128, epochs=100, key=jax.random.PRNGKey(0)):

    # Initialize model
    key_init, key_dropout = jax.random.split(key)
    dummy_input = dists_train[:1]
    params = weightnet_model.init(key_init, dummy_input, train=True)['params']

    state = WeightNetTrainState.create(
        apply_fn=weightnet_model.apply,
        params=params,
        tx=optax.adam(learning_rate=learning_rate)
    )

    for epoch in range(epochs):
        # Shuffle training data
        key, key_dropout = jax.random.split(key)
        num_samples = y_train.shape[0]
        perm = jax.random.permutation(key, num_samples)

        dists_train_shuffled = dists_train[perm]
        mu_pred_train_shuffled = mu_pred_train[perm]
        sigma_pred_train_shuffled = sigma_pred_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_loss = 0.0
        num_batches = int(np.ceil(num_samples / batch_size))

        for batch in range(num_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, num_samples)

            batch_dists = dists_train_shuffled[start:end]
            batch_mu = mu_pred_train_shuffled[start:end]
            batch_sigma = sigma_pred_train_shuffled[start:end]
            batch_y = y_train_shuffled[start:end]

            step_key = jax.random.fold_in(key_dropout, batch)
            state, loss = train_step_weightnet(state, batch_dists, batch_mu, batch_sigma, batch_y, weightnet_model.apply, step_key)
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.6f}")

    return state.params


# --- Initialize and Train the WeightNet ---
# Slice predicted values to match your splits
# Extract μ and σ at that single point
X_intercept_train = jnp.concatenate([jnp.ones_like(X_train_jax), X_train_jax], axis=1)
X_intercept_train = (X_intercept_train - scaler_X.mean_[0]) / scaler_X.scale_[0]
# 2. Use fixed local model (from mid_index) for prediction
mu_pred_train = jnp.sum(X_intercept_train * local_mu_coeffs, axis=1)

log_sigma2_pred_train = jnp.sum(X_intercept_train * local_sigma_coeffs, axis=1)
sigma_pred_train = jnp.exp(0.5 * log_sigma2_pred_train)  # convert log(σ²) to σ

# Compute distances (from train/val to ALL training locations), and standardize
dists_train = calculate_distances(locs_train_jax, locs_train_jax)
dists_train_std = (dists_train - dist_mean_for_sw) / (dist_std_for_sw + DIST_EPSILON)

weightnet_model = WeightNet(hidden_dims=(32, 16), dropout_rate=0.1)

trained_weightnet_params = train_weightnet(
    weightnet_model=weightnet_model,
    dists_train=dists_train_std,
    mu_pred_train=mu_pred_train,
    sigma_pred_train=sigma_pred_train,
    y_train=y_train,
    learning_rate=0.0001,
    batch_size=50,
    epochs=1000,
    key=jax.random.PRNGKey(1234)
)

# --- Visualization of weights on scatter plot ---
def plot_weight_colored_points(X_input, y_input, weights, mu_line=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    weights_np = np.array(weights)
    weights_normalized = (weights_np - weights_np.min()) / (weights_np.max() - weights_np.min() + 1e-8)
    colors = cm.coolwarm(weights_normalized)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_input[:, 0], y_input, color=colors, edgecolors='k', linewidths=0.3, label='Weighted Points')


    if mu_line is not None:
        x_vals = X_input[:, 0]
        x_intercept = np.column_stack([np.ones_like(x_vals), x_vals])
        mu_vals = np.sum(x_intercept * mu_line, axis=1)
        plt.plot(x_vals, mu_vals, linestyle='--', color='green', linewidth=2, label='Local μ Line')

    plt.title('Points Colored by Learned Weights with Local μ Line')
    plt.xlabel('Feature X')
    plt.ylabel('Target y')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


learned_weights_train = weightnet_model.apply(
    {'params': trained_weightnet_params},
    dists_train_std,  # standardized training distances
    train=False
)

plot_weight_colored_points(X_train, y_train, learned_weights_train)