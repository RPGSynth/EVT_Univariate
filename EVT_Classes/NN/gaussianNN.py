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
    # Add intercept term
    X_train_intercept = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)

    # --- Fit OLS model for mean (mu) ---
    ols_mu = LinearRegression(fit_intercept=False)
    ols_mu.fit(X_train_intercept, y_train)
    mu_coeffs = ols_mu.coef_
    mu_preds = ols_mu.predict(X_train_intercept)

    # --- Compute residuals and fit OLS model for log(sigma^2) ---
    residuals = y_train - mu_preds
    log_residual_var = np.log(residuals**2 + 1e-6)  # Add epsilon for stability

    ols_sigma = LinearRegression(fit_intercept=False)
    ols_sigma.fit(X_train_intercept, log_residual_var)
    sigma_coeffs = ols_sigma.coef_

    # Return as JAX arrays
    return jnp.array(mu_coeffs), jnp.array(sigma_coeffs)


# --- Data Generation Function (True Weights based on Conceptual Reference Points) ---
def get_simulated_data(
    n_samples=300,
    key_data=jax.random.PRNGKey(123), # Use a different key or manage keys as needed
    x_minval=-2 * jnp.pi,      # Range for X to show trigonometric cycles
    x_maxval=2 * jnp.pi,
    curve_type='sin',         # 'sin' or 'cos'
    amplitude=1.5,
    frequency=1.0,            # Number of cycles within the x_minval to x_maxval range (approx)
    phase=0.0,                # Phase shift for the trigonometric function
    vertical_offset=0.5,      # This will be like beta0's constant part
    x_slope_coeff=1,        # This will be like beta1 (set to 0 for pure sin/cos)
    noise_y_std=0.1,          # Noise added to the final y
    noise_beta0_std=0.05,     # Noise added to the main curve component
    noise_beta1_std=0.05       # Noise for the x_slope_coeff
):
    key_beta0_noise, key_beta1_noise, key_y_noise = jax.random.split(key_data, 3)

    # 1. Generate X_orig_data as an ordered linspace (single feature)
    X = jnp.linspace(x_minval, x_maxval, n_samples).reshape(-1, 1)

    # 2. Define the main trigonometric component (acting as beta0(X))
    # locs_orig_data will be X itself for this scenario
    locs = X[:, 0]

    if curve_type == 'sin':
        main_curve = amplitude * jnp.sin(frequency * locs + phase)
    elif curve_type == 'cos':
        main_curve = amplitude * jnp.cos(frequency * locs + phase)
    else:
        raise ValueError("curve_type must be 'sin' or 'cos'")

    beta0_noise_values = jax.random.normal(key_beta0_noise, (n_samples,)) * noise_beta0_std
    beta0_values = vertical_offset + main_curve + beta0_noise_values

    # 3. Define the coefficient for X (beta1(X)), keep it simple (e.g., constant)
    beta1_noise_values = jax.random.normal(key_beta1_noise, (n_samples,)) * noise_beta1_std
    beta1_values = x_slope_coeff + beta1_noise_values
    print(beta1_values)
    # 4. Generate y_orig_data
    # y = beta0(X) + beta1(X) * X + noise_y
    y_deterministic = beta0_values + beta1_values * X[:, 0]
    y_noise = jax.random.normal(key_y_noise, (n_samples,)) * noise_y_std
    y = y_deterministic + y_noise

    # Store coefficients
    # true_local_coeffs will represent [beta0(X_i), beta1(X_i)]
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
def mse_loss(mu_pred, targets):
    return jnp.sum((targets - mu_pred)**2 / (2 * 0.1**2) + jnp.log(0.1) + 0.5 * jnp.log(2 * jnp.pi))

# --- GNNWR Prediction Logic (SWNN input: distances to training locs) ---
@partial(jax.jit, static_argnums=(3, 6)) # apply_fn (idx 3) and is_training (idx 6) are static
def gnnwr_predict(swnn_params, swnn_batch_size_stats,
                  dists_to_all_train_locs,  # Raw distances
                  apply_fn,
                  X_independent_vars_batch_size,
                  ols_coeffs,
                  is_training: bool,
                  sw_dist_mean: jnp.ndarray, # Mean for standardization
                  sw_dist_std: jnp.ndarray,  # Std for standardization
                  dropout_key_predict=None):
    X_batch_size_intercept = jnp.concatenate([jnp.ones((X_independent_vars_batch_size.shape[0], 1)), X_independent_vars_batch_size], axis=1)

    # Standardize SWNN input distances
    standardized_dists_input = (dists_to_all_train_locs - sw_dist_mean) / (sw_dist_std + DIST_EPSILON)

    swnn_vars = {'params': swnn_params}
    if swnn_batch_size_stats:
        swnn_vars['batch_size_stats'] = swnn_batch_size_stats

    dropout_rng_key_for_apply = {}
    if dropout_key_predict is not None:
        dropout_rng_key_for_apply = {'dropout': dropout_key_predict}

    if is_training:
        learned_spatial_weights, _ = apply_fn(
            swnn_vars, standardized_dists_input, train=True, mutable=['batch_size_stats'], rngs=dropout_rng_key_for_apply
        )
    else:
        learned_spatial_weights = apply_fn(
            swnn_vars, standardized_dists_input, train=False, rngs=dropout_rng_key_for_apply
        )

    local_coeffs_learned = learned_spatial_weights * ols_coeffs
    y_pred_learned = jnp.sum(X_batch_size_intercept * local_coeffs_learned, axis=1)
    return y_pred_learned, learned_spatial_weights


# --- Training and Evaluation Step Functions ---
@partial(jax.jit, static_argnames=('swnn_apply_fn',))
def train_step(state: GNNWRTrainState,
               batch_size_X_indep, batch_size_locs, batch_size_y,
               all_train_locs_for_dist_calc,
               ols_coeffs_glob,
               swnn_apply_fn, dropout_key_epoch_step,
               sw_dist_mean: jnp.ndarray, # Mean for standardization
               sw_dist_std: jnp.ndarray): # Std for standardization

    dists_batch_size_to_all_train = calculate_distances(batch_size_locs, all_train_locs_for_dist_calc)
    # Standardize SWNN input distances
    standardized_dists_input = (dists_batch_size_to_all_train - sw_dist_mean) / (sw_dist_std + DIST_EPSILON)

    def loss_fn_for_grad(swnn_params_inner):
        swnn_vars_for_apply = {'params': swnn_params_inner, 'batch_size_stats': state.batch_size_stats}
        learned_weights_batch_size, new_model_state = swnn_apply_fn(
            swnn_vars_for_apply, standardized_dists_input, # Use standardized distances
            train=True, mutable=['batch_size_stats'], rngs={'dropout': dropout_key_epoch_step}
        )
        X_batch_size_int = jnp.concatenate([jnp.ones((batch_size_X_indep.shape[0], 1)), batch_size_X_indep], axis=1)
        local_coeffs_batch_size = learned_weights_batch_size * ols_coeffs_glob
        y_pred_batch_size = jnp.sum(X_batch_size_int * local_coeffs_batch_size, axis=1)
        loss_val = mse_loss(y_pred_batch_size, batch_size_y)
        return loss_val, new_model_state['batch_size_stats']

    grad_fn = jax.value_and_grad(loss_fn_for_grad, has_aux=True)
    (loss_item, new_batch_size_stats_updated), grads_val = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads_val)
    new_state = new_state.replace(batch_size_stats=new_batch_size_stats_updated)
    return new_state, loss_item

@partial(jax.jit, static_argnames=('swnn_apply_fn_eval',))
def eval_step(state_params_eval, state_batch_size_stats_eval,
              batch_size_X_indep_eval, batch_size_locs_eval, batch_size_y_eval,
              all_train_locs_for_dist_calc_eval,
              ols_coeffs_glob_eval,
              swnn_apply_fn_eval,
              sw_dist_mean: jnp.ndarray, # Mean for standardization
              sw_dist_std: jnp.ndarray): # Std for standardization

    dists_batch_size_to_all_train_eval = calculate_distances(batch_size_locs_eval, all_train_locs_for_dist_calc_eval)
    y_pred_eval, learned_weights_eval = gnnwr_predict(
        state_params_eval, state_batch_size_stats_eval,
        dists_batch_size_to_all_train_eval, # Pass raw distances
        swnn_apply_fn_eval,
        batch_size_X_indep_eval,
        ols_coeffs_glob_eval,
        is_training=False,
        sw_dist_mean=sw_dist_mean, # Pass mean
        sw_dist_std=sw_dist_std   # Pass std
    )
    loss_eval = mse_loss(y_pred_eval, batch_size_y_eval)
    return loss_eval, y_pred_eval, learned_weights_eval

# --- Data Preparation ---
# ------ GENERATION ------------#
locs, X, y, local_coeffs = \
    get_simulated_data(n_samples=1000, key_data=jax.random.PRNGKey(789),x_slope_coeff=0.5)
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

ols_coeffs, sigma_coeffs = calculate_ols_coefficients(X_train_scaled, y_train)
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
    tx=optax.adam(learning_rate=0.00001),
    batch_size_stats=swnn_vars.get('batch_size_stats', {})
)

epochs = 5000; batch_size = 64; patience = 100
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

        val_loss_item, _, _ = eval_step(state.params, state.batch_size_stats,
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

# Predict using trained model (not in training mode)
ypred_all, weights_all = gnnwr_predict(
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

# --- Plot predictions over the original data ---
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, label='Ground Truth', alpha=0.6, edgecolors='w', linewidth=0.5)
plt.plot(X[:, 0], np.array(ypred_all), label='SWNN Predictions', color='red', linewidth=2)
plt.title('SWNN Predictions vs Ground Truth')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()