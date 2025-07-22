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
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
DIST_EPSILON = 1e-7

# ==============================================================================
# SECTION 1: DATA PREPARATION & UTILITIES
# ==============================================================================

# --- K-Nearest Neighbors Hyperparameter ---
K_NEIGHBORS = 100 # The number of nearest neighbors to consider for each point.
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
    """Calculates global OLS coefficients to serve as a baseline."""
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
    """Generates simulated data with non-stationary properties."""
    config = {'x_minval': -2 * jnp.pi, 'x_maxval': 2 * jnp.pi, 'curve_type': 'sin', 'amplitude': 1.5, 'frequency': 1.0, 'phase': 0.0, 'vertical_offset': 0.5, 'x_slope_coeff': 1.0, 'noise_y_std': 0.3, 'noise_beta0_std': 0.5, 'noise_beta1_std': 0.05, 'noise_type': 'wavy'}
    config.update(kwargs)
    key_beta0, key_beta1, key_y_noise = jax.random.split(key_data, 3)
    X_features = jnp.linspace(config['x_minval'], config['x_maxval'], n_samples).reshape(-1, 1)
    locs = X_features # Using features as locations
    main_curve = config['amplitude'] * jnp.sin(config['frequency'] * locs.flatten() + config['phase'])
    beta0_noise = jax.random.normal(key_beta0, (n_samples,)) * config['noise_beta0_std']
    beta0_values = config['vertical_offset'] + main_curve + beta0_noise
    beta1_noise = jax.random.normal(key_beta1, (n_samples,)) * config['noise_beta1_std']
    beta1_values = config['x_slope_coeff'] + beta1_noise
    y_deterministic = beta0_values + beta1_values * locs.flatten()
    wave = jnp.sin(2.5 * locs.flatten() + 0.3) + jnp.sin(6.3 * locs.flatten() + 1.8)
    pattern = jnp.abs(wave) + 0.2
    pattern /= jnp.mean(pattern)
    dynamic_std = config['noise_y_std'] * pattern
    y_noise = jax.random.normal(key_y_noise, (n_samples,)) * dynamic_std
    y = y_deterministic + y_noise
    return (np.array(locs), np.array(X_features), np.array(y))

# --- Data Preparation ---
locs, X, y = get_simulated_data(n_samples=500, noise_y_std=1, x_slope_coeff=0.9)
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
print("KNN distance matrix computed.")

# --- Calculate distance stats from the pre-computed matrix ---
swnn_input_dist_mean = jnp.mean(precomputed_knn_dists)
swnn_input_dist_std = jnp.std(precomputed_knn_dists)
print(f"Distance stats for k-NN: Mean={swnn_input_dist_mean:.4f}, Std={swnn_input_dist_std:.4f}")

_, _, global_ols_coeffs = calculate_ols_coefficients(X_train_scaled, y_train)

# ==============================================================================
# SECTION 2: DEFINE AND TRAIN THE TEACHER MODEL (CoefficientAdapterNet)
# ==============================================================================
print("\n--- Training Teacher Model (CoefficientAdapterNet) ---")

class CoefficientAdapterNet(nn.Module):
    num_outputs: int
    hidden_dims: tuple = (64,32)
    @nn.compact
    def __call__(self, x, train: bool):
        # The input `x` is now the vector of k-nearest neighbor distances
        for i, h_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=h_dim, name=f'hidden_dense_{i}')(x)
            x = nn.PReLU()(x)
        return nn.Dense(features=self.num_outputs, name='output_factors')(x)

class GNNWRTrainState(train_state.TrainState):
    batch_stats: dict

def nll_loss(mu_pred, sigma_pred, targets):
    """Negative Log-Likelihood loss for Gaussian distribution."""
    # Add a small epsilon to sigma_pred to prevent division by zero and log(0)
    safe_sigma = sigma_pred + DIST_EPSILON
    return jnp.mean(0.5 * jnp.log(2 * jnp.pi * safe_sigma**2) + ((targets - mu_pred)**2 / (2 * safe_sigma**2)))

# --- Prediction function still uses on-the-fly k-NN for flexibility ---
@partial(jax.jit, static_argnums=(3, 6, 10))
def gnnwr_predict(model_params, model_batch_stats, query_locs, apply_fn,
                  x_batch, global_ols_coeffs, is_training: bool,
                  reference_locs, swnn_input_dist_mean, swnn_input_dist_std, k: int):
    """Makes predictions using the k-NN based GNNWR model."""
    knn_dists, _ = find_knn(query_locs, reference_locs, k)
    standardized_dists = (knn_dists - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)
    x_intercept = jnp.concatenate([jnp.ones((x_batch.shape[0], 1)), x_batch], axis=1)
    model_vars = {'params': model_params, 'batch_stats': model_batch_stats}
    modulation_factors = apply_fn(model_vars, standardized_dists, train=is_training)
    if isinstance(modulation_factors, tuple): modulation_factors, _ = modulation_factors
    local_mu_coeffs = modulation_factors[:, :2] * global_ols_coeffs[:2]
    local_log_sigma2_coeffs = modulation_factors[:, 2:] * global_ols_coeffs[2:]
    mu_pred = jnp.sum(x_intercept * local_mu_coeffs, axis=1)
    log_sigma2_pred = jnp.sum(x_intercept * local_log_sigma2_coeffs, axis=1)

    # --- NEW: Clip predicted log variance to prevent explosion ---
    log_sigma2_pred = jnp.clip(log_sigma2_pred, a_min=-15.0, a_max=15.0)

    sigma_pred = jnp.exp(0.5 * log_sigma2_pred)
    local_coeffs = jnp.concatenate([local_mu_coeffs, local_log_sigma2_coeffs], axis=1)
    return mu_pred, sigma_pred, local_coeffs

# --- Training step now uses pre-computed k-NN distances ---
@partial(jax.jit, static_argnames=('adapter_net_apply_fn',))
def train_step(state: GNNWRTrainState, x_batch, knn_dists_batch, y_batch,
               global_ols_coeffs, adapter_net_apply_fn,
               dropout_key, swnn_input_dist_mean, swnn_input_dist_std):
    """Performs a single training step using pre-computed k-NN distances."""
    def loss_fn(model_params):
        model_vars = {'params': model_params, 'batch_stats': state.batch_stats}

        # The k-NN distances are already computed and passed in for the batch.
        standardized_dists = (knn_dists_batch - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)

        x_intercept = jnp.concatenate([jnp.ones((x_batch.shape[0], 1)), x_batch], axis=1)

        modulation_factors, new_model_state = adapter_net_apply_fn(
            model_vars, standardized_dists, train=True, mutable=['batch_stats'], rngs={'dropout': dropout_key}
        )

        mu_pred = jnp.sum(x_intercept * (modulation_factors[:, :2] * global_ols_coeffs[:2]), axis=1)
        log_sigma2_pred = jnp.sum(x_intercept * (modulation_factors[:, 2:] * global_ols_coeffs[2:]), axis=1)

        # --- NEW: Clip predicted log variance to prevent explosion ---
        log_sigma2_pred = jnp.clip(log_sigma2_pred, a_min=-15.0, a_max=15.0)

        sigma_pred = jnp.exp(0.5 * log_sigma2_pred)

        loss = nll_loss(mu_pred, sigma_pred, y_batch)
        return loss, new_model_state['batch_stats']

    (loss_val, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats), loss_val

adapter_net = CoefficientAdapterNet(num_outputs=len(global_ols_coeffs))
key, key_init, key_dropout = jax.random.split(key, 3)

# --- Initialize model with an input of size k ---
dummy_input = jax.random.normal(key_init, (1, K_NEIGHBORS))
model_variables = adapter_net.init(key_init, dummy_input, train=False)

# --- NEW: Add gradient clipping to the optimizer ---
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients to a max norm of 1.0
    optax.adam(learning_rate=0.0005)
)

teacher_state = GNNWRTrainState.create(
    apply_fn=adapter_net.apply, params=model_variables['params'],
    tx=optimizer, batch_stats=model_variables.get('batch_stats', {})
)

epochs, batch_size, patience, best_val_loss, patience_counter = 500, 350,500, float('inf'), 0
for epoch in range(epochs):
    key, key_shuffle = jax.random.split(key)
    perm = jax.random.permutation(key_shuffle, X_train_jax.shape[0])
    
    # --- Shuffle all training data arrays with the same permutation ---
    X_train_s = X_train_jax[perm]
    y_train_s = y_train_jax[perm]
    knn_dists_s = precomputed_knn_dists[perm] # Shuffle the precomputed distances

    epoch_train_loss = 0.0
    num_batches = int(np.ceil(X_train_s.shape[0] / batch_size))
    for i in range(num_batches):
        start_idx, end_idx = i * batch_size, (i + 1) * batch_size
        
        # --- Create batches from pre-shuffled data ---
        x_b = X_train_s[start_idx:end_idx]
        y_b = y_train_s[start_idx:end_idx]
        knn_dists_b = knn_dists_s[start_idx:end_idx] # Get the batch of distances
        
        key, step_key = jax.random.split(key)
        
        # --- Call the simplified training step ---
        teacher_state, loss_item = train_step(
            teacher_state, x_b, knn_dists_b, y_b, global_ols_coeffs,
            adapter_net.apply, step_key, swnn_input_dist_mean, swnn_input_dist_std
        )
        epoch_train_loss += loss_item.item()

        # --- NEW: Check for NaN/inf loss and stop training if it occurs ---
        if jnp.isnan(loss_item) or jnp.isinf(loss_item):
            print(f"\n!!! Training stopped at epoch {epoch+1}, batch {i+1} due to NaN/inf loss. !!!")
            break
    
    if jnp.isnan(loss_item) or jnp.isinf(loss_item):
        break # Break the outer epoch loop as well

    avg_epoch_train_loss = epoch_train_loss / num_batches
    if (epoch + 1) % 10 == 0:
      print(f"Teacher | Epoch {epoch+1:03d} | NLL Loss: {avg_epoch_train_loss:.6f}")

    if avg_epoch_train_loss < best_val_loss:
        best_val_loss, best_teacher_state, patience_counter = avg_epoch_train_loss, teacher_state, 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Teacher training stopped early at epoch {epoch+1}.")
        break

# ==============================================================================
# SECTION 3: DIAGNOSTIC PLOTS FOR THE TEACHER MODEL
# ==============================================================================
print("\n--- Generating Diagnostic Plots for Trained Teacher ---")
plt.style.use('seaborn-v0_8-whitegrid')
X_all_scaled = scaler_X.transform(X)
X_all_jax = jnp.array(X_all_scaled)
locs_all_jax = jnp.array(locs)

# --- Prediction uses the on-the-fly k-NN function as it may be for new data ---
mu_all, sigma_all, local_coeffs_all = gnnwr_predict(
    best_teacher_state.params, best_teacher_state.batch_stats,
    query_locs=locs_all_jax,
    apply_fn=adapter_net.apply,
    x_batch=X_all_jax,
    global_ols_coeffs=global_ols_coeffs,
    is_training=False,
    reference_locs=locs_train_jax,
    swnn_input_dist_mean=swnn_input_dist_mean,
    swnn_input_dist_std=swnn_input_dist_std,
    k=K_NEIGHBORS
)
plt.figure(figsize=(12, 7))
plt.scatter(X[:, 0], y, label='Data Points', alpha=0.3, s=20, color='gray')
plt.plot(X[:, 0], mu_all, label='Predicted Mean (μ)', color='firebrick', linewidth=2.5)
plt.fill_between(X[:, 0], mu_all - 2 * sigma_all, mu_all + 2 * sigma_all,
                 color='firebrick', alpha=0.2, label='Uncertainty (±2σ)')
plt.title('Teacher Model: Predictions with Uncertainty (k-NN based)', fontsize=16)
plt.xlabel('Feature X (Location)'), plt.ylabel('Target y'), plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 7))
plt.scatter(X[:, 0], y, alpha=0.3, s=20, color='gray')
plt.plot(X[:, 0], mu_all, color='firebrick', linewidth=2.5, label='Predicted Mean (μ)')

n_local_lines = 900
indices_to_plot = np.linspace(0, len(X) - 1, n_local_lines, dtype=int)
for idx in indices_to_plot:
    x_center = X[idx, 0]
    # The scaler expects a 2D array, so we reshape.
    x_center_scaled = scaler_X.transform(np.array([[x_center]]))
    x_intercept = np.array([1, x_center_scaled.item()])

    # Extract coefficients for this specific point
    local_mu_coeffs = local_coeffs_all[idx, :2]

    # Define a small range around the center point for plotting the line
    x_range = np.linspace(x_center - 0.5, x_center + 0.5, 2)
    x_range_scaled = scaler_X.transform(x_range.reshape(-1, 1))
    x_range_intercept = np.concatenate([np.ones((2, 1)), x_range_scaled], axis=1)

    y_line = x_range_intercept @ local_mu_coeffs
    plt.plot(x_range, y_line, color='navy', alpha=0.5, linestyle='-')

plt.title('Teacher Model: Learned Local Linear Models (k-NN based)', fontsize=16)
plt.xlabel('Feature X (Location)'), plt.ylabel('Target y'), plt.legend()
plt.tight_layout()
plt.show()

# ==============================================================================
# SECTION 4: GENERATE TARGET COEFFICIENTS FROM THE TRAINED TEACHER
# ==============================================================================
print("\n--- Generating Target Coefficients from Trained Teacher ---")
# --- Use the k-NN prediction function to get coefficients ---
_, _, teacher_coeffs_train = gnnwr_predict(
    best_teacher_state.params, best_teacher_state.batch_stats,
    query_locs=locs_train_jax,
    apply_fn=adapter_net.apply,
    x_batch=X_train_jax,
    global_ols_coeffs=global_ols_coeffs,
    is_training=False,
    reference_locs=locs_train_jax,
    swnn_input_dist_mean=swnn_input_dist_mean,
    swnn_input_dist_std=swnn_input_dist_std,
    k=K_NEIGHBORS
)
print(f"Generated {teacher_coeffs_train.shape[0]} sets of target coefficients using k-NN.")

# ==============================================================================
# SECTION 5: EFFICIENT STUDENT MODEL USING PRE-COMPUTED K-NN
# ==============================================================================
print("\n--- Defining and Training EFFICIENT k-NN Student Model ---")

ENTROPY_PENALTY = 0.1
class LikelihoodWeightNet(nn.Module):
    num_neighbors: int
    hidden_dims: tuple = (64,32)
    @nn.compact
    def __call__(self, x, train: bool):
        for h_dim in self.hidden_dims:
            x = nn.Dense(features=h_dim)(x)
            x = nn.PReLU()(x)
        raw_weights = nn.Dense(features=self.num_neighbors)(x)
        return nn.sigmoid(raw_weights)

def negative_weighted_log_likelihood(beta_params, x_data, y_data, weights):
    """Calculates the weighted NLL for a local regression."""
    num_mu_coeffs = x_data.shape[1]
    beta_mu = beta_params[:num_mu_coeffs]
    beta_sigma = beta_params[num_mu_coeffs:]
    mu_pred = x_data @ beta_mu
    log_sigma2_pred = x_data @ beta_sigma
    log_sigma2_pred = jnp.clip(log_sigma2_pred, a_min=-15.0, a_max=15.0)
    sigma_pred = jnp.exp(0.5 * log_sigma2_pred)
    safe_sigma = sigma_pred + DIST_EPSILON
    nll = 0.5 * jnp.log(2 * jnp.pi * safe_sigma**2) + ((y_data - mu_pred)**2 / (2 * safe_sigma**2))
    return jnp.sum(weights * nll) / jnp.sum(weights)

def find_coeffs_by_optimization_knn(weights, x_knn, y_knn, initial_beta_guess):
    """Performs weighted regression on the k-NN subset."""
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(initial_beta_guess)
    x_knn_intercept = jnp.concatenate([jnp.ones((x_knn.shape[0], 1)), x_knn], axis=1)
    loss_fn_inner = partial(negative_weighted_log_likelihood,
                            x_data=x_knn_intercept, y_data=y_knn, weights=weights)
    grad_fn_inner = jax.value_and_grad(loss_fn_inner)
    def optimization_step(carry, _):
        beta_params, current_opt_state = carry
        loss, grads = grad_fn_inner(beta_params)
        updates, new_opt_state = optimizer.update(grads, current_opt_state)
        new_beta_params = optax.apply_updates(beta_params, updates)
        return (new_beta_params, new_opt_state), loss
    (final_beta, _), _ = jax.lax.scan(optimization_step, (initial_beta_guess, opt_state), jnp.arange(20))
    return final_beta

def gather_knn_data(indices, all_x, all_y):
    return all_x[indices], all_y[indices]

# --- CHANGED: The student training step is now efficient and uses pre-computed data ---
@partial(jax.jit, static_argnames=('apply_fn',))
def train_step_student_knn(
    state,
    knn_dists_batch,      # Pre-computed distances for the batch
    knn_indices_batch,    # Pre-computed indices for the batch
    original_indices_batch, # The original indices of the points in this batch
    apply_fn,
    all_train_x,
    all_train_y,
    all_target_coeffs,
    swnn_input_dist_mean,
    swnn_input_dist_std,
    initial_beta_guess,
):
    # The expensive k-NN search is GONE. We just use the pre-computed inputs.
    standardized_dists = (knn_dists_batch - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)

    batched_gather = jax.vmap(gather_knn_data, in_axes=(0, None, None))
    x_knn_batch, y_knn_batch = batched_gather(knn_indices_batch, all_train_x, all_train_y)

    batched_find_coeffs_knn = jax.vmap(find_coeffs_by_optimization_knn, in_axes=(0, 0, 0, None))

    # Use the original indices to get the correct teacher coefficients and X values
    beta_true = all_target_coeffs[original_indices_batch]
    x_batch = all_train_x[original_indices_batch]

    def loss_fn(params):
        # Predict the weights for the k-nearest neighbors
        weights_batch = apply_fn({'params': params}, standardized_dists, train=True)
        
        # Calculate the predicted coefficients based on the learned weights
        beta_pred = batched_find_coeffs_knn(weights_batch, x_knn_batch, y_knn_batch, initial_beta_guess)

        # --- Primary Loss Calculation (Comparing student and teacher predictions) ---
        x_batch_intercept = jnp.concatenate([jnp.ones((x_batch.shape[0], 1)), x_batch], axis=1)
        
        mu_student = jnp.sum(x_batch_intercept * beta_pred[:, :2], axis=1)
        log_sigma2_student = jnp.sum(x_batch_intercept * beta_pred[:, 2:], axis=1)
        
        mu_teacher = jnp.sum(x_batch_intercept * beta_true[:, :2], axis=1)
        log_sigma2_teacher = jnp.sum(x_batch_intercept * beta_true[:, 2:], axis=1)

        primary_loss = jnp.mean((mu_student - mu_teacher)**2) + jnp.mean((log_sigma2_student - log_sigma2_teacher)**2)

        # --- Entropy Penalty Calculation (to spread out weights) ---
        # 1. Normalize weights for each point to sum to 1, forming a probability distribution.
        #    Add a small epsilon to the denominator to prevent division by zero.
        weights_sum = jnp.sum(weights_batch, axis=1, keepdims=True)
        prob_weights = weights_batch / (weights_sum + DIST_EPSILON)
        
        # 2. Calculate the entropy for each weight distribution.
        #    Add epsilon inside the log to prevent log(0).
        entropy = -jnp.sum(prob_weights * jnp.log(prob_weights + DIST_EPSILON), axis=1)
        
        # 3. We want to MAXIMIZE entropy, so we MINIMIZE its negative.
        #    The penalty term is added to the primary loss.
        entropy_loss = -jnp.mean(entropy)
        
        # 4. Combine the primary loss with the scaled entropy penalty.
        total_loss = primary_loss + ENTROPY_PENALTY * entropy_loss
        
        return total_loss

    # Standard gradient calculation and state update
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


# --- Setup for k-NN Student Model ---
lw_net = LikelihoodWeightNet(num_neighbors=K_NEIGHBORS)
key, lw_key_init = jax.random.split(key)
dummy_student_input = jax.random.normal(lw_key_init, (1, K_NEIGHBORS))
lw_params = lw_net.init(lw_key_init, dummy_student_input, train=False)['params']
student_optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-4)
)
lw_state = train_state.TrainState.create(apply_fn=lw_net.apply, params=lw_params, tx=student_optimizer)

# --- CHANGED: Efficient Training Loop for Student ---
epochs, batch_size, patience, best_val_loss, patience_counter = 5000, 350, 500, float('inf'), 0
initial_beta_guess = global_ols_coeffs

# Create an array of original indices to shuffle
original_indices = jnp.arange(locs_train_jax.shape[0])

for epoch in range(epochs):
    key, shuffle_key = jax.random.split(key)
    perm = jax.random.permutation(shuffle_key, locs_train_jax.shape[0])
    
    # Shuffle the pre-computed data and original indices
    shuffled_dists = precomputed_knn_dists[perm]
    shuffled_indices = precomputed_knn_indices[perm]
    shuffled_original_indices = original_indices[perm]

    epoch_train_loss = 0
    num_batches = int(np.ceil(locs_train_jax.shape[0] / batch_size))
    for i in range(num_batches):
        start_idx, end_idx = i * batch_size, (i + 1) * batch_size
        
        # Create batches of pre-computed data
        dists_b = shuffled_dists[start_idx:end_idx]
        indices_b = shuffled_indices[start_idx:end_idx]
        original_indices_b = shuffled_original_indices[start_idx:end_idx]

        if dists_b.shape[0] == 0: continue

        lw_state, loss_item = train_step_student_knn(
            lw_state, dists_b, indices_b, original_indices_b,
            lw_net.apply, X_train_jax, y_train_jax,
            teacher_coeffs_train,
            swnn_input_dist_mean, swnn_input_dist_std, initial_beta_guess
        )
        epoch_train_loss += loss_item

        if jnp.isnan(loss_item) or jnp.isinf(loss_item):
            print(f"\n!!! Student training stopped at epoch {epoch+1}, batch {i+1} due to NaN/inf loss. !!!")
            break
    if jnp.isnan(loss_item) or jnp.isinf(loss_item):
        break

    avg_train_loss = epoch_train_loss / num_batches
    if (epoch + 1) % 10 == 0:
        print(f"Student | Epoch {epoch+1:03d} | MSE Loss: {avg_train_loss:.8f}")

    if avg_train_loss < best_val_loss:
        best_val_loss, best_lw_state, patience_counter = avg_train_loss, lw_state, 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Student training stopped early at epoch {epoch+1}.")
        break

# ==============================================================================
# SECTION 6: DIAGNOSTIC PLOTS FOR THE k-NN STUDENT MODEL
# ==============================================================================
print("\n--- Generating k-NN Diagnostic Plots for Student Model ---")

# --- First, get the final weights and calculate the resulting student coefficients ---
# 1. Find the k-NN for all training points
final_knn_dists, final_knn_indices = find_knn(locs_train_jax, locs_train_jax, K_NEIGHBORS)
final_std_dists = (final_knn_dists - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)

# 2. Predict the k weights for each point
final_weights = best_lw_state.apply_fn({'params': best_lw_state.params}, final_std_dists, train=False)

# 3. Gather the k-NN data subsets for all training points
batched_gather = jax.vmap(gather_knn_data, in_axes=(0, None, None))
final_x_knn, final_y_knn = batched_gather(final_knn_indices, X_train_jax, y_train_jax)

# 4. Calculate the final student coefficients
batched_find_coeffs_knn = jax.vmap(find_coeffs_by_optimization_knn, in_axes=(0, 0, 0, None))
student_coeffs_train = batched_find_coeffs_knn(final_weights, final_x_knn, final_y_knn, initial_beta_guess)


# --- Plot 1: Correlation of Predictions ---
# We use the full training set's X values to calculate the final mu and sigma
x_train_intercept_jax = jnp.concatenate([jnp.ones((X_train_jax.shape[0], 1)), X_train_jax], axis=1)

mu_student_train = jnp.sum(x_train_intercept_jax * student_coeffs_train[:, :2], axis=1)
log_sigma2_student_train = jnp.sum(x_train_intercept_jax * student_coeffs_train[:, 2:], axis=1)
mu_teacher_train = jnp.sum(x_train_intercept_jax * teacher_coeffs_train[:, :2], axis=1)
log_sigma2_teacher_train = jnp.sum(x_train_intercept_jax * teacher_coeffs_train[:, 2:], axis=1)


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Student vs. Teacher Final Prediction Correlation (k-NN)', fontsize=16)
# Plot for mu (μ)
axes[0].scatter(mu_teacher_train, mu_student_train, alpha=0.6)
axes[0].plot([mu_teacher_train.min(), mu_teacher_train.max()], [mu_teacher_train.min(), mu_teacher_train.max()], 'r--', label='y=x')
axes[0].set_xlabel('Teacher Predicted μ'), axes[0].set_ylabel('Student Predicted μ')
axes[0].set_title('Correlation for Mean (μ)'), axes[0].grid(True), axes[0].legend()
# Plot for log sigma squared (log σ²)
axes[1].scatter(log_sigma2_teacher_train, log_sigma2_student_train, alpha=0.6)
axes[1].plot([log_sigma2_teacher_train.min(), log_sigma2_teacher_train.max()], [log_sigma2_teacher_train.min(), log_sigma2_teacher_train.max()], 'r--', label='y=x')
axes[1].set_xlabel('Teacher Predicted log(σ²)'), axes[1].set_ylabel('Student Predicted log(σ²)')
axes[1].set_title('Correlation for Log Variance (log σ²)'), axes[1].grid(True), axes[1].legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

fig, ax = plt.subplots(figsize=(14, 8))
plt.style.use('seaborn-v0_8-whitegrid')
from matplotlib.animation import FuncAnimation

# --- Plot 3: Animated GIF of k-NN Weight Visualization ---
print("\n--- Generating Diagnostic Plot 3: Animated GIF of k-NN Weight Visualization ---")

# We need to un-scale X for this plot to be intuitive
# This was done in the teacher section, but we ensure it's available here.
scaler_X_anim = StandardScaler()
X_train_scaled_for_plot = scaler_X_anim.fit_transform(X_train)
X_train_unscaled = scaler_X_anim.inverse_transform(X_train_scaled_for_plot)

# 1. Setup the figure and select the frames to animate
fig_anim, ax_anim = plt.subplots(figsize=(12, 7))
plt.style.use('seaborn-v0_8-whitegrid')

X_train_unscaled = scaler_X.inverse_transform(X_train_jax)
sorted_indices = np.argsort(X_train_unscaled.flatten())

# Animate through 30 evenly spaced points from the *sorted* data
num_frames = 30
# Create indices to sample from the `sorted_indices` array
sampler_indices = np.linspace(0, len(sorted_indices) - 1, num_frames, dtype=int)
# Get the final indices to animate, which now guarantee a left-to-right progression
indices_to_animate = sorted_indices[sampler_indices]


# 2. Define the update function that draws each frame
def update_knn(frame_index):
    """Clears the current figure and draws the plot for the next k-NN reference index."""
    ax_anim.clear()

    # Get the specific reference index for this frame
    ref_idx = indices_to_animate[frame_index]

    # --- CHANGED: Create a full weight vector for plotting ---
    # 1. Start with a vector of zeros for all N points.
    weights_for_plotting = jnp.zeros(len(locs_train_jax))
    # 2. Get the indices of the k neighbors for the reference point.
    neighbor_indices = final_knn_indices[ref_idx]
    # 3. Get the k weights for those neighbors.
    neighbor_weights = final_weights[ref_idx]
    # 4. Place the k weights into the full-size vector at the correct indices.
    weights_for_plotting = weights_for_plotting.at[neighbor_indices].set(neighbor_weights)


    ref_x_unscaled = X_train_unscaled[ref_idx, 0]
    ref_y = y_train_jax[ref_idx]

    # Plot all N data points. Non-neighbors will have a color/size of 0.
    ax_anim.scatter(
        X_train_unscaled[:, 0], y_train_jax,
        c=weights_for_plotting,
        s=weights_for_plotting * 500 + 10, # Increased size multiplier for visibility
        alpha=0.7, cmap='viridis', vmin=0, vmax=jnp.max(final_weights)
    )

    # Highlight the current reference point
    ax_anim.scatter(
        ref_x_unscaled, ref_y,
        c='red', s=300, edgecolor='black',
        linewidth=1.5, zorder=5
    )

    # Add a vertical line and set labels/titles for the current frame
    ax_anim.axvline(x=ref_x_unscaled, color='red', linestyle='--', lw=1.5, alpha=0.8)
    ax_anim.set_title(f'k-NN Weight Influence for Reference Point #{ref_idx} (k={K_NEIGHBORS})', fontsize=16)
    ax_anim.set_xlabel('Feature X (Location)')
    ax_anim.set_ylabel('Target y')
    ax_anim.grid(True, alpha=0.4)
    print(f"Processing animation frame {frame_index+1}/{num_frames}...")

# 3. Create the animation object
anim = FuncAnimation(fig_anim, update_knn, frames=len(indices_to_animate), interval=200)

# 4. Save the animation as a GIF
output_filename = r"c:\github\EVT_Univariate\EVT_Classes\NN\student_weight_animation_knn.gif"
anim.save(output_filename, writer='pillow', fps=5)
plt.close(fig_anim)

print(f"\n✅ Animation complete! Saved as '{output_filename}'")

