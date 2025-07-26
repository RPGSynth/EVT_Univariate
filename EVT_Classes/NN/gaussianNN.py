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
from matplotlib.animation import FuncAnimation

# --- JAX Setup ---
key = jax.random.PRNGKey(0)
# Set JAX traceback filtering to 'off' to get more detailed error messages.
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
DIST_EPSILON = 1e-8 # A small constant to prevent division by zero or log(0).

# ==============================================================================
# SECTION 1: DATA PREPARATION & UTILITIES
# ==============================================================================

# --- K-Nearest Neighbors Hyperparameter ---
K_NEIGHBORS =  2500 #The number of nearest neighbors to consider for each point.

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
    dynamic_std = config['noise_y_std'] * pattern**1.5
    y_noise = jax.random.normal(key_y_noise, (n_samples,)) * dynamic_std
    y = y_deterministic + y_noise
    return (np.array(locs), np.array(X_features), np.array(y))

# --- Data Preparation ---
locs, X, y = get_simulated_data(n_samples=10000, noise_y_std=0.5, x_slope_coeff=0.9)
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
        standardized_dists = (knn_dists_batch - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)
        x_intercept = jnp.concatenate([jnp.ones((x_batch.shape[0], 1)), x_batch], axis=1)
        modulation_factors, new_model_state = adapter_net_apply_fn(
            model_vars, standardized_dists, train=True, mutable=['batch_stats'], rngs={'dropout': dropout_key}
        )
        mu_pred = jnp.sum(x_intercept * (modulation_factors[:, :2] * global_ols_coeffs[:2]), axis=1)
        log_sigma2_pred = jnp.sum(x_intercept * (modulation_factors[:, 2:] * global_ols_coeffs[2:]), axis=1)
        log_sigma2_pred = jnp.clip(log_sigma2_pred, a_min=-15.0, a_max=15.0)
        sigma_pred = jnp.exp(0.5 * log_sigma2_pred)
        loss = nll_loss(mu_pred, sigma_pred, y_batch)
        return loss, new_model_state['batch_stats']

    (loss_val, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads).replace(batch_stats=new_batch_stats), loss_val

adapter_net = CoefficientAdapterNet(num_outputs=len(global_ols_coeffs))
key, key_init, key_dropout = jax.random.split(key, 3)
dummy_input = jax.random.normal(key_init, (1, K_NEIGHBORS))
model_variables = adapter_net.init(key_init, dummy_input, train=False)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=0.0005)
)
teacher_state = GNNWRTrainState.create(
    apply_fn=adapter_net.apply, params=model_variables['params'],
    tx=optimizer, batch_stats=model_variables.get('batch_stats', {})
)

epochs, batch_size, patience, best_val_loss, patience_counter = 200, 64, 100, float('inf'), 0
for epoch in range(epochs):
    key, key_shuffle = jax.random.split(key)
    perm = jax.random.permutation(key_shuffle, X_train_jax.shape[0])
    X_train_s, y_train_s, knn_dists_s = X_train_jax[perm], y_train_jax[perm], precomputed_knn_dists[perm]
    
    epoch_train_loss = 0.0
    num_batches = int(np.ceil(X_train_s.shape[0] / batch_size))
    for i in range(num_batches):
        start_idx, end_idx = i * batch_size, (i + 1) * batch_size
        x_b, y_b, knn_dists_b = X_train_s[start_idx:end_idx], y_train_s[start_idx:end_idx], knn_dists_s[start_idx:end_idx]
        key, step_key = jax.random.split(key)
        
        teacher_state, loss_item = train_step(
            teacher_state, x_b, knn_dists_b, y_b, global_ols_coeffs,
            adapter_net.apply, step_key, swnn_input_dist_mean, swnn_input_dist_std
        )
        epoch_train_loss += loss_item.item()

        if jnp.isnan(loss_item) or jnp.isinf(loss_item):
            print(f"\n!!! Training stopped at epoch {epoch+1}, batch {i+1} due to NaN/inf loss. !!!")
            break
    if jnp.isnan(loss_item) or jnp.isinf(loss_item): break

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
# This section is commented out to speed up execution for focusing on the student model.
# You can uncomment it to verify the teacher model's performance.
print("\n--- Generating Diagnostic Plots for Trained Teacher ---")
plt.style.use('seaborn-v0_8-whitegrid')
X_all_scaled = scaler_X.transform(X)
X_all_jax, locs_all_jax = jnp.array(X_all_scaled), jnp.array(locs)
mu_all, sigma_all, _ = gnnwr_predict(
    best_teacher_state.params, best_teacher_state.batch_stats,
    query_locs=locs_all_jax, apply_fn=adapter_net.apply, x_batch=X_all_jax,
    global_ols_coeffs=global_ols_coeffs, is_training=False, reference_locs=locs_train_jax,
    swnn_input_dist_mean=swnn_input_dist_mean, swnn_input_dist_std=swnn_input_dist_std, k=K_NEIGHBORS
)
plt.figure(figsize=(12, 7))
sorted_indices_all = np.argsort(X[:, 0])
plt.scatter(X[:, 0], y, label='Data Points', alpha=0.3, s=20, color='gray')
plt.plot(X[sorted_indices_all, 0], mu_all[sorted_indices_all], label='Predicted Mean (μ)', color='firebrick', linewidth=2.5)
plt.fill_between(X[sorted_indices_all, 0], 
                 (mu_all - 2 * sigma_all)[sorted_indices_all], 
                 (mu_all + 2 * sigma_all)[sorted_indices_all],
                 color='firebrick', alpha=0.2, label='Uncertainty (±2σ)')
plt.title('Teacher Model: Predictions with Uncertainty', fontsize=16)
plt.xlabel('Feature X (Location)'), plt.ylabel('Target y'), plt.legend()
plt.tight_layout(), plt.show()

# ==============================================================================
# SECTION 4: GENERATE TARGET PARAMETERS FROM THE TRAINED TEACHER
# ==============================================================================
print("\n--- Generating Target Parameters (μ, σ) from Trained Teacher ---")
mu_teacher_train, sigma_teacher_train, _ = gnnwr_predict(
    best_teacher_state.params, best_teacher_state.batch_stats,
    query_locs=locs_train_jax, apply_fn=adapter_net.apply, x_batch=X_train_jax,
    global_ols_coeffs=global_ols_coeffs, is_training=False, reference_locs=locs_train_jax,
    swnn_input_dist_mean=swnn_input_dist_mean, swnn_input_dist_std=swnn_input_dist_std,
    k=K_NEIGHBORS
)
print(f"Generated teacher's μ and σ for {mu_teacher_train.shape[0]} training points.")

# ==============================================================================
# SECTION 5: IMPLEMENT AND TRAIN THE DYNAMIC WEIGHTING MODEL
# ==============================================================================
print("\n--- Defining and Training the Dynamic GWR-style Model ---")
print("This model learns a dynamic kernel f(distance)->weight.")
print("The loss for a query 'q' is: sum(w_qi * NLL_i), where 'i' are neighbors of 'q'.")

class DynamicWeightNet(nn.Module):
    """Takes a distance and learns a kernel function to output a weight."""
    hidden_dims: tuple = (32, 16)
    @nn.compact
    def __call__(self, x, train: bool):
        # Input x is a distance value, shaped (batch, 1)
        for h_dim in self.hidden_dims:
            x = nn.Dense(features=h_dim)(x)
            x = nn.relu(x)
        weight = nn.Dense(features=1,bias_init=nn.initializers.constant(5.0))(x)
        # Use softplus to ensure weights are positive
        return nn.sigmoid(weight)

@partial(jax.jit, static_argnames=['apply_fn'])
def train_step_dynamic_weighted_sum_nll(state, apply_fn, neighbor_info_batch):
    """
    Performs a training step for the dynamic weighting model using a fully vectorized approach.
    The loss for each query point's neighborhood is the weighted sum of the neighbors' NLLs.
    This version avoids vmap by reshaping the batch for a single large MLP forward pass.
    """
    # Unpack the pre-gathered neighbor data. Shapes are (batch_size, K_NEIGHBORS)
    y_n_batch, mu_teacher_n_batch, sigma_teacher_n_batch, dists_n_batch = neighbor_info_batch

    def loss_fn(params):
        # Get batch size (B) and neighbor count (K) from the distance matrix
        B, K = dists_n_batch.shape

        # 1. Reshape neighbor distances for a single, large MLP forward pass.
        #    Shape: (B, K) -> (B * K, 1)
        dists_reshaped = dists_n_batch.reshape(-1, 1)

        # 2. Perform one large forward pass to get all weights.
        #    Input: (B * K, 1) -> Output: (B * K, 1)
        weights_reshaped = apply_fn({'params': params}, dists_reshaped, train=True)

        # 3. Reshape weights back to the batch structure.
        #    Shape: (B * K, 1) -> (B, K)
        weights_batch = weights_reshaped.reshape(B, K)
        #max_weights = jnp.max(weights_batch, axis=1, keepdims=True)
        #weights_batch = weights_batch / (max_weights + DIST_EPSILON)
        mean_weight_per_neighborhood = jnp.mean(weights_batch, axis=1)
        density_penalty = jnp.mean(1.0 - mean_weight_per_neighborhood)
        #target_rbf_weights = jnp.exp(-(dists_n_batch**2) / (2 * 10**2))
        #rbf_penalty = jnp.mean((weights_batch[:, 0:] - target_rbf_weights[:, 0:])**2)
        # 4. Calculate the NLL for each neighbor across the entire batch at once.
        safe_sigma = sigma_teacher_n_batch + DIST_EPSILON
        nll_per_neighbor_batch = 0.5 * jnp.log(2 * jnp.pi * safe_sigma**2) + \
                                 ((y_n_batch - mu_teacher_n_batch)**2 / (2 * safe_sigma**2))

        # 5. The loss for each neighborhood is the weighted sum of the NLLs.
        #    This is an element-wise multiplication. Shape: (B, K)
        #mean_weight_per_neighborhood = jnp.mean(weights_batch, axis=1)
        #scale_penalty = jnp.mean((mean_weight_per_neighborhood - 1.0)**2)
        
        weighted_nll_batch = (weights_batch * nll_per_neighbor_batch) 

        sum_of_weights = jnp.sum(weights_batch, axis=1) + DIST_EPSILON
        #jax.debug.print("min sum_of_weights: {s}", s=jnp.mean(sum_of_weights))
        #jax.debug.print("Density Penalty: {d}", d=density_penalty)

        # 6. Sum the weighted NLLs for each neighborhood (sum over K axis) and normalize
        #    by the sum of weights for that neighborhood. Add epsilon for stability.
        #    Shape of numerator and denominator: (B,)
        neighborhood_loss = jnp.sum(weighted_nll_batch, axis=1) / \
                            sum_of_weights + 1E-6
        
        #normalized_weights = weights_batch / sum_of_weights[:, jnp.newaxis]

        # 7. The final loss is the mean of the losses for each neighborhood in the batch.
        return jnp.mean(neighborhood_loss) #+ density_penalty * 0.0001

    loss_val, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss_val

# --- Prepare Data for the Dynamic Model's Training Loop ---
# For each query point in the training set, we need to gather the data of its K neighbors.
print("Pre-gathering neighbor data for efficient training...")
# Use jax.vmap for efficient gathering. It maps a function over the leading axis of an array.
gather_knn_data = jax.vmap(lambda indices, data: data[indices], in_axes=(0, None))

y_neighbors_train = gather_knn_data(precomputed_knn_indices, y_train_jax)
mu_teacher_neighbors_train = gather_knn_data(precomputed_knn_indices, mu_teacher_train)
sigma_teacher_neighbors_train = gather_knn_data(precomputed_knn_indices, sigma_teacher_train)
# The distances are already in the correct (N, K) shape
dists_neighbors_train = precomputed_knn_dists
print("Neighbor data prepared.")

# --- Setup for the Dynamic Weighting Model ---
dynamic_net = DynamicWeightNet()
key, dyn_key_init = jax.random.split(key)
dummy_dyn_input = jnp.zeros((1, 1)) # Input is a single distance
dyn_params = dynamic_net.init(dyn_key_init, dummy_dyn_input, train=False)['params']

dyn_optimizer = optax.chain(
    #optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-4) # Use a smaller learning rate for this more complex model
)
dyn_state = train_state.TrainState.create(
    apply_fn=dynamic_net.apply,
    params=dyn_params,
    tx=dyn_optimizer
)

# --- Training Loop for the Dynamic Model ---
epochs, batch_size, patience, best_val_loss, patience_counter = 100, 64, 1000, float('inf'), 0
print(f"\nTraining dynamic model for {epochs} epochs...")

num_train_points = X_train_jax.shape[0]
for epoch in range(epochs):
    key, shuffle_key = jax.random.split(key)
    perm = jax.random.permutation(shuffle_key, num_train_points)
    
    # Shuffle all corresponding neighbor data arrays together
    shuffled_y_n = y_neighbors_train[perm]
    shuffled_mu_n = mu_teacher_neighbors_train[perm]
    shuffled_sigma_n = sigma_teacher_neighbors_train[perm]
    shuffled_dists_n = dists_neighbors_train[perm]

    epoch_train_loss = 0
    num_batches = int(np.ceil(num_train_points / batch_size))
    for i in range(num_batches):
        start_idx, end_idx = i * batch_size, (i + 1) * batch_size
        
        # Create the batch of neighbor information
        neighbor_info_b = (
            shuffled_y_n[start_idx:end_idx],
            shuffled_mu_n[start_idx:end_idx],
            shuffled_sigma_n[start_idx:end_idx],
            shuffled_dists_n[start_idx:end_idx]
        )

        if neighbor_info_b[0].shape[0] == 0: continue

        dyn_state, loss_item = train_step_dynamic_weighted_sum_nll(
            dyn_state,
            dynamic_net.apply,
            neighbor_info_b
        )
        epoch_train_loss += loss_item

        if jnp.isnan(loss_item) or jnp.isinf(loss_item):
            print(f"\n!!! Training stopped at epoch {epoch+1}, batch {i+1} due to NaN/inf loss. !!!")
            break
    if jnp.isnan(loss_item) or jnp.isinf(loss_item): break

    MIN_WEIGHT_DENSITY_THRESHOLD = 0.5
    avg_train_loss = epoch_train_loss / num_batches
    all_weights_flat = dynamic_net.apply(
        {'params': dyn_state.params},
        dists_neighbors_train.reshape(-1, 1), # Predict on all distances
        train=False
    )
    current_weight_density = jnp.mean(all_weights_flat)
                                      
    if (epoch + 1) % 1 == 0:
        print(f"Dynamic Model | Epoch {epoch+1:05d} | Weighted NLL Sum: {avg_train_loss:.8f} | Validation Density: {current_weight_density:.4f}")

    if current_weight_density < MIN_WEIGHT_DENSITY_THRESHOLD:
        print(f"\n!!! Early stopping at epoch {epoch+1} !!!")
        print(f"Weight density ({current_weight_density:.4f}) fell below threshold ({MIN_WEIGHT_DENSITY_THRESHOLD}).")
        best_dyn_state = dyn_state # Save the state that triggered the stop
        break

    if avg_train_loss < best_val_loss:
        best_val_loss, best_dyn_state, patience_counter = avg_train_loss, dyn_state, 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Training stopped early at epoch {epoch+1}.")
        break

# ==============================================================================
# SECTION 6: VISUALIZE THE DYNAMICALLY LEARNED WEIGHTING KERNEL
# ==============================================================================
print("\n--- Generating Animation of the Learned Dynamic Kernel ---")

# --- Setup the plot elements first ---
fig_anim, ax_anim = plt.subplots(figsize=(12, 7))

# We will animate a subset of points for speed
num_frames = 50 
# Sort points by their x-feature to get a smooth animation across the data space
sorted_indices_anim = np.argsort(X_train_jax.flatten())
sampler_indices = np.linspace(0, len(sorted_indices_anim) - 1, num_frames, dtype=int)
indices_to_animate = sorted_indices_anim[sampler_indices]

# Predict all weights once to get a consistent color scale for the animation
all_predicted_weights = dynamic_net.apply(
    {'params': best_dyn_state.params},
    dists_neighbors_train.reshape(-1, 1), # Flatten distances to predict all weights
    train=False
).reshape(dists_neighbors_train.shape) # Reshape back to (N, K)
vmax = jnp.percentile(all_predicted_weights, 98) # Use 98th percentile for better color contrast

# --- Create the artists ONCE before the animation starts ---

# 1. Main scatter plot of all data points. We'll update its colors later.
#    Initialize with zero weights for the color.
scat = ax_anim.scatter(
    X_train_jax[:, 0], y_train_jax, c=jnp.zeros(num_train_points), s=35,
    alpha=0.8, cmap='viridis', vmin=0, vmax=vmax
)

# 2. Colorbar, created once and attached to the scatter plot artist.
fig_anim.colorbar(scat, ax=ax_anim, label='Dynamic Weight (w_qi)')

# 3. Query point highlight (a red star), initialized off-screen.
query_highlight = ax_anim.scatter([], [], c='red', s=400, marker='*', 
                                  edgecolor='black', linewidth=1.5, zorder=5, label='Query Point')

# 4. Vertical line for the query point.
vline = ax_anim.axvline(x=X_train_jax[0, 0], color='red', linestyle='--', lw=1.5, alpha=0.8)

# 5. Set static labels and title
ax_anim.set_xlabel('Feature X (Scaled)')
ax_anim.set_ylabel('Target y')
ax_anim.legend()
ax_anim.grid(True, alpha=0.4)
title = ax_anim.set_title('Learned Dynamic Kernel', fontsize=16)


# --- Define the animation update function ---
def update_animation(frame_index):
    # This is the current query point for this frame of the animation
    query_idx = indices_to_animate[frame_index]
    
    # Get the specific weights generated for the neighbors of THIS query point
    dynamic_weights_for_query = all_predicted_weights[query_idx]
    neighbor_indices = precomputed_knn_indices[query_idx]
    
    # Create a full weight vector for plotting. Most points will have a weight of 0.
    weights_for_plotting = jnp.zeros(num_train_points)
    weights_for_plotting = weights_for_plotting.at[neighbor_indices].set(dynamic_weights_for_query)
    
    # --- Update the data of the existing artists ---
    
    # Update the color array of the main scatter plot
    scat.set_array(weights_for_plotting)
    # Update the size of the points based on weight
    scat.set_sizes(weights_for_plotting * 200 + 15)

    # Move the query highlight to the new position
    query_x, query_y = X_train_jax[query_idx, 0], y_train_jax[query_idx]
    query_highlight.set_offsets([query_x, query_y])
    
    # Move the vertical line
    vline.set_xdata([query_x])
    
    # Update the title
    title.set_text(f'Learned Dynamic Kernel (Query Point {frame_index+1}/{num_frames})')
    
    print(f"Processing animation frame {frame_index+1}/{num_frames}...")
    
    # Return the tuple of artists that were changed
    return scat, query_highlight, vline, title

# --- Create and save the animation ---
anim = FuncAnimation(fig_anim, update_animation, frames=len(indices_to_animate), 
                       interval=200, blit=False) # Set blit=False for simplicity with text updates
output_filename = r"c:\github\EVT_Univariate\EVT_Classes\NN\student_weight_animation_knn.gif"
anim.save(output_filename, writer='pillow', fps=5)
plt.close(fig_anim)

print(f"\n✅ Animation complete! Saved as '{output_filename}'")
