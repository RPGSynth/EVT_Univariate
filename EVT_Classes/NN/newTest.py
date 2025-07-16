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

def calculate_distances(point_locs, reference_locs):
    """Calculates the Euclidean distance matrix between two sets of locations."""
    return jnp.sqrt(jnp.sum((point_locs[:, jnp.newaxis] - reference_locs[jnp.newaxis, :])**2, axis=-1))

def calculate_ols_coefficients(x_train, y_train):
    """Calculates global OLS coefficients for mu and log(sigma^2)."""
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

def get_simulated_data(n_samples=200, key_data=jax.random.PRNGKey(123), **kwargs):
    """Generates complex simulated data with non-stationary properties."""
    config = {'x_minval': -2 * jnp.pi, 'x_maxval': 2 * jnp.pi, 'curve_type': 'sin', 'amplitude': 0, 'frequency': 1.0, 'phase': 0.0, 'vertical_offset': 0.5, 'x_slope_coeff': 1.0, 'noise_y_std': 0.3, 'noise_beta0_std': 0.5, 'noise_beta1_std': 0.05, 'noise_type': 'wavy'}
    config.update(kwargs)
    key_beta0, key_beta1, key_y_noise = jax.random.split(key_data, 3)
    X_features = jnp.linspace(config['x_minval'], config['x_maxval'], n_samples).reshape(-1, 1)
    locs = X_features
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
locs, X, y = get_simulated_data(n_samples=300, noise_y_std=0.9, x_slope_coeff=1.2)
locs_train_val, locs_test, X_train_val, X_test, y_train_val, y_test = train_test_split(locs, X, y, test_size=0.20, random_state=42)
locs_train, locs_val, X_train, X_val, y_train, y_val = train_test_split(locs_train_val, X_train_val, y_train_val, test_size=0.25, random_state=42)

# --- Sort the Training Data by Location for efficient lookup ---
print("Sorting training data by x-coordinate for intuitive indexing...")
sort_indices = np.argsort(locs_train.flatten())
locs_train, X_train, y_train = locs_train[sort_indices], X_train[sort_indices], y_train[sort_indices]

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_train_jax, y_train_jax = jnp.array(X_train_scaled), jnp.array(y_train)
X_val_jax, y_val_jax = jnp.array(X_val_scaled), jnp.array(y_val)
locs_train_jax, locs_val_jax = jnp.array(locs_train), jnp.array(locs_val)

train_dist_matrix = calculate_distances(locs_train_jax, locs_train_jax)
swnn_input_dist_mean = jnp.mean(train_dist_matrix)
swnn_input_dist_std = jnp.std(train_dist_matrix)
_, _, global_ols_coeffs = calculate_ols_coefficients(X_train_scaled, y_train)

# ==============================================================================
# SECTION 2: DEFINE AND TRAIN THE TEACHER MODEL (CoefficientAdapterNet)
# ==============================================================================
print("\n--- Training Teacher Model (CoefficientAdapterNet) ---")

class CoefficientAdapterNet(nn.Module):
    num_outputs: int
    hidden_dims: tuple = (64, 32)
    @nn.compact
    def __call__(self, x, train: bool):
        for i, h_dim in enumerate(self.hidden_dims):
            x = nn.Dense(features=h_dim, name=f'hidden_dense_{i}')(x)
            x = nn.PReLU()(x)
        return nn.Dense(features=self.num_outputs, name='output_factors')(x)

class GNNWRTrainState(train_state.TrainState):
    batch_stats: dict

def nll_loss(mu_pred, sigma_pred, targets):
    """Negative Log-Likelihood loss for a Gaussian distribution."""
    return jnp.mean(0.5 * jnp.log(2 * jnp.pi * sigma_pred**2) + (targets - mu_pred)**2 / (2 * sigma_pred**2))

@partial(jax.jit, static_argnums=(3, 6))
def gnnwr_predict(model_params, model_batch_stats, dists_to_locs, apply_fn,
                  x_batch, global_ols_coeffs, is_training: bool, swnn_input_dist_mean,
                  swnn_input_dist_std):
    """Performs prediction for the teacher model."""
    x_intercept = jnp.concatenate([jnp.ones((x_batch.shape[0], 1)), x_batch], axis=1)
    standardized_dists = (dists_to_locs - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)
    model_vars = {'params': model_params, 'batch_stats': model_batch_stats}
    modulation_factors = apply_fn(model_vars, standardized_dists, train=is_training)
    if isinstance(modulation_factors, tuple): modulation_factors, _ = modulation_factors
    
    local_mu_coeffs = modulation_factors[:, :2] * global_ols_coeffs[:2]
    local_log_sigma2_coeffs = modulation_factors[:, 2:] * global_ols_coeffs[2:]
    
    mu_pred = jnp.sum(x_intercept * local_mu_coeffs, axis=1)
    log_sigma2_pred = jnp.sum(x_intercept * local_log_sigma2_coeffs, axis=1)
    sigma_pred = jnp.exp(0.5 * log_sigma2_pred)
    
    local_coeffs = jnp.concatenate([local_mu_coeffs, local_log_sigma2_coeffs], axis=1)
    return mu_pred, sigma_pred, local_coeffs

@partial(jax.jit, static_argnames=('adapter_net_apply_fn',))
def train_step_teacher(state: GNNWRTrainState, x_batch, locs_batch, y_batch,
                 all_train_locs, global_ols_coeffs, adapter_net_apply_fn,
                 dropout_key, swnn_input_dist_mean, swnn_input_dist_std):
    """A single training step for the teacher model."""
    def loss_fn(model_params):
        dists = calculate_distances(locs_batch, all_train_locs)
        mu_pred, sigma_pred, _ = gnnwr_predict(
            model_params, state.batch_stats, dists, adapter_net_apply_fn,
            x_batch, global_ols_coeffs, is_training=True,
            swnn_input_dist_mean=swnn_input_dist_mean, swnn_input_dist_std=swnn_input_dist_std
        )
        loss = nll_loss(mu_pred, sigma_pred, y_batch)
        return loss
    
    loss_val, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss_val

# --- Teacher model training loop ---
adapter_net = CoefficientAdapterNet(num_outputs=len(global_ols_coeffs))
key, key_init, key_dropout = jax.random.split(key, 3)
model_variables = adapter_net.init(key_init, jax.random.normal(key_init, (1, locs_train_jax.shape[0])), train=False)
teacher_state = GNNWRTrainState.create(
    apply_fn=adapter_net.apply, params=model_variables['params'],
    tx=optax.adam(learning_rate=0.001), batch_stats=model_variables.get('batch_stats', {})
)

epochs, batch_size, patience, best_val_loss, patience_counter = 500, 64, 10, float('inf'), 0
for epoch in range(epochs):
    key, key_shuffle = jax.random.split(key)
    perm = jax.random.permutation(key_shuffle, X_train_jax.shape[0])
    X_train_s, y_train_s, locs_train_s = X_train_jax[perm], y_train_jax[perm], locs_train_jax[perm]
    
    epoch_train_loss = 0.0
    for i in range(int(np.ceil(X_train_s.shape[0] / batch_size))):
        start_idx, end_idx = i * batch_size, (i + 1) * batch_size
        x_b, l_b, y_b = X_train_s[start_idx:end_idx], locs_train_s[start_idx:end_idx], y_train_s[start_idx:end_idx]
        key, step_key = jax.random.split(key)
        teacher_state, loss_item = train_step_teacher(teacher_state, x_b, l_b, y_b, locs_train_jax, global_ols_coeffs, adapter_net.apply, step_key, swnn_input_dist_mean, swnn_input_dist_std)
        epoch_train_loss += loss_item.item()
        
    avg_epoch_train_loss = epoch_train_loss / (i+1)
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

# --- Generate predictions for the entire dataset for plotting ---
X_all_scaled = scaler_X.transform(X)
X_all_jax = jnp.array(X_all_scaled)
locs_all_jax = jnp.array(locs)
dists_all_to_train = calculate_distances(locs_all_jax, locs_train_jax)

mu_all, sigma_all, local_coeffs_all = gnnwr_predict(
    best_teacher_state.params, best_teacher_state.batch_stats, dists_all_to_train,
    adapter_net.apply, X_all_jax, global_ols_coeffs, is_training=False,
    swnn_input_dist_mean=swnn_input_dist_mean, swnn_input_dist_std=swnn_input_dist_std
)

# --- Plot 1: Predictions with Uncertainty Bands ---
plt.figure(figsize=(12, 7))
plt.scatter(X[:, 0], y, label='Data Points', alpha=0.3, s=20, color='gray')
plt.plot(X[:, 0], mu_all, label='Predicted Mean (μ)', color='firebrick', linewidth=2.5)
plt.fill_between(X[:, 0], mu_all - 2 * sigma_all, mu_all + 2 * sigma_all,
                 color='firebrick', alpha=0.2, label='Uncertainty (±2σ)')
plt.title('Teacher Model: Predictions with Uncertainty', fontsize=16)
plt.xlabel('Feature X (Location)'), plt.ylabel('Target y'), plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Visualization of Local Linear Models ---
plt.figure(figsize=(12, 7))
plt.scatter(X[:, 0], y, alpha=0.3, s=20, color='gray')
plt.plot(X[:, 0], mu_all, color='firebrick', linewidth=2.5, label='Predicted Mean (μ)')

n_local_lines = 80
indices_to_plot = np.linspace(0, len(X) - 1, n_local_lines, dtype=int)
for idx in indices_to_plot:
    x_center = X[idx, 0]
    beta0, beta1 = local_coeffs_all[idx, 0], local_coeffs_all[idx, 1]
    # Create a small range around the center point for the local line
    x_range = np.linspace(x_center - 0.5, x_center + 0.5, 2)
    # The coefficients were trained on scaled data, so we must scale our x_range for prediction
    y_line = beta0 + beta1 * scaler_X.transform(x_range.reshape(-1, 1)).flatten()
    plt.plot(x_range, y_line, color='navy', alpha=0.5, linestyle='-')

plt.title('Teacher Model: Learned Local Linear Models', fontsize=16)
plt.xlabel('Feature X (Location)'), plt.ylabel('Target y'), plt.legend()
plt.tight_layout()
plt.show()


# ==============================================================================
# SECTION 4: GENERATE TARGETS FROM THE TRAINED TEACHER
# ==============================================================================
print("\n--- Generating Target Values and Coefficients from Trained Teacher ---")
dists_train_to_train = calculate_distances(locs_train_jax, locs_train_jax)

# Use the prediction function to get mu, sigma, and coeffs for the entire training set
teacher_mu_train, teacher_sigma_train, teacher_coeffs_train = gnnwr_predict(
    best_teacher_state.params, best_teacher_state.batch_stats, dists_train_to_train,
    adapter_net.apply, X_train_jax, global_ols_coeffs, is_training=False,
    swnn_input_dist_mean=swnn_input_dist_mean, swnn_input_dist_std=swnn_input_dist_std
)
print(f"Generated {teacher_mu_train.shape[0]} sets of target mu/sigma values and coefficients.")


# ==============================================================================
# SECTION 5: DEFINE AND TRAIN THE STUDENT MODEL (LikelihoodWeightNet)
# ==============================================================================
print("\n--- Defining and Training Student Model (LikelihoodWeightNet) ---")

class LikelihoodWeightNet(nn.Module):
    """NN that learns to output spatial weights for any given location."""
    num_training_points: int
    hidden_dims: tuple = (256, 128, 64)
    @nn.compact
    def __call__(self, x, train: bool):
        for h_dim in self.hidden_dims:
            x = nn.Dense(features=h_dim)(x)
            x = nn.PReLU()(x)
        raw_weights = nn.Dense(features=self.num_training_points)(x)
        return nn.softmax(raw_weights)

def negative_weighted_log_likelihood(beta_params, x_data, y_data, weights):
    """The weighted log-likelihood function to be maximized (or minimized as negative)."""
    beta_mu, beta_sigma = beta_params[:x_data.shape[1]], beta_params[x_data.shape[1]:]
    mu_pred = x_data @ beta_mu
    log_sigma2_pred = x_data @ beta_sigma
    sigma_pred = jnp.exp(0.5 * log_sigma2_pred)
    
    # Clamp sigma_pred to avoid log(0) or division by zero
    sigma_pred_safe = jnp.maximum(sigma_pred, DIST_EPSILON)
    
    log_pdf = -0.5 * jnp.log(2 * jnp.pi * sigma_pred_safe**2) - (y_data - mu_pred)**2 / (2 * sigma_pred_safe**2)
    return -jnp.sum(weights * log_pdf)

def find_coeffs_by_optimization(weights, x_train_intercept, y_train, initial_beta_guess):
    """Inner optimization loop to find coefficients for a given set of weights."""
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(initial_beta_guess)
    loss_fn_inner = partial(negative_weighted_log_likelihood, x_data=x_train_intercept, y_data=y_train, weights=weights)
    grad_fn_inner = jax.value_and_grad(loss_fn_inner)
    
    def optimization_step(carry, _):
        beta_params, current_opt_state = carry
        loss, grads = grad_fn_inner(beta_params)
        updates, new_opt_state = optimizer.update(grads, current_opt_state)
        new_beta_params = optax.apply_updates(beta_params, updates)
        return (new_beta_params, new_opt_state), loss
        
    (final_beta, _), _ = jax.lax.scan(optimization_step, (initial_beta_guess, opt_state), jnp.arange(50))
    return final_beta

# Vectorize the optimization function to run on batches
batched_find_coeffs = jax.vmap(find_coeffs_by_optimization, in_axes=(0, None, None, None))

LAMBDA_ENTROPY = 0.75 # Regularization strength for weight entropy

@partial(jax.jit, static_argnames=('apply_fn',))
def train_step_student(state, x_batch, locs_batch, teacher_mu_batch, teacher_sigma_batch, apply_fn, 
                       x_train_intercept_full, y_train_full, all_train_locs, 
                       swnn_input_dist_mean, swnn_input_dist_std, initial_beta_guess):
    """A single training step for the student model, targeting mu and sigma."""
    
    # 1. Calculate distances for the student network input
    dists_batch = (calculate_distances(locs_batch, all_train_locs) - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)
    
    def loss_fn(params):
        # 2. Get spatial weights from the student network
        weights_batch = apply_fn({'params': params}, dists_batch, train=True)
        
        # 3. For each set of weights, find the optimal coefficients by maximizing the weighted log-likelihood
        beta_pred = batched_find_coeffs(weights_batch, x_train_intercept_full, y_train_full, initial_beta_guess)
        
        # 4. Use the found coefficients (beta_pred) to calculate mu and log_sigma2 for the current batch
        x_intercept_batch = jnp.concatenate([jnp.ones((x_batch.shape[0], 1)), x_batch], axis=1)
        mu_pred_batch = jnp.sum(x_intercept_batch * beta_pred[:, :2], axis=1)
        log_sigma2_pred_batch = jnp.sum(x_intercept_batch * beta_pred[:, 2:], axis=1)
        
        # 5. The "true" values are the mu and log(sigma^2) from the teacher model for this batch
        teacher_log_sigma2_batch = jnp.log(teacher_sigma_batch**2)
        
        # 6. Calculate the MSE loss between the student's predictions and the teacher's targets
        loss_mu = jnp.mean((mu_pred_batch - teacher_mu_batch)**2)
        loss_log_sigma2 = jnp.mean((log_sigma2_pred_batch - teacher_log_sigma2_batch)**2)
        mse_loss = loss_mu + loss_log_sigma2

        # 7. Add an entropy penalty to encourage smoother, less peaky weights
        entropy_batch = -jnp.sum(weights_batch * jnp.log(weights_batch + DIST_EPSILON), axis=-1)
        entropy_penalty = -jnp.mean(entropy_batch) # We want to MAXIMIZE entropy, so we MINIMIZE negative entropy
        
        # 8. Combine the losses
        total_loss = mse_loss + LAMBDA_ENTROPY * entropy_penalty
        
        return total_loss

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

# --- Student model training loop ---
lw_net = LikelihoodWeightNet(num_training_points=locs_train_jax.shape[0])
key, lw_key_init = jax.random.split(key)
lw_params = lw_net.init(lw_key_init, jax.random.normal(lw_key_init, (1, locs_train_jax.shape[0])), train=False)['params']
lw_state = train_state.TrainState.create(apply_fn=lw_net.apply, params=lw_params, tx=optax.adam(learning_rate=1e-4))

epochs, batch_size, patience, best_val_loss, patience_counter = 1000, 32, 100, float('inf'), 0
x_train_intercept_jax = jnp.concatenate([jnp.ones((X_train_jax.shape[0], 1)), X_train_jax], axis=1)
initial_beta_guess = global_ols_coeffs

for epoch in range(epochs):
    key, shuffle_key = jax.random.split(key)
    perm = jax.random.permutation(shuffle_key, locs_train_jax.shape[0])
    
    # Shuffle all corresponding training arrays using the same permutation
    shuffled_locs = locs_train_jax[perm]
    shuffled_X = X_train_jax[perm]
    shuffled_teacher_mu = teacher_mu_train[perm]
    shuffled_teacher_sigma = teacher_sigma_train[perm]
    
    epoch_train_loss = 0
    num_batches = int(np.ceil(locs_train_jax.shape[0] / batch_size))
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        
        # Create batches from the shuffled arrays
        locs_b = shuffled_locs[start_idx:end_idx]
        x_b = shuffled_X[start_idx:end_idx]
        teacher_mu_b = shuffled_teacher_mu[start_idx:end_idx]
        teacher_sigma_b = shuffled_teacher_sigma[start_idx:end_idx]
        
        # Call the new training step
        lw_state, loss_item = train_step_student(
            lw_state, x_b, locs_b, teacher_mu_b, teacher_sigma_b, 
            lw_net.apply, x_train_intercept_jax, y_train_jax, locs_train_jax, 
            swnn_input_dist_mean, swnn_input_dist_std, initial_beta_guess
        )
        epoch_train_loss += loss_item
        
    avg_train_loss = epoch_train_loss / num_batches
    if (epoch + 1) % 10 == 0:
      print(f"Student | Epoch {epoch+1:03d} | MSE+Entropy Loss: {avg_train_loss:.8f}")
      
    if avg_train_loss < best_val_loss:
        best_val_loss, best_lw_state, patience_counter = avg_train_loss, lw_state, 0
    else:
        patience_counter += 1
    if patience_counter >= patience:
        print(f"Student training stopped early at epoch {epoch+1}.")
        break

# ==============================================================================
# SECTION 6: DIAGNOSTIC PLOTS FOR THE STUDENT MODEL
# ==============================================================================
print("\n--- Generating Diagnostic Plots for Student Model ---")
# Calculate the final weights and resulting coefficients from the trained student model
dists_all = (calculate_distances(locs_train_jax, locs_train_jax) - swnn_input_dist_mean) / (swnn_input_dist_std + DIST_EPSILON)
final_weights = best_lw_state.apply_fn({'params': best_lw_state.params}, dists_all, train=False)
student_coeffs_train = batched_find_coeffs(final_weights, x_train_intercept_jax, y_train_jax, initial_beta_guess)

# --- Calculate student's mu and log_sigma2 predictions from its found coefficients ---
student_mu_train = jnp.sum(x_train_intercept_jax * student_coeffs_train[:, :2], axis=1)
student_log_sigma2_train = jnp.sum(x_train_intercept_jax * student_coeffs_train[:, 2:], axis=1)
teacher_log_sigma2_train = jnp.log(teacher_sigma_train**2)

# --- Plot 1: Correlation for mu and log(sigma^2) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Student Diagnostic 1: Teacher vs. Student Distribution Parameter Correlation', fontsize=16)

plot_data = [
    (teacher_mu_train, student_mu_train, 'μ (Mean)'),
    (teacher_log_sigma2_train, student_log_sigma2_train, 'log(σ²) (Log Variance)')
]

for i, (true_vals, pred_vals, name) in enumerate(plot_data):
    ax = axes[i]
    ax.scatter(np.array(true_vals), np.array(pred_vals), alpha=0.5, label='Predictions')
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', alpha=0.75, label='Perfect Match (y=x)')
    ax.set_xlabel(f'Teacher {name}')
    ax.set_ylabel(f'Student {name}')
    ax.set_title(f'Correlation for {name}')
    ax.grid(True)
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- Plot 2: Visualization of the learned spatial weight kernels ---
plt.figure(figsize=(14, 7))
plt.style.use('seaborn-v0_8-whitegrid')
plt.title('Student Diagnostic 2: Learned Spatial Weight Kernels', fontsize=16)
# Plot kernels for points at the start, middle, and end of the location range
indices_to_plot = [0, len(locs_train_jax) // 2, len(locs_train_jax) - 1]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
# The data is already sorted by location, so we can plot directly
for i, loc_idx in enumerate(indices_to_plot):
    weights_for_loc = final_weights[loc_idx, :]
    loc_val = locs_train_jax[loc_idx]
    plt.plot(locs_train_jax.flatten(), weights_for_loc, color=colors[i], alpha=0.3)
    plt.fill_between(locs_train_jax.flatten(), weights_for_loc, color=colors[i], alpha=0.2, label=f'Kernel for loc ≈ {loc_val.item():.2f}')
    plt.axvline(x=loc_val, color=colors[i], linestyle='--', lw=2)
plt.xlabel('Training Data Locations'), plt.ylabel('Weight Value'), plt.legend(), plt.grid(True, alpha=0.5), plt.tight_layout()
plt.show()

# --- Plot 3: Data-Weight Visualization ---
print("\n--- Generating Diagnostic Plot 3: Data-Weight Visualization ---")
# Choose a reference point to visualize (e.g., the 30th point in the sorted training data)
ref_idx = 75

# Get the unscaled x-location for plotting labels
ref_x_unscaled = X_train[ref_idx, 0]
ref_y = y_train_jax[ref_idx]

# Get the weights generated by the student model for this specific reference location
weights_for_ref_loc = final_weights[ref_idx, :]

plt.figure(figsize=(14, 8))

# Plot all training data points, colored and sized by their weight for the reference point
scatter = plt.scatter(
    X_train[:, 0], # Use original, unscaled X for plotting
    y_train_jax, 
    c=weights_for_ref_loc, 
    s=weights_for_ref_loc * 200 + 15,  # Scale size for visibility
    alpha=0.7,
    cmap='viridis',
    label='Training Data Points (colored by weight)'
)

# Highlight the reference point itself in red
plt.scatter(
    ref_x_unscaled, 
    ref_y, 
    c='red', 
    s=250,
    edgecolor='black',
    linewidth=1.5,
    zorder=5, 
    label=f'Reference Point (Index {ref_idx})'
)

# Add a vertical line at the reference location
plt.axvline(x=ref_x_unscaled, color='red', linestyle='--', lw=1.5, alpha=0.7)

# Formatting
cbar = plt.colorbar(scatter)
cbar.set_label('Weight Value', rotation=270, labelpad=20)
plt.title(f'Student Diagnostic 3: Weight Influence for Reference Point at x ≈ {ref_x_unscaled:.2f}', fontsize=16)
plt.xlabel('Feature X (Location)')
plt.ylabel('Target y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
