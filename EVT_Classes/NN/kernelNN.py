import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Dummy function: standard squared distance

def calculate_distances(x1, x2):
    return jnp.abs(x1[:, None] - x2[None, :])

# Global OLS coefficients (already trained)
# Assume X_train and y_train are available and scaled

def compute_global_ols(X, y, eps=1e-6):
    """
    Computes two sets of OLS coefficients:
      - beta_mu: for predicting the mean µ(x)
      - beta_sigma: for predicting the standard deviation σ(x) (from residuals)

    Args:
        X: Input features, shape (n_samples, n_features)
        y: Target values, shape (n_samples,)
        eps: Small constant to avoid log(0) or div-by-zero

    Returns:
        beta_mu: Coefficients for predicting mean, shape (n_features + 1,)
        beta_sigma: Coefficients for predicting std deviation, shape (n_features + 1,)
    """
    # Augment with bias term
    X_aug = jnp.concatenate([jnp.ones((X.shape[0], 1)), X], axis=1)

    # Fit for mean µ(x)
    beta_mu, _, _, _ = jnp.linalg.lstsq(X_aug, y, rcond=None)
    y_mu_pred = jnp.dot(X_aug, beta_mu)

    # Estimate residual-based std deviation σ(x)
    residuals = jnp.abs(y - y_mu_pred) + eps  # add eps to avoid zero
    beta_sigma, _, _, _ = jnp.linalg.lstsq(X_aug, residuals, rcond=None)

    return beta_mu, beta_sigma
# Neural kernel model: learns W[i, j] = how much test i uses train j

class NeuralKernel(nn.Module):
    hidden_dims: tuple = (128, 64)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, dists, train: bool):
        x = dists
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        logits = nn.Dense(dists.shape[1])(x)  # output shape (test, train)
        weights = nn.softmax(logits)
        return weights

# Prediction function: combines global OLS prediction with learned weights
def predict_with_kernel(params, apply_fn, X_train, beta_mu, beta_sigma, dists, train=False, rng=None):
    variables = {'params': params}
    
    # Compute attention weights using the learned neural kernel
    if train:
        assert rng is not None, "RNG key is required for training due to dropout."
        weights = apply_fn(variables, dists, train=train, rngs={'dropout': rng})
    else:
        weights = apply_fn(variables, dists, train=train)

    # Augment training features with bias
    X_train_aug = jnp.concatenate([jnp.ones((X_train.shape[0], 1)), X_train], axis=1)

    # Compute OLS-based µ(x) and σ(x) for each training point
    y_mu_train = jnp.dot(X_train_aug, beta_mu)
    y_sigma_train = jnp.dot(X_train_aug, beta_sigma)

    # Apply attention weights to reconstitute µ(x) and σ(x) for test points
    mu_pred = jnp.dot(weights, y_mu_train)
    sigma_pred = jnp.dot(weights, y_sigma_train)

    return mu_pred, sigma_pred, weights

# Define TrainState
class TrainState(train_state.TrainState):
    pass

# MSE Loss

def attention_entropy(weights):
    entropy = -jnp.sum(weights * jnp.log(weights + 1e-8), axis=1)
    return -jnp.mean(entropy)  # Maximize entropy

def gaussian_nll_loss(mu, sigma, y, weights):
    nll_loss = jnp.mean((y - mu)**2 / (2 * 2**2) + jnp.log(2) + 0.5 * jnp.log(2 * jnp.pi))
    loss = nll_loss + 60 * attention_entropy(weights)
    return nll_loss

# Generate dummy data
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (1000, 1))
y_clean = 2.0 + 3.0 * X[:, 0] + 4.0 * jnp.sin(3 * X[:, 0])
sigma_noise = 0.1
y = y_clean + sigma_noise * jax.random.normal(key, (1000,))

# Split
X_train, X_test, y_train, y_test, y_clean_train, y_clean_test = train_test_split(X, y, y_clean, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_jax = jnp.array(X_train_scaled)
X_test_jax = jnp.array(X_test_scaled)
y_train_jax = jnp.array(y_train)
y_test_jax = jnp.array(y_test)
y_clean_test_jax = jnp.array(y_clean_test)

# Compute global OLS
beta_mu, beta_sigma = compute_global_ols(X_train_jax, y_train_jax)

# Compute distances
locs_train = jnp.array(X_train[:, 0])
locs_test = jnp.array(X_test[:, 0])
dists_test_to_train = calculate_distances(locs_test, locs_train)

# Initialize model
model = NeuralKernel()
key, subkey = jax.random.split(key)
params = model.init(subkey, dists_test_to_train, train=True)
state = TrainState.create(apply_fn=model.apply, params=params['params'], tx=optax.adam(1e-3))

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    key, subkey = jax.random.split(key)
    def loss_fn(params):
        mu_pred, sigma_pred, weights = predict_with_kernel(params, model.apply, X_train_jax, beta_mu, beta_sigma, dists_test_to_train, train=True, rng=subkey)
        return gaussian_nll_loss(mu_pred, sigma_pred, y_test_jax,weights)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Final prediction
preds, sigma_pred, weights = predict_with_kernel(
    state.params,
    model.apply,
    X_train_jax,
    beta_mu,
    beta_sigma,
    dists_test_to_train
)
sorted_idx = jnp.argsort(X_test[:, 0])
X_sorted = X_test[sorted_idx, 0]
y_sorted = y_clean_test[sorted_idx] 

# Now it's safe to convert and print
import numpy as np
sigma_np = np.array(sigma_pred)
print(sigma_np[:10])  # Print first 10 values nicely

import matplotlib.pyplot as plt

# Show attention weights for the 10th test point
test_idx = 45  # index of test point to visualize
weights_for_test_point = weights[test_idx]  # shape (n_train,)

plt.figure(figsize=(8, 5))
plt.hist(X_train_scaled, bins=30, edgecolor='black', alpha=0.7)
plt.title("Histogram of X_train_scaled")
plt.xlabel("Standardized X values")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Plot real (true) function vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_clean_test_jax, preds, alpha=0.7)
plt.plot([y_clean_test_jax.min(), y_clean_test_jax.max()], [y_clean_test_jax.min(), y_clean_test_jax.max()], 'r--')
plt.xlabel('True noiseless y (data-generating function)')
plt.ylabel('Predicted y')
plt.title('Neural Kernel Prediction vs True Function')
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Convert to numpy arrays
x_vals = np.array(X_test[:, 0])
true_y = np.array(y_clean_test)
pred_y = np.array(preds)
weights_np = np.array(weights)

# Compute weight density (number of weights > threshold per test point)
threshold = 0.01
weight_density = np.sum(weights_np > threshold, axis=1)
normalized_density = weight_density / weight_density.max()

# Sort everything by x_vals for clean plotting
sorted_idx = np.argsort(x_vals)
x_sorted = x_vals[sorted_idx]
true_y_sorted = true_y[sorted_idx]
pred_y_sorted = pred_y[sorted_idx]
density_sorted = normalized_density[sorted_idx]

# Plot true vs predicted and overlay weight density
fig, ax1 = plt.subplots(figsize=(10, 6))

# Main scatter plot
ax1.scatter(x_sorted, true_y_sorted, label='True function', alpha=0.6)
ax1.scatter(x_sorted, pred_y_sorted, label='Predicted', alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('True vs Predicted Function over X')
ax1.grid(True)
ax1.legend(loc='upper left')

# Secondary axis: weight density
ax2 = ax1.twinx()
ax2.plot(x_sorted, density_sorted, 'k--', label='Weight density (scaled)', linewidth=2)
ax2.set_ylabel('Relative weight density')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()