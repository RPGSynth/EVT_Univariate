# --- Goal: Neural kernel using global OLS coefficients ---
# Predict each test point by learning weights over neighbors
# and combining them with global OLS-based predictions for both mean and variance.

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
def compute_global_ols(X, y):
    X_aug = jnp.concatenate([jnp.ones((X.shape[0], 1)), X], axis=1)
    beta, _, _, _ = jnp.linalg.lstsq(X_aug, y, rcond=None)
    return beta

# Neural kernel model: learns W_mu[i, j] and W_sigma[i, j]
class NeuralKernel(nn.Module):
    hidden_dims: tuple = (128, 64)
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, dists, train: bool):
        x = dists
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.PReLU()(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        logits_mu = nn.Dense(dists.shape[1])(x)
        logits_sigma = nn.Dense(dists.shape[1])(x)
        weights_mu = nn.softmax(logits_mu)
        weights_sigma = nn.softmax(logits_sigma)
        return weights_mu, weights_sigma

# Prediction function

def predict_with_kernel(params, apply_fn, X_test, X_train, y_train, ols_beta_mu, ols_beta_sigma, dists, train=False, rng=None):
    variables = {'params': params}
    if train:
        assert rng is not None, "RNG key is required for training due to dropout."
        weights_mu, weights_sigma = apply_fn(variables, dists, train=train, rngs={'dropout': rng})
    else:
        weights_mu, weights_sigma = apply_fn(variables, dists, train=train)

    X_train_aug = jnp.concatenate([jnp.ones((X_train.shape[0], 1)), X_train], axis=1)
    y_mu = jnp.dot(X_train_aug, ols_beta_mu)
    y_sigma = jnp.dot(X_train_aug, ols_beta_sigma)
    mu_pred = jnp.dot(weights_mu, y_mu)
    sigma_pred = jnp.clip(jnp.dot(weights_sigma, y_sigma), a_min=1e-3)
    return mu_pred, sigma_pred

# Define TrainState
class TrainState(train_state.TrainState):
    pass

# Gaussian NLL Loss

def gaussian_nll_loss(mu_pred, sigma_pred, targets):
    return jnp.sum((targets - mu_pred)**2 / (2 * sigma_pred**2) + jnp.log(sigma_pred) + 0.5 * jnp.log(2 * jnp.pi))

# Generate dummy data
key = jax.random.PRNGKey(0)
X = jax.random.normal(key, (100, 1))
y_clean = 2.0 + 3.0 * X[:, 0] + 4.0 * jnp.sin(3 * X[:, 0])
sigma_noise = 1 + 2 * X[:, 0]
y = y_clean + sigma_noise * jax.random.normal(key, (100,))

# Split\...
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
ols_beta_mu = compute_global_ols(X_train_jax, y_train_jax)
ols_beta_sigma = compute_global_ols(X_train_jax, jnp.abs(y_train_jax - jnp.dot(jnp.concatenate([jnp.ones((X_train_jax.shape[0], 1)), X_train_jax], axis=1), ols_beta_mu)))

# Distances
locs_train = jnp.array(X_train[:, 0])
locs_test = jnp.array(X_test[:, 0])
dists_test_to_train_all = calculate_distances(locs_test, locs_train)

# Initialize model
model = NeuralKernel()
key, subkey = jax.random.split(key)
params = model.init(subkey, dists_test_to_train_all, train=True)
state = TrainState.create(apply_fn=model.apply, params=params['params'], tx=optax.adam(1e-3))

# Mini-batch training
num_epochs = 1000
batch_size = 10
num_batches = X_test_jax.shape[0] // batch_size

for epoch in range(num_epochs):
    perm = jax.random.permutation(key, X_test_jax.shape[0])
    X_test_shuffled = X_test_jax[perm]
    y_test_shuffled = y_test_jax[perm]
    dists_shuffled = dists_test_to_train_all[perm]

    epoch_loss = 0
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        X_batch = X_test_shuffled[start:end]
        y_batch = y_test_shuffled[start:end]
        dists_batch = dists_shuffled[start:end]

        key, subkey = jax.random.split(key)
        def loss_fn(params):
            mu_pred, sigma_pred = predict_with_kernel(params, model.apply, X_batch, X_train_jax, y_train_jax, ols_beta_mu, ols_beta_sigma, dists_batch, train=True, rng=subkey)
            return gaussian_nll_loss(mu_pred, sigma_pred, y_batch)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        epoch_loss += loss

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Avg Loss: {epoch_loss / num_batches:.6f}")

# Final prediction
mu_preds, sigma_preds = predict_with_kernel(state.params, model.apply, X_test_jax, X_train_jax, y_train_jax, ols_beta_mu, ols_beta_sigma, dists_test_to_train_all)

# Sorted display
sorted_idx = jnp.argsort(X_test[:, 0])
X_sorted = X_test[sorted_idx, 0]
y_sorted = y_clean_test[sorted_idx]
pred_sorted = mu_preds[sorted_idx]
sigma_sorted = sigma_preds[sorted_idx]

plt.figure(figsize=(10, 5))
plt.plot(X_sorted, sigma_sorted, label='Predicted σ', color='blue')
plt.xlabel("X (sorted)")
plt.ylabel("Predicted Standard Deviation (σ)")
plt.title("Predicted σ over Sorted Test X")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], y_clean_test, label='True function', alpha=0.6)
plt.scatter(X_test[:, 0], mu_preds, label='Predicted mean', alpha=0.6)
#plt.fill_between(X_test[:, 0], mu_preds - 2*sigma_preds, mu_preds + 2*sigma_preds, color='gray', alpha=0.2, label='±2 std')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Predicted Mean and Uncertainty Band')
plt.legend()
plt.grid(True)
plt.show()

# --- Visualize weights for test sample 0 (first test point) ---

# Extract weights from the model for all test points
weights_mu_all, weights_sigma_all = model.apply({'params': state.params}, dists_test_to_train_all, train=False)

# Select first test point's weights
weights_mu_0 = weights_mu_all[0]
weights_sigma_0 = weights_sigma_all[0]

# Plot the weights
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(locs_train, weights_mu_0, marker='o')
plt.title("Weights for mu (test point 0)")
plt.xlabel("Training locations")
plt.ylabel("Weight")
plt.grid(True)
plt.show()