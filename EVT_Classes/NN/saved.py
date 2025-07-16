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