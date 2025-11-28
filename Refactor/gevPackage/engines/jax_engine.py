import jax
import jax.numpy as jnp
from functools import partial
from ..gev_link import GEVLinkage

linker = GEVLinkage()

def zp_to_mu(zp, sigma, xi, T):
    """
    Inverts the GEV formula to recover Mu from Zp (Return Level).
    mu = zp + (sigma / xi) * ( ( -log(1-1/T) )^(-xi) - 1 )
    """
    y = -jnp.log(1.0 - 1.0/T)
    
    # --- Gumbel Branch (xi -> 0) ---
    # zp = mu - sigma * log(y)  =>  mu = zp + sigma * log(y)
    mu_gumbel = zp + sigma * jnp.log(y)
    
    # --- GEV Branch ---
    xi_safe = jnp.where(jnp.abs(xi) < 1e-5, 1.0, xi)
    
    # Use jnp.exp(log) for stability instead of power
    term = jnp.exp(-xi_safe * jnp.log(y)) 
    
    # Inverted GEV formula
    mu_gev = zp + (sigma / xi_safe) * (1.0 - term)
    
    is_gumbel = jnp.abs(xi) < 1e-5
    return jnp.where(is_gumbel, mu_gumbel, mu_gev)

@partial(jax.jit, static_argnames=('dims', 'reparam_T'))
def predict_parameters(params, exog_loc, exog_scale, exog_shape, dims, reparam_T=None):
    """
    Reconstructs mu, sigma, xi. 
    If reparam_T is set, 'mu' is calculated from the predicted Zp.
    """
    beta_loc, beta_scale, beta_shape = linker.forward(params, dims)
    
    # 1. Calculate the raw first parameter (either Mu or Zp)
    # Note: beta_loc here might be beta_zp if reparam_T is set.
    lin_loc = jnp.einsum('nsk,k->ns', exog_loc, beta_loc)
    
    # 2. Calculate Scale and Shape
    lin_scale = jnp.einsum('nsk,k->ns', exog_scale, beta_scale)
    sigma = linker.transform_scale(lin_scale)
    xi = jnp.einsum('nsk,k->ns', exog_shape, beta_shape)
    
    # 3. Handle Reparameterization
    if reparam_T is not None:
        # lin_loc is actually Zp. Convert to Mu.
        mu = zp_to_mu(lin_loc, sigma, xi, reparam_T)
    else:
        # lin_loc is Mu.
        mu = lin_loc
        
    return mu, sigma, xi

@partial(jax.jit, static_argnames=('dims', 'reparam_T'))
def nloglike_sum(params, endog, exog_loc, exog_scale, exog_shape, weights, dims, reparam_T=None):
    mu, sigma, xi = predict_parameters(params, exog_loc, exog_scale, exog_shape, dims, reparam_T)
    
    # 1. Standardize
    z = (endog - mu) / sigma
    
    # 2. Gumbel Term (xi -> 0)
    # nll = log(sigma) + z + exp(-z)
    nll_gumbel = jnp.log(sigma) + z + jnp.exp(-z)
    
    # 3. GEV Term (xi != 0)
    # Safe xi: preserves sign, ensures abs(xi) >= 1e-7
    xi_sign = jnp.where(xi >= 0, 1.0, -1.0) 
    xi_safe = xi_sign * jnp.maximum(jnp.abs(xi), 1e-7)
    
    op_term = 1 + xi_safe * z
    
    # Domain Constraint: 1 + xi*z > 0
    is_valid_domain = op_term > 0
    
    # Mask invalid values to 1.0 before log/power to avoid NaNs in gradients
    op_term_safe = jnp.where(is_valid_domain, op_term, 1.0)
    
    # nll = log(sigma) + (1 + 1/xi) * log(term) + term^(-1/xi)
    log_op = jnp.log(op_term_safe)
    inv_xi = 1.0 / xi_safe
    t_gev = jnp.exp(-inv_xi * log_op) # Safer than power for gradients
    
    nll_gev = jnp.log(sigma) + (1.0 + inv_xi) * log_op + t_gev
    
    # Apply huge penalty for invalid domain
    nll_gev = jnp.where(is_valid_domain, nll_gev, 1e9)
    
    # 4. Switch
    # Use Gumbel if |xi| < 1e-5
    use_gumbel = jnp.abs(xi) < 1e-5
    nll_point = jnp.where(use_gumbel, nll_gumbel, nll_gev)
    
    return jnp.sum(nll_point * weights)

@partial(jax.jit, static_argnames=('dims', 'reparam_T'))
def compute_sandwich_matrices(params, endog, exog_loc, exog_scale, exog_shape, weights, dims,reparam_T=None):
    """
    Computes the matrices for the Godambe Covariance of the AVERAGE Likelihood.
    L_avg = (1/W) * Sum(w_i * l_i)
    """
    W_total = jnp.sum(weights)

    # 1. Define the Objective Function (Average NLL)
    def objective_avg(p):
        return nloglike_sum(p, endog, exog_loc, exog_scale, exog_shape, weights, dims, reparam_T) / W_total

    # 2. Hessian of Average NLL (The Bread)
    H = jax.hessian(objective_avg)(params)
    
    # 3. Gradients of individual terms (The Meat)
    # We need the gradient of (w_i * l_i / W_total) for each observation
    def row_term_func(params, y_row, exog_loc_row, exog_scale_row, exog_shape_row, w_row):
        # Calculate NLL for this row, weight it, and scale by 1/W_total
        nll_val = nloglike_sum(
            params,
            y_row[None, :], 
            exog_loc_row[None, ...],
            exog_scale_row[None, ...],
            exog_shape_row[None, ...],
            w_row[None, :], 
            dims,
            reparam_T
        )
        return nll_val / W_total

    # Vectorize gradient calculation over N_obs
    grads = jax.vmap(jax.grad(row_term_func), in_axes=(None, 0, 0, 0, 0, 0))(
        params, endog, exog_loc, exog_scale, exog_shape, weights
    )
    
    # B = Sum of outer products of these scaled gradients
    B = jnp.einsum('ni,nj->ij', grads, grads)
    
    return H, B

def return_level_atomic(params, exog_loc, exog_scale, exog_shape, T, dims, reparam_T):
    # predict_parameters expects (Time, Space, Covariates) -> (N, S, K)
    # exog_loc coming in is just (K,).
    # We must expand it to (1, 1, K) to satisfy 'nsk' in einsum.
    
    mu, sigma, xi = predict_parameters(
        params, 
        exog_loc[None, None, :],  # Added extra None
        exog_scale[None, None, :],  # Added extra None
        exog_shape[None, None, :],  # Added extra None
        dims, 
        reparam_T
    )
    
    # The output shape will be (1, 1) because of the fake dims.
    # We squeeze it completely to get a scalar () for the math below.
    mu = jnp.squeeze(mu)
    sigma = jnp.squeeze(sigma)
    xi = jnp.squeeze(xi)
    # --- FIX END ---
    
    y = -jnp.log(1.0 - 1.0/T)
    
    # Gumbel Logic
    rl_gumbel = mu - sigma * jnp.log(y)
    
    # GEV Logic
    xi_safe = jnp.where(jnp.abs(xi) < 1e-5, 1.0, xi)
    term = jnp.exp(-xi_safe * jnp.log(y))
    rl_gev = mu - (sigma / xi_safe) * (1.0 - term)
    
    # Select
    result = jnp.where(jnp.abs(xi) < 1e-5, rl_gumbel, rl_gev)
    return result

# --- 2. The Vectorization (Double Vmap) ---

# Inner Vmap: Vectorize over T (Return Periods)
# Input: (K), (K), (K), (N_periods) -> Output: (N_periods)
rl_over_periods = jax.vmap(
    return_level_atomic, 
    in_axes=(None, None, None, None, 0, None, None)
)

# Outer Vmap: Vectorize over Data Indices (Batch of t/s)
# Input: (Batch, K), (Batch, K), (Batch, K), (N_periods) -> Output: (Batch, N_periods)
rl_batch_2d = jax.vmap(
    rl_over_periods,
    in_axes=(None, 0, 0, 0, None, None, None)
)

# Gradient Vmap: Same logic, but applied to grad()
grad_atomic = jax.grad(return_level_atomic, argnums=0)
grad_over_periods = jax.vmap(grad_atomic, in_axes=(None, None, None, None, 0, None, None))
grad_batch_2d = jax.vmap(grad_over_periods, in_axes=(None, 0, 0, 0, None, None, None))

@partial(jax.jit, static_argnames=('dims', 'reparam_T'))
def compute_return_levels_general(params, exog_loc, exog_scale, exog_shape, T_array, dims, reparam_T=None):
    """
    Computes Point Estimates and Gradients in one highly optimized pass.
    Inputs:
        exog_loc, exog_scale, exog_shape: Shape (N_batch, K_covariates)
        T_array: Shape (N_periods,)
    Returns:
        zp: (N_batch, N_periods)
        grads: (N_batch, N_periods, N_params)
    """
    zp = rl_batch_2d(params, exog_loc, exog_scale, exog_shape, T_array, dims, reparam_T)
    grads = grad_batch_2d(params, exog_loc, exog_scale, exog_shape, T_array, dims, reparam_T)
    return zp, grads