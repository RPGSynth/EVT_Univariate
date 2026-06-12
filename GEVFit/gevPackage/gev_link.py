import jax.numpy as jnp
import numpy as np
from jax.nn import softplus

class GEVLinkage:
    """
    Bijector mapping Real Space (Optimizer) <-> Model Space (GEV parameters).
    """
    
    # ------------------------------------------------------------------
    # JAX Methods (Used by the Engine)
    # ------------------------------------------------------------------
    
    def forward(self, params_flat: jnp.ndarray, dims: tuple):
        """Transforms flat parameters theta -> (mu, sigma, xi)."""
        d_loc, d_scale, d_shape = dims
        
        # Slicing the flat vector based on the tuple dimensions
        beta_loc   = params_flat[:d_loc]
        beta_scale = params_flat[d_loc : d_loc + d_scale]
        beta_shape = params_flat[d_loc + d_scale :]
        
        return beta_loc, beta_scale, beta_shape

    def transform_scale(self, lin_pred_scale):
        """
        Maps linear predictor to positive scale.
        Using Softplus: log(1 + exp(x)).
        
        Why? 
        1. Guarantees sigma > 0.
        2. Linear behavior for large x (avoids overflow unlike exp).
        3. Smooth gradient near zero.
        """
        
        return softplus(lin_pred_scale)

    # ------------------------------------------------------------------
    # NumPy Methods (Used by Results/Plotting to avoid JAX overhead)
    # ------------------------------------------------------------------

    def np_transform_scale(self, x):
        """
        NumPy equivalent of Softplus for plotting routines.
        Mathematically: log(1 + exp(x))
        Numerically stable version: x if x > 20 else log(1+exp(x))
        """
        # Stable implementation to avoid overflow in exp for large x
        #np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

    # ------------------------------------------------------------------
    # Initialization Strategy
    # ------------------------------------------------------------------

    def initial_guess(self, endog, dims, reparam_T=None):
        """
        Generates smart starting values.
        If reparam_T is provided, initializes Location parameters to target the T-year return level.
        """
        d_loc, d_scale, d_shape = dims
        
        # Basic Stats
        mean = np.mean(endog)
        std = np.std(endog)
        
        # Method of Moments for Scale
        scale_mom = (std * np.sqrt(6.0)) / np.pi
        
        if reparam_T is not None:
            # Guess Zp directly using empirical quantile
            # Probability p = 1 - 1/T
            p = 1.0 - 1.0 / reparam_T
            loc_guess = np.quantile(endog, p)
        else:
            # Guess Mu using Gumbel approximation
            euler_gamma = 0.5772156649
            loc_guess = mean - euler_gamma * scale_mom
        
        # --- Construct Vectors ---
        
        # Location / Zp
        beta_loc = jnp.concatenate([jnp.array([loc_guess]), jnp.zeros(d_loc - 1)])
        
        # Scale (Inverse Softplus)
        scale_intercept = jnp.where(
            scale_mom > 20.0, scale_mom, jnp.log(jnp.expm1(scale_mom))
        )
        beta_scale = jnp.concatenate([jnp.array([scale_intercept]), jnp.zeros(d_scale - 1)])
        
        # Shape
        beta_shape = jnp.concatenate([jnp.array([0.1]), jnp.zeros(d_shape - 1)])
        
        return jnp.concatenate([beta_loc, beta_scale, beta_shape])