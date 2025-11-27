import numpy as np
import jax.numpy as jnp
from typing import Union, TYPE_CHECKING
from .engines.jax_engine import compute_return_levels_general

if TYPE_CHECKING:
    from .gev_results import GEVFit

class ReturnLevel:
    def __init__(self, fit: 'GEVFit', t=0, s=0):
        self.fit = fit
        self.reparam_T = getattr(fit, 'reparam_T', None)
        self.params_j = jnp.array(self.fit.params, dtype=float)
        self.cov_j = jnp.array(self.fit.cov_matrix, dtype=float)

        # --- 1. Smart Indexing ---
        # Convert inputs to arrays (atleast_1d)
        t_idx = np.atleast_1d(t)
        s_idx = np.atleast_1d(s)

        # Create a meshgrid of indices (Cartesian Product)
        # Result: If t=[0,1] and s=[0], we get shape (2,1)
        # This allows computing "All requested times for all requested sites"
        self.t_grid, self.s_grid = np.meshgrid(t_idx, s_idx, indexing='ij')
        
        # Save shapes for reconstruction later
        self.shape_grid = self.t_grid.shape # (N_t, N_s)
        
        # --- 2. Extract & Flatten Covariates ---
        # We perform the slicing on the original 3D matrices
        # Then flatten to (Batch_Size, K) for the JAX engine
        # Batch_Size = N_t * N_s
        flat_t = self.t_grid.ravel()
        flat_s = self.s_grid.ravel()

        self.x_l = jnp.array(self.fit.input.exog_loc[flat_t, flat_s, :], dtype=float)
        self.x_s = jnp.array(self.fit.input.exog_scale[flat_t, flat_s, :], dtype=float)
        self.x_x = jnp.array(self.fit.input.exog_shape[flat_t, flat_s, :], dtype=float)

    def compute(self, return_periods):
        T_array = jnp.atleast_1d(jnp.array(return_periods, dtype=float))
        
        # 1. Compute Everything (Double-Vmapped Engine)
        # zp_flat shape: (Batch_Size, N_periods)
        # grads_flat shape: (Batch_Size, N_periods, N_params)
        zp_flat, grads_flat = compute_return_levels_general(
            self.params_j, self.x_l, self.x_s, self.x_x, T_array, 
            self.fit.dims, reparam_T=self.reparam_T
        )
        
        # 2. Vectorized Variance Calculation
        # Einstein Summation over params (p, j)
        # (Batch, Period, Param) @ (Param, Param) @ (Batch, Period, Param)
        # Result: (Batch, Period)
        var_flat = jnp.einsum('bpi,ij,bpj->bp', grads_flat, self.cov_j, grads_flat)
        se_flat = jnp.sqrt(var_flat)
        
        # 3. Reshape and Return
        # We restore the (N_t, N_s) dimensions
        # Final Shape: (N_t, N_s, N_periods)
        final_shape = self.shape_grid + (len(T_array),)
        
        return np.array(zp_flat).reshape(final_shape), np.array(se_flat).reshape(final_shape)
