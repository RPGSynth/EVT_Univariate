import numpy as np
import jax.numpy as jnp
from typing import Union, TYPE_CHECKING
from .engines.jax_engine import compute_return_levels_batch, compute_rl_gradients_batch

if TYPE_CHECKING:
    from .gev_results import GEVFit

class ReturnLevel:
    def __init__(self, fit: 'GEVFit', t_idx: int = 0, s_idx: int = 0):
        self.fit = fit
        self.t = t_idx
        self.s = s_idx
        self.reparam_T = getattr(fit, 'reparam_T', None)
        
        # Correct shapes (1, 1, K)
        self.x_l = jnp.array(self.fit.input.exog_loc[self.t, self.s, :][None, None, :], dtype=float)
        self.x_s = jnp.array(self.fit.input.exog_scale[self.t, self.s, :][None, None, :], dtype=float)
        self.x_x = jnp.array(self.fit.input.exog_shape[self.t, self.s, :][None, None, :], dtype=float)
        
        self.params_j = jnp.array(self.fit.params, dtype=float)
        self.cov_j = jnp.array(self.fit.cov_matrix, dtype=float)

    def compute(self, return_periods: Union[float, np.ndarray, list]):
        T_array = jnp.atleast_1d(jnp.array(return_periods, dtype=float))
        
        # 1. Point Estimates
        zp_jax = compute_return_levels_batch(
            self.params_j, self.x_l, self.x_s, self.x_x, T_array, 
            self.fit.dims, reparam_T=self.reparam_T
        )
        
        # 2. Gradients
        grads_jax = compute_rl_gradients_batch(
            self.params_j, self.x_l, self.x_s, self.x_x, T_array, 
            self.fit.dims, reparam_T=self.reparam_T
        )
        
        # 3. Variance
        var_jax = jnp.einsum('pi,ij,pj->p', grads_jax, self.cov_j, grads_jax)
        se_jax = jnp.sqrt(var_jax)
        
        # zp_jax is now already (N_periods,) because we squeezed inside the engine
        return np.array(zp_jax), np.array(se_jax)

    def plot_on_axis(self, periods, ax, color, show_ci=True, label_prefix=""):
        periods = np.array(periods)
        levels, ses = self.compute(periods)
        
        ax.plot(periods, levels, label=label_prefix, color=color, lw=2)
        
        if show_ci:
            if np.all(np.isfinite(ses)):
                z_crit = 1.96
                lower = levels - z_crit * ses
                upper = levels + z_crit * ses
                ax.fill_between(periods, lower, upper, color=color, alpha=0.15)
            
        ax.set_xlabel("Return Period (Years)")
        ax.set_ylabel("Return Level")
        ax.grid(True, which="both", ls=":", alpha=0.5)