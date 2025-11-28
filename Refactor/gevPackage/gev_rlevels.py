import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Union, TYPE_CHECKING

from .engines.jax_engine import compute_return_levels_general

if TYPE_CHECKING:
    from .gev_results import GEVFit


def _z_from_confidence(confidence: float) -> float:
    if not (0.0 < confidence < 1.0):
        raise ValueError("`confidence` must be in (0, 1).")
    return float(norm.ppf(0.5 + confidence / 2.0))


class ReturnLevel:
    def __init__(
        self,
        fit: "GEVFit",
        t: Union[int, np.ndarray, list] = 0,
        s: Union[int, np.ndarray, list] = 0,
        confidence: float = 0.95,
    ):
        """
        Parameters
        ----------
        fit : GEVFit
            Fitted GEV model.
        t : int or array-like, default 0
            Time indices to consider. Can be a single int or an array of ints.
        s : int or array-like, default 0
            Site indices to consider. Can be a single int or an array of ints.
        confidence : float, default 0.95
            Confidence level for the (symmetric) normal-approximation intervals.
        """
        self.fit = fit
        self.reparam_T = getattr(fit, "reparam_T", None)
        self.params_j = jnp.array(self.fit.params, dtype=float)
        self.cov_j = jnp.array(self.fit.cov_matrix, dtype=float)
        self.confidence = float(confidence)

        # --- 1. Smart indexing over t and s ---
        t_idx = np.atleast_1d(t)
        s_idx = np.atleast_1d(s)

        # meshgrid gives all combinations (t, s)
        self.t_grid, self.s_grid = np.meshgrid(t_idx, s_idx, indexing="ij")
        self.shape_grid = self.t_grid.shape  # (N_t, N_s)

        # Flatten for batch computations
        flat_t = self.t_grid.ravel()
        flat_s = self.s_grid.ravel()

<<<<<<< Updated upstream
        data = self.fit.data
        self.exog_loc = jnp.array(data.exog_loc[flat_t, flat_s, :], dtype=float)
        self.exog_scale = jnp.array(data.exog_scale[flat_t, flat_s, :], dtype=float)
        self.exog_shape = jnp.array(data.exog_shape[flat_t, flat_s, :], dtype=float)
=======
        inp = self.fit.input
        self.exog_loc = jnp.array(inp.exog_loc[flat_t, flat_s, :], dtype=float)
        self.exog_scale = jnp.array(inp.exog_scale[flat_t, flat_s, :], dtype=float)
        self.exog_shape = jnp.array(inp.exog_shape[flat_t, flat_s, :], dtype=float)
>>>>>>> Stashed changes

    def compute(self, T):
        """
        Compute return levels, standard errors, and confidence intervals.

        Parameters
        ----------
        T : array-like
            Return periods T. Shape (N_periods,). The order here defines
            the last axis of the outputs.

        Returns
        -------
        zp : np.ndarray
            Return levels, shape (N_t, N_s, N_periods).
        se : np.ndarray
            Standard errors, shape (N_t, N_s, N_periods).
        ci : np.ndarray
            Confidence intervals, shape (N_t, N_s, N_periods, 2),
            with ci[..., 0] = lower, ci[..., 1] = upper.
        """
        T_array = jnp.atleast_1d(jnp.array(T, dtype=float))

        # 1. Engine call: zp and gradients wrt parameters
        #    zp_flat:    (Batch, N_periods)
        #    grads_flat: (Batch, N_periods, N_params)
        zp_flat, grads_flat = compute_return_levels_general(
            self.params_j,
            self.exog_loc,
            self.exog_scale,
            self.exog_shape,
            T_array,
            self.fit.dims,
            reparam_T=self.reparam_T,
        )

        # 2. Delta-method variance: (Batch, Period)
        var_flat = jnp.einsum("bpi,ij,bpj->bp", grads_flat, self.cov_j, grads_flat)
        se_flat = jnp.sqrt(var_flat)

        # 3. Reshape back to (N_t, N_s, N_periods)
        final_shape = self.shape_grid + (len(T_array),)
        zp = np.asarray(zp_flat).reshape(final_shape)
        se = np.asarray(se_flat).reshape(final_shape)

        # 4. Symmetric normal-approximation CI
        z = _z_from_confidence(self.confidence)
        lower = zp - z * se
        upper = zp + z * se

        # ci: (N_t, N_s, N_periods, 2)
        ci = np.stack((lower, upper), axis=-1)

        return zp, se, ci
