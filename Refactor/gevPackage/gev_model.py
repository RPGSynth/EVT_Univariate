import numpy as np
import jax.numpy as jnp  # <--- Use JAX for linalg
import jaxopt
from typing import Dict, Optional
from .gev_types import GEVInput
from .gev_engine import nloglike_sum, compute_sandwich_matrices, linker
from .gev_results import GEVFit

class GEVModel:
    def __init__(self, max_iter=1000,reparam_T=None):
        """
        Args:
            reparam_T (float, optional): If set (e.g., 100), the model replaces the 
                                         location parameter 'mu' with the T-year return level 'zp'.
        """
        self.max_iter = max_iter
        self.reparam_T = float(reparam_T) if reparam_T is not None else None
    
    def fit(self, endog, exog=None, weights=None) -> GEVFit:
        # 1. Parse Data
        data = GEVInput.from_inputs(endog, exog, weights)
        dims = data.covariate_dims 
        W_total = np.sum(data.weights)
        
        # 2. Init Guess
        print("Initializing...")
        init_params = linker.initial_guess(data.endog, dims, self.reparam_T)
        
        # 3. Define Objective
        def objective(p):
            nll_sum = nloglike_sum(
                p, data.endog, data.exog_loc, data.exog_scale, data.exog_shape, data.weights, dims,reparam_T=self.reparam_T
            )
            return nll_sum / W_total
            
        # 4. Optimize (JAX)
        print(f"Optimizing (Avg NLL) with L-BFGS...")
        solver = jaxopt.LBFGS(fun=objective, maxiter=self.max_iter, tol=1e-12)
        res = solver.run(init_params)
        
        # 5. Sandwich Covariance (JAX)
        print("Calculating Covariance...")
        # H and B are JAX arrays here
        H, B = compute_sandwich_matrices(
            res.params, data.endog, data.exog_loc, data.exog_scale, data.exog_shape, data.weights, dims,reparam_T=self.reparam_T
        )
        
        # If running on GPU, this keeps the matrix on VRAM for the inversion
        try:
            # jnp.linalg.inv is JIT-compatible
            H_inv = jnp.linalg.inv(H)
            cov_matrix_jax = H_inv @ B @ H_inv
        except Exception: 
            # JAX throws slightly different errors than NumPy for singular matrices
            print("Warning: Hessian inversion failed.")
            cov_matrix_jax = jnp.full((len(res.params), len(res.params)), jnp.nan)

        # 6. Result (The Handover)
        # We explicitly cast to np.array() here to move data to CPU
        # so GEVFit can work easily with Pandas/Matplotlib
        return GEVFit(
            params=np.array(res.params),       # JAX -> NumPy
            cov_matrix=np.array(cov_matrix_jax), # JAX -> NumPy
            n_ll_avg=float(res.state.value),   # JAX scalar -> Python float
            input_data=data,
            linker=linker,
            reparam_T=self.reparam_T
        )