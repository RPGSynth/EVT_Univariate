import numpy as np
import jax.numpy as jnp  # <--- Use JAX for linalg
import jaxopt
from typing import Dict, Optional
from .gev_types import GEVInput
from .engines.jax_engine import nloglike_sum, compute_sandwich_matrices, linker
from .gev_results import GEVFit

class GEVModel:
    def __init__(self, max_iter=1000,reparam_T=None,confidence=0.95):
        """
        Args:
            reparam_T (float, optional): If set (e.g., 100), the model replaces the 
                                         location parameter 'mu' with the T-year return level 'zp'.
        """
        self.max_iter = max_iter
        self.reparam_T = float(reparam_T) if reparam_T is not None else None
        self.confidence = confidence
    
    def fit(self, endog, exog=None, weights=None) -> GEVFit:
        # 1. Parse Data
        data = GEVInput.from_inputs(endog, exog, weights)
        dims = data.covariate_dims 
        W_total = np.sum(data.weights)
        
        # 2. Init Guess
        init_params = linker.initial_guess(data.endog, dims, self.reparam_T)
        
        # 3. Define Objective
        def objective(p):
            nll_sum = nloglike_sum(
                p, data.endog, data.exog_loc, data.exog_scale, data.exog_shape, data.weights, dims,reparam_T=self.reparam_T
            )
            return nll_sum / W_total
            
        # 4. Optimize (JAX)
        solver = jaxopt.LBFGS(fun=objective, maxiter=self.max_iter, tol=1e-6)
        res = solver.run(init_params)

        # 4b. Fail fast on optimization issues instead of silently using bad fits.
        state = res.state

        def _scalar_or_none(x):
            if x is None:
                return None
            arr = np.asarray(x)
            if arr.size != 1:
                return None
            return float(arr.reshape(()))

        params_np = np.asarray(res.params, dtype=float)
        if not np.isfinite(params_np).all():
            raise RuntimeError("LBFGS optimization produced non-finite parameters.")

        nll_avg = _scalar_or_none(getattr(state, "value", None))
        if nll_avg is None or not np.isfinite(nll_avg):
            raise RuntimeError("LBFGS optimization produced a non-finite objective value.")

        grad = getattr(state, "grad", None)
        if grad is not None:
            grad_np = np.asarray(grad, dtype=float)
            if not np.isfinite(grad_np).all():
                raise RuntimeError("LBFGS optimization produced non-finite gradients.")

        failed = getattr(state, "failed", None)
        if failed is not None:
            failed_val = bool(np.asarray(failed).reshape(()))
            if failed_val:
                raise RuntimeError("LBFGS optimization reported failure.")

        converged = getattr(state, "converged", None)
        if converged is not None:
            converged_val = bool(np.asarray(converged).reshape(()))
            if not converged_val:
                iter_num = _scalar_or_none(getattr(state, "iter_num", None))
                error = _scalar_or_none(getattr(state, "error", None))
                raise RuntimeError(
                    f"LBFGS did not converge (iter_num={iter_num}, error={error})."
                )
        else:
            iter_num = _scalar_or_none(getattr(state, "iter_num", None))
            error = _scalar_or_none(getattr(state, "error", None))
            tol = _scalar_or_none(getattr(solver, "tol", None))
            if (
                iter_num is not None
                and iter_num >= float(self.max_iter)
                and error is not None
                and tol is not None
                and error > tol
            ):
                raise RuntimeError(
                    f"LBFGS likely stopped at max_iter without convergence "
                    f"(iter_num={iter_num}, error={error}, tol={tol})."
                )
        
        # 5. Sandwich Covariance (JAX)
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
            nll_avg=float(nll_avg),
            data=data,
            linker=linker,
            reparam_T=self.reparam_T,
            confidence=self.confidence
        )
