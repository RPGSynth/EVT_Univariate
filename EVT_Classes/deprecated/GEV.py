# Standard library imports
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import warnings
from typing import Dict, Union, Optional, Callable, Any, List, Tuple, TypeVar
from abc import ABC, abstractmethod
import logging
import time
from functools import partial

# Third-party library imports
import numpy as np
import pandas as pd
import xarray as xrs
import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import minimize, approx_fprime, OptimizeResult
from scipy.stats import norm, chi2, gumbel_r, genextreme
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
from enum import Enum, auto


#Testing Jax
import jax
import jax.numpy as jnp
from jax import grad, hessian, jit, vmap
import jaxopt
logging.basicConfig(level=logging.INFO)  # Set logging level to show info messages

class FitMethod(Enum):
    MLE = auto()
    PROFILE = auto()

# Type Aliases for clarity
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, List[float], List[List[float]], xrs.DataArray]
LinkFunc = Callable[[np.ndarray], np.ndarray]
ExogInput = Optional[Union[Dict[str, Optional[ArrayLike]], ArrayLike]]

class _GEVBase(ABC):
    """
    Abstract base class for GEV statistical models.

    Attributes
    ----------
    loc_link : LinkFunc
        Link function for the location parameter.
    scale_link : LinkFunc
        Link function for the scale parameter.
    shape_link : LinkFunc
        Link function for the shape parameter.
    loc_return_level_reparam : bool
        Flag indicating if location is reparameterized by return level.
    T : Optional[Union[int,float]]
        Return period used for reparameterization.
    endog : np.ndarray
        Endogenous data array, shape (n_obs, n_samples).
    exog : Dict[str, np.ndarray]
        Dictionary of exogenous data arrays, each shape (n_obs, n_covariates, n_samples).
    len_exog : Tuple[int, int, int]
        Number of covariates for location, scale, and shape.
    nparams : int
        Total number of parameters to be fitted.
    """

    endog: np.ndarray
    exog: Dict[str, np.ndarray]
    len_exog: Tuple[int, int, int]
    nparams: int
    loc_link: LinkFunc
    scale_link: LinkFunc
    shape_link: LinkFunc
    loc_return_level_reparam: bool
    T: Optional[Union[int,float]]

    @staticmethod
    def identity(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def exp_link(x: np.ndarray) -> np.ndarray:
        # Add small epsilon to prevent exp(very large negative number) -> 0 issues if needed?
        # Or handle potential overflows/underflows more robustly depending on optimizer behavior.
        with np.errstate(over='ignore'): # Ignore overflow for now, optimizer might handle it
            return np.soft(x)

    def __init__(
        self,
        loc_link: Optional[LinkFunc] = None,
        scale_link: Optional[LinkFunc] = None,
        shape_link: Optional[LinkFunc] = None,
        T : Optional[Union[int,float]] = None
        ) -> None:
        """
        Initializes the GEV model configuration.

        Parameters
        ----------
        loc_link : Optional[LinkFunc], optional
            Link function for location parameter. Defaults to identity.
        scale_link : Optional[LinkFunc], optional
            Link function for scale parameter. Defaults to identity.
        shape_link : Optional[LinkFunc], optional
            Link function for shape parameter. Defaults to identity.
        T : Optional[Union[int,float]], optional
            Return period for reparameterizing location. Defaults to None.
        """
        self.loc_link = loc_link if loc_link is not None else self.identity
        self.scale_link = scale_link if scale_link is not None else self.identity
        self.shape_link = shape_link if shape_link is not None else self.identity
        self.loc_return_level_reparam = T is not None and T>1
        if T is not None:
            if T <= 1: 
                 raise ValueError("Return period T must be greater than 1 for reparameterization.")
            logging.info("ℹ️ The location parameter (μ) will be redefined in terms of return levels (z_p), "
                         "as you have specified a return period (T > 1).")

        self.T = T


    def nloglike(self, params: np.ndarray, weights: Optional[np.ndarray] = None, forced_indices: Optional[Union[int, List[int], np.ndarray]] = None, forced_param_values: Optional[Union[float, List[float], np.ndarray]] = None) -> float:
        """
        Computes the negative log-likelihood of the GEV model.

        Handles multiple series, covariates via link functions, Gumbel case,
        and optional location parameter reparameterization.

        Args:
            params (np.ndarray): Array of model parameters (concatenated for loc, scale, shape).
                                 If profiling, these are the *free* parameters.
            forced_indices (Optional[Union[int, List[int], np.ndarray]]): Indices of parameters fixed for profiling.
            forced_param_values (Optional[Union[float, List[float], np.ndarray]]): Values for fixed parameters.

        Returns:
            float: Negative log-likelihood value. Returns 1e7 for invalid parameter combinations.
        """
        if weights is None:
            weights = getattr(self, "weights", None)
            if weights is None:
                weights = np.ones_like(self.endog)    # all observations get weight 1

        if weights.shape != self.endog.shape:
            raise ValueError("`weights` must have shape (n_obs, n_samples).")

        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("`weights` must be finite and non-negative.")

        total_weight = weights.sum()
        if total_weight == 0:
            raise ValueError("Sum of weights is zero; likelihood is undefined.")

        full_params = params # Placeholder - needs adjustment if called directly with fixed params

        # Extract the number of covariates for each parameter
        i, j, k = self.len_exog # Using self.len_exog assumes it's set by the subclass

        # Parameter slices based on len_exog
        loc_params = full_params[0:i]
        scale_params = full_params[i:i+j]
        shape_params = full_params[i+j:i+j+k] # or full_params[i+j:]

        # Compute scale and shape using einsum and link functions
        # Shape of exog[param]: (n_obs, n_cov_param, n_samples)
        # Shape of params_param: (n_cov_param,)
        # Result shape: (n_obs, n_samples)
        scale = self.scale_link(np.einsum('njp,j->np', self.exog['scale'], scale_params))

        #print("scale = " + str(np.mean(scale)))
        if np.any(scale <= 1e-9): # Use a small epsilon instead of 0 for stability
            return 1e7 # Increased penalty

        shape = self.shape_link(np.einsum('nkp,k->np', self.exog['shape'], shape_params))
        #print("shape = " + str(np.mean(shape)))
        # Compute location parameter (mu or zp)
        if self.loc_return_level_reparam:
            if self.T is None: # Should be set if loc_return_level_reparam is True
                 raise ValueError("T must be set for return level reparameterization.")
            zp = self.loc_link(np.einsum('nip,i->np', self.exog['location'], loc_params))

            # Check for finite zp values
            if np.any(~np.isfinite(zp)):
                return 1e7

            y_p = -np.log(1 - 1 / self.T) # Scalar if T is scalar

            # Safe computation of location (mu) from zp, scale, shape
            with np.errstate(divide='ignore', invalid='ignore'):
                shape_zero_mask = np.isclose(shape, 0)
                location = np.where(
                    shape_zero_mask,
                    zp + scale * np.log(y_p), # Corrected Gumbel inversion: zp = mu - scale*log(yp)
                    zp + scale * (1 - y_p**(-shape)) / shape # GEV inversion: zp = mu + scale/shape * ((-log(1-1/T))**(-shape) - 1) -> mu = zp - scale/shape * (yp**(-shape) - 1) 
                 )

        else:
            # Standard location parameter calculation
            location = self.loc_link(np.einsum('nip,i->np', self.exog['location'], loc_params))
            #print("loc = " + str(np.mean(location)))
        
        if np.any(~np.isfinite(location)):
            return 1e7


        # GEV transformation using calculated location, scale, shape
        # self.endog shape: (n_obs, n_samples)
        normalized_data = (self.endog - location) / scale
        transformed_data = 1 + shape * normalized_data # Shape: (n_obs, n_samples)

        is_gumbel = np.isclose(shape, 0)  # Shape: (n_obs, n_samples)

        # Check for invalid values: 1 + xi * (y - mu) / sigma > 0 for xi != 0
        # This check should technically be done *before* log/power operations
        invalid_gev_domain =  (weights > 0) & (scale <= 0) & (~is_gumbel) & (transformed_data <= 1e-9) # Use epsilon
        if np.any(invalid_gev_domain):
             return 1e7


        # Compute per-observation negative log-likelihood terms
        # Use np.errstate to handle potential numerical issues in log/power
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # Gumbel cases (shape is close to 0)
            log_scale = np.log(scale) # Calculated earlier, ensure scale > 0
            gumbel_term = log_scale + normalized_data + np.exp(-normalized_data)

            # General GEV case (shape is not close to 0)
            # log(transformed_data) requires transformed_data > 0 (checked above)
            # transformed_data ** (-1 / shape) involves power, potentially complex if base is negative (checked above)
            log_transformed = np.log(transformed_data) # Calculated for valid domain
            gev_term = log_scale + transformed_data**(-1 / shape) + (1 + 1 / shape) * log_transformed

            n_ll_terms = np.where(
                is_gumbel,
                gumbel_term,
                gev_term
            )

        # Check for non-finite results after computation (NaN, Inf)
        if not np.all(np.isfinite(n_ll_terms)):
             # This catches issues like exp() overflow or log(neg) etc. that weren't caught by domain checks
            return 1e7
        
        # Total negative log-likelihood (sum over all observations and samples)
        avg_nll = np.sum(n_ll_terms * weights)/total_weight

        # Final check if total NLL is finite
        if not np.isfinite(avg_nll):
             return 1e8 # Return even larger penalty if sum somehow becomes non-finite

        return avg_nll
    

    @staticmethod
    @partial(jit, static_argnames=('loc_link', 'scale_link', 'shape_link', 'loc_return_level_reparam','len_exog'))
    def nloglike_jax(params: jnp.ndarray,
                     endog: jnp.ndarray,
                     exog_loc: jnp.ndarray,
                     exog_scale: jnp.ndarray,
                     exog_shape: jnp.ndarray,
                     weights: jnp.ndarray,
                     len_exog: Tuple[int, int, int],
                     loc_link: Callable,
                     scale_link: Callable,
                     shape_link: Callable,
                     loc_return_level_reparam: bool,
                     T: Optional[float] = None) :
        """
        Computes the negative log-likelihood using JAX.
        This is a static method to be a pure function.
        """
        total_weight = jnp.sum(weights)

        i, j, k = len_exog
        loc_params = params[0:i]
        scale_params = params[i:i+j]
        shape_params = params[i+j:i+j+k]

        scale = scale_link(jnp.einsum('njp,j->np', exog_scale, scale_params))
        
        # JAX requires explicit handling of potential invalid values.
        # We can return a large value if scale is non-positive.
        is_scale_invalid = jnp.any(scale <= 1e-9)

        shape = shape_link(jnp.einsum('nkp,k->np', exog_shape, shape_params))

        if loc_return_level_reparam:
            if T is None:
                # This check should ideally happen before JIT compilation.
                # For simplicity, we assume T is valid if reparam is on.
                raise ValueError("T must be set for return level reparameterization.")
            zp = loc_link(jnp.einsum('nip,i->np', exog_loc, loc_params))
            y_p = -jnp.log(1 - 1 / T)
            
            shape_zero_mask = jnp.isclose(shape, 0)
            # JAX equivalent of the original logic
            location = jnp.where(
                shape_zero_mask,
                zp + scale * jnp.log(y_p),
                zp + scale / shape * (1 - y_p**(-shape))
            )
        else:
            location = loc_link(jnp.einsum('nip,i->np', exog_loc, loc_params))

        is_params_invalid = jnp.any(~jnp.isfinite(location))

        normalized_data = (endog - location) / scale
        transformed_data = 1 + shape * normalized_data
        is_gumbel = jnp.isclose(shape, 0)

        # Check for invalid GEV domain: 1 + xi * (y - mu) / sigma <= 0
        is_domain_invalid = jnp.any((~is_gumbel) & (transformed_data <= 1e-9) & (weights > 0))

        # Gumbel cases
        log_scale = jnp.log(scale)
        gumbel_term = log_scale + normalized_data + jnp.exp(-normalized_data)
        
        # General GEV case
        log_transformed = jnp.log(transformed_data)
        gev_term = log_scale + transformed_data**(-1 / shape) + (1 + 1 / shape) * log_transformed

        n_ll_terms = jnp.where(is_gumbel, gumbel_term, gev_term)

        # Final check for NaN/Inf in the terms themselves
        is_computation_invalid = jnp.any(~jnp.isfinite(n_ll_terms))
        
        # Calculate likelihood, but return a large penalty if any invalid condition was met
        avg_nll = jnp.sum(n_ll_terms * weights) / total_weight

        # Combine all invalid conditions.
        # Note: In a JIT context, all branches are typically evaluated.
        is_invalid = is_scale_invalid | is_params_invalid | is_domain_invalid | is_computation_invalid
        
        return jnp.where(is_invalid, 1e8, avg_nll)


    def loglike(self, params: np.ndarray) -> float:
        """Computes the log-likelihood (negative of nloglike)."""
        return -self.nloglike(params)

    @abstractmethod
    def fit(self, start_params: Optional[np.ndarray] = None, optim_method: str = 'L-BFGS-B', fit_method: FitMethod = FitMethod.MLE) -> 'GEVFit':
        """
        Fit the model to data. Abstract method to be implemented by subclasses.

        Returns
        -------
        GEVFit
            An object containing the results of the fit.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the fit method.")

class GEV(_GEVBase):
    """
    GEV model implementation for fitting a data sample.

    Handles data input, validation, covariate processing, MLE and Profile fitting.
    """
    def __init__(self,
                 endog: ArrayLike,
                 exog: ExogInput = None, # Default to None is handled 
                 weights : Optional[np.ndarray] =None,
                 loc_link: Optional[LinkFunc] = None,
                 scale_link: Optional[LinkFunc] = None,
                 shape_link: Optional[LinkFunc] = None,
                 T: Optional[Union[int,float]] = None,
                 **kwargs: Any) -> None:
        """
        Initializes the GEVSample model.

        Parameters
        ----------
        endog : ArrayLike
            Endogenous variable(s). Shape (n_obs,) or (n_obs, n_samples).
        exog : ExogInput, optional
            Exogenous variables. Can be None, dict, array-like. See _process_exog.
        loc_link : Optional[LinkFunc], optional
            Link function for location. Defaults to identity.
        scale_link : Optional[LinkFunc], optional
            Link function for scale. Defaults to identity.
        shape_link : Optional[LinkFunc], optional
            Link function for shape. Defaults to identity.
        T : Optional[Union[int,float]], optional
            Return period for location reparameterization. Defaults to None.
        kwargs : Any
            Additional keyword arguments (currently unused).

        Raises
        ------
        ValueError
            If data is empty, contains NaNs, or shapes are inconsistent.
        TypeError
            If inputs cannot be converted to numeric arrays or links aren't callable.
        """
        # Initialize base class attributes (links, T, reparam flag)
        super().__init__(loc_link=loc_link, scale_link=scale_link, shape_link=shape_link, T=T)

        self.weights = weights
        #---------- Process Endog input ----------
        if endog is None:
            raise ValueError("The `endog` parameter must not be None.")
        if hasattr(endog, '__len__') and len(endog) == 0:
             raise ValueError("The `endog` parameter must not be empty.")

        try:
            # Convert various inputs to numpy array
            if isinstance(endog, (pd.Series, pd.DataFrame)):
                endog_array = endog.values
            elif isinstance(endog, xrs.DataArray):
                 endog_array = endog.values
            else:
                endog_array = np.asarray(endog)

            # Ensure float type
            self.endog = endog_array.astype(np.float64) # Use float64 for precision

            # Reshape 1D array to 2D column vector (n_obs, 1 sample)
            if self.endog.ndim == 1:
                self.endog = self.endog.reshape(-1, 1)
            elif self.endog.ndim != 2:
                 raise ValueError(f"endog must be 1D or 2D, but got {self.endog.ndim} dimensions.")

        except Exception as e:
            raise TypeError(f"Could not convert endogenous data to a 2D numeric array: {e}")

        if np.isnan(self.endog).any():
            raise ValueError("The `endog` parameter contains NaN values. Please handle missing data before fitting.")

        #---------- Set Data Attributes ----------
        self.n_obs, self.n_samples = self.endog.shape
        self.N_total = self.endog.size # Total number of data points

        #---------- Initial Guesses ----------
        # Use overall mean/variance for initial guesses, even for multiple samples
        endog_mean = np.nanmean(self.endog[:,0])
        endog_var = np.nanmean(np.nanvar(self.endog[:,0], axis=0)) # Average variance across samples

        # Gumbel-based initial guesses (Method of Moments for Gumbel)
        self.scale_guess: float = max(np.sqrt(6 * endog_var) / np.pi, 1e-6) # Ensure positive
        self.shape_guess: float = 0.1 # Common starting point for GEV shape
        euler_gamma = 0.5772156649

        if self.loc_return_level_reparam:
            # Guess for zp (the T-year return level) based on Gumbel approx
            if self.T is None: raise ValueError("T must be set for reparameterization guess.") # Should not happen
            y_p_guess = -np.log(1 - 1 / self.T)
            # Approximate zp using Gumbel quantile formula: zp ~ mu_g - scale_g * log(yp)
            # where mu_g = mean - gamma * scale_g
            mu_gumbel_guess = endog_mean - euler_gamma * self.scale_guess
            self.location_guess = mu_gumbel_guess - self.scale_guess * np.log(y_p_guess)
            # Note: This is the guess for zp, not mu
        else:
            # Guess for mu (location) based on Gumbel approx: mu ~ mean - gamma * scale
             self.location_guess = endog_mean - euler_gamma * self.scale_guess
        #---------- Process Exog Data ----------
        self._process_exog(exog) # Populates self.exog, self.len_exog, self.nparams, self.trans

        #---------- Fit Attributes Init ----------
        self.result: Optional[OptimizeResult] = None # Store optimization result
        self.fitted: bool = False # Flag
        
        self.endog_j = jax.device_put(jnp.asarray(self.endog))
        self.exog_j = {
            'location': jax.device_put(jnp.asarray(self.exog['location'])),
            'scale': jax.device_put(jnp.asarray(self.exog['scale'])),
            'shape': jax.device_put(jnp.asarray(self.exog['shape'])),
        }
        # Also handle weights, ensuring it exists before conversion.
        # Your original __init__ sets self.weights.
        numpy_weights = self.weights if self.weights is not None else np.ones_like(self.endog)
        self.weights_j = jax.device_put(jnp.asarray(numpy_weights))

    def _process_exog(self, exog_input: ExogInput) -> None:
        """
        Processes and validates exogenous data, storing it internally.

        Sets `self.exog`, `self.len_exog`, `self.nparams`, `self.trans`.

        Parameters
        ----------
        exog_input : ExogInput
            Exogenous variables provided by the user.
        """
        param_names: List[str] = ['location', 'scale', 'shape']
        self.exog: Dict[str, np.ndarray] = {}
        processed_exog_arrays: Dict[str, np.ndarray] = {}

        default_exog_array = np.ones((self.n_obs, 1, self.n_samples), dtype=np.float64)

        # ---- Case 1: None provided ----
        if exog_input is None:
            for param in param_names:
                processed_exog_arrays[param] = default_exog_array

        # ---- Case 2: Dictionary provided ----
        elif isinstance(exog_input, dict):
            # Check for invalid keys first
            invalid_keys = set(exog_input.keys()) - set(param_names)
            if invalid_keys:
                raise ValueError(f"Invalid keys in exog dictionary: {invalid_keys}. Expected keys from: {param_names}")

            for param in param_names:
                param_exog_input = exog_input.get(param) # Use .get() for safety

                if param_exog_input is None:
                    processed_exog_arrays[param] = default_exog_array
                else:
                     # Convert various inputs to a NumPy array
                    try:
                        if isinstance(param_exog_input, (pd.Series, pd.DataFrame)):
                            param_exog_array = param_exog_input.values.astype(np.float64)
                        elif isinstance(param_exog_input, xrs.DataArray):
                            param_exog_array = param_exog_input.values.astype(np.float64)
                        else:
                            param_exog_array = np.asarray(param_exog_input, dtype=np.float64)
                    except Exception as e:
                         raise TypeError(f"Could not convert exog['{param}'] to a numeric array: {e}")

                    # Validate shape and add intercept
                    processed_exog_arrays[param] = self._validate_and_reshape_exog_array(param_exog_array, param)

        # ---- Case 3: Single ArrayLike provided (use for all params) ----
        else:
             # Convert various inputs to a NumPy array
            try:
                if isinstance(exog_input, (pd.Series, pd.DataFrame)):
                     param_exog_array = exog_input.values.astype(np.float64)
                elif isinstance(exog_input, xrs.DataArray):
                     param_exog_array = exog_input.values.astype(np.float64)
                else:
                     param_exog_array = np.asarray(exog_input, dtype=np.float64)
            except Exception as e:
                 raise TypeError(f"Could not convert the provided single exog input to a numeric array: {e}")

             # Validate shape and add intercept for each parameter
            for param in param_names:
                 # Process a copy to avoid modifying the original array across loops if it's mutable
                 processed_exog_arrays[param] = self._validate_and_reshape_exog_array(param_exog_array.copy(), param)

        # Store processed arrays in self.exog
        self.exog = processed_exog_arrays

        # Determine if any real covariates were added (beyond intercept)
        self.trans: bool = not all(
             exog_array.shape[1] == 1 for exog_array in self.exog.values()
        )

        # Calculate lengths and total parameters
        self.len_exog: Tuple[int, int, int] = (
            self.exog['location'].shape[1],
            self.exog['scale'].shape[1],
            self.exog['shape'].shape[1]
        )
        self.nparams: int = sum(self.len_exog)

    def _validate_and_reshape_exog_array(self, exog_array: np.ndarray, param_name: str) -> np.ndarray:
        """
        Validates covariate array shape and reshapes to (n_obs, n_covariates + 1, n_samples).
        Adds an intercept column.

        Parameters
        ----------
        exog_array : np.ndarray
            The input covariate array (already converted to numeric).
        param_name : str
            Name of the parameter ('location', 'scale', 'shape') for error messages.

        Returns
        -------
        np.ndarray
            The validated and reshaped array with intercept, shape (n_obs, n_covariates + 1, n_samples).
        """
        target_shape_3d = (self.n_obs, -1, self.n_samples) # Target shape after adding intercept

        # ---- Input shape validation and initial reshape to (n_obs, n_raw_cov, n_samples) ----
        if exog_array.ndim == 1:
            # Assumed shape (n_obs,) -> treat as one covariate varying with observation, constant across samples
            if exog_array.shape[0] != self.n_obs:
                 raise ValueError(f"1D exog['{param_name}'] length ({exog_array.shape[0]}) must match n_obs ({self.n_obs}).")
            # Reshape to (n_obs, 1 covariate, 1 sample) then broadcast to n_samples
            reshaped_exog = exog_array.reshape(self.n_obs, 1, 1)
            reshaped_exog = np.repeat(reshaped_exog, self.n_samples, axis=2) # Shape (n_obs, 1, n_samples)

        elif exog_array.ndim == 2:
            # Could be (n_obs, n_samples) or (n_obs, n_covariates)
            if exog_array.shape == (self.n_obs, self.n_samples):
                 # Assumed one covariate varying with obs and sample
                 reshaped_exog = exog_array.reshape(self.n_obs, 1, self.n_samples) # Shape (n_obs, 1, n_samples)
            elif exog_array.shape[0] == self.n_obs:
                 # Assumed (n_obs, n_covariates), constant across samples
                 n_cov = exog_array.shape[1]
                 reshaped_exog = exog_array.reshape(self.n_obs, n_cov, 1)
                 reshaped_exog = np.repeat(reshaped_exog, self.n_samples, axis=2) # Shape (n_obs, n_cov, n_samples)
            else:
                 raise ValueError(f"2D exog['{param_name}'] shape {exog_array.shape} is incompatible with endog shape ({self.n_obs}, {self.n_samples}). Expected ({self.n_obs}, {self.n_samples}) or ({self.n_obs}, n_covariates).")

        elif exog_array.ndim == 3:
            # Must be (n_obs, n_covariates, n_samples)
            if exog_array.shape[0] != self.n_obs or exog_array.shape[2] != self.n_samples:
                 raise ValueError(f"3D exog['{param_name}'] shape {exog_array.shape} is incompatible with endog shape ({self.n_obs}, {self.n_samples}). Expected ({self.n_obs}, n_covariates, {self.n_samples}).")
            reshaped_exog = exog_array # Already in correct shape structure

        else:
            raise ValueError(f"exog['{param_name}'] has an invalid number of dimensions: {exog_array.ndim}. Expected 1, 2, or 3.")

        # ---- Add Intercept ----
        # Create intercept array matching the shape (n_obs, 1, n_samples)
        intercept = np.ones((self.n_obs, 1, self.n_samples), dtype=np.float64)

        # Concatenate intercept along the covariate axis (axis=1)
        final_exog = np.concatenate([intercept, reshaped_exog], axis=1)

        # Final check for NaNs
        if np.isnan(final_exog).any():
             raise ValueError(f"exog['{param_name}'] contains NaN values after processing.")

        return final_exog
    


    def _get_objective_func(self):
        """
        Helper to create a partial function.
        This is where we perform checks that CANNOT be in the JIT'd code.
        """
        
        if self.loc_return_level_reparam and self.T is None:
            raise ValueError("Return period T must be provided for reparameterization.")
        # --- End of fix ---
        
        return partial(self.nloglike_jax,
                        # ... (all the other arguments as before) ...
                        endog=self.endog_j,
                        exog_loc=self.exog_j['location'],
                        exog_scale=self.exog_j['scale'],
                        exog_shape=self.exog_j['shape'],
                        weights=self.weights_j,
                        len_exog=self.len_exog,
                        loc_link=self.loc_link,
                        scale_link=self.scale_link,
                        shape_link=self.shape_link,
                        loc_return_level_reparam=self.loc_return_level_reparam,
                        T=self.T)

    def hess(self, params: np.ndarray):
        """
        Computes the Hessian matrix of the negative log-likelihood using numdifftools.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector at which to compute the Hessian.

        Returns
        -------
        np.ndarray
            The Hessian matrix.
        """
        hessian_fn = nd.Hessian(self.nloglike)
        hessian = hessian_fn(params.astype(np.float64).flatten())
        # Ensure params is flat float64 for numdifftools
        return hessian

    def hess_inv(self, params: np.ndarray) -> np.ndarray:
        """
        Computes the inverse of the Hessian matrix.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector.

        Returns
        -------
        np.ndarray
            Inverse Hessian matrix (potential covariance matrix).
        """
        hess_matrix = self.hess(params)
        try:
            inv_hess = np.linalg.inv(hess_matrix)
            return inv_hess
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Hessian matrix inversion failed: {e}. Covariance matrix will be invalid.", RuntimeWarning)
            # Return matrix of NaNs with the correct shape
            return np.full((self.nparams, self.nparams), np.nan, dtype=np.float64)


    def score(self, params: np.ndarray):
        """
        Computes the score function (gradient of the log-likelihood / negative gradient of NLL).

        Uses finite differences approximation.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector at which to compute the score.

        Returns
        -------
        np.ndarray
            The score vector.
        """
        # approx_fprime computes gradient of the function (NLL)
        # Score is gradient of LL, so -gradient(NLL)
        # Epsilon might need tuning depending on parameter scales
        epsilon = np.sqrt(np.finfo(float).eps) # Standard epsilon choice
        grad_nll = approx_fprime(params.astype(np.float64).flatten(), self.nloglike, epsilon=epsilon)
        return -grad_nll


    def _generate_profile_params(self, param_idx: int, n_points: int, mle_params: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """
        Generates a range of values for a specific parameter for profiling.

        Parameters
        ----------
        param_idx : int
            Index of the parameter to profile.
        n_points : int
            Number of points to generate in the range.
        mle_params : np.ndarray
            Maximum Likelihood Estimates of all parameters.
        stds : np.ndarray
            Standard errors of the MLE parameters.

        Returns
        -------
        np.ndarray
            Array of values for the specified parameter.
        """
        i, j, k = self.len_exog
        mle_val = mle_params[param_idx]
        std_val = stds[param_idx]

        # Handle cases where std_val might be zero or NaN
        if not np.isfinite(std_val) or std_val <= 1e-9:
            # If std error is invalid, create a small range around MLE
            warnings.warn(f"Invalid standard error for parameter {param_idx}. Using a small fixed range for profiling.", RuntimeWarning)
            param_range = np.linspace(mle_val - 0.1 * abs(mle_val) - 1e-6, mle_val + 0.1 * abs(mle_val) + 1e-6, n_points)
        else:
            # Use +/- 3 standard deviations as a starting point
            lower_bound = mle_val - 3 * std_val
            upper_bound = mle_val + 3 * std_val
             # Special handling for scale parameter: ensure range stays positive if link allows?
             # Example: if scale param corresponds to intercept and uses exp link, the param can be negative,
             # but the actual scale exp(param) > 0. No constraint needed here.
             # If scale used identity link, maybe enforce > 0? But MLE fit should handle this.
             # Let's assume the range is okay for now.

             # Special handling for shape parameter? Sometimes constrained e.g. shape < 0.5?
             # Not enforced here, relies on nloglike penalty.

            param_range = np.linspace(lower_bound, upper_bound, n_points)

        return param_range
    
    def _generate_free_params(self,param_idx, fixed_param_values = None):
        """Efficient single-loop version of the parameter profiling process."""
        if fixed_param_values is None:
            i, j, k = self.len_exog  # Extract external indices

            # Predefine default values for free parameters
            default_params = [self.location_guess if idx == 0 else
                            self.scale_guess if idx == i else
                            self.shape_guess if idx == i + j else 0
                            for idx in range(self.nparams)]

            free_params = default_params[:param_idx] + default_params[param_idx+1:]
            return free_params  
        else:
            return np.concatenate((fixed_param_values[:param_idx],fixed_param_values[param_idx+1:]))

        #Made to be run in a parallel computing framework 
    def _optimize_profile_parallel(self,args):
        param_idx,n,optim_method,mle_params,stds = args
        profile_mles = np.empty(n)
        param_values = np.empty(n)

        #Expected error, the likelihood function was changed and I didn't added profile params yet.
        profile_params = self._generate_profile_params(param_idx=param_idx,n=n,mle_params=mle_params,stds=stds)
        free_params = self._generate_free_params(param_idx=param_idx)
        for n_i, param_value in enumerate(profile_params):
                    nloglike_partial = partial(self.nloglike,forced_indices=param_idx,forced_param_values=param_value)
                    result = minimize(nloglike_partial, free_params, method=optim_method)
                    profile_mles[n_i] = result.fun
                    param_values[n_i] = param_value
        
        return param_idx, profile_mles, param_values

    def fit(self, start_params=None, optim_method='L-BFGS-B', fit_method='MLE'):
        """
        Fits the model using the specified method (MLE or Profile).
        """
        RL_args : dict[str, bool | int] = {"reparam": self.loc_return_level_reparam}

        if RL_args["reparam"]:  # Only include T if reparam is True
            RL_args["T"] = self.T

        match fit_method.lower():
            case "mle":
                return self._fit_mle(start_params=start_params, optim_method=optim_method,RL_args=RL_args)
            case "profile":
                print("oh")
                return self._fit_profile(start_params, optim_method, RL_args)
            case _:
                raise ValueError("Unsupported fit method. Choose 'MLE' or 'Profile'.")
            
    def _fit_mle_jax(self, start_params: Optional[jnp.ndarray] = None, maxiter: int = 500):
        """
        Fits the model using JAX. It optimizes the parameters, then computes
        the Hessian and the robust covariance matrix.
        """
        # Use the same initial guess logic, but ensure it's a JAX array
        if start_params is None:
            i, j, k = self.len_exog
            vec = ([self.location_guess] + [0.] * (i - 1) +
                [self.scale_guess] + [0.] * (j - 1) +
                [self.shape_guess] + [0.] * (k - 1))
            start_params = jnp.array(vec, dtype=jnp.float32)

        # --- 1. Get the JIT-able objective function ---
        #print("Preparing JAX objective function...")
        obj_fun = self._get_objective_func()

        # --- 2. Run the JAX-native optimizer ---
        #print("Running optimization with jaxopt.LBFGS...")
        solver = jaxopt.LBFGS(fun=obj_fun, maxiter=maxiter, tol=1e-5)
        mle_params, state = solver.run(start_params)
        n_ll = state.value

        if state.error > 1e-3: # Check if solver converged
            warnings.warn(f"Optimization may not have converged. Final error: {state.error}", RuntimeWarning)
        #print(f"Optimization finished. Final NLL: {n_ll:.4f}")

        # --- 3. Compute the Hessian ("Bread" of the sandwich) ---
        #print("Calculating Hessian matrix...")
        H_bar = jax.hessian(obj_fun)(mle_params)

        # --- 4. Compute the Robust Covariance Matrix ---
        #print("Calculating robust Godambe (sandwich) covariance matrix...")
        cov_matrix_j = self._compute_godambe_covariance_jax(mle_params, H_bar)

        # --- 5. Package and Return Results ---
        #print("Fit complete. Packaging results.")
        # Convert JAX arrays back to NumPy for the GEVFit object and downstream use
        fitted_loc, fitted_scale, fitted_shape = self._get_gev_params_numpy(np.array(mle_params))
        
        # We will need to update GEVFit to handle these results
        return GEVFit(
            gevSample=self,
            fitted_params=np.array(mle_params),
            gev_params=(fitted_loc, fitted_scale, fitted_shape),
            n_ll=float(n_ll),
            cov_matrix=np.array(cov_matrix_j),
            jacobian=np.array(state.grad), # Gradient at the final step
            # CIs will be computed from the covariance matrix inside GEVFit or a helper
            CIs=self._compute_CIs_mle(np.array(cov_matrix_j), np.array(mle_params), 0.95),
            fit_method='MLE_JAX',
            RL_args={"reparam": self.loc_return_level_reparam, "T": self.T}
        )


    def _get_gev_params_numpy(self, params: np.ndarray):
        """Helper to compute fitted GEV parameters from regression coefficients using NumPy."""
        i, j, k = self.len_exog
        loc_p, scale_p, shape_p = params[:i], params[i:i+j], params[i+j:]
        
        fitted_loc = self.loc_link(np.einsum('nip,i->np', self.exog['location'], loc_p))
        fitted_scale = self.scale_link(np.einsum('njp,j->np', self.exog['scale'], scale_p))
        fitted_shape = self.shape_link(np.einsum('nkp,k->np', self.exog['shape'], shape_p))
        
        return fitted_loc, fitted_scale, fitted_shape
    

    # Add this method to your GEV_JAX_Final class

    def _compute_godambe_covariance_jax(self, mle_params: jnp.ndarray, H_bar: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Godambe information matrix (sandwich covariance) using JAX.
        A = H_bar (Hessian of average NLL)
        B = (1/W^2) * sum(g_i @ g_i.T) where g_i is the sum of weighted gradients for row i.
        """
        w_mat = self.weights_j
        W = jnp.sum(w_mat)
        
        # 1. Define a function that computes NLL for a SINGLE observation row
        def row_nll(w_row, endog_row, ex_loc_row, ex_sca_row, ex_sha_row, params):
            # We need to add a batch dimension for the pure function to work
            w_row, endog_row = w_row[jnp.newaxis, :], endog_row[jnp.newaxis, :]
            ex_loc_row, ex_sca_row, ex_sha_row = (
                ex_loc_row[jnp.newaxis, ...], 
                ex_sca_row[jnp.newaxis, ...], 
                ex_sha_row[jnp.newaxis, ...]
            )
            return self.nloglike_jax(
                params, endog_row, ex_loc_row, ex_sca_row, ex_sha_row, w_row, 
                self.len_exog, self.loc_link, self.scale_link, self.shape_link, 
                self.loc_return_level_reparam, self.T
            )

        # 2. Get the gradient of the per-row NLL function with respect to params.
        grad_row_nll = jax.grad(row_nll, argnums=5) # argnums=5 is `params`

        # 3. Use vmap to "vectorize" the gradient calculation across all rows at once.
        # This is the magic that replaces the slow Python for-loop.
        all_grad_i = jax.vmap(grad_row_nll, in_axes=(0, 0, 0, 0, 0, None))(
            w_mat, self.endog_j, self.exog_j['location'], self.exog_j['scale'], 
            self.exog_j['shape'], mle_params
        )

        # 4. Compute the "Meat" of the sandwich (B)
        W_i = jnp.sum(w_mat, axis=1) # Weight sum for each row
        g_i = W_i[:, None] * all_grad_i # Recover full gradient sum for each row
        
        # B = (1/W^2) * sum(outer(g_i, g_i))
        B = (1.0 / W**2) * jnp.einsum('ip,iq->pq', g_i, g_i)

        # 5. Assemble the sandwich: A_inv @ B @ A_inv
        A = H_bar
        A_inv = jnp.linalg.pinv(A)
            
        cov_matrix = A_inv @ B @ A_inv
        return cov_matrix


    def _fit_mle(self, start_params, optim_method,RL_args):
        """Performs Maximum Likelihood Estimation (MLE)."""
        i, j, k = self.len_exog
        
        if start_params is None:
            start_params = np.array(
                [self.location_guess] + ([0] * (i-1)) +
                [self.scale_guess] + ([0] * (j-1)) +
                [self.shape_guess] + ([0] * (k-1))
            )
        
        self.result = minimize(self.nloglike, start_params, method=optim_method)
        if self.result is None:
            raise ValueError(f"No results was outputed")

        # Handle failure case: maybe return NaNs or raise error
        if not self.result.success:
            raise ValueError(f"Optim issues can often stem from bad automatic starting parameters.\n"
              f"  Suggestion: Verify your `start_params` and/or provide manual initial guesses (especially intercepts) in the start_params attribute of the fit method.", RuntimeWarning)
        mle_params = self.result.x

        hessian_func = nd.Hessian(self.nloglike) 
        H_bar        = hessian_func(mle_params)  

        # ----- Build Godambe (sandwich) covariance -----------------------
        # We will:
        #   A = H_bar  (bread: Hessian of the *average* objective)
        #   B = (1/W^2) Σ_i g_i g_iᵀ,  where g_i = Σ_s w_{i,s} ∇ℓ_{i,s}(θ̂)
        # To avoid rewriting your log-pdf, we get per-observation gradients by
        # calling `self.nloglike` with a one-hot weight mask selecting row i.

        # 0) weights matrix with shape (n_obs, n_samples)
        if getattr(self, "weights", None) is None:
            w_mat = np.ones_like(self.endog, dtype=float)
        else:
            w_mat = np.asarray(self.weights, dtype=float)

        W = float(w_mat.sum())          # total weight mass over all cells
        p = mle_params.size
        A = H_bar                        # bread (already average-scale)
        B = np.zeros((p, p), dtype=float)

        # helper: returns a function θ ↦ average NLL of row `idx`
        def make_row_fun(idx: int):
            # one-hot mask that keeps original per-sample weights on row idx
            row_w = np.zeros_like(w_mat)
            row_w[idx, :] = w_mat[idx, :]
            def _f(theta):
                return self.nloglike(theta, weights=row_w)
            return _f

        # loop over observation rows
        for idx in range(self.n_obs):
            f_i    = make_row_fun(idx)           # scalar function of params
            grad_i = nd.Gradient(f_i)(mle_params)  # ∇ ( average NLL of row i )
            W_i    = float(w_mat[idx, :].sum())    # weight mass in that row

            # Recover g_i = Σ_s w_{i,s} ∇ℓ_{i,s}: grad_i is (1/W_i) * g_i for the average,
            # so multiply back by W_i
            g_i = W_i * grad_i                     # shape (p,)
            B  += (1.0 / W**2) * np.outer(g_i, g_i)

        # Invert A with a tiny ridge if needed
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            ridge = 1e-6 * np.eye(p)
            A_inv = np.linalg.inv(A + ridge)

        cov_matrix = A_inv @ B @ A_inv            # <-- robust Godambe covariance

        # (Optional) quick diagnostic: effective sample size
        n_eff = W**2 / float((w_mat**2).sum())

        fitted_loc = self.loc_link(np.einsum('nip,i->np', self.exog['location'], self.result.x[:i]))
        fitted_scale = self.scale_link(np.einsum('njp,j->np', self.exog['scale'], self.result.x[i:i+j]))
        fitted_shape = self.shape_link(np.einsum('nkp,k->np', self.exog['shape'], self.result.x[i+j:]))
        return GEVFit(
            gevSample = self,
            fitted_params=self.result.x,
            gev_params=(fitted_loc, fitted_scale, fitted_shape),
            n_ll=self.result.fun,
            cov_matrix=cov_matrix,
            jacobian=self.result.jac,
            CIs=self._compute_CIs_mle(cov_matrix, self.result.x, 0.95),
            fit_method='MLE',
            RL_args = RL_args
        )

    def _fit_profile(self, start_params, optim_method, RL_args,**kwargs):
        """Performs Profile Likelihood Estimation."""
        n = 10
        mle_model = self.fit(start_params, optim_method, fit_method='MLE')
        fitted_params = mle_model.fitted_params
        stds = np.sqrt(np.diag(mle_model.cov_matrix))
        all_profile_mles = np.empty((self.nparams, n))
        all_param_values = np.empty((self.nparams, n))
        args = [(param_idx, n, optim_method, fitted_params,stds) for param_idx in range(self.nparams)]
        start_time = time.perf_counter()
        with ProcessPoolExecutor() as executor:
            for param_idx, profile_mles, param_values in executor.map(self._optimize_profile_parallel, args):
                all_profile_mles[param_idx] = profile_mles
                all_param_values[param_idx] = param_values
        
        end_time = time.perf_counter()
        print(f"Execution Time: {end_time - start_time:.4f} seconds")

        i, j, _ = self.len_exog
        fitted_loc = self.loc_link(np.einsum('nip,i->np', self.exog['location'], fitted_params[:i]))
        fitted_scale = self.scale_link(np.einsum('njp,j->np', self.exog['scale'], fitted_params[i:i+j]))
        fitted_shape = self.shape_link(np.einsum('nkp,k->np', self.exog['shape'], fitted_params[i+j:]))
        
        return GEVFit(
            gevSample = self,
            fitted_params=fitted_params,
            gev_params=(fitted_loc, fitted_scale, fitted_shape),
            n_ll=mle_model.n_ll,
            cov_matrix=mle_model.cov_matrix,
            jacobian=mle_model.jacobian,
            CIs=self._compute_CIs_profile(all_param_values, all_profile_mles, fitted_params, 0.95),
            fit_method='Profile',
            RL_args = RL_args
        )


    def _compute_CIs_mle(self,cov_matrix,fitted_params,threshold):
        """
        Compute Confidence Intervals (CIs) for Maximum Likelihood Estimation (MLE)
        
        Parameters:
        - cov_matrix: Covariance matrix of the fitted parameters
        - fitted_params: Array of estimated parameters
        - threshold: Confidence level (e.g., 0.95 for 95% CI)
        
        Returns:
        - CIs: Array containing lower bound, upper bound, p-value, and z-score for each parameter
        """
        # Convert confidence level to critical z-score
        z_critical = norm.ppf(1 - (1 - threshold) / 2)  # Two-tailed z-score

        CIs = np.empty((len(fitted_params), 4))
        se = np.sqrt(np.diag(cov_matrix))  # Standard errors

        for i in range(len(fitted_params)):
            lower_bound = fitted_params[i] - z_critical * se[i]
            upper_bound = fitted_params[i] + z_critical * se[i]
            z_score = fitted_params[i] / se[i]
            p_value = 2 * (1 - norm.cdf(np.abs(z_score)))  # Two-tailed p-value

            CIs[i] = [lower_bound, upper_bound, p_value, z_score]

        return CIs

    def _compute_CIs_profile(self, all_param_values, all_profile_mles, fitted_params, threshold):
        CIs = np.empty((len(fitted_params),4))
        for i in range(len(fitted_params)):
            chi2_threshold   = chi2.ppf(threshold, df=1)/2
            free_params = self._generate_free_params(param_idx = i,fixed_param_values=fitted_params)
            cutoff = all_profile_mles[i] - all_profile_mles[i][np.argmin(all_profile_mles[i])]
            indices = np.where(cutoff<=chi2_threshold)[0]
            lower_bound = all_param_values[i][indices[0]]  # First valid theta value
            upper_bound = all_param_values[i][indices[-1]]

            nloglike_partial = partial(self.nloglike,forced_indices=i,forced_param_values=0.01)
            nll_0 = minimize(nloglike_partial, free_params, method='L-BFGS-B').fun
            #Expected error, the likelihood function was changed and I didn't added profile params yet.
            deviance = 2*(nll_0 - self.nloglike(free_params=fitted_params))
            p_value = chi2.sf(deviance, df=1)
            CIs[i]= [lower_bound,upper_bound,p_value,deviance]
        return CIs

# The fit object, this object serves to compare different fits, print the fitting summary, and produce qq plots as well as data plots. 
class GEVFit():
    def __init__(self, gevSample, fitted_params,gev_params,n_ll,cov_matrix,jacobian, CIs, fit_method, RL_args):
            # Extract relevant attributes from optimize_result and give them meaningful names
            self.gevSample = gevSample
            self.fitted_params = fitted_params
            self.gev_params = gev_params
            self.n_ll = n_ll
            self.cov_matrix = cov_matrix
            self.jacobian = jacobian
            self.nparams = self.fitted_params.size
            self.CIs = CIs
            self.fit_method = fit_method
            self.RL_args = RL_args
    def AIC(self):
        if self.fit_method.lower() != 'mle':
            warnings.warn(f"AIC is based on MLE estimation, not on '{self.fit_method}'.", UserWarning)
        return 2 * self.gevSample.nparams + 2 * (self.n_ll*1)

    def BIC(self):
        if self.fit_method.lower() != 'mle':
            warnings.warn(f"BIC is based on MLE estimation, not on '{self.fit_method}'.", UserWarning)
        return self.gevSample.nparams * np.log(self.gevSample.n_obs) + 2 * (self.n_ll*1)

    def SE(self):
        # Compute standard errors using the inverse Hessian matrix
        if self.cov_matrix is None:
            raise ValueError("Hessian matrix is not available.")
        if self.fit_method.lower() != 'mle':
            warnings.warn(f"SE is based on MLE estimation, not on '{self.fit_method}'.", UserWarning)
        cov_matrix = self.cov_matrix.todense() if hasattr(self.cov_matrix, 'todense') else self.cov_matrix
        se = np.sqrt(np.diag(cov_matrix))
        return se

    
    def deviance(self):
        return 0
    
    def return_level(self, confidence=0.95):
        """
        Internal factory to create the correct return level calculator.
        """
        if self.RL_args["reparam"]:
            return GEVReturnLevelReparam(gevFit=self, confidence=confidence)
        else:
            return GEVReturnLevel(gevFit=self, confidence=confidence)
            
    def get_return_levels(self, T, t=None, s=None, confidence=0.95):
        """
        Calculates and returns the return level data.

        This is a high-level wrapper that provides direct access to the numbers.
        """
        # 1. The factory method is now used internally.
        rl_calculator = self.return_level(confidence=confidence)

        # 2. It calls the calculator's method to get the data.
        z_p, ci_lower, ci_upper = rl_calculator.compute(T=T, t=t, s=s)

        return z_p, ci_lower, ci_upper
    

    def plot_return_levels(self, T, t=None, s=None, show_ci=True, **kwargs):
        """
        Generates and displays a return level plot.

        This is the simplest way to get a standard return level plot.
        """
        # 1. Use the internal factory to get the right calculator.
        rl_calculator = self.return_level()

        # 2. Call the plotting method from that calculator object.
        rl_calculator.time_plot(T=T, t=t, s=s, show_ci=show_ci, **kwargs)
        
    
    def __str__(self): 
        # Calculate fitted values, SE, z-scores, p-values, AIC, and BIC
        aic = self.AIC()
        bic = self.BIC()

        # Note clarification if the fit method is not MLE
        aic_bic_note = ""
        if self.fit_method.lower() != 'mle':
            aic_bic_note = "(AIC and BIC are based on MLE estimation)"
        
        score_note = "Z-score"
        if self.fit_method.lower() != 'mle':
            score_note = "Deviance"

        # Define column widths for uniform formatting
        col_widths = [10, 12, 12, 12, 25, 8]  # Adjust widths as needed

        # Create headers with vertical separators
        header = f"| {'Parameter':<{col_widths[0]}} | {'Estimate':<{col_widths[1]}} | {score_note:<{col_widths[2]}} | {'P>|'+score_note+'|':<{col_widths[3]}} | {'95% CI':<{col_widths[4]}} | {'Signif':<{col_widths[5]}} |"

        # Line separator based on column widths
        separator = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"

        # Start formatting result string
        result_str = "\n"
        result_str += "=" * len(separator) + "\n"
        result_str += "          EVT Results Summary       \n"
        result_str += "=" * len(separator) + "\n"
        result_str += f"Data dimensions : {self.gevSample.endog.shape}\n"
        result_str += f"AIC: {aic:.2f}  {aic_bic_note}\n"
        result_str += f"BIC: {bic:.2f}  \n\n"

        # If reparameterization is active, add a clarification line
        if self.RL_args["reparam"]:
            result_str += f"*Note: Location (μ) was reparameterized using return levels zp for a return period T = {self.RL_args.get('T', None) }\n\n"

        result_str += separator + "\n"
        result_str += header + "\n"
        result_str += separator + "\n"

        len_mu, len_sigma, len_xi = self.gevSample.len_exog
        param_names = [f"{'zp' if self.RL_args['reparam'] else 'μ'}_{n}" for n in range(len_mu)] + \
                    [f"σ_{n}" for n in range(len_sigma)] + \
                    [f"ξ_{n}" for n in range(len_xi)]

        # Iterate over parameters and format each row
        for i, param_name in enumerate(param_names):
            p_value = self.CIs[i][2]

            # Determine significance stars based on p-value
            if p_value < 0.001:
                signif = '***'
            elif p_value < 0.01:
                signif = '**'
            elif p_value < 0.05:
                signif = '*'
            else:
                signif = ''

            # Format confidence intervals and scores
            score = self.CIs[i][3]
            if abs(score) >= 1e6 or abs(score) < 1e-3:
                formatted_score = f"{score:.2e}"  # Scientific notation
            else:
                formatted_score = f"{score:<.2f}"  # Regular fixed-point formatting

            ci_str = f"({self.CIs[i][0]:.4f}, {self.CIs[i][1]:.4f})"

            # Format row with fixed column widths
            result_str += (f"| {param_name:<{col_widths[0]}} | {self.fitted_params[i]:<{col_widths[1]}.4f} "
                        f"| {formatted_score:<{col_widths[2]}} "
                        f"| {p_value:<{col_widths[3]}.4f} "
                        f"| {ci_str:<{col_widths[4]}} "
                        f"| {signif:<{col_widths[5]}} |\n")

        result_str += separator + "\n"
        result_str += "Significance : *** p<0.001, ** p<0.01, * p<0.05\n"

        return result_str

    
    def _gev_params_to_quantiles(self,p):
        """
        Computes quantiles for multiple GEV distributions given a probability p.

        This function maps GEV distribution parameters (location, scale, shape) to 
        corresponding quantile values using the inverse CDF (percent-point function).

        Parameters:
        p (float): Probability for which to compute the quantiles (0 < p < 1).

        Returns:
        np.ndarray: n x 1 array of quantiles for each GEV distribution.
        """

        fitted_loc, fitted_scale, fitted_shape = self.gev_params
        n = len(fitted_loc)  # Number of GEV fits (in time)
        quantiles = np.zeros((n, 1))  # Initialize as an n x 1 array

        for i in range(n):
            # Use scipy's genextreme.ppf to compute the quantile
            quantiles[i, 0] = genextreme.ppf(p, c=-fitted_shape[i], loc=fitted_loc[i], scale=fitted_scale[i]).item()

        return quantiles

class GEVReturnLevelBase(ABC):
    """
    Base class for GEV return level calculations.
    
    This class provides common functionality for calculating and summarizing return levels
    for Generalized Extreme Value (GEV) distribution fits.
    """
    def __init__(self, gevFit, confidence=0.95):
        """
        Initialize the GEV return level calculator.
        
        Args:
            gevFit: The GEV fit object containing model parameters and covariance matrix
            confidence (float, optional): Confidence level for interval estimation. Defaults to 0.95.
        """
        self.gevFit = gevFit
        self.confidence = confidence
    
    @abstractmethod
    def return_level_at(self, *args, **kwargs):
        """
        Abstract method to compute the return level at specific parameters.
        
        Must be implemented by subclasses with their specific parameter requirements.
        
        Returns:
            tuple: Return level estimate and confidence interval (lower, upper).
        """
        raise NotImplementedError("Subclasses must implement return_level_at method")
    
    @abstractmethod
    def compute(self, T, t=None, s=None):
        """
        Abstract method to compute return levels and confidence intervals
        over multiple parameter combinations.
        
        Must be implemented by subclasses with their specific parameter requirements.
        
        Returns:
            np.ndarray: Array of return level estimates and confidence intervals.
        """
        raise NotImplementedError("Subclasses must implement summary method")


class GEVReturnLevel(GEVReturnLevelBase):
    def __init__(self, gevFit, confidence=0.95):
        super().__init__(gevFit, confidence)

    def return_level_at(self,T, t, s):
        """
        Computes the return level of the Generalized Extreme Value (GEV) distribution for a given return period T.
        The confidence interval is estimated using the delta method.

        Args:
            T (float): Return period.
            t (int, optional): Reference year for non-stationary models. Defaults to None.
            s (int, optional): Series index (0-based, from 0 to p-1). Defaults to 1.
            confidence (float, optional): Confidence level for the interval. Defaults to 0.95.

        Returns:
            tuple: (standard error, (lower_ci, upper_ci)) where lower_ci and upper_ci are the confidence interval bounds.
        """
        # Validate return period
        if T <= 1:
            raise ValueError("Return period must be greater than 1.")

        # Get dimensions
        n, p = self.gevFit.gevSample.endog.shape  # n: observations, p: series
        if s < 0 or s >= p:
            raise ValueError(f"Series index s must be between 0 and {p-1}, got {s}.")

        # Get number of covariates for each parameter
        i, j, k = self.gevFit.gevSample.len_exog  # i: location, j: scale, k: shape

        # Extract fitted parameters
        params = self.gevFit.fitted_params  # Shape: (i + j + k,)

        # Determine effective time index
        if not self.gevFit.gevSample.trans:  # Stationary model
            t_eff = 0
            if t is not None:
                warnings.warn(
                    "Reference years are not required in a stationary model. Using t=0.",
                    UserWarning
                )
        else:  # Non-stationary model
            if t is None:
                raise ValueError("A reference year must be provided in a non-stationary model.")
            elif t >= n:
                raise ValueError(f"t must be less than {n}, got {t}.")
            t_eff = t

        # Extract covariates for series s at time t_eff
        X_mu = self.gevFit.gevSample.exog['location'][t_eff, :, s]  # Shape: (i,)
        X_sigma = self.gevFit.gevSample.exog['scale'][t_eff, :, s]  # Shape: (j,)
        X_xi = self.gevFit.gevSample.exog['shape'][t_eff, :, s]     # Shape: (k,)

        # Compute GEV parameters (assuming identity link functions)
        loc = np.dot(X_mu, params[:i])
        scale = np.dot(X_sigma, params[i:i+j])
        shape = np.dot(X_xi, params[i+j:])

        # Compute the return level (Coles, 2001)
        y_p = -np.log(1 - 1 / T)
        if np.isclose(shape, 0) or np.isclose(1 / T, 0):
            z_p = loc - scale * np.log(y_p)
        else:
            z_p = loc - (scale / shape) * (1 - y_p ** (-shape))

        # For gradient computation, use the linear predictors (identity link assumption)
        sigma_t = np.dot(X_sigma, params[i:i+j])
        xi_t = np.dot(X_xi, params[i+j:])

        # Compute derivatives
        if np.isclose(xi_t, 0):
            d_zp_d_mu = 1
            d_zp_d_sigma = -np.log(y_p)
            #d_zp_d_xi = 0 ?? 
            d_zp_d_xi = 0.5 * sigma_t * (np.log(y_p))**2
        else:
            temp = y_p ** (-xi_t)
            d_zp_d_mu = 1
            d_zp_d_sigma = -(1 - temp) / xi_t
            d_zp_d_xi = (sigma_t / xi_t**2) * (1 - temp) - (sigma_t / xi_t) * temp * np.log(y_p)

        # Construct gradient vector with respect to fitted parameters
        gradient = np.concatenate([d_zp_d_mu * X_mu, d_zp_d_sigma * X_sigma, d_zp_d_xi * X_xi])

        # Estimate standard error using the delta method
        variance = gradient @ self.gevFit.cov_matrix @ gradient.T
        std_error = np.sqrt(variance)

        # Compute confidence interval
        alpha = 1 - self.confidence
        z_crit = norm.ppf(1 - alpha / 2)
        ci_lower = z_p - z_crit * std_error
        ci_upper = z_p + z_crit * std_error

        return z_p, (ci_lower, ci_upper)
  
        
    def compute(self, T, t=None, s=None):
        """
        Computes return levels and confidence intervals over multiple return periods and time steps.
        
        Returns:
            tuple of np.ndarray: (z_p_array, ci_lower_array, ci_upper_array) each with shape (len_T, len_t, len_s)
            
        Raises:
            ValueError: If T, t, or confidence is None.
        """
        
        # Set default values for t and s if they are not provided
        if t is None:
            t = [self.gevFit.gevSample.endog.shape[0]-1]
        if s is None:
            s = list(range(self.gevFit.gevSample.endog.shape[1]))

        len_T = len(T)
        len_t = len(t)
        len_s = len(s)

        # Step 1: Define a helper function for parallel computation
        def compute_rl(T_val, t_val, s_val):
            z_p, ci = self.return_level_at(T=T_val, t=t_val, s=s_val)
            return z_p, ci[0], ci[1]

        # Step 2: Compute return levels and CIs in parallel
        results = Parallel(n_jobs=-1,backend='threading')(
            delayed(compute_rl)(T_val, t_val, s_val)
            for T_val in T for t_val in t for s_val in s
        )

        # Step 3: Reshape results and split into three arrays
        results_array = np.array(results, dtype=np.float32).reshape(len_T, len_t, len_s, 3)

        z_p_array = results_array[:, :, :, 0]
        ci_lower_array = results_array[:, :, :, 1]
        ci_upper_array = results_array[:, :, :, 2]

        return z_p_array, ci_lower_array, ci_upper_array
        

    def set_parameters(self, confidence=None, t=None, T=None):
        """
        Sets the summary parameters for return level computations.

        Args:
            confidence (float, optional): Confidence level for return level estimation.
            T (list or np.array, optional): Return period vector.
            t (list or np.array, optional): Time vector.
        """
        if confidence is not None:
            self.confidence = confidence
        if T is not None:
            self.T = T
        if t is not None:
            self.t = t

        if confidence is None and T is None and t is None:
            warnings.warn(
                "No parameters were set. Please specify at least one of `confidence`, `Return Period : T`, or `Reference time : t`.",
                UserWarning
            )

    def time_plot(
        self,
        T, 
        t=None, 
        s=None, 
        show_ci=True,
        save_path=None,
        dpi=300,
        title=None,
        ylabel=None,
        year_offset=1950,
        manual_params=None,          # <-- NEW
    ):
        """
        Plot return-level curves for each time point at a given spatial index *s*.

        [...]
        manual_params : array-like, shape (3, n) or (n, 3), optional
            One or more GEV parameter triples ``[μ, σ, ξ]`` to be drawn as
            dotted-red reference curves.  For example::

                manual_params = [[μ1, σ1, ξ1],
                                [μ2, σ2, ξ2]]

            If *None* (default), no reference curves are added.
        """

        if s is None:
            s_to_plot = 0  # Default to the very first series
        elif isinstance(s, list):
            if len(s) > 1:
             warnings.warn(f"Multiple spatial indices provided: {s}. Plotting the first one: {s[0]}")
            s_to_plot = s[0]
        else:
            s_to_plot = s # Assume it's an integer

        
        if t is None:
            t = list(range(self.gevFit.gevSample.endog.shape[0]))

        # 2. Calculate data ONLY for the single series we need to plot
        z_p_array, ci_lower_array, ci_upper_array = self.compute(T=T, t=t, s=[s_to_plot])
    
        # 3. Prepare variables for plotting
        T_vals = np.asarray(T)
        t_vals = np.asarray(t)
        s_idx = 0
        

        def _gev_return_level(mu, sigma, xi, T_vals):
            """
            Return level z(T) of a GEV distribution for the return period axis `T_vals`.
            """
            p = 1.0 / T_vals  # exceedance probability
            if np.isclose(xi, 0.0):
                # Gumbel limit (ξ → 0)
                return mu - sigma * np.log(-np.log(1.0 - p))
            else:
                return mu + (sigma / xi) * ((-np.log(1.0 - p)) ** (-xi) - 1.0)

        # ------------------------- Style -----------------------------------
        mpl.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.figsize": (10, 6),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.markersize": 6,
            "lines.linewidth": 2,
            "legend.frameon": False,
        })

        # -------------------------- Plotting -------------------------------
        fig, ax = plt.subplots()

        color_cycle = plt.cm.viridis(np.linspace(0, 1, len(t_vals)))
        markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']

        for i, (t_val, color) in enumerate(zip(t_vals, color_cycle)):
            return_levels = z_p_array[:, i, s_idx]
            legend_year   = int(t_val) + 0
            legend_label  = fr"$t = {legend_year}$"

            ax.plot(
                T_vals,
                return_levels,
                label=legend_label,
                color=color,
                marker=markers[i % len(markers)]
            )

            if show_ci:
                lower = ci_lower_array[:, i, s_idx]
                upper = ci_upper_array[:, i, s_idx]
                ax.fill_between(T_vals, lower, upper, color=color, alpha=0.15)

        # ------------- Optional manual reference lines (dotted red) --------
        if manual_params is not None:
            params = np.asarray(manual_params, dtype=float)
            params = np.atleast_2d(params)

            # If already (n, 3), keep it. If (3, n), flip it.
            if params.shape[1] == 3:
                pass  # (n, 3) — good
            elif params.shape[0] == 3:
                params = params.T  # (3, n) -> (n, 3)
            else:
                raise ValueError(
                "`manual_params` must be shape (n,3) or (3,n); got {params.shape}"
            )

            # Allow a single triple to be reused for all t’s
            if params.shape[0] == 1:
                params = np.repeat(params, len(t_vals), axis=0)

            # Otherwise require one triple per t
            if params.shape[0] != len(t_vals):
                raise ValueError(
                f"`manual_params` rows ({params.shape[0]}) must equal len(t) ({len(t_vals)}) "
                "or be a single triple."
            )

            for i, (t_val, color) in enumerate(zip(t_vals, color_cycle)):
                mu, sigma, xi = params[i]
                z_manual = _gev_return_level(mu, sigma, xi, T_vals)
                ax.plot(T_vals, z_manual, linestyle=':', linewidth=2, color=color,
                label=f"True, t={int(t_val)}")


        # ----------------------- Labels & title ----------------------------
        ax.set_xlabel("Return period (years)")
        ax.set_ylabel("Return level (mm/day)" if ylabel is None else ylabel)

        default_title = "Return-level curves for selected reference years"
        ax.set_title(default_title if title is None else title)

        # Legend outside the plot area
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Legend")
        fig.tight_layout()

        # ----------------------- Output ------------------------------------
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        else:
            plt.show()

class GEVReturnLevelReparam(GEVReturnLevelBase):
    def __init__(self, gevFit, confidence=0.95):
        super().__init__(gevFit, confidence)

    def return_level_at(self, t , s):
        """
        Computes the return level at a specific time point using the reparameterized approach.
        
        Args:
            t (int): Reference time point.
            
        Returns:
            tuple: First parameter estimate and confidence interval (lower, upper).
        """
        i,_,_ = self.gevFit.gevSample.len_exog
        zp_cov_matrix = self.gevFit.cov_matrix[:i,:i]
        X_zp_selected = self.gevFit.gevSample.exog['location'][t,:,s]
        se = np.sqrt(np.einsum('i,ij,j->', X_zp_selected, zp_cov_matrix, X_zp_selected))
        zp = np.einsum('i,i -> ',X_zp_selected,self.gevFit.fitted_params[:i])
        z_score = norm.ppf(1-(1-self.confidence)/2)
        lower_bound = zp - z_score * se
        upper_bound = zp + z_score * se
        return zp, (lower_bound.item(),upper_bound.item())
    
    def compute(self, t=None, s=None):
        """
        Computes return levels and confidence intervals over multiple time points and locations.
        
        Returns:
            np.ndarray: Array of return level estimates and confidence intervals.
        """

        if t is None:
            t = list(range(self.gevFit.gevSample.endog.shape[0]))
        if s is None:
            s = list(range(self.gevFit.gevSample.endog.shape[1]))

        len_t = len(t)
        len_s = len(s)

        # Prepare output array: shape (1, len_t, len_s, 6)
        rlArray = np.empty((1, len_t, len_s, 6), dtype=np.float32)

        # Step 1: Fill in T, t, s columns
        # Dimension 0 is fixed = 1; broadcast T, t, s into the correct slices
        rlArray[0, :, :, 0] = self.gevFit.RL_args["T"]
        rlArray[0, :, :, 1] = t[:, np.newaxis]
        rlArray[0, :, :, 2] = s[np.newaxis, :]

    # Step 2: Define a helper function for parallel (or serial) computation
        def compute_one(t_val, s_val):
            z_p, (ci_low, ci_up) = self.return_level_at(t_val, s_val)
            return (z_p, ci_low, ci_up)

    # Step 3: Compute return levels over all (t, s) combinations
        results = Parallel(n_jobs=-1)(
        delayed(compute_one)(t_val, s_val)
        for t_val in t for s_val in s)

        results_array = np.array(results, dtype=np.float32).reshape(len_t, len_s, 3)

        rlArray[0, :, :, 3] = results_array[:, :, 0]  # z_p
        rlArray[0, :, :, 4] = results_array[:, :, 1]  # lower
        rlArray[0, :, :, 5] = results_array[:, :, 2]  # upper

        return rlArray

if __name__ == "__main__":
    # Imports specific to this test
    from scipy.stats import genextreme
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # =============================================================================
    # GROUND TRUTH RECOVERY TEST (MULTI-SAMPLE)
    # =============================================================================
    print("========================================")
    print("  GEV MODEL: MULTI-SAMPLE RECOVERY TEST ")
    print("========================================")

    # 1. CONFIGURATION
    # ----------------
    N_OBS = 100         # 100 Time steps
    N_SAMPLES = 5       # 5 Independent Series (e.g., 5 locations or replicates)
    
    # Time covariate from 0 to 10
    time_covariate = np.linspace(0, 10, N_OBS) 
    
    # TRUE COEFFICIENTS
    # We assume all 5 samples follow the SAME trend (Pooling test)
    
    # Location = 100 + 2.5 * t
    TRUE_MU_INTERCEPT = 100.0
    TRUE_MU_SLOPE     = 2.5    
    
    # Scale = 15 + 0.5 * t
    TRUE_SIG_INTERCEPT = 38
    TRUE_SIG_SLOPE     = 2   
    
    # Shape = 0.15 (Constant)
    TRUE_XI = 0.15 

    print(f"Dimensions: {N_OBS} Observations x {N_SAMPLES} Samples")
    print(f"TRUE Model: Mu    = {TRUE_MU_INTERCEPT} + {TRUE_MU_SLOPE}*t")
    print(f"TRUE Model: Sigma = {TRUE_SIG_INTERCEPT} + {TRUE_SIG_SLOPE}*t")
    print(f"TRUE Model: Xi    = {TRUE_XI}")

    # 2. SIMULATION
    # -------------
    # Calculate the exact GEV parameters for every time step
    # These are vectors of shape (N_OBS,)
    mu_t = TRUE_MU_INTERCEPT + TRUE_MU_SLOPE * time_covariate
    sigma_t = TRUE_SIG_INTERCEPT + TRUE_SIG_SLOPE * time_covariate
    xi_t = np.full(N_OBS, TRUE_XI)
    
    # Generate random variable matrix (N_OBS, N_SAMPLES)
    rng = np.random.default_rng(42)
    endog_sim = np.zeros((N_OBS, N_SAMPLES))
    
    print("\nSimulating data...")
    for i in range(N_OBS):
        # Generate 5 random values for this specific time step
        # params mu_t[i], sigma_t[i] are shared across all 5 samples
        endog_sim[i, :] = genextreme.rvs(
            c=-xi_t[i], 
            loc=mu_t[i], 
            scale=sigma_t[i], 
            size=N_SAMPLES, 
            random_state=rng
        )
        
    # Create the Exogenous input
    # We pass a 1D array for time. The class will broadcast this to (N_OBS, 1, N_SAMPLES)
    exog_sim = {
        'location': time_covariate,
        'scale':    time_covariate,
        'shape':    None 
    }

    # 3. ESTIMATION
    # -------------
    print("Fitting Model (MLE with pooled samples)...")
    
    model = GEV(
        endog=endog_sim,
        exog=exog_sim,
    )
    
    fit_result = model.fit(optim_method='L-BFGS-B', fit_method='MLE')
    
    # 4. NUMERICAL COMPARISON
    # -----------------------
    est = fit_result.fitted_params
    
    true_params_vec = np.array([
        TRUE_MU_INTERCEPT, TRUE_MU_SLOPE,
        TRUE_SIG_INTERCEPT, TRUE_SIG_SLOPE,
        TRUE_XI
    ])
    
    print("\n--- PARAMETER RECOVERY (Pooled Fit) ---")
    print(f"{'Param':<15} | {'True':<10} | {'Estimated':<10} | {'Error %':<10}")
    print("-" * 55)
    
    param_names = ["Mu_Intercept", "Mu_Slope", "Sig_Intercept", "Sig_Slope", "Xi_Intercept"]
    
    for i, name in enumerate(param_names):
        err = abs((est[i] - true_params_vec[i]) / true_params_vec[i]) * 100
        print(f"{name:<15} | {true_params_vec[i]:<10.4f} | {est[i]:<10.4f} | {err:<10.2f}%")

    # 5. VISUAL COMPARISON
    # --------------------
    print("\nGenerating Plot for Sample 0 (Representative)...")
    
    # Indices to plot (Start, Middle, End)
    check_indices = [0, int(N_OBS/2), N_OBS-1]
    
    # Manual Truth for the dotted lines
    manual_truth = []
    for idx in check_indices:
        manual_truth.append([mu_t[idx], sigma_t[idx], xi_t[idx]])
    manual_truth = np.array(manual_truth)

    fit_result.plot_return_levels(
        T=np.logspace(0.1, 3, 100),
        t=check_indices,
        s=0, # Plot the first sample (since all 5 share the same parameters)
        manual_params=manual_truth, 
        title=f"Validation: {N_SAMPLES} Samples Pooled (N_OBS={N_OBS})",
        ylabel="Simulated Value"
    )
    
    print("Test Complete.")