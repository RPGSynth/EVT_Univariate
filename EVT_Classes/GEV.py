# Standard library imports
import os
import warnings

# Third-party library imports
import numpy as np
import pandas as pd
import xarray as xr
import numdifftools as nd
import statsmodels.api as sm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize, approx_fprime
from scipy.stats import norm, chi2, gumbel_r, genextreme
from statsmodels.base.model import GenericLikelihoodModel
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
from functools import partial
import time
from typing import Dict, Union, Optional, Callable, Any, List, Tuple
from abc import ABC, abstractmethod

class GEV(ABC):
    """
    A statistical model supporting GEV models.
    
    Attributes
    ----------
    exog_shape : array-like
        Exogenous variables for the shape parameter.
    exog_scale : array-like
        Exogenous variables for the scale parameter.
    exog_location : array-like
        Exogenous variables for the location parameter.
    model_type : str
        Specifies whether the model is 'GEV' or 'GPD'.
    """
    
    @staticmethod
    def identity(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def exp_link(x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    def __init__(
        self, 
        loc_link: Optional[Callable] = None,
        scale_link: Optional[Callable] = None,
        shape_link: Optional[Callable] = None):
        """
        Initializes the GEV model with specified parameters.
        
        Parameters
        ----------
        To be remade
        """
        self.loc_link = loc_link
        self.scale_link = scale_link
        self.shape_link = shape_link


    def nloglike(self, free_params, forced_index=-1,forced_param_value=0):
        """
        Computes the negative log-likelihood of the GEV model.
        """

        # Extract the number of covariates for each parameter
        i, j, _ = self.len_exog
        params = np.ones(self.nparams)
        free_index = 0

        for k in range(self.nparams):  # Total number of parameters
            if k == forced_index:
                params[k] = forced_param_value  # Use fixed parameter value
            else:
                params[k] = free_params[free_index] # Use optimized parameter
                free_index += 1
        
        # Compute location, scale, and shape parameters
        location = self.loc_link(np.dot(self.exog['location'], params[:i].reshape(-1,1)))
        scale = self.scale_link(np.dot(self.exog['scale'], params[i:i+j].reshape(-1,1)))
        shape = self.shape_link(np.dot(self.exog['shape'], params[i+j:].reshape(-1,1)))
        # GEV transformation
        normalized_data = (self.endog - location) / scale
        if np.allclose(shape, 0):  # Treat shape = 0 separately
            log_likelihood = (
                np.sum(np.log(scale))
                + np.sum(normalized_data)
                + np.sum(np.exp(-normalized_data))
            )
        else:
            # Standard GEV transformation
            transformed_data = 1 + shape * normalized_data

            # Return a large penalty for invalid parameter values
            if np.any(transformed_data <= 0) or np.any(scale <= 0):
                return 1e6

            log_likelihood = (
                np.sum(np.log(scale))
                + np.sum(transformed_data ** (-1 / shape))
                + np.sum(np.log(transformed_data) * (1 + 1 / shape))
            )

        return log_likelihood 

    def loglike(self,params):
        return -(self.nloglike(params))
    
    @abstractmethod
    def fit(self):
        """
        Fit a model to data.
        
        Returns
        -------
        Any
            Fitted model result.
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def return_levels(self):
        pass

class GEVSample(GEV):
    def __init__(self, endog, exog={'shape': None, 'scale': None, 'location': None}, loc_link=GEV.identity, scale_link=GEV.identity, shape_link=GEV.identity, **kwargs):
        """
        Initializes the GEV model for a data sample with specified parameters.
        
        Parameters
        ----------
        endog : array-like
            Endogenous variable. Can be a list, NumPy array, Pandas Series, or any array-like object.
        exog : array-like or dict, optional
            Exogenous variables for parameters. Can be:
            - A list, NumPy array, or Pandas DataFrame to be used for all parameters
            - A dictionary with keys 'shape', 'scale', and 'location', where values can be 
              lists, NumPy arrays, or Pandas Series.
        loc_link : callable, optional
            Link function for the location parameter. Defaults to identity.
        scale_link : callable, optional
            Link function for the scale parameter. Defaults to identity.
        shape_link : callable, optional
            Link function for the shape parameter. Defaults to identity.
        kwargs : dict
            Additional keyword arguments.
            
        Raises
        ------
        ValueError
            If endogenous data is empty or invalid, or if exogenous data shapes are inconsistent.
        TypeError
            If link functions are not callable or if inputs cannot be converted to appropriate types.
        """
        #----------Initiate common parameters-----------

        super().__init__(loc_link=loc_link,scale_link=scale_link,shape_link=shape_link)

        #----------Deal with Endog input-----------------
        # Exceptions

        ## 1 : Endog has a length and elements.
        if endog is None or (hasattr(endog, '__len__') and len(endog) == 0):
            raise ValueError("The `endog` parameter must not be None or empty. Please provide valid endogenous data.")

        
        ## 2 : Endog is a dataframe, series or other array like data and can be converted to numpy.
        try:
            # Handle pandas Series/DataFrame
            if isinstance(endog, (pd.Series, pd.DataFrame)):
                self.endog = endog.values
            # We assume it must be list
            else:
                self.endog = np.asarray(endog)
                
            # Ensure it's a float array and of shape (x,1)
            self.endog = self.endog.astype(float).reshape(-1, 1)
        except Exception as e:
            raise TypeError(f"Could not convert endogenous data to a numeric array: {str(e)}")


        ## 3 : The elements of endog are not Nan.
        if np.isnan(self.endog).any():
            raise ValueError("The `endog` parameter contains NaN values.")
        

        ## 4 : Link functions are callable if they are defined. 
        for name, func in [('loc_link', self.loc_link), ('scale_link', self.scale_link), ('shape_link', self.shape_link)]:
            if func is not None and not callable(func):
                raise TypeError(f"The `{name}` parameter must be callable or None.")


        # ------------------------
        #1 Setting class attributes

        ## 1 : Link functions attributes
        self.loc_link = self.loc_link or self.identity
        self.scale_link = self.scale_link or self.identity
        self.shape_link = self.shape_link or self.identity
        

        ## 2 : Data and fit attributes
        self.len_endog = len(self.endog)
        self.result = None
        self.scale_guess = np.sqrt(6 * np.var(endog)) / np.pi
        self.location_guess = np.mean(endog) - 0.57722 * (np.sqrt(6 * np.var(endog)) / np.pi)
        self.shape_guess = 0.1

        ##3 : Internal attributes
        self._forced_param_index = -1
        self._forced_param_value = 0

        #---------------------------
        # Handle exog data
        self._process_exog(exog, **kwargs)

    def _process_exog(self, 
        exog: Any = None
        ) -> Dict[str, np.ndarray]:
        """
        Processes and validates the exogenous data.

        Parameters
        ----------
        exog : any, optional
            Exogenous variables for parameters. Can be:
            - None: Use default exogenous variables (a single column of ones) for all parameters.
            - dict: A dictionary with keys 'shape', 'scale', 'location' mapping to exogenous variables for each parameter.
                Each value can be:
                - None: Use the default (a column of ones).
                - array-like: An array-like object with shape (n_samples,) or (n_samples, n_features).
            - array-like: An array-like object with shape (n_samples,) or (n_samples, n_features) to be used for all parameters.
        Returns
        -------
        dict
            Processed data dictionary with 'shape', 'scale', and 'location' keys.
            
        Raises
        ------
        ValueError
            If exogenous data has inconsistent shapes or invalid structure.
        TypeError
            If exogenous data cannot be converted to numeric arrays.
        """
        param_names = ['shape', 'scale', 'location']
        self.exog = {}


        # Case 1: No exogenous variables provided (use default ones arrays)
        if exog is None:
            for param in param_names:
                self.exog[param] = np.ones((self.len_endog, 1))

        # Case 2: Dictionary of exogenous variables
        elif isinstance(exog, dict):
            # Check for invalid keys
            invalid_keys = [key for key in exog.keys() if key not in param_names]
            if invalid_keys:
                raise ValueError(f"Invalid keys in exog dictionary: {invalid_keys}. "
                                f"Expected keys are: {param_names}")
            
            for param in param_names:
                param_exog = exog.get(param)
                
                if param_exog is None:
                    # Use default ones array
                    self.exog[param] = np.ones((self.len_endog, 1))
                else:
                    # Convert to numpy array supporting various input types
                    try:
                        if isinstance(param_exog, (pd.Series, pd.DataFrame)):
                            param_exog_array = param_exog.values
                        else:
                            param_exog_array = np.asarray(param_exog)
                            
                        param_exog_array = param_exog_array.astype(float)
                    except Exception as e:
                        raise TypeError(f"Could not convert exog['{param}'] to a numeric array: {str(e)}")
                    
                    # Validate shape
                    if param_exog_array.shape[0] != self.len_endog:
                        raise ValueError(
                            f"The number of rows in exog['{param}'] ({param_exog_array.shape[0]}) "
                            f"must match the number of samples in endog ({self.len_endog})."
                        )
                    
                    # Add constant column
                    if len(param_exog_array.shape) == 1:
                        self.exog[param] = np.concatenate([np.ones((self.len_endog, 1)), param_exog_array.reshape(-1, 1)], axis=1)
                    else:
                        self.exog[param] = np.concatenate([np.ones((self.len_endog, 1)), param_exog_array], axis=1)

        # Case 3 : Array like object           
        else:
            # Convert to numpy array supporting various input types
            try:
                # Handle pandas DataFrame
                if isinstance(exog, pd.DataFrame):
                    exog_array = exog.values
                elif isinstance(exog, pd.Series):
                    exog_array = exog.values.reshape(-1, 1)
                else:
                    exog_array = np.asarray(exog)

                # Ensure it's a float array
                exog_array = exog_array.astype(float)
            except Exception as e:
                raise TypeError(f"Could not convert exog to a numeric array: {str(e)}")

            # Validate shape
            if exog_array.shape[0] != self.len_endog:
                raise ValueError(
                    f"The length of the provided exog array ({exog_array.shape[0]}) "
                    f"must match the length of `endog` ({self.len_endog})."
                )

            if len(exog_array.shape) == 1:
                exog_augmented = np.concatenate([np.ones((self.len_endog, 1)), exog_array.reshape(-1,1)], axis=1)
            else:
                exog_augmented = np.concatenate([np.ones((self.len_endog, 1)), exog_array], axis=1)
            
            # Use the same exog array for all parameters
            for param in param_names:
                self.exog[param] = exog_augmented
        
        if all(np.array_equal(self.exog[key], np.ones((self.len_endog, 1))) for key in ['shape', 'scale', 'location']):
            self.trans = False
        else:
            self.trans = True

        self.len_exog = (self.exog['location'].shape[1],self.exog['scale'].shape[1],self.exog['shape'].shape[1])
        self.nparams = sum(self.len_exog)

    #Careful works on 1D par
    def hess(self, params=None):
        """
        Computes the Hessian matrix of the negative log-likelihood.
        Be careful that hessian_fn accepts only 1D arrays for params, reshaping will be necessary when called.
        """
        if params is not None:
            hessian_fn = nd.Hessian(self.nloglike)
            return hessian_fn(params)
        elif self.fitted:
            return np.linalg.inv(self.result.hess_inv.todense())
        else:
            raise ValueError("Model is not fitted. Cannot compute the Hessian at optimal parameters.")

    def hess_inv(self, params=None):
        """
        Computes the inverse of the Hessian matrix of the negative log-likelihood.
        """
        return np.linalg.inv(self.hess(params))

    def score(self, params=None):
        """
        Computes the score function (gradient of the log-likelihood).
        """
        if params is not None:
            return approx_fprime(params, self.nloglike, 1e-5)
        elif self.fitted:
            return self.result.jac
        else:
            raise ValueError("Model is not fitted. Cannot compute the score at optimal parameters.")

    def _generate_profile_params(self, param_idx, n,mle_params):
        """Generate profile parameter values based on param index rules."""
        i, j, k = self.len_exog  # Extract external indices
        
        param_ranges = {
            0: np.linspace(mle_params[0] / 1.5, mle_params[0] * 1.5, n),
            i: np.linspace(mle_params[i] / 1.5, mle_params[i] * 1.5, n),
            i + j: np.linspace(mle_params[i+j] - 0.2, mle_params[i+j] + 0.2, n)
        }
        return param_ranges.get(param_idx, np.linspace(mle_params[param_idx] / 1.5, mle_params[param_idx] * 1.5, n))

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
        param_idx,n,optim_method,mle_params = args
        profile_mles = np.empty(n)
        param_values = np.empty(n)

        profile_params = self._generate_profile_params(param_idx=param_idx,n=n,mle_params=mle_params)
        free_params = self._generate_free_params(param_idx=param_idx)
        for n_i, param_value in enumerate(profile_params):
                    nloglike_partial = partial(self.nloglike,forced_index=param_idx,forced_param_value=param_value)
                    result = minimize(nloglike_partial, free_params, method=optim_method)
                    profile_mles[n_i] = result.fun
                    param_values[n_i] = param_value
        
        return param_idx, profile_mles, param_values

    def fit(self, start_params=None, optim_method='L-BFGS-B', fit_method='MLE'):
        """
        Fits the model using the specified method (MLE or Profile).
        """
        if fit_method.lower() == 'mle':
            return self._fit_mle(start_params, optim_method)
        elif fit_method.lower() == 'profile':
            return self._fit_profile(start_params, optim_method)
        else:
            raise ValueError("Unsupported fit method. Choose 'MLE' or 'Profile'.")

    def _fit_mle(self, start_params, optim_method):
        """Performs Maximum Likelihood Estimation (MLE)."""
        i, j, k = self.len_exog
        
        if start_params is None:
            start_params = np.array(
                [self.location_guess] + ([0] * (i-1)) +
                [self.scale_guess] + ([0] * (j-1)) +
                [self.shape_guess] + ([0] * (k-1))
            )
        
        self.result = minimize(self.nloglike, start_params, method=optim_method)

        fitted_loc = self.loc_link(self.exog['location'] @ self.result.x[:i]).reshape(-1, 1)
        fitted_scale = self.scale_link(self.exog['scale'] @ self.result.x[i:i+j]).reshape(-1, 1)
        fitted_shape = self.shape_link(self.exog['shape'] @ self.result.x[i+j:]).reshape(-1, 1)

        return GEVFit(
            fitted_params=self.result.x,
            endog=self.endog,
            len_endog=self.len_endog,
            exog=self.exog,
            len_exog=self.len_exog,
            trans=self.trans,
            gev_params=(fitted_loc, fitted_scale, fitted_shape),
            log_likelihood=self.result.fun,
            cov_matrix=self.result.hess_inv,
            jacobian=self.result.jac,
            CIs=self._compute_CIs_mle(self.result.hess_inv.todense(), self.result.x, 0.95),
            fit_method='MLE'
        )

    def _fit_profile(self, start_params, optim_method, **kwargs):
        """Performs Profile Likelihood Estimation."""
        n = 1000
        mle_model = self.fit(start_params, optim_method, fit_method='MLE')
        fitted_params = mle_model.fitted_params
        
        all_profile_mles = np.empty((self.nparams, n))
        all_param_values = np.empty((self.nparams, n))
        args = [(param_idx, n, optim_method, fitted_params) for param_idx in range(self.nparams)]
        
        start_time = time.perf_counter()
        with ProcessPoolExecutor() as executor:
            for param_idx, profile_mles, param_values in executor.map(self._optimize_profile_parallel, args):
                all_profile_mles[param_idx] = profile_mles
                all_param_values[param_idx] = param_values
        
        end_time = time.perf_counter()
        print(f"Execution Time: {end_time - start_time:.4f} seconds")

        i, j, k = self.len_exog
        fitted_loc = self.loc_link(self.exog['location'] @ fitted_params[:i]).reshape(-1, 1)
        fitted_scale = self.scale_link(self.exog['scale'] @ fitted_params[i:i+j]).reshape(-1, 1)
        fitted_shape = self.shape_link(self.exog['shape'] @ fitted_params[i+j:]).reshape(-1, 1)
        
        return GEVFit(
            fitted_params=fitted_params,
            endog=self.endog,
            len_endog=self.len_endog,
            exog=self.exog,
            len_exog=self.len_exog,
            trans=self.trans,
            gev_params=(fitted_loc, fitted_scale, fitted_shape),
            log_likelihood=mle_model.log_likelihood,
            cov_matrix=mle_model.cov_matrix,
            jacobian=mle_model.jacobian,
            CIs=self._compute_CIs_profile(all_param_values, all_profile_mles, fitted_params, 0.95),
            fit_method='Profile'
        )


    def _compute_CIs_mle(self,cov_matrix,fitted_params,treshold):
        CIs = np.empty((len(fitted_params),4))
        se =  np.sqrt(np.diag(cov_matrix))
        for i in range(len(fitted_params)):
            lower_bound = fitted_params[i] - 1.96 * se[i]
            upper_bound = fitted_params[i] + 1.96 * se[i]
            z_score = fitted_params[i] / se[i]
            p_value = 2 * (1 - norm.cdf(np.abs(z_score)))
            CIs[i][0],CIs[i][1],CIs[i][2],CIs[i][3] = (lower_bound,upper_bound,p_value,z_score)
        return CIs

    def _compute_CIs_profile(self, all_param_values, all_profile_mles, fitted_params, treshold):
        CIs = np.empty((len(fitted_params),4))
        for i in range(len(fitted_params)):
            chi2_threshold   = chi2.ppf(treshold, df=1)/2
            free_params = self._generate_free_params(param_idx = i,fixed_param_values=fitted_params)
            cutoff = all_profile_mles[i] - all_profile_mles[i][np.argmin(all_profile_mles[i])]
            indices = np.where(cutoff<=chi2_threshold)[0]
            lower_bound = all_param_values[i][indices[0]]  # First valid theta value
            upper_bound = all_param_values[i][indices[-1]]

            deviance = 2*(self.nloglike(free_params=free_params,forced_index=i,forced_param_value=0.0001) - self.nloglike(free_params=fitted_params))
            p_value = chi2.sf(deviance, df=1)
            CIs[i][0],CIs[i][1],CIs[i][2],CIs[i][3] = (lower_bound,upper_bound,p_value,deviance)
        return CIs

    def _compute_confidence_interval(self, params, cov_matrix, compute_z_p, z_p, confidence=0.95):
        """
        Computes the confidence interval using the delta method.

        Parameters:
        - params: ndarray, model parameters
        - compute_z_p: function, function to compute z_p
        - cov_matrix: ndarray, covariance matrix
        - z_p: ndarray, point estimate
        - confidence: float, confidence level (default: 0.95)

        Returns:
        - z_p[0]: float, point estimate
        - (ci_lower, ci_upper): tuple, confidence interval
        """
        eps = np.sqrt(np.finfo(float).eps)
        gradient = approx_fprime(params.flatten(), compute_z_p, eps).flatten()

        # Estimate standard error using the delta method
        variance = gradient @ cov_matrix @ gradient.T
        std_error = np.sqrt(variance)

        # Compute confidence interval
        alpha = 1 - confidence
        z_crit = norm.ppf(1 - alpha / 2)
        ci_lower = z_p - z_crit * std_error
        ci_upper = z_p + z_crit * std_error

        return z_p[0], (ci_lower[0], ci_upper[0])

    def return_level(self, gevFit, return_period, confidence=0.95, ref_year=None):
        """
        Computes the return level of the Generalized Extreme Value (GEV) distribution for a given return period T.
        If method is set to "delta", the confidence interval is estimated using the delta method.

        Args:
            T (float): Return period.
            shape (float): Shape parameter (ξ) of the GEV distribution.
            location (float): Location parameter (μ) of the GEV distribution.
            scale (float): Scale parameter (σ) of the GEV distribution.
            cov_matrix (np.ndarray, optional): Covariance matrix of the parameter estimates. Required for the delta method.
            method (str, optional): Method for confidence interval estimation. Defaults to "delta".
            confidence (float, optional): Confidence level for the interval. Defaults to 0.95.

        Returns:
            tuple: Return level estimate and confidence interval (lower, upper).
        """
        if return_period <= 1:
            raise ValueError("Return period must be greater than 1.")
        
        if not gevFit.trans:
            if ref_year is not None:
                warnings.warn(
                    "Reference years are not required in a stationary model. Each year has the same return level for a given return period.",
                    UserWarning
                )
            loc, scale, shape = gevFit.gev_params[0][0], gevFit.gev_params[1][0], gevFit.gev_params[2][0]
        else:
            if ref_year is None:
                raise ValueError(
                    "A reference year must be provided in a non-stationary model since return levels vary over time for a given return period."
                )
            elif ref_year >= self.len_endog:
                raise ValueError(
                    "You can not get the future return levels but only present or past return levels."
                )
            loc, scale, shape = gevFit.gev_params[0][ref_year], gevFit.gev_params[1][ref_year], gevFit.gev_params[2][ref_year]

        
         # Compute the return level (Coles, 2001)
        y_p = -np.log(1 - 1/return_period)
        if shape == 0 or np.isclose(1/return_period, 0):
            z_p = loc - scale * np.log(y_p)
        else:
            z_p = loc - (scale / shape) * (1 - y_p**(-shape))

        if not self.trans:
            def compute_z_p(params):
                loc, scale, shape = params
                y_p = -np.log(1 - 1/return_period)
                if shape == 0 or np.isclose(1/return_period, 0):
                    return loc - scale * np.log(y_p)
                else:
                    return loc - (scale / shape) * (1 - y_p**(-shape))

            params = np.array([loc, scale, shape])

        else:
            i, j, k = self.len_exog
            def compute_z_p(params):
                loc = np.dot(self.exog['location'][ref_year][0:i],params[0:i])
                scale = np.dot(self.exog['scale'][ref_year][0:j],params[i:i+j])
                shape = np.dot(self.exog['shape'][ref_year][0:k],params[i+j:])
                y_p = -np.log(1 - 1/return_period)
                if shape == 0 or np.isclose(1/return_period, 0):
                    return loc - scale * np.log(y_p)
                else:
                    return loc - (scale / shape) * (1 - y_p**(-shape))

            params = gevFit.fitted_params
            
        return(self._compute_confidence_interval(params=params,cov_matrix=gevFit.cov_matrix.todense(),compute_z_p=compute_z_p,z_p=z_p,confidence=confidence))

    def return_levels(self, gevFit, confidence=0.95, N=30):
        T = np.arange(10, N, 10)  # Generate return periods from 10 to <N in steps of 10
        T = np.insert(T, 0, 2)   # Prepend 2
        len_T = len(T)
        rlArray = np.empty((self.len_endog, len_T, 4), dtype=np.float16)
        for t in range(self.len_endog):
            for n in range(len_T):
                z_p, ci = self.return_level(gevFit=gevFit, return_period=T[n], ref_year=t, confidence=confidence)
                rlArray[t, n, 1] = z_p
                rlArray[t, n, 2] = ci[0]
                rlArray[t, n, 3] = ci[1]
        rlArray[:, :, 0] = T      # Assign return periods to first column
        return rlArray

# The fit object, this object serves to compare different fits, print the fitting summary, and produce qq plots as well as data plots. 
class GEVFit():
    def __init__(self, fitted_params, endog, len_endog, exog, len_exog, trans,gev_params,log_likelihood,cov_matrix,jacobian, CIs, fit_method):
            # Extract relevant attributes from optimize_result and give them meaningful names
            self.fitted_params = fitted_params
            self.endog = endog
            self.len_endog = len_endog
            self.exog = exog
            self.len_exog = len_exog
            self.trans = trans
            self.gev_params = gev_params
            self.log_likelihood = log_likelihood
            self.cov_matrix = cov_matrix
            self.jacobian = jacobian
            self.nparams = self.fitted_params.size
            self.CIs = CIs
            self.fit_method = fit_method
    def AIC(self):
        if self.fit_method.lower() != 'mle':
            warnings.warn(f"AIC is based on MLE estimation, not on '{self.fit_method}'.", UserWarning)
        return 2 * self.nparams + 2 * self.log_likelihood

    def BIC(self):
        if self.fit_method.lower() != 'mle':
            warnings.warn(f"BIC is based on MLE estimation, not on '{self.fit_method}'.", UserWarning)
        return self.nparams * np.log(self.len_endog) + 2 * self.log_likelihood

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
        result_str += f"AIC: {aic:.2f}  {aic_bic_note}\n"
        result_str += f"BIC: {bic:.2f}  \n\n"
        result_str += separator + "\n"
        result_str += header + "\n"
        result_str += separator + "\n"

        len_mu, len_sigma, len_xi = self.len_exog
        param_names = [f"μ_{n}" for n in range(len_mu)] + \
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
        result_str += "Notes: *** p<0.001, ** p<0.01, * p<0.05\n"

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

    #To be remade
    def data_plot(self, time, toggle=None):
        """
        Creates a scatter plot of self.endog against the given time vector. Optionally overlays
        quantile lines at 0.05, 0.95 (green, bold) and 0.01, 0.99 (blue, dashed) based on `self.trans` 
        or the `toggle` parameter.

        Parameters:
        - time: Array-like, representing the x-axis values (time points).
        - toggle (bool, optional): If provided, overrides the self.trans attribute to toggle quantile lines.
                                If None, falls back to using self.trans.
        """
        if len(time) != len(self.endog):
            raise ValueError("Length of time vector must match the length of self.endog.")

        # Determine whether to show quantile lines
        show_quantiles = toggle if toggle is not None else self.trans

        # Configure matplotlib for a modern font
        rcParams['font.family'] = 'DejaVu Sans'
        rcParams['font.size'] = 14

        # Define a consistent, aesthetically pleasing color for all points
        point_color = '#76C7C0'

        # Create the scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(time, self.endog, color=point_color, alpha=0.8, edgecolors='black', linewidth=0.5)

        # Plot quantile lines if toggle is True or self.trans is True
        if show_quantiles:
            # Compute quantiles
            q_05 = self._gev_params_to_quantiles()(0.05)
            q_95 = self._gev_params_to_quantiles()(0.95)
            q_01 = self._gev_params_to_quantiles()(0.01)
            q_99 = self._gev_params_to_quantiles()(0.99)

            # Add quantile lines
            plt.plot(time, q_05.flatten(), color='green', linewidth=2, linestyle='-', label='5th & 95th Percentiles')
            plt.plot(time, q_95.flatten(), color='green', linewidth=2, linestyle='-')
            plt.plot(time, q_01.flatten(), color='blue', linewidth=2, linestyle='--', label='1st & 99th Percentiles')
            plt.plot(time, q_99.flatten(), color='blue', linewidth=2, linestyle='--')

            # Add legend inside the plot
            plt.legend(
                fontsize=12,
                loc='upper left',
                frameon=True,
                facecolor='white',
                edgecolor='black',
                title="Quantile Lines"
            )

        # Add plot aesthetics
        plt.title("Block Maximum Over Time with GEV Quantiles", fontsize=18, pad=15, fontweight='normal', color='darkslategray')
        plt.xlabel("Time", fontsize=16, labelpad=10)
        plt.ylabel("Values", fontsize=16, labelpad=10)

        # Gridlines with light styling
        plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.6)

        # Custom ticks
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        # Tight layout to maximize use of space
        plt.tight_layout()

        # Show the plot
        plt.show()
class GEVReturnLevel():
    def __init__(self,GEVFit,rlArray):
        self.rlArray = rlArray
        self.GEVFit = GEVFit

    def get_returnLevel(self,returnPeriod,t):
        return 0
    
class GEV_WWA(GEV):
    def __init__(self, endog, exog=None, loc_link=None, scale_link=None, shape_link=None, **kwargs):
        super().__init__(endog, exog, loc_link, scale_link, shape_link, **kwargs)

    def _process_data(self, endog, exog=None, **kwargs):
        """
        Processes and validates the endogenous and exogenous data.

        Parameters
        ----------
        endog : array-like
            Endogenous variable.
        exog : dict or array-like, optional
            Exogenous variables for parameters. Can be a single array (applied to all) or a dict with keys
            'shape', 'location'.
        kwargs : dict
            Additional arguments for data handling.
            
        Returns
        -------
        dict
            Processed data for internal use or external reference.
        """
        if exog is None:
            raise ValueError(
                "The WWA model requires exogenous data. Compatible formats are dictionaries or numpy arrays."
            )

        elif isinstance(exog, np.ndarray):
            if exog.shape[0] != self.len_endog:
                raise ValueError(
                    f"The length of the provided exog array ({exog.shape[0]}) must match the length of `endog` ({self.len_endog})."
                )

            if len(exog.shape) == 1:
                exog_augmented = exog.reshape(-1, 1)
            else:
                exog_augmented = exog

            self.exog = {
                "shape": np.ones((self.len_endog, 1)),
                "scale": exog_augmented,
                "location": exog_augmented,
            }

        elif isinstance(exog, dict):
            self.exog = {}

            if exog.get("shape") is not None:
                raise ValueError("The WWA model does not allow a non-stationary shape.")

            for key in ["shape", "scale", "location"]:
                value = exog.get(key)

                if value is None:
                    self.exog[key] = np.ones((self.len_endog, 1))
                else:
                    value_array = np.asarray(value)
                    if value_array.shape[0] != self.len_endog:
                        raise ValueError(
                            f"The number of rows in exog['{key}'] ({value_array.shape[0]}) "
                            f"must match the number of rows in `endog` ({self.len_endog})."
                        )
                    if len(value_array.shape) == 1:
                        self.exog[key] = value_array.reshape(-1, 1)
                    else:
                        self.exog[key] = value_array

            # Check if 'scale' and 'location' have the same size
            if self.exog["scale"].shape != self.exog["location"].shape:
                raise ValueError(
                    f"The WWA model requires 'scale' and 'location' to have the same shape, "
                    f"but got {self.exog['scale'].shape} and {self.exog['location'].shape}."
                )

        else:
            raise ValueError(
                "`exog` must be either a dictionary (default) or a NumPy array of shape (n, >=1)."
            )

        # Determine if transformations are needed
        if all(
            np.array_equal(self.exog[key], np.ones((self.len_endog, 1)))
            for key in ["shape", "scale", "location"]
        ):
            raise ValueError("The WWA model requires exogenous data for the location and the scale. Compatible formats are dictionaries or numpy arrays.")
        else:
            self.trans = True
        
        self.endog = np.asarray(endog).reshape(-1, 1)
        self.len_exog = (self.exog['location'].shape[1],self.exog['scale'].shape[1],self.exog['shape'].shape[1])

class GEV_WWA_Likelihood(GEV_WWA):
    def __init__(self, endog, loc_link=GEV.exp_link, scale_link=GEV.exp_link, shape_link=GEV.identity, exog={'shape': None, 'scale': None, 'location': None}, **kwargs):
        """
        Initializes the GEVSample model with given parameters.
        """
        super().__init__(endog=endog, exog=exog, loc_link=loc_link, scale_link=scale_link, shape_link=shape_link, **kwargs)
        model_0 = GEVSample(endog=self.endog,exog={}).fit()
        self.mu_0 = model_0.gev_params[0][0].item()
        self.sigma_0 = model_0.gev_params[1][0].item()
        
    def nloglike(self, params):
        """
        Computes the negative log-likelihood of the GEV model.
        """
        # Extract the number of covariates for each parameter
        i,_,_ = self.len_exog
        # Compute location, scale, and shape parameters
        location = self.mu_0 * self.loc_link(np.dot(self.exog['location'], params[:i].reshape(-1,1)) / self.mu_0)
        scale = self.sigma_0 * self.scale_link(np.dot(self.exog['scale'], params[:i].reshape(-1,1)) / self.mu_0)
        shape = self.shape_link(np.dot(self.exog['shape'],  params[i:].reshape(-1,1)))
        # GEV transformation
        normalized_data = (self.endog - location) / scale
        transformed_data = 1 + shape * normalized_data
        # Return a large penalty for invalid parameter values
        if np.any(transformed_data <= 0) or np.any(scale <= 0) or np.any(shape>=0.4):
            return 1e6

        return np.sum(np.log(scale)) + np.sum(transformed_data ** (-1 / shape)) + np.sum(np.log(transformed_data) * (1 / shape + 1))

    def fit(self, start_params=None, method='L-BFGS-B', **kwargs):
        """
        Fits the model using maximum likelihood estimation.

        Parameters
        ----------
        start_params : array_like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is an array of zeros.
        method : str, optional
            The `method` determines which solver from `scipy.optimize`
            is used, and it can be chosen from among the following strings:

            - 'newton' for Newton-Raphson, 'nm' for Nelder-Mead
            - 'bfgs' for Broyden-Fletcher-Goldfarb-Shanno (BFGS)
            - 'lbfgs' for limited-memory BFGS with optional box constraints
            - 'powell' for modified Powell's method
            - 'cg' for conjugate gradient
            - 'ncg' for Newton-conjugate gradient
            - 'basinhopping' for global basin-hopping solver
            - 'minimize' for generic wrapper of scipy minimize (BFGS by default)
        """
        i,_,_ = self.len_exog

        if start_params is None:
            start_params = np.array(
            [1]*(i) +
            [self.shape_guess] 
            )
        
        self.fitted = True
        self.result = minimize(self.nloglike, start_params, method=method, **kwargs)

        fitted_loc = self.mu_0 * self.loc_link(self.exog['location'] @ self.result.x[:i] / self.mu_0).reshape(-1,1)
        fitted_scale = self.sigma_0 * self.scale_link(self.exog['scale'] @ self.result.x[:i] / self.mu_0).reshape(-1,1)
        fitted_shape = self.shape_link(self.exog['shape'] @ self.result.x[i:]).reshape(-1,1)

        return GEV_WWA_Fit(
            optimize_result=self.result,
            endog=self.endog,
            len_endog=len(self.endog),
            exog=self.exog,
            len_exog=(self.exog['location'].shape[1],self.exog['scale'].shape[1],self.exog['shape'].shape[1]),
            trans=self.trans,
            gev_params = (fitted_loc,fitted_scale,fitted_shape),
            mu0 = self.mu_0,
            sigma0 = self.sigma_0
        )
    
class GEV_WWA_Fit(GEVFit):
    def __init__(self, optimize_result, endog, len_endog, exog, len_exog, trans, gev_params, mu0, sigma0):
        super().__init__(optimize_result, endog, len_endog, exog, len_exog, trans, gev_params)
        self.mu0 = mu0
        self.sigma0 = sigma0
        

    #Override
    def return_level(self, return_period, method="delta", confidence=0.95, ref_year=None):
        if ref_year is None:
            raise ValueError(
                "A reference year must be provided in a non-stationary model since return levels vary over time for a given return period.")
        elif ref_year >= self.len_endog:
            raise ValueError(
                "You can not get the future return levels but only present or past return levels.")

        loc, scale, shape = self.gev_params[0][ref_year], self.gev_params[1][ref_year], self.gev_params[2][ref_year]
        y_p = -np.log(1 - 1/return_period)
        if shape == 0 or np.isclose(1/return_period, 0):
            z_p = loc - scale * np.log(y_p)
        else:
            z_p = loc - (scale / shape) * (1 - y_p**(-shape))

        def compute_z_p(params):
            loc = self.mu0 * np.exp(np.dot(self.exog['location'][ref_year][0], params[0]).item() / self.mu0)
            scale = self.sigma0 * np.exp(np.dot(self.exog['scale'][ref_year][0], params[0]).item() / self.mu0)
            #To change for larger WWA model. 
            shape = params[1]
            y_p = -np.log(1 - 1/return_period)
            if shape == 0 or np.isclose(1/return_period, 0):
                return loc - scale * np.log(y_p)
            else:
                return loc - (scale / shape) * (1 - y_p**(-shape))
        return(self._compute_confidence_interval(params=self.fitted_params,cov_matrix=self.cov_matrix.todense(),compute_z_p=compute_z_p,z_p=z_p,confidence=confidence))


    
if __name__ == "__main__":
    EOBS = pd.read_csv(r"c:\ThesisData\EOBS\Blockmax\blockmax_temp.csv")

    EOBS["random_value"] = np.random.uniform(-2, 2, size=len(EOBS))
    #n = len(EOBS["prmax"].values.reshape(-1,1))
    #
    # Dummy endog variable (10 samples)
    
    exog = {"location" : EOBS[["tempanomalyMean","random_value"]], "scale" :  EOBS[["tempanomalyMean","random_value"]]}
    #model = GEVSample(endog=EOBS["prmax"].values.reshape(-1,1),exog=exog)
    #gev_result_1 = model.fit()
    #gev_result_1.probability_plot()
    #gev_result_1.data_plot(EOBS["year"].values)

    #test = GEV_WWA_Likelihood(endog=EOBS["prmax"].values.reshape(-1,1),exog=exog).fit()
    #test.data_plot(time=EOBS["year"])

    a1 = GEVSample(endog=EOBS["prmax"].values.reshape(-1,1),exog=exog)
    print(a1.return_levels(gevFit=a1.fit(fit_method='mle'))[30][0])
    #a1 = GEVSample(endog=EOBS["prmax"].values.reshape(-1,1))
    #print(a1.len_exog)


    #a2 = GEVSample(endog=EOBS["prmax"].values.reshape(-1,1),exog=exog).fit()
    #a1.data_plot(time=EOBS["year"])

    #e = a1.return_level(return_period=5,ref_year=74)
    #a1.return_level_plot()
