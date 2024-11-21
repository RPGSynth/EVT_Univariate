import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
import pandas as pd
import numdifftools as nd
import statsmodels.api as sm
from scipy.optimize import approx_fprime
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import gumbel_r

class GEV:
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
    def identity(x):
        return x

    @staticmethod
    def exp_link(x):
        return np.exp(x)

    def __init__(self, endog, exog=None, loc_link=None, scale_link=None, shape_link=None, **kwargs):
        """
        Initializes the GEV model with specified parameters.
        
        Parameters
        ----------
        endog : array-like
            Endogenous variable.
        exog : dict of array-like, optional
            Dictionary where keys are 'shape', 'scale', and 'location', specifying exogenous variables for each parameter.
        loc_link : callable, optional
            Link function for the location parameter. Defaults to identity.
        scale_link : callable, optional
            Link function for the scale parameter. Defaults to identity.
        shape_link : callable, optional
            Link function for the shape parameter. Defaults to identity.
        kwargs : dict
            Additional keyword arguments.
        """
        if endog is None or len(endog) == 0:
            raise ValueError("The `endog` parameter must not be None or empty. Please provide valid endogenous data.")
        
        # Store attributes
        self.nobs = len(endog)
        self.result = None
        self.fitted = False
        self.scale_guess = np.sqrt(6 * np.var(endog)) / np.pi
        self.shape_guess = 0.1
        self.loc_link = loc_link or self.identity
        self.scale_link = scale_link or self.identity
        self.shape_link = shape_link or self.identity

        # Handle data
        self.data = self._process_data(endog, exog, **kwargs)

    def _process_data(self, endog, exog=None, **kwargs):
        """
        Processes and validates the endogenous and exogenous data.

        Parameters
        ----------
        endog : array-like
            Endogenous variable.
        exog : dict or array-like, optional
            Exogenous variables for parameters. Can be a single array (applied to all) or a dict with keys
            'shape', 'scale', and 'location'.
        kwargs : dict
            Additional arguments for data handling.
            
        Returns
        -------
        dict
            Processed data for internal use or external reference.
        """
        # Convert `endog` to numpy array if needed
        endog = np.asarray(endog)

        exog_dict = {'shape': None, 'scale': None, 'location': None}
        # Initialize exog dictionary
        if exog is not None:
            if isinstance(exog, dict):
                for param in exog_dict.keys():
                    if param in exog and exog[param] is not None:
                        exog_array = np.asarray(exog[param])
                        if len(exog_array) != len(endog):
                            raise ValueError(
                                f"The length of exog['{param}'] ({len(exog_array)}) "
                                f"must match the length of `endog` ({len(endog)})."
                            )
                        exog_dict[param] = exog_array
                    else:
                        exog_dict[param] = np.ones_like(endog).reshape(-1,1)
            else:
                # If a single exog array is provided, apply it to all parameters
                exog_array = np.asarray(exog)
                if len(exog_array) != len(endog):
                    raise ValueError(
                        f"The length of the provided exog array ({len(exog_array)}) "
                        f"must match the length of `endog` ({len(endog)})."
                    )
                exog_dict = {key: exog_array for key in exog_dict}
        else:
            # If exog is None, replace all three parameters with vectors of ones of the same length as endog
            ones_array = np.ones(len(endog)).reshape(-1,1)
            exog_dict = {key: ones_array for key in exog_dict}

        # Set instance attributes for exogenous variables
        self.endog = endog
        self.exog_shape = exog_dict['shape']
        self.exog_scale = exog_dict['scale']
        self.exog_location = exog_dict['location']
        def is_trans(vals):
            # Check if the shape is greater than (1,): implies it may contain useful information beyond default values
            if vals.shape[1] > 1:
                return True
            # Check if the parameter is different from a vector of ones (only if shape condition is not met)
            default_ones = np.ones(len(endog)).reshape(-1,1)
            if not np.array_equal(vals, default_ones):
                return True
            return False
        self.trans = any(is_trans(v) for v in exog_dict.values())

        # Attach remaining kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        return {
            'endog': endog,
            'exog': exog_dict
        }

    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

    def predict(self, params, exog=None, *args, **kwargs):
        """
        Predict fitted values after model fitting.

        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError
    
class GEVLikelihood(GEV):
    def __init__(self, endog, loc_link=GEV.identity, scale_link=GEV.identity, shape_link=GEV.identity, exog={}, **kwargs):
        """
        Initializes the GEVLikelihood model with given parameters.
        """
        super().__init__(endog=endog, exog=exog, loc_link=loc_link, scale_link=scale_link, shape_link=shape_link, **kwargs)

    def loglike(self, params):
        """
        Computes the log-likelihood of the model.
        """
        return -(self.nloglike(params))

    def hess(self, params=None):
        """
        Computes the Hessian matrix of the negative log-likelihood.
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

    def nloglike(self, params):
        """
        Computes the negative log-likelihood of the GEV model.
        """
        # Extract the number of covariates for each parameter
        x1 = self.exog_location.shape[1] if self.exog_location is not None else 0
        x2 = self.exog_scale.shape[1] if self.exog_scale is not None else 0
        x3 = self.exog_shape.shape[1] if self.exog_shape is not None else 0

        # Compute location, scale, and shape parameters
        location = self.loc_link(np.dot(self.exog_location, params[:x1])).reshape(-1,1)
        scale = self.scale_link(np.dot(self.exog_scale, params[x1:x1 + x2])).reshape(-1,1)
        shape = self.shape_link(np.dot(self.exog_shape, params[x1 + x2:])).reshape(-1,1)
        # GEV transformation
        normalized_data = (self.endog - location) / scale
        transformed_data = 1 + shape * normalized_data
        
        # Return a large penalty for invalid parameter values
        if np.any(transformed_data <= 0) or np.any(scale <= 0):
            return 1e6

        return np.sum(np.log(scale)) + np.sum(transformed_data ** (-1 / shape)) + np.sum(np.log(transformed_data) * (1 / shape + 1))