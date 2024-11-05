import os
import pandas as pd
import xarray as xr
import cftime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as ftr
from IPython.display import HTML
import numpy as np
from scipy.optimize import minimize
from statsmodels.base.model import GenericLikelihoodModel
import numpy as np
import pandas as pd
import warnings
from scipy import stats
import math
import numdifftools as nd
import statsmodels.api as sm
from scipy.optimize import approx_fprime
from scipy.stats import norm
from scipy.stats import chi2
from scipy.stats import gumbel_r

class EVTModel:
    """
    A statistical model for EVT (Extreme Value Theory), supporting GEV and GPD models.
    
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

    def __init__(self, endog, exog={}, loc_link = None, scale_link = None, shape_link = None, **kwargs):
        """
        Initializes the EVTModel with specified parameters.
        
        Parameters
        ----------
        endog : array-like
            Endogenous variable.
        exog : dict of array-like, optional
            Dictionary where keys are 'shape', 'scale', and 'location', specifying exogenous variables for each parameter.
        model_type : str, optional
            Type of EVT model, either 'GEV' or 'GPD'. Default is 'GEV'.
        kwargs : dict
            Additional keyword arguments.
        """
        # Store the model type
        self.nobs = len(endog)
        self.result = None
        self.fitted = False
        self.scale_guess = np.sqrt(6 * np.var(endog)) / np.pi
        self.shape_guess = 0.1
        self.loc_link = loc_link if loc_link is not None else self.identity
        self.scale_link = scale_link if scale_link is not None else self.identity
        self.shape_link = shape_link if shape_link is not None else self.identity
        # Handle missing values and constant term
        missing = kwargs.pop('missing', 'none')
        hasconst = kwargs.pop('hasconst', None)

        # Process the data (endog and exog)
        self.data = self._handle_data(endog, exog, missing, hasconst, **kwargs)

    def _handle_data(self, endog, exog, missing='none', hasconst=None, **kwargs):
        """
        Processes and validates the endogenous and exogenous data, handling multiple exogenous variables
        for each EVT model parameter.

        Parameters
        ----------
        endog : array-like
            Endogenous variable.
        exog : dict or array-like
            Exogenous variables for parameters. Can be a single array (applied to all) or a dict with 'shape', 'scale', 'location'.
        missing : str
            Specifies handling of missing values.
        hasconst : bool or None
            Indicates if a constant is present in the exogenous variables.
        kwargs : dict
            Additional arguments for data handling.
            
        Returns
        -------
        data : dict
            Dictionary containing processed `endog` and `exog` data.
        """
        # Convert `endog` to a numpy array if provided as list or tuple
        if isinstance(endog, (list, tuple)):
            endog = np.asarray(endog)
        
        # Ensure `exog` is in dictionary format with numpy arrays for each parameter
        processed_exog = {}
        if isinstance(exog, dict):
            for param in ['shape', 'scale', 'location']:
                if exog.get(param) is not None:
                    processed_exog[param] = (
                        np.asarray(exog[param]) if isinstance(exog[param], (list, tuple)) else exog[param]
                    )
                else:
                    processed_exog[param] = None
        else:
            exog_array = np.asarray(exog) if isinstance(exog, (list, tuple)) else exog
            processed_exog = {'shape': exog_array, 'scale': exog_array, 'location': exog_array}
        
        # Attach processed exog for each parameter as instance attributes
        self.endog = endog
        self.exog_shape = processed_exog['shape']
        self.exog_scale = processed_exog['scale']
        self.exog_location = processed_exog['location']
        self.trans = any(v is not None for v in [self.exog_location, self.exog_shape, self.exog_scale])
        # Attach remaining kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Return a dictionary of processed data for internal use or external reference
        return {
            'endog': endog,
            'exog': processed_exog,
            'missing': missing,
            'hasconst': hasconst
        }

    
    def fit(self):
        """
        Fit a model to data.
        """
        raise NotImplementedError

    
    def predict(self, params, exog=None, *args, **kwargs):
        """
        After a model has been fit predict returns the fitted values.

        This is a placeholder intended to be overwritten by individual models.
        """
        raise NotImplementedError

class EVTLikelihood(EVTModel):

    def __init__(self, endog, loc_link = EVTModel.identity, scale_link = EVTModel.identity, shape_link = EVTModel.identity, exog={}, **kwargs):
        super().__init__(endog=endog, exog=exog, loc_link = loc_link,scale_link = scale_link, shape_link = shape_link, **kwargs)

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
            raise ValueError("Model was not fitted so you can not find the hessian at optimal parameters.")

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
        return ValueError("Model was not fitted so you can not find the hessian at optimal parameters.")

    def nloglike(self, params):
        pass

    def fit(self, start_params, method='L-BFGS-B', maxiter=100, full_output=True, **kwargs):
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
        self.fitted = True
        self.result = minimize(self.nloglike, start_params, method=method, **kwargs)
        self.result.endog = self.endog
        self.result.len_endog = len(self.endog)
        self.result.trans = self.trans
        return GEVResult(self.result)

class GEV(EVTLikelihood):
    def __init__(self, endog, exog={}, loc_link = EVTModel.identity, scale_link = EVTModel.identity, shape_link = EVTModel.identity, **kwargs):
        super().__init__(endog=endog, exog=exog, loc_link = loc_link,scale_link = scale_link, shape_link = shape_link, **kwargs)
        self.location_guess = np.mean(endog) - 0.57722 * (np.sqrt(6 * np.var(endog)) / np.pi)
        def initialize_exog(exog):
            if exog is None:
                return np.ones((self.nobs, 1))
            return sm.add_constant(exog) if exog.ndim == 1 else exog

        self.exog_location = initialize_exog(self.exog_location)
        self.exog_shape = initialize_exog(self.exog_shape)
        self.exog_scale = initialize_exog(self.exog_scale)
        self.data['exog']['location'] = self.exog_location
        self.data['exog']['scale'] = self.exog_scale
        self.data['exog']['shape'] = self.exog_shape

    def fit(self, start_params=None, method='L-BFGS-B', maxiter=1000, full_output=True, **kwargs):
        if start_params is None:
            start_params = np.array(
            [self.location_guess] +
            ([0] * (self.exog_location.shape[1] - 1)) +
            [self.scale_guess] +
            ([0] * (self.exog_scale.shape[1] - 1)) +
            [self.shape_guess] +
            ([0] * (self.exog_shape.shape[1] - 1))
            )

        # Fit the model
        result = super().fit(start_params, method, maxiter, full_output, **kwargs)

        # Efficiently compute fitted parameters with slicing
        loc_end = self.exog_location.shape[1]
        scale_end = loc_end + self.exog_scale.shape[1]
        fitted_loc = self.loc_link(self.exog_location @ result.like.x[:loc_end])
        fitted_scale = self.scale_link(self.exog_scale @ result.like.x[loc_end:scale_end])
        fitted_shape = self.shape_link(self.exog_shape @ result.like.x[scale_end:])
        result.like.params = [np.mean(fitted_loc),np.mean(fitted_scale),np.mean(fitted_shape)]
        
        # Apply transformation if needed
        result.like.data = (
            -np.log((1 + (fitted_shape * (self.endog - fitted_loc)) / fitted_scale) ** (-1 / fitted_shape))
            if self.trans else self.endog
        )

        return result

    
    def nloglike(self,params):
        # Extract the number of covariates for each parameter
        x1 = self.exog_location.shape[1] if self.exog_location is not None else 0
        x2 = self.exog_scale.shape[1] if self.exog_scale is not None else 0
        x3 = self.exog_shape.shape[1] if self.exog_shape is not None else 0

        # Compute location, scale, and shape parameters
        location = self.loc_link(np.dot(self.exog_location, params[:x1]))
        scale = self.scale_link(np.dot(self.exog_scale, params[x1:x1 + x2]))
        shape = self.shape_link(np.dot(self.exog_shape, params[x1 + x2:]))

        # GEV transformation
        normalized_data = (self.endog - location) / scale
        transformed_data = 1 + shape * normalized_data

        # Return a large penalty for invalid parameter values
        if np.any(transformed_data <= 0) or np.any(scale <= 0):
            return 1e6
        
        return np.sum(np.log(scale)) + np.sum(transformed_data ** (-1 / shape)) + np.sum(np.log(transformed_data) * (1 / shape + 1))

class GEVTradowsky(EVTLikelihood):
    def __init__(self, endog, exog=None, **kwargs):
        super().__init__(endog=endog, exog=exog, **kwargs)
        self.location_guess = np.mean(endog) - 0.57722 * (np.sqrt(6 * np.var(endog)) / np.pi)
        if self.exog_location is None or self.exog_location.ndim != 1:
            raise ValueError("Location must depend on exactly one covariate in the Tradowsky model")
        else:
            self.exog_location = self.exog_location.reshape(-1,1)

        if self.exog_scale is None or self.exog_scale.ndim != 1:
            raise ValueError("Scale must depend on exactly one covariate in the Tradowsky model")
        else:
            self.exog_scale = self.exog_scale.reshape(-1,1)

        if self.exog_shape is not None:
            raise ValueError("There are no covariates for the shape in the Tradowsky model")
        else:
            self.exog_shape = np.ones((self.nobs, 1))

        self.data['exog']['shape'] = self.exog_shape
        model_0 = GEV(endog=self.endog,full_output=True).fit()
        self.mu_0 = model_0.fitted_params()[0]
        self.sigma_0 = model_0.fitted_params()[1]


    def fit(self, start_params=None, method='L-BFGS-B', maxiter=1000, full_output=True, **kwargs):
        if start_params is None:
            start_params = np.array([1,0.1])
        return(super().fit(start_params, method, maxiter, full_output, **kwargs))
    
    def nloglike(self,params):
        # Compute location, scale, and shape parameters
        location = self.mu_0 * EVTModel.exp_link(np.dot(self.exog_location, params[:1]) / self.mu_0)
        scale = self.sigma_0 * EVTModel.exp_link(np.dot(self.exog_scale, params[:1]) / self.mu_0)
        shape = np.dot(self.exog_shape, params[1:2])

        # GEV transformation
        normalized_data = (self.endog - location) / scale
        transformed_data = 1 + shape * normalized_data

        # Return a large penalty for invalid parameter values
        if np.any(transformed_data <= 0) or np.any(scale <= 0) or np.any(shape>=0.4):
            return 1e6
        
        return np.sum(np.log(scale)) + np.sum(transformed_data ** (-1 / shape)) + np.sum(np.log(transformed_data) * (1 / shape + 1))


class EVTResults():
    def __init__(self, evtLikelihood):
        self.like = evtLikelihood
        self.num_params = self.like.x.size

    def AIC(self):
        return 2 * self.num_params + 2 * self.like.fun

    def BIC(self):
        return self.num_params * np.log(self.like.len_endog) + 2 * self.like.fun

    def SE(self):
        # Compute standard errors using the inverse Hessian matrix
        if self.like.hess_inv is None:
            raise ValueError("Hessian matrix is not available.")
        hessian_inv = self.like.hess_inv.todense() if hasattr(self.like.hess_inv, 'todense') else self.like.hess_inv
        se = np.sqrt(np.diag(hessian_inv))
        return se

    def fitted_params(self):
        # Return the fitted parameters
        return self.like.x

    def __str__(self):
        # Calculate fitted values, SE, z-scores, p-values, AIC, and BIC
        fitted_params = self.fitted_params()
        se = self.SE()
        z_scores = fitted_params / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        aic = self.AIC()
        bic = self.BIC()

        # Dynamically calculate the width of the separator lines
        header = "Parameter  Estimate   SE     z     P>|z|  95% CI          Signif."
        line_length = len(header)
        separator = "-" * line_length
        
        # Format the output string
        result_str = "\n"
        result_str += "=" * line_length + "\n"
        result_str += "          EVT Results Summary       \n"
        result_str += "=" * line_length + "\n"
        result_str += f"AIC: {aic:.2f}\n"
        result_str += f"BIC: {bic:.2f}\n\n"
        result_str += separator + "\n"
        result_str += header + "\n"
        result_str += separator + "\n"
        
        for i in range(self.num_params):
            # Determine significance stars based on p-value
            if p_values[i] < 0.001:
                signif = '***'
            elif p_values[i] < 0.01:
                signif = '**'
            elif p_values[i] < 0.05:
                signif = '*'
            else:
                signif = ''
            
            # Calculate the 95% confidence interval
            ci_lower = fitted_params[i] - 1.96 * se[i]
            ci_upper = fitted_params[i] + 1.96 * se[i]
            
            result_str += (f"{i+1:<10} {fitted_params[i]:<10.4f} {se[i]:<7.4f} {z_scores[i]:<6.2f} "
                           f"{p_values[i]:<.4f}  ({ci_lower:.4f}, {ci_upper:.4f}) {signif}\n")
        
        result_str += separator + "\n"
        result_str += "Notes: *** p<0.001, ** p<0.01, * p<0.05\n"
        
        return result_str

class GEVResult(EVTResults):
    def __init__(self, evtLikelihood):
        return super().__init__(evtLikelihood=evtLikelihood)
    def probability_plot(self,ax=None):
        sorted_data  = np.sort(self.like.data)
        if ax is None:
            ax = plt.gca()

        print("Plotting on axis:", ax)
        if self.like.trans:
            # Sort the data for plotting
            sorted_data = np.sort(self.like.data)
            n = self.like.len_endog
            x = np.arange(1, n + 1) / (n + 1)

            # Empirical vs Model Plot
            ax.scatter(x, np.exp(-np.exp(-sorted_data)), color="#4a90e2", label="Empirical vs Model")
            ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=1.5, label="45° Line")  # Reference line
            ax.set_xlabel("Empirical", fontsize=12)
            ax.set_ylabel("Model", fontsize=12)
            ax.set_title("Residual Probability Plot", fontsize=14)
            ax.grid(True, linestyle=':', color='grey', alpha=0.6)
            ax.legend()
        else:
            n = self.like.len_endog
            # Empirical probabilities
            empirical_probs = np.arange(1, len(self.like.data) + 1) / len(self.like.data)
            # Calculate the model probabilities using the GEV distribution function
            model_probs = self.gevf(self.like.x, sorted_data)
            
            # Plot the probability plot
            ax.plot(empirical_probs, model_probs, 'o', label="Model vs. Empirical")
            ax.plot([0, 1], [0, 1], 'r--', label="y = x")  # Reference line for a perfect fit
            ax.set_xlabel("Empirical")
            ax.set_ylabel("Model")
            ax.set_title("Probability Plot")
            ax.legend()

    def quantile_plot(self,ax=None):
        sorted_data = np.sort(self.like.data)
        n = self.like.len_endog
        if ax is None:
            ax = plt.gca()
        if self.like.trans:
            x = np.arange(1, n + 1) / (n + 1)
            # Quantile Plot (Gumbel Scale) with Dynamic Range and Points
            plt.scatter(-np.log(-np.log(x)), sorted_data, color="#f5a623", label="Model vs Empirical")
            # Setting dynamic range based on data
            min_val, max_val = min(-np.log(-np.log(x)).min(), sorted_data.min()), max(-np.log(-np.log(x)).max(), sorted_data.max())
            ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="45° Line")  # Reference line
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel("Empirical", fontsize=12)
            ax.set_title("Quantile Plot", fontsize=14)
            ax.grid(True, linestyle=':', color='grey', alpha=0.6)
            ax.legend()
        else:
            model_quantiles = self.gevq(self.like.x, 1 - (np.arange(1, n + 1) / (n + 1)))
            # Sort data
            empirical_quantiles = sorted_data
            # Plot
            ax.plot(model_quantiles, empirical_quantiles, 'o', label="Data")
            ax.plot([min(model_quantiles), max(model_quantiles)], [min(model_quantiles), max(model_quantiles)], 'b-', label="y = x")
            ax.set_xlabel("Model")
            ax.set_ylabel("Empirical")
            ax.set_title("Quantile Plot")
            plt.legend()
        
    def diag(self):
        fig, axs = plt.subplots(1,2, figsize=(14, 6))  # Specify figure size here for combined plots

        self.probability_plot(ax=axs[0])  # Plot on the first subplot
        self.quantile_plot(ax=axs[1])     # Plot on the second subplot
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()  

    def gevf(self,a, z):
        """
        Calculates the GEV distribution function (cdf) for given parameters.

        Parameters:
        a (list or array): MLE parameters for the GEV distribution [location, scale, shape]
        z (array): Sorted data points
                
        Returns:
        array: CDF values of the GEV model for data points
        """
        location, scale, shape = a[0], a[1], a[2]
                
        if shape != 0:
            # Use the GEV CDF formula for non-zero shape
            cdf_values = np.exp(- (1 + (shape * (z - location)) / scale) ** (-1 / shape))
        else:
            # If shape parameter is zero, use Gumbel distribution as a special case of GEV
            cdf_values = np.exp(-np.exp(-(z - location) / scale))
                
        return cdf_values

    def gevq(self,a,p):
        if a[2] != 0:
            return a[0] + (a[1] * ((-np.log(1 - p))**(-a[2]) - 1)) / a[2]
        else:
            return gumbel_r.ppf(p, loc=a[0], scale=a[1])

    def gev_profxi(self,xlow=-0.5, xup=0.5, conf=0.9, nint=100):
        """
        Plots profile log-likelihood for the shape parameter (xi) in a GEV model.

        Parameters:
        z : dict
            A dictionary containing 'mle' (maximum likelihood estimates for location and scale)
            and 'data' (the observed data).
        xlow : float
            Lower bound for the shape parameter (xi).
        xup : float
            Upper bound for the shape parameter (xi).
        conf : float, optional
            Confidence level for the profile likelihood interval (default is 0.95).
        nint : int, optional
            Number of intervals to evaluate in the shape parameter range (default is 100).
        """
        print("If routine fails, try changing the plotting interval.")
        if not self.like.trans:
            # Initialize storage for log-likelihood values and xi values
            z = {
                'mle': [55.8, 12.3],
                'data': self.like.data,
                'nllh': self.like.fun
            }
            v = np.zeros(nint)
            x = np.linspace(xup, xlow, nint)
            sol = np.array([z['mle'][0], z['mle'][1]])

            def gev_plikxi(a, xi):
                """Calculates the profile negative log-likelihood for given parameters."""
                y = (z['data'] - a[0]) / a[1]
                if abs(xi) < 1e-6:  # Approximate to Gumbel when xi is very small
                    if a[1] <= 0:
                        return 1e6
                    return len(y) * np.log(a[1]) + np.sum(np.exp(-y)) + np.sum(y)
                else:
                    y = 1 + xi * y
                    if a[1] <= 0 or np.any(y <= 0):
                        return 1e6
                    return len(y) * np.log(a[1]) + np.sum(y ** (-1 / xi)) + (1 / xi + 1) * np.sum(np.log(y))
        
        # Loop through xi values, optimize the parameters for each, and store the log-likelihood
            for i in range(nint):
                xi = x[i]
                opt = minimize(lambda a: gev_plikxi(a, xi), sol)
                sol = opt.x  # Update solution with optimal parameters
                v[i] = opt.fun  # Store the minimized negative log-likelihood

            # Plotting
            plt.plot(x, -v, label="Profile Log-likelihood")
            plt.xlabel("Shape Parameter (xi)")
            plt.ylabel("Profile Log-likelihood")
            
            # Draw confidence level line
            ma = -z['nllh']  # Maximum log-likelihood value
            plt.axhline(y=ma, color="blue", linestyle="-", label="Maximum Log-likelihood")
            plt.axhline(y=ma - 0.5 * chi2.ppf(conf, 1), color="blue", linestyle="--", label=f"{int(conf*100)}% Confidence Level")
            print(ma - 0.5 * chi2.ppf(conf, 1))
            plt.legend()
            plt.grid(True, linestyle=':', color='grey', alpha=0.6)
            plt.show()
        else:
            print("NOT YET IMPLEMENTED : Non stationnary profile likelihoods.")

    def gev_profR(self,m, xlow=-0.9, xup=0.9, conf=0.95, nint=100):
        """
        Plots profile log-likelihood for the m-year return level in a GEV model.

        Parameters:
        z : dict
            A dictionary containing 'mle' (maximum likelihood estimates for scale and shape)
            and 'data' (the observed data).
        m : float
            The return period (e.g., 100 for a 100-year return level).
        xlow : float
            Lower bound for the return level to be evaluated.
        xup : float
            Upper bound for the return level to be evaluated.
        conf : float, optional
            Confidence level for the profile likelihood interval (default is 0.95).
        nint : int, optional
            Number of intervals to evaluate in the return level range (default is 100).
        """
        if not self.like.trans:
            if m <= 1:
                raise ValueError("`m` must be greater than one")
            
            print("If routine fails, try changing the plotting interval.")
            z = {
                'mle': [10, 0.025],
                'data': self.like.data,
                'nllh': self.like.fun
            }
            p = 1 / m  # Exceedance probability
            v = np.zeros(nint)
            x = np.linspace(xup, xlow, nint)
            sol = np.array([z['mle'][0], z['mle'][1]])  # Starting guess for scale and shape

            def gev_plik(a, xp):
                """Calculates the profile negative log-likelihood for given parameters."""
                scale, shape = a
                if abs(shape) < 1e-6:  # Approximate to Gumbel when shape is close to zero
                    mu = xp + scale * np.log(-np.log(1 - p))
                    y = (z['data'] - mu) / scale
                    if np.isinf(mu) or scale <= 0:
                        return 1e6
                    return len(y) * np.log(scale) + np.sum(np.exp(-y)) + np.sum(y)
                else:
                    mu = xp - scale / shape * ((-np.log(1 - p)) ** (-shape) - 1)
                    y = (z['data'] - mu) / scale
                    y = 1 + shape * y
                    if np.isinf(mu) or scale <= 0 or np.any(y <= 0):
                        return 1e6
                    return len(y) * np.log(scale) + np.sum(y ** (-1 / shape)) + (1 / shape + 1) * np.sum(np.log(y))
        
            for i in range(nint):
                xp = x[i]  # Current return level
                opt = minimize(lambda a: gev_plik(a, xp), sol)  # Minimize negative log-likelihood with xp fixed
                sol = opt.x  # Update solution with optimal parameters for next iteration
                v[i] = opt.fun  # Store the minimized negative log-likelihood
            # Plotting
            plt.plot(x, -v, label="Profile Log-likelihood")
            plt.xlabel("Return Level")
            plt.ylabel("Profile Log-likelihood")

            # Draw confidence level line
            ma = -z['nllh']  # Maximum log-likelihood value
            plt.axhline(y=ma, color="blue", linestyle="-", label="Maximum Log-likelihood")
            plt.axhline(y=ma - 0.5 * chi2.ppf(conf, 1), color="blue", linestyle="--", label=f"{int(conf*100)}% Confidence Level")
            plt.legend()
            plt.grid(True, linestyle=':', color='grey', alpha=0.6)
            plt.show()