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
        self.len_endog = len(endog)
        self.result = None
        self.fitted = False
        self.scale_guess = np.sqrt(6 * np.var(endog)) / np.pi
        self.location_guess = np.mean(endog) - 0.57722 * (np.sqrt(6 * np.var(endog)) / np.pi)
        self.shape_guess = 0.1
        self.loc_link = loc_link or self.identity
        self.scale_link = scale_link or self.identity
        self.shape_link = shape_link or self.identity
        self._forced_param_index = -1
        self._forced_param_value = 0

        # Handle data
        self._process_data(endog, exog, **kwargs)

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
        if exog is None:
            # Initialize exog as a dictionary with ones as default values for each parameter
            self.exog = {
                'shape': np.ones((self.len_endog, 1)),
                'scale': np.ones((self.len_endog, 1)),
                'location': np.ones((self.len_endog, 1))
            }
        elif isinstance(exog, np.ndarray):
            if exog.shape[0] != self.len_endog:
                raise ValueError(
                    f"The length of the provided exog array ({exog.shape[0]}) must match the length of `endog` ({self.len_endog})."
                )
            if len(exog.shape) == 1:
                exog_augmented = np.concatenate([np.ones((self.len_endog, 1)), exog.reshape(-1,1)], axis=1)
            else:
                # Use the same `exog` array for all three parameters
                exog_augmented = np.concatenate([np.ones((self.len_endog, 1)), exog], axis=1)
            self.exog = {
                'shape': exog_augmented,
                'scale': exog_augmented,
                'location': exog_augmented
            }
        elif isinstance(exog, dict):
            # Initialize the exog dictionary by iterating over the keys
            self.exog = {}
            for key in ['shape', 'scale', 'location']:
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
                        self.exog[key] = np.concatenate([np.ones((self.len_endog, 1)), value_array.reshape(-1, 1)], axis=1)
                    else:
                        self.exog[key] = np.concatenate([np.ones((self.len_endog, 1)), value_array], axis=1)
        else:
            raise ValueError("`exog` must be either a dictionary (default), or a NumPy array of shape (n,>=1).")
        
        if all(np.array_equal(self.exog[key], np.ones((self.len_endog, 1))) for key in ['shape', 'scale', 'location']):
            self.trans = False
        else:
            self.trans = True

        self.endog = np.asarray(endog).reshape(-1, 1)
        self.len_exog = (self.exog['location'].shape[1],self.exog['scale'].shape[1],self.exog['shape'].shape[1])
        self.nparams = self.len_exog[0]+self.len_exog[1]+self.len_exog[2]

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
    def __init__(self, endog, loc_link=GEV.identity, scale_link=GEV.identity, shape_link=GEV.identity, exog={'shape': None, 'scale': None, 'location': None}, **kwargs):
        """
        Initializes the GEVLikelihood model with given parameters.
        """
        super().__init__(endog=endog, exog=exog, loc_link=loc_link, scale_link=scale_link, shape_link=shape_link, **kwargs)

    def loglike(self, params):
        """
        Computes the log-likelihood of the model.
        """
        return -(self.nloglike(params))

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
        
    def _nloglike_formula(self,transformed_data,scale,shape):
        return np.sum(np.log(scale)) + np.sum(transformed_data ** (-1 / shape)) + np.sum(np.log(transformed_data) * (1 + 1 / shape))

    def nloglike(self, free_params):
        """
        Computes the negative log-likelihood of the GEV model.
        """

        # Extract the number of covariates for each parameter
        i, j, _ = self.len_exog
        params = np.ones(self.nparams)
        free_index = 0

        for k in range(self.nparams):  # Total number of parameters
            if k == self._forced_param_index:
                params[k] = self._forced_param_value  # Use fixed parameter value
            else:
                params[k] = free_params[free_index] # Use optimized parameter
                free_index += 1
        
        # Compute location, scale, and shape parameters
        location = self.loc_link(np.dot(self.exog['location'], params[:i].reshape(-1,1)))
        scale = self.scale_link(np.dot(self.exog['scale'], params[i:i+j].reshape(-1,1)))
        shape = self.shape_link(np.dot(self.exog['shape'], params[i+j:].reshape(-1,1)))
        # GEV transformation
        normalized_data = (self.endog - location) / scale
        transformed_data = 1 + shape * normalized_data
        # Return a large penalty for invalid parameter values
        if np.any(transformed_data <= 0) or np.any(scale <=0):
            return 1e6

        return self._nloglike_formula(transformed_data,scale,shape)
    

    def loglike(self,params):
        return -(self.nloglike(params))

    def fit(self, start_params=None, optim_method='L-BFGS-B', fit_method ='MLE', **kwargs):
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
        i,j,k = self.len_exog
        if fit_method.lower() == 'mle':
            if start_params is None:
                free_params = np.array(
                [self.location_guess] +
                ([0] * (i-1)) +
                [self.scale_guess] +
                ([0] * (j-1)) +
                [self.shape_guess] +
                ([0] * (k-1))
                )
            
            # Compute plot_data based on transformation

            #Investigate fitted is it useful in the code ? I don't think so. 
            self.result = minimize(self.nloglike, free_params, method=optim_method, **kwargs)

            fitted_loc = self.loc_link(self.exog['location'] @ self.result.x[:i]).reshape(-1,1)
            fitted_scale = self.scale_link(self.exog['scale'] @ self.result.x[i:i+j]).reshape(-1,1)
            fitted_shape = self.shape_link(self.exog['shape'] @ self.result.x[i+j:]).reshape(-1,1)
            if self.trans:
                plot_data = -np.log((1 + (fitted_shape * (self.endog - fitted_loc)) / fitted_scale) ** (-1 / fitted_shape))
            else:
                plot_data = self.endog

            return GEVFit(
                optimize_result = self.result,
                endog=self.endog,
                len_endog=self.len_endog,
                exog = self.exog,
                len_exog = self.len_exog,
                trans=self.trans,
                plot_data = plot_data,
                gev_params = (fitted_loc,fitted_scale,fitted_shape)
            )
        else:
            n=500
            profile_mles = np.empty((self.nparams,n))
            param_values = np.empty((self.nparams,n))
            fits = np.empty(self.nparams)
            for l in range(self.nparams):
                if l == 0:
                    profile_params = np.linspace(self.location_guess / 5, self.location_guess * 2, n)
                elif l == i:
                    profile_params = np.linspace(self.scale_guess / 5, self.scale_guess * 2, n)
                elif l == i + j:
                    profile_params = np.linspace(-0.6, 0.6, n)
                else:
                    profile_params = np.linspace(-50, 50, n)

                free_params = []  # Initialize an empty list
                for param_index in range(self.nparams):
                    if param_index == l:
                        continue  # Skip the fixed parameter
                    if param_index == 0:
                        free_params.append(self.location_guess)
                    elif param_index == i:
                        free_params.append(self.scale_guess)
                    elif param_index == i+j:
                        free_params.append(self.shape_guess)
                    else:
                        free_params.append(0)

                # Loop only over values in all_mus
                for m, param_value in enumerate(profile_params):
                    self._forced_param_value = param_value  # Update only this value per iteration
                    self._forced_param_index = l
                    result = minimize(self.nloglike, free_params, method=optim_method, **kwargs)
                    profile_mles[l][m] = result.fun
                    param_values[l][m] = param_value

            for n in range(self.nparams):
                fits[n] = param_values[n][np.argmin(profile_mles[n])]
            return fits


class GEVFit():
    def __init__(self, optimize_result, endog, len_endog, exog, len_exog, trans,plot_data,gev_params):
            # Extract relevant attributes from optimize_result and give them meaningful names
            self.fitted_params = optimize_result.x
            self.endog = endog
            self.len_endog = len_endog
            self.exog = exog
            self.len_exog = len_exog
            self.trans = trans
            self.plot_data = plot_data
            self.gev_params = gev_params
            self.log_likelihood = optimize_result.fun
            self.cov_matrix = optimize_result.hess_inv
            self.jacobian = optimize_result.jac
            self.success = optimize_result.success
            self.nparams = self.fitted_params.size

    def AIC(self):
        return 2 * self.nparams + 2 * self.log_likelihood
    def BIC(self):
        return self.nparams * np.log(self.len_endog) + 2 * self.log_likelihood

    def SE(self):
        # Compute standard errors using the inverse Hessian matrix
        if self.cov_matrix is None:
            raise ValueError("Hessian matrix is not available.")
        cov_matrix = self.cov_matrix.todense() if hasattr(self.cov_matrix, 'todense') else self.cov_matrix
        se = np.sqrt(np.diag(cov_matrix))
        return se
    
    def __str__(self):
        # Calculate fitted values, SE, z-scores, p-values, AIC, and BIC
        se = self.SE()
        z_scores = self.fitted_params / se
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
        
        for i in range(self.nparams):
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
            ci_lower = self.fitted_params[i] - 1.96 * se[i]
            ci_upper = self.fitted_params[i] + 1.96 * se[i]
            
            result_str += (f"{i+1:<10} {self.fitted_params[i]:<10.4f} {se[i]:<7.4f} {z_scores[i]:<6.2f} "
                           f"{p_values[i]:<.4f}  ({ci_lower:.4f}, {ci_upper:.4f}) {signif}\n")
        
        result_str += separator + "\n"
        result_str += "Notes: *** p<0.001, ** p<0.01, * p<0.05\n"
        
        return result_str

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

    def return_level(self,return_period, method="delta", confidence=0.95, ref_year=None):
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

        if not self.trans:
            if ref_year is not None:
                warnings.warn(
                    "Reference years are not required in a stationary model. Each year has the same return level for a given return period.",
                    UserWarning
                )
            loc, scale, shape = self.gev_params[0][0], self.gev_params[1][0], self.gev_params[2][0]
        else:
            if ref_year is None:
                raise ValueError(
                    "A reference year must be provided in a non-stationary model since return levels vary over time for a given return period."
                )
            elif ref_year >= self.len_endog:
                raise ValueError(
                    "You can not get the future return levels but only present or past return levels."
                )
            loc, scale, shape = self.gev_params[0][ref_year], self.gev_params[1][ref_year], self.gev_params[2][ref_year]

        
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

            params = self.fitted_params
            
        return(self._compute_confidence_interval(params=params,cov_matrix=self.cov_matrix.todense(),compute_z_p=compute_z_p,z_p=z_p,confidence=confidence))

    def _return_level_plot_values(self, method="delta", confidence=0.95, ref_year=None):
        all_return_periods = np.empty(len(np.arange(1.1, 1000, 0.5)))
        all_return_levels = np.empty(len(np.arange(1.1, 1000, 0.5)))  # Determine the number of iterations
        all_ci_lower = np.empty(len(np.arange(1.1, 1000, 0.5)))
        all_ci_upper = np.empty(len(np.arange(1.1, 1000, 0.5)))

        for i, T in enumerate(np.arange(1.1, 1000, 0.5)):
            z_p, ci = self.return_level(T,method,confidence,ref_year)
            all_return_periods[i] = T
            all_return_levels[i] = z_p
            all_ci_lower[i], all_ci_upper[i] = ci
        return all_return_periods, all_return_levels, all_ci_lower, all_ci_upper


    def return_level_plot(self,method="delta",confidence=0.95,ref_year=None):
        all_return_periods, all_return_levels, all_ci_lower, all_ci_upper = self._return_level_plot_values(method,confidence,ref_year)

        fig = go.Figure(go.Scatter(x=all_return_periods,y=all_return_levels,mode="lines",name="Return level Curve",line_color = "blue"))

        fig.add_trace(go.Scatter(x=np.concatenate([all_return_periods, all_return_periods[::-1]]),
                         y=np.concatenate([all_ci_upper, all_ci_lower[::-1]])))

        # Customize layout
        fig.update_layout(title='Return Level Plot with Confidence Interval',
                        xaxis_title='Return Period (Years)',
                        yaxis_title='Return Level',
                        template='plotly_white')

        # Show the plot
        fig.show()

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
        Initializes the GEVLikelihood model with given parameters.
        """
        super().__init__(endog=endog, exog=exog, loc_link=loc_link, scale_link=scale_link, shape_link=shape_link, **kwargs)
        model_0 = GEVLikelihood(endog=self.endog,exog={}).fit()
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

        #To be Modified
        if self.trans:
            plot_data = "a"
        else:
            plot_data = "b"

        return GEV_WWA_Fit(
            optimize_result=self.result,
            endog=self.endog,
            len_endog=len(self.endog),
            exog=self.exog,
            len_exog=(self.exog['location'].shape[1],self.exog['scale'].shape[1],self.exog['shape'].shape[1]),
            trans=self.trans,
            plot_data = plot_data,
            gev_params = (fitted_loc,fitted_scale,fitted_shape),
            mu0 = self.mu_0,
            sigma0 = self.sigma_0
        )
    
class GEV_WWA_Fit(GEVFit):
    def __init__(self, optimize_result, endog, len_endog, exog, len_exog, trans, plot_data, gev_params, mu0, sigma0):
        super().__init__(optimize_result, endog, len_endog, exog, len_exog, trans, plot_data, gev_params)
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
    n = len(EOBS["prmax"].values.reshape(-1,1))
    #tempanomalyMean
    exog = {"location" : EOBS["tempanomalyMean"], "scale" : EOBS["tempanomalyMean"]}
    #model = GEVLikelihood(endog=EOBS["prmax"].values.reshape(-1,1),exog=exog)
    #gev_result_1 = model.fit()
    #gev_result_1.probability_plot()
    #gev_result_1.data_plot(EOBS["year"].values)

    #test = GEV_WWA_Likelihood(endog=EOBS["prmax"].values.reshape(-1,1),exog=exog).fit()
    #test.data_plot(time=EOBS["year"])

    a1 = GEVLikelihood(endog=EOBS["prmax"].values.reshape(-1,1),exog=exog).fit(fit_method='i')
    print(a1)
    #a2 = GEVLikelihood(endog=EOBS["prmax"].values.reshape(-1,1),exog=exog).fit()
    #a1.data_plot(time=EOBS["year"])

    #e = a1.return_level(return_period=5,ref_year=74)
    #a1.return_level_plot()
