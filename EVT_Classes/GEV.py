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
        self.location_guess = np.mean(endog) - 0.57722 * (np.sqrt(6 * np.var(endog)) / np.pi)
        self.shape_guess = 0.1
        self.loc_link = loc_link or self.identity
        self.scale_link = scale_link or self.identity
        self.shape_link = shape_link or self.identity

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
        nobs = len(endog)

        if exog is None:
            # Initialize exog as a dictionary with ones as default values for each parameter
            self.exog = {
                'shape': np.ones((nobs, 1)),
                'scale': np.ones((nobs, 1)),
                'location': np.ones((nobs, 1))
            }
        elif isinstance(exog, np.ndarray):
            if exog.shape[0] != nobs:
                raise ValueError(
                    f"The length of the provided exog array ({exog.shape[0]}) must match the length of `endog` ({nobs})."
                )
            if len(exog.shape) == 1:
                exog_augmented = np.concatenate([np.ones((nobs, 1)), exog.reshape(-1,1)], axis=1)
            else:
                # Use the same `exog` array for all three parameters
                exog_augmented = np.concatenate([np.ones((nobs, 1)), exog], axis=1)
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
                    self.exog[key] = np.ones((nobs, 1))
                else:
                    value_array = np.asarray(value)
                    if value_array.shape[0] != nobs:
                        raise ValueError(
                            f"The number of rows in exog['{key}'] ({value_array.shape[0]}) "
                            f"must match the number of rows in `endog` ({nobs})."
                        )
                    if len(value_array.shape) == 1:
                        self.exog[key] = np.concatenate([np.ones((nobs, 1)), value_array.reshape(-1, 1)], axis=1)
                    else:
                        self.exog[key] = np.concatenate([np.ones((nobs, 1)), value_array], axis=1)
        else:
            raise ValueError("`exog` must be either a dictionary (default), or a NumPy array of shape (n,>=1).")
        
        if all(np.array_equal(self.exog[key], np.ones((nobs, 1))) for key in ['shape', 'scale', 'location']):
            self.trans = False
        else:
            self.trans = True

        self.endog = np.asarray(endog).reshape(-1, 1)

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

    def nloglike(self, params):
        """
        Computes the negative log-likelihood of the GEV model.
        """
        # Extract the number of covariates for each parameter
        x1 = self.exog['location'].shape[1] 
        x2 = self.exog['scale'].shape[1] 
        x3 = self.exog['shape'].shape[1] 

        # Compute location, scale, and shape parameters
        location = self.loc_link(np.dot(self.exog['location'], params[:x1].reshape(-1,1)))
        scale = self.scale_link(np.dot(self.exog['scale'], params[x1:(x1 + x2)].reshape(-1,1)))
        shape = self.shape_link(np.dot(self.exog['shape'], params[(x1 + x2):].reshape(-1,1)))
        # GEV transformation
        normalized_data = (self.endog - location) / scale
        transformed_data = 1 + shape * normalized_data
        # Return a large penalty for invalid parameter values
        if np.any(transformed_data <= 0) or np.any(scale <= 0):
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
        if start_params is None:
            start_params = np.array(
            [self.location_guess] +
            ([0] * (self.exog['location'].shape[1] - 1)) +
            [self.scale_guess] +
            ([0] * (self.exog['scale'].shape[1] - 1)) +
            [self.shape_guess] +
            ([0] * (self.exog['shape'].shape[1] - 1))
            )
        
        # Compute plot_data based on transformation

        self.fitted = True
        self.result = minimize(self.nloglike, start_params, method=method, **kwargs)
        self.result.endog = self.endog
        self.result.len_endog = len(self.endog)
        self.result.trans = self.trans

        loc_end = self.exog['location'].shape[1]
        scale_end = loc_end + self.exog['scale'].shape[1]
        fitted_loc = self.loc_link(self.exog['location'] @ self.result.x[:loc_end]).reshape(-1,1)
        fitted_scale = self.scale_link(self.exog['scale'] @ self.result.x[loc_end:scale_end]).reshape(-1,1)
        fitted_shape = self.shape_link(self.exog['shape'] @ self.result.x[scale_end:]).reshape(-1,1)
        if self.trans:
            plot_data = -np.log((1 + (fitted_shape * (self.endog - fitted_loc)) / fitted_scale) ** (-1 / fitted_shape))
        else:
            plot_data = self.endog

        return GEVResult(
            self.result,
            endog=self.endog,
            len_endog=len(self.endog),
            trans=self.trans,
            plot_data = plot_data,
            gev_params = (fitted_loc,fitted_scale,fitted_shape)
        )


class GEVResult():
    def __init__(self, optimize_result, endog, len_endog, trans,plot_data,gev_params):
            # Extract relevant attributes from optimize_result and give them meaningful names
            self.fitted_params = optimize_result.x
            self.log_likelihood = optimize_result.fun
            self.hessian_inverse = optimize_result.hess_inv
            self.jacobian = optimize_result.jac
            self.success = optimize_result.success
            self.plot_data = plot_data
            self.gev_params = gev_params
            self.len_endog = len_endog
            self.trans = trans
            self.nparams = self.fitted_params.size

    def AIC(self):
        return 2 * self.nparams + 2 * self.log_likelihood

    def BIC(self):
        return self.nparams * 1 + 2 * self.log_likelihood

    def SE(self):
        # Compute standard errors using the inverse Hessian matrix
        if self.hessian_inverse is None:
            raise ValueError("Hessian matrix is not available.")
        hessian_inv = self.hessian_inverse.todense() if hasattr(self.hessian_inverse, 'todense') else self.hessian_inverse
        se = np.sqrt(np.diag(hessian_inv))
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
    
    def probability_plot(self,ax=None):
        sorted_data  = np.sort(self.plot_data, axis=0)
        if ax is None:
            ax = plt.gca()

        if self.trans:
            # Sort the data for plotting
            n = self.len_endog
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
            n = self.len_endog
            # Empirical probabilities
            empirical_probs = np.arange(1, len(self.plot_data) + 1) / len(self.plot_data)
            # Calculate the model probabilities using the GEV distribution function
            model_probs = self.gevf(self.fitted_params, sorted_data)
            
            # Plot the probability plot
            ax.plot(empirical_probs, model_probs, 'o', label="Model vs. Empirical")
            ax.plot([0, 1], [0, 1], 'r--', label="y = x")  # Reference line for a perfect fit
            ax.set_xlabel("Empirical")
            ax.set_ylabel("Model")
            ax.set_title("Probability Plot")
            ax.legend()

if __name__ == "__main__":
    EOBS = pd.read_csv(r"c:\ThesisData\EOBS\Blockmax\blockmax_temp.csv")

    n = len(EOBS["prmax"].values.reshape(-1,1))
    exog = {'scale':EOBS["tempanomalyMean"].values.reshape(-1,1)}
    model = GEVLikelihood(endog=EOBS["prmax"].values.reshape(-1,1),exog=exog)
    gev_result_1 = model.fit()
    gev_result_1.probability_plot()
    plt.show()