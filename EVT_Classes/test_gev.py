import unittest
import numpy as np
import pandas as pd
from gev_copy import GEV, GEVLikelihood, GEVResult # Replace with the actual module name where GEV is defined
from unittest.mock import patch, Mock, MagicMock
from scipy.optimize import approx_fprime
from scipy.optimize import OptimizeResult
from scipy.stats import norm


EOBS = pd.read_csv(r"c:\ThesisData\OUTPUTS\UCL_blockmax.csv")
TRUE_PARAMS = np.array([55,10,12,2,0.2,1])

class TestGEV(unittest.TestCase):
    
    def setUp(self):
        """
        Set up test cases with sample data for GEV initialization.
        """
        self.endog = np.random.randn(100).reshape(-1, 1)  # Simulated endogenous variable
        self.exog = {
            'shape': np.random.rand(100).reshape(-1, 1),
            'scale': np.random.rand(100).reshape(-1, 1),
            'location': np.random.rand(100).reshape(-1, 1)
        }

        self.likelihood = GEVLikelihood(self.endog)

                #make a fake GEVLikelihood object to get a GEVResult
        self.mock_optimize_result = MagicMock(spec=OptimizeResult)
        self.mock_optimize_result.x = np.array([0.5, 1.0, -0.3])
        self.mock_optimize_result.fun = -100.0
        self.mock_optimize_result.hess_inv = np.eye(3)  # Mocking Hessian inverse as identity matrix
        self.mock_optimize_result.jac = np.array([0.01, 0.02, 0.03])
        self.mock_optimize_result.success = True
        self.mock_optimize_result.message = "Optimization terminated successfully."
        
        # Additional parameters for GEVResult
        self.mock_endog = np.array([1.2, 2.3, 3.4, 4.5, 5.6])  
        self.mock_len_endog = len(self.endog)
        self.mock_trans = False
        self.mock_plot_data = self.mock_endog
        self.mock_gev_params = (1,1,1)

        self.mock_gev_result = GEVResult(self.mock_optimize_result, self.mock_endog, self.mock_len_endog, self.mock_trans,self.mock_plot_data,self.mock_gev_params)

    def test_initialization_with_dict_exog(self):
        """
        Test initialization with exog provided as a dictionary.
        """
        model = GEV(self.endog, exog=self.exog)
        self.assertEqual(len(model.endog), len(self.endog))
        self.assertTrue(np.array_equal(model.exog['shape'][:,1].reshape(-1,1), self.exog['shape']))
        self.assertTrue(np.array_equal(model.exog['scale'][:,1].reshape(-1,1), self.exog['scale']))
        self.assertTrue(np.array_equal(model.exog['location'][:,1].reshape(-1,1), self.exog['location']))
        self.assertTrue(model.trans)
    
    def test_initialization_with_single_exog(self):
        """
        Test initialization with a single exog array applied to all parameters.
        """
        exog = np.random.randn(100).reshape(-1, 1)
        model = GEV(self.endog, exog=exog)
        self.assertEqual(len(model.endog), len(self.endog))
        self.assertTrue(np.array_equal(model.exog['shape'][:,1].reshape(-1,1), exog))
        self.assertTrue(np.array_equal(model.exog['scale'][:,1].reshape(-1,1), exog))
        self.assertTrue(np.array_equal(model.exog['location'][:,1].reshape(-1,1), exog))
        self.assertTrue(model.trans)
    
    def test_initialization_with_no_exog(self):
        """
        Test initialization with no exog provided.
        """
        model = GEV(endog = self.endog)

        # Ensure `endog` is correctly assigned
        self.assertEqual(len(model.endog), len(self.endog))

        # Check that `exog_location`, `exog_shape`, and `exog_scale` default to vectors of ones of length equal to `endog`
        expected_length = len(self.endog)
        expected_default = np.ones(expected_length).reshape(-1,1)

        # Assert that exog_shape, exog_scale, and exog_location are not None and match the expected default
        np.testing.assert_array_equal(model.exog['location'], expected_default)
        np.testing.assert_array_equal(model.exog['scale'], expected_default)
        np.testing.assert_array_equal(model.exog['shape'], expected_default)

        # Verify the `trans` attribute (assuming it represents transformation, which should be False initially)
        self.assertFalse(model.trans)
    
    def test_link_functions(self):
        """
        Test default and custom link functions.
        """
        # Default link functions
        model = GEV(self.endog)
        self.assertEqual(model.loc_link(5), 5)
        self.assertEqual(model.scale_link(5), 5)
        self.assertEqual(model.shape_link(5), 5)

        # Custom link functions
        model = GEV(self.endog, loc_link=np.exp, scale_link=np.log, shape_link=np.sqrt)
        self.assertEqual(model.loc_link(1), np.exp(1))
        self.assertEqual(model.scale_link(np.exp(1)), 1)
        self.assertEqual(model.shape_link(4), 2)

    def test_missing_fit_and_predict(self):
        """
        Test that fit and predict methods raise NotImplementedError.
        """
        model = GEV(self.endog, exog=self.exog)
        with self.assertRaises(NotImplementedError):
            model.fit()
        with self.assertRaises(NotImplementedError):
            model.predict(None)
    
    def test_edge_case_empty_endog(self):
        """
        Test behavior when endog is an empty array.
        """
        with self.assertRaises(ValueError):
            GEV(None)
        with self.assertRaises(ValueError):
            GEV(np.array([]))
    
    def test_edge_case_mismatched_exog_length(self):
        """
        Test behavior when exog lengths do not match endog.
        """
        mismatched_exog = {
            'shape': np.random.rand(50),
            'scale': np.random.rand(50),
            'location': np.random.rand(50)
        }    
        with self.assertRaises(ValueError) as context:
            GEV(self.endog, exog=mismatched_exog)
        
        print("Caught error:", context.exception)
    
    def test_loglike(self):
        """
        Test the loglike method to ensure it returns an approriate loglike value with and without exog data
        """
        exog2 = {"location" : EOBS["tempanomalyMean"].values.reshape(-1,1)}
        model1 = GEVLikelihood(endog=EOBS["prmax"].values.reshape(-1,1))
        model2= GEVLikelihood(endog=EOBS["prmax"].values.reshape(-1,1), exog = exog2)
        params = np.array([55,12,0.2])
        nloglike_value = model1.nloglike(params)
        loglike_value = model1.loglike(params)
        self.assertEqual(loglike_value, -nloglike_value)
        nloglike_value_True = model1.nloglike(params)
        nloglike2_value_True = model2.nloglike(np.array([52,10,11.3,0.07])) 
        self.assertAlmostEqual(nloglike_value_True, 309, delta=1)
        self.assertAlmostEqual(nloglike2_value_True, 303, delta=1)

    
    def test_hessian_with_params(self):
        """
        Test the hessian method when parameters are provided.
        """
        model = GEVLikelihood(self.endog, exog=self.exog)
        params = np.random.rand(model.exog['location'].shape[1] + model.exog['scale'].shape[1] + model.exog['shape'].shape[1])
        hessian = model.hess(params)

        # Assert that the Hessian matrix has the correct shape (should be square)
        self.assertEqual(hessian.shape[0], len(params))
        self.assertEqual(hessian.shape[1], len(params))

    def test_hessian_not_fitted(self):
        """
        Test that the hessian method raises an exception when the model is not fitted and no parameters are provided.
        """
        model = GEVLikelihood(self.endog)
        with self.assertRaises(ValueError) as context:
            model.hess()
        self.assertEqual(str(context.exception), "Model is not fitted. Cannot compute the Hessian at optimal parameters.")

    def test_hessian_inverse(self):
        """
        Test the hess_inv method.
        """
        # Generate some random parameters for testing
        model = GEVLikelihood(endog=EOBS["prmax"].values.reshape(-1,1))
        # Compute the Hessian and its inverse
        hessian = model.hess(np.array([55,10,0.2]))
        hessian_inv = model.hess_inv(np.array([55,10,0.2]))

        # Verify that hessian_inv is the inverse of hessian (i.e., Hessian * Hessian^-1 should be close to identity matrix)
        identity_approx = np.dot(hessian, hessian_inv)
        identity = np.eye(len(np.array([55,10,0.2])))

        # Assert that the product is close to the identity matrix within a tolerance
        np.testing.assert_array_almost_equal(identity_approx, identity, decimal=5)

    def test_score_with_params(self):
        """
        Test the score method when parameters are provided.
        """
        model = GEVLikelihood(endog=EOBS["prmax"].values.reshape(-1,1))
        # Mocking the nloglike function for approx_fprime calculation
        params = np.array([0.5, 1.0, -0.3])
        model.nloglike = lambda x: np.sum(x**2)  # Mock nloglike as a simple function

        score = model.score(params=params)
        expected_score = approx_fprime(params, model.nloglike, 1e-5)
        np.testing.assert_almost_equal(score, expected_score, decimal=5)

    def test_score_when_not_fitted(self):
        """
        Test that the score method raises a ValueError when the model is not fitted and no parameters are provided.
        """
        # Set the fitted status to False
        self.likelihood.fitted = False

        # Verify that a ValueError is raised when attempting to compute the score without fitting
        with self.assertRaises(ValueError) as context:
            self.likelihood.score()
        
        self.assertEqual(str(context.exception), "Model is not fitted. Cannot compute the score at optimal parameters.")

    @patch('gev_copy.minimize')
    def test_fit_method(self, mock_minimize):
        """
        Test the fit method to ensure the optimization works correctly and results are stored properly.
        """
        # Mocking the return value of minimize
        mock_result = OptimizeResult()
        mock_result.x = np.array([0.5, 1.0, -0.3])
        mock_result.success = True
        mock_result.fun = -100.0
        mock_result.jac = np.array([0.01, 0.02, 0.03])
        mock_result.hess_inv = np.eye(3)  # Mocking Hessian inverse as identity matrix
        mock_minimize.return_value = mock_result

        # Define start_params for the fitting process
        start_params = np.array([0.1, 0.1, 0.1])

        # Fit the model
        result = self.likelihood.fit(start_params=start_params, maxiter=200)

        # Ensure that the fit method updates fitted attribute
        self.assertTrue(self.likelihood.fitted)

        # Ensure the mock minimize function was called with the correct arguments
        mock_minimize.assert_called_once_with(self.likelihood.nloglike, start_params, method='L-BFGS-B', maxiter=200)

        # Ensure result is of type GEVResults
        self.assertIsInstance(result, GEVResult)

        # Ensure the result contains expected fitted values
        self.assertTrue(hasattr(self.likelihood.result, 'x'))
        np.testing.assert_array_equal(self.likelihood.result.x, mock_result.x)

        # Ensure result attributes are set correctly
        np.testing.assert_array_equal(self.likelihood.result.endog, self.endog)
        self.assertEqual(self.likelihood.result.len_endog, len(self.endog))

    @patch('gev_copy.minimize')
    def test_fit_method_without_start_params(self, mock_minimize):
        """
        Test the fit method to ensure the optimization works correctly when start_params are not provided.
        """
        # Mocking the return value of minimize
        mock_result = OptimizeResult()
        mock_result.x = np.array([0.5, 1.0, -0.3])
        mock_result.success = True
        mock_result.fun = -100.0
        mock_result.jac = np.array([0.01, 0.02, 0.03])
        mock_result.hess_inv = np.eye(3)  # Mocking Hessian inverse as identity matrix
        mock_minimize.return_value = mock_result

        # Fit the model without specifying start_params
        result = self.likelihood.fit(start_params=None, maxiter=200)

        # Define expected default start_params based on the given fit method logic
        expected_start_params = np.array(
            [self.likelihood.location_guess] +
            ([0] * (self.likelihood.exog['location'].shape[1] - 1)) +
            [self.likelihood.scale_guess] +
            ([0] * (self.likelihood.exog['scale'].shape[1] - 1)) +
            [self.likelihood.shape_guess] +
            ([0] * (self.likelihood.exog['shape'].shape[1] - 1))
        )

        print(expected_start_params.shape)

        actual_call_args = mock_minimize.call_args
        actual_nloglike, actual_start_params = actual_call_args[0]  # Positional arguments
        actual_kwargs = actual_call_args[1]  # Keyword arguments

        # Manually compare the arrays using numpy
        self.assertTrue(np.array_equal(actual_start_params, expected_start_params))
        # Check the other arguments used in the call
        self.assertEqual(actual_nloglike, self.likelihood.nloglike)
        self.assertEqual(actual_kwargs['method'], 'L-BFGS-B')
        self.assertEqual(actual_kwargs['maxiter'], 200)

        # Ensure the fit method updates the fitted attribute
        self.assertTrue(self.likelihood.fitted)

        # Ensure result is of type GEVResult
        self.assertIsInstance(result, GEVResult)

        # Ensure the result contains expected fitted values
        self.assertTrue(hasattr(self.likelihood.result, 'x'))
        np.testing.assert_array_equal(self.likelihood.result.x, mock_result.x)

        # Ensure result attributes are set correctly
        np.testing.assert_array_equal(self.likelihood.result.endog, self.likelihood.endog)
        self.assertEqual(self.likelihood.result.len_endog, len(self.likelihood.endog))

    def test_aic(self):
        # AIC = 2 * nparams + 2 * likelihood value
        expected_aic = 2 * len(self.mock_optimize_result.x) + 2 * self.mock_optimize_result.fun
        calculated_aic = self.mock_gev_result.AIC()
        self.assertAlmostEqual(calculated_aic, expected_aic, places=4)

    def test_bic(self):
        # BIC = nparams * log(len_endog) + 2 * likelihood value
        expected_bic = len(self.mock_optimize_result.x) * np.log(self.mock_len_endog) + 2 * self.mock_optimize_result.fun
        calculated_bic = self.mock_gev_result.BIC()
        self.assertAlmostEqual(calculated_bic, expected_bic, places=4)

    def test_se(self):
        # Standard error is the sqrt of the diagonal of Hessian inverse
        expected_se = np.sqrt(np.diag(self.mock_optimize_result.hess_inv))
        calculated_se = self.mock_gev_result.SE()
        np.testing.assert_array_almost_equal(calculated_se, expected_se, decimal=4)

    def test_str(self):
        # Get the string representation of the GEVResult instance
        result_str = str(self.mock_gev_result)

        # Check for presence of header and summary information
        self.assertIn("EVT Results Summary", result_str)
        self.assertIn("AIC:", result_str)
        self.assertIn("BIC:", result_str)

        # Calculate expected AIC and BIC for verification
        expected_aic = 2 * len(self.mock_optimize_result.x) + 2 * self.mock_optimize_result.fun
        expected_bic = len(self.mock_optimize_result.x) * np.log(self.mock_len_endog) + 2 * self.mock_optimize_result.fun

        # Check that the calculated AIC and BIC are in the result string
        self.assertIn(f"AIC: {expected_aic:.2f}", result_str)
        self.assertIn(f"BIC: {expected_bic:.2f}", result_str)

        # Calculate standard errors, z-scores, and p-values
        se = np.sqrt(np.diag(self.mock_optimize_result.hess_inv))
        z_scores = self.mock_optimize_result.x / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

        # Check parameter estimates and related statistics in the output
        for i in range(len(self.mock_optimize_result.x)):
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
            ci_lower = self.mock_optimize_result.x[i] - 1.96 * se[i]
            ci_upper = self.mock_optimize_result.x[i] + 1.96 * se[i]

            # Check that each parameter's statistics are in the result string
            self.assertIn(f"{i+1:<10} {self.mock_optimize_result.x[i]:<10.4f} {se[i]:<7.4f} {z_scores[i]:<6.2f} "
                          f"{p_values[i]:<.4f}  ({ci_lower:.4f}, {ci_upper:.4f}) {signif}", result_str)

        # Check for presence of separator and note lines
        self.assertIn("Notes: *** p<0.001, ** p<0.01, * p<0.05", result_str)
        print(self.mock_gev_result)



if __name__ == "__main__":
    unittest.main()
