import unittest
import numpy as np
import pandas as pd
from gev_copy import GEV, GEVLikelihood # Replace with the actual module name where GEV is defined


EOBS = pd.read_csv(r"c:\ThesisData\OUTPUTS\UCL_blockmax.csv")
TRUE_PARAMS = np.array([55,12,0.2])

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
        self.single_exog = np.random.rand(100).reshape(-1, 1)  # Single exog applied to all parameters
        self.gev = GEV(self.endog, exog=self.exog)
        self.like = GEVLikelihood(self.endog, exog=self.exog)
        self.trueLike = GEVLikelihood(EOBS["prmax"].values.reshape(-1,1))
    
    def test_initialization_with_dict_exog(self):
        """
        Test initialization with exog provided as a dictionary.
        """
        model = GEV(self.endog, exog=self.exog)
        self.assertEqual(len(model.endog), len(self.endog))
        self.assertTrue(np.array_equal(model.exog_shape, self.exog['shape']))
        self.assertTrue(np.array_equal(model.exog_scale, self.exog['scale']))
        self.assertTrue(np.array_equal(model.exog_location, self.exog['location']))
        self.assertTrue(model.trans)
    
    def test_initialization_with_single_exog(self):
        """
        Test initialization with a single exog array applied to all parameters.
        """
        model = GEV(self.endog, exog=self.single_exog)
        self.assertEqual(len(model.endog), len(self.endog))
        self.assertTrue(np.array_equal(model.exog_shape, self.single_exog))
        self.assertTrue(np.array_equal(model.exog_scale, self.single_exog))
        self.assertTrue(np.array_equal(model.exog_location, self.single_exog))
        self.assertTrue(model.trans)
    
    def test_initialization_with_no_exog(self):
        """
        Test initialization with no exog provided.
        """
        model = GEV(self.endog)

        # Ensure `endog` is correctly assigned
        self.assertEqual(len(model.endog), len(self.endog))

        # Check that `exog_location`, `exog_shape`, and `exog_scale` default to vectors of ones of length equal to `endog`
        expected_length = len(self.endog)
        expected_default = np.ones(expected_length).reshape(-1,1)

        # Assert that exog_shape, exog_scale, and exog_location are not None and match the expected default
        np.testing.assert_array_equal(model.exog_location, expected_default)
        np.testing.assert_array_equal(model.exog_scale, expected_default)
        np.testing.assert_array_equal(model.exog_shape, expected_default)

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
    
    def test_data_processing(self):
        """
        Test the _process_data method for correct processing.
        """
        model = GEV(self.endog, exog=self.exog)
        data = model._process_data(self.endog, self.exog)
        self.assertTrue(np.array_equal(data['endog'], self.endog))
        self.assertTrue(np.array_equal(data['exog']['shape'], self.exog['shape']))
        self.assertTrue(np.array_equal(data['exog']['scale'], self.exog['scale']))
        self.assertTrue(np.array_equal(data['exog']['location'], self.exog['location']))

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
            GEV(None, exog=self.exog)
        with self.assertRaises(ValueError):
            GEV(np.array([]), exog=self.exog)
    
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
        params = np.random.rand(self.like.exog_location.shape[1] + self.like.exog_scale.shape[1] + self.like.exog_shape.shape[1])
        nloglike_value = self.like.nloglike(params)
        loglike_value = self.like.loglike(params)
        self.assertEqual(loglike_value, -nloglike_value)
        nloglike_value_True = self.trueLike.nloglike(TRUE_PARAMS)
        self.assertAlmostEqual(nloglike_value_True, 309, delta=1)

    
    def test_hessian_with_params(self):
        """
        Test the hessian method when parameters are provided.
        """
        model = GEVLikelihood(self.endog, exog=self.exog)
        params = np.random.rand(model.exog_location.shape[1] + model.exog_scale.shape[1] + model.exog_shape.shape[1])
        hessian = model.hess(params)

        # Assert that the Hessian matrix has the correct shape (should be square)
        self.assertEqual(hessian.shape[0], len(params))
        self.assertEqual(hessian.shape[1], len(params))

    def test_hessian_not_fitted(self):
        """
        Test that the hessian method raises an exception when the model is not fitted and no parameters are provided.
        """
        with self.assertRaises(ValueError) as context:
            self.like.hess()
        self.assertEqual(str(context.exception), "Model is not fitted. Cannot compute the Hessian at optimal parameters.")

    def test_hessian_inverse(self):
        """
        Test the hess_inv method.
        """
        # Generate some random parameters for testing
        model = GEVLikelihood(endog=EOBS["prmax"].values.reshape(-1,1))
        # Compute the Hessian and its inverse
        hessian = model.hess(TRUE_PARAMS)
        print(hessian)
        hessian_inv = model.hess_inv(TRUE_PARAMS)

        # Verify that hessian_inv is the inverse of hessian (i.e., Hessian * Hessian^-1 should be close to identity matrix)
        identity_approx = np.dot(hessian, hessian_inv)
        identity = np.eye(len(TRUE_PARAMS))

        # Assert that the product is close to the identity matrix within a tolerance
        np.testing.assert_array_almost_equal(identity_approx, identity, decimal=5)


if __name__ == "__main__":
    unittest.main()
