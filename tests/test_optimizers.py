import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from Refactor.likelihood import GEVLikelihood
from Refactor.optimizers import SciPyMLE


def build_design():
    rng = np.random.default_rng(123)
    n_obs, n_series = 6, 2
    endog = rng.normal(size=(n_obs, n_series))
    exog = {
        'location': np.ones((n_obs, n_series)),
        'scale': np.ones((n_obs, n_series)),
        'shape': np.zeros((n_obs, n_series)),
    }
    return endog, exog


def test_scipy_mle_runs():
    endog, exog = build_design()
    likelihood = GEVLikelihood(endog=endog, exog=exog, add_intercept=False)
    optimizer = SciPyMLE(options={'maxiter': 50})
    outcome = optimizer.fit(likelihood)

    assert outcome.success
    assert outcome.params.shape[0] == likelihood.nparams
    assert np.isfinite(outcome.fun)

if __name__ == '__main__':
    pytest.main([__file__])
