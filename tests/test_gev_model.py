import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest


from Refactor.model import GEV


def build_dataset():
    rng = np.random.default_rng(1234)
    n_obs, n_series = 5, 2
    endog = rng.normal(size=(n_obs, n_series))
    exog = {
        'location': None,
        'scale': np.ones((n_obs, n_series)),
        'shape': np.zeros((n_obs, n_series)),
    }
    return endog, exog


def test_gev_end_to_end():
    endog, exog = build_dataset()
    model = GEV(endog=endog, exog=exog, add_intercept={'location': True, 'scale': False, 'shape': True})
    solution = model.fit()

    assert solution.success
    assert solution.params.shape[0] == sum(model.len_exog)
    loc, scale, shape = solution.param_blocks
    assert loc.shape[0] == model.len_exog[0]
    assert scale.shape[0] == model.len_exog[1]
    assert shape.shape[0] == model.len_exog[2]

if __name__ == '__main__':
    pytest.main([__file__])