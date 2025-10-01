import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from Refactor.likelihood import GEVLikelihood
from Refactor.solution import GEVSolution


def build_solution():
    rng = np.random.default_rng(42)
    n_obs, n_series = 5, 2
    endog = rng.normal(size=(n_obs, n_series))
    exog = {'location': None, 'scale': None, 'shape': np.zeros((n_obs, n_series))}
    likelihood = GEVLikelihood(endog=endog, exog=exog)
    params = likelihood.initial_params()
    success = True
    message = ""
    nll = likelihood.nloglike(params)
    return GEVSolution(likelihood, params, success, message, nll, "MockOptimizer")


def test_solution_blocks():
    solution = build_solution()
    loc, scale, shape = solution.param_blocks
    assert loc.shape[0] == solution.likelihood.len_exog[0]
    assert scale.shape[0] == solution.likelihood.len_exog[1]
    assert shape.shape[0] == solution.likelihood.len_exog[2]
    assert solution.success
    assert np.isfinite(solution.nll)
