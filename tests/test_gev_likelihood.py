import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from Refactor.likelihood import GEVLikelihood


def build_simple_design(add_intercept=True):
    rng = np.random.default_rng(0)
    n_obs, n_series = 5, 3
    endog = rng.normal(size=(n_obs, n_series))

    exog = {
        'location': None,
        'scale': None,
        'shape': np.zeros((n_obs, n_series))
    }

    return endog, exog, add_intercept


def test_gev_likelihood_default_intercept():
    endog, exog, add_intercept = build_simple_design(add_intercept=True)
    likelihood = GEVLikelihood(endog=endog, exog=exog, add_intercept=add_intercept)

    assert likelihood.endog.shape == endog.shape
    for key, block in likelihood.exog.items():
        assert block.shape[:2] == endog.shape
        assert block.shape[2] >= 1  # intercept exists

    params = np.zeros(likelihood.nparams)
    value = likelihood.nloglike(params)
    assert np.isfinite(value)


def test_gev_likelihood_no_intercept():
    endog, exog, _ = build_simple_design(add_intercept=False)
    for key in exog:
        if exog[key] is None:
            exog[key] = np.zeros_like(endog)

    likelihood = GEVLikelihood(endog=endog, exog=exog, add_intercept=False)

    for block in likelihood.exog.values():
        assert block.shape[:2] == endog.shape
        assert block.shape[2] == 1  # only provided covariate

    params = np.zeros(likelihood.nparams)
    value = likelihood.nloglike(params)
    assert np.isfinite(value)


def test_mixed_intercept_settings():
    rng = np.random.default_rng(1)
    n_obs, n_series = 4, 2
    endog = rng.normal(size=(n_obs, n_series))

    exog = {
        'location': None,
        'scale': rng.normal(size=(n_obs, n_series)),
        'shape': None,
    }

    likelihood = GEVLikelihood(
        endog=endog,
        exog=exog,
        add_intercept={'location': True, 'scale': False, 'shape': True},
    )

    assert likelihood.exog['location'].shape[2] == 1  # intercept only
    assert likelihood.exog['scale'].shape[2] == 1  # provided covariate only
    assert likelihood.exog['shape'].shape[2] == 1  # intercept only

    assert likelihood.nparams == sum(likelihood.len_exog)


def test_invalid_exog_type():
    endog, _, _ = build_simple_design()
    with pytest.raises(TypeError):
        GEVLikelihood(endog=endog, exog=[None, None, None])  # not a dict


def test_invalid_add_intercept_type():
    endog, exog, _ = build_simple_design()
    with pytest.raises(TypeError):
        GEVLikelihood(endog=endog, exog=exog, add_intercept=123)


def test_nan_endog_raises():
    endog, exog, _ = build_simple_design()
    endog[0, 0] = np.nan
    with pytest.raises(ValueError):
        GEVLikelihood(endog=endog, exog=exog)


def test_exog_shape_mismatch_raises():
    endog, exog, _ = build_simple_design()
    bad_exog = exog.copy()
    bad_exog['location'] = np.zeros((endog.shape[0], endog.shape[1] - 1))
    with pytest.raises(ValueError):
        GEVLikelihood(endog=endog, exog=bad_exog)


def test_weights_override_changes_output():
    endog, exog, _ = build_simple_design()
    likelihood = GEVLikelihood(endog=endog, exog=exog, scale_link=np.exp)

    params = np.zeros(likelihood.nparams)
    base_value = likelihood.nloglike(params)
    rng = np.random.default_rng(42)
    weights = rng.uniform(0.5, 1.5, size=endog.shape)
    override_value = likelihood.nloglike(params, weights=weights)

    assert np.isfinite(override_value)
    assert override_value != base_value


def test_len_exog_matches_exog_shapes():
    endog, exog, _ = build_simple_design()
    likelihood = GEVLikelihood(endog=endog, exog=exog)

    for idx, key in enumerate(['location', 'scale', 'shape']):
        assert likelihood.len_exog[idx] == likelihood.exog[key].shape[2]



def test_initial_params_matches_structure():
    endog, exog, _ = build_simple_design(add_intercept=True)
    likelihood = GEVLikelihood(endog=endog, exog=exog)

    params = likelihood.initial_params()
    assert params.shape[0] == likelihood.nparams
    assert params[0] == pytest.approx(likelihood.location_guess)


if __name__ == '__main__':
    pytest.main([__file__])
