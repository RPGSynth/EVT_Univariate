"""Likelihood-focused components for the refactored GEV model.

Target architecture:
    GEV -> GEVLikelihood + OptimizerStrategy -> GEVSolution -> ReturnLevelCalculator/Plotter
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

try:  # Optional JAX support mirrors the current module behaviour
    import jax
    import jax.numpy as jnp
    from jax import jit
except ImportError:  # pragma: no cover - JAX is optional
    jax = None  # type: ignore
    jnp = None  # type: ignore

    def jit(func=None, **_ignored):  # type: ignore
        return func


LinkFunc = Callable[[np.ndarray], np.ndarray]
ExogDict = Dict[str, Optional[np.ndarray]]
AddIntercept = Union[bool, Dict[str, bool]]


@dataclass
class ProcessedDesign:
    """Container for validated endogenous/exogenous data."""

    endog: np.ndarray  # (n_obs, n_series)
    exog: Dict[str, np.ndarray]  # each (n_obs, n_series, n_covariates)
    weights: np.ndarray  # (n_obs, n_series)
    len_exog: Tuple[int, int, int]
    trans: bool


class GEVLikelihood:
    """Encapsulates data preparation and likelihood evaluation for GEV models."""

    def __init__(
        self,
        endog: np.ndarray,
        exog: Optional[ExogDict] = None,
        weights: Optional[np.ndarray] = None,
        *,
        loc_link: Optional[LinkFunc] = None,
        scale_link: Optional[LinkFunc] = None,
        shape_link: Optional[LinkFunc] = None,
        T: Optional[float] = None,
        add_intercept: AddIntercept = True,
    ) -> None:
        self.loc_link = loc_link or self.identity
        self.scale_link = scale_link or self.identity
        self.shape_link = shape_link or self.identity
        self.T = T
        self.loc_return_level_reparam = T is not None and T > 1
        self._add_intercept = self._normalise_add_intercept(add_intercept)

        self._design = self._build_design(endog=endog, exog=exog, weights=weights)
        self.n_obs, self.n_samples = self._design.endog.shape
        self.nparams = sum(self._design.len_exog)

        endog_first = self._design.endog[:, 0]
        endog_mean = float(np.nanmean(endog_first))
        endog_var = float(np.nanvar(endog_first))
        euler_gamma = 0.5772156649

        self.scale_guess = max(np.sqrt(6 * endog_var) / np.pi, 1e-6)
        self.shape_guess = 0.1
        if self.loc_return_level_reparam:
            if T is None:
                raise ValueError("Return period T must be provided when reparameterising the location.")
            y_p_guess = -np.log(1 - 1 / T)
            mu_gumbel_guess = endog_mean - euler_gamma * self.scale_guess
            self.location_guess = mu_gumbel_guess - self.scale_guess * np.log(y_p_guess)
        else:
            self.location_guess = endog_mean - euler_gamma * self.scale_guess

    @property
    def endog(self) -> np.ndarray:
        return self._design.endog

    @property
    def exog(self) -> Dict[str, np.ndarray]:
        return self._design.exog

    @property
    def weights(self) -> np.ndarray:
        return self._design.weights

    @property
    def len_exog(self) -> Tuple[int, int, int]:
        return self._design.len_exog

    @property
    def trans(self) -> bool:
        return self._design.trans

    def nloglike(
        self,
        params: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        params = self._as_float_array(params, "params", ndim=1)
        weights_array = self.weights if weights is None else self._as_float_array(weights, "weights", ndim=2)
        if weights_array.shape != self.endog.shape:
            raise ValueError("`weights` must have shape (n_obs, n_series).")
        if np.any(weights_array < 0) or not np.all(np.isfinite(weights_array)):
            raise ValueError("`weights` must be finite and non-negative.")

        total_weight = float(weights_array.sum())
        if total_weight <= 0:
            raise ValueError("Sum of weights must be positive.")

        i, j, k = self.len_exog
        loc_params = params[0:i]
        scale_params = params[i:i + j]
        shape_params = params[i + j:i + j + k]

        scale = self.scale_link(np.einsum('nij,j->ni', self.exog['scale'], scale_params))
        if np.any(scale <= 1e-9):
            return 1e7

        shape_vals = self.shape_link(np.einsum('nij,j->ni', self.exog['shape'], shape_params))

        if self.loc_return_level_reparam:
            location = self._zp_to_location(loc_params, scale, shape_vals)
        else:
            location = self.loc_link(np.einsum('nij,j->ni', self.exog['location'], loc_params))

        if np.any(~np.isfinite(location)):
            return 1e7

        normalized_data = (self.endog - location) / scale
        transformed_data = 1 + shape_vals * normalized_data
        is_gumbel = np.isclose(shape_vals, 0)

        invalid_domain = (weights_array > 0) & (~is_gumbel) & (transformed_data <= 1e-9)
        if np.any(invalid_domain):
            return 1e7

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            log_scale = np.log(scale)
            gumbel_term = log_scale + normalized_data + np.exp(-normalized_data)
            log_transformed = np.log(transformed_data)
            gev_term = log_scale + transformed_data ** (-1 / shape_vals) + (1 + 1 / shape_vals) * log_transformed
            n_ll_terms = np.where(is_gumbel, gumbel_term, gev_term)

        if not np.all(np.isfinite(n_ll_terms)):
            return 1e7

        return float(np.sum(n_ll_terms * weights_array) / total_weight)

    def initial_params(self) -> np.ndarray:
        """Heuristic starting values for optimisation."""
        i, j, k = self.len_exog
        total = i + j + k
        params = np.zeros(total, dtype=float)
        offset = 0

        if i > 0:
            if self._should_add_intercept('location'):
                params[offset] = self.location_guess
            offset += i

        if j > 0:
            if self._should_add_intercept('scale'):
                params[offset] = self._inverse_link_initial_value(self.scale_link, self.scale_guess)
            offset += j

        if k > 0 and self._should_add_intercept('shape'):
            params[offset] = self._inverse_link_initial_value(self.shape_link, self.shape_guess)

        return params

    def _build_design(
        self,
        *,
        endog: np.ndarray,
        exog: Optional[ExogDict],
        weights: Optional[np.ndarray],
    ) -> ProcessedDesign:
        endog_array = self._as_float_array(endog, "endog", ndim=2)
        if np.isnan(endog_array).any():
            raise ValueError("`endog` contains NaN values. Please clean the data before fitting.")

        if weights is None:
            weights_array = np.ones_like(endog_array, dtype=float)
        else:
            weights_array = self._as_float_array(weights, "weights", ndim=2)
            if weights_array.shape != endog_array.shape:
                raise ValueError("`weights` must have shape (n_obs, n_series).")

        exog_dict = self._process_exog(endog_array, exog)
        len_exog = (
            exog_dict['location'].shape[2],
            exog_dict['scale'].shape[2],
            exog_dict['shape'].shape[2],
        )
        trans = any(block.shape[2] > int(self._should_add_intercept(name)) for name, block in exog_dict.items())

        return ProcessedDesign(
            endog=endog_array,
            exog=exog_dict,
            weights=weights_array,
            len_exog=len_exog,
            trans=trans,
        )

    def _process_exog(self, endog: np.ndarray, exog_input: Optional[ExogDict]) -> Dict[str, np.ndarray]:
        param_names = ['location', 'scale', 'shape']
        if exog_input is not None and not isinstance(exog_input, dict):
            raise TypeError("`exog` must be a dict mapping parameter names to ndarray values or None.")

        result: Dict[str, np.ndarray] = {}
        for name in param_names:
            block = None if exog_input is None else exog_input.get(name)
            result[name] = self._prepare_exog_block(block, endog, name)

        if exog_input:
            invalid = set(exog_input.keys()) - set(param_names)
            if invalid:
                raise ValueError(f"Invalid keys in exog dictionary: {invalid}.")
        return result

    def _prepare_exog_block(self, block: Optional[np.ndarray], endog: np.ndarray, name: str) -> np.ndarray:
        n_obs, n_series = endog.shape
        add_intercept = self._should_add_intercept(name)

        if block is None:
            if not add_intercept:
                raise ValueError(
                    f"exog['{name}'] must be provided when add_intercept is False for that parameter."
                )
            return np.ones((n_obs, n_series, 1), dtype=float)

        arr = self._as_float_array(block, f"exog['{name}']", ndim=None)
        if arr.ndim == 2:
            if arr.shape != endog.shape:
                raise ValueError(
                    f"2D exog['{name}'] must have shape (n_obs, n_series); got {arr.shape}."
                )
            arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3:
            if arr.shape[0] != n_obs or arr.shape[1] != n_series:
                raise ValueError(
                    f"3D exog['{name}'] must have shape (n_obs, n_series, n_covariates); got {arr.shape}."
                )
        else:
            raise ValueError(
                f"exog['{name}'] must be 2D or 3D ndarray with shape (n_obs, n_series [, n_covariates])."
            )

        if np.isnan(arr).any():
            raise ValueError(f"exog['{name}'] contains NaN values after processing.")

        if add_intercept:
            intercept = np.ones((n_obs, n_series, 1), dtype=float)
            arr = np.concatenate([intercept, arr], axis=2)
        return arr

    def _inverse_link_initial_value(self, link: LinkFunc, value: float) -> float:
        name = getattr(link, '__name__', '')
        if link is self.identity or name == "identity":
            return float(value)
        if name == "exp" or link is np.exp:
            safe_value = max(float(value), 1e-6)
            return float(np.log(safe_value))
        return float(value)

    def _normalise_add_intercept(self, add_intercept: AddIntercept) -> Dict[str, bool]:
        param_names = ['location', 'scale', 'shape']
        if isinstance(add_intercept, bool):
            return {name: add_intercept for name in param_names}
        if not isinstance(add_intercept, dict):
            raise TypeError("add_intercept must be a bool or a dict keyed by parameter names.")
        mapping = {name: add_intercept.get(name, True) for name in param_names}
        invalid = set(add_intercept.keys()) - set(param_names)
        if invalid:
            raise ValueError(f"Invalid keys in add_intercept dict: {invalid}.")
        return mapping

    def _should_add_intercept(self, name: str) -> bool:
        return self._add_intercept.get(name, True)

    @staticmethod
    def _as_float_array(value: np.ndarray, label: str, *, ndim: Optional[int]) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{label} must be a NumPy ndarray.")
        arr = np.asarray(value, dtype=float)
        if ndim is not None and arr.ndim != ndim:
            raise ValueError(f"{label} must have {ndim} dimensions; got {arr.ndim}.")
        if arr.size == 0:
            raise ValueError(f"{label} must contain data.")
        return arr

    def _zp_to_location(self, loc_params: np.ndarray, scale: np.ndarray, shape_vals: np.ndarray) -> np.ndarray:
        if self.T is None:
            raise ValueError("T must be set when using return-level reparameterisation.")
        zp = self.loc_link(np.einsum('nij,j->ni', self.exog['location'], loc_params))
        if np.any(~np.isfinite(zp)):
            return np.full_like(zp, np.nan)
        y_p = -np.log(1 - 1 / self.T)
        shape_zero_mask = np.isclose(shape_vals, 0)
        location = np.where(
            shape_zero_mask,
            zp + scale * np.log(y_p),
            zp + scale * (1 - y_p ** (-shape_vals)) / shape_vals,
        )
        return location

    @staticmethod
    def identity(x: np.ndarray) -> np.ndarray:
        return x


__all__ = ["GEVLikelihood", "ProcessedDesign", "LinkFunc", "AddIntercept"]

