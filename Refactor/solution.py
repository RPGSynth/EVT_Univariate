"""Solution object for fitted GEV models."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from .likelihood import GEVLikelihood


_FD_STEP = 1e-5


def _as_float_array(value: Sequence[float]) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _ensure_1d(array: np.ndarray) -> float:
    value = np.asarray(array, dtype=float)
    return float(np.atleast_1d(value).ravel()[0])


@dataclass
class GEVSolution:
    """Lightweight container describing a fitted GEV model."""

    likelihood: GEVLikelihood
    params: np.ndarray
    success: bool
    message: str
    nll: float
    optimizer: str
    raw_result: Any | None = None

    _hessian_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _score_outer_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _cov_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _cov_robust_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _surface_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.params = _as_float_array(self.params)
        if self.params.ndim != 1:
            raise ValueError("`params` must be a one-dimensional array.")
        if self.params.size != sum(self.likelihood.len_exog):
            raise ValueError(
                "Parameter vector length does not match the likelihood design."  # pragma: no cover
            )

    # ------------------------------------------------------------------
    # Parameter bookkeeping
    # ------------------------------------------------------------------
    def _split_params(self, params: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = self.params if params is None else _as_float_array(params)
        i, j, k = self.likelihood.len_exog
        loc = theta[0:i]
        scale = theta[i:i + j]
        shape = theta[i + j:i + j + k]
        return loc, scale, shape

    @property
    def param_blocks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the location, scale, and shape parameter blocks."""
        return self._split_params()

    @property
    def location_params(self) -> np.ndarray:
        return self.param_blocks[0]

    @property
    def scale_params(self) -> np.ndarray:
        return self.param_blocks[1]

    @property
    def shape_params(self) -> np.ndarray:
        return self.param_blocks[2]

    @property
    def n_params(self) -> int:
        return int(self.params.size)

    @property
    def n_obs(self) -> int:
        return int(self.likelihood.n_obs)

    @property
    def n_series(self) -> int:
        return int(self.likelihood.n_samples)

    @property
    def total_weight(self) -> float:
        return float(self.likelihood.weights.sum())

    @property
    def effective_sample_size(self) -> float:
        weights = self.likelihood.weights
        denom = float(np.sum(weights ** 2))
        if denom == 0:
            return 0.0
        return (self.total_weight ** 2) / denom

    # ------------------------------------------------------------------
    # Objective transformations
    # ------------------------------------------------------------------
    @property
    def total_nll(self) -> float:
        return float(self.nll * self.total_weight)

    @property
    def log_likelihood(self) -> float:
        return float(-self.total_nll)

    @property
    def aic(self) -> float:
        return 2 * self.n_params + 2 * self.total_nll

    @property
    def bic(self) -> float:
        n_eff = max(self.effective_sample_size, 1.0)
        return self.n_params * np.log(n_eff) + 2 * self.total_nll

    @property
    def tic(self) -> float:
        return 2 * self.total_nll + 2 * self._trace_jh()

    # ------------------------------------------------------------------
    # Numerical derivatives
    # ------------------------------------------------------------------
    def numerical_gradient(
        self,
        func: Callable[[np.ndarray], float],
        *,
        params: Optional[np.ndarray] = None,
        step: float = _FD_STEP,
    ) -> np.ndarray:
        theta = self.params if params is None else _as_float_array(params)
        grad = np.zeros_like(theta)
        if theta.size == 0:
            return grad
        for idx in range(theta.size):
            h = step if np.isfinite(theta[idx]) else _FD_STEP
            h = max(h, _FD_STEP * max(1.0, abs(theta[idx])))
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[idx] += h
            theta_minus[idx] -= h
            f_plus = func(theta_plus)
            f_minus = func(theta_minus)
            grad[idx] = (f_plus - f_minus) / (2 * h)
        return grad

    def _objective(self, theta: np.ndarray, *, weights: Optional[np.ndarray] = None) -> float:
        return float(self.likelihood.nloglike(theta, weights=weights))

    def _gradient(self, theta: Optional[np.ndarray] = None, *, weights: Optional[np.ndarray] = None) -> np.ndarray:
        theta_array = self.params if theta is None else _as_float_array(theta)
        return self.numerical_gradient(lambda prm: self._objective(prm, weights=weights), params=theta_array)

    def _hessian(self, *, weights: Optional[np.ndarray] = None) -> np.ndarray:
        if self.n_params == 0:
            return np.zeros((0, 0))
        theta = self.params
        hessian = np.zeros((self.n_params, self.n_params), dtype=float)
        for idx in range(self.n_params):
            h = max(_FD_STEP, _FD_STEP * max(1.0, abs(theta[idx])))
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[idx] += h
            theta_minus[idx] -= h
            grad_plus = self._gradient(theta_plus, weights=weights)
            grad_minus = self._gradient(theta_minus, weights=weights)
            hessian[:, idx] = (grad_plus - grad_minus) / (2 * h)
        return _symmetrize(hessian)

    @property
    def hessian(self) -> np.ndarray:
        if self._hessian_cache is None:
            self._hessian_cache = self._hessian()
        return self._hessian_cache

    def _score_outer(self) -> np.ndarray:
        if self._score_outer_cache is not None:
            return self._score_outer_cache
        if self.n_params == 0:
            self._score_outer_cache = np.zeros((0, 0))
            return self._score_outer_cache
        weights = self.likelihood.weights
        total_weight = self.total_weight
        j_matrix = np.zeros((self.n_params, self.n_params), dtype=float)
        for index, weight in np.ndenumerate(weights):
            if weight <= 0:
                continue
            mask = np.zeros_like(weights)
            mask[index] = weight
            grad_term = self._gradient(weights=mask)
            factor = (weight / total_weight) ** 2
            j_matrix += factor * np.outer(grad_term, grad_term)
        self._score_outer_cache = _symmetrize(j_matrix)
        return self._score_outer_cache

    def _trace_jh(self) -> float:
        if self.n_params == 0:
            return 0.0
        H_inv = self._pinv(self.hessian)
        return float(np.trace(self._score_outer() @ H_inv))

    # ------------------------------------------------------------------
    # Covariance summaries
    # ------------------------------------------------------------------
    @staticmethod
    def _pinv(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        return np.linalg.pinv(matrix, hermitian=True)

    @property
    def covariance_matrix(self) -> np.ndarray:
        if self._cov_cache is None:
            self._cov_cache = self._pinv(self.hessian)
        return self._cov_cache

    @property
    def covariance_matrix_robust(self) -> np.ndarray:
        if self._cov_robust_cache is None:
            H_inv = self._pinv(self.hessian)
            J = self._score_outer()
            self._cov_robust_cache = H_inv @ J @ H_inv
        return self._cov_robust_cache

    def standard_errors(self, *, robust: bool = True) -> np.ndarray:
        cov = self.covariance_matrix_robust if robust else self.covariance_matrix
        return np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))

    def confidence_intervals(self, *, alpha: float = 0.05, robust: bool = True) -> np.ndarray:
        z_crit = norm.ppf(1 - alpha / 2)
        se = self.standard_errors(robust=robust)
        lower = self.params - z_crit * se
        upper = self.params + z_crit * se
        return np.column_stack((lower, upper))

    def parameter_table(self, *, alpha: float = 0.05, robust: bool = True) -> np.ndarray:
        se = self.standard_errors(robust=robust)
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.divide(self.params, se, out=np.zeros_like(self.params), where=se > 0)
        p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
        ci = self.confidence_intervals(alpha=alpha, robust=robust)
        return np.column_stack((self.params, se, z_scores, p_values, ci))

    def summary(self, *, alpha: float = 0.05, robust: bool = True) -> str:
        stats = self.parameter_table(alpha=alpha, robust=robust)
        header = ["Parameter", "Estimate", "Std.Err", "Z", "P>|Z|", f"{100*(1-alpha):.1f}% CI"]
        lines = []
        lines.append(f"Optimizer: {self.optimizer}")
        lines.append(f"Converged: {self.success}")
        if self.message:
            lines.append(f"Message : {self.message}")
        lines.append("")
        lines.append(f"Log-likelihood: {self.log_likelihood:.4f}")
        lines.append(f"AIC: {self.aic:.4f}  BIC: {self.bic:.4f}")
        lines.append(f"TIC: {self.tic:.4f}")
        lines.append("")
        lines.append(" | ".join(header))
        lines.append("-" * 72)
        names = self._parameter_names()
        for name, row in zip(names, stats):
            est, se, z_score, p_val, lower, upper = row
            ci_str = f"({lower:.4f}, {upper:.4f})"
            lines.append(f"{name:<12} | {est:>9.4f} | {se:>8.4f} | {z_score:>6.2f} | {p_val:>7.4f} | {ci_str}")
        return "\n".join(lines)

    def _parameter_names(self) -> list[str]:
        names: list[str] = []
        len_loc, len_scale, len_shape = self.likelihood.len_exog
        prefix_loc = "zp" if self.likelihood.loc_return_level_reparam else "mu"
        names.extend(f"{prefix_loc}_{idx}" for idx in range(len_loc))
        names.extend(f"sigma_{idx}" for idx in range(len_scale))
        names.extend(f"xi_{idx}" for idx in range(len_shape))
        return names

    # ------------------------------------------------------------------
    # Predictive utilities
    # ------------------------------------------------------------------
    def _component_surfaces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self._surface_cache is not None:
            return self._surface_cache
        loc_params, scale_params, shape_params = self.param_blocks
        exog = self.likelihood.exog
        scale_linear = np.einsum('nij,j->ni', exog['scale'], scale_params)
        shape_linear = np.einsum('nij,j->ni', exog['shape'], shape_params)
        scale = self.likelihood.scale_link(scale_linear)
        shape = self.likelihood.shape_link(shape_linear)
        if self.likelihood.loc_return_level_reparam:
            zp_linear = np.einsum('nij,j->ni', exog['location'], loc_params)
            zp = self.likelihood.loc_link(zp_linear)
            location = self.likelihood._zp_to_location(loc_params, scale, shape)
            self._surface_cache = (location, scale, shape, zp)
        else:
            loc_linear = np.einsum('nij,j->ni', exog['location'], loc_params)
            location = self.likelihood.loc_link(loc_linear)
            self._surface_cache = (location, scale, shape, None)
        return self._surface_cache

    @property
    def fitted_location(self) -> np.ndarray:
        return self._component_surfaces()[0]

    @property
    def fitted_scale(self) -> np.ndarray:
        return self._component_surfaces()[1]

    @property
    def fitted_shape(self) -> np.ndarray:
        return self._component_surfaces()[2]

    @property
    def fitted_return_level_surface(self) -> Optional[np.ndarray]:
        return self._component_surfaces()[3]

    def _predict_triplet(self, params: np.ndarray, t: int, s: int) -> Tuple[float, float, float]:
        loc_params, scale_params, shape_params = self._split_params(params)
        exog = self.likelihood.exog
        loc_linear = float(np.dot(exog['location'][t, s, :], loc_params))
        scale_linear = float(np.dot(exog['scale'][t, s, :], scale_params))
        shape_linear = float(np.dot(exog['shape'][t, s, :], shape_params))
        location = _ensure_1d(self.likelihood.loc_link(np.array([loc_linear])))
        scale = _ensure_1d(self.likelihood.scale_link(np.array([scale_linear])))
        shape = _ensure_1d(self.likelihood.shape_link(np.array([shape_linear])))
        return location, scale, shape

    # ------------------------------------------------------------------
    # Return level interface
    # ------------------------------------------------------------------
    def return_level(self, *, confidence: float = 0.95, robust: bool = True) -> 'ReturnLevelCalculatorBase':
        if self.likelihood.loc_return_level_reparam:
            return ReturnLevelCalculatorReparam(self, confidence=confidence, robust=robust)
        return ReturnLevelCalculatorStandard(self, confidence=confidence, robust=robust)

    def get_return_levels(
        self,
        T: Sequence[float],
        *,
        t: Optional[Sequence[int]] = None,
        s: Optional[Sequence[int]] = None,
        confidence: float = 0.95,
        robust: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        calculator = self.return_level(confidence=confidence, robust=robust)
        return calculator.compute(T=T, t=t, s=s)

    def plot_return_levels(
        self,
        T: Sequence[float],
        *,
        t: Optional[Sequence[int]] = None,
        s: Optional[Sequence[int]] = None,
        confidence: float = 0.95,
        robust: bool = True,
        show_ci: bool = True,
        **kwargs: Any,
    ) -> Any:
        calculator = self.return_level(confidence=confidence, robust=robust)
        plotter = ReturnLevelPlotter(calculator)
        return plotter.time_plot(T=T, t=t, s=s, show_ci=show_ci, **kwargs)


class ReturnLevelCalculatorBase:
    def __init__(self, solution: GEVSolution, *, confidence: float = 0.95, robust: bool = True) -> None:
        self.solution = solution
        self.confidence = confidence
        self.robust = robust

    def _covariance(self) -> np.ndarray:
        return self.solution.covariance_matrix_robust if self.robust else self.solution.covariance_matrix

    def _z_critical(self) -> float:
        return norm.ppf(1 - (1 - self.confidence) / 2)

    def _normalise_indices(self, idx: Optional[Sequence[int]], upper: int) -> np.ndarray:
        if idx is None:
            return np.arange(upper)
        arr = np.atleast_1d(idx).astype(int)
        if (arr < 0).any() or (arr >= upper).any():
            raise IndexError("Index out of bounds for return level computation.")
        return arr

    def compute(
        self,
        T: Sequence[float],
        *,
        t: Optional[Sequence[int]] = None,
        s: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        periods = np.atleast_1d(T).astype(float)
        t_idx = self._normalise_indices(t, self.solution.n_obs)
        s_idx = self._normalise_indices(s, self.solution.n_series)
        len_T, len_t, len_s = len(periods), len(t_idx), len(s_idx)
        estimates = np.empty((len_T, len_t, len_s), dtype=float)
        lower = np.empty_like(estimates)
        upper = np.empty_like(estimates)
        for i_T, T_val in enumerate(periods):
            for i_t, t_val in enumerate(t_idx):
                for i_s, s_val in enumerate(s_idx):
                    estimate, (ci_low, ci_high) = self.return_level_at(T_val, int(t_val), int(s_val))
                    estimates[i_T, i_t, i_s] = estimate
                    lower[i_T, i_t, i_s] = ci_low
                    upper[i_T, i_t, i_s] = ci_high
        return estimates, lower, upper

    def return_level_at(self, T: float, t: int, s: int) -> Tuple[float, Tuple[float, float]]:
        raise NotImplementedError


class ReturnLevelCalculatorStandard(ReturnLevelCalculatorBase):
    def _gev_return_level(self, mu: float, sigma: float, xi: float, T: float) -> float:
        y_p = -np.log(1 - 1 / T)
        if np.isclose(xi, 0.0):
            return mu - sigma * np.log(y_p)
        return mu - (sigma / xi) * (1 - y_p ** (-xi))

    def _value(self, params: np.ndarray, T: float, t: int, s: int) -> float:
        mu, sigma, xi = self.solution._predict_triplet(params, t, s)
        return self._gev_return_level(mu, sigma, xi, T)

    def return_level_at(self, T: float, t: int, s: int) -> Tuple[float, Tuple[float, float]]:
        value = self._value(self.solution.params, T, t, s)
        grad = self.solution.numerical_gradient(lambda prm: self._value(prm, T, t, s))
        cov = self._covariance()
        variance = float(grad @ cov @ grad)
        variance = max(variance, 0.0)
        se = float(np.sqrt(variance))
        z_crit = self._z_critical()
        return value, (value - z_crit * se, value + z_crit * se)


class ReturnLevelCalculatorReparam(ReturnLevelCalculatorBase):
    def _value(self, params: np.ndarray, t: int, s: int) -> float:
        loc_params, _, _ = self.solution._split_params(params)
        exog_loc = self.solution.likelihood.exog['location'][t, s, :]
        linear = float(np.dot(exog_loc, loc_params))
        return _ensure_1d(self.solution.likelihood.loc_link(np.array([linear])))

    def return_level_at(self, T: float, t: int, s: int) -> Tuple[float, Tuple[float, float]]:
        value = self._value(self.solution.params, t, s)
        grad = self.solution.numerical_gradient(lambda prm: self._value(prm, t, s))
        cov = self._covariance()
        variance = float(grad @ cov @ grad)
        variance = max(variance, 0.0)
        se = float(np.sqrt(variance))
        z_crit = self._z_critical()
        return value, (value - z_crit * se, value + z_crit * se)


class ReturnLevelPlotter:
    def __init__(self, calculator: ReturnLevelCalculatorBase) -> None:
        self.calculator = calculator

    def time_plot(
        self,
        T: Sequence[float],
        *,
        t: Optional[Sequence[int]] = None,
        s: Optional[Sequence[int]] = None,
        show_ci: bool = True,
        ax: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required for plotting return levels.") from exc

        periods = np.atleast_1d(T).astype(float)
        estimates, lower, upper = self.calculator.compute(T=periods, t=t, s=s)
        t_idx = self.calculator._normalise_indices(t, self.calculator.solution.n_obs)
        s_idx = self.calculator._normalise_indices(s, self.calculator.solution.n_series)

        if s_idx.size > 1:
            warnings.warn("Multiple series provided; plotting the first one only.", UserWarning)
        s_pos = int(s_idx[0]) if s_idx.size else 0

        if ax is None:
            _, ax = plt.subplots()

        for idx, t_val in enumerate(t_idx if t_idx.size else [0]):
            series = estimates[:, idx, s_pos]
            ax.plot(periods, series, marker='o', label=f"t={int(t_val)}", **kwargs)
            if show_ci:
                ax.fill_between(periods, lower[:, idx, s_pos], upper[:, idx, s_pos], alpha=0.2)

        ax.set_xlabel("Return period")
        ax.set_ylabel("Return level")
        ax.legend()
        return ax
