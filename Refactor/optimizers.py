"""Optimization strategies for the refactored GEV model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize

from .likelihood import GEVLikelihood


@dataclass
class FitOutcome:
    """Container for optimisation results."""

    params: np.ndarray
    success: bool
    message: str = ""
    fun: float = np.nan
    raw_result: Any | None = None


class OptimizerStrategy(ABC):
    """Base class for optimisation strategies."""

    @abstractmethod
    def fit(
        self,
        likelihood: GEVLikelihood,
        *,
        start_params: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> FitOutcome:
        """Optimise the likelihood and return the fitted parameters."""


class SciPyMLE(OptimizerStrategy):
    """Maximum Likelihood Estimation via :func:scipy.optimize.minimize."""

    def __init__(
        self,
        method: str = "L-BFGS-B",
        bounds: Optional[list[tuple[float | None, float | None]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.method = method
        self.bounds = bounds
        self.options = options or {"maxiter": 500}

    def fit(
        self,
        likelihood: GEVLikelihood,
        *,
        start_params: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> FitOutcome:
        theta0 = np.asarray(start_params, dtype=float) if start_params is not None else likelihood.initial_params()

        def objective(theta: np.ndarray) -> float:
            return likelihood.nloglike(theta, weights=weights)

        result = minimize(
            objective,
            theta0,
            method=self.method,
            bounds=self.bounds,
            options=self.options,
        )

        params = np.asarray(result.x, dtype=float)
        fun = float(result.fun) if result.fun is not None else np.nan
        return FitOutcome(
            params=params,
            success=bool(result.success),
            message=result.message,
            fun=fun,
            raw_result=result,
        )


class JAXMLE(OptimizerStrategy):
    """Placeholder for a JAX-based optimiser."""

    def fit(
        self,
        likelihood: GEVLikelihood,
        *,
        start_params: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> FitOutcome:
        raise NotImplementedError("JAXMLE optimiser has not been implemented yet.")


class ProfileLikelihood(OptimizerStrategy):
    """Placeholder for a profile-likelihood optimiser."""

    def fit(
        self,
        likelihood: GEVLikelihood,
        *,
        start_params: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> FitOutcome:
        raise NotImplementedError("ProfileLikelihood optimiser has not been implemented yet.")
