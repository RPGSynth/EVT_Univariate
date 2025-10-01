"""High-level facade for the refactored GEV model."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .likelihood import GEVLikelihood, LinkFunc, AddIntercept
from .optimizers import FitOutcome, OptimizerStrategy, SciPyMLE
from .solution import GEVSolution


class GEV:
    """Facade that composes :class:GEVLikelihood with optimisation backends."""

    def __init__(
        self,
        endog: np.ndarray,
        exog: Optional[Dict[str, Optional[np.ndarray]]] = None,
        weights: Optional[np.ndarray] = None,
        *,
        loc_link: Optional[LinkFunc] = None,
        scale_link: Optional[LinkFunc] = None,
        shape_link: Optional[LinkFunc] = None,
        T: Optional[float] = None,
        add_intercept: AddIntercept = True,
    ) -> None:
        self.likelihood = GEVLikelihood(
            endog=endog,
            exog=exog,
            weights=weights,
            loc_link=loc_link,
            scale_link=scale_link,
            shape_link=shape_link,
            T=T,
            add_intercept=add_intercept,
        )

    @property
    def endog(self) -> np.ndarray:
        return self.likelihood.endog

    @property
    def exog(self) -> Dict[str, np.ndarray]:
        return self.likelihood.exog

    @property
    def weights(self) -> np.ndarray:
        return self.likelihood.weights

    @property
    def len_exog(self) -> tuple[int, int, int]:
        return self.likelihood.len_exog

    def nloglike(self, params: np.ndarray, *, weights: Optional[np.ndarray] = None) -> float:
        return self.likelihood.nloglike(params, weights=weights)

    def fit(
        self,
        optimizer: Optional[OptimizerStrategy] = None,
        *,
        start_params: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> GEVSolution:
        """Fit the model using the provided optimiser and return a solution."""
        optimiser = optimizer or SciPyMLE()
        outcome: FitOutcome = optimiser.fit(
            self.likelihood,
            start_params=start_params,
            weights=weights,
        )
        return GEVSolution(
            likelihood=self.likelihood,
            params=outcome.params,
            success=outcome.success,
            message=outcome.message,
            nll=outcome.fun,
            optimizer=optimiser.__class__.__name__,
            raw_result=outcome.raw_result,
        )


__all__ = ["GEV"]
