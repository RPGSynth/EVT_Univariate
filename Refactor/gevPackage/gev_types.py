import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

@dataclass
class GEVInput:
    """
    Strict immutable container for GEV data.
    Enforces (N, S, K) layout with zero ambiguity guessing.
    """
    endog: np.ndarray
    exog_loc: np.ndarray
    exog_scale: np.ndarray
    exog_shape: np.ndarray
    weights: np.ndarray
    
    n_obs: int = field(init=False)
    n_samples: int = field(init=False)

    def __post_init__(self):
        self.n_obs = self.endog.shape[0]
        self.n_samples = self.endog.shape[1]

    @classmethod
    def from_inputs(cls, endog: np.ndarray, exog: Optional[Dict[str, np.ndarray]], weights: Optional[np.ndarray]):
        # 1. Enforce Endog Shape (N_obs, N_samples)
        endog = np.asarray(endog, dtype=np.float64)
        
        # Transform 1D (N,) -> 2D (N, 1)
        if endog.ndim == 1:
            endog = endog[:, np.newaxis]
        
        if endog.ndim != 2:
            raise ValueError(f"Endog must be 2D (N_obs, N_samples). Got {endog.shape}")
        
        n_obs, n_samples = endog.shape

        # 2. Enforce Weights
        if weights is None:
            weights = np.ones_like(endog)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if weights.ndim == 0:
                weights = np.full_like(endog, weights)
            if weights.shape != endog.shape:
                 raise ValueError(f"Weights shape {weights.shape} must match endog {endog.shape}")

        # 3. Parse Exog Dictionary
        if exog is None: exog = {}
        
        exog_loc = cls._validate_covariate(exog.get('location'), n_obs, n_samples, "Location")
        exog_scale = cls._validate_covariate(exog.get('scale'), n_obs, n_samples, "Scale")
        exog_shape = cls._validate_covariate(exog.get('shape'), n_obs, n_samples, "Shape")

        return cls(endog, exog_loc, exog_scale, exog_shape, weights)

    @staticmethod
    def _validate_covariate(arr: Optional[np.ndarray], n_obs: int, n_samples: int, name: str) -> np.ndarray:
        """
        Validates covariate dimensions with STRICT matching rules.
        """
        # Create Intercept (N, S, 1)
        intercept = np.ones((n_obs, n_samples, 1), dtype=np.float64)

        if arr is None:
            return intercept

        arr = np.asarray(arr, dtype=np.float64)
        
        # --- Case 1: 1D Array (N,) ---
        # ALLOWED: Convenience for Time/Global vectors.
        # Broadcast: (N,) -> (N, 1, 1) -> Tiled to (N, S, 1)
        if arr.ndim == 1:
            if arr.shape[0] != n_obs:
                raise ValueError(f"{name} (1D) length mismatch. Expected {n_obs}, got {arr.shape[0]}.")
            user_cov = np.tile(arr[:, None, None], (1, n_samples, 1))

        # --- Case 2: 2D Array (Rows, Cols) ---
        # STRICT: Cols must equal n_samples. No guessing K.
        elif arr.ndim == 2:
            rows, cols = arr.shape
            if rows != n_obs:
                raise ValueError(f"{name} (2D) row mismatch. Expected {n_obs}, got {rows}.")
            
            if cols != n_samples:
                raise ValueError(
                    f"{name} (2D) dimension mismatch. \n"
                    f" - Endog has {n_samples} samples.\n"
                    f" - Input has {cols} columns.\n"
                    f"STRICT RULE: 2D covariates must match (N_obs, N_samples).\n"
                    f"If you intend to provide {cols} covariates per sample, please reshape to 3D: ({rows}, {n_samples}, {cols})."
                )
            
            # Match confirmed. Shape is (N, S). Reshape to (N, S, 1)
            user_cov = arr[:, :, np.newaxis]

        # --- Case 3: 3D Array (N, S, K) ---
        # STRICT: Must match N and S.
        elif arr.ndim == 3:
            if arr.shape[0] != n_obs:
                raise ValueError(f"{name} (3D) row mismatch. Expected {n_obs}, got {arr.shape[0]}.")
            if arr.shape[1] != n_samples:
                raise ValueError(f"{name} (3D) sample mismatch. Expected {n_samples}, got {arr.shape[1]}.")
            
            user_cov = arr
        
        else:
            raise ValueError(f"{name} has invalid dimensions: {arr.ndim}. Accepted: 1D, 2D (N,S), 3D (N,S,K).")

        # Concatenate Intercept along the last axis (K)
        return np.concatenate([intercept, user_cov], axis=2)

    @property
    def covariate_dims(self) -> Tuple[int, int, int]:
        return (self.exog_loc.shape[2], self.exog_scale.shape[2], self.exog_shape.shape[2])