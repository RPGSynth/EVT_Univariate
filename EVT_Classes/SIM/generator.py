"""
Space–time GEV simulator with a single, flexible generator and clean API.

This module simulates Generalized Extreme Value (GEV) fields over a 2D spatial grid evolving in time.
It exposes **one** generator, :func:`generate_gev_dataset`, which supports multiple covariate regimes
via keywords while defaulting to *linear time* and *linear space*.

Covariate regimes
-----------------
Temporal (``temporal=``)
  - ``"linear"`` *(default)* — linearly increasing covariate (z-scored).
  - ``"late_exp"`` — late-onset exponential (z-scored).

Spatial (``spatial=``)
  - ``"linear"`` *(default)* — planar field ``s' = a x + b y`` (z-scored).
  - ``"gstools"`` — gstools Matern field; optional multi-scale composition (z-scored).

Both covariates enter linearly into the GEV location and log-scale. Sampling uses
``scipy.stats.genextreme`` with the SciPy parameterization ``c = -xi``.

Exports
-------
- linear_time_covariate(n_time, slope, intercept, standardize)
- late_exp_curve(n_time, onset, p, alpha)
- linear_space_covariate(x, y, a, b, standardize)
- generate_gev_dataset(...)
- plot_random_time_series(data, meta, seed)
- plot_random_spatial_slice(data, meta, seed)

Requires: numpy, scipy, gstools, matplotlib
"""

from __future__ import annotations

from typing import Dict, Tuple, TypedDict

import numpy as np
from scipy.stats import genextreme
import matplotlib.pyplot as plt

__all__ = [
    "generate_gev_dataset_linear",
    "plot_random_time_series",
    "plot_random_spatial_slice",
]

# ----------------------------------------------------------------------------
# Covariates
# -----------------------------------------------------------------------------

def linear_time_covariate(n_time: int, *, slope: float, intercept: float, standardize: bool = True) -> np.ndarray:
    t = slope * np.arange(n_time) + intercept
    if standardize:
        t = (t - t.mean()) / (t.std() + 1e-12)
    return t

def late_exp_curve(n_time: int, *, onset: float, p: float, alpha: float) -> np.ndarray:
    # Simple late-exponential-like curve on [0,1] with onset ∈ [0,1)
    u = np.linspace(0.0, 1.0, n_time)
    z = np.where(u < onset, 0.0, ((u - onset) / max(1e-12, 1 - onset)) ** p)
    z = (np.exp(alpha * z) - 1.0) / max(1e-12, alpha)
    z = (z - z.mean()) / (z.std() + 1e-12)
    return z

def linear_space_covariate(x: np.ndarray, y: np.ndarray, *, a: float, b: float, standardize: bool = True) -> np.ndarray:
    X, Y = np.meshgrid(x, y, indexing="xy")
    s = a * X + b * Y
    if standardize:
        s = (s - s.mean()) / (s.std() + 1e-12)
    return s

class SpaceTimeMeta(TypedDict):
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    s_field: np.ndarray
    t_curve: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    xi: np.ndarray
    params: dict

def generate_gev_dataset_linear(
    n_lat: int = 50,
    n_lon: int = 50,
    n_time: int = 120,
    Lx: float = 100.0,
    Ly: float = 100.0,
    *,
    # temporal regime: "linear" or "late_exp"
    temporal: str = "linear",
    # linear time params
    t_slope: float = 1.0,
    t_intercept: float = 0.0,
    # late-exp time params
    t_onset: float = 0.0,
    t_p: float = 3.0,
    t_alpha: float = 0.25,
    # linear space params
    a: float = 1.0,
    b: float = 1.0,
    # regression (mu, log-sigma, xi)
    beta_mu0: float = 10.0,
    beta_mu_s: float = 1.0,
    beta_mu_t: float = 0.5,
    beta_ls0: float = 0.0,
    beta_ls_s: float = 0.5,
    beta_ls_t: float = 0.0,
    beta_xi0: float = 0.1,
    beta_xi_s: float = 0.0,
    beta_xi_t: float = 0.0,
    # RNG
    seed: int = 2025,
) -> Tuple[np.ndarray, SpaceTimeMeta]:
    """
    Simulate a space–time GEV dataset (linear covariates in space and time only).

    Temporal covariate can be 'linear' or 'late_exp'; spatial covariate is a linear plane.
    Parameters for mu, log(sigma), and xi are linear in those covariates.

    By default, xi has no covariate effect and equals 0.1 everywhere.

    Returns
    -------
    data : (n_time, n_lat, n_lon)
    meta : dict with fields x, y, t, s_field, t_curve, mu, sigma, xi, params
    """
    if n_lat < 1 or n_lon < 1 or n_time < 1:
        raise ValueError("n_lat, n_lon, n_time must be >= 1")
    if temporal not in {"linear", "late_exp"}:
        raise ValueError("temporal must be 'linear' or 'late_exp'")

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, Lx, n_lon)
    y = np.linspace(0.0, Ly, n_lat)

    # temporal covariate
    if temporal == "linear":
        t_curve = linear_time_covariate(n_time, slope=t_slope, intercept=t_intercept, standardize=True)
    else:
        t_curve = late_exp_curve(n_time, onset=t_onset, p=t_p, alpha=t_alpha)

    # spatial covariate (always linear)
    s_field = linear_space_covariate(x, y, a=a, b=b, standardize=True)

    # broadcast covariates to (t, i, j)
    s3 = s_field[None, :, :]
    t3 = t_curve[:, None, None]

    # regression
    mu = beta_mu0 + beta_mu_s * s3 + beta_mu_t * t3
    log_sigma = beta_ls0 + beta_ls_s * s3 + beta_ls_t * t3  # varies only if you set nonzero betas
    sigma = np.exp(log_sigma)
    xi = beta_xi0 + beta_xi_s * s3 + beta_xi_t * t3        # default: constant 0.1

    # sampling: SciPy genextreme uses c = -xi
    data = np.empty((n_time, n_lat, n_lon))
    for ti in range(n_time):
        data[ti] = genextreme.rvs(c=-xi[ti], loc=mu[ti], scale=sigma[ti], random_state=rng)

    meta: SpaceTimeMeta = {
        "x": x,
        "y": y,
        "t": np.arange(n_time),
        "s_field": s_field,
        "t_curve": t_curve,
        "mu": mu,
        "sigma": sigma,
        "xi": xi,
        "params": {
            "Lx": Lx, "Ly": Ly,
            "temporal": temporal,
            "t_slope": t_slope, "t_intercept": t_intercept,
            "t_onset": t_onset, "t_p": t_p, "t_alpha": t_alpha,
            "a": a, "b": b,
            "beta_mu0": beta_mu0, "beta_mu_s": beta_mu_s, "beta_mu_t": beta_mu_t,
            "beta_ls0": beta_ls0, "beta_ls_s": beta_ls_s, "beta_ls_t": beta_ls_t,
            "beta_xi0": beta_xi0, "beta_xi_s": beta_xi_s, "beta_xi_t": beta_xi_t,
            "seed": seed,
        },
    }
    return data, meta

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_random_time_series(
    data: np.ndarray,
    meta: SpaceTimeMeta | None = None,
    seed: int = 123,
) -> None:
    """Plot a random location's time series.

    ``meta`` improves titles and axis labels when provided.
    """
    rng = np.random.default_rng(seed)
    n_time, n_lat, n_lon = data.shape
    i = int(rng.integers(0, n_lat))
    j = int(rng.integers(0, n_lon))

    t = meta["t"] if (meta and "t" in meta) else np.arange(n_time)
    loc_str = f"[i={i}, j={j}]"
    if meta and "x" in meta and "y" in meta:
        loc_str = f"(y={float(np.asarray(meta['y'])[i]):.2f}, x={float(np.asarray(meta['x'])[j]):.2f}) [i={i}, j={j}]"

    plt.figure()
    plt.plot(t, data[:, i, j])
    plt.xlabel("time")
    plt.ylabel("GEV value")
    plt.title(f"Time series at random location {loc_str}")
    plt.tight_layout()
    plt.show()


def plot_random_spatial_slice(
    data: np.ndarray,
    meta: SpaceTimeMeta | None = None,
    seed: int = 456,
) -> None:
    """Plot a random time slice as a 2D field.

    ``meta`` improves titles and axis labels when provided.
    """
    rng = np.random.default_rng(seed)
    n_time, _, _ = data.shape
    k = int(rng.integers(0, n_time))

    field = data[k]
    vmax = float(np.quantile(field[np.isfinite(field)], 0.98))

    extent = None
    title = f"Spatial slice at index t={k}"
    if meta and "x" in meta and "y" in meta:
        x, y = np.asarray(meta["x"]), np.asarray(meta["y"])
        extent = (x.min(), x.max(), y.min(), y.max())
        title = f"Spatial slice at t={meta['t'][k] if 't' in meta else k} (index {k})"

    plt.figure()
    im = plt.imshow(
        field,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="none",
        vmax=vmax,
    )
    plt.colorbar(im, label="GEV value")
    plt.xlabel("x" if extent is not None else "lon index")
    plt.ylabel("y" if extent is not None else "lat index")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# CLI/demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # default: linear time + linear space
    data, meta = generate_gev_dataset_linear(n_lat=50, n_lon=50, n_time=120)
    print(
        "data:", data.shape,
        "| s_field sd:", np.std(meta["s_field"]).round(3),
        "| t_curve sd:", np.std(meta["t_curve"]).round(3),
    )
    plot_random_time_series(data, meta)
    plot_random_spatial_slice(data, meta)

    # alternative regime example
    # data2, meta2 = generate_gev_dataset(
    #     n_lat=50, n_lon=50, n_time=120,
    #     spatial="gstools", temporal="late_exp",
    #     t_onset=0.2, t_p=3.0, t_alpha=0.3,
    # )
