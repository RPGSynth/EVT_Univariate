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
import gstools as gs
import matplotlib.pyplot as plt

__all__ = [
    "linear_time_covariate",
    "late_exp_curve",
    "linear_space_covariate",
    "generate_gev_dataset",
    "plot_random_time_series",
    "plot_random_spatial_slice",
]


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------
class SpaceTimeMeta(TypedDict):
    """Metadata returned alongside simulated data.

    Keys
    ----
    x, y : ndarray
        1D coordinate arrays with lengths ``n_lon`` and ``n_lat`` respectively.
    t : ndarray
        1D array of time indices with length ``n_time``.
    s_field : ndarray
        Standardized spatial covariate with shape ``(n_lat, n_lon)``.
    t_curve : ndarray
        Standardized temporal covariate with shape ``(n_time,)``.
    mu : ndarray
        Location parameter, shape ``(n_time, n_lat, n_lon)``.
    sigma : ndarray
        Scale parameter (>0), shape ``(n_time, n_lat, n_lon)``.
    xi : float
        Shape parameter of the GEV distribution.
    params : Dict[str, object]
        Dictionary of input parameters used for the simulation.
    """


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Return a z-scored copy of ``arr`` (zero mean, unit variance).

    If the standard deviation is zero, a zero array of the same shape is returned.
    """
    arr = np.asarray(arr, dtype=float)
    sd = arr.std()
    if sd == 0:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / sd


# -----------------------------------------------------------------------------
# Covariates
# -----------------------------------------------------------------------------


def linear_time_covariate(
    n_time: int,
    slope: float = 1.0,
    intercept: float = 0.0,
    standardize: bool = True,
) -> np.ndarray:
    """Linear-in-time covariate ``t'``.

    ``t'`` is ``intercept + slope * (t / (n_time - 1))`` for ``t=0..n_time-1``.
    Optionally standardized via z-scoring.
    """
    if n_time < 1:
        raise ValueError("n_time must be >= 1")
    t = np.arange(n_time, dtype=float)
    u = t / max(1.0, n_time - 1)
    t_raw = intercept + slope * u
    return _zscore(t_raw) if standardize else t_raw


def late_exp_curve(
    n_time: int,
    onset: float = 0.0,
    p: float = 3.0,
    alpha: float = 0.25,
) -> np.ndarray:
    """Late-onset exponential temporal covariate (standardized).

    Flat early and accelerates near the end; returned array is z-scored.
    """
    if n_time < 1:
        raise ValueError("n_time must be >= 1")
    t = np.arange(n_time, dtype=float)
    u = t / max(1.0, n_time - 1)
    v = np.clip((u - onset) / max(1e-12, 1 - onset), 0, 1)
    y_raw = np.expm1(alpha * (v**p))
    return _zscore(y_raw)


def linear_space_covariate(
    x: np.ndarray,
    y: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    standardize: bool = True,
) -> np.ndarray:
    """Planar spatial covariate ``s' = a x + b y`` over the grid.

    Uses ``indexing='xy'`` so the returned array has shape ``(n_lat, n_lon)``.
    """
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float), indexing="xy")
    s_raw = a * X + b * Y
    return _zscore(s_raw) if standardize else s_raw


def _gstools_spatial_field(
    x: np.ndarray,
    y: np.ndarray,
    *,
    nu_L: float = 1.5,
    var_L: float = 1.0,
    len_Lx: float = 80.0,
    len_Ly: float = 60.0,
    angle_deg: float = 30.0,
    use_multiscale: bool = True,
    var_M: float = 0.4,
    len_Mx: float = 60.0,
    len_My: float = 40.0,
    var_S: float = 0.1,
    len_Sx: float = 15.0,
    len_Sy: float = 10.0,
    seed: int = 2025,
) -> np.ndarray:
    """gstools-based spatial covariate, optionally multi-scale; standardized.

    Returns an array of shape ``(n_lat, n_lon)`` matching ``(y, x)``.
    """
    def matern_field(var, len_x, len_y, nu, angle_deg, seed_offset=0):
        model = gs.Matern(
            dim=2, var=var, len_scale=[len_x, len_y], nu=nu, angles=np.deg2rad(angle_deg)
        )
        srf = gs.SRF(model, seed=seed + seed_offset)
        return srf.structured((x, y))

    s_L = matern_field(var_L, len_Lx, len_Ly, nu_L, angle_deg, seed_offset=1)
    if use_multiscale:
        s_M = matern_field(var_M, len_Mx, len_My, 1.0, angle_deg, seed_offset=2)
        s_S = matern_field(var_S, len_Sx, len_Sy, 0.5, angle_deg, seed_offset=3)
        s_raw = s_L + s_M + s_S
    else:
        s_raw = s_L
    return _zscore(s_raw)


# -----------------------------------------------------------------------------
# Single generator
# -----------------------------------------------------------------------------


def generate_gev_dataset(
    n_lat: int = 50,
    n_lon: int = 50,
    n_time: int = 120,
    Lx: float = 100.0,
    Ly: float = 100.0,
    *,
    # regime selectors
    temporal: str = "linear",
    spatial: str = "linear",
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
    # gstools space params
    nu_L: float = 1.5,
    var_L: float = 1.0,
    len_Lx: float = 80.0,
    len_Ly: float = 60.0,
    angle_deg: float = 30.0,
    use_multiscale: bool = True,
    var_M: float = 0.4,
    len_Mx: float = 60.0,
    len_My: float = 40.0,
    var_S: float = 0.1,
    len_Sx: float = 15.0,
    len_Sy: float = 10.0,
    # regression
    beta_mu0: float = 10.0,
    beta_mu_s: float = 1.0,
    beta_mu_t: float = 0.5,
    beta_ls0: float = 0,
    beta_ls_s: float = 0.0,
    beta_ls_t: float = 0.3,
    # GEV
    xi: float = 0.1,
    seed: int = 2025,
) -> Tuple[np.ndarray, SpaceTimeMeta]:
    """Simulate a space–time GEV dataset with configurable covariate regimes.

    The temporal covariate is chosen with ``temporal`` (``"linear"`` or ``"late_exp"``) and the spatial
    covariate with ``spatial`` (``"linear"`` or ``"gstools"``). Linear regimes are the defaults.

    Parameters
    ----------
    n_lat, n_lon, n_time : int
        Grid size and number of time steps (all >= 1).
    Lx, Ly : float
        Extents of the grid along x and y used to build coordinates via ``linspace``.
    temporal : {"linear", "late_exp"}
        Temporal covariate regime.
    spatial : {"linear", "gstools"}
        Spatial covariate regime.
    t_slope, t_intercept : float
        Linear time parameters (used if ``temporal='linear'``).
    t_onset, t_p, t_alpha : float
        Late-exponential time parameters (used if ``temporal='late_exp'``).
    a, b : float
        Linear space parameters for ``s' = a x + b y`` (used if ``spatial='linear'``).
    nu_L, var_L, len_Lx, len_Ly, angle_deg, use_multiscale, var_M, len_Mx, len_My, var_S, len_Sx, len_Sy
        gstools spatial parameters (used if ``spatial='gstools'``).
    beta_mu0, beta_mu_s, beta_mu_t : float
        Coefficients for the location parameter ``mu``.
    beta_ls0, beta_ls_s, beta_ls_t : float
        Coefficients for the log-scale parameter ``log(sigma)``.
    xi : float
        GEV shape parameter.
    seed : int
        RNG seed used in sampling.

    Returns
    -------
    data : ndarray
        Simulated field of shape ``(n_time, n_lat, n_lon)``.
    meta : SpaceTimeMeta
        Metadata containing covariates, parameters, and coordinates.

    Examples
    --------
    >>> # default: linear time + linear space
    >>> data, meta = generate_gev_dataset(n_lat=20, n_lon=30, n_time=40)
    >>> # gstools spatial + late exponential temporal
    >>> data2, meta2 = generate_gev_dataset(
    ...     n_lat=20, n_lon=30, n_time=40,
    ...     spatial="gstools", temporal="late_exp",
    ...     nu_L=1.0, var_L=1.0, len_Lx=50, len_Ly=40,
    ...     t_onset=0.2, t_p=3.0, t_alpha=0.3,
    ... )
    """
    if n_lat < 1 or n_lon < 1 or n_time < 1:
        raise ValueError("n_lat, n_lon, n_time must be >= 1")

    if temporal not in {"linear", "late_exp"}:
        raise ValueError("temporal must be 'linear' or 'late_exp'")
    if spatial not in {"linear", "gstools"}:
        raise ValueError("spatial must be 'linear' or 'gstools'")

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, Lx, n_lon)
    y = np.linspace(0.0, Ly, n_lat)

    # temporal covariate
    if temporal == "linear":
        t_curve = linear_time_covariate(n_time, slope=t_slope, intercept=t_intercept, standardize=True)
    else:
        t_curve = late_exp_curve(n_time, onset=t_onset, p=t_p, alpha=t_alpha)

    # spatial covariate
    if spatial == "linear":
        s_field = linear_space_covariate(x, y, a=a, b=b, standardize=True)
    else:
        s_field = _gstools_spatial_field(
            x,
            y,
            nu_L=nu_L,
            var_L=var_L,
            len_Lx=len_Lx,
            len_Ly=len_Ly,
            angle_deg=angle_deg,
            use_multiscale=use_multiscale,
            var_M=var_M,
            len_Mx=len_Mx,
            len_My=len_My,
            var_S=var_S,
            len_Sx=len_Sx,
            len_Sy=len_Sy,
            seed=seed,
        )

    # regression and sampling
    s3 = s_field[None, :, :]
    t3 = t_curve[:, None, None]

    mu = beta_mu0 + beta_mu_s * s3 + beta_mu_t * t3
    log_sigma = beta_ls0 + 0*(beta_ls_s * s3 + beta_ls_t * t3)
    sigma = np.exp(log_sigma)

    data = np.empty((n_time, n_lat, n_lon))
    c = -xi  # SciPy's shape parameterization
    for ti in range(n_time):
        data[ti] = genextreme.rvs(c=c, loc=mu[ti], scale=sigma[ti], random_state=rng)

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
            "Lx": Lx,
            "Ly": Ly,
            "temporal": temporal,
            "spatial": spatial,
            "t_slope": t_slope,
            "t_intercept": t_intercept,
            "t_onset": t_onset,
            "t_p": t_p,
            "t_alpha": t_alpha,
            "a": a,
            "b": b,
            "nu_L": nu_L,
            "var_L": var_L,
            "len_Lx": len_Lx,
            "len_Ly": len_Ly,
            "angle_deg": angle_deg,
            "use_multiscale": use_multiscale,
            "var_M": var_M,
            "len_Mx": len_Mx,
            "len_My": len_My,
            "var_S": var_S,
            "len_Sx": len_Sx,
            "len_Sy": len_Sy,
            "beta_mu0": beta_mu0,
            "beta_mu_s": beta_mu_s,
            "beta_mu_t": beta_mu_t,
            "beta_ls0": beta_ls0,
            "beta_ls_s": beta_ls_s,
            "beta_ls_t": beta_ls_t,
            "xi": xi,
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
    data, meta = generate_gev_dataset(n_lat=50, n_lon=50, n_time=120)
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
