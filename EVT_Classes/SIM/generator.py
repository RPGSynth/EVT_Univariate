"""
Space–time GEV simulator with a single, flexible generator and clean API.

This module simulates Generalized Extreme Value (GEV) fields over a 2D spatial grid evolving in time.
It exposes **one** generator, :func:`generate_gev_dataset`, which supports multiple covariate regimes
via keywords while defaulting to *linear time* and *linear space*.

Covariate regimes
-----------------
Temporal: fixed to a linear covariate on [0, 1].
Spatial: either a linear plane on [0, 1] or a gstools Matern field on [0, 1].

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
import gstools as gs
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
        t_min, t_max = t.min(), t.max()
        t = (t - t_min) / (t_max - t_min + 1e-12)
    return t

def linear_space_covariate(x: np.ndarray, y: np.ndarray, *, a: float, b: float, standardize: bool = True) -> np.ndarray:
    X, Y = np.meshgrid(x, y, indexing="xy")
    s = a * X + b * Y
    if standardize:
        s_min, s_max = s.min(), s.max()
        s = (s - s_min) / (s_max - s_min + 1e-12)
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
    # linear time params
    t_slope: float = 1.0,
    t_intercept: float = 0.0,
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
    # optional xi noise
    xi_noise: bool = False,
    xi_noise_amp: float = 0.1,
    # RNG
    seed: int = 2025,
) -> Tuple[np.ndarray, SpaceTimeMeta]:
    """
    Simulate a space–time GEV dataset (linear covariates in space and time only).

    Temporal covariate is linear on [0, 1]; spatial covariate is a linear plane on [0, 1].
    Parameters for mu, log(sigma), and xi are linear in those covariates.

    By default, xi has no covariate effect and equals 0.1 everywhere.

    Returns
    -------
    data : (n_time, n_lat, n_lon)
    meta : dict with fields x, y, t, s_field, t_curve, mu, sigma, xi, params
    """
    if n_lat < 1 or n_lon < 1 or n_time < 1:
        raise ValueError("n_lat, n_lon, n_time must be >= 1")

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, Lx, n_lon)
    y = np.linspace(0.0, Ly, n_lat)

    # temporal covariate (always linear, scaled to [0, 1])
    t_curve = linear_time_covariate(n_time, slope=t_slope, intercept=t_intercept, standardize=True)

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

    # optional additive noise on xi (±xi_noise_amp)
    if xi_noise:
        xi = xi + rng.choice([-xi_noise_amp, xi_noise_amp], size=xi.shape)

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
            "t_slope": t_slope, "t_intercept": t_intercept,
            "a": a, "b": b,
            "beta_mu0": beta_mu0, "beta_mu_s": beta_mu_s, "beta_mu_t": beta_mu_t,
            "beta_ls0": beta_ls0, "beta_ls_s": beta_ls_s, "beta_ls_t": beta_ls_t,
            "beta_xi0": beta_xi0, "beta_xi_s": beta_xi_s, "beta_xi_t": beta_xi_t,
            "xi_noise": xi_noise, "xi_noise_amp": xi_noise_amp,
            "seed": seed,
        },
    }
    return data, meta

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def generate_gev_dataset_blobs(
    n_lat: int = 50,
    n_lon: int = 50,
    n_time: int = 100,
    Lx: float = 100.0,
    Ly: float = 100.0,
    *,
    # --- Temporal Params (Linear) ---
    t_slope: float = 1.0,
    t_intercept: float = 0.0,
    
    # --- Spatial Params (Blobs/Anisotropy) ---
    # len_scale: [lat_scale, lon_scale] for anisotropy
    blob_scale_x: float = 20.0, 
    blob_scale_y: float = 8.0, 
    blob_angle: float = np.pi / 4, 
    blob_smoothness: float = 1.5, 
    blob_variance: float = 1.0,

    # --- Regression Params ---
    beta_mu0: float = 10.0,
    beta_mu_s: float = 2.0,   
    beta_mu_t: float = 0.5,   
    beta_ls0: float = 0.0,
    beta_ls_s: float = 0.5,   
    beta_ls_t: float = 0.0,
    beta_xi0: float = 0.1,
    beta_xi_s: float = 0.0,   
    beta_xi_t: float = 0.0,
    xi_noise: bool = False,
    xi_noise_amp: float = 0.1,
    
    seed: int = 2025,
) -> Tuple[np.ndarray, SpaceTimeMeta]:
    """
    Simulates a Space-Time GEV dataset using Gaussian Random Fields for space.
    Corrected to fix gstools TypeError.
    """
    if n_lat < 1 or n_lon < 1 or n_time < 1:
        raise ValueError("Dims must be >= 1")

    # 1. Setup Grid
    # We define axis vectors, not a meshgrid yet
    x = np.linspace(0.0, Lx, n_lon)
    y = np.linspace(0.0, Ly, n_lat)
    
    # 2. Generate Temporal Covariate (Linear)
    t_curve = linear_time_covariate(n_time, slope=t_slope, intercept=t_intercept, standardize=True)

    # 3. Generate Spatial Covariate (Anisotropic Random Field)
    # Note: gstools uses [y_scale, x_scale] convention often, but we can align 
    # it by passing axes explicitly.
    model = gs.Matern(
        dim=2, 
        var=blob_variance, 
        len_scale=[blob_scale_y, blob_scale_x], # [y, x] is standard for dim=2 in some versions, but check angle
        angles=blob_angle, 
        nu=blob_smoothness
    )
    
    srf = gs.SRF(model, seed=seed)

    s_field_raw = srf.structured([y, x]) 
    
    # Standardize
    s_min, s_max = s_field_raw.min(), s_field_raw.max()
    s_field = (s_field_raw - s_min) / (s_max - s_min + 1e-12)

    # 4. Regression (Linear combination)
    # Broadcast to (t, y, x)
    s3 = s_field[None, :, :]      # (1, n_lat, n_lon)
    t3 = t_curve[:, None, None]   # (n_time, 1, 1)

    mu = beta_mu0 + beta_mu_s * s3 + beta_mu_t * t3
    log_sigma = beta_ls0 + beta_ls_s * s3 + beta_ls_t * t3
    sigma = np.exp(log_sigma)
    xi = beta_xi0 + beta_xi_s * s3 + beta_xi_t * t3
    if xi_noise:
        xi = xi + rng.choice([-xi_noise_amp, xi_noise_amp], size=xi.shape)

    # 5. Sampling
    rng = np.random.default_rng(seed)
    data = np.empty((n_time, n_lat, n_lon))
    
    for ti in range(n_time):
        data[ti] = genextreme.rvs(c=-xi[ti], loc=mu[ti], scale=sigma[ti], random_state=rng)

    # 6. Metadata
    meta: SpaceTimeMeta = {
        "x": x, "y": y, "t": np.arange(n_time),
        "s_field": s_field,
        "t_curve": t_curve,
        "mu": mu, "sigma": sigma, "xi": xi,
        "params": {
            "type": "blob_anisotropic_gstools",
            "blob_scale": [blob_scale_x, blob_scale_y],
            "blob_angle": blob_angle,
            "seed": seed,
            "beta_xi0": beta_xi0, "beta_xi_s": beta_xi_s, "beta_xi_t": beta_xi_t,
            "xi_noise": xi_noise, "xi_noise_amp": xi_noise_amp,
        }
    }
    
    return data, meta

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
