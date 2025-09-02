"""
Space–time GEV simulator with gstools spatial covariate + quick sanity plots.

Requires: numpy, scipy, gstools, matplotlib
Exports:
  - late_exp_curve(n_time, onset, p, alpha)
  - generate_gev_dataset(...)
  - plot_random_time_series(data, meta, seed=123)
  - plot_random_spatial_slice(data, meta, seed=456)
"""

from __future__ import annotations
import numpy as np
from scipy.stats import genextreme
import gstools as gs
import matplotlib.pyplot as plt

# ---------------------- temporal covariate ----------------------

def late_exp_curve(n_time: int, onset: float = 0.0, p: float = 3.0, alpha: float = 0.25) -> np.ndarray:
    """Late-onset exponential (flat early, accelerates near the end). Returns standardized (n_time,) array."""
    t = np.arange(n_time, dtype=float)
    u = t / max(1.0, n_time - 1)
    v = np.clip((u - onset) / max(1e-12, 1 - onset), 0, 1)
    y_raw = np.expm1(alpha * (v ** p))
    return (y_raw - y_raw.mean()) / y_raw.std()

# ------------------------ data generator -----------------------

def generate_gev_dataset(
    n_lat: int = 50, n_lon: int = 50, n_time: int = 120, Lx: float = 100.0, Ly: float = 100.0,
    nu_L: float = 1.5, var_L: float = 1.0, len_Lx: float = 80.0, len_Ly: float = 60.0, angle_deg: float = 30.0,
    use_multiscale: bool = True,
    var_M: float = 0.4, len_Mx: float = 60.0, len_My: float = 40.0,
    var_S: float = 0.1, len_Sx: float = 15.0, len_Sy: float = 10.0,
    t_onset: float = 0.0, t_p: float = 3.0, t_alpha: float = 0.25,
    beta_mu0: float = 10.0, beta_mu_s: float = 1.0, beta_mu_t: float = 0.5,
    beta_ls0: float = -1.0, beta_ls_s: float = 0.0, beta_ls_t: float = 0.3,
    xi: float = 0.1,
    seed: int = 2025,
):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, Lx, n_lon)
    y = np.linspace(0.0, Ly, n_lat)

    def matern_field(var, len_x, len_y, nu, angle_deg, seed_offset=0):
        model = gs.Matern(dim=2, var=var, len_scale=[len_x, len_y], nu=nu, angles=np.deg2rad(angle_deg))
        srf = gs.SRF(model, seed=seed + seed_offset)
        return srf.structured((x, y))  # (n_lat, n_lon)

    s_L = matern_field(var_L, len_Lx, len_Ly, nu_L, angle_deg, seed_offset=1)
    if use_multiscale:
        s_M = matern_field(var_M, len_Mx, len_My, 1.0, angle_deg, seed_offset=2)
        s_S = matern_field(var_S, len_Sx, len_Sy, 0.5, angle_deg, seed_offset=3)
        s_raw = s_L + s_M + s_S
    else:
        s_raw = s_L
    s_field = (s_raw - s_raw.mean()) / s_raw.std()

    t_curve = late_exp_curve(n_time, onset=t_onset, p=t_p, alpha=t_alpha)
    s3 = s_field[None, :, :]
    t3 = t_curve[:, None, None]

    mu = beta_mu0 + beta_mu_s * s3 + beta_mu_t * t3
    log_sigma = beta_ls0 + beta_ls_s * s3 + beta_ls_t * t3
    sigma = np.exp(log_sigma)

    data = np.empty((n_time, n_lat, n_lon))
    c = -xi  # scipy uses c = -xi
    for ti in range(n_time):
        data[ti] = genextreme.rvs(c=c, loc=mu[ti], scale=sigma[ti], random_state=rng)

    out = dict(
        x=x, y=y, t=np.arange(n_time),
        s_field=s_field, t_curve=t_curve,
        mu=mu, sigma=sigma, xi=xi,
        params=dict(
            nu_L=nu_L, var_L=var_L, len_Lx=len_Lx, len_Ly=len_Ly, angle_deg=angle_deg,
            use_multiscale=use_multiscale, var_M=var_M, len_Mx=len_Mx, len_My=len_My,
            var_S=var_S, len_Sx=len_Sx, len_Sy=len_Sy,
            beta_mu0=beta_mu0, beta_mu_s=beta_mu_s, beta_mu_t=beta_mu_t,
            beta_ls0=beta_ls0, beta_ls_s=beta_ls_s, beta_ls_t=beta_ls_t,
            xi=xi, seed=seed, t_onset=t_onset, t_p=t_p, t_alpha=t_alpha,
        ),
    )
    return data, out

# --------------------------- quick sanity plots ---------------------------

def plot_random_time_series(data: np.ndarray, meta: dict | None = None, seed: int = 123):
    rng = np.random.default_rng(seed)
    n_time, n_lat, n_lon = data.shape
    i = int(rng.integers(0, n_lat)); j = int(rng.integers(0, n_lon))
    t = meta["t"] if (meta and "t" in meta) else np.arange(n_time)
    loc_str = f"[i={i}, j={j}]"
    if meta and "x" in meta and "y" in meta:
        loc_str = f"(y={float(meta['y'][i]):.2f}, x={float(meta['x'][j]):.2f}) [i={i}, j={j}]"
    plt.figure()
    plt.plot(t, data[:, i, j])
    plt.xlabel("time"); plt.ylabel("GEV value"); plt.title(f"Time series at random location {loc_str}")
    plt.tight_layout(); plt.show()

def plot_random_spatial_slice(data: np.ndarray, meta: dict | None = None, seed: int = 456):
    rng = np.random.default_rng(seed)
    n_time, _, _ = data.shape
    k = int(rng.integers(0, n_time))
    field = data[k]
    vmax = float(np.quantile(field[np.isfinite(field)], 0.98))
    extent = None; title = f"Spatial slice at index t={k}"
    if meta and "x" in meta and "y" in meta:
        x, y = np.asarray(meta["x"]), np.asarray(meta["y"])
        extent = (x.min(), x.max(), y.min(), y.max())
        title = f"Spatial slice at t={meta['t'][k] if 't' in meta else k} (index {k})"
    plt.figure()
    im = plt.imshow(field, origin="lower", extent=extent, aspect="auto", interpolation="none", vmax=vmax)
    plt.colorbar(im, label="GEV value")
    plt.xlabel("x" if extent is not None else "lon index"); plt.ylabel("y" if extent is not None else "lat index")
    plt.title(title); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    data, meta = generate_gev_dataset(n_lat=50, n_lon=50, n_time=120)
    print("data:", data.shape, "| s_field sd:", np.std(meta["s_field"]).round(3), "| t_curve sd:", np.std(meta["t_curve"]).round(3))
    plot_random_time_series(data, meta)
    plot_random_spatial_slice(data, meta)
