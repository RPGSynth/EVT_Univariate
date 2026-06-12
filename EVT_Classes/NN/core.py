from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import genextreme

from EVT_Classes.selector import select_spatial_neighborhood

_THIS_FILE = Path(__file__).resolve()
_SIM_DIR = _THIS_FILE.parents[1] / "SIM"
DEFAULT_TRAIN_SELECTOR_ARGS = {
    "mode": "circle",
    "radius": None,
    "max_points": 100,
    "include_center": False,
}
DEFAULT_MARGINAL_CACHE_PATH = _SIM_DIR / "_cache" / "marginal_init_end1.npz"

__all__ = [
    "KernelContext",
    "TrainingConfig",
    "TrainedWeightModel",
    "BootstrapRunSummary",
    "BootstrapTrainingResult",
    "KernelFeatureBank",
    "EnergyKernelJointGEV",
    "FlexibleKernelWeightAdapterGPlus",
    "BootstrapAveragedWeightAdapterGPlus",
    "prepare_kernel_context",
    "build_teacher_marginal_fullfield",
    "train_weight_model",
    "train_weight_model_bootstrap",
    "build_gplus_adapter",
    "save_weight_artifact",
    "load_weight_artifact",
]

@dataclass
class KernelContext:
    data: np.ndarray
    Y_np: np.ndarray
    Y: torch.Tensor
    device: torch.device
    T: int
    H: int
    W: int
    S: int
    selector_args: dict[str, Any]
    nbr_idx: torch.Tensor
    nbr_d2: torch.Tensor
    nbr_mask: torch.Tensor
    d_space_norm_train: torch.Tensor
    space_scale: float

@dataclass
class TrainingConfig:
    random_seed: int = 2025
    split_seed: int | None = None
    train_fraction: float = 0.8
    early_stop_min_delta: float = 1e-3
    early_stop_patience: int = 5
    use_marginal_init: bool = True
    force_refit_marginals: bool = False
    outer_iters: int = 600
    kernel_epochs_per_outer: int = 2
    theta_epochs_per_outer: int = 3
    lr: float = 5e-3
    print_every: int = 10
    verbose: bool = True
    tau: float = 1.0

@dataclass
class TrainedWeightModel:
    model: "EnergyKernelJointGEV"
    feature_bank: "KernelFeatureBank"
    context: KernelContext
    history: dict[str, list[float]]
    config: TrainingConfig
    marginal_cache_path: Path
    train_time_idx: np.ndarray
    val_time_idx: np.ndarray
    selected_state_val_nll: float
    default_adapter: "FlexibleKernelWeightAdapterGPlus | None" = None
    meta: dict[str, Any] | None = None

    @property
    def device(self) -> torch.device:
        return self.context.device

@dataclass
class BootstrapRunSummary:
    run_idx: int
    split_seed: int
    selected_state_val_nll: float
    best_history_val_nll: float
    best_history_theta_train_nll: float

@dataclass
class BootstrapTrainingResult:
    best_model: TrainedWeightModel
    best_adapter: Callable[..., np.ndarray]
    ensemble_adapter: Callable[..., np.ndarray]
    run_summaries: list[BootstrapRunSummary]
    best_run_idx: int
    best_score: float
    n_bootstrap: int
    run_models: list[TrainedWeightModel] | None = None
    meta: dict[str, Any] | None = None

def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def _validate_data_3d(data: Any) -> np.ndarray:
    data_np = np.asarray(data, dtype=np.float64)
    if data_np.ndim != 3:
        raise ValueError(f"`data` must have shape (T, H, W), got {data_np.shape}")
    return data_np

def _validate_training_data(data: Any) -> np.ndarray:
    data_np = _validate_data_3d(data)
    if data_np.shape[0] < 2:
        raise ValueError(
            f"Training requires at least 2 time steps for a non-empty train/validation split, got T={data_np.shape[0]}"
        )
    if not np.isfinite(data_np).all():
        raise ValueError("Training data must be finite. Generated upstream data should not contain NaN or Inf.")
    return data_np

def _split_time_indices(
    T: int,
    *,
    train_fraction: float = 0.8,
    seed: int = 2025,
) -> tuple[np.ndarray, np.ndarray]:
    T = int(T)
    if T < 2:
        raise ValueError(f"Time split requires T >= 2, got T={T}")

    train_fraction = float(train_fraction)
    if not np.isfinite(train_fraction) or not (0.0 < train_fraction < 1.0):
        raise ValueError(f"`train_fraction` must be in (0, 1), got {train_fraction!r}")

    rng = np.random.default_rng(int(seed))
    all_t = np.arange(T, dtype=np.int64)
    rng.shuffle(all_t)

    n_train_t = max(1, min(T - 1, int(round(train_fraction * T))))
    tr_idx = np.sort(all_t[:n_train_t])
    va_idx = np.sort(all_t[n_train_t:])
    if tr_idx.size == 0 or va_idx.size == 0:
        raise RuntimeError("Time split unexpectedly produced an empty train or validation set.")
    return tr_idx, va_idx

def _resolve_split_seed(config: TrainingConfig) -> int:
    return int(config.random_seed if config.split_seed is None else config.split_seed)

def _validate_index(i: int, j: int, n_lat: int, n_lon: int) -> tuple[int, int]:
    i = int(i)
    j = int(j)
    if not (0 <= i < n_lat and 0 <= j < n_lon):
        raise ValueError(f"Pixel ({i}, {j}) is outside grid ({n_lat}, {n_lon})")
    return i, j

def _normalize_train_selector_args(selector_args: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_TRAIN_SELECTOR_ARGS)
    if selector_args is not None:
        cfg.update(selector_args)

    mode = str(cfg["mode"]).lower()
    if mode == "full":
        cfg["mode"] = "circle"
        cfg["radius"] = None
    elif mode not in {"square", "circle"}:
        raise ValueError(
            f"Unsupported training selector mode={cfg['mode']!r}. Expected 'square', 'circle', or 'full'."
        )

    return cfg

def _format_progress_bar(current: int, total: int, *, width: int = 28) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    width = max(8, int(width))
    filled = int(round(width * current / total))
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {current:03d}/{total:03d}"

def _train_split_hash(train_time_idx: np.ndarray) -> str:
    idx = np.asarray(train_time_idx, dtype=np.int64).reshape(-1)
    return hashlib.sha1(idx.tobytes()).hexdigest()[:16]

def _train_data_hash(train_data: np.ndarray) -> str:
    arr = np.asarray(train_data, dtype=np.float64, order="C")
    hasher = hashlib.sha1()
    hasher.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    hasher.update(arr.tobytes())
    return hasher.hexdigest()[:16]

def _resolve_split_marginal_cache_path(
    marginal_cache_path: Path,
    train_time_idx: np.ndarray,
    train_data: np.ndarray,
) -> Path:
    cache_path = Path(marginal_cache_path)
    suffix = cache_path.suffix if cache_path.suffix else ".npz"
    stem = cache_path.stem if cache_path.suffix else cache_path.name
    split_hash = _train_split_hash(train_time_idx)
    data_hash = _train_data_hash(train_data)
    return cache_path.parent / f"{stem}__data_{data_hash}__train_{split_hash}{suffix}"

def _clone_plain_object(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _clone_plain_object(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clone_plain_object(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_plain_object(item) for item in value)
    return value

def _config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    return {str(key): _clone_plain_object(value) for key, value in asdict(config).items()}

def _config_from_dict(payload: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(**{str(key): value for key, value in payload.items()})

def _cpu_state_dict(model: EnergyKernelJointGEV) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

def build_neighbor_tables(
    n_lat: int,
    n_lon: int,
    selector_args: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_total = n_lat * n_lon
    idx_list: list[np.ndarray] = []
    d2_list: list[np.ndarray] = []

    for site_idx in range(s_total):
        i0, j0 = divmod(site_idx, n_lon)
        sel = select_spatial_neighborhood(
            n_lat=n_lat,
            n_lon=n_lon,
            i=i0,
            j=j0,
            mode=selector_args["mode"],
            radius=selector_args["radius"],
            max_points=selector_args["max_points"],
            include_center=selector_args["include_center"],
        )

        flat = np.asarray(sel.flat_idx, dtype=np.int64)
        if flat.size == 0:
            flat = np.array([site_idx], dtype=np.int64)

        ii = flat // n_lon
        jj = flat % n_lon
        d2 = ((ii - i0) ** 2 + (jj - j0) ** 2).astype(np.float64)

        idx_list.append(flat)
        d2_list.append(d2)

    k_max = max(len(values) for values in idx_list)
    nbr_idx = np.zeros((s_total, k_max), dtype=np.int64)
    nbr_d2 = np.zeros((s_total, k_max), dtype=np.float64)
    nbr_mask = np.zeros((s_total, k_max), dtype=bool)

    for site_idx, (flat, d2) in enumerate(zip(idx_list, d2_list)):
        k = len(flat)
        nbr_idx[site_idx, :k] = flat
        nbr_d2[site_idx, :k] = d2
        nbr_mask[site_idx, :k] = True
        if k < k_max:
            nbr_idx[site_idx, k:] = flat[0]
            nbr_d2[site_idx, k:] = d2[0]

    return (
        torch.as_tensor(nbr_idx, dtype=torch.long, device=device),
        torch.as_tensor(nbr_d2, dtype=torch.float64, device=device),
        torch.as_tensor(nbr_mask, dtype=torch.bool, device=device),
    )

def prepare_kernel_context(
    data: Any,
    selector_args: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> KernelContext:
    data_np = _validate_data_3d(data)
    T, H, W = data_np.shape
    S = H * W
    device_t = _resolve_device(device)
    train_selector_args = _normalize_train_selector_args(selector_args)

    Y_np = data_np.reshape(T, S).astype(np.float64)
    Y = torch.from_numpy(Y_np).to(device=device_t, dtype=torch.float64)

    nbr_idx, nbr_d2, nbr_mask = build_neighbor_tables(H, W, train_selector_args, device_t)
    d_space = torch.sqrt(nbr_d2.clamp_min(0.0))
    d_valid = d_space[nbr_mask]
    d_pos = d_valid[d_valid > 0]
    space_scale = float(torch.median(d_pos).item()) if d_pos.numel() > 0 else 1.0
    space_scale = max(space_scale, 1e-8)

    d_space_norm_train = (d_space / space_scale) * nbr_mask.to(dtype=torch.float64)

    return KernelContext(
        data=data_np,
        Y_np=Y_np,
        Y=Y,
        device=device_t,
        T=T,
        H=H,
        W=W,
        S=S,
        selector_args=train_selector_args,
        nbr_idx=nbr_idx,
        nbr_d2=nbr_d2,
        nbr_mask=nbr_mask,
        d_space_norm_train=d_space_norm_train,
        space_scale=space_scale,
    )

def build_teacher_marginal_fullfield(
    data: Any,
    n_jobs: int = -1,
    backend: str = "threading",
) -> dict[str, np.ndarray]:
    data_np = _validate_data_3d(data)
    T, H, W = data_np.shape

    def fit_one(y_t: np.ndarray) -> tuple[float, float, float, bool]:
        try:
            c0, mu0, sigma0 = genextreme.fit(np.asarray(y_t, dtype=np.float64))
            xi0 = float(-c0)
            mu0 = float(mu0)
            sigma0 = float(sigma0)
            if (not np.isfinite(mu0)) or (not np.isfinite(sigma0)) or (not np.isfinite(xi0)) or sigma0 <= 0.0:
                raise ValueError("Invalid SciPy GEV fit.")
            return mu0, sigma0, xi0, True
        except Exception:
            return np.nan, np.nan, np.nan, False

    fits = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        delayed(fit_one)(data_np[:, i, j]) for i in range(H) for j in range(W)
    )

    mu2 = np.empty((H, W), dtype=np.float64)
    sigma2 = np.empty((H, W), dtype=np.float64)
    xi2 = np.empty((H, W), dtype=np.float64)
    ok2 = np.zeros((H, W), dtype=bool)

    offset = 0
    for i in range(H):
        for j in range(W):
            mu2[i, j], sigma2[i, j], xi2[i, j], ok2[i, j] = fits[offset]
            offset += 1

    return {
        "mu_hat": np.repeat(mu2[None, :, :], T, axis=0),
        "sigma_hat": np.repeat(sigma2[None, :, :], T, axis=0),
        "xi_hat": np.repeat(xi2[None, :, :], T, axis=0),
        "ok_mask": ok2,
    }

def inv_softplus_np(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.maximum(arr, 1e-12)
    return np.log(np.expm1(arr) + 1e-12)

def robust_pair_scale_1d(v: Any, num_pairs: int = 200_000, seed: int = 2025) -> float:
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    n = arr.size
    if n < 2:
        return 1.0
    m = min(int(num_pairs), max(1, n * (n - 1)))
    rng = np.random.default_rng(int(seed))
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n - 1, size=m)
    j = j + (j >= i)
    d = np.abs(arr[i] - arr[j])
    return float(np.median(d)) + 1e-8

def _make_x_space_infer_fn(context: KernelContext) -> Callable[..., np.ndarray]:
    def infer_x_space(
        i: int,
        j: int,
        coords: np.ndarray,
        flat: np.ndarray,
        target_flat: int,
        params_np: dict[str, np.ndarray],
    ) -> np.ndarray:
        del flat, target_flat, params_np
        return np.sqrt((coords[:, 0] - i) ** 2 + (coords[:, 1] - j) ** 2) / float(context.space_scale)

    return infer_x_space

class KernelFeatureBank:
    def __init__(self) -> None:
        self.static_specs: list[dict[str, Any]] = []
        self.dynamic_specs: list[dict[str, Any]] = []

    def add_static(self, name: str, train_tensor_sk: Any, infer_fn: Callable[..., np.ndarray]) -> None:
        if not torch.is_tensor(train_tensor_sk):
            train_tensor = torch.as_tensor(train_tensor_sk, dtype=torch.float64)
        else:
            train_tensor = train_tensor_sk.to(dtype=torch.float64)
        self.static_specs.append(
            {
                "name": str(name),
                "train_tensor": train_tensor,
                "infer_fn": infer_fn,
            }
        )

    def add_dynamic_sqdiff(self, name: str, param_key: str, scale2: float) -> None:
        self.dynamic_specs.append(
            {
                "name": str(name),
                "param_key": str(param_key),
                "scale2": float(max(scale2, 1e-12)),
            }
        )

    @property
    def in_dim(self) -> int:
        return len(self.static_specs) + len(self.dynamic_specs)

    @property
    def feature_names(self) -> list[str]:
        return [spec["name"] for spec in self.static_specs] + [spec["name"] for spec in self.dynamic_specs]

    def build_batch(
        self,
        q_idx: torch.Tensor,
        idx: torch.Tensor,
        params_t: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.in_dim == 0:
            raise ValueError("No kernel features configured in KernelFeatureBank.")

        feat_list: list[torch.Tensor] = []

        for spec in self.static_specs:
            train_tensor = spec["train_tensor"]
            if train_tensor.device != q_idx.device:
                train_tensor = train_tensor.to(q_idx.device)
            feat_list.append(train_tensor[q_idx])

        for spec in self.dynamic_specs:
            values = params_t[spec["param_key"]]
            q = values[q_idx].unsqueeze(1)
            n = values[idx]
            feat = ((n - q) ** 2) / spec["scale2"]
            feat_list.append(feat)

        return torch.stack(feat_list, dim=-1)

    def build_infer(
        self,
        *,
        i: int,
        j: int,
        coords: np.ndarray,
        flat: np.ndarray,
        target_flat: int,
        params_np: dict[str, np.ndarray],
    ) -> np.ndarray:
        if self.in_dim == 0:
            raise ValueError("No kernel features configured in KernelFeatureBank.")

        feat_list: list[np.ndarray] = []

        for spec in self.static_specs:
            values = np.asarray(spec["infer_fn"](i, j, coords, flat, target_flat, params_np), dtype=np.float64)
            feat_list.append(values)

        for spec in self.dynamic_specs:
            values = np.asarray(params_np[spec["param_key"]], dtype=np.float64)
            feat = ((values[flat] - values[target_flat]) ** 2) / spec["scale2"]
            feat_list.append(feat)

        return np.stack(feat_list, axis=-1)

class EnergyKernelJointGEV(nn.Module):
    def __init__(
        self,
        y_ts_np: np.ndarray,
        kernel_in_dim: int,
        energy_hidden: tuple[int, ...] = (64, 32),
        xi_bound: float = 0.35,
        sigma_min: float = 1e-4,
        xi_init: float = 0.1,
    ) -> None:
        super().__init__()
        mu0 = np.mean(y_ts_np, axis=0)
        sig0 = np.clip(np.std(y_ts_np, axis=0, ddof=1), 1e-3, None)

        self.mu = nn.Parameter(torch.as_tensor(mu0, dtype=torch.float64))
        self.raw_sigma = nn.Parameter(torch.as_tensor(inv_softplus_np(sig0), dtype=torch.float64))

        xi0 = float(np.clip(xi_init, -0.95 * xi_bound, 0.95 * xi_bound))
        raw_xi0 = np.arctanh(xi0 / xi_bound)
        self.raw_xi = nn.Parameter(torch.full_like(self.mu, raw_xi0))

        self.kernel_in_dim = int(kernel_in_dim)
        dims = [self.kernel_in_dim, *[int(hidden) for hidden in energy_hidden], self.kernel_in_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.energy_net = nn.Sequential(*layers)

        self.xi_bound = float(xi_bound)
        self.sigma_min = float(sigma_min)
        self.alpha_floor = 1e-8

    def constrained(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = F.softplus(self.raw_sigma) + self.sigma_min
        xi = self.xi_bound * torch.tanh(self.raw_xi)
        return self.mu, sigma, xi

    def kernel_energy_from_input(
        self,
        x_bkf: torch.Tensor,
        return_parts: bool = False,
    ) -> Any:
        batch_size, n_neighbors, feature_dim = x_bkf.shape
        x_pos = x_bkf.clamp_min(0.0)
        alpha_raw = self.energy_net(x_pos.reshape(batch_size * n_neighbors, feature_dim))
        alpha = F.softplus(alpha_raw).reshape(batch_size, n_neighbors, feature_dim) + self.alpha_floor
        per_feat = alpha * x_pos
        energy = per_feat.sum(dim=-1)
        if return_parts:
            return energy, alpha, per_feat
        return energy

    def kernel_logits(self, x_bkf: torch.Tensor, return_parts: bool = False) -> Any:
        if return_parts:
            energy, alpha, per_feat = self.kernel_energy_from_input(x_bkf, return_parts=True)
            return -energy, energy, alpha, per_feat
        energy = self.kernel_energy_from_input(x_bkf, return_parts=False)
        return -energy, energy

def _clip_if_any(params: list[torch.nn.Parameter], max_norm: float = 5.0) -> None:
    params_with_grad = [param for param in params if param.grad is not None]
    if params_with_grad:
        torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm)

def _set_requires_grad(params: list[torch.nn.Parameter], flag: bool) -> None:
    for param in params:
        param.requires_grad_(flag)

def _current_params_t(model: EnergyKernelJointGEV) -> dict[str, torch.Tensor]:
    mu_all, sigma_all, xi_all = model.constrained()
    return {
        "mu": mu_all,
        "sigma": sigma_all,
        "xi": xi_all,
        "logsigma": torch.log(sigma_all.clamp_min(1e-12)),
    }

def _masked_softmax_weights(
    logits_bk: torch.Tensor,
    mask_bk: torch.Tensor,
    *,
    tau: float = 1.0,
) -> torch.Tensor:
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError(f"`tau` must be positive, got {tau}")

    logits_masked = logits_bk.masked_fill(~mask_bk, -1e9)
    weights = torch.softmax(logits_masked / tau, dim=1)
    weights = weights * mask_bk
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

def _mean_ess_from_weights(weights_sk: torch.Tensor) -> torch.Tensor:
    if weights_sk.ndim != 2:
        raise ValueError(f"`weights_sk` must have shape (S, K), got {tuple(weights_sk.shape)}")
    ess_s = 1.0 / weights_sk.pow(2).sum(dim=1).clamp_min(1e-12)
    return ess_s.mean()

def _weighted_gev_nll_from_weights(
    model: EnergyKernelJointGEV,
    Y_ts: torch.Tensor,
    nbr_idx: torch.Tensor,
    weights_sk: torch.Tensor,
    *,
    xi_eps: float = 1e-6,
    penalty: float = 1e6,
) -> torch.Tensor:
    if tuple(weights_sk.shape) != tuple(nbr_idx.shape):
        raise ValueError(f"`weights_sk` shape {tuple(weights_sk.shape)} must match nbr_idx shape {tuple(nbr_idx.shape)}")

    params_t = _current_params_t(model)
    mu_all = params_t["mu"]
    sigma_all = params_t["sigma"]
    xi_all = params_t["xi"]

    y_tsk = Y_ts[:, nbr_idx]
    mu3 = mu_all[None, :, None]
    sg3 = sigma_all[None, :, None]
    z = (y_tsk - mu3) / sg3

    xi_safe = xi_all + xi_eps * torch.tanh(xi_all / xi_eps)
    xi_safe3 = xi_safe[None, :, None]
    t = 1.0 + xi_safe3 * z
    valid = t > 0.0
    log_t = torch.log(torch.clamp(t, min=1e-12))
    pow_term = torch.exp(torch.clamp(-log_t / xi_safe3, min=-40.0, max=40.0))
    nll_gev = torch.log(sg3) + (1.0 + 1.0 / xi_safe3) * log_t + pow_term
    nll_gev = torch.where(valid, nll_gev, torch.full_like(nll_gev, penalty))

    row_nll = (weights_sk[None, :, :] * nll_gev).sum(dim=2)
    return row_nll.mean()

def _build_default_feature_bank(
    context: KernelContext,
    mu_init_np: np.ndarray,
    sig_init_np: np.ndarray,
    xi_init_np: np.ndarray,
    *,
    seed: int,
) -> KernelFeatureBank:
    mu_scale = robust_pair_scale_1d(mu_init_np, num_pairs=200_000, seed=seed)
    mu_scale2 = max(mu_scale ** 2, 1e-8)

    sig_init_np = np.clip(sig_init_np, 1e-6, None)
    logsig_init_np = np.log(sig_init_np)
    logsig_scale = robust_pair_scale_1d(logsig_init_np, num_pairs=200_000, seed=seed)
    logsig_scale2 = max(logsig_scale ** 2, 1e-8)

    xi_scale = robust_pair_scale_1d(xi_init_np, num_pairs=200_000, seed=seed)
    xi_scale2 = max(xi_scale ** 2, 1e-8)

    feature_bank = KernelFeatureBank()

    feature_bank.add_static(
        name="x_space",
        train_tensor_sk=context.d_space_norm_train,
        infer_fn=_make_x_space_infer_fn(context),
    )
    feature_bank.add_dynamic_sqdiff(name="x_mu", param_key="mu", scale2=mu_scale2)
    feature_bank.add_dynamic_sqdiff(name="x_logsigma", param_key="logsigma", scale2=logsig_scale2)
    feature_bank.add_dynamic_sqdiff(name="x_xi", param_key="xi", scale2=xi_scale2)

    if feature_bank.in_dim == 0:
        raise ValueError("No kernel features configured in feature_bank.")

    return feature_bank

def _feature_bank_to_metadata(feature_bank: KernelFeatureBank) -> dict[str, Any]:
    return {
        "static_names": [str(spec["name"]) for spec in feature_bank.static_specs],
        "dynamic_specs": [
            {
                "name": str(spec["name"]),
                "param_key": str(spec["param_key"]),
                "scale2": float(spec["scale2"]),
            }
            for spec in feature_bank.dynamic_specs
        ],
    }

def _feature_bank_from_metadata(context: KernelContext, metadata: dict[str, Any]) -> KernelFeatureBank:
    feature_bank = KernelFeatureBank()
    static_names = [str(name) for name in metadata.get("static_names", [])]
    if static_names != ["x_space"]:
        raise ValueError(f"Unsupported static feature specification: {static_names}")

    feature_bank.add_static(
        name="x_space",
        train_tensor_sk=context.d_space_norm_train,
        infer_fn=_make_x_space_infer_fn(context),
    )
    for spec in metadata.get("dynamic_specs", []):
        feature_bank.add_dynamic_sqdiff(
            name=str(spec["name"]),
            param_key=str(spec["param_key"]),
            scale2=float(spec["scale2"]),
        )
    return feature_bank

def _model_to_metadata(model: EnergyKernelJointGEV) -> dict[str, Any]:
    linear_layers = [layer for layer in model.energy_net if isinstance(layer, nn.Linear)]
    if len(linear_layers) < 2:
        raise ValueError("Energy network must contain at least two Linear layers.")
    return {
        "kernel_in_dim": int(model.kernel_in_dim),
        "energy_hidden": tuple(int(layer.out_features) for layer in linear_layers[:-1]),
        "xi_bound": float(model.xi_bound),
        "sigma_min": float(model.sigma_min),
    }

def _load_or_fit_marginal_initialization(
    context: KernelContext,
    *,
    train_data: np.ndarray,
    train_time_idx: np.ndarray,
    config: TrainingConfig,
    marginal_cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Path]:
    train_data_np = _validate_data_3d(train_data)
    if train_data_np.shape[0] < 1:
        raise ValueError("`train_data` must include at least one time step.")
    if not np.isfinite(train_data_np).all():
        raise ValueError("`train_data` must be finite.")
    if train_data_np.shape[1:] != (context.H, context.W):
        raise ValueError(
            f"`train_data` spatial shape {train_data_np.shape[1:]} does not match context shape {(context.H, context.W)}"
        )

    train_flat = train_data_np.reshape(train_data_np.shape[0], -1)
    mu_fb = np.mean(train_flat, axis=0)
    ddof = 1 if train_flat.shape[0] > 1 else 0
    sig_fb = np.clip(np.std(train_flat, axis=0, ddof=ddof), 1e-6, None)
    xi_fb = np.full_like(mu_fb, 0.02, dtype=np.float64)
    split_cache_path = _resolve_split_marginal_cache_path(marginal_cache_path, train_time_idx, train_data_np)

    if not config.use_marginal_init:
        ok_t = np.zeros(context.S, dtype=bool)
        return mu_fb.copy(), sig_fb.copy(), xi_fb.copy(), ok_t, split_cache_path

    cache_path = Path(split_cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    loaded = False

    if cache_path.exists() and not config.force_refit_marginals:
        try:
            cached = np.load(cache_path)
            mu_raw = np.asarray(cached["mu"], dtype=np.float64)
            sigma_raw = np.asarray(cached["sigma"], dtype=np.float64)
            xi_raw = np.asarray(cached["xi"], dtype=np.float64)
            ok_raw = np.asarray(cached["ok"], dtype=bool)
            if (
                mu_raw.shape == (context.H, context.W)
                and sigma_raw.shape == (context.H, context.W)
                and xi_raw.shape == (context.H, context.W)
                and ok_raw.shape == (context.H, context.W)
            ):
                mu_t0 = mu_raw.reshape(-1)
                sig_t0 = sigma_raw.reshape(-1)
                xi_t0 = xi_raw.reshape(-1)
                ok_t = ok_raw.reshape(-1)
                loaded = True
                if config.verbose:
                    print(f"[Cell G] Loaded marginal cache: {cache_path}")
            elif config.verbose:
                print(f"[Cell G] Ignoring incompatible marginal cache shape at {cache_path}")
        except Exception:
            loaded = False

    if not loaded:
        teacher_init = build_teacher_marginal_fullfield(train_data_np, n_jobs=-1, backend="threading")
        mu_t0 = np.asarray(teacher_init["mu_hat"][0], dtype=np.float64).reshape(-1)
        sig_t0 = np.asarray(teacher_init["sigma_hat"][0], dtype=np.float64).reshape(-1)
        xi_t0 = np.asarray(teacher_init["xi_hat"][0], dtype=np.float64).reshape(-1)
        ok_t = np.asarray(teacher_init.get("ok_mask", np.ones((context.H, context.W), dtype=bool)), dtype=bool).reshape(-1)

        np.savez_compressed(
            cache_path,
            mu=mu_t0.reshape(context.H, context.W),
            sigma=sig_t0.reshape(context.H, context.W),
            xi=xi_t0.reshape(context.H, context.W),
            ok=ok_t.reshape(context.H, context.W),
        )
        if config.verbose:
            print(f"[Cell G] Fitted marginals + saved cache: {cache_path}")

    ok_t = ok_t & np.isfinite(mu_t0) & np.isfinite(sig_t0) & np.isfinite(xi_t0) & (sig_t0 > 0)
    mu_init_np = np.where(ok_t, mu_t0, mu_fb).astype(np.float64)
    sig_init_np = np.where(ok_t, sig_t0, sig_fb).astype(np.float64)
    xi_init_np = np.where(ok_t, xi_t0, xi_fb).astype(np.float64)
    return mu_init_np, sig_init_np, xi_init_np, ok_t, cache_path

def _build_feature_snapshot(
    model: EnergyKernelJointGEV,
    feature_bank: KernelFeatureBank,
    nbr_idx: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    S_local, K_local = nbr_idx.shape
    F_local = feature_bank.in_dim

    with torch.no_grad():
        q_idx = torch.arange(S_local, device=nbr_idx.device, dtype=torch.long)
        params_t = _current_params_t(model)
        snapshot = feature_bank.build_batch(q_idx=q_idx, idx=nbr_idx, params_t=params_t).detach()
        expected = (S_local, K_local, F_local)
        if snapshot.shape != expected:
            raise RuntimeError(f"Unexpected X_snapshot shape {tuple(snapshot.shape)}, expected {expected}")
        return snapshot

def _weights_from_features(
    model: EnergyKernelJointGEV,
    X_bkf: torch.Tensor,
    mask_bk: torch.Tensor,
    *,
    tau: float = 1.0,
) -> torch.Tensor:
    logits_bk, _ = model.kernel_logits(X_bkf, return_parts=False)
    return _masked_softmax_weights(logits_bk, mask_bk, tau=tau)

def _validation_nll(
    model: EnergyKernelJointGEV,
    Y_ts: torch.Tensor,
    feature_bank: KernelFeatureBank,
    nbr_idx: torch.Tensor,
    nbr_mask: torch.Tensor,
    *,
    tau: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_idx = torch.arange(nbr_idx.shape[0], device=nbr_idx.device, dtype=torch.long)
    params_t = _current_params_t(model)
    X_current = feature_bank.build_batch(q_idx=q_idx, idx=nbr_idx, params_t=params_t)
    weights_sk = _weights_from_features(model, X_current, nbr_mask, tau=tau)
    nll = _weighted_gev_nll_from_weights(
        model=model,
        Y_ts=Y_ts,
        nbr_idx=nbr_idx,
        weights_sk=weights_sk,
    )
    return nll, _mean_ess_from_weights(weights_sk)

class FlexibleKernelWeightAdapterGPlus:
    def __init__(
        self,
        trained_model: EnergyKernelJointGEV,
        feature_bank: KernelFeatureBank,
        H: int,
        W: int,
        *,
        device: str | torch.device = "cpu",
        tau: float = 1.0,
        self_frac_cutoff: float | None = 0.01,
    ) -> None:
        self.model = trained_model
        self.model.eval()
        self.feature_bank = feature_bank
        self.H = int(H)
        self.W = int(W)
        self.S = self.H * self.W
        self.device = _resolve_device(device)
        self.tau = float(tau)
        self.self_frac_cutoff = None if self_frac_cutoff is None else float(self_frac_cutoff)

        ii, jj = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        self.coords_full = np.column_stack([ii.ravel(), jj.ravel()]).astype(np.int64)
        self.flat_full = (self.coords_full[:, 0] * self.W + self.coords_full[:, 1]).astype(np.int64)

    def _current_params_np(self) -> dict[str, np.ndarray]:
        with torch.no_grad():
            mu_all, sigma_all, xi_all = self.model.constrained()

        mu_np = mu_all.detach().cpu().numpy().astype(np.float64)
        sigma_np = sigma_all.detach().cpu().numpy().astype(np.float64)
        xi_np = xi_all.detach().cpu().numpy().astype(np.float64)
        return {
            "mu": mu_np,
            "sigma": sigma_np,
            "xi": xi_np,
            "logsigma": np.log(np.clip(sigma_np, 1e-12, None)),
        }

    def __call__(self, i: int, j: int, t_idx: int, data_shape: tuple[int, int, int], **kwargs: Any) -> np.ndarray:
        del kwargs, t_idx
        n_time, n_lat, n_lon = tuple(int(value) for value in data_shape)
        if (n_lat, n_lon) != (self.H, self.W):
            raise ValueError(
                f"data_shape spatial dims {(n_lat, n_lon)} do not match adapter dims {(self.H, self.W)}"
            )

        i, j = _validate_index(i, j, n_lat, n_lon)
        target_flat = int(i * n_lon + j)
        params_np = self._current_params_np()

        X_np = self.feature_bank.build_infer(
            i=i,
            j=j,
            coords=self.coords_full,
            flat=self.flat_full,
            target_flat=target_flat,
            params_np=params_np,
        )
        center_mask = self.flat_full == target_flat
        if not center_mask.any():
            raise RuntimeError("Target site not found in the full candidate set.")
        ref_idx = int(np.where(center_mask)[0][0])

        with torch.inference_mode():
            x_t = torch.from_numpy(X_np).to(self.device, dtype=torch.float64).unsqueeze(0)
            logits, _ = self.model.kernel_logits(x_t, return_parts=False)
            probs = torch.softmax(logits.squeeze(0) / self.tau, dim=0).detach().cpu().numpy().astype(np.float64)

        if (not np.isfinite(probs).all()) or float(probs.sum()) <= 0.0:
            probs = np.zeros(self.S, dtype=np.float64)
            probs[ref_idx] = 1.0

        if self.self_frac_cutoff is not None:
            ref_w = float(probs[ref_idx])
            cutoff = self.self_frac_cutoff * ref_w
            probs[probs < cutoff] = 0.0
            probs[ref_idx] = max(probs[ref_idx], ref_w)
            mass = float(probs.sum())
            if mass > 0.0:
                probs = probs / mass
            else:
                probs = np.zeros(self.S, dtype=np.float64)
                probs[ref_idx] = 1.0

        w_map = probs.reshape(self.H, self.W)
        return np.repeat(w_map[None, :, :], n_time, axis=0)

def _normalize_weight_cube(weight_cube: np.ndarray, *, i: int, j: int) -> np.ndarray:
    cube = np.asarray(weight_cube, dtype=np.float64).copy()
    if cube.ndim != 3:
        raise ValueError(f"`weight_cube` must have shape (T, H, W), got {tuple(cube.shape)}")

    T = cube.shape[0]
    for t in range(T):
        plane = np.asarray(cube[t], dtype=np.float64)
        plane = np.where(np.isfinite(plane), plane, 0.0)
        plane = np.clip(plane, 0.0, None)
        mass = float(plane.sum())
        if (not np.isfinite(mass)) or mass <= 0.0:
            plane.fill(0.0)
            plane[i, j] = 1.0
        else:
            plane = plane / mass
        cube[t] = plane
    return cube

class BootstrapAveragedWeightAdapterGPlus:
    def __init__(
        self,
        adapters: list[Callable[..., Any]],
        H: int,
        W: int,
        *,
        self_frac_cutoff: float | None = 0.01,
    ) -> None:
        if len(adapters) == 0:
            raise ValueError("`adapters` must be non-empty.")

        self.adapters = list(adapters)
        self.H = int(H)
        self.W = int(W)
        self.self_frac_cutoff = None if self_frac_cutoff is None else float(self_frac_cutoff)

    def __call__(self, i: int, j: int, t_idx: int, data_shape: tuple[int, int, int], **kwargs: Any) -> np.ndarray:
        n_time, n_lat, n_lon = tuple(int(value) for value in data_shape)
        if (n_lat, n_lon) != (self.H, self.W):
            raise ValueError(
                f"data_shape spatial dims {(n_lat, n_lon)} do not match adapter dims {(self.H, self.W)}"
            )

        i, j = _validate_index(i, j, n_lat, n_lon)
        cubes: list[np.ndarray] = []
        for adapter in self.adapters:
            w_full = np.asarray(adapter(i=i, j=j, t_idx=t_idx, data_shape=data_shape, **kwargs), dtype=np.float64)
            if w_full.shape != tuple(data_shape):
                raise ValueError(f"Child adapter returned shape {w_full.shape}, expected {tuple(data_shape)}")
            cubes.append(w_full)

        avg = np.mean(np.stack(cubes, axis=0), axis=0)
        avg = _normalize_weight_cube(avg, i=i, j=j)

        if self.self_frac_cutoff is not None:
            for t in range(n_time):
                ref_w = float(avg[t, i, j])
                cutoff = self.self_frac_cutoff * ref_w
                avg[t][avg[t] < cutoff] = 0.0
                avg[t, i, j] = max(float(avg[t, i, j]), ref_w)
            avg = _normalize_weight_cube(avg, i=i, j=j)

        return avg

def build_gplus_adapter(
    trained: TrainedWeightModel,
    *,
    tau: float | None = None,
    self_frac_cutoff: float | None = 0.01,
) -> FlexibleKernelWeightAdapterGPlus:
    return FlexibleKernelWeightAdapterGPlus(
        trained_model=trained.model,
        feature_bank=trained.feature_bank,
        H=trained.context.H,
        W=trained.context.W,
        device=trained.device,
        tau=trained.config.tau if tau is None else tau,
        self_frac_cutoff=self_frac_cutoff,
    )

def _serialize_trained_weight_model(trained: TrainedWeightModel) -> dict[str, Any]:
    return {
        "config": _config_to_dict(trained.config),
        "selector_args": _clone_plain_object(trained.context.selector_args),
        "history": {
            str(key): [float(value) for value in values]
            for key, values in trained.history.items()
        },
        "marginal_cache_path": str(trained.marginal_cache_path),
        "train_time_idx": np.asarray(trained.train_time_idx, dtype=np.int64).copy(),
        "val_time_idx": np.asarray(trained.val_time_idx, dtype=np.int64).copy(),
        "selected_state_val_nll": float(trained.selected_state_val_nll),
        "feature_bank": _feature_bank_to_metadata(trained.feature_bank),
        "model_meta": _model_to_metadata(trained.model),
        "model_state": _cpu_state_dict(trained.model),
        "data_shape": tuple(int(value) for value in trained.context.data.shape),
    }

def _rebuild_trained_weight_model(
    payload: dict[str, Any],
    *,
    data: np.ndarray,
    meta: dict[str, Any] | None,
    device: torch.device,
) -> TrainedWeightModel:
    data_np = _validate_training_data(data)
    expected_shape = tuple(int(value) for value in payload["data_shape"])
    if tuple(data_np.shape) != expected_shape:
        raise ValueError(f"Artifact data shape {tuple(data_np.shape)} does not match saved shape {expected_shape}")

    selector_args = _clone_plain_object(payload.get("selector_args"))
    context = prepare_kernel_context(data_np, selector_args=selector_args, device=device)
    feature_bank = _feature_bank_from_metadata(context, payload["feature_bank"])
    model_meta = payload["model_meta"]
    model = EnergyKernelJointGEV(
        y_ts_np=context.Y_np,
        kernel_in_dim=int(model_meta["kernel_in_dim"]),
        energy_hidden=tuple(int(value) for value in model_meta["energy_hidden"]),
        xi_bound=float(model_meta["xi_bound"]),
        sigma_min=float(model_meta["sigma_min"]),
        xi_init=0.0,
    ).to(context.device).double()
    model.load_state_dict(payload["model_state"])

    trained = TrainedWeightModel(
        model=model,
        feature_bank=feature_bank,
        context=context,
        history={
            str(key): [float(value) for value in values]
            for key, values in payload["history"].items()
        },
        config=_config_from_dict(payload["config"]),
        marginal_cache_path=Path(payload["marginal_cache_path"]),
        train_time_idx=np.asarray(payload["train_time_idx"], dtype=np.int64).copy(),
        val_time_idx=np.asarray(payload["val_time_idx"], dtype=np.int64).copy(),
        selected_state_val_nll=float(payload["selected_state_val_nll"]),
        meta=_clone_plain_object(meta),
    )
    trained.default_adapter = build_gplus_adapter(trained)
    return trained

def save_weight_artifact(obj: TrainedWeightModel | BootstrapTrainingResult, path: str | Path) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, BootstrapTrainingResult):
        if not obj.run_models:
            raise ValueError("BootstrapTrainingResult must include `run_models` to be saved.")
        run_models = obj.run_models
        data = np.asarray(run_models[0].context.data, dtype=np.float64)
        meta = _clone_plain_object(obj.meta if obj.meta is not None else run_models[0].meta)
        payload = {
            "artifact_version": 1,
            "artifact_type": "bootstrap",
            "data": data,
            "meta": meta,
            "run_payloads": [_serialize_trained_weight_model(model) for model in run_models],
            "run_summaries": [asdict(summary) for summary in obj.run_summaries],
            "best_run_idx": int(obj.best_run_idx),
            "best_score": float(obj.best_score),
            "n_bootstrap": int(obj.n_bootstrap),
            "ensemble_self_frac_cutoff": (
                None
                if not isinstance(obj.ensemble_adapter, BootstrapAveragedWeightAdapterGPlus)
                else obj.ensemble_adapter.self_frac_cutoff
            ),
        }
    elif isinstance(obj, TrainedWeightModel):
        data = np.asarray(obj.context.data, dtype=np.float64)
        payload = {
            "artifact_version": 1,
            "artifact_type": "single",
            "data": data,
            "meta": _clone_plain_object(obj.meta),
            "run_payloads": [_serialize_trained_weight_model(obj)],
            "run_summaries": [],
            "best_run_idx": 0,
            "best_score": float(obj.selected_state_val_nll),
            "n_bootstrap": 1,
            "ensemble_self_frac_cutoff": None,
        }
    else:
        raise TypeError(f"Unsupported artifact object type: {type(obj)!r}")

    torch.save(payload, artifact_path)
    return artifact_path

def load_weight_artifact(
    path: str | Path,
    device: str | torch.device | None = None,
) -> TrainedWeightModel | BootstrapTrainingResult:
    artifact_path = Path(path)
    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(artifact_path, map_location="cpu")
    if int(payload.get("artifact_version", -1)) != 1:
        raise ValueError(f"Unsupported artifact version: {payload.get('artifact_version')!r}")

    artifact_type = str(payload["artifact_type"])
    data_np = _validate_training_data(payload["data"])
    meta = _clone_plain_object(payload.get("meta"))
    resolved_device = _resolve_device(device)

    run_models = [
        _rebuild_trained_weight_model(run_payload, data=data_np, meta=meta, device=resolved_device)
        for run_payload in payload["run_payloads"]
    ]
    if len(run_models) == 0:
        raise ValueError("Artifact did not contain any saved run models.")

    if artifact_type == "single":
        return run_models[0]

    if artifact_type != "bootstrap":
        raise ValueError(f"Unsupported artifact type: {artifact_type!r}")

    run_summaries = [
        BootstrapRunSummary(
            run_idx=int(summary["run_idx"]),
            split_seed=int(summary["split_seed"]),
            selected_state_val_nll=float(summary["selected_state_val_nll"]),
            best_history_val_nll=float(summary["best_history_val_nll"]),
            best_history_theta_train_nll=float(summary["best_history_theta_train_nll"]),
        )
        for summary in payload["run_summaries"]
    ]
    best_run_idx = int(payload["best_run_idx"])
    best_model = run_models[best_run_idx]
    best_adapter = best_model.default_adapter if best_model.default_adapter is not None else build_gplus_adapter(best_model)
    ensemble_children = [build_gplus_adapter(model, self_frac_cutoff=None) for model in run_models]
    ensemble_adapter = BootstrapAveragedWeightAdapterGPlus(
        adapters=ensemble_children,
        H=best_model.context.H,
        W=best_model.context.W,
        self_frac_cutoff=payload.get("ensemble_self_frac_cutoff", 0.01),
    )
    return BootstrapTrainingResult(
        best_model=best_model,
        best_adapter=best_adapter,
        ensemble_adapter=ensemble_adapter,
        run_summaries=run_summaries,
        best_run_idx=best_run_idx,
        best_score=float(payload["best_score"]),
        n_bootstrap=int(payload["n_bootstrap"]),
        run_models=run_models,
        meta=meta,
    )

def train_weight_model(
    data: Any,
    *,
    selector_args: dict[str, Any] | None = None,
    config: TrainingConfig | None = None,
    marginal_cache_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> TrainedWeightModel:
    cfg = TrainingConfig() if config is None else config
    if int(cfg.outer_iters) < 1:
        raise ValueError(f"`outer_iters` must be >= 1, got {cfg.outer_iters}")
    if int(cfg.kernel_epochs_per_outer) < 1:
        raise ValueError(f"`kernel_epochs_per_outer` must be >= 1, got {cfg.kernel_epochs_per_outer}")
    if int(cfg.theta_epochs_per_outer) < 1:
        raise ValueError(f"`theta_epochs_per_outer` must be >= 1, got {cfg.theta_epochs_per_outer}")

    data_np = _validate_training_data(data)
    split_seed = _resolve_split_seed(cfg)
    tr_idx_np, va_idx_np = _split_time_indices(
        data_np.shape[0],
        train_fraction=cfg.train_fraction,
        seed=split_seed,
    )
    train_data_np = np.asarray(data_np[tr_idx_np], dtype=np.float64)
    context = prepare_kernel_context(data_np, selector_args=selector_args, device=device)
    cache_base_path = DEFAULT_MARGINAL_CACHE_PATH if marginal_cache_path is None else Path(marginal_cache_path)

    mu_init_np, sig_init_np, xi_init_np, ok_t, cache_path = _load_or_fit_marginal_initialization(
        context,
        train_data=train_data_np,
        train_time_idx=tr_idx_np,
        config=cfg,
        marginal_cache_path=cache_base_path,
    )
    if cfg.verbose:
        print(f"[Cell G] marginal init valid sites: {int(ok_t.sum())}/{ok_t.size}")

    feature_bank = _build_default_feature_bank(
        context,
        mu_init_np=mu_init_np,
        sig_init_np=sig_init_np,
        xi_init_np=xi_init_np,
        seed=cfg.random_seed,
    )
    if cfg.verbose:
        print("[Cell G] Kernel features:", feature_bank.feature_names)
        print("[Cell G] Kernel input dimension:", feature_bank.in_dim)

    tr_t = torch.as_tensor(tr_idx_np, dtype=torch.long, device=context.device)
    va_t = torch.as_tensor(va_idx_np, dtype=torch.long, device=context.device)
    Y_tr = context.Y.index_select(0, tr_t)
    Y_va = context.Y.index_select(0, va_t)

    model = EnergyKernelJointGEV(
        y_ts_np=context.Y_np,
        kernel_in_dim=feature_bank.in_dim,
        energy_hidden=(64, 32),
        xi_bound=0.35,
        sigma_min=1e-4,
        xi_init=float(np.nanmedian(xi_init_np)),
    ).to(context.device).double()

    with torch.no_grad():
        model.mu.copy_(torch.as_tensor(mu_init_np, dtype=torch.float64, device=context.device))
        model.raw_sigma.copy_(
            torch.as_tensor(inv_softplus_np(np.clip(sig_init_np, 1e-6, None)), dtype=torch.float64, device=context.device)
        )
        xi_clip = np.clip(xi_init_np, -0.95 * model.xi_bound, 0.95 * model.xi_bound)
        model.raw_xi.copy_(
            torch.as_tensor(np.arctanh(xi_clip / model.xi_bound), dtype=torch.float64, device=context.device)
        )

    marginal_params = [model.mu, model.raw_sigma, model.raw_xi]
    marginal_ids = {id(param) for param in marginal_params}
    kernel_params = [param for _, param in model.named_parameters() if id(param) not in marginal_ids]
    if not kernel_params:
        raise RuntimeError("Kernel parameter block is empty.")

    opt_theta = torch.optim.Adam(marginal_params, lr=cfg.lr)
    opt_kernel = torch.optim.Adam(kernel_params, lr=cfg.lr)

    if cfg.verbose:
        print(f"[Cell G] time split: train={tr_t.numel()}, val={va_t.numel()} (T={context.T})")
        print(
            f"[Cell G] nbr_idx.shape={tuple(context.nbr_idx.shape)}, "
            f"nbr_mask.shape={tuple(context.nbr_mask.shape)}, K={context.nbr_idx.shape[1]}"
        )
        valid_neighbors = context.nbr_mask.sum(dim=1)
        print(
            "[Cell G] valid neighbors per site: "
            f"min={int(valid_neighbors.min().item())}, "
            f"mean={float(valid_neighbors.double().mean().item()):.3f}, "
            f"max={int(valid_neighbors.max().item())}"
        )

    history: dict[str, list[float]] = {
        "outer_iter": [],
        "weight_step_train_nll": [],
        "theta_step_train_nll": [],
        "val_nll": [],
        "weight_step_mean_ess": [],
        "theta_step_mean_ess": [],
        "val_mean_ess": [],
    }

    best_train = float("inf")
    best_val = float("inf")
    best_outer = 0
    prev_val_nll: float | None = None
    early_stop_bad_count = 0
    best_state: dict[str, torch.Tensor] | None = None
    early_stop_min_delta = float(cfg.early_stop_min_delta)
    if not np.isfinite(early_stop_min_delta) or early_stop_min_delta < 0.0:
        raise ValueError(f"`early_stop_min_delta` must be finite and >= 0, got {cfg.early_stop_min_delta!r}")
    early_stop_patience = int(cfg.early_stop_patience)
    if early_stop_patience < 1:
        raise ValueError(f"`early_stop_patience` must be >= 1, got {cfg.early_stop_patience}")

    for outer in range(1, int(cfg.outer_iters) + 1):
        X_snapshot = _build_feature_snapshot(
            model=model,
            feature_bank=feature_bank,
            nbr_idx=context.nbr_idx,
        )

        _set_requires_grad(marginal_params, False)
        _set_requires_grad(kernel_params, True)
        model.train()

        weight_step_nlls: list[float] = []
        weight_step_esss: list[float] = []
        for _ in range(int(cfg.kernel_epochs_per_outer)):
            weights_sk = _weights_from_features(
                model=model,
                X_bkf=X_snapshot,
                mask_bk=context.nbr_mask,
                tau=cfg.tau,
            )
            nll_w = _weighted_gev_nll_from_weights(
                model=model,
                Y_ts=Y_tr,
                nbr_idx=context.nbr_idx,
                weights_sk=weights_sk,
            )
            opt_kernel.zero_grad(set_to_none=True)
            nll_w.backward()
            _clip_if_any(kernel_params, max_norm=5.0)
            opt_kernel.step()
            weight_step_nlls.append(float(nll_w.detach().cpu()))
            weight_step_esss.append(float(_mean_ess_from_weights(weights_sk).detach().cpu()))

        weight_step_train_nll = float(np.mean(weight_step_nlls))
        weight_step_mean_ess = float(np.mean(weight_step_esss))

        _set_requires_grad(kernel_params, False)
        _set_requires_grad(marginal_params, True)
        model.train()

        with torch.no_grad():
            theta_weights_fixed = _weights_from_features(
                model=model,
                X_bkf=X_snapshot,
                mask_bk=context.nbr_mask,
                tau=cfg.tau,
            ).detach()
            theta_step_mean_ess = float(_mean_ess_from_weights(theta_weights_fixed).detach().cpu())

        theta_step_nlls: list[float] = []
        for _ in range(int(cfg.theta_epochs_per_outer)):
            nll_t = _weighted_gev_nll_from_weights(
                model=model,
                Y_ts=Y_tr,
                nbr_idx=context.nbr_idx,
                weights_sk=theta_weights_fixed,
            )
            opt_theta.zero_grad(set_to_none=True)
            nll_t.backward()
            _clip_if_any(marginal_params, max_norm=5.0)
            opt_theta.step()
            theta_step_nlls.append(float(nll_t.detach().cpu()))

        theta_step_train_nll = float(np.mean(theta_step_nlls))

        _set_requires_grad(marginal_params, True)
        _set_requires_grad(kernel_params, True)

        model.eval()
        with torch.no_grad():
            val_nll_t, val_mean_ess_t = _validation_nll(
                model=model,
                Y_ts=Y_va,
                feature_bank=feature_bank,
                nbr_idx=context.nbr_idx,
                nbr_mask=context.nbr_mask,
                tau=cfg.tau,
            )
            val_nll = float(val_nll_t.detach().cpu())
            val_mean_ess = float(val_mean_ess_t.detach().cpu())

        history["outer_iter"].append(float(outer))
        history["weight_step_train_nll"].append(weight_step_train_nll)
        history["theta_step_train_nll"].append(theta_step_train_nll)
        history["val_nll"].append(val_nll)
        history["weight_step_mean_ess"].append(weight_step_mean_ess)
        history["theta_step_mean_ess"].append(theta_step_mean_ess)
        history["val_mean_ess"].append(val_mean_ess)

        if cfg.verbose and (outer == 1 or outer % int(cfg.print_every) == 0):
            print(
                f"outer={outer:03d} | "
                f"nll[W={weight_step_train_nll:.5f}, T={theta_step_train_nll:.5f}, val={val_nll:.5f}] | "
                f"ess[W={weight_step_mean_ess:.3f}, T={theta_step_mean_ess:.3f}, val={val_mean_ess:.3f}]"
            )

        if theta_step_train_nll < best_train:
            best_train = theta_step_train_nll

        if val_nll < best_val - early_stop_min_delta:
            best_val = val_nll
            best_outer = outer
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            early_stop_bad_count = 0
        elif prev_val_nll is not None and val_nll > prev_val_nll + early_stop_min_delta:
            early_stop_bad_count += 1
            if early_stop_bad_count >= early_stop_patience:
                if cfg.verbose:
                    print(
                        f"[Cell G] early stop at outer={outer:03d}: "
                        f"val NLL increased for {early_stop_bad_count} consecutive outer steps "
                        f"(last {prev_val_nll:.5f} -> {val_nll:.5f}, min_delta={early_stop_min_delta:.5g}). "
                        f"Restoring best validation checkpoint from outer={best_outer:03d}."
                    )
                break
        else:
            early_stop_bad_count = 0

        prev_val_nll = val_nll

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        selected_state_val_nll_t, _ = _validation_nll(
            model=model,
            Y_ts=Y_va,
            feature_bank=feature_bank,
            nbr_idx=context.nbr_idx,
            nbr_mask=context.nbr_mask,
            tau=cfg.tau,
        )
        selected_state_val_nll = float(selected_state_val_nll_t.detach().cpu())

    if cfg.verbose:
        print("Best theta-step train NLL (Cell G):", best_train)
        print(f"Best validation NLL checkpoint (Cell G): {best_val:.5f} at outer={best_outer:03d}")

    trained = TrainedWeightModel(
        model=model,
        feature_bank=feature_bank,
        context=context,
        history=history,
        config=cfg,
        marginal_cache_path=cache_path,
        train_time_idx=np.asarray(tr_idx_np, dtype=np.int64).copy(),
        val_time_idx=np.asarray(va_idx_np, dtype=np.int64).copy(),
        selected_state_val_nll=selected_state_val_nll,
    )
    trained.default_adapter = build_gplus_adapter(trained)
    return trained

def train_weight_model_bootstrap(
    data: Any,
    *,
    selector_args: dict[str, Any] | None = None,
    config: TrainingConfig | None = None,
    marginal_cache_path: str | Path | None = None,
    device: str | torch.device | None = None,
    n_bootstrap: int = 50,
    bootstrap_seed: int | None = None,
) -> BootstrapTrainingResult:
    cfg_base = TrainingConfig() if config is None else replace(config)
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError(f"`n_bootstrap` must be >= 1, got {n_bootstrap}")

    base_seed = int(cfg_base.random_seed if bootstrap_seed is None else bootstrap_seed)
    run_models: list[TrainedWeightModel] = []
    run_summaries: list[BootstrapRunSummary] = []
    best_idx = 0
    best_score = float("inf")

    for run_idx in range(n_bootstrap):
        child_cfg = replace(
            cfg_base,
            split_seed=base_seed + run_idx,
            verbose=False,
        )
        trained = train_weight_model(
            data,
            selector_args=selector_args,
            config=child_cfg,
            marginal_cache_path=marginal_cache_path,
            device=device,
        )
        run_models.append(trained)

        best_history_val_nll = float(min(trained.history["val_nll"]))
        best_history_theta_train_nll = float(min(trained.history["theta_step_train_nll"]))
        summary = BootstrapRunSummary(
            run_idx=run_idx,
            split_seed=int(child_cfg.split_seed),
            selected_state_val_nll=float(trained.selected_state_val_nll),
            best_history_val_nll=best_history_val_nll,
            best_history_theta_train_nll=best_history_theta_train_nll,
        )
        run_summaries.append(summary)

        if summary.selected_state_val_nll < best_score:
            best_idx = run_idx
            best_score = summary.selected_state_val_nll

        if cfg_base.verbose:
            progress_bar = _format_progress_bar(run_idx + 1, n_bootstrap)
            status = (
                f"{progress_bar} | "
                f"split_seed={summary.split_seed} | "
                f"val={summary.selected_state_val_nll:.5f} | "
                f"best={best_score:.5f}"
            )
            end = "\n" if run_idx + 1 == n_bootstrap else "\r"
            print(status, end=end, flush=True)

    best_model = run_models[best_idx]
    best_adapter = best_model.default_adapter if best_model.default_adapter is not None else build_gplus_adapter(best_model)
    ensemble_children = [build_gplus_adapter(model, self_frac_cutoff=None) for model in run_models]
    ensemble_adapter = BootstrapAveragedWeightAdapterGPlus(
        adapters=ensemble_children,
        H=best_model.context.H,
        W=best_model.context.W,
        self_frac_cutoff=0.01,
    )

    if cfg_base.verbose:
        best_summary = run_summaries[best_idx]
        print(
            f"bootstrap best | run={best_idx + 1:03d}/{n_bootstrap:03d} | "
            f"split_seed={best_summary.split_seed} | "
            f"val={best_summary.selected_state_val_nll:.5f}"
        )

    return BootstrapTrainingResult(
        best_model=best_model,
        best_adapter=best_adapter,
        ensemble_adapter=ensemble_adapter,
        run_summaries=run_summaries,
        best_run_idx=best_idx,
        best_score=best_score,
        n_bootstrap=n_bootstrap,
        run_models=run_models,
    )

def _resolve_artifact_runtime_state(
    artifact_obj: TrainedWeightModel | BootstrapTrainingResult,
) -> tuple[TrainedWeightModel, Callable[..., np.ndarray], dict[str, Any] | None, np.ndarray, int]:
    if isinstance(artifact_obj, BootstrapTrainingResult):
        trained = artifact_obj.best_model
        adapter = artifact_obj.ensemble_adapter
        meta = artifact_obj.meta if artifact_obj.meta is not None else trained.meta
        n_bootstrap = int(artifact_obj.n_bootstrap)
    else:
        trained = artifact_obj
        adapter = trained.default_adapter if trained.default_adapter is not None else build_gplus_adapter(trained)
        meta = trained.meta
        n_bootstrap = 1
    data = np.asarray(trained.context.data, dtype=np.float64)
    return trained, adapter, meta, data, n_bootstrap
