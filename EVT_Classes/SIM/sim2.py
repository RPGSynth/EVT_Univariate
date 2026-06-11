from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import genextreme

from EVT_Classes.SIM.generator import generate_gev_dataset_blobs, generate_gev_dataset_linear
from EVT_Classes.selector import select_spatial_neighborhood

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
    "compute_combined_sim_from_data",
    "compute_combined_sim",
    "train_weight_model",
    "train_weight_model_bootstrap",
    "build_gplus_adapter",
    "plot_generated_data_map",
    "plot_weight_map",
    "run_all_locations_sim",
    "save_weight_artifact",
    "load_weight_artifact",
]

DEFAULT_SIM_SELECTOR_ARGS = {
    "mode": "full",
    "radius": None,
    "max_points": None,
    "include_center": True,
}

DEFAULT_TRAIN_SELECTOR_ARGS = {
    "mode": "circle",
    "radius": None,
    "max_points": 100,
    "include_center": False,
}

DEFAULT_MARGINAL_CACHE_PATH = _THIS_FILE.parent / "_cache" / "marginal_init_end1.npz"
DEFAULT_DEMO_PLOT_BUNDLE_PATH = _THIS_FILE.parent / "_cache" / "demo_plot_bundle.npz"
DEFAULT_WEIGHT_PLOT_EXPORT_DIR = _THIS_FILE.parent / "_output" / "weight_maps"
DEFAULT_WEIGHT_ARTIFACT_DIR = _THIS_FILE.parent / "_cache" / "weight_artifacts"
DEFAULT_SIM_RESULT_DIR = _THIS_FILE.parent / "_output" / "sim_results"
DEFAULT_SIM_ALL_RESULT_DIR = _THIS_FILE.parent / "_output" / "sim_results_all"
DEFAULT_WEIGHT_PLOT_VMAX_QUANTILE = 0.99
DEMO_DEFAULTS = {
    "generator": "blob",
    "n_bootstrap": 100,
    "show_plot": True,
    "run_sim": False,
    "force_retrain": False,
    "sim_n_runs": 100,
    "ref_i": 5,
    "ref_j": 6,
}
DEMO_VIEWER_SCRIPT_PATH = _THIS_FILE.parent / "_demo_viewer.py"
BLOB_DATASET_WORKER_SCRIPT_PATH = _THIS_FILE.parent / "_blob_dataset_worker.py"
WEIGHT_MAP_EXPORTER_SCRIPT_PATH = _THIS_FILE.parent / "_weight_map_exporter.py"
SIM_ALL_LOCATIONS_WORKER_SCRIPT_PATH = _THIS_FILE.parent / "_sim_all_locations_worker.py"


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


def _validate_fit_mode(fit_mode: str) -> str:
    mode = str(fit_mode).lower()
    if mode not in {"pointwise", "full"}:
        raise ValueError(f"Unsupported fit_mode={fit_mode!r}. Expected 'pointwise' or 'full'.")
    return mode


def _normalize_sim_selector_args(selector_args: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_SIM_SELECTOR_ARGS)
    if selector_args is not None:
        cfg.update(selector_args)
    cfg["mode"] = str(cfg["mode"]).lower()
    return cfg


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


def _validate_meta_for_generated_data(meta: dict[str, Any], data_shape: tuple[int, int, int]) -> None:
    required = ("x", "y", "mu", "sigma", "xi")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"`meta` is missing required keys: {missing}")

    n_time, n_lat, n_lon = data_shape

    x = np.asarray(meta["x"], dtype=np.float64)
    y = np.asarray(meta["y"], dtype=np.float64)
    mu = np.asarray(meta["mu"], dtype=np.float64)
    sigma = np.asarray(meta["sigma"], dtype=np.float64)
    xi = np.asarray(meta["xi"], dtype=np.float64)

    if x.shape != (n_lon,):
        raise ValueError(f"`meta['x']` must have shape ({n_lon},), got {x.shape}")
    if y.shape != (n_lat,):
        raise ValueError(f"`meta['y']` must have shape ({n_lat},), got {y.shape}")
    for name, value in (("mu", mu), ("sigma", sigma), ("xi", xi)):
        if value.shape != data_shape:
            raise ValueError(f"`meta['{name}']` must have shape {data_shape}, got {value.shape}")


def _is_blob_generator(generate_fn: Callable[..., Any]) -> bool:
    return (
        getattr(generate_fn, "__name__", "") == "generate_gev_dataset_blobs"
        and getattr(generate_fn, "__module__", "") == "EVT_Classes.SIM.generator"
    )


def _load_generated_dataset_bundle(bundle_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    bundle_path = Path(bundle_path)
    with np.load(bundle_path) as bundle:
        data = np.asarray(bundle["data"], dtype=np.float64)
        meta: dict[str, Any] = {
            "x": np.asarray(bundle["x"], dtype=np.float64),
            "y": np.asarray(bundle["y"], dtype=np.float64),
            "t": np.asarray(bundle["t"], dtype=np.float64),
            "s_field": np.asarray(bundle["s_field"], dtype=np.float64),
            "t_curve": np.asarray(bundle["t_curve"], dtype=np.float64),
            "mu": np.asarray(bundle["mu"], dtype=np.float64),
            "sigma": np.asarray(bundle["sigma"], dtype=np.float64),
            "xi": np.asarray(bundle["xi"], dtype=np.float64),
            "params": json.loads(str(np.asarray(bundle["params_json"]).item())),
        }
    return data, meta


def _generate_blob_dataset_out_of_process(gen_kwargs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir = _THIS_FILE.parent / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, output_path_str = tempfile.mkstemp(prefix="blob_dataset_", suffix=".npz", dir=str(cache_dir))
    os.close(fd)
    output_path = Path(output_path_str)
    try:
        cmd = [
            sys.executable,
            str(BLOB_DATASET_WORKER_SCRIPT_PATH),
            "--output",
            str(output_path),
            "--kwargs-json",
            json.dumps(gen_kwargs),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"return code {completed.returncode}"
            raise RuntimeError(f"Blob dataset worker failed: {detail}")
        return _load_generated_dataset_bundle(output_path)
    finally:
        output_path.unlink(missing_ok=True)


def _format_progress_bar(current: int, total: int, *, width: int = 28) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    width = max(8, int(width))
    filled = int(round(width * current / total))
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {current:03d}/{total:03d}"


def _import_plotting_modules() -> tuple[Any, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    return plt, Rectangle


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or (not np.isfinite(float(seconds))) or float(seconds) < 0.0:
        return "--:--:--"
    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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


def _artifact_key_payload(
    *,
    gen_kwargs: dict[str, Any],
    config: TrainingConfig,
    n_bootstrap: int,
    generator_kind: str | None = None,
    selector_args: dict[str, Any] | None = None,
) -> str:
    payload = {
        "generator_kind": None if generator_kind is None else str(generator_kind),
        "gen_kwargs": _clone_plain_object(gen_kwargs),
        "config": _config_to_dict(config),
        "n_bootstrap": int(n_bootstrap),
        "selector_args": _clone_plain_object(selector_args),
    }
    payload["config"].pop("verbose", None)
    payload["config"].pop("print_every", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _default_demo_artifact_path(
    *,
    gen_kwargs: dict[str, Any],
    config: TrainingConfig,
    n_bootstrap: int,
    generator_kind: str | None = None,
    selector_args: dict[str, Any] | None = None,
) -> Path:
    DEFAULT_WEIGHT_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_key = _artifact_key_payload(
        gen_kwargs=gen_kwargs,
        config=config,
        n_bootstrap=n_bootstrap,
        generator_kind=generator_kind,
        selector_args=selector_args,
    )
    digest = hashlib.sha1(artifact_key.encode("utf-8")).hexdigest()[:16]
    return DEFAULT_WEIGHT_ARTIFACT_DIR / f"demo_weights_k{int(n_bootstrap):03d}_{digest}.pt"


def _save_json_file(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_clone_plain_object(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _default_demo_sim_result_dir(
    *,
    generator_kind: str,
    n_bootstrap: int,
    i_ref: int,
    j_ref: int,
    t_ref: int,
    sim_n_runs: int,
) -> Path:
    return (
        DEFAULT_SIM_RESULT_DIR
        / str(generator_kind)
        / f"bootstrap_k{int(n_bootstrap):03d}"
        / f"ref_i{int(i_ref):02d}_j{int(j_ref):02d}"
        / f"t_{int(t_ref):03d}"
        / f"runs_{int(sim_n_runs):03d}"
    )


def _save_demo_sim_reports(
    *,
    generator_kind: str,
    n_bootstrap: int,
    i_ref: int,
    j_ref: int,
    t_ref: int,
    sim_n_runs: int,
    artifact_path: str | Path,
    weight_export_dir: str | Path,
    target_ess: float,
    selector_points: int,
    selector_args_local: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> Path:
    result_dir = _default_demo_sim_result_dir(
        generator_kind=generator_kind,
        n_bootstrap=n_bootstrap,
        i_ref=i_ref,
        j_ref=j_ref,
        t_ref=t_ref,
        sim_n_runs=sim_n_runs,
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generator": str(generator_kind),
        "n_bootstrap": int(n_bootstrap),
        "ref_i": int(i_ref),
        "ref_j": int(j_ref),
        "t_ref": int(t_ref),
        "sim_n_runs": int(sim_n_runs),
        "artifact_path": str(Path(artifact_path)),
        "weight_export_dir": str(Path(weight_export_dir)),
        "weighted_ess": float(target_ess),
        "local_selector_points": int(selector_points),
        "local_selector_args": _clone_plain_object(selector_args_local),
        "report_names": list(reports.keys()),
    }
    _save_json_file(result_dir / "metadata.json", metadata)
    _save_json_file(result_dir / "reports.json", reports)
    for report_name, report in reports.items():
        _save_json_file(result_dir / f"{report_name}.json", report)
    return result_dir


def _default_all_locations_sim_result_dir(
    *,
    generator_kind: str,
    n_bootstrap: int,
    t_idx: int,
    sim_n_runs: int,
    artifact_path: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    root_dir = DEFAULT_SIM_ALL_RESULT_DIR if output_dir is None else Path(output_dir)
    artifact_stem = Path(artifact_path).stem
    return (
        root_dir
        / str(generator_kind)
        / f"bootstrap_k{int(n_bootstrap):03d}"
        / f"t_{int(t_idx):03d}"
        / f"runs_{int(sim_n_runs):03d}"
        / artifact_stem
    )


def _location_label(i: int, j: int) -> str:
    return f"({int(i)},{int(j)})"


def _raw_location_output_path(
    raw_dir: str | Path,
    *,
    location_index: int,
    i: int,
    j: int,
    total_locations: int,
) -> Path:
    width = max(3, len(str(max(0, int(total_locations) - 1))))
    return Path(raw_dir) / f"loc_{int(location_index):0{width}d}_i{int(i):02d}_j{int(j):02d}.json"


def _location_coordinates(meta: dict[str, Any] | None, i: int, j: int) -> tuple[float, float]:
    if meta is not None and "x" in meta and "y" in meta:
        x = np.asarray(meta["x"], dtype=np.float64)
        y = np.asarray(meta["y"], dtype=np.float64)
        if x.ndim == 1 and y.ndim == 1 and 0 <= j < x.size and 0 <= i < y.size:
            return float(x[j]), float(y[i])
    return float(j), float(i)


def _metric_value_from_report(report: dict[str, Any], metric_name: str) -> float:
    metric_name = str(metric_name)
    if metric_name in report["gev_params"]:
        return float(report["gev_params"][metric_name]["rmse"])
    rl_key = metric_name
    if rl_key.lower().startswith("rl_"):
        rl_key = rl_key[3:]
    return float(report["return_levels"][str(rl_key)]["rmse"])


def _write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return output_path


def _load_json_file(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(payload)!r}")
    return payload


def _collect_all_location_payloads(raw_dir: str | Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        if not _location_payload_is_complete(path):
            continue
        payloads.append(_load_json_file(path))
    payloads.sort(key=lambda payload: int(payload["location_index"]))
    return payloads


def _location_payload_is_complete(path: str | Path) -> bool:
    try:
        payload = _load_json_file(path)
    except Exception:
        return False
    reports = payload.get("reports")
    if not isinstance(reports, dict):
        return False
    expected = {"pointwise", "weighted", "local", "full"}
    return expected.issubset(reports.keys())


def _all_location_metric_specs() -> list[tuple[str, str]]:
    return [
        ("mu", "rmse_mu.csv"),
        ("sigma", "rmse_sigma.csv"),
        ("xi", "rmse_xi.csv"),
        ("RL_10", "rmse_rl_10.csv"),
        ("RL_100", "rmse_rl_100.csv"),
    ]


def _rebuild_all_location_result_tables(result_dir: str | Path) -> dict[str, Path]:
    result_path = Path(result_dir)
    raw_dir = result_path / "raw"
    tables_dir = result_path / "tables"
    payloads = _collect_all_location_payloads(raw_dir)

    summary_rows: list[dict[str, Any]] = []
    for payload in payloads:
        reports = payload["reports"]
        summary_rows.append(
            {
                "location_index": int(payload["location_index"]),
                "location": str(payload["location"]),
                "i": int(payload["i"]),
                "j": int(payload["j"]),
                "x": float(payload["x"]),
                "y": float(payload["y"]),
                "weighted_ess": float(payload["weighted_ess"]),
                "local_selector_points": int(payload["local_selector_points"]),
                "pointwise_success_rate": float(reports["pointwise"]["success_rate"]),
                "weighted_success_rate": float(reports["weighted"]["success_rate"]),
                "local_success_rate": float(reports["local"]["success_rate"]),
                "full_success_rate": float(reports["full"]["success_rate"]),
                "raw_report_path": str(payload.get("raw_report_path", "")),
            }
        )

    summary_path = _write_csv_rows(
        result_path / "summary.csv",
        [
            "location_index",
            "location",
            "i",
            "j",
            "x",
            "y",
            "weighted_ess",
            "local_selector_points",
            "pointwise_success_rate",
            "weighted_success_rate",
            "local_success_rate",
            "full_success_rate",
            "raw_report_path",
        ],
        summary_rows,
    )

    table_paths: dict[str, Path] = {}
    for metric_name, filename in _all_location_metric_specs():
        table_rows: list[dict[str, Any]] = []
        for payload in payloads:
            reports = payload["reports"]
            table_rows.append(
                {
                    "location_index": int(payload["location_index"]),
                    "location": str(payload["location"]),
                    "i": int(payload["i"]),
                    "j": int(payload["j"]),
                    "x": float(payload["x"]),
                    "y": float(payload["y"]),
                    "pointwise": _metric_value_from_report(reports["pointwise"], metric_name),
                    "weighted": _metric_value_from_report(reports["weighted"], metric_name),
                    "local": _metric_value_from_report(reports["local"], metric_name),
                    "full": _metric_value_from_report(reports["full"], metric_name),
                }
            )
        table_paths[metric_name] = _write_csv_rows(
            tables_dir / filename,
            ["location_index", "location", "i", "j", "x", "y", "pointwise", "weighted", "local", "full"],
            table_rows,
        )

    return {"summary": summary_path, **table_paths}


def _demo_generator_spec(generator_kind: str) -> tuple[dict[str, Any], Callable[..., Any]]:
    kind = str(generator_kind).strip().lower()
    common_kwargs = {
        "n_lat": 10,
        "n_lon": 10,
        "n_time": 100,
        "beta_mu0": 70,
        "beta_mu_t": 0,
        "beta_ls0": float(np.log(11.0)),
        "beta_ls_s": float(np.log(2.0)),
        "beta_mu_s": 30,
        "xi_noise": True,
        "xi_noise_amp": 0.05,
        "seed_field": 32,
        "seed_sample": 2,
    }
    if kind == "blob":
        gen_kwargs = {
            **common_kwargs,
            "blob_scale_x": 24.0,
            "blob_scale_y": 9.0,
            "blob_angle": float(np.pi / 4.0),
            "blob_smoothness": 1.5,
            "blob_variance": 1.0,
        }
        return gen_kwargs, generate_gev_dataset_blobs
    if kind == "linear":
        gen_kwargs = {
            **common_kwargs,
            "a": 1.0,
            "b": 1.0,
        }
        return gen_kwargs, generate_gev_dataset_linear
    raise ValueError(f"Unsupported generator_kind={generator_kind!r}. Expected 'blob' or 'linear'.")


def _generate_demo_dataset(generator_kind: str, gen_kwargs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    kind = str(generator_kind).strip().lower()
    if kind == "blob":
        return _generate_blob_dataset_out_of_process(gen_kwargs)
    if kind == "linear":
        return generate_gev_dataset_linear(**gen_kwargs)
    raise ValueError(f"Unsupported generator_kind={generator_kind!r}. Expected 'blob' or 'linear'.")


def _import_gev() -> Any:
    import Refactor.gevPackage as gev

    return gev


def _resolve_sim_selection(
    n_lat: int,
    n_lon: int,
    i: int,
    j: int,
    fit_mode: str,
    selector_args: dict[str, Any],
) -> tuple[np.ndarray | None, int]:
    if fit_mode != "full":
        return None, 0

    s_total = n_lat * n_lon
    mode = str(selector_args["mode"]).lower()

    if mode == "full":
        return np.arange(s_total, dtype=np.int64), int(i * n_lon + j)

    sel = select_spatial_neighborhood(
        n_lat=n_lat,
        n_lon=n_lon,
        i=i,
        j=j,
        mode=mode,
        radius=selector_args["radius"],
        max_points=selector_args["max_points"],
        include_center=bool(selector_args["include_center"]),
    )
    selected_idx = np.asarray(sel.flat_idx, dtype=np.int64)
    s_idx = int(sel.target_local_idx)
    if selected_idx.size == 0:
        raise ValueError("Selector returned no points.")
    if s_idx < 0:
        raise ValueError("Target point is not included in the selected neighborhood.")
    return selected_idx, s_idx


def _calc_metrics(
    hats: np.ndarray,
    trues: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
) -> dict[str, float]:
    valid = (
        np.isfinite(hats)
        & np.isfinite(trues)
        & np.isfinite(lowers)
        & np.isfinite(uppers)
    )
    n_valid = int(valid.sum())

    if n_valid == 0:
        return {
            "call_true": np.nan,
            "rmse": np.nan,
            "nrmse": np.nan,
            "std": np.nan,
            "nstd": np.nan,
            "bias": np.nan,
            "nbias": np.nan,
            "n_valid": 0,
        }

    hats_v = hats[valid]
    trues_v = trues[valid]
    diff = hats_v - trues_v
    bias = float(np.mean(diff))
    std = float(np.std(diff, ddof=0))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    true_val = float(np.mean(trues_v))
    if np.isfinite(true_val) and true_val != 0.0:
        nrmse = float(rmse / abs(true_val))
        nstd = float(std / abs(true_val))
        nbias = float(bias / abs(true_val))
    else:
        nrmse = np.nan
        nstd = np.nan
        nbias = np.nan

    return {
        "call_true": true_val,
        "rmse": rmse,
        "nrmse": nrmse,
        "std": std,
        "nstd": nstd,
        "bias": bias,
        "nbias": nbias,
        "n_valid": n_valid,
    }


def _coerce_weights_for_fit(
    w_full: Any,
    *,
    data_shape: tuple[int, int, int],
    fit_mode: str,
    selected_idx: np.ndarray | None,
    target_flat: int,
    i: int,
    j: int,
) -> np.ndarray | None:
    if w_full is None:
        return None

    n_time, n_lat, n_lon = data_shape
    s_total = n_lat * n_lon
    if hasattr(w_full, "detach"):
        w_arr = w_full.detach().cpu().numpy()
    else:
        w_arr = np.asarray(w_full)
    w_arr = np.asarray(w_arr, dtype=np.float64)

    if fit_mode == "full":
        if w_arr.shape == data_shape:
            w2d = w_arr.reshape(n_time, s_total)
        elif w_arr.shape == (n_time, s_total):
            w2d = w_arr
        elif np.isscalar(w_arr) or w_arr.size == 1:
            w2d = np.full((n_time, s_total), float(np.ravel(w_arr)[0]), dtype=np.float64)
        else:
            raise ValueError(
                f"Unsupported full weights shape {w_arr.shape}. Expected (T,H,W), (T,S), or scalar."
            )

        if selected_idx is None:
            raise ValueError("selected_idx must be provided for fit_mode='full'.")
        w_subset = w2d[:, selected_idx]

    else:
        if w_arr.shape == data_shape:
            w_vec = w_arr[:, i, j]
        elif w_arr.shape == (n_time, s_total):
            w_vec = w_arr[:, target_flat]
        elif w_arr.ndim == 1 and w_arr.shape == (n_time,):
            w_vec = w_arr
        elif np.isscalar(w_arr) or w_arr.size == 1:
            w_vec = np.full(n_time, float(np.ravel(w_arr)[0]), dtype=np.float64)
        else:
            raise ValueError(
                f"Unsupported pointwise weights shape {w_arr.shape}. Expected (T,H,W), (T,S), (T,), or scalar."
            )

        w_subset = np.asarray(w_vec, dtype=np.float64).reshape(n_time, 1)

    w_subset = np.nan_to_num(w_subset, nan=0.0, posinf=0.0, neginf=0.0)
    w_subset = np.clip(w_subset, 0.0, None)
    if (not np.isfinite(w_subset).all()) or float(w_subset.sum()) <= 0.0:
        return None
    return w_subset


def _build_exog(
    meta: dict[str, Any],
    *,
    fit_mode: str,
    use_spatial_covariates: bool,
    selected_idx: np.ndarray | None,
    n_time: int,
    n_lat: int,
    n_lon: int,
) -> dict[str, np.ndarray | None]:
    if not use_spatial_covariates:
        return {"location": None, "scale": None, "shape": None}
    if fit_mode != "full" or selected_idx is None or selected_idx.size <= 1:
        return {"location": None, "scale": None, "shape": None}

    lat_vals = np.asarray(meta["y"], dtype=np.float64)
    lon_vals = np.asarray(meta["x"], dtype=np.float64)
    lat_idx, lon_idx = np.unravel_index(selected_idx, (n_lat, n_lon))

    lat = np.broadcast_to(lat_vals[lat_idx][None, :], (n_time, selected_idx.size))
    lon = np.broadcast_to(lon_vals[lon_idx][None, :], (n_time, selected_idx.size))
    exog_loc = np.stack([lat, lon], axis=2)
    exog_scale = np.stack([lat, lon], axis=2)

    return {
        "location": exog_loc,
        "scale": exog_scale,
        "shape": None,
    }


def _evaluate_params_at_target(
    fit: Any,
    params: np.ndarray,
    *,
    t_idx: int,
    s_idx: int,
) -> tuple[float, float, float]:
    p = np.asarray(params, dtype=np.float64).ravel()
    d_loc, d_scale, d_shape = fit.dims

    beta_loc = p[:d_loc]
    beta_scale = p[d_loc:d_loc + d_scale]
    beta_shape = p[d_loc + d_scale:d_loc + d_scale + d_shape]

    x_loc = np.asarray(fit.data.exog_loc[t_idx, s_idx, :], dtype=np.float64)
    x_scale = np.asarray(fit.data.exog_scale[t_idx, s_idx, :], dtype=np.float64)
    x_shape = np.asarray(fit.data.exog_shape[t_idx, s_idx, :], dtype=np.float64)

    mu = float(x_loc @ beta_loc)
    raw_scale = float(x_scale @ beta_scale)
    sigma = float(fit.linker.np_transform_scale(np.array([raw_scale], dtype=np.float64))[0])
    xi = float(x_shape @ beta_shape)
    return mu, sigma, xi


def _fit_single_dataset(
    data: Any,
    meta: dict[str, Any],
    i: int,
    j: int,
    *,
    return_periods: list[int | float] | tuple[int | float, ...] | np.ndarray | None,
    t_idx: int,
    confidence: float,
    fit_mode: str,
    weights: Callable[..., Any] | None,
    selector_args: dict[str, Any] | None,
    use_spatial_covariates: bool,
) -> tuple[dict[str, tuple[float, float, float, float]], dict[int | float, tuple[float, float, float, float]]]:
    data_np = _validate_data_3d(data)
    _validate_meta_for_generated_data(meta, tuple(data_np.shape))
    n_time, n_lat, n_lon = data_np.shape
    i, j = _validate_index(i, j, n_lat, n_lon)
    fit_mode = _validate_fit_mode(fit_mode)

    if not (0.0 < float(confidence) < 1.0):
        raise ValueError("`confidence` must be in (0, 1).")

    t_idx = int(t_idx)
    if not (0 <= t_idx < n_time):
        raise ValueError(f"`t_idx` must be in [0, {n_time - 1}], got {t_idx}")

    if return_periods is None:
        return_periods = [100]
    t_arr = np.atleast_1d(np.asarray(return_periods, dtype=np.float64))
    if t_arr.size == 0:
        raise ValueError("`return_periods` must be a non-empty sequence.")

    sel_cfg = _normalize_sim_selector_args(selector_args)
    selected_idx, s_idx = _resolve_sim_selection(n_lat, n_lon, i, j, fit_mode, sel_cfg)

    s_total = n_lat * n_lon
    target_flat = int(i * n_lon + j)
    if fit_mode == "full":
        endog = data_np.reshape(n_time, s_total)[:, selected_idx].astype(np.float64, copy=False)
    else:
        endog = data_np[:, i, j].astype(np.float64, copy=False)

    if weights is None:
        w_subset = None
    else:
        w_full = weights(i=i, j=j, t_idx=t_idx, data_shape=data_np.shape)
        w_subset = _coerce_weights_for_fit(
            w_full,
            data_shape=tuple(data_np.shape),
            fit_mode=fit_mode,
            selected_idx=selected_idx,
            target_flat=target_flat,
            i=i,
            j=j,
        )

    exog = _build_exog(
        meta,
        fit_mode=fit_mode,
        use_spatial_covariates=use_spatial_covariates,
        selected_idx=selected_idx,
        n_time=n_time,
        n_lat=n_lat,
        n_lon=n_lon,
    )

    fit = _import_gev().GEVModel().fit(endog=endog, exog=exog, weights=w_subset)
    theta = np.asarray(fit.params, dtype=np.float64).ravel()
    lo, hi = fit.ci(confidence=confidence)
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()

    mu_hat, sigma_hat, xi_hat = _evaluate_params_at_target(fit, theta, t_idx=t_idx, s_idx=s_idx)
    mu_lo, sigma_lo, xi_lo = _evaluate_params_at_target(fit, lo, t_idx=t_idx, s_idx=s_idx)
    mu_hi, sigma_hi, xi_hi = _evaluate_params_at_target(fit, hi, t_idx=t_idx, s_idx=s_idx)

    mu_true = float(np.asarray(meta["mu"], dtype=np.float64)[t_idx, i, j])
    sigma_true = float(np.asarray(meta["sigma"], dtype=np.float64)[t_idx, i, j])
    xi_true = float(np.asarray(meta["xi"], dtype=np.float64)[t_idx, i, j])

    res_params = {
        "mu": (mu_hat, mu_true, mu_lo, mu_hi),
        "sigma": (sigma_hat, sigma_true, sigma_lo, sigma_hi),
        "xi": (xi_hat, xi_true, xi_lo, xi_hi),
    }

    rl_true_arr = genextreme.ppf(
        1.0 - 1.0 / t_arr,
        c=-xi_true,
        loc=mu_true,
        scale=sigma_true,
    )
    rl_obj, _, ci_obj = fit.return_level(t=t_idx, s=s_idx).compute(T=t_arr, confidence=confidence)

    res_rls: dict[int | float, tuple[float, float, float, float]] = {}
    for idx, return_period in enumerate(t_arr.tolist()):
        res_rls[return_period] = (
            float(rl_obj[0, 0, idx]),
            float(rl_true_arr[idx]),
            float(ci_obj[0, 0, idx, 0]),
            float(ci_obj[0, 0, idx, 1]),
        )

    return res_params, res_rls


def _assemble_sim_report(
    *,
    i: int,
    j: int,
    t_idx: int,
    n_runs: int,
    n_success: int,
    raw_params: dict[str, list[np.ndarray]],
    raw_rls: dict[int | float, list[np.ndarray]],
    true_params: dict[str, float] | None,
    true_rls: dict[str, float] | None,
) -> dict[str, Any]:
    return {
        "pixel": (int(i), int(j)),
        "t_idx": int(t_idx),
        "n_runs": int(n_runs),
        "success_rate": float(n_success / max(1, n_runs)),
        "true_params": true_params or {},
        "true_return_levels": true_rls or {},
        "gev_params": {name: _calc_metrics(*raw_params[name]) for name in raw_params},
        "return_levels": {_return_period_key(key): _calc_metrics(*raw_rls[key]) for key in raw_rls},
    }


def _return_period_key(value: int | float) -> str:
    value_f = float(value)
    if np.isfinite(value_f) and value_f.is_integer():
        return str(int(value_f))
    return str(value)


def _print_sim_report(
    report: dict[str, Any],
    *,
    fit_mode: str,
    weighted: bool,
    use_spatial_covariates: bool,
) -> None:
    i, j = report["pixel"]
    w_str = "[Weighted]" if weighted else "[Unweighted]"
    cov_str = "[SpatialCov]" if use_spatial_covariates else "[NoSpatialCov]"
    print("=" * 110)
    print(f" SIMULATION REPORT | Pixel ({i},{j}) | {fit_mode} | {w_str} | {cov_str}")
    print(
        f" Success: {int(round(report['success_rate'] * report['n_runs']))}/{report['n_runs']} "
        f"({report['success_rate']:.1%}) | T_idx: {report['t_idx']}"
    )
    print("-" * 110)

    h_fmt = "{:<12} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}"
    row_fmt = "{:<12} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10.4f} | {:>10.4f}"
    print(h_fmt.format("Metric", "True", "RMSE", "NRMSE", "Std", "Nstd", "Bias", "Nbias"))
    print("-" * 110)

    for key in ("mu", "sigma", "xi"):
        stats = report["gev_params"][key]
        print(
            row_fmt.format(
                key,
                stats["call_true"],
                stats["rmse"],
                stats["nrmse"],
                stats["std"],
                stats["nstd"],
                stats["bias"],
                stats["nbias"],
            )
        )

    print("-" * 110)
    rl_keys = sorted(report["return_levels"].keys(), key=lambda value: float(value))
    for rl_key in rl_keys:
        stats = report["return_levels"][rl_key]
        print(
            row_fmt.format(
                f"RL_{rl_key}",
                stats["call_true"],
                stats["rmse"],
                stats["nrmse"],
                stats["std"],
                stats["nstd"],
                stats["bias"],
                stats["nbias"],
            )
        )
    print("=" * 110 + "\n")


def compute_combined_sim_from_data(
    data: Any,
    meta: dict[str, Any],
    i: int,
    j: int,
    *,
    n_runs: int = 1,
    return_periods: list[int | float] | tuple[int | float, ...] | np.ndarray | None = None,
    t_idx: int = 30,
    confidence: float = 0.95,
    fit_mode: str = "pointwise",
    weights: Callable[..., Any] | None = None,
    selector_args: dict[str, Any] | None = None,
    use_spatial_covariates: bool = False,
) -> dict[str, Any]:
    if int(n_runs) != 1:
        raise ValueError("`compute_combined_sim_from_data` works on one dataset and requires n_runs=1.")

    res_params, res_rls = _fit_single_dataset(
        data=data,
        meta=meta,
        i=i,
        j=j,
        return_periods=return_periods,
        t_idx=t_idx,
        confidence=confidence,
        fit_mode=fit_mode,
        weights=weights,
        selector_args=selector_args,
        use_spatial_covariates=use_spatial_covariates,
    )

    params_names = ("mu", "sigma", "xi")
    return_periods_arr = np.atleast_1d(np.asarray([100] if return_periods is None else return_periods, dtype=np.float64))
    raw_params = {name: [np.full(1, np.nan) for _ in range(4)] for name in params_names}
    raw_rls = {value: [np.full(1, np.nan) for _ in range(4)] for value in return_periods_arr.tolist()}

    for name, values in res_params.items():
        for metric_idx in range(4):
            raw_params[name][metric_idx][0] = values[metric_idx]

    for return_period, values in res_rls.items():
        for metric_idx in range(4):
            raw_rls[return_period][metric_idx][0] = values[metric_idx]

    true_params = {name: float(values[1]) for name, values in res_params.items()}
    true_rls = {_return_period_key(key): float(values[1]) for key, values in res_rls.items()}

    return _assemble_sim_report(
        i=i,
        j=j,
        t_idx=t_idx,
        n_runs=1,
        n_success=1,
        raw_params=raw_params,
        raw_rls=raw_rls,
        true_params=true_params,
        true_rls=true_rls,
    )


def compute_combined_sim(
    generate_fn: Callable[..., tuple[np.ndarray, dict[str, Any]]],
    gen_kwargs: dict[str, Any],
    i: int,
    j: int,
    *,
    n_runs: int = 5,
    return_periods: list[int | float] | tuple[int | float, ...] | np.ndarray | None = None,
    t_idx: int = 30,
    n_jobs_runs: int = -1,
    backend: str = "threading",
    verbose: bool = True,
    confidence: float = 0.95,
    fit_mode: str = "pointwise",
    weights: Callable[..., Any] | None = None,
    selector_args: dict[str, Any] | None = None,
    use_spatial_covariates: bool = False,
) -> dict[str, Any]:
    n_runs = int(n_runs)
    if n_runs < 1:
        raise ValueError("`n_runs` must be >= 1.")

    fit_mode = _validate_fit_mode(fit_mode)
    return_periods_arr = np.atleast_1d(np.asarray([100] if return_periods is None else return_periods, dtype=np.float64))

    params_names = ("mu", "sigma", "xi")
    raw_params = {name: [np.full(n_runs, np.nan) for _ in range(4)] for name in params_names}
    raw_rls = {value: [np.full(n_runs, np.nan) for _ in range(4)] for value in return_periods_arr.tolist()}

    def one_run(
        run_idx: int,
    ) -> tuple[
        int,
        dict[str, tuple[float, float, float, float]] | None,
        dict[int | float, tuple[float, float, float, float]] | None,
        str | None,
    ]:
        run_kwargs = dict(gen_kwargs)
        seed_field = int(run_kwargs.get("seed_field", 2025))
        run_kwargs["seed_field"] = seed_field

        if run_kwargs.get("seed_sample") is not None:
            seed_sample_base = int(run_kwargs["seed_sample"])
        elif run_kwargs.get("seed") is not None:
            seed_sample_base = int(run_kwargs["seed"])
        else:
            seed_sample_base = seed_field + 1

        run_kwargs["seed_sample"] = seed_sample_base + run_idx
        run_kwargs["seed"] = None

        try:
            if _is_blob_generator(generate_fn):
                data, meta = _generate_blob_dataset_out_of_process(run_kwargs)
            else:
                data, meta = generate_fn(**run_kwargs)
            res_params, res_rls = _fit_single_dataset(
                data=data,
                meta=meta,
                i=i,
                j=j,
                return_periods=return_periods_arr,
                t_idx=t_idx,
                confidence=confidence,
                fit_mode=fit_mode,
                weights=weights,
                selector_args=selector_args,
                use_spatial_covariates=use_spatial_covariates,
            )
            return run_idx, res_params, res_rls, None
        except Exception as exc:
            return run_idx, None, None, repr(exc)

    if n_jobs_runs == 1:
        results = [one_run(run_idx) for run_idx in range(n_runs)]
    else:
        results = Parallel(n_jobs=n_jobs_runs, backend=backend, verbose=0)(
            delayed(one_run)(run_idx) for run_idx in range(n_runs)
        )

    n_success = 0
    true_params: dict[str, float] | None = None
    true_rls: dict[str, float] | None = None
    first_error: str | None = None

    for run_idx, p_res, r_res, err in results:
        if err is not None:
            if first_error is None:
                first_error = err
            continue

        n_success += 1
        if true_params is None:
            true_params = {name: float(values[1]) for name, values in p_res.items()}
        if true_rls is None:
            true_rls = {_return_period_key(key): float(values[1]) for key, values in r_res.items()}

        for name in params_names:
            values = p_res[name]
            for metric_idx in range(4):
                raw_params[name][metric_idx][run_idx] = values[metric_idx]

        for return_period in return_periods_arr.tolist():
            values = r_res[return_period]
            for metric_idx in range(4):
                raw_rls[return_period][metric_idx][run_idx] = values[metric_idx]

    final_report = _assemble_sim_report(
        i=i,
        j=j,
        t_idx=t_idx,
        n_runs=n_runs,
        n_success=n_success,
        raw_params=raw_params,
        raw_rls=raw_rls,
        true_params=true_params,
        true_rls=true_rls,
    )

    if verbose:
        if first_error is not None and n_success < n_runs:
            print(f"First failed run error: {first_error}")
        _print_sim_report(
            final_report,
            fit_mode=fit_mode,
            weighted=weights is not None,
            use_spatial_covariates=use_spatial_covariates,
        )

    return final_report


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


def _ess_from_weight_map(weight_map: Any) -> float:
    weights = np.asarray(weight_map, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(f"`weight_map` must have shape (H, W), got {weights.shape}")
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, None)
    mass = float(weights.sum())
    if (not np.isfinite(mass)) or mass <= 0.0:
        return 1.0
    weights = weights / mass
    return float(1.0 / np.clip(np.sum(weights ** 2), 1e-12, None))


def _selector_args_for_comparable_ess(
    target_ess: float,
    *,
    n_lat: int,
    n_lon: int,
) -> tuple[dict[str, Any], int]:
    s_total = int(n_lat) * int(n_lon)
    selector_points = int(np.clip(int(round(float(target_ess))), 1, s_total))
    selector_args = {
        "mode": "circle",
        "radius": None,
        "max_points": selector_points,
        "include_center": True,
    }
    return selector_args, selector_points


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


def plot_weight_map(
    adapter: Callable[..., Any],
    data_shape: tuple[int, int, int],
    i: int,
    j: int,
    t_idx: int,
    *,
    ax: Any = None,
    weight_vmin: float | None = None,
    weight_vmax: float | None = None,
) -> np.ndarray:
    plt, _ = _import_plotting_modules()
    w_full = adapter(i=i, j=j, t_idx=t_idx, data_shape=data_shape)
    w_arr = np.asarray(w_full, dtype=np.float64)
    if w_arr.shape != tuple(data_shape):
        raise ValueError(f"Adapter returned shape {w_arr.shape}, expected {tuple(data_shape)}")
    if not (0 <= int(t_idx) < int(data_shape[0])):
        raise ValueError(f"`t_idx` must be in [0, {int(data_shape[0]) - 1}], got {t_idx}")

    w_map = np.asarray(w_arr[t_idx], dtype=np.float64)
    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
        created_fig = True

    if isinstance(adapter, BootstrapAveragedWeightAdapterGPlus):
        n_models = len(adapter.adapters)
        weight_title = f"Bootstrap-averaged weights (K={n_models}) at (i,j)=({i},{j}), t={t_idx}"
    else:
        weight_title = f"Weights at (i,j)=({i},{j}), t={t_idx}"

    im = ax.imshow(w_map, origin="lower", cmap="viridis", vmin=weight_vmin, vmax=weight_vmax)
    ax.scatter([j], [i], c="red", marker="x", s=140, linewidths=2, label="Reference point")
    ax.set_title(weight_title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.legend(frameon=False, loc="upper right")
    plt.colorbar(im, ax=ax, label="Weight")
    return w_map


def plot_generated_data_map(
    data: Any,
    meta: dict[str, Any] | None,
    i: int,
    j: int,
    t_idx: int,
    *,
    field_name: str = "mu",
    ax: Any = None,
) -> np.ndarray:
    plt, Rectangle = _import_plotting_modules()
    data_np = _validate_data_3d(data)
    n_time, n_lat, n_lon = data_np.shape
    i, j = _validate_index(i, j, n_lat, n_lon)
    t_idx = int(t_idx)
    if not (0 <= t_idx < n_time):
        raise ValueError(f"`t_idx` must be in [0, {n_time - 1}], got {t_idx}")

    field_name = str(field_name)
    plot_title = f"Generated {field_name} field at t={t_idx}"
    colorbar_label = field_name
    cmap = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    if meta is not None and field_name in meta:
        field_raw = np.asarray(meta[field_name], dtype=np.float64)
        if field_raw.shape == (n_time, n_lat, n_lon):
            field = np.asarray(field_raw[t_idx], dtype=np.float64)
        elif field_raw.shape == (n_lat, n_lon):
            field = np.asarray(field_raw, dtype=np.float64)
            plot_title = f"Generated {field_name} field"
        else:
            raise ValueError(
                f"meta[{field_name!r}] has shape {field_raw.shape}, expected {(n_time, n_lat, n_lon)} or {(n_lat, n_lon)}"
            )
    else:
        field = np.asarray(data_np[t_idx], dtype=np.float64)
        plot_title = f"Generated data realization at t={t_idx}"
        colorbar_label = "GEV value"

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    extent = None
    xlabel = "j"
    ylabel = "i"
    rectangle_patch: Any = None
    use_rectangle = field_name == "s_field"
    if meta is not None and "x" in meta and "y" in meta:
        x = np.asarray(meta["x"], dtype=np.float64)
        y = np.asarray(meta["y"], dtype=np.float64)
        if x.ndim == 1 and y.ndim == 1 and x.size == n_lon and y.size == n_lat:
            if use_rectangle:
                dx = float(x[1] - x[0]) if x.size > 1 else 1.0
                dy = float(y[1] - y[0]) if y.size > 1 else 1.0
                x_edges = np.linspace(float(x[0]) - dx / 2.0, float(x[-1]) + dx / 2.0, x.size + 1)
                y_edges = np.linspace(float(y[0]) - dy / 2.0, float(y[-1]) + dy / 2.0, y.size + 1)
                extent = (float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1]))
                rectangle_patch = Rectangle(
                    (float(x_edges[j]), float(y_edges[i])),
                    float(x_edges[j + 1] - x_edges[j]),
                    float(y_edges[i + 1] - y_edges[i]),
                    fill=False,
                    edgecolor="red",
                    linewidth=2.5,
                )
            else:
                extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
            xlabel = "x"
            ylabel = "y"
            x_ref = float(x[j])
            y_ref = float(y[i])
            scatter_x = [x_ref]
            scatter_y = [y_ref]
        else:
            scatter_x = [j]
            scatter_y = [i]
    else:
        scatter_x = [j]
        scatter_y = [i]

    if field_name == "s_field":
        plot_title = "Spatial covariate field"
        colorbar_label = "Field intensity (0-1)"
        cmap = "YlGnBu"

    finite = field[np.isfinite(field)]
    if vmax is None:
        vmax = float(np.quantile(finite, 0.98)) if finite.size else None
    if field_name == "s_field":
        im = ax.imshow(
            field,
            origin="lower",
            cmap="YlGnBu",
            extent=extent,
            aspect="auto",
        )
    else:
        im = ax.imshow(
            field,
            origin="lower",
            cmap=cmap,
            extent=extent,
            aspect="auto",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )
    if rectangle_patch is not None:
        ax.add_patch(rectangle_patch)
    else:
        ax.scatter(scatter_x, scatter_y, c="red", marker="x", s=140, linewidths=2, label="Reference point")
    ax.set_title(plot_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if rectangle_patch is None:
        ax.legend(frameon=False, loc="upper right")
    plt.colorbar(im, ax=ax, label=colorbar_label)
    return field


def _save_demo_plot_bundle(
    data: np.ndarray,
    meta: dict[str, Any] | None,
    adapter: Callable[..., Any],
    *,
    i: int,
    j: int,
    t_idx: int,
    field_name: str = "mu",
    bundle_path: str | Path | None = None,
    weight_vmin: float | None = None,
    weight_vmax: float | None = None,
) -> Path:
    data_np = _validate_data_3d(data)
    n_time, n_lat, n_lon = data_np.shape
    i, j = _validate_index(i, j, n_lat, n_lon)
    t_idx = int(t_idx)
    if not (0 <= t_idx < n_time):
        raise ValueError(f"`t_idx` must be in [0, {n_time - 1}], got {t_idx}")

    weight_cube = np.asarray(adapter(i=i, j=j, t_idx=t_idx, data_shape=data_np.shape), dtype=np.float64)
    if weight_cube.shape != data_np.shape:
        raise ValueError(f"Adapter returned shape {weight_cube.shape}, expected {data_np.shape}")

    path = DEFAULT_DEMO_PLOT_BUNDLE_PATH if bundle_path is None else Path(bundle_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(meta["x"], dtype=np.float64) if meta is not None and "x" in meta else np.array([], dtype=np.float64)
    y = np.asarray(meta["y"], dtype=np.float64) if meta is not None and "y" in meta else np.array([], dtype=np.float64)
    if isinstance(adapter, BootstrapAveragedWeightAdapterGPlus):
        n_models = len(adapter.adapters)
        weight_title = f"Bootstrap-averaged weights (K={n_models}) at (i,j)=({i},{j}), t={t_idx}"
    else:
        weight_title = f"Weights at (i,j)=({i},{j}), t={t_idx}"
    if meta is not None and field_name in meta:
        field_raw = np.asarray(meta[field_name], dtype=np.float64)
        if field_raw.shape == (n_time, n_lat, n_lon):
            generated_map = np.asarray(field_raw[t_idx], dtype=np.float64)
            generated_title = f"Generated {field_name} field at t={t_idx}"
        elif field_raw.shape == (n_lat, n_lon):
            generated_map = np.asarray(field_raw, dtype=np.float64)
            generated_title = f"Generated {field_name} field"
        else:
            raise ValueError(
                f"meta[{field_name!r}] has shape {field_raw.shape}, expected {(n_time, n_lat, n_lon)} or {(n_lat, n_lon)}"
            )
        generated_label = field_name
    else:
        generated_map = np.asarray(data_np[t_idx], dtype=np.float64)
        generated_title = f"Generated data realization at t={t_idx}"
        generated_label = "GEV value"
    np.savez(
        path,
        generated_map=generated_map,
        weight_map=np.asarray(weight_cube[t_idx], dtype=np.float64),
        x=x,
        y=y,
        i=np.asarray(i, dtype=np.int64),
        j=np.asarray(j, dtype=np.int64),
        t_idx=np.asarray(t_idx, dtype=np.int64),
        generated_title=np.asarray(generated_title),
        generated_label=np.asarray(generated_label),
        generated_field_name=np.asarray(field_name),
        weight_title=np.asarray(weight_title),
        weight_vmin=np.asarray(np.nan if weight_vmin is None else float(weight_vmin), dtype=np.float64),
        weight_vmax=np.asarray(np.nan if weight_vmax is None else float(weight_vmax), dtype=np.float64),
    )
    return path


def _launch_demo_plot_viewer(bundle_path: str | Path) -> None:
    bundle_path = Path(bundle_path).resolve()
    cmd = [sys.executable, str(DEMO_VIEWER_SCRIPT_PATH), "--bundle", str(bundle_path), "--show"]
    child_env = dict(os.environ)
    child_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    popen_kwargs: dict[str, Any] = {
        "cwd": str(_REPO_ROOT),
        "close_fds": True,
        "env": child_env,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **popen_kwargs)


def _save_all_weight_maps(
    adapter: Callable[..., Any],
    data_shape: tuple[int, int, int],
    *,
    t_idx: int,
    output_dir: str | Path | None = None,
) -> Path:
    n_time, n_lat, n_lon = tuple(int(value) for value in data_shape)
    t_idx = int(t_idx)
    if not (0 <= t_idx < n_time):
        raise ValueError(f"`t_idx` must be in [0, {n_time - 1}], got {t_idx}")

    root_dir = DEFAULT_WEIGHT_PLOT_EXPORT_DIR if output_dir is None else Path(output_dir)
    export_dir = root_dir / f"t_{t_idx:03d}"
    export_dir.mkdir(parents=True, exist_ok=True)

    weight_maps = np.empty((n_lat, n_lon, n_lat, n_lon), dtype=np.float64)
    if isinstance(adapter, BootstrapAveragedWeightAdapterGPlus):
        n_models = len(adapter.adapters)
        title_prefix = f"Bootstrap-averaged weights (K={n_models})"
    else:
        title_prefix = "Weights"

    for i in range(n_lat):
        for j in range(n_lon):
            w_full = np.asarray(adapter(i=i, j=j, t_idx=t_idx, data_shape=data_shape), dtype=np.float64)
            if w_full.shape != tuple(data_shape):
                raise ValueError(f"Adapter returned shape {w_full.shape}, expected {tuple(data_shape)}")
            weight_maps[i, j] = np.asarray(w_full[t_idx], dtype=np.float64)

    finite = weight_maps[np.isfinite(weight_maps)]
    weight_vmin = 0.0
    if finite.size:
        weight_vmax = float(np.quantile(finite, DEFAULT_WEIGHT_PLOT_VMAX_QUANTILE))
        true_max = float(finite.max())
        if (not np.isfinite(weight_vmax)) or weight_vmax <= weight_vmin:
            weight_vmax = true_max
    else:
        weight_vmax = 1.0
    if (not np.isfinite(weight_vmax)) or weight_vmax <= weight_vmin:
        weight_vmax = 1.0
    print(
        f"weight plot color scale: vmin={weight_vmin:.6f}, "
        f"vmax={weight_vmax:.6f} (quantile={DEFAULT_WEIGHT_PLOT_VMAX_QUANTILE:.3f})"
    )

    cache_dir = DEFAULT_MARGINAL_CACHE_PATH.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, bundle_path_str = tempfile.mkstemp(prefix="weight_map_export_", suffix=".npz", dir=str(cache_dir))
    os.close(fd)
    bundle_path = Path(bundle_path_str)
    np.savez(
        bundle_path,
        weight_maps=weight_maps,
        t_idx=np.asarray(t_idx, dtype=np.int64),
        title_prefix=np.asarray(title_prefix),
        weight_vmin=np.asarray(weight_vmin, dtype=np.float64),
        weight_vmax=np.asarray(weight_vmax, dtype=np.float64),
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(WEIGHT_MAP_EXPORTER_SCRIPT_PATH),
                "--bundle",
                str(bundle_path),
                "--output-dir",
                str(export_dir),
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE")},
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Weight-map export worker failed:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    finally:
        bundle_path.unlink(missing_ok=True)

    return export_dir, weight_vmin, weight_vmax


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


def _load_or_train_demo_artifact(
    *,
    generator_kind: str,
    n_bootstrap: int,
    force_retrain: bool,
) -> tuple[TrainedWeightModel | BootstrapTrainingResult, Path, dict[str, Any], Callable[..., Any], TrainingConfig]:
    generator_kind = str(generator_kind).strip().lower()
    gen_kwargs, generate_fn = _demo_generator_spec(generator_kind)

    demo_cfg = TrainingConfig(
        outer_iters=100,
        kernel_epochs_per_outer=1,
        theta_epochs_per_outer=10,
        print_every=1,
        verbose=True,
    )
    n_bootstrap = int(n_bootstrap)
    if n_bootstrap < 1:
        raise ValueError(f"`n_bootstrap` must be >= 1, got {n_bootstrap}")
    artifact_path = _default_demo_artifact_path(
        gen_kwargs=gen_kwargs,
        config=demo_cfg,
        n_bootstrap=n_bootstrap,
        generator_kind=generator_kind,
        selector_args=None,
    )

    artifact_obj: TrainedWeightModel | BootstrapTrainingResult | None = None
    if artifact_path.exists() and not force_retrain:
        try:
            artifact_obj = load_weight_artifact(artifact_path)
            print(f"loaded weight artifact from {artifact_path}")
        except Exception as exc:
            print(f"failed to load weight artifact at {artifact_path}; retraining ({exc})")

    if artifact_obj is None:
        data, meta = _generate_demo_dataset(generator_kind, gen_kwargs)
        if n_bootstrap == 1:
            trained_single = train_weight_model(data, config=demo_cfg)
            trained_single.meta = _clone_plain_object(meta)
            save_weight_artifact(trained_single, artifact_path)
            artifact_obj = trained_single
        else:
            bootstrap_result = train_weight_model_bootstrap(
                data,
                config=demo_cfg,
                n_bootstrap=n_bootstrap,
            )
            bootstrap_result.meta = _clone_plain_object(meta)
            bootstrap_result.best_model.meta = _clone_plain_object(meta)
            if bootstrap_result.run_models:
                for model in bootstrap_result.run_models:
                    model.meta = _clone_plain_object(meta)
            save_weight_artifact(bootstrap_result, artifact_path)
            artifact_obj = bootstrap_result
        print(f"saved weight artifact to {artifact_path}")

    return artifact_obj, artifact_path, gen_kwargs, generate_fn, demo_cfg


def _compute_single_location_sim_payload(
    *,
    artifact_path: str | Path,
    generator_kind: str,
    i: int,
    j: int,
    t_idx: int,
    sim_n_runs: int,
) -> dict[str, Any]:
    artifact_obj = load_weight_artifact(artifact_path)
    trained, adapter, meta, data, _ = _resolve_artifact_runtime_state(artifact_obj)
    i, j = _validate_index(i, j, data.shape[1], data.shape[2])
    t_idx = int(t_idx)
    if not (0 <= t_idx < data.shape[0]):
        raise ValueError(f"`t_idx` must be in [0, {data.shape[0] - 1}], got {t_idx}")

    gen_kwargs, generate_fn = _demo_generator_spec(generator_kind)
    weight_cube = np.asarray(adapter(i=i, j=j, t_idx=t_idx, data_shape=data.shape), dtype=np.float64)
    if weight_cube.shape != tuple(data.shape):
        raise ValueError(f"Adapter returned shape {weight_cube.shape}, expected {tuple(data.shape)}")

    target_ess = _ess_from_weight_map(weight_cube[t_idx])
    selector_args_local, selector_points = _selector_args_for_comparable_ess(
        target_ess,
        n_lat=data.shape[1],
        n_lon=data.shape[2],
    )
    reports = {
        "weighted": compute_combined_sim(
            generate_fn=generate_fn,
            gen_kwargs=gen_kwargs,
            i=i,
            j=j,
            n_runs=sim_n_runs,
            fit_mode="full",
            t_idx=t_idx,
            return_periods=[10, 100],
            weights=adapter,
            verbose=False,
        ),
        "local": compute_combined_sim(
            generate_fn=generate_fn,
            gen_kwargs=gen_kwargs,
            i=i,
            j=j,
            n_runs=sim_n_runs,
            fit_mode="full",
            t_idx=t_idx,
            return_periods=[10, 100],
            selector_args=selector_args_local,
            weights=None,
            verbose=False,
        ),
        "pointwise": compute_combined_sim(
            generate_fn=generate_fn,
            gen_kwargs=gen_kwargs,
            i=i,
            j=j,
            n_runs=sim_n_runs,
            fit_mode="pointwise",
            t_idx=t_idx,
            return_periods=[10, 100],
            weights=None,
            verbose=False,
        ),
        "full": compute_combined_sim(
            generate_fn=generate_fn,
            gen_kwargs=gen_kwargs,
            i=i,
            j=j,
            n_runs=sim_n_runs,
            fit_mode="full",
            t_idx=t_idx,
            return_periods=[10, 100],
            selector_args=None,
            weights=None,
            verbose=False,
        ),
    }
    x_coord, y_coord = _location_coordinates(meta, i, j)
    location_index = int(i * data.shape[2] + j)
    return {
        "location_index": location_index,
        "location": _location_label(i, j),
        "i": int(i),
        "j": int(j),
        "x": float(x_coord),
        "y": float(y_coord),
        "t_idx": int(t_idx),
        "sim_n_runs": int(sim_n_runs),
        "weighted_ess": float(target_ess),
        "local_selector_points": int(selector_points),
        "local_selector_args": _clone_plain_object(selector_args_local),
        "reports": reports,
    }


def _run_all_locations_worker(
    *,
    artifact_path: str | Path,
    generator_kind: str,
    i: int,
    j: int,
    t_idx: int,
    sim_n_runs: int,
    output_path: str | Path,
) -> None:
    cmd = [
        sys.executable,
        str(SIM_ALL_LOCATIONS_WORKER_SCRIPT_PATH),
        "--artifact-path",
        str(Path(artifact_path).resolve()),
        "--generator",
        str(generator_kind),
        "--i",
        str(int(i)),
        "--j",
        str(int(j)),
        "--t-idx",
        str(int(t_idx)),
        "--sim-n-runs",
        str(int(sim_n_runs)),
        "--output",
        str(Path(output_path).resolve()),
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu"),
            "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK", "TRUE"),
            "MPLBACKEND": os.environ.get("MPLBACKEND", "Agg"),
        },
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip() or (completed.stdout or "").strip() or f"return code {completed.returncode}"
        raise RuntimeError(f"All-locations worker failed for ({i},{j}): {detail}")


def run_all_locations_sim(
    *,
    artifact_path: str | Path | None = None,
    generator_kind: str = "blob",
    n_bootstrap: int = 1,
    sim_n_runs: int = 100,
    t_idx: int | None = None,
    output_dir: str | Path | None = None,
    force_retrain: bool = False,
    force_rerun: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    generator_kind = str(generator_kind).strip().lower()
    sim_n_runs = int(sim_n_runs)
    if sim_n_runs < 1:
        raise ValueError(f"`sim_n_runs` must be >= 1, got {sim_n_runs}")

    if artifact_path is None:
        artifact_obj, resolved_artifact_path, _, _, _ = _load_or_train_demo_artifact(
            generator_kind=generator_kind,
            n_bootstrap=n_bootstrap,
            force_retrain=force_retrain,
        )
    else:
        resolved_artifact_path = Path(artifact_path)
        artifact_obj = load_weight_artifact(resolved_artifact_path)

    trained, _, meta, data, resolved_n_bootstrap = _resolve_artifact_runtime_state(artifact_obj)
    resolved_t_idx = min(30, data.shape[0] - 1) if t_idx is None else int(t_idx)
    if not (0 <= resolved_t_idx < data.shape[0]):
        raise ValueError(f"`t_idx` must be in [0, {data.shape[0] - 1}], got {resolved_t_idx}")

    result_dir = _default_all_locations_sim_result_dir(
        generator_kind=generator_kind,
        n_bootstrap=resolved_n_bootstrap,
        t_idx=resolved_t_idx,
        sim_n_runs=sim_n_runs,
        artifact_path=resolved_artifact_path,
        output_dir=output_dir,
    )
    raw_dir = result_dir / "raw"
    tables_dir = result_dir / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    total_locations = int(data.shape[1] * data.shape[2])
    metadata = {
        "generator": str(generator_kind),
        "artifact_path": str(Path(resolved_artifact_path)),
        "artifact_type": "bootstrap" if isinstance(artifact_obj, BootstrapTrainingResult) else "single",
        "n_bootstrap": int(resolved_n_bootstrap),
        "t_idx": int(resolved_t_idx),
        "sim_n_runs": int(sim_n_runs),
        "grid_shape": [int(data.shape[1]), int(data.shape[2])],
        "data_shape": [int(v) for v in data.shape],
        "return_periods": [10, 100],
        "result_dir": str(result_dir),
        "raw_dir": str(raw_dir),
        "tables_dir": str(tables_dir),
        "force_rerun": bool(force_rerun),
    }
    metadata_path = _save_json_file(result_dir / "metadata.json", metadata)

    raw_paths: list[Path] = []
    pending: list[tuple[int, int, int, Path]] = []
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            location_index = int(i * data.shape[2] + j)
            raw_path = _raw_location_output_path(
                raw_dir,
                location_index=location_index,
                i=i,
                j=j,
                total_locations=total_locations,
            )
            raw_paths.append(raw_path)
            if force_rerun or (not raw_path.exists()) or (not _location_payload_is_complete(raw_path)):
                pending.append((location_index, i, j, raw_path))

    rebuilt_paths = _rebuild_all_location_result_tables(result_dir)
    if verbose:
        initial_completed = total_locations - len(pending)
        print(
            f"{_format_progress_bar(initial_completed, total_locations)} | "
            f"elapsed={_format_seconds(0.0)} | eta={_format_seconds(None)} | out={result_dir}"
        )

    start_time = time.perf_counter()
    completed_runtime = 0
    for location_index, i, j, raw_path in pending:
        _run_all_locations_worker(
            artifact_path=resolved_artifact_path,
            generator_kind=generator_kind,
            i=i,
            j=j,
            t_idx=resolved_t_idx,
            sim_n_runs=sim_n_runs,
            output_path=raw_path,
        )
        payload = _load_json_file(raw_path)
        payload["raw_report_path"] = str(raw_path)
        _save_json_file(raw_path, payload)
        rebuilt_paths = _rebuild_all_location_result_tables(result_dir)
        completed_runtime += 1
        if verbose:
            completed_total = total_locations - len(pending) + completed_runtime
            elapsed = time.perf_counter() - start_time
            avg_seconds = elapsed / completed_runtime if completed_runtime > 0 else None
            eta_seconds = None if avg_seconds is None else avg_seconds * (len(pending) - completed_runtime)
            print(
                f"{_format_progress_bar(completed_total, total_locations)} | "
                f"loc={location_index:03d} {_location_label(i, j)} | "
                f"elapsed={_format_seconds(elapsed)} | eta={_format_seconds(eta_seconds)} | out={result_dir}"
            )

    return {
        "result_dir": result_dir,
        "metadata_path": metadata_path,
        "summary_path": rebuilt_paths["summary"],
        "table_paths": {key: value for key, value in rebuilt_paths.items() if key != "summary"},
        "raw_dir": raw_dir,
        "artifact_path": Path(resolved_artifact_path),
        "completed_locations": total_locations,
    }


def _demo(
    *,
    run_sim: bool = False,
    show_plot: bool = False,
    n_bootstrap: int = 1,
    force_retrain: bool = False,
    generator_kind: str = "blob",
    ref_i: int = 5,
    ref_j: int = 6,
    sim_n_runs: int | None = None,
) -> None:
    generator_kind = str(generator_kind).strip().lower()
    artifact_obj, artifact_path, gen_kwargs, generate_fn, _ = _load_or_train_demo_artifact(
        generator_kind=generator_kind,
        n_bootstrap=n_bootstrap,
        force_retrain=force_retrain,
    )
    trained, adapter, meta, data, _ = _resolve_artifact_runtime_state(artifact_obj)

    i_ref, j_ref = _validate_index(ref_i, ref_j, data.shape[1], data.shape[2])
    t_ref = min(30, data.shape[0] - 1)
    weight_export_dir, weight_vmin, weight_vmax = _save_all_weight_maps(adapter, data.shape, t_idx=t_ref)
    print(f"saved weight maps for all spatial locations to {weight_export_dir}")
    if show_plot:
        bundle_path = _save_demo_plot_bundle(
            data,
            meta,
            adapter,
            i=i_ref,
            j=j_ref,
            t_idx=t_ref,
            field_name="s_field",
            weight_vmin=weight_vmin,
            weight_vmax=weight_vmax,
        )
        _launch_demo_plot_viewer(bundle_path)
    else:
        plt, _ = _import_plotting_modules()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_generated_data_map(data, meta, i_ref, j_ref, t_ref, field_name="s_field", ax=axes[0])
        plot_weight_map(
            adapter,
            data.shape,
            i_ref,
            j_ref,
            t_ref,
            ax=axes[1],
            weight_vmin=weight_vmin,
            weight_vmax=weight_vmax,
        )
        fig.suptitle(f"Demo views for reference site (i, j)=({i_ref}, {j_ref}) at t={t_ref}")

    if run_sim:
        resolved_sim_n_runs = int(DEMO_DEFAULTS["sim_n_runs"] if sim_n_runs is None else sim_n_runs)
        if resolved_sim_n_runs < 1:
            raise ValueError(f"`DEMO_DEFAULTS['sim_n_runs']` must be >= 1, got {sim_n_runs}")

        weighted_report = compute_combined_sim(
            generate_fn=generate_fn,
            gen_kwargs=gen_kwargs,
            i=i_ref,
            j=j_ref,
            n_runs=resolved_sim_n_runs,
            fit_mode="full",
            t_idx=t_ref,
            return_periods=[10, 100],
            weights=adapter,
            verbose=False,
        )
        weight_cube = np.asarray(adapter(i=i_ref, j=j_ref, t_idx=t_ref, data_shape=data.shape), dtype=np.float64)
        if weight_cube.shape != tuple(data.shape):
            raise ValueError(f"Adapter returned shape {weight_cube.shape}, expected {tuple(data.shape)}")
        target_ess = _ess_from_weight_map(weight_cube[t_ref])
        selector_args_cmp, selector_points = _selector_args_for_comparable_ess(
            target_ess,
            n_lat=data.shape[1],
            n_lon=data.shape[2],
        )
        selector_report = compute_combined_sim(
            generate_fn=generate_fn,
            gen_kwargs=gen_kwargs,
            i=i_ref,
            j=j_ref,
            n_runs=resolved_sim_n_runs,
            fit_mode="full",
            t_idx=t_ref,
            return_periods=[10, 100],
            selector_args=selector_args_cmp,
            weights=None,
            verbose=False,
        )
        pointwise_report = compute_combined_sim(
            generate_fn=generate_fn,
            gen_kwargs=gen_kwargs,
            i=i_ref,
            j=j_ref,
            n_runs=resolved_sim_n_runs,
            fit_mode="pointwise",
            t_idx=t_ref,
            return_periods=[10, 100],
            weights=None,
            verbose=False,
        )
        full_unweighted_report = compute_combined_sim(
            generate_fn=generate_fn,
            gen_kwargs=gen_kwargs,
            i=i_ref,
            j=j_ref,
            n_runs=resolved_sim_n_runs,
            fit_mode="full",
            t_idx=t_ref,
            return_periods=[10, 100],
            selector_args=None,
            weights=None,
            verbose=False,
        )
        sim_reports = {
            "pointwise": pointwise_report,
            "weighted": weighted_report,
            "local": selector_report,
            "full": full_unweighted_report,
        }
        sim_result_dir = _save_demo_sim_reports(
            generator_kind=generator_kind,
            n_bootstrap=n_bootstrap,
            i_ref=i_ref,
            j_ref=j_ref,
            t_ref=t_ref,
            sim_n_runs=resolved_sim_n_runs,
            artifact_path=artifact_path,
            weight_export_dir=weight_export_dir,
            target_ess=target_ess,
            selector_points=selector_points,
            selector_args_local=selector_args_cmp,
            reports=sim_reports,
        )
        print(
            f"SIM comparison for ({i_ref},{j_ref}) at t={t_ref} using generator='{generator_kind}' "
            f"with {resolved_sim_n_runs} Monte Carlo runs per report."
        )
        print(
            f"Comparable-ESS selector baseline: weighted ESS={target_ess:.3f}, "
            f"selector points={selector_points}"
        )
        print(f"saved SIM comparison results to {sim_result_dir}")
        print("[pointwise]")
        _print_sim_report(
            pointwise_report,
            fit_mode="pointwise",
            weighted=False,
            use_spatial_covariates=False,
        )
        print("[weighted]")
        _print_sim_report(
            weighted_report,
            fit_mode="full",
            weighted=True,
            use_spatial_covariates=False,
        )
        print("[local]")
        _print_sim_report(
            selector_report,
            fit_mode="full",
            weighted=False,
            use_spatial_covariates=False,
        )
        print("[full]")
        _print_sim_report(
            full_unweighted_report,
            fit_mode="full",
            weighted=False,
            use_spatial_covariates=False,
        )
    if not show_plot:
        plt, _ = _import_plotting_modules()
        plt.close("all")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SIM neural-weight utilities. Running the script trains the demo model; SIM is optional."
    )
    parser.set_defaults(
        with_sim=bool(DEMO_DEFAULTS["run_sim"]),
        with_sim_all_locations=False,
        show_plot=bool(DEMO_DEFAULTS["show_plot"]),
        force_retrain=bool(DEMO_DEFAULTS["force_retrain"]),
        force_rerun_sim_all_locations=False,
    )
    parser.add_argument(
        "--with-sim",
        action="store_true",
        help="Also run the single-reference four-mode SIM comparison after the demo training run.",
    )
    parser.add_argument(
        "--no-sim",
        dest="with_sim",
        action="store_false",
        help="Do not run the single-reference SIM comparison after the demo training run.",
    )
    parser.add_argument(
        "--with-sim-all-locations",
        dest="with_sim_all_locations",
        action="store_true",
        help="Run the four SIM comparison modes for every spatial location and save batch RMSE tables.",
    )
    parser.add_argument(
        "--show-plot",
        dest="show_plot",
        action="store_true",
        help="Display the demo generated-data and weight-map figures interactively.",
    )
    parser.add_argument(
        "--no-show-plot",
        dest="show_plot",
        action="store_false",
        help="Skip displaying the interactive demo plot window.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=int(DEMO_DEFAULTS["n_bootstrap"]),
        help="Run the demo trainer this many bootstrap splits. The right plot uses the averaged bootstrap adapter when > 1.",
    )
    parser.add_argument(
        "--sim-n-runs",
        type=int,
        default=int(DEMO_DEFAULTS["sim_n_runs"]),
        help="Monte Carlo run count to use for SIM reports.",
    )
    parser.add_argument(
        "--generator",
        choices=("blob", "linear"),
        default=str(DEMO_DEFAULTS["generator"]),
        help="Choose which generated demo dataset to use for training and plotting.",
    )
    parser.add_argument(
        "--force-retrain",
        dest="force_retrain",
        action="store_true",
        help="Ignore any saved weight artifact and retrain the demo model from scratch.",
    )
    parser.add_argument(
        "--use-saved-artifact",
        dest="force_retrain",
        action="store_false",
        help="Reuse a saved weight artifact when available.",
    )
    parser.add_argument(
        "--force-rerun-sim-all-locations",
        dest="force_rerun_sim_all_locations",
        action="store_true",
        help="Recompute all all-locations SIM raw outputs even if they already exist on disk.",
    )
    parser.add_argument(
        "--ref-i",
        type=int,
        default=int(DEMO_DEFAULTS["ref_i"]),
        help="Reference-grid row index used for the demo plots and SIM comparison outputs.",
    )
    parser.add_argument(
        "--ref-j",
        type=int,
        default=int(DEMO_DEFAULTS["ref_j"]),
        help="Reference-grid column index used for the demo plots and SIM comparison outputs.",
    )
    args = parser.parse_args(argv)

    if args.with_sim_all_locations:
        result = run_all_locations_sim(
            artifact_path=None,
            generator_kind=args.generator,
            n_bootstrap=args.n_bootstrap,
            sim_n_runs=args.sim_n_runs,
            force_retrain=args.force_retrain,
            force_rerun=args.force_rerun_sim_all_locations,
            verbose=True,
        )
        print(f"saved all-locations SIM results to {result['result_dir']}")
        print(f"summary CSV: {result['summary_path']}")
        return 0

    _demo(
        run_sim=args.with_sim,
        show_plot=args.show_plot,
        n_bootstrap=args.n_bootstrap,
        force_retrain=args.force_retrain,
        generator_kind=args.generator,
        ref_i=args.ref_i,
        ref_j=args.ref_j,
        sim_n_runs=args.sim_n_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
