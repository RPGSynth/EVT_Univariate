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

from EVT_Classes.NN import (
    BootstrapAveragedWeightAdapterGPlus,
    BootstrapRunSummary,
    BootstrapTrainingResult,
    EnergyKernelJointGEV,
    FlexibleKernelWeightAdapterGPlus,
    KernelContext,
    KernelFeatureBank,
    TrainedWeightModel,
    TrainingConfig,
    build_gplus_adapter,
    build_teacher_marginal_fullfield,
    load_weight_artifact,
    prepare_kernel_context,
    save_weight_artifact,
    train_weight_model,
    train_weight_model_bootstrap,
)
from EVT_Classes.NN.core import (
    _clone_plain_object,
    _config_to_dict,
    _format_progress_bar,
    _resolve_artifact_runtime_state,
    _validate_data_3d,
    _validate_index,
)

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
    import GEVFit.gevPackage as gev

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
    payload = _compute_single_location_sim_payload(
        artifact_path=Path(artifact_path),
        generator_kind=str(generator_kind),
        i=int(i),
        j=int(j),
        t_idx=int(t_idx),
        sim_n_runs=int(sim_n_runs),
    )
    payload["raw_report_path"] = str(Path(output_path).resolve())
    _save_json_file(Path(output_path), payload)


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
