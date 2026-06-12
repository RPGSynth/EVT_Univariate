from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


_THIS_FILE = Path(__file__).resolve()
_DEFAULT_RESULTS_ROOT = _THIS_FILE.parent / "_output" / "sim_results_all"
_DEFAULT_METRICS: list[tuple[str, str, str]] = [
    ("mu", "rmse_mu.csv", "mu"),
    ("sigma", "rmse_sigma.csv", "sigma"),
    ("xi", "rmse_xi.csv", "xi"),
    ("RL_10", "rmse_rl_10.csv", "RL 10"),
    ("RL_100", "rmse_rl_100.csv", "RL 100"),
]
_METHODS: list[tuple[str, str, str]] = [
    ("pointwise", "Pointwise", "#4c78a8"),
    ("weighted", "Weighted", "#f58518"),
    ("local", "Local", "#54a24b"),
    ("full", "Full", "#e45756"),
]


def _find_latest_result_dir(results_root: Path) -> Path:
    metadata_paths = sorted(results_root.rglob("metadata.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata.json found under {results_root}")
    return metadata_paths[0].parent


def _load_metric_table(result_dir: Path, filename: str) -> pd.DataFrame:
    table_path = result_dir / "tables" / filename
    if not table_path.exists():
        raise FileNotFoundError(f"Missing metric table: {table_path}")
    return pd.read_csv(table_path)


def _make_regular_grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot(index="i", columns="j", values=value_col).sort_index(axis=0).sort_index(axis=1)
    i_vals = pivot.index.to_numpy(dtype=int)
    j_vals = pivot.columns.to_numpy(dtype=int)
    x_min = float(np.nanmin(df["x"]))
    x_max = float(np.nanmax(df["x"]))
    y_min = float(np.nanmin(df["y"]))
    y_max = float(np.nanmax(df["y"]))
    extent = np.array([x_min, x_max, y_min, y_max], dtype=np.float64)
    return pivot.to_numpy(dtype=np.float64), extent, np.array([i_vals.size, j_vals.size], dtype=np.int64)


def _build_density_curve(values: np.ndarray, n_points: int = 256) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([0.0, 0.0], dtype=np.float64)
    if arr.size == 1 or np.allclose(arr, arr[0]):
        x = np.linspace(arr[0] - 0.5, arr[0] + 0.5, n_points)
        y = np.zeros_like(x)
        y[np.argmin(np.abs(x - arr[0]))] = 1.0
        return x, y

    x_min = float(arr.min())
    x_max = float(arr.max())
    pad = max((x_max - x_min) * 0.1, 1e-6)
    x = np.linspace(x_min - pad, x_max + pad, n_points)

    try:
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(arr)
        y = np.asarray(kde(x), dtype=np.float64)
        return x, y
    except Exception:
        hist, bin_edges = np.histogram(arr, bins=min(20, max(5, arr.size // 4)), density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        y = np.interp(x, centers, hist, left=0.0, right=0.0)
        return x, y


def _plot_weighted_minus_local_maps(result_dir: Path, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes_flat = axes.ravel()

    for axis, (metric_key, filename, title) in zip(axes_flat, _DEFAULT_METRICS):
        df = _load_metric_table(result_dir, filename)
        diff = np.asarray(df["weighted"], dtype=np.float64) - np.asarray(df["local"], dtype=np.float64)
        df_plot = df.copy()
        df_plot["diff"] = diff
        grid, extent, grid_shape = _make_regular_grid(df_plot, "diff")
        finite = grid[np.isfinite(grid)]
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmax = max(vmax, 1e-8)
        im = axis.imshow(
            grid,
            origin="lower",
            extent=extent,
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            aspect="auto",
        )
        axis.set_title(f"{title}: weighted - local")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.text(
            0.02,
            0.02,
            f"{int(grid_shape[0])}x{int(grid_shape[1])}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
        )
        fig.colorbar(im, ax=axis, shrink=0.82, label="RMSE diff")

    axes_flat[-1].axis("off")
    fig.suptitle(f"Weighted vs Local RMSE Difference Maps\n{result_dir.name}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_method_density_curves(result_dir: Path, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    axes_flat = axes.ravel()

    for axis, (metric_key, filename, title) in zip(axes_flat, _DEFAULT_METRICS):
        df = _load_metric_table(result_dir, filename)
        for column, label, color in _METHODS:
            x, y = _build_density_curve(np.asarray(df[column], dtype=np.float64))
            axis.plot(x, y, label=label, color=color, linewidth=2.0)
        axis.set_title(f"{title} RMSE density")
        axis.set_xlabel("RMSE")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.25, linewidth=0.6)

    legend_axis = axes_flat[-1]
    legend_axis.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    legend_axis.legend(handles, labels, loc="center", frameon=False, fontsize=12)
    legend_axis.set_title("Methods", fontsize=13)
    fig.suptitle(f"RMSE Density Curves by Method\n{result_dir.name}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_location_plots(result_dir: Path) -> dict[str, Path]:
    resolved_result_dir = Path(result_dir).resolve()
    plots_dir = resolved_result_dir / "plots"
    diff_map_path = _plot_weighted_minus_local_maps(
        resolved_result_dir,
        plots_dir / "weighted_minus_local_rmse_maps.png",
    )
    density_path = _plot_method_density_curves(
        resolved_result_dir,
        plots_dir / "rmse_density_curves.png",
    )
    return {
        "result_dir": resolved_result_dir,
        "plots_dir": plots_dir,
        "diff_map_path": diff_map_path,
        "density_path": density_path,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all-locations SIM RMSE summaries from saved CSV tables.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Specific all-locations SIM result directory. Defaults to the most recent result under _output/sim_results_all.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result_dir = _find_latest_result_dir(_DEFAULT_RESULTS_ROOT) if args.result_dir is None else Path(args.result_dir)
    outputs = generate_all_location_plots(result_dir)
    print(f"result dir: {outputs['result_dir']}")
    print(f"diff map: {outputs['diff_map_path']}")
    print(f"density curves: {outputs['density_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
