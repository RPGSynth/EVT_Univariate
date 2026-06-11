from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def _load_bundle(bundle_path: str | Path) -> dict[str, np.ndarray | int]:
    bundle_path = Path(bundle_path)
    with np.load(bundle_path) as bundle:
        return {
            "generated_map": np.asarray(bundle["generated_map"], dtype=np.float64),
            "weight_map": np.asarray(bundle["weight_map"], dtype=np.float64),
            "x": np.asarray(bundle["x"], dtype=np.float64),
            "y": np.asarray(bundle["y"], dtype=np.float64),
            "i": int(np.asarray(bundle["i"]).item()),
            "j": int(np.asarray(bundle["j"]).item()),
            "t_idx": int(np.asarray(bundle["t_idx"]).item()),
            "generated_title": str(np.asarray(bundle["generated_title"]).item()) if "generated_title" in bundle else "",
            "generated_label": str(np.asarray(bundle["generated_label"]).item()) if "generated_label" in bundle else "",
            "generated_field_name": str(np.asarray(bundle["generated_field_name"]).item()) if "generated_field_name" in bundle else "",
            "weight_title": str(np.asarray(bundle["weight_title"]).item()) if "weight_title" in bundle else "",
            "weight_vmin": float(np.asarray(bundle["weight_vmin"]).item()) if "weight_vmin" in bundle else np.nan,
            "weight_vmax": float(np.asarray(bundle["weight_vmax"]).item()) if "weight_vmax" in bundle else np.nan,
        }


def _plot_bundle(bundle: dict[str, np.ndarray | int]) -> plt.Figure:
    generated_map = np.asarray(bundle["generated_map"], dtype=np.float64)
    weight_map = np.asarray(bundle["weight_map"], dtype=np.float64)
    x = np.asarray(bundle["x"], dtype=np.float64)
    y = np.asarray(bundle["y"], dtype=np.float64)
    i = int(bundle["i"])
    j = int(bundle["j"])
    t_idx = int(bundle["t_idx"])
    generated_title = str(bundle["generated_title"]) if bundle["generated_title"] else f"Generated field at t={t_idx}"
    generated_label = str(bundle["generated_label"]) if bundle["generated_label"] else "Value"
    generated_field_name = str(bundle["generated_field_name"]) if bundle["generated_field_name"] else ""
    weight_title = str(bundle["weight_title"]) if bundle["weight_title"] else f"Weights at (i,j)=({i},{j}), t={t_idx}"
    weight_vmin_raw = float(bundle["weight_vmin"]) if "weight_vmin" in bundle else float("nan")
    weight_vmax_raw = float(bundle["weight_vmax"]) if "weight_vmax" in bundle else float("nan")
    weight_vmin = None if not np.isfinite(weight_vmin_raw) else weight_vmin_raw
    weight_vmax = None if not np.isfinite(weight_vmax_raw) else weight_vmax_raw
    if generated_field_name == "s_field":
        generated_title = "Spatial covariate field"
        generated_label = "Field intensity (0-1)"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    extent = None
    xlabel = "j"
    ylabel = "i"
    scatter_x = [j]
    scatter_y = [i]
    rectangle_patch: Rectangle | None = None
    if x.ndim == 1 and y.ndim == 1 and x.size == generated_map.shape[1] and y.size == generated_map.shape[0]:
        if generated_field_name == "s_field":
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
        scatter_x = [float(x[j])]
        scatter_y = [float(y[i])]

    finite = generated_map[np.isfinite(generated_map)]
    if generated_field_name == "s_field":
        cmap0 = "YlGnBu"
        vmin0 = None
        vmax0 = None
    else:
        cmap0 = "viridis"
        vmin0 = None
        vmax0 = float(np.quantile(finite, 0.98)) if finite.size else None
    if generated_field_name == "s_field":
        im0 = axes[0].imshow(
            generated_map,
            origin="lower",
            cmap="YlGnBu",
            extent=extent,
            aspect="auto",
        )
    else:
        im0 = axes[0].imshow(
            generated_map,
            origin="lower",
            cmap=cmap0,
            extent=extent,
            aspect="auto",
            interpolation="none",
            vmin=vmin0,
            vmax=vmax0,
        )
    if rectangle_patch is not None:
        axes[0].add_patch(rectangle_patch)
    else:
        axes[0].scatter(scatter_x, scatter_y, c="red", marker="x", s=140, linewidths=2, label="Reference point")
    axes[0].set_title(generated_title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    if rectangle_patch is None:
        axes[0].legend(frameon=False, loc="upper right")
    fig.colorbar(im0, ax=axes[0], label=generated_label)

    im1 = axes[1].imshow(weight_map, origin="lower", cmap="viridis", vmin=weight_vmin, vmax=weight_vmax)
    axes[1].scatter([j], [i], c="red", marker="x", s=140, linewidths=2, label="Reference point")
    axes[1].set_title(weight_title)
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("i")
    axes[1].legend(frameon=False, loc="upper right")
    fig.colorbar(im1, ax=axes[1], label="Weight")

    fig.suptitle(f"Demo views for reference site (i, j)=({i}, {j}) at t={t_idx}")
    return fig


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Display or save SIM demo plots from a precomputed bundle.")
    parser.add_argument("--bundle", required=True, help="Path to the .npz bundle produced by sim2.py")
    parser.add_argument("--output", help="Optional PNG output path")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args(argv)

    bundle = _load_bundle(args.bundle)
    fig = _plot_bundle(bundle)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
