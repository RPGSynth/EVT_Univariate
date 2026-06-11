from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_bundle(bundle_path: str | Path) -> dict[str, np.ndarray | str | int]:
    bundle_path = Path(bundle_path)
    with np.load(bundle_path) as bundle:
        return {
            "weight_maps": np.asarray(bundle["weight_maps"], dtype=np.float64),
            "t_idx": int(np.asarray(bundle["t_idx"]).item()),
            "title_prefix": str(np.asarray(bundle["title_prefix"]).item()) if "title_prefix" in bundle else "Weights",
            "weight_vmin": float(np.asarray(bundle["weight_vmin"]).item()) if "weight_vmin" in bundle else np.nan,
            "weight_vmax": float(np.asarray(bundle["weight_vmax"]).item()) if "weight_vmax" in bundle else np.nan,
        }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Save all spatial weight maps from a precomputed bundle.")
    parser.add_argument("--bundle", required=True, help="Path to the .npz bundle produced by sim2.py")
    parser.add_argument("--output-dir", required=True, help="Directory where the PNGs will be written")
    args = parser.parse_args(argv)

    bundle = _load_bundle(args.bundle)
    weight_maps = np.asarray(bundle["weight_maps"], dtype=np.float64)
    t_idx = int(bundle["t_idx"])
    title_prefix = str(bundle["title_prefix"])
    weight_vmin_raw = float(bundle["weight_vmin"]) if "weight_vmin" in bundle else float("nan")
    weight_vmax_raw = float(bundle["weight_vmax"]) if "weight_vmax" in bundle else float("nan")
    weight_vmin = None if not np.isfinite(weight_vmin_raw) else weight_vmin_raw
    weight_vmax = None if not np.isfinite(weight_vmax_raw) else weight_vmax_raw
    if weight_maps.ndim != 4:
        raise ValueError(f"`weight_maps` must have shape (H_ref, W_ref, H, W), got {weight_maps.shape}")

    n_lat, n_lon, map_lat, map_lon = weight_maps.shape
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_lat):
        for j in range(n_lon):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(weight_maps[i, j], origin="lower", cmap="viridis", vmin=weight_vmin, vmax=weight_vmax)
            ax.scatter([j], [i], c="red", marker="x", s=140, linewidths=2, label="Reference point")
            ax.set_title(f"{title_prefix} at (i,j)=({i},{j}), t={t_idx}")
            ax.set_xlabel("j")
            ax.set_ylabel("i")
            ax.set_xlim(-0.5, map_lon - 0.5)
            ax.set_ylim(-0.5, map_lat - 0.5)
            ax.legend(frameon=False, loc="upper right")
            fig.colorbar(im, ax=ax, label="Weight")
            fig.savefig(output_dir / f"weight_map_i{i:02d}_j{j:02d}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
