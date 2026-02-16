from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class SpatialSelection:
    """Container for selected spatial points around a reference (i, j)."""

    coords: np.ndarray
    flat_idx: np.ndarray
    mask: np.ndarray
    target_flat: int
    target_local_idx: int


Mode = Literal["square", "circle"]


def _validate_grid(n_lat: int, n_lon: int, i: int, j: int) -> tuple[int, int, int, int]:
    n_lat = int(n_lat)
    n_lon = int(n_lon)
    i = int(i)
    j = int(j)

    if n_lat <= 0 or n_lon <= 0:
        raise ValueError(f"Grid dimensions must be positive, got n_lat={n_lat}, n_lon={n_lon}")
    if not (0 <= i < n_lat and 0 <= j < n_lon):
        raise ValueError(f"Reference point (i={i}, j={j}) is outside grid ({n_lat}, {n_lon})")

    return n_lat, n_lon, i, j


def _coords_to_flat(coords: np.ndarray, n_lon: int) -> np.ndarray:
    return coords[:, 0] * n_lon + coords[:, 1]


def _full_grid_coords(n_lat: int, n_lon: int) -> np.ndarray:
    """Return all (i, j) coordinates on the grid in row-major order."""
    ii, jj = np.meshgrid(
        np.arange(n_lat, dtype=np.int64),
        np.arange(n_lon, dtype=np.int64),
        indexing="ij",
    )
    return np.column_stack([ii.ravel(), jj.ravel()])


def _sort_coords_by_distance(coords: np.ndarray, i: int, j: int) -> np.ndarray:
    """Sort coords by Euclidean distance to (i, j), tie-break by row/col."""
    if coords.shape[0] == 0:
        return coords
    di = coords[:, 0] - i
    dj = coords[:, 1] - j
    d2 = di * di + dj * dj
    order = np.lexsort((coords[:, 1], coords[:, 0], d2))
    return coords[order]


def _apply_sort_and_limit(
    coords: np.ndarray,
    *,
    i: int,
    j: int,
    sort_by_distance: bool,
    max_points: int | None,
) -> np.ndarray:
    """
    Apply optional distance sorting and direct point-count truncation.
    If max_points is provided, nearest-first ordering is always used.
    """
    if max_points is not None:
        max_points = int(max_points)
        if max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {max_points}")

    must_sort = sort_by_distance or (max_points is not None)
    if must_sort:
        coords = _sort_coords_by_distance(coords, i=i, j=j)

    if max_points is not None and coords.shape[0] > max_points:
        coords = coords[:max_points]

    return coords


def _finalize_selection(
    coords: np.ndarray,
    *,
    n_lat: int,
    n_lon: int,
    i: int,
    j: int,
    include_center: bool,
) -> SpatialSelection:
    target_flat = i * n_lon + j

    if coords.size == 0:
        coords = np.empty((0, 2), dtype=np.int64)
    else:
        coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)

    if not include_center and coords.shape[0] == 0:
        # Valid output: empty selection around the target.
        flat_idx = np.empty((0,), dtype=np.int64)
        mask = np.zeros((n_lat, n_lon), dtype=bool)
        return SpatialSelection(
            coords=coords,
            flat_idx=flat_idx,
            mask=mask,
            target_flat=int(target_flat),
            target_local_idx=-1,
        )

    flat_idx = _coords_to_flat(coords, n_lon).astype(np.int64)

    mask = np.zeros((n_lat, n_lon), dtype=bool)
    if coords.shape[0] > 0:
        mask[coords[:, 0], coords[:, 1]] = True

    where_target = np.where(flat_idx == target_flat)[0]
    target_local_idx = int(where_target[0]) if where_target.size else -1

    return SpatialSelection(
        coords=coords,
        flat_idx=flat_idx,
        mask=mask,
        target_flat=int(target_flat),
        target_local_idx=target_local_idx,
    )


def select_square(
    n_lat: int,
    n_lon: int,
    i: int,
    j: int,
    *,
    radius: int,
    include_center: bool = True,
    sort_by_distance: bool = False,
    max_points: int | None = None,
) -> SpatialSelection:
    """
    Select points in a square neighborhood around (i, j).

    Parameters
    ----------
    radius : int
        Half-width of square window. radius=1 gives a 3x3 neighborhood.
    include_center : bool
        If False, remove (i, j) from selection.
    sort_by_distance : bool
        If True, sort selected points by Euclidean distance to (i, j).
        Tie-break is row-major (i then j).
    max_points : int or None
        If provided, keep at most this many selected points (nearest-first).
    """
    n_lat, n_lon, i, j = _validate_grid(n_lat, n_lon, i, j)
    radius = int(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    i0 = max(0, i - radius)
    i1 = min(n_lat - 1, i + radius)
    j0 = max(0, j - radius)
    j1 = min(n_lon - 1, j + radius)

    ii, jj = np.meshgrid(
        np.arange(i0, i1 + 1, dtype=np.int64),
        np.arange(j0, j1 + 1, dtype=np.int64),
        indexing="ij",
    )
    coords = np.column_stack([ii.ravel(), jj.ravel()])

    if not include_center:
        keep = ~((coords[:, 0] == i) & (coords[:, 1] == j))
        coords = coords[keep]

    coords = _apply_sort_and_limit(
        coords,
        i=i,
        j=j,
        sort_by_distance=sort_by_distance,
        max_points=max_points,
    )

    return _finalize_selection(
        coords,
        n_lat=n_lat,
        n_lon=n_lon,
        i=i,
        j=j,
        include_center=include_center,
    )


def select_circle(
    n_lat: int,
    n_lon: int,
    i: int,
    j: int,
    *,
    radius: float,
    include_center: bool = True,
    sort_by_distance: bool = True,
    max_points: int | None = None,
) -> SpatialSelection:
    """
    Select points in a circular neighborhood around (i, j).

    Parameters
    ----------
    radius : float
        Euclidean radius in grid units.
    include_center : bool
        If False, remove (i, j) from selection.
    sort_by_distance : bool
        If True, sort selected points by Euclidean distance to (i, j).
        Tie-break is row-major (i then j).
    max_points : int or None
        If provided, keep at most this many selected points (nearest-first).
    """
    n_lat, n_lon, i, j = _validate_grid(n_lat, n_lon, i, j)
    radius = float(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")

    r_int = int(np.ceil(radius))
    i0 = max(0, i - r_int)
    i1 = min(n_lat - 1, i + r_int)
    j0 = max(0, j - r_int)
    j1 = min(n_lon - 1, j + r_int)

    ii, jj = np.meshgrid(
        np.arange(i0, i1 + 1, dtype=np.int64),
        np.arange(j0, j1 + 1, dtype=np.int64),
        indexing="ij",
    )

    di = ii - i
    dj = jj - j
    d2 = di * di + dj * dj
    keep = d2 <= (radius * radius + 1e-12)

    coords = np.column_stack([ii[keep], jj[keep]])

    if not include_center:
        keep_center = ~((coords[:, 0] == i) & (coords[:, 1] == j))
        coords = coords[keep_center]

    coords = _apply_sort_and_limit(
        coords,
        i=i,
        j=j,
        sort_by_distance=sort_by_distance,
        max_points=max_points,
    )

    return _finalize_selection(
        coords,
        n_lat=n_lat,
        n_lon=n_lon,
        i=i,
        j=j,
        include_center=include_center,
    )


def select_spatial_neighborhood(
    n_lat: int,
    n_lon: int,
    i: int,
    j: int,
    *,
    mode: Mode,
    radius: float | None,
    include_center: bool = True,
    sort_by_distance: bool | None = None,
    max_points: int | None = None,
) -> SpatialSelection:
    """
    Generic dispatcher for spatial neighborhood selection.

    mode='square' uses integer radius (half-window size).
    mode='circle' uses floating Euclidean radius.
    If radius is None, selection falls back to the full grid.
    """
    mode = str(mode).lower()
    n_lat, n_lon, i, j = _validate_grid(n_lat, n_lon, i, j)

    if mode not in ("square", "circle"):
        raise ValueError(f"Unknown mode '{mode}'. Expected 'square' or 'circle'.")

    if sort_by_distance is None:
        sort_by_distance = (mode == "circle")

    # Fallback strategy: no radius -> full grid, then optional sort/truncation.
    if radius is None:
        coords = _full_grid_coords(n_lat, n_lon)
        if not include_center:
            keep = ~((coords[:, 0] == i) & (coords[:, 1] == j))
            coords = coords[keep]

        coords = _apply_sort_and_limit(
            coords,
            i=i,
            j=j,
            sort_by_distance=bool(sort_by_distance),
            max_points=max_points,
        )
        return _finalize_selection(
            coords,
            n_lat=n_lat,
            n_lon=n_lon,
            i=i,
            j=j,
            include_center=include_center,
        )

    if mode == "square":
        return select_square(
            n_lat=n_lat,
            n_lon=n_lon,
            i=i,
            j=j,
            radius=int(radius),
            include_center=include_center,
            sort_by_distance=bool(sort_by_distance),
            max_points=max_points,
        )

    if mode == "circle":
        return select_circle(
            n_lat=n_lat,
            n_lon=n_lon,
            i=i,
            j=j,
            radius=float(radius),
            include_center=include_center,
            sort_by_distance=bool(sort_by_distance),
            max_points=max_points,
        )
