from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from EVT_Classes.SIM.generator import generate_gev_dataset_blobs


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a blob-based GEV dataset and save it to an .npz bundle.")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--kwargs-json", required=True, help="JSON-encoded kwargs for generate_gev_dataset_blobs")
    args = parser.parse_args(argv)

    gen_kwargs = json.loads(args.kwargs_json)
    data, meta = generate_gev_dataset_blobs(**gen_kwargs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        data=np.asarray(data, dtype=np.float64),
        x=np.asarray(meta["x"], dtype=np.float64),
        y=np.asarray(meta["y"], dtype=np.float64),
        t=np.asarray(meta["t"], dtype=np.float64),
        s_field=np.asarray(meta["s_field"], dtype=np.float64),
        t_curve=np.asarray(meta["t_curve"], dtype=np.float64),
        mu=np.asarray(meta["mu"], dtype=np.float64),
        sigma=np.asarray(meta["sigma"], dtype=np.float64),
        xi=np.asarray(meta["xi"], dtype=np.float64),
        params_json=np.asarray(json.dumps(meta.get("params", {}))),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
