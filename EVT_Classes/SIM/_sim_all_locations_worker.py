from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from EVT_Classes.SIM.sim2 import _compute_single_location_sim_payload, _save_json_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the all-locations SIM comparison for one location.")
    parser.add_argument("--artifact-path", required=True, help="Path to the saved weight artifact.")
    parser.add_argument("--generator", choices=("blob", "linear"), required=True, help="Demo generator kind.")
    parser.add_argument("--i", type=int, required=True, help="Reference row index.")
    parser.add_argument("--j", type=int, required=True, help="Reference column index.")
    parser.add_argument("--t-idx", type=int, required=True, help="Reference time index.")
    parser.add_argument("--sim-n-runs", type=int, required=True, help="Monte Carlo run count per report.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args(argv)

    payload = _compute_single_location_sim_payload(
        artifact_path=Path(args.artifact_path),
        generator_kind=str(args.generator),
        i=int(args.i),
        j=int(args.j),
        t_idx=int(args.t_idx),
        sim_n_runs=int(args.sim_n_runs),
    )
    payload["raw_report_path"] = str(Path(args.output).resolve())
    _save_json_file(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
