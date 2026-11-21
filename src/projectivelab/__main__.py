"""Estimate a homography from JSON source/target correspondences."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .core import GeometryError, normalized_dlt, ransac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--method", choices=["dlt", "ransac"], default="ransac")
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.5,
        help="Forward-error threshold in target coordinate units",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-trials", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, help="Write JSON exclusively; otherwise print to stdout"
    )
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if args.method == "dlt":
            result = normalized_dlt(data["source"], data["target"])
            payload = {
                "status": "found",
                "matrix": result.matrix.tolist(),
                "design_rank": result.design_rank,
                "constraint_condition": result.constraint_condition,
            }
        else:
            result = ransac(
                data["source"],
                data["target"],
                threshold=args.threshold,
                seed=args.seed,
                max_trials=args.max_trials,
                min_trials=min(20, args.max_trials),
            )
            payload = {
                "status": result.status,
                "matrix": result.matrix.tolist() if result.matrix is not None else None,
                "inlier_indices": [
                    i for i, value in enumerate(result.inliers) if value
                ],
                "trials": result.trials,
                "skipped_degenerate": result.skipped_degenerate,
            }
        text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        else:
            print(text, end="")
        return 0 if payload["status"] == "found" else 1
    except (GeometryError, np.linalg.LinAlgError, OSError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
