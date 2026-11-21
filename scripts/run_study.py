"""Preserve synthetic correspondences, model fits, errors, RANSAC traces and timings."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from projectivelab.core import (
    GeometryError,
    normalized_dlt,
    project,
    ransac,
    transfer_errors,
)
from projectivelab.fixtures import SEED, cases


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rmse(errors):
    return float(np.sqrt(np.mean(np.square(errors))))


def run(output, repetitions):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Choose a new empty directory; preserved runs are not overwritten"
        )
    if type(repetitions) is not int or not 1 <= repetitions <= 30:
        raise ValueError("Repetitions must lie in 1..30")
    for folder in ["inputs", "models", "errors", "traces"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    sources = [
        "src/projectivelab/core.py",
        "src/projectivelab/fixtures.py",
        "scripts/run_study.py",
    ]
    hashes = {
        path: hashlib.sha256(
            (ROOT / path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        ).hexdigest()
        for path in sources
    }
    timings = []
    summaries = []
    order_rng = random.Random(1307)
    for case in cases():
        name = case["name"]
        write_json(output / "inputs" / f"{name}.json", case)
        source = np.asarray(case["source"])
        target = np.asarray(case["target"])
        clean = np.asarray(case["clean_target"])
        truth = np.asarray(case["true_inlier"], dtype=bool)
        write_csv(
            output / "inputs" / f"{name}.csv",
            [
                {
                    "index": i,
                    "source_x": s[0],
                    "source_y": s[1],
                    "observed_target_x": t[0],
                    "observed_target_y": t[1],
                    "clean_target_x": c[0],
                    "clean_target_y": c[1],
                    "synthetic_true_inlier": int(inlier),
                }
                for i, (s, t, c, inlier) in enumerate(zip(source, target, clean, truth))
            ],
        )
        operations = {
            "all_point_dlt": lambda: normalized_dlt(source, target),
            "ransac": lambda: ransac(
                source,
                target,
                threshold=2.5,
                max_trials=1000,
                min_trials=20,
                confidence=0.995,
                seed=17,
            ),
        }
        # The first exact call is an untimed warmup and supplies the preserved model.
        results = {method: operation() for method, operation in operations.items()}
        for repeat in range(repetitions):
            methods = list(operations)
            order_rng.shuffle(methods)
            for order, method in enumerate(methods):
                begin = time.perf_counter_ns()
                result = operations[method]()
                elapsed = time.perf_counter_ns() - begin
                np.testing.assert_array_equal(result.matrix, results[method].matrix)
                if method == "ransac":
                    assert (
                        result.inliers == results[method].inliers
                        and result.trace == results[method].trace
                    )
                timings.append(
                    {
                        "case": name,
                        "method": method,
                        "repeat": repeat,
                        "order": order,
                        "elapsed_ns": elapsed,
                    }
                )
        for method, result in results.items():
            matrix = result.matrix
            if matrix is None:
                raise RuntimeError("A planned study case did not find a model")
            observed_error = transfer_errors(matrix, source, target)
            clean_error = transfer_errors(matrix, source, clean)
            held_error = transfer_errors(
                matrix, case["held_source"], case["held_target"]
            )
            # These finite fixtures are deliberately away from the projective horizon.
            if not np.isfinite(held_error).all():
                raise RuntimeError("Unexpected invalid held-out projection")
            inferred = observed_error <= 2.5
            true_positive = int(np.sum(inferred & truth))
            false_positive = int(np.sum(inferred & ~truth))
            if method == "ransac":
                model = {
                    "matrix": matrix.tolist(),
                    "inliers": list(result.inliers),
                    "status": result.status,
                    "trials": result.trials,
                    "skipped_degenerate": result.skipped_degenerate,
                    "adaptive_limit": result.adaptive_limit,
                    "threshold_target_px": 2.5,
                    "confidence_assumption": 0.995,
                    "seed": 17,
                    "max_trials": 1000,
                    "min_trials": 20,
                    "local_refits": 3,
                }
                write_json(
                    output / "traces" / f"{name}.json",
                    {"iterations": result.trace, "improvements": result.improvements},
                )
                write_csv(
                    output / "traces" / f"{name}.csv",
                    [
                        {
                            "trial": event["trial"],
                            "sample": " ".join(str(i) for i in event["sample"]),
                            "candidate_inliers": event["candidate_inliers"],
                            "candidate_clipped_squared_error": event[
                                "candidate_clipped_squared_error"
                            ],
                            "skipped": int(event["skipped"]),
                            "improved": int(event["improved"]),
                            "best_inliers": event["best_inliers"],
                            "adaptive_limit": event["adaptive_limit"],
                        }
                        for event in result.trace
                    ],
                )
            else:
                model = {
                    "matrix": matrix.tolist(),
                    "design_rank": result.design_rank,
                    "singular_values": result.singular_values,
                    "constraint_condition": result.constraint_condition,
                    "used_full_basis": result.used_full_basis,
                    "threshold_mask_for_evaluation_only": inferred.tolist(),
                }
            write_json(output / "models" / f"{name}__{method}.json", model)
            write_csv(
                output / "errors" / f"{name}__{method}__train.csv",
                [
                    {
                        "index": i,
                        "observed_transfer_error_px": float(o),
                        "clean_transfer_error_px": float(c),
                        "synthetic_true_inlier": int(t),
                        "threshold_inlier": int(v),
                    }
                    for i, (o, c, t, v) in enumerate(
                        zip(observed_error, clean_error, truth, inferred)
                    )
                ],
            )
            mapped = project(matrix, case["held_source"])
            write_csv(
                output / "errors" / f"{name}__{method}__held.csv",
                [
                    {
                        "index": i,
                        "source_x": s[0],
                        "source_y": s[1],
                        "true_target_x": t[0],
                        "true_target_y": t[1],
                        "predicted_x": p[0],
                        "predicted_y": p[1],
                        "transfer_error_px": float(error),
                    }
                    for i, (s, t, p, error) in enumerate(
                        zip(
                            case["held_source"],
                            case["held_target"],
                            mapped.points,
                            held_error,
                        )
                    )
                ],
            )
            durations = [
                row["elapsed_ns"]
                for row in timings
                if row["case"] == name and row["method"] == method
            ]
            summaries.append(
                {
                    "case": name,
                    "method": method,
                    "train_points": len(source),
                    "held_points": len(held_error),
                    "outlier_fraction": case["outlier_fraction"],
                    "noise_sigma_target_px": case["noise_sigma_target_px"],
                    "threshold_inliers": int(inferred.sum()),
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "precision": (
                        true_positive / int(inferred.sum()) if inferred.any() else ""
                    ),
                    "recall": true_positive / int(truth.sum()),
                    "train_observed_true_inlier_rmse_px": rmse(observed_error[truth]),
                    "train_clean_rmse_px": rmse(clean_error),
                    "held_clean_rmse_px": rmse(held_error),
                    "held_clean_median_px": float(np.median(held_error)),
                    "held_clean_max_px": float(np.max(held_error)),
                    "trials": result.trials if method == "ransac" else 0,
                    "skipped_degenerate": (
                        result.skipped_degenerate if method == "ransac" else 0
                    ),
                    "median_ns": statistics.median(durations),
                    "min_ns": min(durations),
                    "max_ns": max(durations),
                }
            )
    degeneracies = []
    for name, points in [
        ("coincident", np.zeros((4, 2))),
        ("collinear", np.array([[0, 0], [1, 0], [2, 0], [3, 0]])),
        ("three_collinear_minimal", np.array([[0, 0], [1, 0], [2, 0], [0, 1]])),
    ]:
        try:
            normalized_dlt(points, points)
        except GeometryError as exc:
            degeneracies.append(
                {
                    "case": name,
                    "source": points.tolist(),
                    "target": points.tolist(),
                    "status": "rejected",
                    "reason": str(exc),
                }
            )
        else:
            raise AssertionError("A degenerate fixture was accepted")
    write_json(output / "degeneracies.json", degeneracies)
    write_csv(output / "measurements.csv", timings)
    write_csv(output / "summary.csv", summaries)
    write_json(
        output / "environment.json",
        {
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
            "architecture": platform.machine(),
            "repetitions": repetitions,
            "timed_calls": len(timings),
            "fixture_seed": SEED,
            "ransac_seed": 17,
            "order_seed": 1307,
            "source_sha256_lf": hashes,
            "source_hash_policy": "CRLF and CR normalized to LF before SHA-256; captured before measurements",
            "precision": "float64 coordinates and NumPy SVD; algorithm and RANSAC implemented in this repository",
            "timing_scope": "Complete in-memory estimation calls, including RANSAC sampling/refinement/trace; no input generation, image warp, plotting or disk I/O",
            "provenance": "All correspondences, corruption labels, held-out points and images are procedural synthetic planar data",
        },
    )
    manifest = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": 4,
                "models": 8,
                "held_points_per_case": 165,
                "timed_calls": len(timings),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/latest")
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    run(args.out, args.repetitions)
