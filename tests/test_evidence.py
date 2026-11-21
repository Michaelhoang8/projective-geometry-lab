"""Reconcile source hashes, fitted matrices, errors, masks, traces and media."""

import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import unittest

import numpy as np

from projectivelab.core import normalized_dlt, project, ransac, transfer_errors

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


class EvidenceTests(unittest.TestCase):
    def test_complete_study_and_visual_artifacts_reconcile(self):
        data = Path(os.environ.get("PROJECTIVE_RESULTS", ROOT / "results/published"))
        environment = json.loads((data / "environment.json").read_text())
        for relative, expected in environment["source_sha256_lf"].items():
            normalized = (
                (ROOT / relative)
                .read_bytes()
                .replace(b"\r\n", b"\n")
                .replace(b"\r", b"\n")
            )
            self.assertEqual(hashlib.sha256(normalized).hexdigest(), expected)
        for item in json.loads((data / "manifest.json").read_text()):
            content = (data / item["path"]).read_bytes()
            self.assertEqual(len(content), item["bytes"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), item["sha256"])
        measurements = rows(data / "measurements.csv")
        self.assertEqual(len(measurements), 4 * 2 * environment["repetitions"])
        summaries = rows(data / "summary.csv")
        self.assertEqual(len(summaries), 8)
        for row in summaries:
            name, method = row["case"], row["method"]
            case = json.loads((data / "inputs" / f"{name}.json").read_text())
            model = json.loads((data / "models" / f"{name}__{method}.json").read_text())
            matrix = np.array(model["matrix"])
            observed = transfer_errors(matrix, case["source"], case["target"])
            clean = transfer_errors(matrix, case["source"], case["clean_target"])
            held = transfer_errors(matrix, case["held_source"], case["held_target"])
            train = rows(data / "errors" / f"{name}__{method}__train.csv")
            held_rows = rows(data / "errors" / f"{name}__{method}__held.csv")
            np.testing.assert_allclose(
                [float(r["observed_transfer_error_px"]) for r in train],
                observed,
                rtol=1e-9,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                [float(r["clean_transfer_error_px"]) for r in train],
                clean,
                rtol=1e-9,
                atol=1e-9,
            )
            np.testing.assert_allclose(
                [float(r["transfer_error_px"]) for r in held_rows],
                held,
                rtol=1e-9,
                atol=1e-9,
            )
            predicted = project(matrix, case["held_source"]).points
            np.testing.assert_allclose(
                [[float(r["predicted_x"]), float(r["predicted_y"])] for r in held_rows],
                predicted,
                rtol=1e-10,
                atol=1e-9,
            )
            truth = np.array(case["true_inlier"])
            mask = observed <= 2.5
            self.assertEqual(int(row["threshold_inliers"]), int(mask.sum()))
            self.assertEqual(int(row["true_positive"]), int(np.sum(mask & truth)))
            self.assertEqual(int(row["false_positive"]), int(np.sum(mask & ~truth)))
            self.assertAlmostEqual(
                float(row["held_clean_rmse_px"]),
                float(np.sqrt(np.mean(held**2))),
                places=7,
            )
            if method == "ransac":
                np.testing.assert_array_equal(model["inliers"], mask)
                result = ransac(case["source"], case["target"], seed=17)
                np.testing.assert_allclose(result.matrix, matrix, rtol=1e-7, atol=1e-9)
                np.testing.assert_array_equal(result.inliers, mask)
                trace = json.loads((data / "traces" / f"{name}.json").read_text())
                self.assertEqual(len(trace["iterations"]), model["trials"])
                self.assertEqual(
                    trace["iterations"][-1]["best_inliers"], int(mask.sum())
                )
                for index, event in enumerate(trace["iterations"], 1):
                    self.assertEqual(event["trial"], index)
                    self.assertEqual(len(set(event["sample"])), 4)
                    self.assertTrue(all(0 <= i < len(truth) for i in event["sample"]))
                np.testing.assert_allclose(
                    trace["improvements"][-1]["matrix"], matrix, rtol=0, atol=0
                )
            else:
                np.testing.assert_allclose(
                    normalized_dlt(case["source"], case["target"]).matrix,
                    matrix,
                    rtol=1e-8,
                    atol=1e-10,
                )
            samples = [
                int(m["elapsed_ns"])
                for m in measurements
                if m["case"] == name and m["method"] == method
            ]
            self.assertEqual(len(samples), environment["repetitions"])
            self.assertTrue(all(value > 0 for value in samples))
            self.assertEqual(float(row["median_ns"]), statistics.median(samples))
            self.assertEqual(int(row["min_ns"]), min(samples))
            self.assertEqual(int(row["max_ns"]), max(samples))
        visual = json.loads((ROOT / "docs/images/provenance.json").read_text())
        for mapping in [visual["input_sha256"], visual["output_sha256"]]:
            for relative, expected in mapping.items():
                self.assertEqual(
                    hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected
                )
        replay = json.loads((ROOT / "docs/media/comparison.json").read_text())
        for relative, expected in replay["input_sha256"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected
            )
        self.assertEqual(
            hashlib.sha256(
                (ROOT / "docs/media/fit-comparison.mp4").read_bytes()
            ).hexdigest(),
            replay["output_sha256"],
        )
        renderer = (
            (ROOT / "scripts/render_comparison.py")
            .read_bytes()
            .replace(b"\r\n", b"\n")
            .replace(b"\r", b"\n")
        )
        self.assertEqual(
            hashlib.sha256(renderer).hexdigest(), replay["renderer_sha256_lf"]
        )
        self.assertEqual(replay["frames"] / replay["fps"], replay["duration_seconds"])


if __name__ == "__main__":
    unittest.main()
