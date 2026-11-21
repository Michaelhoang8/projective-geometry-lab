"""Analytic homography anchors, independent errors and robust-estimation checks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from projectivelab.core import (
    GeometryError,
    canonical_matrix,
    normalized_dlt,
    project,
    ransac,
    transfer_errors,
    warp_image,
)
from projectivelab.fixtures import cases


def independently_project(matrix, points):
    outputs = []
    for x, y in points:
        u = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        v = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        w = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        outputs.append([u / w, v / w])
    return np.array(outputs)


class GeometryTests(unittest.TestCase):
    def test_minimal_full_basis_and_exact_analytic_homographies(self):
        source = np.array([[1, 1], [5, 1], [5, 4], [1, 4]], dtype=float)
        held = np.array([[2, 2], [3, 1.5], [4, 3]])
        matrices = [
            np.eye(3),
            np.array([[1, 0, 13], [0, 1, -8], [0, 0, 1.0]]),
            np.array([[0.9, -0.3, 2], [0.3, 0.9, 4], [0, 0, 1.0]]),
            np.array([[1, 0.2, 2], [-0.1, 0.8, 3], [0.03, -0.01, 1.0]]),
        ]
        for matrix in matrices:
            target = independently_project(matrix, source)
            result = normalized_dlt(source, target)
            self.assertTrue(result.used_full_basis)
            self.assertEqual(result.design_rank, 8)
            np.testing.assert_allclose(
                project(result.matrix, held).points,
                independently_project(matrix, held),
                rtol=2e-12,
                atol=2e-12,
            )

    def test_valid_h33_zero_and_projective_scale_invariance(self):
        matrix = np.array([[1, 0, 1], [0, 1, 2], [0.2, 0.1, 0.0]])
        source = np.array([[1, 1], [4, 1], [4, 3], [1, 3]], dtype=float)
        target = independently_project(matrix, source)
        result = normalized_dlt(source, target)
        self.assertLess(abs(result.matrix[2, 2]), 1e-12)
        self.assertAlmostEqual(np.linalg.norm(result.matrix), 1)
        for scale in [1e-100, -1, 1e100]:
            projection = project(matrix * scale, source)
            self.assertTrue(projection.valid.all())
            np.testing.assert_allclose(
                projection.points, target, rtol=2e-14, atol=2e-14
            )
        np.testing.assert_allclose(
            project(result.matrix, source).points, target, rtol=2e-12, atol=2e-12
        )

    def test_near_infinity_is_invalid_and_errors_are_infinite(self):
        matrix = np.array([[1, 0, 1], [0, 1, 2], [1, 0, -1.0]])
        points = np.array([[1, 0], [1 + 1e-14, 0], [2, 1]])
        mapped = project(matrix, points)
        np.testing.assert_array_equal(mapped.valid, [False, False, True])
        self.assertTrue(np.isnan(mapped.points[:2]).all())
        errors = transfer_errors(matrix, points, np.zeros((3, 2)))
        self.assertTrue(np.isinf(errors[:2]).all())
        self.assertAlmostEqual(errors[2], np.hypot(3, 3))

    def test_degeneracy_and_invalid_correspondences_are_rejected(self):
        cases = [
            np.zeros((4, 2)),
            np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
            np.array([[0, 0], [1, 0], [2, 0], [0, 1]]),
            np.array([[0, 0], [1, 0], [1, 0], [0, 1]]),
        ]
        for points in cases:
            with self.assertRaises(GeometryError):
                normalized_dlt(points, points)
        with self.assertRaises(GeometryError):
            normalized_dlt([[0, 0]] * 3, [[0, 0]] * 3)
        with self.assertRaises(GeometryError):
            normalized_dlt([[0, 0], [1, 0], [1, 1], [0, float("nan")]], [[0, 0]] * 4)
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        result = ransac(points, points, max_trials=7, min_trials=1)
        self.assertEqual(result.status, "no_model")
        self.assertEqual(result.trials, 7)
        self.assertEqual(result.skipped_degenerate, 7)

    def test_large_coordinate_normalization(self):
        source = np.array(
            [
                [1e6, 2e6],
                [1e6 + 300, 2e6],
                [1e6 + 300, 2e6 + 200],
                [1e6, 2e6 + 200],
                [1e6 + 170, 2e6 + 83],
            ]
        )
        matrix = np.array([[1.1, 0.05, 12], [-0.03, 0.97, -34], [1e-8, -2e-8, 1.0]])
        target = independently_project(matrix, source)
        result = normalized_dlt(source, target)
        np.testing.assert_allclose(
            project(result.matrix, source).points, target, rtol=0, atol=2e-8
        )

    def test_ransac_outliers_mask_error_and_determinism(self):
        case = next(case for case in cases() if case["name"] == "outliers_35")
        result = ransac(case["source"], case["target"], seed=17)
        repeated = ransac(case["source"], case["target"], seed=17)
        self.assertEqual(result.status, "found")
        self.assertEqual(result.trace, repeated.trace)
        np.testing.assert_array_equal(result.matrix, repeated.matrix)
        errors = transfer_errors(result.matrix, case["source"], case["target"])
        np.testing.assert_array_equal(result.inliers, errors <= 2.5)
        actual = independently_project(result.matrix, case["held_source"])
        rmse = np.sqrt(
            np.mean(np.sum((actual - np.array(case["held_target"])) ** 2, axis=1))
        )
        self.assertLess(rmse, 0.6)
        truth = np.array(case["true_inlier"])
        self.assertGreater(np.sum(np.array(result.inliers) & truth) / truth.sum(), 0.95)
        self.assertGreaterEqual(result.trials, 20)
        self.assertLessEqual(result.trials, 1000)

    def test_inverse_warp_identity_translation_and_bilinear_anchor(self):
        image = np.arange(20, dtype=float).reshape(4, 5)
        warped, valid = warp_image(image, np.eye(3), (4, 5))
        np.testing.assert_allclose(warped, image, atol=1e-14)
        self.assertTrue(valid.all())
        matrix = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1.0]])
        shifted, mask = warp_image(image, matrix, (4, 5))
        np.testing.assert_allclose(shifted[:, 1:], image[:, :-1])
        self.assertFalse(mask[:, 0].any())
        half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]])
        interpolated, mask = warp_image(image, half, (4, 5))
        self.assertAlmostEqual(
            interpolated[1, 1],
            (image[0, 0] + image[0, 1] + image[1, 0] + image[1, 1]) / 4,
        )
        self.assertTrue(mask[1, 1])

    def test_cli_exact_case_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            output = Path(directory) / "result.json"
            points = [[0, 0], [1, 0], [1, 1], [0, 1]]
            path.write_text(json.dumps({"source": points, "target": points}))
            command = [
                sys.executable,
                "-m",
                "projectivelab",
                str(path),
                "--method",
                "dlt",
                "--output",
                str(output),
            ]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads(output.read_text())["status"], "found")
            self.assertNotEqual(
                subprocess.run(command, capture_output=True).returncode, 0
            )

    def test_warp_roundoff_halo_preserves_boundary_without_admitting_real_outside(self):
        image = np.arange(4096, dtype=float).reshape(64, 64)
        warped, valid = warp_image(image, np.eye(3), (64, 64))
        self.assertEqual(int(valid.sum()), 4096)
        np.testing.assert_allclose(warped, image, rtol=0, atol=2e-12)
        # Inverse mapping shifts the right edge genuinely outside by 1e-9,
        # well beyond the documented roughly 4.55e-13 boundary tolerance.
        moved = np.array([[1, 0, -1e-9], [0, 1, 0], [0, 0, 1.0]])
        _, valid = warp_image(image, moved, (64, 64))
        self.assertFalse(valid[:, -1].any())
        self.assertTrue(valid[:, :-1].all())


if __name__ == "__main__":
    unittest.main()
