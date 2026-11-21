"""Independent affine interpolation and projective scale anchors."""

import unittest
import numpy as np
from projectivelab.core import normalized_dlt, project, warp_image


class ReviewTests(unittest.TestCase):
    def test_half_pixel_translation_interpolates_affine_rgb_field(self):
        y, x = np.indices((64, 64), dtype=float)
        image = np.stack((2 * x + 3 * y, x - y, np.ones_like(x)), axis=-1)
        matrix = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1.0]])
        warped, valid = warp_image(image, matrix, (64, 64))
        self.assertEqual(int(valid.sum()), 63 * 63)
        self.assertFalse(valid[0].any())
        self.assertFalse(valid[:, 0].any())
        self.assertTrue(valid[1:, 1:].all())
        np.testing.assert_allclose(
            warped[1:, 1:, 0], 2 * (x[1:, 1:] - 0.5) + 3 * (y[1:, 1:] - 0.5), atol=1e-12
        )
        np.testing.assert_allclose(warped[1:, 1:, 1], x[1:, 1:] - y[1:, 1:], atol=1e-12)
        np.testing.assert_allclose(warped[1:, 1:, 2], 1, atol=1e-14)
        self.assertTrue(np.all(warped[~valid] == 0))

    def test_minimal_dlt_recovers_homography_with_zero_bottom_right(self):
        source = np.array([[1, 1], [5, 1], [5, 4], [1, 4]], dtype=float)

        # Exact projective map, evaluated directly without production projection.
        def transform(points):
            x, y = points[:, 0], points[:, 1]
            return np.column_stack(
                ((x + 2) / (0.2 * x + 0.1 * y), (y + 1) / (0.2 * x + 0.1 * y))
            )

        result = normalized_dlt(source, transform(source))
        self.assertEqual(result.design_rank, 8)
        self.assertTrue(result.used_full_basis)
        self.assertLess(abs(result.matrix[2, 2]), 1e-12)
        held_out = np.array([[2, 2], [3, 1.5], [4, 3]], dtype=float)
        projected = project(result.matrix, held_out)
        self.assertTrue(projected.valid.all())
        np.testing.assert_allclose(
            projected.points, transform(held_out), atol=2e-12, rtol=2e-12
        )

    def test_projection_scale_and_point_at_infinity(self):
        matrix = np.array([[1, 0, 2], [0, 1, 1], [0.2, 0.1, 0.0]])
        points = np.array([[1, -2], [0, 1], [2, 3]], dtype=float)
        for scale in (1.0, -1.0, 1e250, -1e-250):
            result = project(matrix * scale, points)
            np.testing.assert_array_equal(result.valid, [False, True, True])
            self.assertTrue(np.isnan(result.points[0]).all())
            np.testing.assert_allclose(
                result.points[1:], [[20, 20], [4 / 0.7, 4 / 0.7]], atol=2e-12
            )
