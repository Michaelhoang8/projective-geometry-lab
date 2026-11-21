"""Owned synthetic planar correspondences with labeled corruption for evaluation."""

import numpy as np

SEED = 20261121
TRUE_H = np.array([[0.91, 0.13, 28.0], [-0.06, 1.02, 40.0], [0.00055, -0.00028, 1.0]])


def analytic_transfer(points, matrix=TRUE_H):
    # Direct coordinate equations, separate from the estimator's projection API.
    result = []
    for x, y in points:
        denominator = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
        result.append(
            (
                (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denominator,
                (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denominator,
            )
        )
    return np.asarray(result)


def cases():
    configurations = [
        ("exact", 80, 0.0, 0.0),
        ("noise_only", 100, 0.7, 0.0),
        ("outliers_35", 120, 0.7, 0.35),
        ("outliers_60", 160, 0.7, 0.60),
    ]
    result = []
    held_source = np.array(
        [(x, y) for y in np.linspace(12, 348, 11) for x in np.linspace(12, 500, 15)]
    )
    for index, (name, count, sigma, outlier_fraction) in enumerate(configurations):
        rng = np.random.default_rng(SEED + index)
        source = rng.uniform([20, 20], [492, 340], size=(count, 2))
        clean = analytic_transfer(source)
        observed = clean + rng.normal(0, sigma, size=clean.shape)
        inlier = np.ones(count, dtype=bool)
        corrupted = rng.choice(
            count, size=round(count * outlier_fraction), replace=False
        )
        inlier[corrupted] = False
        observed[corrupted] = rng.uniform([0, 0], [512, 420], size=(len(corrupted), 2))
        result.append(
            {
                "name": name,
                "seed": SEED + index,
                "noise_sigma_target_px": sigma,
                "outlier_fraction": outlier_fraction,
                "source": source.tolist(),
                "target": observed.tolist(),
                "clean_target": clean.tolist(),
                "true_inlier": inlier.tolist(),
                "true_matrix": TRUE_H.tolist(),
                "held_source": held_source.tolist(),
                "held_target": analytic_transfer(held_source).tolist(),
            }
        )
    return result
