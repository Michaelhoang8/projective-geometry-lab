"""Normalized DLT, bounded RANSAC and inverse image mapping; no OpenCV estimator."""

from dataclasses import dataclass
from itertools import combinations
import math
import random

import numpy as np


class GeometryError(ValueError):
    """Invalid input or numerically degenerate projective geometry."""


def _points(value, minimum=0):
    if np.iscomplexobj(value):
        raise GeometryError("Points must be real")
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryError("Points must be numeric") from exc
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < minimum
        or not np.isfinite(points).all()
    ):
        raise GeometryError(f"Expected at least {minimum} finite 2D points")
    return points


def canonical_matrix(value):
    if np.iscomplexobj(value):
        raise GeometryError("Matrix must be real")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise GeometryError("Expected a finite 3x3 matrix")
    maximum = float(np.max(np.abs(matrix)))
    if maximum == 0:
        raise GeometryError("The zero matrix is not a homography")
    matrix = matrix / maximum
    matrix = matrix / np.linalg.norm(matrix)
    if matrix.flat[int(np.argmax(np.abs(matrix)))] < 0:
        matrix = -matrix
    return matrix


@dataclass(frozen=True)
class Projection:
    points: np.ndarray
    valid: np.ndarray


def project(matrix, points, relative_epsilon=1e-12):
    """Project finite coordinates, marking near-infinite results invalid/NaN.

    The denominator is compared to the largest homogeneous component, so the
    decision is invariant to a nonzero scale of the homography.
    """
    matrix = canonical_matrix(matrix)
    points = _points(points)
    if not math.isfinite(relative_epsilon) or not 0 < relative_epsilon < 1:
        raise GeometryError("Projection tolerance must lie in (0,1)")
    homogeneous = np.column_stack((points, np.ones(len(points))))
    with np.errstate(over="ignore", invalid="ignore"):
        mapped = homogeneous @ matrix.T
    scale = np.max(np.abs(mapped), axis=1)
    valid = np.isfinite(mapped).all(axis=1) & (
        np.abs(mapped[:, 2]) > relative_epsilon * scale
    )
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    result[valid] = mapped[valid, :2] / mapped[valid, 2, None]
    return Projection(result, valid)


def transfer_errors(matrix, source, target):
    """Forward Euclidean target-coordinate error; invalid projections are inf."""
    source = _points(source)
    target = _points(target)
    if source.shape != target.shape:
        raise GeometryError("Correspondence counts must match")
    mapped = project(matrix, source)
    errors = np.full(len(source), np.inf)
    difference = mapped.points[mapped.valid] - target[mapped.valid]
    errors[mapped.valid] = np.hypot(difference[:, 0], difference[:, 1])
    return errors


def _normalize(points):
    centroid = np.mean(points, axis=0)
    shifted = points - centroid
    distances = np.hypot(shifted[:, 0], shifted[:, 1])
    mean_distance = float(np.mean(distances))
    if (
        not np.isfinite(shifted).all()
        or not math.isfinite(mean_distance)
        or mean_distance <= 0
    ):
        raise GeometryError("Coincident or numerically unscalable point set")
    scale = math.sqrt(2) / mean_distance
    normalized = shifted * scale
    transformation = np.array(
        [
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1.0],
        ]
    )
    spread = np.linalg.svd(normalized, compute_uv=False)
    if spread[-1] <= spread[0] * 1e-10:
        raise GeometryError("Collinear or nearly collinear points")
    if len(points) == 4:
        # Minimal projective bases require no three points on one line in either image.
        for a, b, c in combinations(normalized, 3):
            area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(area) <= 1e-9:
                raise GeometryError("Degenerate minimal sample: three collinear points")
    return normalized, transformation


@dataclass(frozen=True)
class DLTResult:
    matrix: np.ndarray
    design_rank: int
    singular_values: tuple[float, ...]
    constraint_condition: float
    used_full_basis: bool


def normalized_dlt(source, target, rank_tolerance=1e-10):
    """Estimate a nonsingular homography by normalized algebraic least squares."""
    source = _points(source, 4)
    target = _points(target, 4)
    if source.shape != target.shape:
        raise GeometryError("Correspondence counts must match")
    if not math.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
        raise GeometryError("Rank tolerance must lie in (0,1)")
    if len(np.unique(source, axis=0)) < 4 or len(np.unique(target, axis=0)) < 4:
        raise GeometryError("At least four distinct points are required in each image")
    src, T_source = _normalize(source)
    dst, T_target = _normalize(target)
    design = np.zeros((2 * len(source), 9), dtype=np.float64)
    for index, ((x, y), (u, v)) in enumerate(zip(src, dst)):
        design[2 * index] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        design[2 * index + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]
    # An 8x9 minimal system needs Vh's ninth row: reduced SVD omits its null vector.
    full_basis = design.shape[0] < design.shape[1]
    _, singular, vh = np.linalg.svd(design, full_matrices=full_basis)
    rank = int(np.count_nonzero(singular > rank_tolerance * singular[0]))
    if rank < 8:
        raise GeometryError("DLT design has fewer than eight independent constraints")
    normalized = vh[-1].reshape(3, 3)
    matrix_singular = np.linalg.svd(normalized, compute_uv=False)
    if matrix_singular[-1] <= matrix_singular[0] * 1e-12:
        raise GeometryError("Estimated normalized homography is numerically singular")
    matrix = np.linalg.solve(T_target, normalized @ T_source)
    matrix = canonical_matrix(matrix)
    return DLTResult(
        matrix,
        rank,
        tuple(float(x) for x in singular),
        float(singular[0] / singular[7]),
        full_basis,
    )


def _score(errors, threshold):
    mask = errors <= threshold
    clipped = float(np.minimum(errors, threshold).dot(np.minimum(errors, threshold)))
    return (int(mask.sum()), -clipped), mask


def _required_trials(consensus, total, confidence, maximum):
    if consensus < 4:
        return maximum
    # Exact all-in-consensus probability for a uniformly sampled four-subset.
    probability = math.prod((consensus - i) / (total - i) for i in range(4))
    if probability >= 1:
        return 1
    if probability <= 0:
        return maximum
    return min(
        maximum, max(1, math.ceil(math.log1p(-confidence) / math.log1p(-probability)))
    )


@dataclass(frozen=True)
class RansacResult:
    status: str
    matrix: np.ndarray | None
    inliers: tuple[bool, ...]
    trials: int
    skipped_degenerate: int
    adaptive_limit: int
    trace: tuple[dict, ...]
    improvements: tuple[dict, ...]


def ransac(
    source,
    target,
    threshold=2.5,
    max_trials=1000,
    min_trials=20,
    confidence=0.995,
    seed=0,
    local_refits=3,
):
    """Bounded seeded RANSAC with consensus refinement and explicit audit trace.

    Confidence-based stopping uses the current consensus as an estimate of the
    true inlier set; it is a conditional sampling model, not a success guarantee.
    """
    source = _points(source, 4)
    target = _points(target, 4)
    if source.shape != target.shape:
        raise GeometryError("Correspondence counts must match")
    if not math.isfinite(threshold) or threshold <= 0:
        raise GeometryError("Threshold must be finite and positive")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise GeometryError("Confidence must lie in (0,1)")
    if (
        type(max_trials) is not int
        or type(min_trials) is not int
        or not 1 <= min_trials <= max_trials
    ):
        raise GeometryError("Trial bounds must satisfy 1 <= min <= max")
    if type(local_refits) is not int or not 0 <= local_refits <= 20:
        raise GeometryError("Local refits must be an integer in 0..20")
    rng = random.Random(seed)
    best_score = (-1, -math.inf)
    best_matrix = None
    best_mask = np.zeros(len(source), dtype=bool)
    limit = max_trials
    trials = skipped = 0
    trace = []
    improvements = []
    while trials < limit:
        indices = tuple(sorted(rng.sample(range(len(source)), 4)))
        trials += 1
        event = {
            "trial": trials,
            "sample": indices,
            "candidate_inliers": None,
            "candidate_clipped_squared_error": None,
            "skipped": False,
            "improved": False,
        }
        try:
            candidate = normalized_dlt(
                source[list(indices)], target[list(indices)]
            ).matrix
            score, mask = _score(transfer_errors(candidate, source, target), threshold)
            if score > best_score and score[0] >= 4:
                for _ in range(local_refits):
                    try:
                        refined = normalized_dlt(source[mask], target[mask]).matrix
                    except (GeometryError, np.linalg.LinAlgError):
                        break
                    refined_score, refined_mask = _score(
                        transfer_errors(refined, source, target), threshold
                    )
                    if refined_score <= score:
                        break
                    candidate, score, mask = refined, refined_score, refined_mask
                if score > best_score:
                    best_matrix, best_score, best_mask = candidate, score, mask
                    limit = max(
                        min_trials,
                        _required_trials(score[0], len(source), confidence, max_trials),
                    )
                    event["improved"] = True
                    improvements.append(
                        {
                            "trial": trials,
                            "inlier_count": score[0],
                            "matrix": candidate.tolist(),
                        }
                    )
            event["candidate_inliers"] = score[0]
            event["candidate_clipped_squared_error"] = -score[1]
        except (GeometryError, np.linalg.LinAlgError):
            skipped += 1
            event["skipped"] = True
        event.update({"best_inliers": max(0, best_score[0]), "adaptive_limit": limit})
        trace.append(event)
    return RansacResult(
        "found" if best_matrix is not None else "no_model",
        best_matrix,
        tuple(bool(x) for x in best_mask),
        trials,
        skipped,
        limit,
        tuple(trace),
        tuple(improvements),
    )


def warp_image(image, matrix, output_shape):
    """Inverse-map pixel centers with bilinear sampling and zero outside support."""
    source = np.asarray(image, dtype=np.float64)
    if (
        source.ndim not in (2, 3)
        or min(source.shape[:2]) < 1
        or not np.isfinite(source).all()
    ):
        raise GeometryError("Expected a finite nonempty gray or multichannel image")
    if len(output_shape) != 2 or any(
        type(size) is not int or size <= 0 for size in output_shape
    ):
        raise GeometryError(
            "Output shape must contain positive integer height and width"
        )
    try:
        inverse = np.linalg.inv(canonical_matrix(matrix))
    except np.linalg.LinAlgError as exc:
        raise GeometryError("Cannot invert singular homography") from exc
    height, width = output_shape
    yy, xx = np.indices((height, width))
    mapped = project(inverse, np.column_stack((xx.ravel(), yy.ravel())))
    x, y = mapped.points[:, 0], mapped.points[:, 1]
    # Homogeneous division can place an exact boundary center a few ulps beyond
    # its integer coordinate. Admit and clamp only this explicitly bounded halo.
    boundary_tolerance = 32 * np.finfo(np.float64).eps * max(source.shape[:2])
    valid = (
        mapped.valid
        & (x >= -boundary_tolerance)
        & (y >= -boundary_tolerance)
        & (x <= source.shape[1] - 1 + boundary_tolerance)
        & (y <= source.shape[0] - 1 + boundary_tolerance)
    )
    x = np.clip(x, 0, source.shape[1] - 1)
    y = np.clip(y, 0, source.shape[0] - 1)
    result = np.zeros((height * width,) + source.shape[2:], dtype=np.float64)
    x0 = np.floor(x[valid]).astype(int)
    y0 = np.floor(y[valid]).astype(int)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    dx = x[valid] - x0
    dy = y[valid] - y0
    if source.ndim == 3:
        dx = dx[:, None]
        dy = dy[:, None]
    result[valid] = (
        (1 - dx) * (1 - dy) * source[y0, x0]
        + dx * (1 - dy) * source[y0, x1]
        + (1 - dx) * dy * source[y1, x0]
        + dx * dy * source[y1, x1]
    )
    return result.reshape((height, width) + source.shape[2:]), valid.reshape(
        height, width
    )
