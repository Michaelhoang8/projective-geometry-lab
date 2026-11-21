# Algorithms and numerical assumptions

## Mapping and residual

Coordinates use `[x, y]`, with x increasing rightward and y downward in rendered images. A homography maps homogeneous source coordinates to target coordinates:

```text
[u', v', w']^T = H [x, y, 1]^T
target = [u'/w', v'/w']
```

Matrices differing by a nonzero scalar represent the same projective mapping. The implementation first divides by the largest absolute entry, then uses unit Frobenius norm and chooses the sign making the largest-magnitude entry positive. It never requires `h33 = 1`. A valid nonsingular homography can have `h33 = 0`.

Projection marks a point invalid when the result is nonfinite or when `abs(w') <= 1e-12 * max(abs(u'), abs(v'), abs(w'))`. This relative rule is invariant to homography scaling. Invalid projected coordinates are NaN; their transfer errors are infinite. The threshold deliberately treats very near-horizon projections as numerically unusable rather than returning arbitrarily large coordinates.

The estimator's residual is forward Euclidean target-space error:

```text
e_i = || project(H, source_i) - observed_target_i ||_2
```

It is measured in the target coordinate unit, called pixels in these synthetic fixtures. Source coordinates are exact in the generation model; noise is added to target coordinates. Forward error is asymmetric. The project does not use symmetric transfer error, reprojection bundle adjustment or a calibrated camera likelihood.

## Normalized DLT

For each coordinate set, subtract the centroid and scale so its mean distance from the origin is √2. Denote the similarity transforms by `T_source` and `T_target`. For each normalized correspondence `(x,y) -> (u,v)`, the design matrix receives two rows:

```text
[-x, -y, -1,  0,  0,  0, u*x, u*y, u]
[ 0,  0,  0, -x, -y, -1, v*x, v*y, v]
```

The right-singular vector with smallest singular value is reshaped into the normalized homography. Denormalization is `solve(T_target, H_normalized @ T_source)`. This follows the normalized DLT construction in Hartley and Zisserman, *Multiple View Geometry in Computer Vision*, second edition, section 4.4 and Algorithm 4.2; the [authors' book page](https://www.robots.ox.ac.uk/~vgg/hzbook/) provides the reference and errata.

For exactly four pairs, the design is 8 × 9. A reduced SVD returns only eight right-singular vectors and omits the one-dimensional null vector. The implementation therefore requests `full_matrices=True` in this case. For at least five pairs it uses the reduced decomposition, which still contains all nine right-singular vectors while avoiding an unnecessarily large left basis. The output-shape behavior is specified in the [NumPy 2.3 SVD documentation](https://numpy.org/doc/2.3/reference/generated/numpy.linalg.svd.html).

DLT minimizes a normalized algebraic residual, not geometric transfer error. Consensus refitting uses the same DLT primitive; it is not nonlinear maximum-likelihood optimization.

## Degeneracy policy

The implementation rejects:

- Fewer than four pairs, unequal counts, nonfinite/complex coordinates or fewer than four distinct points in either image.
- Coincident sets and point clouds whose second centered singular value is at most `1e-10` times the first.
- Any three nearly collinear points in a minimal four-point set, using absolute doubled triangle area at most `1e-9` in normalized coordinates.
- A DLT design with fewer than eight singular values above `1e-10` times its largest singular value.
- A normalized estimated homography whose smallest singular value is at most `1e-12` times its largest.

Overdetermined noisy designs commonly have rank nine and are accepted. `constraint_condition` reports the largest design singular value divided by its eighth; it is a diagnostic of the eight required constraint directions, not a coordinate-invariant error guarantee. The tolerances are numerical policy choices, tested for the stated coordinate ranges. Very ill-conditioned valid geometry can be rejected.

## RANSAC

Each trial samples four distinct indices using a seeded Python random generator. A degenerate sample is counted and skipped. A candidate is scored first by the number of points with `e_i <= threshold`, then by the negative clipped squared-error sum `-sum(min(e_i, threshold)^2)` to break equal-count ties.

When a candidate can improve the best score, up to three local refits estimate DLT on its current consensus. A refit is accepted only if the same score improves. The returned mask is recomputed from the returned matrix; it is not a stale sample mask. Improvement matrices, sampled subsets, candidate scores and current trial limits are preserved.

Sampling-based consensus follows the approach introduced by Fischler and Bolles in [“Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography” (1981)](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/f/p381-fischler.pdf), DOI `10.1145/358669.358692`. This repository's particular score, refit cap and stopping policy are specified here rather than attributed to that paper as an implementation.

## Conditional trial calculation

If `I` out of `N` points were known to be true inliers, a uniform sample of four distinct indices would have all-inlier probability:

```text
p = product((I-j)/(N-j), j=0..3)
K = ceil(log(1-confidence) / log(1-p))
```

This uses sampling without replacement within each subset; subsets can repeat across trials. The implementation substitutes its current consensus count for unknown `I`, clamps the result to the configured maximum, and enforces a minimum trial count. Published runs use confidence 0.995, minimum 20, maximum 1000 and seed 17.

The calculation assumes that the current consensus approximates the true inlier set and an all-inlier subset can yield a useful model. Noise, near-degeneracy, threshold mismatch and structured outliers can violate those assumptions. Skipped degenerate draws still count toward the actual budget. Thus the confidence setting is not an observed success rate or a correctness guarantee. If no usable model appears within the cap, the API returns `no_model` with its trace.

## Image resampling

Images are inverse-mapped at integer pixel centers. For each target center, the inverse homography locates a source coordinate. Valid source support is `[0, width-1] × [0, height-1]`; bilinear interpolation combines its four neighboring source samples. At the last row/column the neighboring index is clamped to the boundary.

Homogeneous division may put a mathematically exact boundary a few floating-point units outside it. The implementation permits a halo of `32 * float64_epsilon * max(source_height, source_width)` and clamps accepted coordinates to the boundary. A 64 × 64 identity regression verifies that no final row/column disappears; a genuine `1e-9` shift beyond support is still rejected. This tolerance is numerical, not an expanded geometric footprint.

Invalid/out-of-support destinations have zero numerical output; the visualization renderer colors them dark. Bilinear interpolation is not a minification anti-alias filter. Output images use a 512 × 420 canvas and may crop the transformed plane. Correspondence estimation is independent of that texture and crop.
