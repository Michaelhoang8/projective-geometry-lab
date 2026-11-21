# Experimental methodology

## Synthetic cases

The generator uses one fixed known planar homography and four independently seeded training sets:

| Case | Pairs | Target Gaussian noise σ, px | Replaced targets |
| --- | ---: | ---: | ---: |
| Exact | 80 | 0 | 0% |
| Noise only | 100 | 0.7 | 0% |
| Outliers 35 | 120 | 0.7 | 35% |
| Outliers 60 | 160 | 0.7 | 60% |

Source points are uniformly sampled within x = 20–492 and y = 20–340. The known coordinate mapping produces clean targets. Independent Gaussian noise is added to target coordinates only. A known subset of targets is then replaced by unrelated uniform coordinates in the 512 × 420 target region. Seeds are `20261121` through `20261124`, using NumPy's default generator with the pinned NumPy version.

These inputs are explicitly selected synthetic tests. The percentages do not model a real feature matcher's error distribution, and uniformly replaced outliers are easier than some structured false-match patterns. No external images or feature detector are involved.

## Held-out evaluation

A 15 × 11 grid provides 165 source points unused during fitting. Their clean target coordinates are calculated from the known homography. The grid spans x = 12–500 and y = 12–348, slightly beyond the random training interior. All four cases use that same held-out grid and same ground-truth transform, so they are repeated evaluations of one geometric mapping rather than independent real-scene trials.

Each fitted model is evaluated using forward Euclidean target-coordinate errors. The study preserves per-point errors and reports held-out RMSE, median and maximum. Training summaries separately report error against noisy observations on the synthetic true-inlier subset and error against clean synthetic targets. Known corruption labels are evaluation metadata; they are never passed to DLT or RANSAC.

Precision and recall compare the returned threshold mask with synthetic corruption labels. For the all-point DLT model, a threshold mask is computed only for evaluation; it is not used in its fit. Zero accepted points yields undefined precision, represented by an empty CSV field rather than zero.

## Estimation configurations

The baseline fits normalized DLT to every training correspondence. The robust method uses the same DLT primitive inside RANSAC, with a 2.5 target-pixel forward threshold, seed 17, minimum 20 draws, maximum 1000 draws, confidence setting 0.995 and at most three improving consensus refits per candidate. These choices are fixed across cases.

Three additional degeneracy fixtures are preserved in `degeneracies.json`: coincident points, all-collinear points and a minimal set with three collinear points. Tests also cover valid `h33 = 0`, projective-scale changes, near-infinite projections and large coordinate offsets. The implementation's exact numerical thresholds are documented in [algorithms.md](algorithms.md).

## Timing

Each exact method/case call is warmed once and supplies the preserved model. Five timed repetitions follow. Method order is shuffled each repetition with independent seed `1307`. Every timed matrix must exactly match its untimed counterpart on the same process; RANSAC masks and traces must also match.

`time.perf_counter_ns()` records complete in-memory estimation calls. DLT timing includes validation, normalization, design construction, SVD and denormalization. RANSAC additionally includes sampling, degeneracy tests, scoring, consensus refits and trace construction. Input generation, file I/O, held-out evaluation and all image rendering are excluded.

The run contains `4 cases × 2 methods × 5 repetitions = 40` elapsed observations. Summaries preserve medians and observed minima/maxima without dropping samples. Ranges are not confidence intervals. There is no CPU pinning, isolated process or background-load control. The study makes no real-time or general estimator-speed claim. CI creates a separate two-repetition smoke run without timing thresholds.

## Numerical reproducibility

Coordinates and matrices use `float64`; NumPy `2.3.5` supplies SVD and linear-system primitives. Fixed seeds make the sampled subsets reproducible for the recorded environment. Tiny floating-point differences across BLAS implementations can affect exact-case tie scores and equivalent matrices, so cross-platform verification compares mappings/matrices with documented numerical tolerances rather than promising bit-identical SVD results.

The recorded core, generator and study script are hashed before measurement, normalizing CRLF/CR to LF. `environment.json` includes these hashes, actual UTC timestamps, Python/NumPy versions, architecture, seeds and timing scope. `manifest.json` stores exact-byte hashes and sizes of every other experiment artifact. Generated text is UTF-8 with LF line endings.

New runs require an empty output directory and do not overwrite `results/published/`. Evidence tests verify source and artifact hashes, recompute transfer errors from saved matrices, check masks and trace structure, rerun estimators with tolerance, and reconcile raw timing aggregates.

## Procedural imagery and animation

`scripts/build_visuals.py` draws the source plane using owned shapes, grids, marker patterns and literal coordinate labels. The point correspondences are generated independently of those pixels; the project does not claim to detect the patterns. The known, all-point and robust matrices inverse-warp this same source graphic using the project's own bilinear sampler. Images share a 512 × 420 target canvas, which crops some transformed content. Dark regions indicate unsupported source coordinates.

The eight-second silent MP4 contains four two-second crossfades between the known target and each fixed fitted warp. Its weights vary as a cosine, revealing misalignment. It is a fixed-estimate comparison, not a visualization of the optimization process; actual sampled RANSAC trials remain available in the CSV/JSON traces. The animation is 1200 × 660 at 30 frames/s and never autoplays.

Image and video metadata record input and output hashes, renderer versions and timing policy. NumPy `2.3.5`, Matplotlib `3.11.2` and Pillow `13.0.0` are pinned. Optional MP4 regeneration requires ffmpeg/libx264, with its actual version recorded. Codec bytes may vary across encoder versions, while the committed file is hash-checked.

## Limits

The homography assumption applies to the declared planar mapping. The experiment excludes nonplanar scenes, camera calibration, radial distortion, rolling shutter, feature detection, uncertain source coordinates, photometric matching, real imagery and deployment. The limited cases demonstrate algorithm mechanics and selected failure behavior; they do not establish general RANSAC reliability or camera-reconstruction accuracy. This project was developed with AI assistance.
