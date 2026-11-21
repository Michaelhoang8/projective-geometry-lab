# Findings

## All-point fitting is sensitive to the replaced targets

With 35% uniformly replaced targets, all-point normalized DLT has held-out RMSE 246.208 px. RANSAC with consensus refitting reduces it to 0.269 px in the recorded case. The returned mask contains all 78 synthetic inliers and none of the 42 replacements.

With 60% replacements, all-point held-out RMSE rises to 2465.869 px, while the robust model gives 0.298 px. Its worst held-out error is 1.153 px despite a subpixel RMSE, showing why the maximum and per-point distribution are preserved as well. The sampled RANSAC count increases from 30 to 217 between these cases.

The distorted all-point image is a direct warp using the fitted matrix. It is not an illustrative deformation added to exaggerate the difference. The comparison animation uses those fixed fitted images and the known target.

## Noise-only fitting does not require outlier rejection to work

The noise-only case gives held-out RMSE 0.235 px for all-point DLT and 0.210 px for RANSAC. One of the 100 true noisy inliers falls outside the chosen 2.5 px threshold for both fitted models, producing recall 0.99. A threshold mask is a geometric decision, not a perfect oracle for the corruption label.

The slight error difference on this one draw does not establish that robust fitting is generally better on uncontaminated data. It also costs more sampled fits. Exact correspondences give errors around 10⁻¹³ px for both methods, consistent with numerical precision in this configuration.

## Geometry checks matter independently of noise

The minimal 8 × 9 DLT system requires the full right-singular basis. Analytic tests verify exact transfer of held-out points, including a valid transform with `h33 = 0`. The matrix gauge is therefore independent of that particular coefficient.

Degenerate fixtures are rejected instead of producing a plausible-looking matrix. Near-horizon projections are explicitly invalid. Image resampling tests also caught and fixed floating-point support loss on the last row and column of a 64 × 64 identity warp; the regression distinguishes a few-ulp boundary error from a genuine out-of-support coordinate.

## Conditional confidence and scope

The 0.995 RANSAC setting controls a consensus-based trial calculation. It is not a measured success probability. These cases use uniform replacement outliers, a known planar mapping and one fixed threshold. Structured mismatches, heavy noise and nearly degenerate samples can behave differently.

The held-out grid is unused by fitting, but all cases share one true homography. The findings support this inspectable synthetic study, not a general real-image performance claim.
