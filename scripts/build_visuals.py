"""Generate owned plane texture and inverse warps from the preserved fitted matrices."""

import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from projectivelab.core import project, warp_image

DATA = ROOT / "results/published"
OUT = ROOT / "docs/images"
FIG = ROOT / "docs/figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
PAPER = "#f2f4f7"
INK = "#192a45"
BLUE = "#3155d6"
CORAL = "#d2573b"
MUTED = "#647087"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#b8c2d2",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 170,
    }
)


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.tight_layout(pad=1.3)
    for suffix in ["png", "svg"]:
        destination = FIG / f"{name}.{suffix}"
        fig.savefig(destination, bbox_inches="tight")
        if suffix == "svg":
            # Preserve exact artifact hashes across LF-normalized Git checkouts.
            destination.write_bytes(destination.read_bytes().replace(b"\r\n", b"\n"))
    plt.close(fig)


def source_texture():
    image = Image.new("RGB", (512, 360), "#f4edda")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans"), 12)
    for x in range(0, 512, 32):
        draw.line((x, 0, x, 359), fill="#c8c4b9", width=1)
    for y in range(0, 360, 32):
        draw.line((0, y, 511, y), fill="#c8c4b9", width=1)
    draw.rectangle((12, 12, 499, 347), outline=INK, width=3)
    draw.rectangle((24, 24, 200, 70), fill=BLUE)
    draw.text((35, 39), "PLANE 512 x 360", font=font, fill="white")
    draw.ellipse((50, 112, 218, 280), fill=CORAL, outline=INK, width=4)
    draw.ellipse((90, 152, 178, 240), fill="#f4edda", outline=INK, width=3)
    draw.polygon(
        [(300, 90), (452, 135), (415, 260), (270, 220)],
        fill="#e8bc44",
        outline=INK,
        width=4,
    )
    draw.line((290, 300, 465, 300), fill=BLUE, width=10)
    for marker, (x0, y0) in enumerate([(235, 30), (425, 30), (30, 295), (210, 292)]):
        for row in range(5):
            for col in range(5):
                if (row * 3 + col + marker) % 3 or row in {0, 4} or col in {0, 4}:
                    draw.rectangle(
                        (
                            x0 + col * 7,
                            y0 + row * 7,
                            x0 + col * 7 + 6,
                            y0 + row * 7 + 6,
                        ),
                        fill=INK,
                    )
    for x, y in [(55, 85), (260, 95), (235, 250), (450, 275)]:
        draw.line((x - 9, y, x + 9, y), fill=INK, width=2)
        draw.line((x, y - 9, x, y + 9), fill=INK, width=2)
    return image


source = source_texture()
source.save(OUT / "source.png")
pixels = np.asarray(source, dtype=np.float64) / 255
case = json.loads((DATA / "inputs/outliers_35.json").read_text())
matrices = {"reference": np.asarray(case["true_matrix"])}
for method in ["all_point_dlt", "ransac"]:
    matrices[method] = np.asarray(
        json.loads((DATA / "models" / f"outliers_35__{method}.json").read_text())[
            "matrix"
        ]
    )
coverage = {}
for name, matrix in matrices.items():
    warped, valid = warp_image(pixels, matrix, (420, 512))
    # Dark background denotes destinations outside the source image support.
    warped[~valid] = np.array([25, 42, 69]) / 255
    Image.fromarray(np.uint8(np.clip(np.rint(warped * 255), 0, 255))).save(
        OUT / f"{name}.png"
    )
    Image.fromarray(np.uint8(valid) * 255).save(OUT / f"{name}_support.png")
    coverage[name] = int(valid.sum())

summary = read_rows(DATA / "summary.csv")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for ax, key, title in zip(
    axes,
    ["reference", "all_point_dlt", "ransac"],
    ["Known homography", "All-point DLT", "RANSAC + consensus refit"],
):
    ax.imshow(Image.open(OUT / f"{key}.png"))
    ax.axis("off")
    metric = (
        "Synthetic target image"
        if key == "reference"
        else f"Held-out RMSE {float(next(row for row in summary if row['case']=='outliers_35' and row['method']==key)['held_clean_rmse_px']):.3f} px"
    )
    ax.set_title(title + "\n" + metric, loc="left", fontsize=12, pad=10)
save(fig, "warp_comparison")

model = json.loads((DATA / "models/outliers_35__ransac.json").read_text())
src = np.array(case["source"])
dst = np.array(case["target"])
truth = np.array(case["true_inlier"])
inferred = np.array(model["inliers"])
fig, axes = plt.subplots(1, 2, figsize=(11.8, 5))
axes[0].imshow(source)
axes[1].imshow(Image.open(OUT / "reference.png"))
for ax, points, mask in [(axes[0], src, truth), (axes[1], dst, inferred)]:
    ax.scatter(
        points[mask, 0],
        points[mask, 1],
        s=27,
        facecolors="white",
        edgecolors=BLUE,
        linewidths=1.4,
        label="Inlier",
    )
    ax.scatter(
        points[~mask, 0],
        points[~mask, 1],
        s=32,
        c=CORAL,
        marker="x",
        linewidths=1.5,
        label="Outlier",
    )
    ax.set_xlabel("x / pixels")
    ax.set_ylabel("y / pixels")
    ax.legend(frameon=True, fontsize=8, loc="lower left")
axes[0].set_title(
    "Source coordinates / synthetic corruption labels", loc="left", fontsize=11
)
axes[1].set_title(
    "Observed targets / inferred consensus labels", loc="left", fontsize=11
)
save(fig, "correspondences")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for method, color, label in [
    ("all_point_dlt", CORAL, "All-point DLT"),
    ("ransac", BLUE, "RANSAC + refit"),
]:
    errors = np.sort(
        [
            float(row["transfer_error_px"])
            for row in read_rows(DATA / "errors" / f"outliers_35__{method}__held.csv")
        ]
    )
    axes[0].plot(
        errors,
        np.arange(1, len(errors) + 1) / len(errors),
        color=color,
        label=label,
        lw=1.7,
    )
    points = [
        row for row in summary if row["method"] == method and row["case"] != "exact"
    ]
    offset = -0.16 if method == "all_point_dlt" else 0.16
    axes[1].bar(
        np.arange(3) + offset,
        [float(row["held_clean_rmse_px"]) for row in points],
        width=0.3,
        color=color,
        label=label,
    )
axes[0].set_xscale("log")
axes[0].set_xlabel("Held-out forward error (px) / log scale")
axes[0].set_ylabel("Fraction of 165 held-out points")
axes[0].set_ylim(0, 1.02)
axes[0].legend(frameon=False)
axes[1].set_xticks(range(3), ["Noise only", "35% outliers", "60% outliers"])
axes[1].set_yscale("log")
axes[1].set_ylabel("Held-out RMSE (px) / log scale")
axes[1].legend(frameon=False)
save(fig, "held_out_errors")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, color, label in [
    ("outliers_35", BLUE, "35% outliers"),
    ("outliers_60", CORAL, "60% outliers"),
]:
    trace = json.loads((DATA / "traces" / f"{name}.json").read_text())["iterations"]
    axes[0].step(
        [row["trial"] for row in trace],
        [row["best_inliers"] for row in trace],
        where="post",
        color=color,
        label=label,
    )
axes[0].set_xlabel("Actual sampled trial")
axes[0].set_ylabel("Best consensus count")
axes[0].legend(frameon=False)
for method, color, label, offset in [
    ("all_point_dlt", CORAL, "All-point DLT", -0.16),
    ("ransac", BLUE, "RANSAC + refit", 0.16),
]:
    rows = [row for row in summary if row["method"] == method]
    med = np.array([float(row["median_ns"]) / 1e6 for row in rows])
    low = np.array([float(row["min_ns"]) / 1e6 for row in rows])
    high = np.array([float(row["max_ns"]) / 1e6 for row in rows])
    axes[1].bar(np.arange(4) + offset, med, width=0.3, color=color, label=label)
    axes[1].errorbar(
        np.arange(4) + offset,
        med,
        yerr=[med - low, high - med],
        fmt="none",
        color=INK,
        elinewidth=0.7,
        capsize=2,
    )
axes[1].set_xticks(range(4), ["Exact", "Noise only", "35% outliers", "60% outliers"])
axes[1].set_ylabel("Milliseconds / median and range")
axes[1].legend(frameon=False)
save(fig, "trials_and_runtime")

inputs = [
    DATA / "inputs/outliers_35.json",
    DATA / "models/outliers_35__all_point_dlt.json",
    DATA / "models/outliers_35__ransac.json",
]
files = sorted([*OUT.glob("*.png"), *FIG.glob("*.png"), *FIG.glob("*.svg")])
metadata = {
    "numpy": np.__version__,
    "matplotlib": matplotlib.__version__,
    "pillow": pillow_version,
    "input_sha256": {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    },
    "output_sha256": {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    },
    "valid_destination_pixels": coverage,
    "source_policy": "Owned procedural plane graphic, no external image or feature detector. Correspondences are generated independently of the texture.",
    "warp_policy": "Inverse pixel-center mapping with bilinear interpolation; zero/out-of-support regions rendered dark",
}
(OUT / "provenance.json").write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
)
print(OUT)
