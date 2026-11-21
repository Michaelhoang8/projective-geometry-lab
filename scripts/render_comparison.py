"""Crossfade fixed fitted warps against the known synthetic target; not an optimizer trace."""

import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import matplotlib.font_manager
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/media"
OUT.mkdir(parents=True, exist_ok=True)
IMAGES = ROOT / "docs/images"
WIDTH, HEIGHT, FPS, SECONDS = 1200, 660, 30, 8
PAPER = "#f2f4f7"
INK = "#192a45"
BLUE = "#3155d6"
CORAL = "#d2573b"
MUTED = "#647087"
font_path = matplotlib.font_manager.findfont("DejaVu Sans")
fonts = {size: ImageFont.truetype(font_path, size) for size in [15, 18, 23, 39]}
with (ROOT / "results/published/summary.csv").open() as stream:
    summary = list(csv.DictReader(stream))
reference = Image.open(IMAGES / "reference.png").convert("RGB")
fitted = {
    name: Image.open(IMAGES / f"{name}.png").convert("RGB")
    for name in ["all_point_dlt", "ransac"]
}
ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    raise RuntimeError("ffmpeg with libx264 support must be on PATH")
command = [
    ffmpeg,
    "-y",
    "-loglevel",
    "error",
    "-f",
    "rawvideo",
    "-pixel_format",
    "rgb24",
    "-video_size",
    f"{WIDTH}x{HEIGHT}",
    "-framerate",
    str(FPS),
    "-i",
    "-",
    "-an",
    "-c:v",
    "libx264",
    "-crf",
    "22",
    "-preset",
    "medium",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(OUT / "fit-comparison.mp4"),
]
process = subprocess.Popen(command, stdin=subprocess.PIPE)
try:
    for frame in range(FPS * SECONDS):
        alpha = 0.5 - 0.5 * math.cos(2 * math.pi * frame / (2 * FPS))
        canvas = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
        draw = ImageDraw.Draw(canvas)
        draw.text((40, 22), "Homography fit comparison", font=fonts[39], fill=INK)
        draw.text(
            (42, 78),
            "35% synthetic outliers / fixed fitted matrices / same known target",
            font=fonts[18],
            fill=MUTED,
        )
        for x0, name, label, color in [
            (40, "all_point_dlt", "All-point DLT", CORAL),
            (648, "ransac", "RANSAC + consensus refit", BLUE),
        ]:
            error = float(
                next(
                    row
                    for row in summary
                    if row["case"] == "outliers_35" and row["method"] == name
                )["held_clean_rmse_px"]
            )
            draw.text((x0, 119), label, font=fonts[23], fill=color)
            canvas.paste(Image.blend(reference, fitted[name], alpha), (x0, 158))
            draw.text(
                (x0, 590), f"Held-out RMSE {error:.3f} px", font=fonts[18], fill=color
            )
        draw.text(
            (40, 628),
            f"Reference {100*(1-alpha):.0f}% / fitted warp {100*alpha:.0f}%   |   comparison crossfade, not optimization progress",
            font=fonts[15],
            fill=MUTED,
        )
        process.stdin.write(canvas.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode comparison")
except BaseException:
    process.kill()
    process.wait()
    raise
inputs = [
    IMAGES / "reference.png",
    IMAGES / "all_point_dlt.png",
    IMAGES / "ransac.png",
    ROOT / "results/published/summary.csv",
]
metadata = {
    "duration_seconds": SECONDS,
    "frames": FPS * SECONDS,
    "fps": FPS,
    "width": WIDTH,
    "height": HEIGHT,
    "audio": False,
    "policy": "Four two-second cosine crossfades between known target and each fixed fitted warp; not live estimation or an iteration trace",
    "input_sha256": {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs
    },
    "output_sha256": hashlib.sha256(
        (OUT / "fit-comparison.mp4").read_bytes()
    ).hexdigest(),
    "renderer_sha256_lf": hashlib.sha256(
        Path(__file__).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest(),
    "pillow": pillow_version,
    "ffmpeg": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
}
(OUT / "comparison.json").write_text(
    json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n"
)
print(OUT / "fit-comparison.mp4")
