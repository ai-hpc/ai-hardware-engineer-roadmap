"""Encode/decode a synthetic clip on CPU and verify pixels, count, and timing."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


WIDTH, HEIGHT, FRAMES, FPS = 64, 48, 12, 30


def fixture() -> bytes:
    return bytes(220 if (x - frame * 3) % WIDTH < 12 else 30 + y
                 for frame in range(FRAMES) for y in range(HEIGHT) for x in range(WIDTH))


def validate(reference: bytes, decoded: bytes, probe: dict, count: int = FRAMES) -> dict:
    if len(reference) != WIDTH * HEIGHT * count or len(decoded) != len(reference):
        raise ValueError("missing, extra, or wrongly shaped decoded pixels")
    stream = probe["streams"][0]
    if (stream["width"], stream["height"]) != (WIDTH, HEIGHT):
        raise ValueError("decoded stream geometry changed")
    frames = probe["frames"]
    if len(frames) != count or int(stream["nb_read_frames"]) != count:
        raise ValueError("frame count changed")
    base = Fraction(stream["time_base"])
    pts = [int(frame["best_effort_timestamp"]) * base for frame in frames]
    if pts != [Fraction(index, FPS) for index in range(count)]:
        raise ValueError("presentation timestamps changed")
    squared_error = sum((a - b) ** 2 for a, b in zip(reference, decoded))
    mse = squared_error / len(reference)
    return {"frames": count, "mse_gray_8bit": mse,
            "psnr_gray_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "lossless": mse == 0, "max_absolute_pixel_error": max(abs(a-b) for a,b in zip(reference,decoded)),
            "pts_seconds": [float(value) for value in pts]}


def run(directory: Path) -> dict:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("this optional exercise needs FFmpeg and ffprobe")
    directory.mkdir(parents=True, exist_ok=True)
    source, encoded, decoded = (directory / name for name in ("source.gray", "clip.mp4", "decoded.gray"))
    reference = fixture()
    source.write_bytes(reference)
    commands = [
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
         "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", str(source),
         "-frames:v", str(FRAMES), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
         "-crf", "23", "-bf", "0", "-threads", "1", str(encoded)],
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(encoded), "-threads", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", str(decoded)],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    probe_command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                     "-show_streams", "-show_frames", "-of", "json", str(encoded)]
    probe = json.loads(subprocess.check_output(probe_command, text=True, timeout=30))
    report = validate(reference, decoded.read_bytes(), probe)
    report.update({"ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                   "encoded_bytes": encoded.stat().st_size,
                   "reference_sha256": hashlib.sha256(reference).hexdigest(),
                   "encoded_sha256": hashlib.sha256(encoded.read_bytes()).hexdigest(),
                   "commands": commands + [probe_command],
                   "boundary": "synthetic CPU libx264 fixture; no Jetson encoder or latency claim"})
    (directory / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output_dir), indent=2))
