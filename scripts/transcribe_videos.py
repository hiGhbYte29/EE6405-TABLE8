#!/usr/bin/env python3
"""Transcribe EE6405 lecture videos with timestamped JSON and text output."""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import Iterable


def add_conda_cuda_library_paths() -> None:
    """Expose CUDA libraries bundled in the active conda env (no install needed)."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    nvidia_root = Path(conda_prefix) / "lib/python3.11/site-packages/nvidia"
    library_dirs = [path / "lib" for path in nvidia_root.iterdir()] if nvidia_root.is_dir() else []
    existing = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if os.environ.get("LD_LIBRARY_PATH") else []
    paths = [str(path) for path in library_dirs if path.is_dir()]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(paths + existing))


add_conda_cuda_library_paths()

from faster_whisper import WhisperModel

LOG = logging.getLogger("transcribe_videos")
VIDEO_EXTENSIONS = {".mp4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/raw/EE6405_Raw_Dataset/02_Lecture_Videos"),
        help="Root directory containing W01..W07 video folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/transcripts"),
        help="Directory for one .json and one .txt per video.",
    )
    parser.add_argument("--model", default="large-v3", help="faster-whisper model name.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type (for example float16 or int8_float16).",
    )
    parser.add_argument("--language", default="en", help="Language code, or omit for detection.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Regenerate outputs that already exist.")
    parser.add_argument("--video", type=Path, help="Process only this video path.")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("--no-auto-downgrade", action="store_true", help="Fail on CUDA OOM instead of retrying with int8_float16.")
    return parser.parse_args()


def iter_videos(video_dir: Path) -> Iterable[Path]:
    yield from sorted(
        (p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )


def lecture_metadata(video: Path, video_dir: Path) -> tuple[str, str, str]:
    relative = video.relative_to(video_dir)
    week_match = re.search(r"W\d+", str(relative), re.IGNORECASE)
    week = week_match.group(0).upper() if week_match else video.parent.name
    lecture_match = re.search(r"[_-]L(\d+)", video.stem, re.IGNORECASE)
    lecture = f"L{int(lecture_match.group(1)):02d}" if lecture_match else "L01"
    return str(relative), week, lecture


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def available_vram_mb() -> int | None:
    """Return free CUDA memory in MiB, or None when nvidia-smi is unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return min(values) if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def load_model(args: argparse.Namespace, compute_type: str) -> WhisperModel:
    LOG.info("Loading model=%s device=%s compute_type=%s", args.model, args.device, compute_type)
    return WhisperModel(
        args.model,
        device=args.device,
        compute_type=compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )


def transcribe_one(model: WhisperModel, video: Path, args: argparse.Namespace) -> tuple[dict, str]:
    relative, week, lecture = lecture_metadata(video, args.video_dir)
    LOG.info("Transcribing %s", relative)
    segments, info = model.transcribe(
        str(video),
        language=args.language or None,
        beam_size=args.beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=True,
        word_timestamps=False,
    )
    segment_list = []
    text_lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        start = float(segment.start)
        end = float(segment.end)
        segment_list.append({"start": start, "end": end, "text": text})
        text_lines.append(f"[{timestamp(start)}] {text}")

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    detected_language = getattr(info, "language", None) or args.language
    payload = {
        "source": relative,
        "week": week,
        "lecture": lecture,
        "duration": duration,
        "model": args.model,
        "language": detected_language,
        "segments": segment_list,
    }
    return payload, "\n".join(text_lines) + ("\n" if text_lines else "")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    if not args.video_dir.is_dir():
        LOG.error("Video directory does not exist: %s", args.video_dir)
        return 2
    videos = list(iter_videos(args.video_dir))
    if args.video is not None:
        requested = args.video if args.video.is_absolute() else (args.video_dir / args.video)
        requested = requested.resolve()
        videos = [video for video in videos if video.resolve() == requested]
        if not videos:
            LOG.error("Requested video was not found under %s: %s", args.video_dir, args.video)
            return 2
    if not videos:
        LOG.error("No mp4 videos found under %s", args.video_dir)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for video in videos:
        json_path = args.output_dir / f"{video.stem}.json"
        text_path = args.output_dir / f"{video.stem}.txt"
        if not args.force and json_path.exists() and text_path.exists():
            LOG.info("Skipping existing %s", video.name)
            continue
        pending.append((video, json_path, text_path))
    LOG.info("Found %d videos; %d pending", len(videos), len(pending))
    if not pending:
        return 0

    compute_type = args.compute_type
    free_vram = available_vram_mb() if args.device == "cuda" else None
    if free_vram is not None:
        LOG.info("CUDA free memory before model load: %d MiB", free_vram)
        if compute_type == "float16" and free_vram < 6000:
            LOG.warning("Only %d MiB CUDA memory is free; selecting int8_float16 before loading", free_vram)
            compute_type = "int8_float16"
    try:
        model = load_model(args, compute_type)
    except (RuntimeError, ValueError) as error:
        if args.no_auto_downgrade or args.device != "cuda" or compute_type == "int8_float16":
            raise
        LOG.warning("Could not load %s/%s (%s); retrying with int8_float16", args.model, compute_type, error)
        compute_type = "int8_float16"
        model = load_model(args, compute_type)

    for index, (video, json_path, text_path) in enumerate(pending, 1):
        LOG.info("[%d/%d] %s", index, len(pending), video.name)
        try:
            payload, plain_text = transcribe_one(model, video, args)
        except RuntimeError as error:
            if args.no_auto_downgrade or args.device != "cuda" or compute_type == "int8_float16" or "out of memory" not in str(error).lower():
                raise
            LOG.warning("CUDA memory error; retrying %s with int8_float16", video.name)
            del model
            compute_type = "int8_float16"
            model = load_model(args, compute_type)
            payload, plain_text = transcribe_one(model, video, args)
        payload["compute_type"] = compute_type
        atomic_write(json_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        atomic_write(text_path, plain_text)
        LOG.info("Wrote %s and %s (%d segments, %.1fs)", json_path, text_path, len(payload["segments"]), payload["duration"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
