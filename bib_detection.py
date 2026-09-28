"""Detect and track race bibs in a video.

Full sample video:
    uv run python bib_detection.py --input sample1/video.mp4 --output-dir runs/full

First 150 frames:
    uv run python bib_detection.py --max-frames 150 --output-dir runs/short
"""

import argparse
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import cv2 as cv
from loguru import logger

import detector
from detector import BibDetection
from tracker import Track, Tracker, TrackResult

ROOT = Path(__file__).resolve().parent
BIB_MODEL = detector.DetectorConfig(
    str(ROOT / "bibobj/RBNR_custom-yolov4-tiny-detector.cfg"),
    str(ROOT / "bibobj/RBNR_custom-yolov4-tiny-detector_best.weights"),
    ("bib",),
)
DIGIT_MODEL = detector.DetectorConfig(
    str(ROOT / "bibobj/SVHN_custom-yolov4-tiny-detector.cfg"),
    str(ROOT / "bibobj/SVHN_custom-yolov4-tiny-detector_best.weights"),
    tuple(str(number) for number in range(10)),
)


class Timings(TypedDict):
    detection: float
    optical_flow: float
    digit_reading: float
    video_writing: float
    total: float


class Summary(TypedDict):
    input: str
    fps: float
    processed_frames: int
    detect_every: int
    read_every: int
    detection_runs: int
    detection_frames: list[int]
    tracks: list[TrackResult]
    seconds: Timings


def detect_bibs(frame: cv.typing.MatLike, timings: Timings | None = None) -> list[BibDetection]:
    """Find bib boxes and read their numbers in one video frame."""
    started = perf_counter()
    boxes = detector.find_bibs(frame, BIB_MODEL)
    if timings is not None:
        timings["detection"] += perf_counter() - started

    started = perf_counter()
    bibs = detector.read_bibs(frame, boxes, DIGIT_MODEL)
    if timings is not None:
        timings["digit_reading"] += perf_counter() - started
    return bibs


def draw_tracks(frame: cv.typing.MatLike, tracks: list[Track]) -> None:
    """Draw active track boxes and bib readings on frame."""
    for track in tracks:
        x, y, width, height = (int(value) for value in track.bbox)
        label = f"T{track.track_id} bib:{track.best_bib or '?'}"
        if track.conflicting:
            label += " conflict"
        cv.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)
        cv.putText(
            frame,
            label,
            (x, max(20, y - 8)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )


def _open_video(input_path: Path, output_dir: Path) -> tuple[cv.VideoCapture, cv.VideoWriter, float]:
    capture = cv.VideoCapture(str(input_path))
    writer: cv.VideoWriter | None = None
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {input_path}")
        fps = capture.get(cv.CAP_PROP_FPS)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("Video must have a positive FPS")

        output_dir.mkdir(parents=True, exist_ok=True)
        video_path = output_dir / "annotated.mp4"
        summary_path = output_dir / "summary.json"
        if video_path.exists() or summary_path.exists():
            raise FileExistsError(f"Output directory already contains results: {output_dir}")

        size = (
            int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
        writer = cv.VideoWriter(str(video_path), cv.VideoWriter.fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise OSError(f"Cannot write video in: {output_dir}")
        return capture, writer, fps
    except Exception:
        capture.release()
        if writer is not None:
            writer.release()
        raise


def process_video(
    input_path: Path,
    output_dir: Path,
    detect_every: int = 100,
    read_every: int = 10,
    max_frames: int | None = None,
) -> Summary:
    """Track bibs through input_path and write an annotated video and summary."""
    if detect_every < 1 or read_every < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("frame intervals and max_frames must be positive")

    started = perf_counter()
    capture, writer, fps = _open_video(input_path, output_dir)
    timings: Timings = {
        "detection": 0.0,
        "optical_flow": 0.0,
        "digit_reading": 0.0,
        "video_writing": 0.0,
        "total": 0.0,
    }
    processed_frames = 0
    detection_frames: list[int] = []
    tracker = Tracker()

    try:
        while max_frames is None or processed_frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id = processed_frames

            stage_started = perf_counter()
            tracker.follow(frame, frame_id)
            timings["optical_flow"] += perf_counter() - stage_started

            scheduled = frame_id % detect_every == 0
            if scheduled:
                detections = detect_bibs(frame, timings)
                detection_frames.append(frame_id)
                tracker.correct(frame, detections, frame_id)

            if frame_id % read_every == 0 and tracker.active_tracks:
                stage_started = perf_counter()
                tracker.read(frame, detector.get_detector(DIGIT_MODEL), frame_id)
                timings["digit_reading"] += perf_counter() - stage_started

            stage_started = perf_counter()
            draw_tracks(frame, tracker.active_tracks)
            writer.write(frame)
            timings["video_writing"] += perf_counter() - stage_started
            processed_frames += 1
    finally:
        capture.release()
        writer.release()

    timings["total"] = perf_counter() - started
    summary: Summary = {
        "input": str(input_path.resolve()),
        "fps": fps,
        "processed_frames": processed_frames,
        "detect_every": detect_every,
        "read_every": read_every,
        "detection_runs": len(detection_frames),
        "detection_frames": detection_frames,
        "tracks": tracker.results(fps),
        "seconds": timings,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Processed {processed_frames} frames, found {len(tracker.tracks)} tracks in {timings['total']:.2f}s")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "sample1/video.mp4")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/full")
    parser.add_argument("--detect-every", type=int, default=100)
    parser.add_argument("--read-every", type=int, default=10)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    process_video(
        args.input,
        args.output_dir,
        args.detect_every,
        args.read_every,
        args.max_frames,
    )


if __name__ == "__main__":
    main()
    #  uv run python bib_detection.py
