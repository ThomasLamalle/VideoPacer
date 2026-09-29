"""Detect and track race bibs in a video.

Detection runs every ``--detect-every`` frames. A smaller interval finds more
runners and costs proportionally more detector time, because the tracker can only
follow the bibs an earlier scan located.

Full sample video:
    uv run python bib_detection.py --input sample1/video.mp4 --output-dir runs/full

First 150 frames:
    uv run python bib_detection.py --max-frames 150 --output-dir runs/short
"""

import argparse
import json
import logging
import re
from datetime import datetime
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import cv2 as cv
from attrs import evolve

import detector
from detector import BibDetection
from tracker import Track, Tracker, TrackResult

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
# The locator cfg declares a 416x416 input and keeps its anchors in absolute pixels
# for that input, so squashing 1920x1080 down to 416 leaves a bib at ~14x18 px, below
# the smallest anchor either head can match. Running the same weights at 832 puts a
# bib at ~28x36 px and locates runners up to 7.5s earlier. Recall peaks between 768
# and 896 and falls off on both sides, so this is a plateau and not a lucky value.
BIB_INPUT_SIZE = 832
BIB_MODEL = detector.DetectorConfig(
    str(ROOT / "bibobj/RBNR_custom-yolov4-tiny-detector.cfg"),
    str(ROOT / "bibobj/RBNR_custom-yolov4-tiny-detector_best.weights"),
    ("bib",),
    input_size=BIB_INPUT_SIZE,
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
    input_size: int
    bib_pattern: str
    detection_frames: list[int]
    tracks: list[TrackResult]
    seconds: Timings


def detect_bibs(
    frame: cv.typing.MatLike,
    model: detector.DetectorConfig,
    timings: Timings | None = None,
    bib_pattern: str = detector.BIB_PATTERN,
) -> list[BibDetection]:
    """Find bib boxes and read their numbers in one video frame."""
    started = perf_counter()
    boxes = detector.find_bibs(frame, model)
    if timings is not None:
        timings["detection"] += perf_counter() - started

    started = perf_counter()
    bibs = detector.read_bibs(frame, boxes, DIGIT_MODEL, bib_pattern)
    if timings is not None:
        timings["digit_reading"] += perf_counter() - started
    return bibs


def draw_tracks(frame: cv.typing.MatLike, tracks: list[Track], frame_id: int) -> None:
    """Draw track boxes in the color of the event on this frame."""
    for track in tracks:
        detected = track.last_detection_frame == frame_id
        read = track.last_read_frame == frame_id
        event, color = (
            ("BOTH", (255, 0, 255))
            if detected and read
            else ("READ", (0, 255, 0))
            if read
            else ("DETECT", (0, 0, 255))
            if detected
            else ("FLOW", (255, 255, 0))
        )
        x, y, width, height = (int(value) for value in track.bbox)
        label = f"T{track.track_id} {event} bib:{track.best_bib or '?'}"
        if track.conflicting:
            label += " conflict"
        cv.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv.putText(frame, label, (x, max(20, y - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


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


# The run options are independent knobs rather than parts of one object yet; a model
# registry will absorb them (see features_todo.md).
def process_video(  # noqa: PLR0913, PLR0917
    input_path: Path,
    output_dir: Path,
    detect_every: int = 20,
    read_every: int = 10,
    max_frames: int | None = None,
    input_size: int = BIB_INPUT_SIZE,
    bib_pattern: str = detector.BIB_PATTERN,
) -> Summary:
    """Track bibs through input_path and write an annotated video and summary.

    Detection runs on every ``detect_every`` frame. Every frame is still followed
    by optical flow, so the interval only decides how often a fresh bib box is
    located for a runner the tracker does not have yet. ``input_size`` sets the
    square size frames are squashed to for the locator; larger finds small bibs
    earlier and costs quadratically. ``bib_pattern`` is the regex a reading must
    match to count as a bib number; a box that reads no bib number starts no track.
    """
    if detect_every < 1 or read_every < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("frame intervals and max_frames must be positive")
    try:
        re.compile(bib_pattern)
    except re.error as error:
        raise ValueError(f"invalid bib pattern: {bib_pattern!r}") from error
    bib_model = evolve(BIB_MODEL, input_size=input_size)

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
    tracker = Tracker(bib_pattern)

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
                detections = detect_bibs(frame, bib_model, timings, bib_pattern)
                detection_frames.append(frame_id)
                tracker.correct(frame, detections, frame_id)

            if frame_id % read_every == 0 and tracker.active_tracks:
                stage_started = perf_counter()
                tracker.read(frame, detector.get_detector(DIGIT_MODEL), frame_id)
                timings["digit_reading"] += perf_counter() - stage_started

            stage_started = perf_counter()
            draw_tracks(frame, tracker.active_tracks, frame_id)
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
        "input_size": input_size,
        "bib_pattern": bib_pattern,
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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--detect-every", type=int, default=20)
    parser.add_argument("--read-every", type=int, default=10)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--input-size",
        type=int,
        default=BIB_INPUT_SIZE,
        help="Square size frames are squashed to for the locator (default: %(default)s)",
    )
    parser.add_argument(
        "--bib-pattern",
        default=detector.BIB_PATTERN,
        help="Regex a digit reading must match to count as a bib number (default: %(default)s)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    process_video(
        args.input,
        output_dir,
        args.detect_every,
        args.read_every,
        args.max_frames,
        args.input_size,
        args.bib_pattern,
    )


if __name__ == "__main__":
    main()
    #  uv run python bib_detection.py
