"""Detect and track race bibs in a video.

Detection runs every ``--detect-every`` frames. A smaller interval finds more
runners and costs proportionally more detector time, because the tracker can only
follow the bibs an earlier scan located.

Full sample video:
    uv run python bib_detection.py --input sample1/video.mp4 --output-dir runs/full

First 150 frames:
    uv run python bib_detection.py --max-frames 150 --output-dir runs/short
"""

import json
import logging
import re
from datetime import datetime
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Annotated, TypedDict

import cv2 as cv
import typer
from attrs import evolve, frozen

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
BIB_CONFIDENCE = 0.1
BIB_MODEL = detector.DetectorConfig(
    str(ROOT / "bibobj/RBNR_custom-yolov4-tiny-detector.cfg"),
    str(ROOT / "bibobj/RBNR_custom-yolov4-tiny-detector_best.weights"),
    ("bib",),
    input_size=BIB_INPUT_SIZE,
    confidence=BIB_CONFIDENCE,
)
DIGIT_MODEL = detector.DetectorConfig(
    str(ROOT / "bibobj/SVHN_custom-yolov4-tiny-detector.cfg"),
    str(ROOT / "bibobj/SVHN_custom-yolov4-tiny-detector_best.weights"),
    tuple(str(number) for number in range(10)),
)
BIB_DETECTOR = "yolo"
DIGIT_READER = "yolo"


@frozen
class RunConfig:
    """One run's knobs: schedule, locator tuning, bib pattern and implementations.

    ``bib_detector`` and ``digit_reader`` name entries in ``detector.BIB_DETECTORS`` and
    ``detector.BIB_READERS``, so switching a backend is a registry entry plus a name here.
    """

    detect_every: int = 20
    read_every: int = 10
    max_frames: int | None = None
    input_size: int = BIB_INPUT_SIZE
    confidence: float = BIB_CONFIDENCE
    bib_pattern: str = detector.BIB_PATTERN
    bib_detector: str = BIB_DETECTOR
    digit_reader: str = DIGIT_READER

    def __attrs_post_init__(self) -> None:
        if self.detect_every < 1 or self.read_every < 1 or (self.max_frames is not None and self.max_frames < 1):
            raise ValueError("frame intervals and max_frames must be positive")
        try:
            re.compile(self.bib_pattern)
        except re.error as error:
            raise ValueError(f"invalid bib pattern: {self.bib_pattern!r}") from error
        if self.bib_detector not in detector.BIB_DETECTORS:
            raise ValueError(f"unknown bib detector: {self.bib_detector!r}")
        if self.digit_reader not in detector.BIB_READERS:
            raise ValueError(f"unknown digit reader: {self.digit_reader!r}")


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
    confidence: float
    bib_pattern: str
    bib_detector: str
    digit_reader: str
    detection_frames: list[int]
    tracks: list[TrackResult]
    seconds: Timings


def detect_bibs(
    frame: cv.typing.MatLike,
    model: detector.DetectorConfig,
    config: RunConfig,
    timings: Timings | None = None,
) -> list[BibDetection]:
    """Find bib boxes and read their numbers in one video frame."""
    started = perf_counter()
    boxes = detector.BIB_DETECTORS[config.bib_detector](frame, model)
    if timings is not None:
        timings["detection"] += perf_counter() - started

    started = perf_counter()
    bibs = detector.read_bibs(frame, boxes, DIGIT_MODEL, config.bib_pattern, config.digit_reader)
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


def process_video(
    input_path: Path,
    output_dir: Path,
    config: RunConfig | None = None,
) -> Summary:
    """Track bibs through input_path and write an annotated video and summary.

    Detection runs on every ``config.detect_every`` frame. Every frame is still followed
    by optical flow, so the interval only decides how often a fresh bib box is located
    for a runner the tracker does not have yet. ``config`` also holds the locator tuning,
    the bib pattern, and the detector and reader implementations to use.
    """
    config = config or RunConfig()
    bib_model = evolve(BIB_MODEL, input_size=config.input_size, confidence=config.confidence)

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
    tracker = Tracker(config.bib_pattern, detector.BIB_READERS[config.digit_reader])

    try:
        while config.max_frames is None or processed_frames < config.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id = processed_frames

            stage_started = perf_counter()
            tracker.follow(frame, frame_id)
            timings["optical_flow"] += perf_counter() - stage_started

            scheduled = frame_id % config.detect_every == 0
            if scheduled:
                detections = detect_bibs(frame, bib_model, config, timings)
                detection_frames.append(frame_id)
                tracker.correct(frame, detections, frame_id)

            if frame_id % config.read_every == 0 and tracker.active_tracks:
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
        "detect_every": config.detect_every,
        "read_every": config.read_every,
        "input_size": config.input_size,
        "confidence": config.confidence,
        "bib_pattern": config.bib_pattern,
        "bib_detector": config.bib_detector,
        "digit_reader": config.digit_reader,
        "detection_frames": detection_frames,
        "tracks": tracker.results(fps),
        "seconds": timings,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Processed {processed_frames} frames, found {len(tracker.tracks)} tracks in {timings['total']:.2f}s")
    return summary


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(  # noqa: PLR0913, PLR0917 -- a CLI exposes one argument per tunable knob
    input_path: Annotated[Path, typer.Option("--input", help="Video file to process.")] = ROOT / "sample1/video.mp4",
    output_dir: Annotated[
        Path | None, typer.Option(help="Where to write the results (default: runs/<timestamp>).")
    ] = None,
    detect_every: Annotated[int, typer.Option(min=1, help="Run the bib detector every N frames.")] = 20,
    read_every: Annotated[int, typer.Option(min=1, help="Try reading track digits every N frames.")] = 10,
    max_frames: Annotated[int | None, typer.Option(min=1, help="Stop after this many frames.")] = None,
    input_size: Annotated[
        int, typer.Option(help="Square size frames are squashed to for the locator.")
    ] = BIB_INPUT_SIZE,
    confidence: Annotated[float, typer.Option(help="Score below which a bib detection is dropped.")] = BIB_CONFIDENCE,
    bib_pattern: Annotated[
        str, typer.Option(help="Regex a reading must match to count as a bib number.")
    ] = detector.BIB_PATTERN,
    bib_detector: Annotated[
        str, typer.Option(help=f"Bib box detector to use ({', '.join(sorted(detector.BIB_DETECTORS))}).")
    ] = BIB_DETECTOR,
    digit_reader: Annotated[
        str, typer.Option(help=f"Digit reader to use ({', '.join(sorted(detector.BIB_READERS))}).")
    ] = DIGIT_READER,
) -> None:
    """Run bib detection and tracking on a video."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = output_dir or ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    process_video(
        input_path,
        output_dir,
        RunConfig(
            detect_every=detect_every,
            read_every=read_every,
            max_frames=max_frames,
            input_size=input_size,
            confidence=confidence,
            bib_pattern=bib_pattern,
            bib_detector=bib_detector,
            digit_reader=digit_reader,
        ),
    )


if __name__ == "__main__":
    app()
