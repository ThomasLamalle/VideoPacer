"""Detect and track race bibs in a video.

Detection runs every ``--detect-every`` frames. A smaller interval finds more
runners and costs proportionally more detector time, because the tracker can only
follow the bibs an earlier scan located.

Full sample video:
    uv run python bib_detection.py --input sample1/video.mp4 --output-dir runs/full

First 150 frames:
    uv run python bib_detection.py --max-frames 150 --output-dir runs/short
"""

import csv
import json
import logging
import platform
import re
import subprocess
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
# One row per full run, so each trial can be traced to the speed and recognition it gave.
HISTORY = ROOT / "performance_history.csv"
logger = logging.getLogger(__name__)
# The locator cfg declares a 416x416 input and keeps its anchors in absolute pixels
# for that input, so squashing 1920x1080 down to 416 leaves a bib at ~14x18 px, below
# the smallest anchor either head can match. Running the same weights at 832 puts a
# bib at ~28x36 px and locates runners up to 7.5s earlier. Recall peaks between 768
# and 896 and falls off on both sides, so this is a plateau and not a lucky value.
# The default YOLO26n detector also did best at 832 on sample1. With the OCR reader and
# no stretch, it found 12 of 13 runners, against 11 at 1024 and 10 at 1280, and it runs
# 2.5x faster than at 1280.
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
BIB_DETECTOR = "yolo26n-v025"
DIGIT_READER = "ppocr"


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


class GroundTruth(TypedDict):
    expected: int
    matched: list[str]
    missing: list[str]
    extra: list[str]


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
    ground_truth: GroundTruth | None


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


def _open_video(input_path: Path, output_dir: Path | None) -> tuple[cv.VideoCapture, cv.VideoWriter | None, float]:
    capture = cv.VideoCapture(str(input_path))
    writer: cv.VideoWriter | None = None
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {input_path}")
        fps = capture.get(cv.CAP_PROP_FPS)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("Video must have a positive FPS")

        if output_dir is None:
            return capture, None, fps

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


def log_ground_truth(input_path: Path, tracks: list[TrackResult]) -> GroundTruth | None:
    """Compare detected bibs with a ground_truth.csv beside the input video, if present, and log the result."""
    truth_path = input_path.parent / "ground_truth.csv"
    if not truth_path.is_file():
        return None
    with truth_path.open(newline="", encoding="utf-8") as file:
        expected = {row["bib"] for row in csv.DictReader(file)}
    found = {track["best_bib"] for track in tracks if track["best_bib"]}
    truth: GroundTruth = {
        "expected": len(expected),
        "matched": sorted(found & expected),
        "missing": sorted(expected - found),
        "extra": sorted(found - expected),
    }
    logger.info("Ground truth: %d/%d bibs found", len(truth["matched"]), truth["expected"])
    logger.info("Matched: %s", ", ".join(truth["matched"]) or "none")
    logger.info("Missing: %s", ", ".join(truth["missing"]) or "none")
    logger.info("Extra: %s", ", ".join(truth["extra"]) or "none")
    return truth


def _git_commit() -> str:
    """Return the short commit hash, marked "+changes" when Python files or dependencies differ from it."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    changed = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "*.py", "uv.lock"], cwd=ROOT, check=False)
    return f"{commit}+changes" if changed.returncode else commit


def _cpu() -> str:
    """Return the processor model. Python leaves it empty on Linux, so it is read from /proc/cpuinfo there."""
    cpuinfo = Path("/proc/cpuinfo")
    lines = cpuinfo.read_text().splitlines() if cpuinfo.is_file() else []
    names = [line.split(":", 1)[1].strip() for line in lines if line.startswith("model name")]
    return names[0] if names else platform.processor()


def _power_source() -> str:
    """Return "AC" or "battery" on a Linux laptop, and "" elsewhere. Runs on battery can take twice as long."""

    def read(path: Path) -> str:
        return path.read_text().strip() if path.is_file() else ""

    # A wireless mouse also reports itself "online", so only chargers count.
    chargers = [path for path in Path("/sys/class/power_supply").glob("*") if read(path / "type") in {"Mains", "USB"}]
    if not chargers:
        return ""
    return "AC" if any(read(path / "online") == "1" for path in chargers) else "battery"


def record_performance(summary: Summary, note: str, report_path: Path, history_path: Path) -> None:
    """Write a performance report for one run and add the same fields as a row of the performance history.

    ``note`` says what the run tries. ``other_s`` is the time outside the timed stages, mostly video decoding.
    """
    seconds = summary["seconds"]
    staged = seconds["detection"] + seconds["optical_flow"] + seconds["digit_reading"] + seconds["video_writing"]
    truth = summary["ground_truth"]
    video = Path(summary["input"])
    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "note": note,
        "video": f"{video.parent.name}/{video.name}",
        "frames": summary["processed_frames"],
        "x_realtime": round(summary["processed_frames"] / summary["fps"] / seconds["total"], 2),
        "total_s": round(seconds["total"], 1),
        "detection_s": round(seconds["detection"], 1),
        "optical_flow_s": round(seconds["optical_flow"], 1),
        "digit_reading_s": round(seconds["digit_reading"], 1),
        "video_writing_s": round(seconds["video_writing"], 1),
        "other_s": round(seconds["total"] - staged, 1),
        "found": f"{len(truth['matched'])}/{truth['expected']}" if truth else "",
        "missing": " ".join(truth["missing"]) if truth else "",
        "extra": " ".join(truth["extra"]) if truth else "",
        "detect_every": summary["detect_every"],
        "read_every": summary["read_every"],
        "input_size": summary["input_size"],
        "confidence": summary["confidence"],
        "bib_detector": summary["bib_detector"],
        "digit_reader": summary["digit_reader"],
        "bib_pattern": summary["bib_pattern"],
        "cpu": _cpu(),
        "power": _power_source(),
    }
    report = "\n".join(f"- {key}: {value}".rstrip() for key, value in row.items())
    report_path.write_text(f"# Performance report\n\n{report}\n", encoding="utf-8")
    # ponytail: the header is written once, so a change of columns needs a new history file.
    is_new = not history_path.is_file()
    with history_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def process_video(
    input_path: Path,
    output_dir: Path | None,
    config: RunConfig | None = None,
) -> Summary:
    """Track bibs through input_path; write outputs only when output_dir is set.

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
                # Seeding new tracking points belongs to the tracking cost, so it is counted with optical flow.
                stage_started = perf_counter()
                tracker.correct(frame, detections, frame_id)
                timings["optical_flow"] += perf_counter() - stage_started

            if frame_id % config.read_every == 0 and tracker.active_tracks:
                stage_started = perf_counter()
                tracker.read(frame, detector.get_detector(DIGIT_MODEL), frame_id)
                timings["digit_reading"] += perf_counter() - stage_started

            if writer is not None:
                stage_started = perf_counter()
                draw_tracks(frame, tracker.active_tracks, frame_id)
                writer.write(frame)
                timings["video_writing"] += perf_counter() - stage_started
            processed_frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    timings["total"] = perf_counter() - started
    summary: Summary = {
        "input": str(input_path.resolve()),
        "fps": fps,
        "processed_frames": processed_frames,
        "detect_every": config.detect_every,
        "read_every": config.read_every,
        "input_size": {"roboflow_2.0": 416, "rfdetr-large-t1": 640}.get(config.bib_detector, config.input_size),
        "confidence": config.confidence,
        "bib_pattern": config.bib_pattern,
        "bib_detector": config.bib_detector,
        "digit_reader": config.digit_reader,
        "detection_frames": detection_frames,
        "tracks": tracker.results(fps),
        "seconds": timings,
        "ground_truth": None,
    }
    logger.info(f"Processed {processed_frames} frames, found {len(tracker.tracks)} tracks in {timings['total']:.2f}s")
    summary["ground_truth"] = log_ground_truth(input_path, summary["tracks"])
    if output_dir is not None:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
        int, typer.Option(help="YOLO input size (Roboflow v5 and RF-DETR use fixed 416/640 inputs).")
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
    note: Annotated[str, typer.Option(help="What this run tries, saved in the performance history.")] = "",
) -> None:
    """Run bib detection and tracking on a video.

    A run over the whole video also writes performance.md beside its outputs and adds a row to
    performance_history.csv, so a trial can be compared with the ones before it.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = output_dir or ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary = process_video(
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
    # A shortened run is not comparable with the history, which holds whole videos only.
    if max_frames is None:
        record_performance(summary, note, output_dir / "performance.md", HISTORY)


if __name__ == "__main__":
    app()
