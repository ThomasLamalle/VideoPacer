"""Detect bibs periodically, track every frame, and read the tracked image crops.

Run these commands from the repository directory (requires ``uv sync``):

    # Process the FULL sample video; omitting --max-frames reads until EOF.
    uv run python bib_detection.py --input sample1/video.mp4

    # Full video, explicit intervals and output directory (use a fresh directory).
    uv run python bib_detection.py --input sample1/video.mp4 --output-dir runs/full-video

    # First 150 frames, approximately five seconds of the sample, with more scans.
    uv run python bib_detection.py --detect-every 10 --max-frames 150

    # Full video without annotation/encoding; still save observations and votes.
    uv run python bib_detection.py --input sample1/video.mp4 --no-video

Defaults: detect every 100 frames, read tracked crops every 10 frames, and save
to a new timestamped directory under runs/. Recovery scans can run between the
periodic scans. Every detector scan also reads digits, regardless of --read-every.
Outputs are annotated.mp4 (unless --no-video), observations.jsonl and summary.json.
Existing output files are never overwritten. Run with --help for all options.
"""

import argparse
import json
import platform
from datetime import datetime
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2 as cv
from attrs import asdict, define
from loguru import logger

import detector
from detector import BibDetection
from optical_flow import BibTracker, FlowTrack

__all__ = ["BibDetection", "detect_bibs", "display_bib_detections", "process_video"]

ROOT = Path(__file__).resolve().parent
BIB_DETECTOR_CONFIG = detector.DetectorConfig(
    cfg=str(ROOT / "bibobj/RBNR_custom-yolov4-tiny-detector.cfg"),
    weights=str(ROOT / "bibobj/RBNR_custom-yolov4-tiny-detector_best.weights"),
    classes=("bib",),
)
DIGIT_READER_CONFIG = detector.DetectorConfig(
    cfg=str(ROOT / "bibobj/SVHN_custom-yolov4-tiny-detector.cfg"),
    weights=str(ROOT / "bibobj/SVHN_custom-yolov4-tiny-detector_best.weights"),
    classes=tuple(str(digit) for digit in range(10)),
)


@define
class ScanScheduler:
    """Decide when the slower full-frame detector should run."""

    detect_every: int
    retry_every: int
    last_scan: int
    recovery_pending: bool = False

    @classmethod
    def create(cls, detect_every: int) -> ScanScheduler:
        return cls(detect_every, min(10, detect_every), -detect_every)

    def request_recovery(self) -> None:
        """Ask for an early scan because a track failed or no track exists."""
        self.recovery_pending = True

    def reason(self, frame_id: int) -> str | None:
        """Return ``periodic``, ``recovery``, or None for this frame."""
        if frame_id % self.detect_every == 0:
            return "periodic"
        if self.recovery_pending and frame_id - self.last_scan >= self.retry_every:
            return "recovery"
        return None

    def record_result(self, frame_id: int, detections: list[BibDetection]) -> None:
        """Remember a scan; retry recovery only while the result stays empty."""
        self.last_scan = frame_id
        self.recovery_pending = self.recovery_pending and not detections


def detect_bibs(
    frame: Any, frame_id: int, timings: dict[str, float] | None = None
) -> list[BibDetection]:
    """Keep located bibs even when the digit reader returns no reading."""
    return detector.get_rbns(frame, BIB_DETECTOR_CONFIG, DIGIT_READER_CONFIG, frame_id, timings)


def _draw_box(frame: Any, bbox: detector.BBox, label: str, color: tuple[int, int, int]) -> None:
    """Clip drawing coordinates without changing the tracked box."""
    x, y, w, h = bbox
    if not all(isfinite(value) for value in bbox) or w <= 0 or h <= 0:
        return
    height, width = frame.shape[:2]
    x1, y1 = max(0, min(width - 1, int(x))), max(0, min(height - 1, int(y)))
    x2, y2 = max(0, min(width - 1, int(x + w))), max(0, min(height - 1, int(y + h)))
    if x2 <= x1 or y2 <= y1:
        return
    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    position = (x1, min(height - 5, max(20, y1 - 8)))
    cv.putText(frame, label, position, cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv.putText(frame, label, position, cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def display_bib_detections(frame: Any, bib_detections: list[BibDetection]) -> Any:
    """Still-image helper, also useful when inspecting raw detector output."""
    annotated = frame.copy()
    for detection in bib_detections:
        _draw_box(annotated, detection.bbox, detection.bib_string or "?", (0, 255, 0))
    return annotated


def annotate_tracks(
    frame: Any, tracks: list[FlowTrack], tracker: BibTracker, frame_id: int, fps: float
) -> Any:
    """Return an annotated copy with track IDs, provisional readings and scan age.

    Green means a detector matched this frame; cyan means accepted optical flow.
    The age measures time since detector confirmation, not time since digit reading.
    """
    annotated = frame.copy()
    for track in tracks:
        history = tracker.histories[track.track_id]
        age = (frame_id - history.last_detection_frame) / fps
        label = f"T{track.track_id} bib:{history.provisional_bib or '?'} {track.source} {age:.1f}s"
        if history.conflicting:
            label += " conflict"
        color = (0, 255, 0) if track.source == "detection" else (255, 255, 0)
        _draw_box(annotated, track.bbox, label, color)
    cv.rectangle(annotated, (0, 0), (min(850, frame.shape[1] - 1), 36), (0, 0, 0), -1)
    cv.putText(
        annotated,
        f"Frame {frame_id} | {frame_id / fps:.2f}s | green: detected / cyan: optical flow",
        (10, 25),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return annotated


def _read_tracked_bibs(frame: Any, tracker: BibTracker, timings: dict[str, float]) -> None:
    """Read each active track that was not already read by detection this frame."""
    tracks_to_read = [track for track in tracker.tracks if not track.reading_ran]
    if not tracks_to_read:
        return
    number_reader = detector.get_detector(DIGIT_READER_CONFIG)
    for track in tracks_to_read:
        reading = detector.read_bib(frame, track.bbox, number_reader, timings)
        tracker.record_reading(track.track_id, reading)


def _observation_record(
    frame_id: int,
    fps: float,
    scan_reason: str | None,
    detections: list[BibDetection] | None,
    tracker: BibTracker,
) -> dict[str, Any]:
    """Build one record; None means detection was skipped and [] means empty."""
    return {
        "frame_id": frame_id,
        "timestamp_seconds": frame_id / fps,
        "detector_ran": detections is not None,
        "scan_reason": scan_reason,
        "detections": [asdict(detection) for detection in detections]
        if detections is not None
        else None,
        "tracks": [track.snapshot() for track in tracker.tracks],
    }


def process_video(  # noqa: PLR0915, PLR0913, PLR0912 -- keep sequential video processing together
    video_path: Path,
    output_dir: Path,
    detect_every: int = 100,
    max_frames: int | None = None,
    *,
    read_every: int = 10,
    save_video: bool = True,
) -> dict[str, Any]:
    """Process a video sequentially and return the same summary saved to JSON.

    Args:
        video_path: Input video readable by OpenCV.
        output_dir: Destination for JSONL observations, JSON summary and optional
            annotated video. Existing files with these names cause an error.
        detect_every: Full-frame scan interval in source frames, starting at zero.
            Recovery can trigger extra scans after min(10, detect_every) frames.
        max_frames: Maximum decoded frames, or None to process the entire video.
        read_every: Interval for extra digit reads on active tracked crops. Scans
            always read digits; a track is read at most once on a given frame.
        save_video: Whether to annotate and encode every processed source frame.

    Timestamps use the source frame clock (frame_id / FPS), suitable for this
    constant-frame-rate file pipeline. First/last seen include accepted optical
    flow; last detector confirmation is recorded separately. Neither is a finish
    time. Stage timings accumulate wall-clock seconds; detector_total overlaps
    the detection/read stages and must not be added to them.

    Raises:
        ValueError: An interval/limit or the source FPS is invalid.
        OSError: Input/output cannot be opened, or an output file already exists.
    """
    if min(detect_every, read_every) < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("detect_every, read_every and max_frames must be positive")
    started = perf_counter()
    cap = cv.VideoCapture(str(video_path))
    writer = None
    timings = dict.fromkeys(("decode", "bib_detection", "digit_reading", "tracking", "output"), 0.0)
    detector_seconds = 0.0
    processed_frames = detector_runs = recovery_scans = 0
    scheduler = ScanScheduler.create(detect_every)
    try:
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video_path}")
        fps = cap.get(cv.CAP_PROP_FPS)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("Video must provide a positive source FPS")
        width, height = (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
        tracker = BibTracker()
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("annotated.mp4", "observations.jsonl", "summary.json"):
            if (output_dir / name).exists():
                raise FileExistsError(f"Output already exists: {output_dir / name}")
        if save_video:
            writer = cv.VideoWriter(
                str(output_dir / "annotated.mp4"),
                cv.VideoWriter.fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise OSError(f"Cannot open video writer in {output_dir}")

        with (output_dir / "observations.jsonl").open("w", encoding="utf-8") as observations:
            while max_frames is None or processed_frames < max_frames:
                stage_start = perf_counter()
                ok, frame = cap.read()
                timings["decode"] += perf_counter() - stage_start
                if not ok:
                    break
                frame_id = processed_frames
                # Move boxes into this frame before matching fresh detections to them.
                stage_start = perf_counter()
                flow_failed = tracker.advance(frame, frame_id)
                timings["tracking"] += perf_counter() - stage_start
                if flow_failed or not tracker.tracks:
                    scheduler.request_recovery()
                scan_reason = scheduler.reason(frame_id)
                detections = None
                # None means no scan; [] means the detector ran and found no bibs.
                if scan_reason is not None:
                    stage_start = perf_counter()
                    detections = detect_bibs(frame, frame_id, timings)
                    detector_seconds += perf_counter() - stage_start
                    detector_runs += 1
                    recovery_scans += int(scan_reason == "recovery")
                    stage_start = perf_counter()
                    tracker.correct(detections)
                    timings["tracking"] += perf_counter() - stage_start
                    scheduler.record_result(frame_id, detections)
                if frame_id % read_every == 0:
                    _read_tracked_bibs(frame, tracker, timings)
                stage_start = perf_counter()
                if writer is not None:
                    writer.write(annotate_tracks(frame, tracker.tracks, tracker, frame_id, fps))
                record = _observation_record(frame_id, fps, scan_reason, detections, tracker)
                observations.write(json.dumps(record) + "\n")
                timings["output"] += perf_counter() - stage_start
                processed_frames += 1
    finally:
        cap.release()
        if writer is not None:
            stage_start = perf_counter()
            writer.release()
            timings["output"] += perf_counter() - stage_start

    elapsed = perf_counter() - started
    histories = []
    for history in tracker.histories.values():
        record = asdict(history)
        record.update(
            first_seen_seconds=history.first_seen_frame / fps,
            last_seen_seconds=history.last_seen_frame / fps,
            last_detection_seconds=history.last_detection_frame / fps,
            provisional_bib=history.provisional_bib,
            conflicting=history.conflicting,
        )
        histories.append(record)
    summary = {
        "input": str(video_path.resolve()),
        "source_fps": fps,
        "resolution": [width, height],
        "detect_every": detect_every,
        "read_every": read_every,
        "retry_every": scheduler.retry_every,
        "max_detection_gap": tracker.max_detection_gap,
        "save_video": save_video,
        "max_frames": max_frames,
        "processed_frames": processed_frames,
        "detector_runs": detector_runs,
        "recovery_scans": recovery_scans,
        "lost_tracks": tracker.lost_tracks,
        "digit_crops": int(timings.pop("digit_crops", 0)),
        "elapsed_seconds": elapsed,
        "processing_fps": processed_frames / elapsed,
        "video_seconds_per_wall_second": processed_frames / fps / elapsed,
        "stage_seconds": timings,
        "detector_total_seconds_including_model_load": detector_seconds,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "opencv": cv.__version__,
            "tracker": "OpenCV sparse Lucas-Kanade optical flow",
            "opencv_threads": cv.getNumThreads(),
        },
        "tracks": histories,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Saved {} frames, {} detector runs, {} tracks in {:.2f}s to {}",
        processed_frames,
        detector_runs,
        len(histories),
        elapsed,
        output_dir,
    )
    return summary


def main() -> None:
    """Parse CLI options, choose a fresh default output directory and run the video."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "sample1/video.mp4",
        help="Input video (default: sample1/video.mp4)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: a new timestamped directory under runs/)",
    )
    parser.add_argument(
        "--detect-every",
        type=int,
        default=100,
        help="Periodic bib scan interval in frames; recovery may scan sooner (default: 100)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Process only the first N frames (default: the full video)",
    )
    parser.add_argument(
        "--read-every",
        type=int,
        default=10,
        help="Extra tracked-crop reading interval in frames (default: 10)",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Skip annotation/encoding for timing"
    )
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    try:
        process_video(
            args.input,
            output_dir,
            args.detect_every,
            args.max_frames,
            read_every=args.read_every,
            save_video=not args.no_video,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
