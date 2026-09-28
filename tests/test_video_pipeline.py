"""Tests for the direct video-processing loop."""

from pathlib import Path
from time import sleep

import cv2 as cv
import numpy as np
import pytest

import bib_detection
from detector import BibDetection


def make_video(path, frames: int = 7, disappear_after: int | None = None) -> None:
    writer = cv.VideoWriter(str(path), cv.VideoWriter.fourcc(*"mp4v"), 10, (96, 64))
    assert writer.isOpened()
    texture = np.random.default_rng(4).integers(80, 255, (20, 40, 3), dtype=np.uint8)
    for frame_id in range(frames):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        if disappear_after is None or frame_id <= disappear_after:
            frame[10:30, 10:50] = texture
        writer.write(frame)
    writer.release()


def count_frames(path) -> int:
    capture = cv.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def test_detects_on_schedule_and_writes_every_frame(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source)
    scans = []

    def detect(_frame, _model, _timings=None, _bib_pattern=None):
        scans.append(len(scans))
        return [BibDetection((10, 10, 40, 20), "0012", 0.9)]

    monkeypatch.setattr(bib_detection, "detect_bibs", detect)
    summary = bib_detection.process_video(source, tmp_path / "out", detect_every=3)

    assert len(scans) == 3
    assert summary["processed_frames"] == 7
    assert summary["detection_runs"] == 3
    assert summary["tracks"][0]["best_bib"] == "0012"
    assert summary["tracks"][0]["first_seconds"] == 0
    assert summary["tracks"][0]["last_seconds"] == pytest.approx(0.6)
    assert count_frames(tmp_path / "out/annotated.mp4") == 7


def test_detects_only_on_schedule(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=25)
    monkeypatch.setattr(bib_detection, "detect_bibs", lambda _frame, _model, _timings=None, _bib_pattern=None: [])

    summary = bib_detection.process_video(source, tmp_path / "out", detect_every=10)

    assert summary["detection_frames"] == [0, 10, 20]


def test_an_active_track_does_not_change_the_schedule(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=25)
    monkeypatch.setattr(
        bib_detection,
        "detect_bibs",
        lambda _frame, _model, _timings=None, _bib_pattern=None: [BibDetection((10, 10, 40, 20), "12", 0.9)],
    )

    summary = bib_detection.process_video(source, tmp_path / "out", detect_every=10)

    assert summary["detection_frames"] == [0, 10, 20]


def test_summary_contains_simple_stage_timings(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=2)
    monkeypatch.setattr(bib_detection, "detect_bibs", lambda _frame, _model, _timings=None, _bib_pattern=None: [])

    summary = bib_detection.process_video(source, tmp_path / "out")

    timing_names = {
        "detection",
        "optical_flow",
        "digit_reading",
        "video_writing",
        "total",
    }
    assert set(summary["seconds"]) == timing_names
    assert summary["seconds"]["detection"] >= 0
    assert summary["seconds"]["optical_flow"] >= 0
    assert summary["seconds"]["digit_reading"] >= 0
    assert summary["seconds"]["video_writing"] >= 0
    assert summary["seconds"]["total"] >= 0


def test_max_frames_limits_the_output(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=7)
    monkeypatch.setattr(bib_detection, "detect_bibs", lambda _frame, _model, _timings=None, _bib_pattern=None: [])

    summary = bib_detection.process_video(source, tmp_path / "out", max_frames=3)

    assert summary["processed_frames"] == 3
    assert count_frames(tmp_path / "out/annotated.mp4") == 3


def test_reads_a_surviving_track_that_detection_misses(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=2)
    scans = 0

    def detect(_frame, _model, _timings=None, _bib_pattern=None):
        nonlocal scans
        scans += 1
        return [BibDetection((10, 10, 40, 20), "0012", 0.9)] if scans == 1 else []

    class Reader:
        def detect(self, _image, _confidence):
            digits = ("0", "0", "1", "2")
            return [
                bib_detection.detector.Detection(digit, (index, 0, 1, 1), 0.9) for index, digit in enumerate(digits)
            ]

    monkeypatch.setattr(bib_detection, "detect_bibs", detect)
    monkeypatch.setattr(bib_detection.detector, "get_detector", lambda _config: Reader())

    summary = bib_detection.process_video(source, tmp_path / "out", detect_every=1, read_every=1)

    assert summary["tracks"][0]["votes"] == {"0012": 2}


def test_pipeline_applies_the_configured_bib_pattern(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=2)
    monkeypatch.setattr(
        bib_detection,
        "detect_bibs",
        lambda _frame, _model, _timings=None, _bib_pattern=None: [BibDetection((10, 10, 40, 20), "0012", 0.9)],
    )

    class Reader:
        def detect(self, _image, _confidence):
            return [bib_detection.detector.Detection("A", (0, 0, 1, 1), 0.9)]

    monkeypatch.setattr(bib_detection.detector, "get_detector", lambda _config: Reader())

    summary = bib_detection.process_video(source, tmp_path / "out", detect_every=2, read_every=1, bib_pattern="[A-Z]")

    assert summary["bib_pattern"] == "[A-Z]"
    assert summary["tracks"][0]["votes"] == {"0012": 1, "A": 1}


def test_setup_error_releases_the_open_capture(tmp_path, monkeypatch):
    class Capture:
        released = False

        def isOpened(self):  # noqa: N802 - OpenCV uses this method name.
            return True

        def get(self, _property):
            return 10.0

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(bib_detection.cv, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError()))

    with pytest.raises(PermissionError):
        bib_detection.process_video(tmp_path / "source.mp4", tmp_path / "out")

    assert capture.released


def test_detector_ocr_has_its_own_timing(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=1)
    monkeypatch.setattr(
        bib_detection.detector,
        "find_bibs",
        lambda _frame, _config: [bib_detection.detector.Detection("bib", (10, 10, 40, 20), 0.9)],
    )

    def read_bibs(_frame, _boxes, _config, _bib_pattern=None):
        sleep(0.002)
        return [BibDetection((10, 10, 40, 20), "12", 0.9)]

    monkeypatch.setattr(bib_detection.detector, "read_bibs", read_bibs)

    summary = bib_detection.process_video(source, tmp_path / "out")

    assert summary["seconds"]["digit_reading"] >= 0.002
