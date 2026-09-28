"""Tests for the direct video-processing loop."""

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

    def detect(_frame):
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


def test_lost_track_triggers_one_immediate_recovery_scan(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=7, disappear_after=0)
    scan_frames = []

    def detect(_frame):
        frame_id = len(scan_frames)
        scan_frames.append(frame_id)
        if len(scan_frames) == 1:
            return [BibDetection((10, 10, 40, 20), "12", 0.9)]
        return []

    monkeypatch.setattr(bib_detection, "detect_bibs", detect)
    summary = bib_detection.process_video(source, tmp_path / "out", detect_every=5)

    assert summary["detection_frames"] == [0, 1, 5]
    assert summary["recovery_runs"] == 1


def test_summary_contains_simple_stage_timings(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=2)
    monkeypatch.setattr(bib_detection, "detect_bibs", lambda _frame: [])

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
    monkeypatch.setattr(bib_detection, "detect_bibs", lambda _frame: [])

    summary = bib_detection.process_video(source, tmp_path / "out", max_frames=3)

    assert summary["processed_frames"] == 3
    assert count_frames(tmp_path / "out/annotated.mp4") == 3
