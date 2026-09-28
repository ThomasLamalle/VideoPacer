"""Exercise decoding, scheduling, timestamps and saved video with a tiny clip."""

import json

import cv2 as cv
import numpy as np
import pytest

import bib_detection
from detector import BibDetection


def make_video(path, frames=7):
    writer = cv.VideoWriter(str(path), cv.VideoWriter.fourcc(*"mp4v"), 10, (96, 64))
    assert writer.isOpened()
    for frame_id in range(frames):
        frame = np.full((64, 96, 3), 10 * frame_id, dtype=np.uint8)
        frame[10:30, 10:50] = np.random.default_rng(4).integers(
            80, 255, (20, 40, 3), dtype=np.uint8
        )
        writer.write(frame)
    writer.release()


def test_processes_every_frame_but_detects_only_on_schedule(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source)
    calls = []

    def detect(frame, frame_id, _timings=None):
        calls.append((frame_id, float(frame.mean())))
        return [BibDetection(frame_id, (10, 10, 40, 20), "0012", 0.9)]

    monkeypatch.setattr(bib_detection, "detect_bibs", detect)
    summary = bib_detection.process_video(source, tmp_path / "out", detect_every=3)
    assert [index for index, _ in calls] == [0, 3, 6]
    assert calls[0][1] < calls[1][1] < calls[2][1]
    assert summary["processed_frames"] == 7
    assert summary["detector_runs"] == 3
    records = [
        json.loads(line) for line in (tmp_path / "out/observations.jsonl").read_text().splitlines()
    ]
    assert [row["frame_id"] for row in records] == list(range(7))
    assert records[6]["timestamp_seconds"] == pytest.approx(0.6)
    assert records[1]["tracks"][0]["source"] == "flow"
    assert records[1]["tracks"][0]["reading_ran"] is False
    assert summary["tracks"][0]["votes"] == {"0012": 3}
    assert summary["tracks"][0]["last_seen_seconds"] == pytest.approx(0.6)
    cap = cv.VideoCapture(str(tmp_path / "out/annotated.mp4"))
    assert cap.get(cv.CAP_PROP_FPS) == pytest.approx(10)
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == 7


def test_frame_limit_and_empty_scans(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source)
    monkeypatch.setattr(bib_detection, "detect_bibs", lambda *_args: [])
    summary = bib_detection.process_video(source, tmp_path / "out", detect_every=3, max_frames=4)
    assert summary["processed_frames"] == 4
    assert summary["detector_runs"] == 2
    assert summary["tracks"] == []


@pytest.mark.parametrize("interval", [0, -1])
def test_invalid_interval_rejected_before_creating_outputs(tmp_path, interval):
    with pytest.raises(ValueError, match="positive"):
        bib_detection.process_video(tmp_path / "missing.mp4", tmp_path / "out", interval)
    assert not (tmp_path / "out").exists()


def test_reads_flow_crops_between_full_frame_scans(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source)
    monkeypatch.setattr(
        bib_detection,
        "detect_bibs",
        lambda _frame, frame_id, _times: [BibDetection(frame_id, (10, 10, 40, 20), "0012", 0.9)],
    )
    monkeypatch.setattr(bib_detection.detector, "get_detector", lambda _config: None)
    monkeypatch.setattr(bib_detection.detector, "read_bib", lambda *_args: "0012")
    summary = bib_detection.process_video(
        source, tmp_path / "out", detect_every=100, read_every=2, save_video=False
    )
    assert summary["detector_runs"] == 1
    assert summary["tracks"][0]["votes"] == {"0012": 4}
    assert summary["tracks"][0]["last_detection_frame"] == 0
    assert not (tmp_path / "out/annotated.mp4").exists()


def test_flow_failure_triggers_early_detection_with_retry_limit(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=25)
    scans = []

    def detect(_frame, frame_id, _timings):
        scans.append(frame_id)
        # Empty-texture box cannot be followed, requesting re-detection.
        return [BibDetection(frame_id, (60, 35, 20, 20), "12", 0.9)]

    monkeypatch.setattr(bib_detection, "detect_bibs", detect)
    summary = bib_detection.process_video(source, tmp_path / "out", detect_every=100)
    assert scans == [0, 10, 20]
    assert summary["lost_tracks"] == 3
    assert summary["recovery_scans"] == 2


def test_empty_recovery_scan_retries_while_another_track_survives(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source, frames=25)
    scans = []

    def detect(_frame, frame_id, _timings):
        scans.append(frame_id)
        if frame_id:
            return []
        return [
            BibDetection(0, (60, 35, 20, 20), "12", 0.9),
            BibDetection(0, (10, 10, 40, 20), "34", 0.9),
        ]

    monkeypatch.setattr(bib_detection, "detect_bibs", detect)
    summary = bib_detection.process_video(
        source, tmp_path / "out", detect_every=100, read_every=100
    )
    assert scans == [0, 10, 20]
    assert summary["tracks"][1]["last_seen_frame"] == 24
