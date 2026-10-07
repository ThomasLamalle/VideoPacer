"""Golden check on the bundled full video and its bib ground truth."""

import csv
import logging
from pathlib import Path

import pytest

from bib_detection import RunConfig, process_video

ROOT = Path(__file__).resolve().parents[1]


def test_full_video_matches_current_golden_and_reports_ground_truth(caplog: pytest.LogCaptureFixture) -> None:
    """Run the real models without writing an annotated video or summary file."""
    with (ROOT / "sample1/ground_truth.csv").open(newline="", encoding="utf-8") as file:
        expected = {row["bib"] for row in csv.DictReader(file)}

    with caplog.at_level(logging.INFO):
        summary = process_video(ROOT / "sample1/video.mp4", None, RunConfig())
    found = {track["best_bib"] for track in summary["tracks"] if track["best_bib"]}

    assert summary["processed_frames"] == 523
    assert found == expected
    assert "Ground truth: 13/13 bibs found" in caplog.text
    assert "Missing: none" in caplog.text
    assert "Extra: none" in caplog.text
