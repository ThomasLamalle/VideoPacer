"""Tests for optical-flow tracks and bib votes."""

import numpy as np
import pytest

from detector import BibDetection, Detection
from tracker import Tracker


def frame(x: int = 20) -> np.ndarray:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    texture = np.random.default_rng(4).integers(40, 255, (20, 35, 3), dtype=np.uint8)
    image[20:40, x : x + 35] = texture
    return image


def test_optical_flow_moves_a_track():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "0012", 0.9)], 0)

    lost = tracker.follow(frame(23), 1)

    assert not lost
    assert tracker.active_tracks[0].bbox[:2] == pytest.approx((23, 20), abs=1)
    assert tracker.active_tracks[0].last_frame == 1


def test_failed_flow_deactivates_the_track():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), None, 0.9)], 0)

    assert tracker.follow(np.zeros_like(frame()), 1)
    assert tracker.active_tracks == []
    assert tracker.tracks[0].last_frame == 0


def test_detection_updates_an_existing_track_and_its_votes():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "12", 0.9)], 0)
    tracker.follow(frame(22), 1)

    tracker.correct(frame(22), [BibDetection((22, 20, 35, 20), "12", 0.2)], 1)

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].votes == {"12": 2}
    assert tracker.tracks[0].last_detection_frame == 1


def test_votes_report_ties_and_conflicts():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "12", 0.9)], 0)

    class Reader:
        def detect(self, _crop, _confidence):
            return [Detection("13", (0, 0, 2, 8), 0.9)]

    reader = Reader()

    tracker.read(frame(), reader)

    track = tracker.tracks[0]
    assert track.best_bib is None
    assert track.conflicting
    assert track.votes == {"12": 1, "13": 1}


def test_results_include_best_bib_and_visible_times():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "007", 0.9)], 0)
    tracker.follow(frame(22), 1)

    assert tracker.results(fps=10) == [
        {
            "track_id": 0,
            "best_bib": "007",
            "votes": {"007": 1},
            "conflicting": False,
            "first_frame": 0,
            "last_frame": 1,
            "first_seconds": 0.0,
            "last_seconds": 0.1,
        }
    ]
