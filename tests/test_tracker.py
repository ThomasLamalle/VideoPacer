"""Tests for optical-flow tracks and bib votes."""

import cv2 as cv
import numpy as np
import pytest

import tracker as tracker_module
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

    tracker.follow(frame(23), 1)

    assert tracker.active_tracks[0].bbox[:2] == pytest.approx((23, 20), abs=1)
    assert tracker.active_tracks[0].last_frame == 1


def two_textures(top_x: int, bottom_x: int) -> np.ndarray:
    """Two textured patches, so each can move on its own."""
    image = np.zeros((80, 200, 3), dtype=np.uint8)
    rng = np.random.default_rng(4)
    image[5:25, top_x : top_x + 35] = rng.integers(40, 255, (20, 35, 3), dtype=np.uint8)
    image[45:65, bottom_x : bottom_x + 35] = rng.integers(40, 255, (20, 35, 3), dtype=np.uint8)
    return image


def test_tracks_sharing_one_flow_call_keep_their_own_motion():
    tracker = Tracker()
    tracker.follow(two_textures(20, 120), 0)
    boxes = [BibDetection((20, 5, 35, 20), "0012", 0.9), BibDetection((120, 45, 35, 20), "0034", 0.9)]
    tracker.correct(two_textures(20, 120), boxes, 0)

    tracker.follow(two_textures(23, 118), 1)

    first, second = tracker.active_tracks
    assert first.bbox[:2] == pytest.approx((23, 5), abs=1)
    assert second.bbox[:2] == pytest.approx((118, 45), abs=1)


@pytest.mark.parametrize("bbox", [(30, 40, 35, 20), (-5, -5, 30, 25), (380, 280, 40, 40), (500, 500, 10, 10)])
def test_corner_search_near_a_box_matches_a_whole_frame_search(bbox):
    gray = np.random.default_rng(7).integers(0, 255, (300, 400), dtype=np.uint8)
    x, y, width, height = bbox
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[max(0, y) : y + height, max(0, x) : x + width] = 255
    whole_frame = cv.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01, minDistance=3, mask=mask)

    found = tracker_module._find_points(gray, bbox)

    assert np.array_equal(found, whole_frame if whole_frame is not None else np.empty((0, 1, 2), np.float32))


def test_failed_flow_deactivates_the_track():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "0012", 0.9)], 0)

    tracker.follow(np.zeros_like(frame()), 1)

    assert tracker.active_tracks == []
    assert tracker.tracks[0].last_frame == 0


def test_unreadable_detection_does_not_start_a_track():
    tracker = Tracker()
    tracker.follow(frame(), 0)

    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), None, 0.1)], 0)

    assert tracker.tracks == []


def test_unreadable_detection_still_moves_an_existing_track():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "0012", 0.9)], 0)
    tracker.follow(frame(22), 1)

    tracker.correct(frame(22), [BibDetection((22, 20, 35, 20), None, 0.1)], 1)

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].votes == {"0012": pytest.approx(0.9)}
    assert tracker.tracks[0].last_detection_frame == 1
    assert tracker.tracks[0].last_read_frame == 0


def test_detection_updates_an_existing_track_and_its_votes():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "0012", 0.9)], 0)
    tracker.follow(frame(22), 1)

    tracker.correct(frame(22), [BibDetection((22, 20, 35, 20), "0012", 0.2)], 1)

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].votes == {"0012": pytest.approx(1.1)}
    assert tracker.tracks[0].last_detection_frame == 1
    assert tracker.tracks[0].last_read_frame == 1


def test_votes_are_weighted_by_reading_confidence():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "0012", 1.0)], 0)

    class Reader:
        def detect(self, _crop, _confidence):
            return [Detection("0013", (0, 0, 2, 8), 0.3)]

    reader = Reader()
    for frame_id in (1, 2, 3):
        tracker.follow(frame(22), frame_id)
        tracker.read(frame(22), reader, frame_id)

    track = tracker.tracks[0]
    assert track.votes["0013"] == pytest.approx(0.9)
    assert track.best_bib == "0012"
    assert track.last_read_frame == 3


def test_a_longer_bib_outweighs_a_few_shorter_votes_but_not_many():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "1234", 1.0)], 0)
    track = tracker.tracks[0]

    track.add_vote("12345", 0.3)
    assert track.best_bib == "12345"

    track.add_vote("1234", 0.6)
    assert track.best_bib == "1234"


def test_votes_report_ties_and_conflicts():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "0012", 0.9)], 0)

    class Reader:
        def detect(self, _crop, _confidence):
            return [Detection("0013", (0, 0, 2, 8), 0.9)]

    reader = Reader()

    tracker.follow(frame(22), 1)
    tracker.read(frame(22), reader, 1)

    track = tracker.tracks[0]
    assert track.best_bib == "0012"
    assert track.conflicting
    assert track.votes == {"0012": pytest.approx(0.9), "0013": pytest.approx(0.9)}


def test_a_reading_that_does_not_match_the_bib_pattern_does_not_vote():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "0012", 0.9)], 0)

    class Reader:
        def detect(self, _crop, _confidence):
            return [Detection("3", (0, 0, 2, 8), 0.9)]

    tracker.follow(frame(22), 1)
    tracker.read(frame(22), Reader(), 1)

    assert tracker.tracks[0].votes == {"0012": pytest.approx(0.9)}


def test_a_tracker_can_be_told_another_bib_pattern():
    tracker = Tracker(bib_pattern=r"\d{2}")
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "12", 0.9)], 0)

    class Reader:
        def detect(self, _crop, _confidence):
            return [Detection("3", (0, 0, 2, 8), 0.9), Detection("4", (3, 0, 2, 8), 0.9)]

    tracker.follow(frame(22), 1)
    tracker.read(frame(22), Reader(), 1)

    assert tracker.tracks[0].votes == {"12": pytest.approx(0.9), "34": pytest.approx(0.9)}


def test_results_include_best_bib_and_visible_times():
    tracker = Tracker()
    tracker.follow(frame(), 0)
    tracker.correct(frame(), [BibDetection((20, 20, 35, 20), "007", 0.9)], 0)
    tracker.follow(frame(22), 1)

    assert tracker.results(fps=10) == [
        {
            "track_id": 0,
            "best_bib": "007",
            "votes": {"007": pytest.approx(0.9)},
            "conflicting": False,
            "first_frame": 0,
            "last_frame": 1,
            "first_seconds": 0.0,
            "last_seconds": 0.1,
        }
    ]
