"""Track known image motion with the real OpenCV optical-flow implementation."""

import cv2 as cv
import numpy as np
import pytest

from detector import BibDetection
from optical_flow import BibTracker


def moving_bib(x=20, y=20):
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    patch = np.random.default_rng(7).integers(40, 255, (30, 40, 3), dtype=np.uint8)
    frame[y : y + 30, x : x + 40] = patch
    return frame


def make_tracker():
    return BibTracker()


def test_follows_image_motion_without_new_detections_or_votes():
    tracker = make_tracker()
    tracker.advance(moving_bib(), 0)
    tracker.correct([BibDetection(0, (20, 20, 40, 30), "0012", 0.9)])
    first_id = tracker.tracks[0].track_id
    for frame_id in range(1, 8):
        assert not tracker.advance(moving_bib(20 + 2 * frame_id, 20 + frame_id), frame_id)
        track = tracker.tracks[0]
        assert track.track_id == first_id
        assert track.bbox[:2] == pytest.approx((20 + 2 * frame_id, 20 + frame_id), abs=1)
        assert track.source == "flow"
        assert not track.reading_ran
    history = tracker.histories[first_id]
    assert history.votes == {"0012": 1}
    assert history.last_detection_frame == 0
    assert history.last_seen_frame == 7


def test_detector_refresh_preserves_id_and_counts_actual_readings():
    tracker = make_tracker()
    tracker.advance(moving_bib(), 0)
    tracker.correct([BibDetection(0, (20, 20, 40, 30), "0012", 0.9)])
    track_id = tracker.tracks[0].track_id
    tracker.advance(moving_bib(22), 1)
    tracker.record_reading(track_id, "0013")
    assert tracker.histories[track_id].provisional_bib is None
    assert tracker.histories[track_id].conflicting
    tracker.advance(moving_bib(24), 2)
    tracker.correct([BibDetection(2, (24, 20, 40, 30), "0012", 0.9)])
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == track_id
    assert tracker.histories[track_id].votes == {"0012": 2, "0013": 1}
    assert tracker.histories[track_id].last_detection_frame == 2


def test_disappearance_stops_tracking_and_requests_detection():
    tracker = make_tracker()
    tracker.advance(moving_bib(), 0)
    tracker.correct([BibDetection(0, (20, 20, 40, 30), None, 0.9)])
    assert tracker.advance(np.zeros((100, 160, 3), dtype=np.uint8), 1)
    assert tracker.tracks == []
    assert tracker.lost_tracks == 1
    assert tracker.histories[0].last_seen_frame == 0


def test_flow_handles_moderate_scale_change():
    tracker = make_tracker()
    source = moving_bib()
    tracker.advance(source, 0)
    tracker.correct([BibDetection(0, (20, 20, 40, 30), "12", 0.9)])
    transform = cv.getRotationMatrix2D((40, 35), 0, 1.05)
    frame = cv.warpAffine(source, transform, (160, 100))
    assert not tracker.advance(frame, 1)
    assert tracker.tracks[0].bbox == pytest.approx((19, 19.25, 42, 31.5), abs=1)


def test_unreadable_detections_and_weak_recovery_keep_separate_ids():
    tracker = make_tracker()
    frame = moving_bib()
    frame[:, 80:] = moving_bib()[:, :80]
    tracker.advance(frame, 0)
    tracker.correct(
        [
            BibDetection(0, (20, 20, 40, 30), None, 0.9),
            BibDetection(0, (100, 20, 40, 30), "0012", 0.9),
        ]
    )
    assert len({track.track_id for track in tracker.tracks}) == 2
    tracker.advance(frame, 1)
    tracker.correct(
        [
            BibDetection(1, (100, 20, 40, 30), "0012", 0.2),
            BibDetection(1, (20, 20, 40, 30), "0012", 0.9),
        ]
    )
    assert tracker.histories[0].votes == {"0012": 1}
    assert tracker.histories[1].votes == {"0012": 2}


def test_blank_bib_is_not_followed_without_features():
    tracker = make_tracker()
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    tracker.advance(frame, 0)
    tracker.correct([BibDetection(0, (20, 20, 40, 30), "12", 0.9)])
    assert tracker.advance(frame, 1)
    assert not tracker.tracks


def test_duplicate_read_in_same_frame_does_not_add_votes():
    tracker = make_tracker()
    tracker.advance(moving_bib(), 0)
    tracker.correct([BibDetection(0, (20, 20, 40, 30), "12", 0.9)])
    with pytest.raises(ValueError, match="already read"):
        tracker.record_reading(0, "12")


def test_valid_flow_survives_empty_detector_scans():
    tracker = BibTracker()
    frame = moving_bib()
    tracker.advance(frame, 0)
    tracker.correct([BibDetection(0, (20, 20, 40, 30), "12", 0.9)])
    for frame_id in range(1, 32):
        tracker.advance(frame, frame_id)
        if frame_id % 10 == 0:
            tracker.correct([])
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].track_id == 0


def test_flow_still_has_a_bounded_lifetime_without_detector_confirmation():
    tracker = BibTracker()
    frame = moving_bib()
    tracker.advance(frame, 0)
    tracker.correct([BibDetection(0, (20, 20, 40, 30), "12", 0.9)])
    for frame_id in range(1, 202):
        tracker.advance(frame, frame_id)
    assert tracker.tracks == []
