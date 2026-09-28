"""Follow bib image features between detector scans; never extrapolate lost boxes."""

from collections import Counter
from typing import Any

import cv2 as cv
import numpy as np
from attrs import define, field

from detector import BBox, BibDetection

MIN_POINTS = 6
MAX_FORWARD_BACKWARD_ERROR = 1.5  # Pixels in the original-resolution image.
MIN_INLIER_FRACTION = 0.6
MIN_SCALE, MAX_SCALE = 0.8, 1.25  # Reject implausible one-frame size changes.
MATCH_IOU = 0.3
NEW_TRACK_CONFIDENCE = 0.5


def _features(gray: Any, bbox: BBox) -> np.ndarray:
    """Select strong image corners inside a bib box for Lucas-Kanade tracking.

    The two-pixel inset avoids choosing corners on the detector rectangle boundary
    or the surrounding background. An empty array is returned when the crop has no
    useful texture; such a track will fail cleanly on the following frame.
    """
    x, y, w, h = (int(value) for value in bbox)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    # Stay inside the bib instead of selecting corners on its background edge.
    mask[max(0, y + 2) : max(0, y + h - 2), max(0, x + 2) : max(0, x + w - 2)] = 255
    points = cv.goodFeaturesToTrack(
        gray, maxCorners=60, qualityLevel=0.01, minDistance=3, mask=mask
    )
    return points if points is not None else np.empty((0, 1, 2), dtype=np.float32)


def _consistent_points(
    previous: Any, current: Any, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return point pairs that survive a forward-and-backward flow check."""
    next_points, forward_ok, _ = cv.calcOpticalFlowPyrLK(
        previous, current, points, np.empty_like(points), winSize=(21, 21), maxLevel=3
    )
    if next_points is None:
        return None
    back_points, backward_ok, _ = cv.calcOpticalFlowPyrLK(
        current, previous, next_points, np.empty_like(next_points), winSize=(21, 21), maxLevel=3
    )
    if back_points is None:
        return None
    error = np.linalg.norm(points - back_points, axis=2).ravel()
    valid = (forward_ok.ravel() == 1) & (backward_ok.ravel() == 1)
    valid &= np.isfinite(error) & (error < MAX_FORWARD_BACKWARD_ERROR)
    old, new = points[valid].reshape(-1, 2), next_points[valid].reshape(-1, 2)
    if len(new) < MIN_POINTS:
        return None
    return old, new


def _estimate_motion(old: np.ndarray, new: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a robust translation/rotation/uniform-scale transform to point pairs."""
    transform, inliers = cv.estimateAffinePartial2D(
        old, new, method=cv.RANSAC, ransacReprojThreshold=2
    )
    if transform is None or inliers is None or not np.isfinite(transform).all():
        return None
    inlier_mask = inliers.ravel().astype(bool)
    if inlier_mask.sum() < MIN_POINTS or inlier_mask.mean() < MIN_INLIER_FRACTION:
        return None
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    if not MIN_SCALE <= scale <= MAX_SCALE:
        return None
    return transform, inlier_mask


def _move_box(
    bbox: BBox,
    transform: np.ndarray,
    new_points: np.ndarray,
    inliers: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[BBox, np.ndarray] | None:
    """Apply the fitted motion and retain only points inside the moved box."""
    scale = float(np.hypot(transform[0, 0], transform[1, 0]))
    x, y, w, h = bbox
    center = transform @ np.array([x + w / 2, y + h / 2, 1])
    w, h = w * scale, h * scale
    bbox = (float(center[0] - w / 2), float(center[1] - h / 2), w, h)
    x, y, w, h = bbox
    points = new_points[inliers]
    inside = (points[:, 0] >= max(0, x)) & (points[:, 0] < min(image_shape[1], x + w))
    inside &= (points[:, 1] >= max(0, y)) & (points[:, 1] < min(image_shape[0], y + h))
    points = points[inside]
    if len(points) < MIN_POINTS:
        return None
    return bbox, points.reshape(-1, 1, 2)


def _follow(previous: Any, current: Any, track: FlowTrack) -> tuple[BBox, np.ndarray] | None:
    """Move one track to the current frame, returning None when motion is unreliable.

    The three steps are: follow points in both directions, fit one robust motion,
    then move the box and discard points that left it.
    """
    if len(track.points) < MIN_POINTS:
        return None
    pairs = _consistent_points(previous, current, track.points)
    if pairs is None:
        return None
    old_points, new_points = pairs
    motion = _estimate_motion(old_points, new_points)
    if motion is None:
        return None
    transform, inliers = motion
    return _move_box(track.bbox, transform, new_points, inliers, current.shape)


def _iou(first: BBox, second: BBox) -> float:
    """Return intersection over union for two ``(x, y, width, height)`` boxes."""
    x, y, w, h = first
    other_x, other_y, other_w, other_h = second
    overlap_w = max(0, min(x + w, other_x + other_w) - max(x, other_x))
    overlap_h = max(0, min(y + h, other_y + other_h) - max(y, other_y))
    intersection = overlap_w * overlap_h
    union = w * h + other_w * other_h - intersection
    return intersection / union if union > 0 else 0.0


@define
class TrackHistory:
    """Evidence retained after an active track disappears.

    ``first_seen_frame`` and ``last_seen_frame`` include accepted optical-flow
    positions. ``last_detection_frame`` records the latest detector confirmation.
    Votes contain only real digit-model readings; optical flow never creates one.
    """

    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    last_detection_frame: int
    last_read_frame: int = -1
    reading_attempts: int = 0
    votes: Counter[str] = field(factory=Counter)

    @property
    def provisional_bib(self) -> str | None:
        """Return the unique most frequent reading, or None for no votes/a tie."""
        best = self.votes.most_common(2)
        if not best or (len(best) > 1 and best[0][1] == best[1][1]):
            return None
        return best[0][0]

    @property
    def conflicting(self) -> bool:
        """Report whether more than one distinct bib string has been read."""
        return len(self.votes) > 1


@define
class FlowTrack:
    """Current box and feature points for one temporary runner identity.

    ``source`` describes how the box was obtained on the current frame. The
    temporary ``track_id`` is deliberately independent from ``bib_string``.
    """

    track_id: int
    bbox: BBox
    points: np.ndarray
    source: str = "detection"
    bib_string: str | None = None
    reading_ran: bool = False

    def snapshot(self) -> dict[str, Any]:
        """Return the JSON-safe per-frame representation written to observations."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "source": self.source,
            "bib_string": self.bib_string,
            "reading_ran": self.reading_ran,
            "point_count": len(self.points),
        }


class BibTracker:
    """Track bib boxes with sparse optical flow and accumulate digit readings.

    Call :meth:`advance` exactly once for every consecutive source frame, starting
    at frame zero. When a detector runs, call :meth:`correct` afterward with its
    complete result, including an empty list. Additional crop readings can then be
    submitted through :meth:`record_reading`.

    A flow failure removes the active track immediately. The history and votes are
    retained, but a later detection starts a new temporary ID; identities are never
    reconstructed from equal bib text. Tracks also expire after
    ``max_detection_gap`` frames without a matching detector box.
    """

    def __init__(self, max_detection_gap: int = 200) -> None:
        if max_detection_gap < 1:
            raise ValueError("max_detection_gap must be positive")
        # Independent of scan frequency: a missed scan is not failed flow.
        self.max_detection_gap = max_detection_gap
        self.frame_id = -1
        self.gray: Any = None
        self.tracks: list[FlowTrack] = []
        self.histories: dict[int, TrackHistory] = {}
        self.lost_tracks = 0

    def advance(self, frame: Any, frame_id: int) -> bool:
        """Advance all active tracks by one frame.

        Returns True if at least one active track was dropped because flow failed
        or its detector-confirmation age exceeded ``max_detection_gap``. A true
        result lets the video loop request an earlier recovery scan.
        """
        if frame_id != self.frame_id + 1:
            raise ValueError("Optical flow needs consecutive frames starting at zero")
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        survivors = []
        lost = False
        for track in self.tracks:
            history = self.histories[track.track_id]
            followed = _follow(self.gray, gray, track)
            if followed is None or frame_id - history.last_detection_frame > self.max_detection_gap:
                self.lost_tracks += 1
                lost = True
                continue
            track.bbox, track.points = followed
            track.source, track.bib_string, track.reading_ran = "flow", None, False
            history.last_seen_frame = frame_id
            survivors.append(track)
        self.tracks, self.gray, self.frame_id = survivors, gray, frame_id
        return lost

    def correct(self, detections: list[BibDetection]) -> None:
        """Correct active boxes and initialize eligible unmatched detections.

        Candidate pairs are processed from highest to lowest IoU. Each track and
        detection is used at most once, with ``MATCH_IOU`` as the acceptance floor.
        Detector confidence does not block correction of an existing track, while
        a new track requires ``NEW_TRACK_CONFIDENCE``. Bib text never participates
        in identity matching.
        """
        if any(d.frame_id != self.frame_id for d in detections):
            raise ValueError("Detections must belong to the current frame")
        candidates = sorted(
            (
                (_iou(track.bbox, detection.bbox), ti, di)
                for ti, track in enumerate(self.tracks)
                for di, detection in enumerate(detections)
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for overlap, ti, di in candidates:
            if overlap < MATCH_IOU:
                break
            if ti in used_tracks or di in used_detections:
                continue
            self._correct_track(self.tracks[ti], detections[di])
            used_tracks.add(ti)
            used_detections.add(di)
        for di, detection in enumerate(detections):
            if di not in used_detections and detection.confidence >= NEW_TRACK_CONFIDENCE:
                track_id = len(self.histories)
                track = FlowTrack(track_id, detection.bbox, np.empty((0, 1, 2), dtype=np.float32))
                self.tracks.append(track)
                self.histories[track_id] = TrackHistory(
                    track_id, self.frame_id, self.frame_id, self.frame_id
                )
                self._correct_track(track, detection)

    def _correct_track(self, track: FlowTrack, detection: BibDetection) -> None:
        """Reset a matched track from a detector box and record its digit reading."""
        track.bbox = detection.bbox
        track.points = _features(self.gray, track.bbox)
        track.source = "detection"
        self.histories[track.track_id].last_detection_frame = self.frame_id
        self.record_reading(track.track_id, detection.bib_string)

    def record_reading(self, track_id: int, reading: str | None) -> None:
        """Record one real digit-model attempt for a track on the current frame.

        ``None`` counts as an attempt but not a vote. Strings are kept verbatim so
        leading zeroes survive. A second attempt on the same track and frame is an
        error, preventing detector OCR and scheduled crop OCR from voting twice.
        """
        history = self.histories[track_id]
        if history.last_read_frame == self.frame_id:
            raise ValueError("This track was already read on this frame")
        track = next(track for track in self.tracks if track.track_id == track_id)
        track.bib_string, track.reading_ran = reading, True
        history.last_read_frame = self.frame_id
        history.reading_attempts += 1
        if reading is not None:
            history.votes[reading] += 1
