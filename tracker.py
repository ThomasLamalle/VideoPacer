"""Follow bib boxes between detector frames and collect bib readings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import cv2 as cv
import numpy as np
from attrs import define, field

from detector import BIB_PATTERN, BIB_READERS, BBox, BibDetection, BibReading, DetectorLike

MIN_POINTS = 6
MAX_FLOW_ERROR = 1.5
MATCH_IOU = 0.3
# A corner score looks at a few pixels around each pixel (gradient, scoring window and peak picking), so the
# corner search runs on the box plus this many pixels. Scoring a whole 1080p frame for one box cost about 55 ms.
CORNER_CONTEXT = 4


class TrackResult(TypedDict):
    track_id: int
    best_bib: str | None
    votes: dict[str, float]
    conflicting: bool
    first_frame: int
    last_frame: int
    first_seconds: float
    last_seconds: float


def _find_points(gray: cv.typing.MatLike, bbox: BBox) -> np.ndarray:
    """Return up to 50 trackable corners inside a box, in frame coordinates.

    The search runs on a small window around the box, with a mask that keeps the corners inside the box. The
    window keeps enough real pixels around the box that the corners are the same as a search of the whole frame.
    """
    x, y, width, height = (int(value) for value in bbox)
    image_height, image_width = gray.shape[:2]
    left, top = max(0, x), max(0, y)
    right, bottom = min(image_width, x + width), min(image_height, y + height)
    if right <= left or bottom <= top:
        return np.empty((0, 1, 2), dtype=np.float32)
    window_left, window_top = max(0, left - CORNER_CONTEXT), max(0, top - CORNER_CONTEXT)
    window_right, window_bottom = min(image_width, right + CORNER_CONTEXT), min(image_height, bottom + CORNER_CONTEXT)
    window = gray[window_top:window_bottom, window_left:window_right]
    mask = np.zeros(window.shape, dtype=np.uint8)
    mask[top - window_top : bottom - window_top, left - window_left : right - window_left] = 255
    points = cv.goodFeaturesToTrack(window, maxCorners=50, qualityLevel=0.01, minDistance=3, mask=mask)
    if points is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return points + np.array([window_left, window_top], dtype=np.float32)


def _move_tracks(
    previous_gray: cv.typing.MatLike,
    gray: cv.typing.MatLike,
    tracks: list[Track],
) -> list[tuple[BBox, np.ndarray] | None]:
    """Move each track by the median motion of its points, or give None for a track that lost them.

    All tracks share one optical-flow call each way. A call spends most of its time building image pyramids of
    both frames, so one call per track multiplied that cost by the number of tracks. Each point is followed on its
    own, so sharing the call does not change any result. A point counts only if following it back to the previous
    frame lands within MAX_FLOW_ERROR pixels of where it started.
    """
    moves: list[tuple[BBox, np.ndarray] | None] = [None] * len(tracks)
    followed = [index for index, track in enumerate(tracks) if len(track.points) >= MIN_POINTS]
    if not followed:
        return moves
    points = np.concatenate([tracks[index].points for index in followed])
    moved, forward_status, _ = cv.calcOpticalFlowPyrLK(
        previous_gray, gray, points, np.empty_like(points), winSize=(21, 21), maxLevel=3
    )
    returned, backward_status, _ = cv.calcOpticalFlowPyrLK(
        gray, previous_gray, moved, np.empty_like(moved), winSize=(21, 21), maxLevel=3
    )
    error = np.linalg.norm(points - returned, axis=2).ravel()
    valid = (forward_status.ravel() == 1) & (backward_status.ravel() == 1) & (error < MAX_FLOW_ERROR)

    start = 0
    for index in followed:
        track = tracks[index]
        end = start + len(track.points)
        keep = valid[start:end]
        if keep.sum() >= MIN_POINTS:
            old_points = points[start:end][keep].reshape(-1, 2)
            new_points = moved[start:end][keep].reshape(-1, 2)
            dx, dy = np.median(new_points - old_points, axis=0)
            x, y, width, height = track.bbox
            moves[index] = (x + float(dx), y + float(dy), width, height), new_points.reshape(-1, 1, 2)
        start = end
    return moves


def _iou(first: BBox, second: BBox) -> float:
    x1, y1, width1, height1 = first
    x2, y2, width2, height2 = second
    overlap_width = max(0.0, min(x1 + width1, x2 + width2) - max(x1, x2))
    overlap_height = max(0.0, min(y1 + height1, y2 + height2) - max(y1, y2))
    intersection = overlap_width * overlap_height
    union = width1 * height1 + width2 * height2 - intersection
    return intersection / union if union else 0.0


@define
class Track:
    track_id: int
    bbox: BBox
    points: np.ndarray
    first_frame: int
    last_frame: int
    last_detection_frame: int
    last_read_frame: int = -1
    votes: dict[str, float] = field(factory=dict)
    active: bool = True

    @property
    def best_bib(self) -> str | None:
        """Return the reading with the most confidence behind it.

        Prefer the longest reading, then its accumulated confidence. A tie returns
        the reading seen first, in vote order.
        """
        if not self.votes:
            return None
        return max(self.votes, key=lambda bib: (len(bib), self.votes[bib]))

    @property
    def conflicting(self) -> bool:
        return len(self.votes) > 1

    def add_vote(self, bib: str, confidence: float) -> None:
        """Accumulate confidence for one reading."""
        self.votes[bib] = self.votes.get(bib, 0.0) + confidence


class Tracker:
    def __init__(
        self,
        bib_pattern: str = BIB_PATTERN,
        reader_fn: Callable[[cv.typing.MatLike, BBox, DetectorLike, str], BibReading | None] | None = None,
    ) -> None:
        """Follow bib boxes between detector frames and collect their readings.

        ``bib_pattern`` is the regex a reading must match to count as a bib number; it
        is applied whenever the tracker reads a bib itself.
        """
        self.bib_pattern = bib_pattern
        self.reader_fn = reader_fn or BIB_READERS["yolov4"]
        self.previous_gray: cv.typing.MatLike | None = None
        self.tracks: list[Track] = []

    @property
    def active_tracks(self) -> list[Track]:
        return [track for track in self.tracks if track.active]

    def follow(self, frame: cv.typing.MatLike, frame_id: int) -> None:
        """Move active tracks one frame forward and deactivate the ones that fail.

        A track that loses optical flow is deactivated rather than deleted, so the
        caller sees it leave ``active_tracks`` and can rescan before the next
        scheduled scan.
        """
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return

        tracks = self.active_tracks
        for track, moved in zip(tracks, _move_tracks(self.previous_gray, gray, tracks), strict=True):
            if moved is None:
                track.active = False
                continue
            track.bbox, track.points = moved
            track.last_frame = frame_id
        self.previous_gray = gray

    def correct(
        self,
        frame: cv.typing.MatLike,
        detections: list[BibDetection],
        frame_id: int,
    ) -> None:
        """Match detector boxes to active tracks and start a track for the rest.

        Only a box with a bib number starts a track. A box whose digits could not be
        read, or read as something that is not a bib number, still moves an existing
        track but starts none of its own, so junk boxes add no tracks. Detector
        confidence does not gate new tracks.
        """
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        used_tracks: set[int] = set()
        for detection in detections:
            candidates = [
                track
                for track in self.active_tracks
                if track.track_id not in used_tracks and _iou(track.bbox, detection.bbox) >= MATCH_IOU
            ]
            if candidates:
                track = max(candidates, key=lambda item: _iou(item.bbox, detection.bbox))
                # avoid recomputing the IoU for the same track twice, since we compute it above already
                # TODO: Use Hungarian algorithm for better matching
            elif detection.bib_string is None:
                continue
            else:
                track = Track(
                    len(self.tracks),
                    detection.bbox,
                    np.empty((0, 1, 2), dtype=np.float32),
                    frame_id,
                    frame_id,
                    frame_id,
                )
                self.tracks.append(track)

            track.bbox = detection.bbox
            track.points = _find_points(gray, detection.bbox)
            track.last_frame = frame_id
            track.last_detection_frame = frame_id
            if detection.bib_string is not None:
                track.add_vote(detection.bib_string, detection.confidence)
                track.last_read_frame = frame_id
            used_tracks.add(track.track_id)

    def read(self, frame: cv.typing.MatLike, reader: DetectorLike, frame_id: int) -> None:
        """Read active bibs that the detector did not already read on this frame."""
        for track in self.active_tracks:
            if track.last_detection_frame == frame_id:
                continue
            reading = self.reader_fn(frame, track.bbox, reader, self.bib_pattern)
            if reading is not None:
                track.add_vote(reading.bib_string, reading.confidence)
                track.last_read_frame = frame_id

    def results(self, fps: float) -> list[TrackResult]:
        """Return the final bib and visible times for every track."""
        return [
            {
                "track_id": track.track_id,
                "best_bib": track.best_bib,
                "votes": dict(track.votes),
                "conflicting": track.conflicting,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "first_seconds": track.first_frame / fps,
                "last_seconds": track.last_frame / fps,
            }
            for track in self.tracks
        ]
