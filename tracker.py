"""Follow bib boxes between detector frames and collect bib readings."""

from __future__ import annotations

from collections import Counter
from typing import TypedDict

import cv2 as cv
import numpy as np
from attrs import define, field

from detector import BBox, BibDetection, DetectorLike, read_bib

MIN_POINTS = 6
MAX_FLOW_ERROR = 1.5
MATCH_IOU = 0.3
NEW_TRACK_CONFIDENCE = 0.5


class TrackResult(TypedDict):
    track_id: int
    best_bib: str | None
    votes: dict[str, int]
    conflicting: bool
    first_frame: int
    last_frame: int
    first_seconds: float
    last_seconds: float


def _find_points(gray: cv.typing.MatLike, bbox: BBox) -> np.ndarray:
    x, y, width, height = (int(value) for value in bbox)
    image_height, image_width = gray.shape[:2]
    left, top = max(0, x), max(0, y)
    right, bottom = min(image_width, x + width), min(image_height, y + height)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    points = cv.goodFeaturesToTrack(
        gray, maxCorners=50, qualityLevel=0.01, minDistance=3, mask=mask
    )
    return points if points is not None else np.empty((0, 1, 2), dtype=np.float32)


def _move_track(
    previous_gray: cv.typing.MatLike,
    gray: cv.typing.MatLike,
    track: Track,
) -> tuple[BBox, np.ndarray] | None:
    if len(track.points) < MIN_POINTS:
        return None
    moved, forward_status, _ = cv.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        track.points,
        np.empty_like(track.points),
        winSize=(21, 21),
        maxLevel=3,
    )
    if moved is None or forward_status is None:
        return None
    returned, backward_status, _ = cv.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        moved,
        np.empty_like(moved),
        winSize=(21, 21),
        maxLevel=3,
    )
    if returned is None or backward_status is None:
        return None
    error = np.linalg.norm(track.points - returned, axis=2).ravel()
    valid = (
        (forward_status.ravel() == 1) & (backward_status.ravel() == 1) & (error < MAX_FLOW_ERROR)
    )
    if valid.sum() < MIN_POINTS:
        return None
    old_points = track.points[valid].reshape(-1, 2)
    new_points = moved[valid].reshape(-1, 2)
    dx, dy = np.median(new_points - old_points, axis=0)
    x, y, width, height = track.bbox
    return (x + float(dx), y + float(dy), width, height), new_points.reshape(-1, 1, 2)


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
    votes: Counter[str] = field(factory=Counter)
    active: bool = True

    @property
    def best_bib(self) -> str | None:
        """Return the leading bib reading, or None when there is a tie."""
        leaders = self.votes.most_common(2)
        if not leaders or (len(leaders) > 1 and leaders[0][1] == leaders[1][1]):
            return None
        return leaders[0][0]

    @property
    def conflicting(self) -> bool:
        return len(self.votes) > 1


class Tracker:
    def __init__(self) -> None:
        self.previous_gray: cv.typing.MatLike | None = None
        self.tracks: list[Track] = []

    @property
    def active_tracks(self) -> list[Track]:
        return [track for track in self.tracks if track.active]

    def follow(self, frame: cv.typing.MatLike, frame_id: int) -> bool:
        """Move active tracks to frame and return whether any track was lost."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return False

        lost = False
        for track in self.active_tracks:
            moved = _move_track(self.previous_gray, gray, track)
            if moved is None:
                track.active = False
                lost = True
                continue
            track.bbox, track.points = moved
            track.last_frame = frame_id
        self.previous_gray = gray
        return lost

    def correct(
        self,
        frame: cv.typing.MatLike,
        detections: list[BibDetection],
        frame_id: int,
    ) -> None:
        """Match detector boxes to active tracks and start unmatched tracks."""
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        used_tracks: set[int] = set()
        for detection in detections:
            candidates = [
                track
                for track in self.active_tracks
                if track.track_id not in used_tracks
                and _iou(track.bbox, detection.bbox) >= MATCH_IOU
            ]
            if candidates:
                track = max(candidates, key=lambda item: _iou(item.bbox, detection.bbox))
            elif detection.confidence >= NEW_TRACK_CONFIDENCE:
                track = Track(
                    len(self.tracks),
                    detection.bbox,
                    np.empty((0, 1, 2), dtype=np.float32),
                    frame_id,
                    frame_id,
                    frame_id,
                )
                self.tracks.append(track)
            else:
                continue

            track.bbox = detection.bbox
            track.points = _find_points(gray, detection.bbox)
            track.last_frame = frame_id
            track.last_detection_frame = frame_id
            if detection.number is not None:
                track.votes[detection.number] += 1
            used_tracks.add(track.track_id)

    def read(self, frame: cv.typing.MatLike, reader: DetectorLike) -> None:
        """Read each active bib box and add successful readings to its votes."""
        for track in self.active_tracks:
            number = read_bib(frame, track.bbox, reader)
            if number is not None:
                track.votes[number] += 1

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
