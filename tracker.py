"""Follow bib boxes between detector frames and collect bib readings."""

from __future__ import annotations

import math
from collections.abc import Callable
from itertools import count
from typing import TypedDict

import cv2 as cv
import numpy as np
from attrs import Factory, define, field

from detector import BIB_PATTERN, BIB_READERS, BBox, BibDetection, BibReading, DetectorLike

MIN_POINTS = 6
MAX_FLOW_ERROR = 1.5
MATCH_IOU = 0.3
# A corner score looks at a few pixels around each pixel (gradient, scoring window and peak picking), so the
# corner search runs on the box plus this many pixels. Scoring a whole 1080p frame for one box cost about 55 ms.
CORNER_CONTEXT = 4
# Optical flow follows the points on frames shrunk by this factor. Building the image pyramids of both frames,
# each way, was most of the tracking cost, and a half-size frame has a quarter of the pixels.
FLOW_SCALE = 0.5
# Each digit multiplies a reading's votes by this weight. The reader drops one of two repeated digits far more
# often than it invents one (on sample1, 43 wrong votes against 6), so a longer reading should weigh more. Weights
# of 3 to 5 did best there. Plain sums lost 11116 to 1116, and strict length priority let a single misread
# 12461 beat thirteen votes for 3246.
DIGIT_WEIGHT = 5
# A track is locked once its best number stayed the same over this many readings in a row. A locked track is read
# only every LOCKED_READ_EVERY frames instead of every read_every, so later readings can still correct it. On
# sample1 this cut reader calls from 238 to 123 with the same 13/13 (30 frames: 152, 120: 109). Never reading a
# locked track again (94 calls) kept 6240 locked on 5240 after reading 5240, 6240, 6740.
LOCK_READS = 3
LOCKED_READ_EVERY = 60
# A track that ended merges into a newer track with the same best number started at most this many frames later
# (10 s at 30 fps), when both have at least MERGE_MIN_READINGS readings. A runner hidden for a moment comes back
# as a new track, and on sample1 the gap was 6 frames. The readings minimum keeps one misread from merging two
# runners.
MERGE_GAP_FRAMES = 300
MERGE_MIN_READINGS = 3
# The newer track must also start near where the older one was lost: within MERGE_NEAR_HEIGHTS box heights, plus
# MERGE_SPEED_HEIGHTS per frame of gap. On sample1 bibs moved at most 0.15 box heights per frame, and the newer
# 11461 track started 1.4 heights from the older one after 6 frames.
# ponytail: a straight radius ignores direction; extrapolate the older track's motion if long gaps merge wrongly.
MERGE_NEAR_HEIGHTS = 2.0
MERGE_SPEED_HEIGHTS = 0.15


class Reading(TypedDict):
    frame: int
    seconds: float
    bib: str
    confidence: float


class TrackResult(TypedDict):
    track_id: int
    best_bib: str | None
    votes: dict[str, float]
    readings: list[Reading]
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
    frame lands within MAX_FLOW_ERROR pixels of where it started. Points live on the frames shrunk by FLOW_SCALE,
    so their motion is scaled back up to move the full-size box.
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
            dx, dy = np.median(new_points - old_points, axis=0) / FLOW_SCALE
            x, y, width, height = track.bbox
            moves[index] = (x + float(dx), y + float(dy), width, height), new_points.reshape(-1, 1, 2)
        start = end
    return moves


def _best(votes: dict[str, float]) -> str:
    """Return the number with the most votes, weighted by DIGIT_WEIGHT per digit. A tie keeps the first seen."""
    return max(votes, key=lambda bib: votes[bib] * DIGIT_WEIGHT ** len(bib))


def _extends(longer: str, shorter: str) -> bool:
    """Whether ``longer`` is ``shorter`` with digits added, as 11461 is to 1146: a digit the reader had dropped."""
    digits = iter(longer)
    return len(longer) > len(shorter) and all(digit in digits for digit in shorter)


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
    # Where the track started, to tell whether it continues a track that ended nearby.
    start_bbox: BBox = field(default=Factory(lambda track: track.bbox, takes_self=True))
    votes: dict[str, float] = field(factory=dict)
    # Every reading in order, as (frame, bib, confidence), so a rule can weigh agreement over time.
    readings: list[tuple[int, str, float]] = field(factory=list)
    active: bool = True

    @property
    def best_bib(self) -> str | None:
        """Return the reading with the most confidence behind it.

        A reading's accumulated confidence is weighted by DIGIT_WEIGHT per digit, so one more digit counts five
        times as much. A tie returns the reading seen first, in vote order.
        """
        return _best(self.votes) if self.votes else None

    @property
    def locked(self) -> bool:
        """Whether the best number stayed the same over the last LOCK_READS readings.

        A reading that extends the best number (11461 when the best is 1146) restarts the count, since the reader
        drops digits far more often than it invents them. Any other disagreeing reading counts as long as the best
        number does not change, so a single misread does not unlock a well-read track.
        """
        votes: dict[str, float] = {}
        best, stable = None, 0
        for _frame, bib, confidence in self.readings:
            votes[bib] = votes.get(bib, 0.0) + confidence
            new_best = _best(votes)
            if new_best != best:
                stable = 1
            elif best is not None and _extends(bib, best):
                stable = 0
            else:
                stable += 1
            best = new_best
        return stable >= LOCK_READS

    def wants_reading(self, frame_id: int) -> bool:
        """Whether this track should be read on this frame: always while unlocked, then every LOCKED_READ_EVERY."""
        return not self.locked or frame_id - self.last_read_frame >= LOCKED_READ_EVERY

    @property
    def conflicting(self) -> bool:
        return len(self.votes) > 1

    def add_vote(self, bib: str, confidence: float, frame_id: int) -> None:
        """Accumulate confidence for one reading and remember when it was read."""
        self.votes[bib] = self.votes.get(bib, 0.0) + confidence
        self.readings.append((frame_id, bib, confidence))
        self.last_read_frame = frame_id


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
        # Merged tracks leave ``tracks``, so IDs come from a counter rather than the list length.
        self._track_ids = count()

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
        gray = cv.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv.INTER_AREA)
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
                    next(self._track_ids),
                    detection.bbox,
                    np.empty((0, 1, 2), dtype=np.float32),
                    frame_id,
                    frame_id,
                    frame_id,
                )
                self.tracks.append(track)

            track.bbox = detection.bbox
            # Corners are found at full size, so a small bib keeps as many points, then moved onto the shrunk frames.
            track.points = _find_points(gray, detection.bbox) * FLOW_SCALE
            track.last_frame = frame_id
            track.last_detection_frame = frame_id
            if detection.bib_string is not None:
                track.add_vote(detection.bib_string, detection.confidence, frame_id)
            used_tracks.add(track.track_id)
        self._merge_duplicates()

    def needs_reading(self, bbox: BBox, frame_id: int) -> bool:
        """Whether a detected box should be read, which is not when it matches a locked track not due for reading."""
        return not any(
            _iou(track.bbox, bbox) >= MATCH_IOU and not track.wants_reading(frame_id) for track in self.active_tracks
        )

    def read(self, frame: cv.typing.MatLike, reader: DetectorLike, frame_id: int) -> None:
        """Read active bibs that the detector did not already read on this frame."""
        for track in self.active_tracks:
            if track.last_detection_frame == frame_id or not track.wants_reading(frame_id):
                continue
            reading = self.reader_fn(frame, track.bbox, reader, self.bib_pattern)
            if reading is not None:
                track.add_vote(reading.bib_string, reading.confidence, frame_id)
        self._merge_duplicates()

    def _merge_duplicates(self) -> None:
        """Fold a newer active track into an older one with the same best number that belongs to the same runner.

        One runner wears one bib, so two tracks with the same number are one runner when either:
        - both are active and their boxes overlap, typically a new track started after an occlusion while the old
          one still followed something nearby;
        - the older one ended at most MERGE_GAP_FRAMES before the newer one started, near where the older one was
          lost, and both have at least MERGE_MIN_READINGS readings.

        The older track keeps its ID and first frame, and becomes active again. It takes the votes and readings of
        the newer one, and its box and points when the newer one was detected more recently or the older one ended.
        """

        def same_runner(older: Track, newer: Track) -> bool:
            if newer.best_bib is None or older.best_bib != newer.best_bib:
                return False
            if older.active:
                return _iou(older.bbox, newer.bbox) > 0
            gap = max(0, newer.first_frame - older.last_frame)
            x, y, width, height = older.bbox
            start_x, start_y, start_width, start_height = newer.start_bbox
            distance = math.dist(
                (x + width / 2, y + height / 2), (start_x + start_width / 2, start_y + start_height / 2)
            )
            return (
                gap <= MERGE_GAP_FRAMES
                and min(len(older.readings), len(newer.readings)) >= MERGE_MIN_READINGS
                and distance <= height * (MERGE_NEAR_HEIGHTS + MERGE_SPEED_HEIGHTS * gap)
            )

        for newer in self.active_tracks:
            older = next(
                (track for track in self.tracks[: self.tracks.index(newer)] if same_runner(track, newer)), None
            )
            if older is None:
                continue
            for bib, confidence in newer.votes.items():
                older.votes[bib] = older.votes.get(bib, 0.0) + confidence
            older.readings = sorted(older.readings + newer.readings)
            older.last_frame = max(older.last_frame, newer.last_frame)
            older.last_read_frame = max(older.last_read_frame, newer.last_read_frame)
            if not older.active or newer.last_detection_frame > older.last_detection_frame:
                older.bbox, older.points = newer.bbox, newer.points
                older.last_detection_frame = newer.last_detection_frame
            older.active = True
            newer.active = False
            self.tracks.remove(newer)

    def results(self, fps: float) -> list[TrackResult]:
        """Return the final bib and visible times for every track."""
        return [
            {
                "track_id": track.track_id,
                "best_bib": track.best_bib,
                "votes": dict(track.votes),
                "readings": [
                    {"frame": frame, "seconds": frame / fps, "bib": bib, "confidence": confidence}
                    for frame, bib, confidence in track.readings
                ],
                "conflicting": track.conflicting,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "first_seconds": track.first_frame / fps,
                "last_seconds": track.last_frame / fps,
            }
            for track in self.tracks
        ]
