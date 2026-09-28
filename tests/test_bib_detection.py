"""Integration tests for bib detection.

Each test frame is a JPG extracted from ``sample1/video.mp4`` at a video
position where ``bib_detection.py`` previously read a specific bib number. The
tests run the real detection pipeline (``bib_detection.detect_bibs``) on the
frame and assert the expected bib number is read back.

The frames live in ``tests/frames/``.

Run the tests with:

    pytest
"""

from pathlib import Path

import cv2 as cv
import pytest

from bib_detection import detect_bibs

FRAMES_DIR = Path(__file__).resolve().parent / "frames"

# (expected bib number, frame id as logged by bib_detection.py) for each test
# frame. bib_detection.py only runs the detector every N-th loop iteration, so
# the frame id it logs is not the real video index (the real index is the
# logged id // 100).
FRAME_SPECS: list[tuple[str, int]] = [
    ("13426", 0),
    ("6500", 8600),
    ("6240", 22400),
    ("11461", 33800),
    ("21893", 39800),
]


@pytest.mark.parametrize(
    ("frame_file", "expected_bib"),
    [
        pytest.param(
            FRAMES_DIR / f"bib_{bib_string}_frame_{frame_id}.jpg",
            bib_string,
            id=f"bib_{bib_string}_frame_{frame_id}",
        )
        for bib_string, frame_id in FRAME_SPECS
    ],
)
def test_bib_is_detected_on_frame(frame_file: Path, expected_bib: str) -> None:
    """The expected bib number is read from its extracted frame."""
    assert frame_file.exists(), f"Missing test frame fixture {frame_file}"

    frame = cv.imread(str(frame_file))
    assert frame is not None, f"Could not read frame {frame_file}"

    detections = detect_bibs(frame)
    detected_bibs = {detection.number for detection in detections if detection.number is not None}

    assert expected_bib in detected_bibs, (
        f"Expected bib {expected_bib} on {frame_file.name}, but detected {sorted(detected_bibs)}"
    )
