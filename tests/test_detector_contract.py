"""Tests for bib detection and digit reading."""

from types import SimpleNamespace
from typing import cast

import cv2 as cv
import numpy as np
import pytest

import detector


def test_detector_keeps_confidence():
    model = object.__new__(detector.Detector)
    model.classes = ("bib",)
    model.input_size = 416
    model.layer_names = ["out"]
    model.net = cast(
        cv.dnn.Net,
        SimpleNamespace(
            setInput=lambda _blob: None,
            forward=lambda _names: [np.array([[0.5, 0.5, 0.4, 0.2, 1.0, 0.2]])],
        ),
    )

    result = model.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0.1)

    assert result[0].bbox == (30, 40, 40, 20)
    assert result[0].confidence == pytest.approx(0.2)


def test_detector_squashes_frames_to_the_configured_input_size(monkeypatch):
    sizes = []
    monkeypatch.setattr(
        cv.dnn,
        "blobFromImage",
        lambda _image, _scale, size, **_kwargs: sizes.append(size) or np.zeros((1, 3, 416, 416)),
    )
    model = object.__new__(detector.Detector)
    model.classes = ("bib",)
    model.input_size = 832
    model.layer_names = ["out"]
    model.net = cast(
        cv.dnn.Net,
        SimpleNamespace(
            setInput=lambda _blob: None,
            forward=lambda _names: [np.array([[0.5, 0.5, 0.4, 0.2, 1.0, 0.2]])],
        ),
    )

    model.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0.1)

    assert sizes == [(832, 832)]


class DigitsReader:
    """A stand-in digit reader that returns these digits from left to right."""

    def __init__(self, *digits: str) -> None:
        self.digits = digits

    def detect(self, _image: cv.typing.MatLike, _confidence: float) -> list[detector.Detection]:
        return [detector.Detection(digit, (index * 3, 0, 2, 8), 0.9) for index, digit in enumerate(self.digits)]


@pytest.mark.parametrize(
    "digits, expected",
    [
        (["0", "0", "0", "7"], detector.BibReading("0007", 0.9)),
        ([], None),
    ],
)
def test_read_bib_preserves_zeroes_and_unreadable_bibs(digits, expected):
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    assert detector.read_bib(image, (-5, -5, 15, 15), DigitsReader(*digits)) == expected


@pytest.mark.parametrize(
    "digits, expected",
    [
        (["1", "2", "3"], None),
        (["1", "2", "3", "4"], detector.BibReading("1234", 0.9)),
        (["1", "2", "3", "4", "5"], detector.BibReading("12345", 0.9)),
        (["1", "2", "3", "4", "5", "6"], None),
    ],
)
def test_read_bib_keeps_only_readings_that_match_the_bib_pattern(digits, expected):
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    assert detector.read_bib(image, (-5, -5, 15, 15), DigitsReader(*digits)) == expected


def test_read_bib_confidence_is_the_mean_of_the_digit_scores():
    class Reader:
        def detect(self, _image, _confidence):
            return [
                detector.Detection("1", (0, 0, 2, 8), 1.0),
                detector.Detection("2", (3, 0, 2, 8), 0.5),
                detector.Detection("3", (6, 0, 2, 8), 0.0),
                detector.Detection("4", (9, 0, 2, 8), 0.5),
            ]

    reading = detector.read_bib(np.zeros((20, 20, 3), dtype=np.uint8), (-5, -5, 15, 15), Reader())

    assert reading is not None
    assert reading.bib_string == "1234"
    assert reading.confidence == pytest.approx(0.5)


def test_read_bib_accepts_a_configured_bib_pattern():
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    reader = DigitsReader("A", "1", "2", "3")

    assert detector.read_bib(image, (-5, -5, 15, 15), reader, bib_pattern=r"[A-Z]\d{3}") == detector.BibReading(
        "A123", 0.9
    )


def test_detect_bibs_retains_an_unreadable_box(monkeypatch):
    bib_detector = SimpleNamespace(detect=lambda _image, _threshold: [detector.Detection("bib", (2, 3, 10, 8), 0.8)])
    digit_reader = SimpleNamespace(detect=lambda _crop, _threshold: [])
    models = iter([bib_detector, digit_reader])
    monkeypatch.setattr(detector, "get_detector", lambda _config: next(models))
    config = detector.DetectorConfig("unused", "unused", ("bib",))

    result = detector.detect_bibs(np.zeros((20, 20, 3), dtype=np.uint8), config, config)

    assert result == [detector.BibDetection((2, 3, 10, 8), None, 0.0)]


def test_read_bib_does_not_send_an_empty_crop_to_the_model():
    class Reader:
        def detect(self, _image, _confidence):
            pytest.fail("empty crop sent to model")

    reader = Reader()

    result = detector.read_bib(np.zeros((20, 20, 3), dtype=np.uint8), (50, 50, 10, 10), reader)

    assert result is None
