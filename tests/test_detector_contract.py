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


@pytest.mark.parametrize("digits, expected", [(["0", "0", "7"], "007"), ([], None)])
def test_read_bib_preserves_zeroes_and_unreadable_bibs(digits, expected):
    class Reader:
        def detect(self, _crop, _threshold):
            return [detector.Detection(number, (index * 3, 0, 2, 8), 0.9) for index, number in enumerate(digits)]

    reader = Reader()
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    assert detector.read_bib(image, (-5, -5, 15, 15), reader) == expected


def test_detect_bibs_retains_an_unreadable_box(monkeypatch):
    bib_detector = SimpleNamespace(detect=lambda _image, _threshold: [detector.Detection("bib", (2, 3, 10, 8), 0.8)])
    digit_reader = SimpleNamespace(detect=lambda _crop, _threshold: [])
    models = iter([bib_detector, digit_reader])
    monkeypatch.setattr(detector, "get_detector", lambda _config: next(models))
    config = detector.DetectorConfig("unused", "unused", ("bib",))

    result = detector.detect_bibs(np.zeros((20, 20, 3), dtype=np.uint8), config, config)

    assert result == [detector.BibDetection((2, 3, 10, 8), None, 0.8)]


def test_read_bib_does_not_send_an_empty_crop_to_the_model():
    class Reader:
        def detect(self, _image, _confidence):
            pytest.fail("empty crop sent to model")

    reader = Reader()

    result = detector.read_bib(np.zeros((20, 20, 3), dtype=np.uint8), (50, 50, 10, 10), reader)

    assert result is None
