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


def test_read_bibs_retains_an_unreadable_box(monkeypatch):
    digit_reader = SimpleNamespace(detect=lambda _crop, _threshold: [])
    monkeypatch.setattr(detector, "get_detector", lambda _config: digit_reader)
    config = detector.DetectorConfig("unused", "unused", ("bib",))
    boxes = [detector.Detection("bib", (2, 3, 10, 8), 0.8)]

    result = detector.read_bibs(np.zeros((20, 20, 3), dtype=np.uint8), boxes, config)

    assert result == [detector.BibDetection((2, 3, 10, 8), None, 0.0)]


@pytest.mark.parametrize(
    "name, model_id",
    [("roboflow_2.0", "bib-detection/5"), ("rfdetr-large-t1", "bib-detection/7"), ("yolo26n-t1", "bib-detection/8")],
)
def test_roboflow_detector_converts_and_filters_boxes(monkeypatch, name, model_id):
    """Each Roboflow version uses its own resizing and returns video-coordinate boxes."""
    label = "0" if name == "yolo26n-t1" else "Bib"
    predictions = [
        SimpleNamespace(class_name=label, x=30, y=40, width=20, height=10, confidence=0.9),
        SimpleNamespace(class_name=label, x=30, y=40, width=20, height=10, confidence=0.05),
    ]
    called = []

    def load(identifier):
        def infer(_image, **kwargs):
            called.append((identifier, kwargs))
            return [SimpleNamespace(predictions=predictions)]

        return SimpleNamespace(infer=infer)

    monkeypatch.setattr(detector, "get_roboflow_detector", load)
    config = detector.DetectorConfig("unused", "unused", ("bib",), confidence=0.1)

    assert detector.BIB_DETECTORS[name](np.zeros((80, 80, 3), dtype=np.uint8), config) == [
        detector.Detection(label, (20, 35, 20, 10), 0.9)
    ]
    expected_options = {"image_size": 416} if name == "yolo26n-t1" else {}
    assert called == [(model_id, {"confidence": 0.1, **expected_options})]


def test_read_bib_does_not_send_an_empty_crop_to_the_model():
    class Reader:
        def detect(self, _image, _confidence):
            pytest.fail("empty crop sent to model")

    reader = Reader()

    result = detector.read_bib(np.zeros((20, 20, 3), dtype=np.uint8), (50, 50, 10, 10), reader)

    assert result is None


def test_openvino_detector_letterboxes_the_frame_and_maps_boxes_back(monkeypatch):
    shapes = []

    def compiled(blob):
        shapes.append(blob.shape)
        # Two candidates in network pixels: centre x, centre y, width, height, score. The second is below threshold.
        return [np.array([[[416, 100], [240, 100], [52, 10], [26, 10], [0.9, 0.05]]], dtype=np.float32)]

    monkeypatch.setattr(detector, "get_openvino_model", lambda _path, _height, _width: compiled)
    config = detector.DetectorConfig("unused", "unused", ("bib",), input_size=832, confidence=0.1)

    (found,) = detector.find_openvino_bibs(np.zeros((1080, 1920, 3), dtype=np.uint8), config)

    # 1920x1080 scales by 832/1920 to 832x468, padded to 832x480 with 6 px above the frame.
    scale = 832 / 1920
    assert shapes == [(1, 3, 480, 832)]
    assert found.bbox == pytest.approx(((416 - 26) / scale, (240 - 13 - 6) / scale, 52 / scale, 26 / scale))
    assert found.confidence == pytest.approx(0.9)


def fake_ocr(monkeypatch, columns: list[tuple[str, float]]) -> None:
    """Make the recognizer return these (best character, probability) columns, "" meaning no character."""
    alphabet = ["", *"0123456789B", " "]
    probabilities = np.full((len(columns), len(alphabet)), 0.001, dtype=np.float32)
    for column, (character, probability) in enumerate(columns):
        probabilities[column, alphabet.index(character)] = probability
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="x")], run=lambda _outputs, _feed: [probabilities[None]]
    )
    monkeypatch.setattr(detector, "get_ocr", lambda _path: (session, alphabet))


def test_ocr_reader_merges_repeats_and_keeps_the_longest_run_of_digits(monkeypatch):
    # Reads "1B 3325": a wave label, then the number. The doubled 3 only counts twice because "" splits it.
    columns = [("1", 0.9), ("B", 0.9), (" ", 0.9), ("3", 0.8), ("3", 0.8), ("", 0.9), ("3", 0.6), ("2", 1.0)]
    fake_ocr(monkeypatch, [*columns, ("5", 1.0), ("", 0.9)])

    reading = detector.read_bib_ocr(np.zeros((40, 60, 3), dtype=np.uint8), (0, 0, 60, 40), DigitsReader())

    assert reading is not None
    assert reading.bib_string == "3325"
    assert reading.confidence == pytest.approx((0.8 + 0.6 + 1.0 + 1.0) / 4)


def test_ocr_reader_rejects_a_number_that_does_not_match_the_pattern(monkeypatch):
    fake_ocr(monkeypatch, [("1", 0.9), ("2", 0.9), ("3", 0.9)])

    assert detector.read_bib_ocr(np.zeros((40, 60, 3), dtype=np.uint8), (0, 0, 60, 40), DigitsReader()) is None


def test_ocr_reader_does_not_run_on_a_box_outside_the_image(monkeypatch):
    monkeypatch.setattr(detector, "get_ocr", lambda _path: pytest.fail("empty crop sent to the recognizer"))

    assert detector.read_bib_ocr(np.zeros((20, 20, 3), dtype=np.uint8), (50, 50, 10, 10), DigitsReader()) is None
