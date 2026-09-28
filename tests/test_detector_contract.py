"""Check confidence and reading semantics without loading model weights."""

from types import SimpleNamespace

import numpy as np
import pytest

import detector


def test_nms_preserves_weak_detection_and_confidence(monkeypatch):
    model = object.__new__(detector.Detector)
    model.classes = ("bib",)
    model.layer_names = ["out"]
    net = SimpleNamespace(
        setInput=lambda _blob: None,
        forward=lambda _names: [np.array([[0.5, 0.5, 0.4, 0.2, 1.0, 0.2]])],
    )
    monkeypatch.setattr(model, "net", net, raising=False)
    results = model.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0.1)
    assert len(results) == 1
    assert results[0].confidence == pytest.approx(0.2)
    assert results[0].bbox == (30, 40, 40, 20)


@pytest.mark.parametrize("read_digits, expected", [(True, "007"), (False, None)])
def test_readings_preserve_zeros_and_unreadable_boxes(monkeypatch, read_digits, expected):
    def read_crop(crop, _threshold):
        assert crop.shape == (10, 10, 3)
        if not read_digits:
            return []
        return [
            detector.Detection("7", (7, 0, 2, 8), 0.9),
            detector.Detection("0", (0, 0, 2, 8), 0.9),
            detector.Detection("0", (3, 0, 2, 8), 0.9),
        ]

    bib_model = SimpleNamespace(
        detect=lambda _img, _threshold: [detector.Detection("bib", (-5, -5, 15, 15), 0.8)]
    )
    digits_model = SimpleNamespace(detect=read_crop)
    models = iter([bib_model, digits_model])
    monkeypatch.setattr(detector, "get_detector", lambda _cfg: next(models))
    config = detector.DetectorConfig("unused", "unused", ("bib",))
    results = detector.get_rbns(np.zeros((20, 20, 3), dtype=np.uint8), config, config)
    assert len(results) == 1
    assert results[0].bib_string == expected
    assert results[0].confidence == pytest.approx(0.8)


def test_invalid_crop_retains_unreadable_tracking_box(monkeypatch):
    def unexpected_crop(*_args):
        pytest.fail("Digit reader must not receive an empty crop")

    models = iter(
        [
            SimpleNamespace(
                detect=lambda *_args: [detector.Detection("bib", (50, 50, 10, 10), 0.8)]
            ),
            SimpleNamespace(detect=unexpected_crop),
        ]
    )
    monkeypatch.setattr(detector, "get_detector", lambda _cfg: next(models))
    config = detector.DetectorConfig("unused", "unused", ("bib",))
    result = detector.get_rbns(np.zeros((20, 20, 3), dtype=np.uint8), config, config)
    assert len(result) == 1
    assert result[0].bib_string is None
    assert result[0].bbox == (50, 50, 10, 10)
