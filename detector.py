"""Detect bib boxes and read the digits inside them."""

from functools import cache
from typing import Protocol

import cv2 as cv
import numpy as np
from attrs import frozen

type BBox = tuple[float, float, float, float]


@frozen
class Detection:
    class_name: str
    bbox: BBox
    confidence: float


@frozen
class BibDetection:
    bbox: BBox
    number: str | None
    confidence: float


@frozen
class DetectorConfig:
    cfg: str
    weights: str
    classes: tuple[str, ...]


class DetectorLike(Protocol):
    def detect(self, image: cv.typing.MatLike, confidence: float, /) -> list[Detection]: ...


class Detector:
    def __init__(self, config: DetectorConfig) -> None:
        self.classes = config.classes
        self.net = cv.dnn.readNetFromDarknet(config.cfg, config.weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        names = self.net.getLayerNames()
        self.layer_names = [names[index - 1] for index in self.net.getUnconnectedOutLayers()]

    def detect(self, image: cv.typing.MatLike, confidence: float) -> list[Detection]:
        """Return model detections in image coordinates."""
        blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)
        height, width = image.shape[:2]
        image_scale = np.array([width, height, width, height])
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for raw_detection in output:
                class_id = int(np.argmax(raw_detection[5:]))
                score = float(raw_detection[5 + class_id])
                if score <= confidence:
                    continue
                center_x, center_y, box_width, box_height = (
                    raw_detection[:4] * image_scale
                ).astype(int)
                boxes.append(
                    [
                        int(center_x - box_width / 2),
                        int(center_y - box_height / 2),
                        int(box_width),
                        int(box_height),
                    ]
                )
                scores.append(score)
                class_ids.append(class_id)

        kept = np.asarray(cv.dnn.NMSBoxes(boxes, scores, confidence, 0.4)).flatten()
        return [
            Detection(self.classes[class_ids[index]], tuple(boxes[index]), scores[index])
            for index in kept
        ]


@cache
def get_detector(config: DetectorConfig) -> Detector:
    """Load a model once and reuse it."""
    return Detector(config)


def read_bib(image: cv.typing.MatLike, bbox: BBox, reader: DetectorLike) -> str | None:
    """Return the digits inside a bib box, or None when none can be read."""
    x, y, width, height = (int(value) for value in bbox)
    image_height, image_width = image.shape[:2]
    left, top = max(0, x), max(0, y)
    right, bottom = min(image_width, x + width), min(image_height, y + height)
    if right <= left or bottom <= top:
        return None

    digits = reader.detect(image[top:bottom, left:right], 0.5)
    if not digits:
        return None
    digits.sort(key=lambda digit: digit.bbox[0])
    return "".join(digit.class_name for digit in digits)


def detect_bibs(
    image: cv.typing.MatLike,
    bib_config: DetectorConfig,
    digit_config: DetectorConfig,
) -> list[BibDetection]:
    """Return the bib boxes and numbers found in one image."""
    bib_detector = get_detector(bib_config)
    boxes = bib_detector.detect(image, 0.1)
    if not boxes:
        return []

    digit_reader = get_detector(digit_config)
    return [
        BibDetection(box.bbox, read_bib(image, box.bbox, digit_reader), box.confidence)
        for box in boxes
    ]
