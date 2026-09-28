"""Utilities for detecting race bibs and reading the numbers printed on them.

This module wraps OpenCV's Darknet (YOLO) DNN backend: :class:`Detector` builds
a network from a Darknet cfg/weights pair, and :func:`get_rbns` chains a bib
locator together with a digit reader to turn a single frame into bib numbers.
"""

from functools import cache
from time import perf_counter
from typing import Any

import cv2 as cv
import numpy as np
from attrs import frozen

type BBox = tuple[float, float, float, float]  # x, y, width, height


@frozen
class Detection:
    """One raw model result after confidence filtering and suppression."""

    class_name: str
    bbox: BBox
    confidence: float


@frozen
class BibDetection:
    """A located bib with optional OCR text and the locator confidence."""

    frame_id: int
    bbox: BBox
    bib_string: str | None
    confidence: float


@frozen
class DetectorConfig:
    """Darknet model files and class names used to build a :class:`Detector`."""

    cfg: str
    weights: str
    classes: tuple[str, ...]


class Detector:
    """A YOLO object detector backed by OpenCV's Darknet DNN module.

    Attributes
        classes (tuple): class names the model was trained on
        net (obj): OpenCV network object
        layer_names (list): names of the network's output layers
    """

    def __init__(self, config: DetectorConfig) -> None:
        """Build the detection network described by ``config``."""
        self.classes = config.classes
        self.net = cv.dnn.readNetFromDarknet(config.cfg, config.weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

        # Determine the network's output layers
        layer_names = self.net.getLayerNames()
        self.layer_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, img: Any, conf: float) -> list[Detection]:
        """Detect objects in an image above a confidence threshold.

        Args
            img (numpy array): image array from openCV .imread
            conf (float): prediction confidence threshold

        Returns
            Class, bounding box and confidence for each surviving detection.
        """

        # Format the image for detection
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Get raw detections
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        h_img, w_img = img.shape[:2]
        image_scale = np.array([w_img, h_img, w_img, h_img])

        boxes: list[list[int]] = []
        confidences: list[float] = []
        class_ids: list[int] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                # Keep only detections above the requested confidence
                if confidence > conf:
                    box = detection[:4] * image_scale
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Apply non-maximal suppression so each object is reported once
        indices = np.asarray(cv.dnn.NMSBoxes(boxes, confidences, conf, 0.4)).flatten()

        return [
            Detection(
                self.classes[class_ids[i]],
                (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]),
                confidences[i],
            )
            for i in indices
        ]


# Detectors are expensive to build (the weights are read from disk), so keep
# one instance per configuration and reuse it.
@cache
def get_detector(config: DetectorConfig) -> Detector:
    """Return a cached detector for the given model configuration.

    Building a Detector loads the whole Darknet model from disk, which is slow.
    Calling this instead of Detector(...) makes repeated detections (e.g. once
    per video frame) reuse the already loaded network.

    Args
        config (DetectorConfig): model files and class names to build from

    Returns
        Detector object
    """

    return Detector(config)


def get_rbns(
    img: Any,
    bib_detector_cfg: DetectorConfig,
    number_reader_cfg: DetectorConfig,
    frame_id: int = 0,
    timings: dict[str, float] | None = None,
) -> list[BibDetection]:
    """Return bib numbers and bib bounding boxes for the detected bibs.

    Args
        img (numpy array): image array given by openCV .imread
        bib_detector_cfg (DetectorConfig): model that locates bibs in the image
        number_reader_cfg (DetectorConfig): model that reads the digits inside a
            located bib

    Returns
        Bib boxes with confidence and an optional reading. None means unreadable;
        strings preserve leading zeros. Confidence is the existing OpenCV Darknet
        class score, not an independently calibrated probability.
    """

    # Instantiate detectors (reused across calls, see get_detector)
    bib_detector = get_detector(bib_detector_cfg)
    # Make bib location predictions
    started = perf_counter()
    bib_detections = bib_detector.detect(img, 0.1)
    if timings is not None:
        timings["bib_detection"] += perf_counter() - started
    if not bib_detections:
        return []

    number_reader = get_detector(number_reader_cfg)
    bib_readings: list[BibDetection] = []
    for detection in bib_detections:
        bib_string = read_bib(img, detection.bbox, number_reader, timings)
        bib_readings.append(
            BibDetection(frame_id, detection.bbox, bib_string, detection.confidence)
        )

    return bib_readings


def read_bib(
    img: Any, bbox: BBox, number_reader: Detector, timings: dict[str, float] | None = None
) -> str | None:
    """Read left-to-right digits from a detected or optically tracked bib crop.

    The box is clipped to image boundaries before slicing. Returns None for an
    empty/invalid crop or when the digit detector finds nothing. The result is a
    string, rather than an integer, so leading zeroes are preserved.
    """
    started = perf_counter()
    x, y, w, h = (int(value) for value in bbox)
    image_h, image_w = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_w, x + w), min(image_h, y + h)
    digits = []
    if x2 > x1 and y2 > y1:
        digits = number_reader.detect(img[y1:y2, x1:x2], 0.5)
        if timings is not None:
            timings["digit_crops"] = timings.get("digit_crops", 0) + 1
    if timings is not None:
        timings["digit_reading"] += perf_counter() - started
    if not digits:
        return None
    return "".join(digit.class_name for digit in sorted(digits, key=lambda digit: digit.bbox[0]))
