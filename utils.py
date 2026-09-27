"""Utilities for detecting race bibs and reading the numbers printed on them.

This module wraps OpenCV's Darknet (YOLO) DNN backend: :class:`Detector` builds
a network from a Darknet cfg/weights pair, and :func:`get_rbns` chains a bib
locator together with a digit reader to turn a single frame into bib numbers.
"""

from typing import Any

import cv2 as cv
import numpy as np
from attrs import frozen


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

    def detect(self, img: Any, conf: float) -> list[list[Any]]:
        """Detect objects in an image above a confidence threshold.

        Args
            img (numpy array): image array from openCV .imread
            conf (float): prediction confidence threshold

        Returns
            List of detections in the form [<class name>, [x, y, width, height]]
        """

        # Format the image for detection
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Get raw detections
        self.net.setInput(blob)
        outputs = self.net.forward(self.layer_names)

        h_img, w_img = img.shape[:2]

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
                    box = detection[:4] * np.array([w_img, h_img, w_img, h_img])
                    (center_x, center_y, width, height) = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Apply non-maximal suppression so each object is reported once
        indices = np.asarray(cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)).flatten()

        return [[self.classes[class_ids[i]], boxes[i]] for i in indices]


# Detectors are expensive to build (the weights are read from disk), so keep
# one instance per configuration and reuse it.
_DETECTOR_CACHE: dict[DetectorConfig, Detector] = {}


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

    detector = _DETECTOR_CACHE.get(config)
    if detector is None:
        detector = Detector(config)
        _DETECTOR_CACHE[config] = detector

    return detector


def get_rbns(
    img: Any, bib_detector_cfg: DetectorConfig, number_reader_cfg: DetectorConfig
) -> list[list[Any]]:
    """Return bib numbers and bib bounding boxes for the detected bibs.

    Args
        img (numpy array): image array given by openCV .imread
        bib_detector_cfg (DetectorConfig): model that locates bibs in the image
        number_reader_cfg (DetectorConfig): model that reads the digits inside a
            located bib

    Returns
        List of detected bib numbers and corresponding bounding boxes in
        the format [<bib number>, [x, y, width, height]]. A bib number of 0
        means a bib was located but no digit could be read inside it.
    """

    # Instantiate detectors (reused across calls, see get_detector)
    bib_detector = get_detector(bib_detector_cfg)
    number_reader = get_detector(number_reader_cfg)

    # Make bib location predictions
    bib_detections = bib_detector.detect(img, 0.25)

    bib_readings: list[list[Any]] = []
    for detection in bib_detections:
        bbox = detection[1]
        (x, y, w, h) = bbox

        # Crop out the bib so the digit reader only sees the bib itself
        crop_img = img[y : y + h, x : x + w]

        # Detect the digits inside the bib
        digit_detections = number_reader.detect(crop_img, 0.5)
        if digit_detections:
            # Sort the digits left to right and join them into a single number
            digits = sorted((digit[1][0], str(digit[0])) for digit in digit_detections)
            bib_number = int("".join(value for _, value in digits))
        else:
            bib_number = 0  # bib detection but no digit detection

        bib_readings.append([bib_number, bbox])

    return bib_readings
