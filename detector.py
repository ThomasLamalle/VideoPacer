"""Detect bib boxes and read the digits inside them."""

import os
import re
from functools import cache, partial
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
class BibReading:
    """A bib number read from a box and how sure the digit reader was.

    ``confidence`` is the mean of the per-digit scores, so a confident reading casts a
    heavier vote than an unsure one.
    """

    bib_string: str
    confidence: float


@frozen
class BibDetection:
    """A bib box and the number read inside it.

    An unreadable box keeps its place with a None ``bib_string`` and zero confidence.
    """

    bbox: BBox
    bib_string: str | None
    confidence: float


@frozen
class DetectorConfig:
    """Darknet model files, class names and the network input size to run at.

    ``input_size`` is the square size every frame is squashed to before inference.
    It can differ from the ``width``/``height`` declared in the cfg, but the anchors
    stay absolute pixel sizes, so raising it rescales the objects the network sees.
    A cfg trained at 416 hands its smallest anchor to objects that shrink far below
    it once a widescreen frame is squashed, and those objects match no anchor at all.

    ``confidence`` is the score below which a detection is dropped. It belongs to the
    model, not the pipeline: a different backend needs its own threshold.
    """

    cfg: str
    weights: str
    classes: tuple[str, ...]
    input_size: int = 416
    confidence: float = 0.1


class DetectorLike(Protocol):
    def detect(self, image: cv.typing.MatLike, confidence: float, /) -> list[Detection]: ...


class Detector:
    def __init__(self, config: DetectorConfig) -> None:
        self.classes = config.classes
        self.input_size = config.input_size
        self.net = cv.dnn.readNetFromDarknet(config.cfg, config.weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        names = self.net.getLayerNames()
        self.layer_names = [names[index - 1] for index in self.net.getUnconnectedOutLayers()]

    def detect(self, image: cv.typing.MatLike, confidence: float) -> list[Detection]:
        """Return model detections in image coordinates."""
        size = (self.input_size, self.input_size)
        blob = cv.dnn.blobFromImage(image, 1 / 255.0, size, swapRB=True, crop=False)
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
                center_x, center_y, box_width, box_height = (raw_detection[:4] * image_scale).astype(int)
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
        return [Detection(self.classes[class_ids[index]], tuple(boxes[index]), scores[index]) for index in kept]


@cache
def get_detector(config: DetectorConfig) -> Detector:
    """Load a model once and reuse it."""
    return Detector(config)


# The digit model was trained on SVHN crops, where a digit fills roughly two thirds
# of its 32x32 window. A bib box is tight around the digits, so the crop is grown by
# this fraction of the box height before reading. Without it the same bib reads as a
# different number from frame to frame, because the digits fill the whole crop.
DIGIT_CROP_MARGIN = 0.5

# A race bib number has a fixed shape. A reading that does not look like a bib number is a
# fragment of a partly visible bib or a couple of stray digits, so it is not treated as a
# bib number. The pattern is a regex matched against the whole reading, so a race whose
# bibs mix digits and letters only needs the pattern changed, e.g. r"[A-Z0-9]{4,5}".
BIB_PATTERN = r"\d{4,5}"

# The digit reader's own detection threshold, for the same reason as DetectorConfig.confidence.
DIGIT_THRESHOLD = 0.5


def read_bib(
    image: cv.typing.MatLike,
    bbox: BBox,
    reader: DetectorLike,
    bib_pattern: str = BIB_PATTERN,
) -> BibReading | None:
    """Return the digits inside a bib box, or None when none can be read.

    ``confidence`` is the mean score of the digits read. A reading that does not match
    ``bib_pattern`` reads as None, so it counts as an attempt that found nothing and
    casts no vote for a bib number.
    """
    x, y, width, height = (int(value) for value in bbox)
    image_height, image_width = image.shape[:2]
    margin = int(height * DIGIT_CROP_MARGIN)
    left, top = max(0, x - margin), max(0, y - margin)
    right, bottom = min(image_width, x + width + margin), min(image_height, y + height + margin)
    if right <= left or bottom <= top:
        return None

    digits = reader.detect(image[top:bottom, left:right], DIGIT_THRESHOLD)
    if not digits:
        return None
    digits.sort(key=lambda digit: digit.bbox[0])
    bib_string = "".join(digit.class_name for digit in digits)
    if re.fullmatch(bib_pattern, bib_string) is None:
        return None
    confidence = sum(digit.confidence for digit in digits) / len(digits)
    return BibReading(bib_string, confidence)


def find_bibs(image: cv.typing.MatLike, config: DetectorConfig) -> list[Detection]:
    """Find bib boxes in one video frame."""
    return get_detector(config).detect(image, config.confidence)


@cache
def get_roboflow_detector(model_id: str):
    """Load and cache a Roboflow model by its version ID."""
    from inference import get_model  # noqa: PLC0415 -- keep the heavy backend optional for YOLO runs.

    return get_model(model_id, api_key=os.environ["__ROBOFLOW_API_KEY__"])


def find_roboflow_bibs(
    image: cv.typing.MatLike, config: DetectorConfig, model_id: str = "bib-detection/5"
) -> list[Detection]:
    """Use model preprocessing and convert its boxes to video coordinates."""
    options = {"image_size": config.input_size} if model_id == "bib-detection/8" else {}
    response = get_roboflow_detector(model_id).infer(image, confidence=config.confidence, **options)[0]
    # Version 8's training labels include bibs as '0' and 'tegnumber' (race number).
    bib_classes = {"bib", "0", "tegnumber"} if model_id == "bib-detection/8" else {"bib"}
    return [
        Detection(
            prediction.class_name,
            (
                prediction.x - prediction.width / 2,
                prediction.y - prediction.height / 2,
                prediction.width,
                prediction.height,
            ),
            prediction.confidence,
        )
        for prediction in response.predictions
        if prediction.confidence > config.confidence and prediction.class_name.lower() in bib_classes
    ]


def read_bibs(
    image: cv.typing.MatLike,
    boxes: list[Detection],
    digit_config: DetectorConfig,
    bib_pattern: str = BIB_PATTERN,
    digit_reader: str = "yolov4",
) -> list[BibDetection]:
    """Read the number inside each detected bib box.

    Boxes whose digits cannot be read, or read as something that is not a bib number,
    keep their box with a None ``bib_string``.
    """
    if not boxes:
        return []
    reader = get_detector(digit_config)
    read_crop = BIB_READERS[digit_reader]
    results = []
    for box in boxes:
        reading = read_crop(image, box.bbox, reader, bib_pattern)
        results.append(
            BibDetection(box.bbox, reading.bib_string, reading.confidence)
            if reading is not None
            else BibDetection(box.bbox, None, 0.0)
        )
    return results


# Swap implementations here: a detector takes (image, config), a reader takes
# (image, bbox, digit_reader, bib_pattern). Add an entry and pass its name to RunConfig.
BIB_DETECTORS = {
    "yolov4": find_bibs,
    "roboflow_2.0": find_roboflow_bibs,
    "rfdetr-large-t1": partial(find_roboflow_bibs, model_id="bib-detection/7"),
    "yolo26n-t1": partial(find_roboflow_bibs, model_id="bib-detection/8"),
}
BIB_READERS = {"yolov4": read_bib}
