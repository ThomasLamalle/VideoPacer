"""Detect bib boxes and read the digits inside them."""

import math
import os
import re
from functools import cache, partial
from pathlib import Path
from typing import Protocol

import cv2 as cv
import numpy as np
import openvino as ov
from attrs import frozen

type BBox = tuple[float, float, float, float]

MODELS = Path(__file__).resolve().parent / "bibobj"


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


def _crop(image: cv.typing.MatLike, bbox: BBox, margin: float) -> cv.typing.MatLike | None:
    """Cut a box out of the image, grown on every side by ``margin`` times its height.

    Return None when nothing of the box is inside the image.
    """
    x, y, width, height = (int(value) for value in bbox)
    image_height, image_width = image.shape[:2]
    grow = int(height * margin)
    left, top = max(0, x - grow), max(0, y - grow)
    right, bottom = min(image_width, x + width + grow), min(image_height, y + height + grow)
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


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
    crop = _crop(image, bbox, DIGIT_CROP_MARGIN)
    if crop is None:
        return None

    digits = reader.detect(crop, DIGIT_THRESHOLD)
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


# YOLO26n trained at 1280 on BibBoxes v025 (TrainBibDetector run 932e5ca1), exported to OpenVINO with any input
# shape allowed: yolo export model=best.pt format=openvino dynamic=True
YOLO26_MODEL = MODELS / "yolo26n_v025_openvino_model/y26n_v025.xml"
# A YOLO network shrinks the image by up to 32 times, so each side of its input must be a multiple of 32.
YOLO_STRIDE = 32


@cache
def get_openvino_model(model_path: Path, height: int, width: int) -> ov.CompiledModel:
    """Compile a model for one input shape, once.

    The model file accepts any shape. Fixing the shape before compiling lets OpenVINO plan for it, and the cache
    keeps one compiled model per shape.
    """
    core = ov.Core()
    model = core.read_model(model_path)
    model.reshape([1, 3, height, width])
    return core.compile_model(model, "CPU")


def find_openvino_bibs(
    image: cv.typing.MatLike, config: DetectorConfig, model_path: Path = YOLO26_MODEL
) -> list[Detection]:
    """Find bibs with a one-class Ultralytics YOLO model exported to OpenVINO.

    The frame is prepared as in training: scaled so its long side is ``config.input_size``, centred, and padded
    with gray to the next multiple of 32 on the short side. A 1920x1080 frame at 832 runs at 832x480, about half
    the pixels of an 832x832 square. The model returns many overlapping candidates, so overlapping boxes are
    reduced to the best one, as in the YOLOv4 detector.
    """
    height, width = image.shape[:2]
    scale = config.input_size / max(height, width)
    resized_width, resized_height = round(width * scale), round(height * scale)
    net_width = math.ceil(resized_width / YOLO_STRIDE) * YOLO_STRIDE
    net_height = math.ceil(resized_height / YOLO_STRIDE) * YOLO_STRIDE
    left, top = (net_width - resized_width) // 2, (net_height - resized_height) // 2
    canvas = np.full((net_height, net_width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = cv.resize(image, (resized_width, resized_height))
    blob = cv.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
    model = get_openvino_model(model_path, net_height, net_width)

    # One column per candidate: centre x, centre y, width, height and score, in network pixels.
    center_x, center_y, box_width, box_height, scores = model(blob)[0][0]
    keep = scores > config.confidence
    boxes = np.stack(
        [
            (center_x[keep] - box_width[keep] / 2 - left) / scale,
            (center_y[keep] - box_height[keep] / 2 - top) / scale,
            box_width[keep] / scale,
            box_height[keep] / scale,
        ],
        axis=1,
    ).tolist()
    kept_scores = scores[keep].tolist()
    kept = np.asarray(cv.dnn.NMSBoxes(boxes, kept_scores, config.confidence, 0.4)).flatten()
    return [Detection("bib", tuple(boxes[index]), kept_scores[index]) for index in kept]


# PaddleOCR's PP-OCRv6 small text recognizer, as shipped by RapidOCR 3.9, run with ONNX Runtime. It reads one line
# of text from a whole crop, so it needs no box per digit. Its alphabet is stored inside the model file.
OCR_MODEL = MODELS / "PP-OCRv6_rec_small.onnx"
# The recognizer was trained on text lines 48 px high, padded to at least 320 px wide. Narrower padding is
# faster but changed 8 of 40 test readings.
OCR_HEIGHT = 48
OCR_MIN_WIDTH = 320
# The crop is grown by this fraction of the box height, so digits at the edge of a box that optical flow has
# moved, or that the bib has outgrown while approaching, are not cut.
OCR_CROP_MARGIN = 0.1
# The recognizer gives one prediction per 8 px column of its 48 px high input. On a whole-bib crop the digits are
# much shorter than the crop, so a digit gets about two columns, and two equal digits side by side ("11") can merge
# into one. Stretching the crop sideways gives each digit more columns. On 40 labelled bib crops, 1.5 and 2 both
# read 27 of 28 readable bibs against 24 unstretched, and 1.5 did best on sample1.
OCR_STRETCH = 1.5
DIGITS = set("0123456789")


@cache
def get_ocr(model_path: Path) -> tuple[ov.CompiledModel, list[str]]:
    """Load the recognizer and its alphabet once. Index 0 is "no character" and the last entry is a space.

    OpenVINO runs the ONNX file directly, with the same outputs as ONNX Runtime and about half the time per crop on
    the Ryzen 5 5500U. The input width stays dynamic, since a few crops are wider than ``OCR_MIN_WIDTH``.
    """
    core = ov.Core()
    model = core.read_model(model_path)
    alphabet = ["", *model.get_rt_info(["framework", "character"]).astype(str).splitlines(), " "]
    return core.compile_model(model, "CPU"), alphabet


def read_bib_ocr(
    image: cv.typing.MatLike,
    bbox: BBox,
    _digit_model: DetectorLike,
    bib_pattern: str = BIB_PATTERN,
) -> BibReading | None:
    """Read a bib number with the text recognizer, or return None when no bib number is read.

    ``_digit_model`` is unused. It keeps the signature that every reader shares.

    For each column of the crop, the recognizer gives the probability of every character and of "no character".
    Keeping the most likely one per column, merging repeats and dropping "no character" gives the text. The
    longest run of digits in it is the bib number, so a letter or a word printed next to the number is ignored.
    ``confidence`` is the mean probability of those digits.
    """
    crop = _crop(image, bbox, OCR_CROP_MARGIN)
    if crop is None:
        return None
    resized_width = math.ceil(OCR_HEIGHT * crop.shape[1] / crop.shape[0] * OCR_STRETCH)
    batch = np.zeros((1, 3, OCR_HEIGHT, max(OCR_MIN_WIDTH, resized_width)), dtype=np.float32)
    # Pixels go to the range -1 to 1, and the padding stays at 0, as in training.
    batch[0, :, :, :resized_width] = cv.resize(crop, (resized_width, OCR_HEIGHT)).transpose(2, 0, 1) / 127.5 - 1

    model, alphabet = get_ocr(OCR_MODEL)
    probabilities = model(batch)[0][0]
    best = probabilities.argmax(axis=1)
    kept = (best != 0) & np.r_[True, best[1:] != best[:-1]]
    scores = probabilities.max(axis=1)[kept]
    # One character per kept column, with anything that is not a digit as "-", so positions match ``scores``.
    text = "".join(alphabet[index] if alphabet[index] in DIGITS else "-" for index in best[kept])
    runs = [match.span() for match in re.finditer(r"\d+", text)]
    if not runs:
        return None
    start, end = max(runs, key=lambda span: span[1] - span[0])
    if re.fullmatch(bib_pattern, text[start:end]) is None:
        return None
    return BibReading(text[start:end], float(scores[start:end].mean()))


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
    "yolo26n-v025": find_openvino_bibs,
}
BIB_READERS = {"yolov4": read_bib, "ppocr": read_bib_ocr}
