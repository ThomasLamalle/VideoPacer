from pathlib import Path
from typing import Any

import cv2 as cv
from attrs import frozen
from loguru import logger

import utils as ut

# Bib detection model config (RBNR: locates the bib on a runner)
BD_CONFIG = ut.DetectorConfig(
    cfg="./bibobj/RBNR_custom-yolov4-tiny-detector.cfg",
    weights="./bibobj/RBNR_custom-yolov4-tiny-detector_best.weights",
    classes=("bib",),
)

# Number reader config (SVHN: reads the digits inside a detected bib)
NR_CONFIG = ut.DetectorConfig(
    cfg="./bibobj/SVHN_custom-yolov4-tiny-detector.cfg",
    weights="./bibobj/SVHN_custom-yolov4-tiny-detector_best.weights",
    classes=tuple(str(digit) for digit in range(10)),
)


@frozen
class BibDetection:
    frame_id: int
    bbox: Any
    bib_string: str


@frozen
class Bib:
    bib_string: str
    detections: list[BibDetection]
    locked: bool = False

    @classmethod
    def from_bib_detection(cls, detection: BibDetection) -> Bib:
        return Bib(detection.bib_string, [detection])


def detect_bibs(frame: Any, frame_id: int) -> list[BibDetection]:
    """Detect bibs on a single frame and read the numbers inside them."""
    results = ut.get_rbns(frame, BD_CONFIG, NR_CONFIG)

    if not results:
        return []

    detections: list[BibDetection] = []
    for bib_number, bbox in results:
        # A bib number of 0 means a bib was located but no digit could be read
        # inside it, so it cannot be matched to a runner.
        if not bib_number:
            continue
        detections.append(BibDetection(frame_id=frame_id, bbox=bbox, bib_string=str(bib_number)))

    return detections


def display_bib_detections(frame: Any, bib_detections: list[BibDetection]) -> Any:
    """Draw bib bounding boxes and their readings on a copy of the frame.

    Args
        frame (numpy array): frame from openCV
        bib_detections (list[BibDetection]): detections to display

    Returns
        Copy of the frame with a bounding box and the read bib number drawn
        on top of each detection
    """

    annotated_frame = frame.copy()
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (0, 255, 0)  # BGR green

    for bib_detection in bib_detections:
        (x, y, w, h) = bib_detection.bbox[:4]
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Draw the bib bounding box
        cv.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)

        # Place the reading label above the box, or below it if there is no
        # room at the top of the frame
        label = bib_detection.bib_string
        (_, text_h), _ = cv.getTextSize(label, font, font_scale, thickness)
        text_y = y - 10 if y - text_h - 10 >= 0 else y + h + text_h + 10
        text_y = min(max(text_y, text_h), annotated_frame.shape[0] - 1)

        # Draw a dark outline first so the label stays readable on any frame
        cv.putText(annotated_frame, label, (x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
        cv.putText(annotated_frame, label, (x, text_y), font, font_scale, color, thickness)

    return annotated_frame


def main() -> None:
    video_path = Path(__file__).resolve().parent / "sample1" / "video.mp4"

    only_inspect_x_frames = 100
    output_path = video_path.with_name(
        f"{video_path.stem}_annotated_{only_inspect_x_frames}_fps.mp4"
    )

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.info(f"Can't open video {video_path}. Exiting ...")
        return

    # Read the source properties so the saved video matches the input
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    writer = cv.VideoWriter(
        str(output_path),
        cv.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        logger.info(f"Can't open video writer for {output_path}. Exiting ...")
        cap.release()
        return

    bibs_dict: dict[str, Bib] = {}
    frame_id = 0
    while cap.isOpened():
        if frame_id % only_inspect_x_frames != 0:
            frame_id += 1
            continue

        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            logger.info("Can't receive frame (stream end?). Exiting ...")
            break

        bib_detections = detect_bibs(frame, frame_id)

        # Annotate the frame with the detections and save it to the video. The
        # results go to a video file rather than an interactive preview window.
        annotated_frame = display_bib_detections(frame, bib_detections)
        writer.write(annotated_frame)

        for bib_detection in bib_detections:
            logger.info(f"Found {bib_detection.bib_string} bibs on frame {frame_id}")
            if bib_detection.bib_string not in bibs_dict:
                bibs_dict[bib_detection.bib_string] = Bib.from_bib_detection(bib_detection)
            else:
                bibs_dict[bib_detection.bib_string].detections.append(bib_detection)

        frame_id += 1

    cap.release()
    writer.release()

    logger.info(f"Saved annotated video to {output_path}")


if __name__ == "__main__":
    main()
