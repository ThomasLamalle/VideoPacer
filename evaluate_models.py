"""Evaluate all bib detectors against a COCO test set (bib boxes only)."""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2 as cv
from attrs import evolve

import bib_detection
import detector

PARAMETERS = {
    "yolov4": {"input_size": 832, "confidence": 0.01, "model": "bundled YOLOv4-tiny"},
    "roboflow_2.0": {"input_size": 416, "confidence": 0.1, "model": "bib-detection/5"},
    "rfdetr-large-t1": {"input_size": 640, "confidence": 0.1, "model": "bib-detection/7"},
    "yolo26n-t1": {"input_size": 1024, "confidence": 0.01, "model": "bib-detection/8"},
    "yolo26n-v025": {"input_size": 832, "confidence": 0.01, "model": "BibBoxes v025 YOLO26n, OpenVINO"},
}

DEFAULT_DATASET = Path("/home/thomas/Projects/Datasets/test_dataset_rizvi.coco")
DEFAULT_RESULTS = Path("/home/thomas/Projects/Datasets/results")
IOU_THRESHOLDS = [round(0.5 + 0.05 * n, 2) for n in range(10)]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a, area_b = max(0, a[2]) * max(0, a[3]), max(0, b[2]) * max(0, b[3])
    return overlap / (area_a + area_b - overlap) if area_a + area_b > overlap else 0.0


def score(predictions, ground_truth, threshold, total_gt):
    """Greedy confidence-ranked one-to-one matching, then 101-point interpolated AP."""
    matched = defaultdict(set)
    tp = fp = 0
    curve = []
    for image_id, bbox, _confidence in sorted(predictions, key=lambda p: -p[2]):
        candidates = ground_truth[image_id]
        best = max(
            ((iou(bbox, gt), index) for index, gt in enumerate(candidates) if index not in matched[image_id]),
            default=(0.0, -1),
        )
        if best[0] >= threshold and best[1] >= 0:
            tp += 1
            matched[image_id].add(best[1])
        else:
            fp += 1
        curve.append((tp / (tp + fp), tp / total_gt))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    ap = sum(max((p for p, r in curve if r >= step / 100), default=0.0) for step in range(101)) / 101
    return {
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": total_gt - tp,
    }


def annotate(frame, truth, detections):
    annotated = frame.copy()
    boxes = [(bbox, (0, 255, 0), "GT") for bbox in truth]
    boxes += [(box.bbox, (0, 0, 255), f"{box.confidence:.3f}") for box in detections]
    for bbox, color, label in boxes:
        x, y, width, height = (round(value) for value in bbox)
        cv.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        cv.putText(annotated, label, (x, max(15, y - 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("test", "valid"), default="test")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save annotated images under <output-root>/<timestamp>_<dataset>/<model>/",
    )
    args = parser.parse_args()

    annotations = json.loads((args.dataset / args.split / "_annotations.coco.json").read_text(encoding="utf-8"))
    images = annotations["images"]
    ground_truth = defaultdict(list)
    for annotation in annotations["annotations"]:
        # ponytail: these single-class bib datasets use different label names; COCO id 0 is background.
        if annotation["category_id"] != 0 and not annotation.get("iscrowd", 0):
            ground_truth[annotation["image_id"]].append(annotation["bbox"])
    total_gt = sum(map(len, ground_truth.values()))
    if not images or not total_gt:
        raise ValueError(f"Dataset must contain {args.split} images and bib annotations")

    args.output_root.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d_%H%M%S}_{args.dataset.name}"
    if args.split != "test":
        stem += f"_{args.split}"
    path = args.output_root / f"{stem}.json"
    result = {
        "dataset": str(args.dataset.resolve()),
        "split": args.split,
        "images": len(images),
        "ground_truth_bibs": total_gt,
        "metrics": {
            "mAP": "Mean 101-point interpolated average precision across IoU 0.50 to 0.95 (step 0.05).",
            "precision": "At IoU 0.50: correctly matched detections / all detections (fewer false alarms is better).",
            "recall": "At IoU 0.50: correctly matched detections / annotated bibs (fewer misses is better).",
            "f1": "At IoU 0.50: harmonic mean of precision and recall.",
        },
        "comparison": {},
        "models": {},
    }
    try:
        for name, parameters in PARAMETERS.items():
            print(f"Running {name}...", flush=True)
            model = evolve(
                bib_detection.BIB_MODEL, input_size=parameters["input_size"], confidence=parameters["confidence"]
            )
            predictions = []
            image_dir = args.output_root / stem / name
            if args.images:
                image_dir.mkdir(parents=True, exist_ok=True)
            started = perf_counter()
            for image in images:
                image_path = args.dataset / args.split / image["file_name"]
                frame = cv.imread(str(image_path))
                if frame is None:
                    raise OSError(f"Cannot read image: {image_path}")
                boxes = detector.BIB_DETECTORS[name](frame, model)
                predictions.extend((image["id"], box.bbox, box.confidence) for box in boxes)
                if args.images:
                    annotated = annotate(frame, ground_truth[image["id"]], boxes)
                    destination = image_dir / image_path.name
                    if not cv.imwrite(str(destination), annotated):
                        raise OSError(f"Cannot write annotated image: {destination}")
            scores = {
                f"{threshold:.2f}": score(predictions, ground_truth, threshold, total_gt)
                for threshold in IOU_THRESHOLDS
            }
            at_50 = scores["0.50"]
            result["models"][name] = {
                "parameters": parameters,
                "mAP": round(sum(s["ap"] for s in scores.values()) / len(scores), 3),
                "AP50": round(at_50["ap"], 3),
                "precision": round(at_50["precision"], 3),
                "recall": round(at_50["recall"], 3),
                "f1": round(at_50["f1"], 3),
                "tp": at_50["tp"],
                "fp": at_50["fp"],
                "fn": at_50["fn"],
                "detections": len(predictions),
                "seconds": round(perf_counter() - started, 3),
            }
            result["comparison"][name] = {
                metric: result["models"][name][metric] for metric in ("mAP", "precision", "recall", "f1")
            }
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        print(path)


if __name__ == "__main__":
    main()
