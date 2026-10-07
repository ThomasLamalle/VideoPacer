# VideoPacer

**Race-bib detection and tracking in finish-line video.** VideoPacer combines periodic bib detection, digit recognition, and optical-flow tracking to follow runners between detector passes. It aggregates repeated readings into a best bib number and exports an annotated video plus machine-readable results.

## Demo

![Annotated race-bib detection and tracking demo](sample1/annotated-demo.gif)

Boxes are labeled with a temporary track ID and the current best bib reading. `DETECT` marks a detector update, `READ` a digit-reading update, `FLOW` an optical-flow-only frame, and `BOTH` a frame with both updates. The GIF is a reduced-size preview; the pipeline writes full-resolution MP4 output.


## Run

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). The sample input video and the model files are not included in this repository. Use your own video. The default detector and reader expect their model files under `bibobj/`:

- `yolo26n_v025_openvino_model/`: the bib detector, a YOLO26n trained on BibBoxes v025 and exported from TrainBibDetector with `yolo export model=best.pt format=openvino dynamic=True`.
- `PP-OCRv6_rec_small.onnx`: PaddleOCR's text recognizer as shipped by RapidOCR 3.9, from [ModelScope](https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx) (SHA-256 `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884`).

The older YOLOv4 detector and SVHN digit reader still work with `--bib-detector yolov4 --digit-reader yolov4` and their Darknet files in the same folder.

OpenVINO sends anonymous usage data to Intel each time it is imported, unless declined. To decline it once for your user account:

```bash
mkdir -p ~/intel && printf 0 > ~/intel/openvino_telemetry
```

```bash
uv sync
uv run python bib_detection.py --input /path/to/video.mp4 --output-dir runs/sample
```

The output directory contains `annotated.mp4` and `summary.json`. To process only an initial portion or tune the schedules:

```bash
uv run python bib_detection.py \
  --input sample1/video.mp4 \
  --max-frames 150 \
  --detect-every 20 \
  --read-every 10 \
  --output-dir runs/short
```

To use the hosted Roboflow bib detector, provide `__ROBOFLOW_API_KEY__` in the environment and select a backend:

```bash
export ROBOFLOW_API_KEY="your-key"
uv run python bib_detection.py --input /path/to/video.mp4 --output-dir runs/roboflow --bib-detector roboflow_2.0
```

## Project layout

- `bib_detection.py` — CLI, frame-processing pipeline, output and run configuration.
- `detector.py` — bib localization and digit-reading backends.
- `tracker.py` — optical-flow tracking, matching, and vote aggregation.
- `tests/` — unit and video-pipeline tests.

## Development checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
```
