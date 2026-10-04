# VideoPacer

**Race-bib detection and tracking in finish-line video.** VideoPacer combines periodic bib detection, digit recognition, and optical-flow tracking to follow runners between detector passes. It aggregates repeated readings into a best bib number and exports an annotated video plus machine-readable results.

## Demo

![Annotated race-bib detection and tracking demo](sample1/annotated-demo.gif)

Boxes are labeled with a temporary track ID and the current best bib reading. `DETECT` marks a detector update, `READ` a digit-reading update, `FLOW` an optical-flow-only frame, and `BOTH` a frame with both updates. The GIF is a reduced-size preview; the pipeline writes full-resolution MP4 output.


## Run

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). The sample input video and local Darknet model weights are not included in this repository. Use your own video; the default backend also expects its model files under `bibobj/` (see `bib_detection.py`).

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
