# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Coding rules for every change: @AGENT.md

## Project

VideoPacer times runners from a finish-line video instead of chips. It detects race bibs, follows them with optical flow between detector passes, reads their numbers and votes per track. `features_todo.md` is the roadmap; its P1 to P3 items are the current priorities. The bib detector is trained in the sibling repo `~/Projects/TrainBibDetector`, on datasets in `~/Projects/Datasets`.

## Commands

```bash
uv sync
uv run python bib_detection.py --input sample1/video.mp4 --note "what this run tries"   # full run, outputs in runs/<timestamp>/
uv run python bib_detection.py --max-frames 150 --output-dir runs/short                 # quick partial run
uv run pytest                                                            # all tests, about 20 s
uv run pytest tests/test_tracker.py::test_optical_flow_moves_a_track    # one test
uv run pytest tests/test_full_video.py                                   # golden test on the full sample video
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pre-commit install    # hooks: file hygiene, ruff, ty, and only the golden test
```

## Local files the code needs

`bibobj/` (model files, sources in README.md) and `sample1/video.mp4` are not in git. Without them the default pipeline, `tests/test_full_video.py` and `tests/test_bib_detection.py` fail. The YOLOv4 SVHN digit model in `bibobj/` is loaded even when the `ppocr` reader is used, because every reader receives a digit model argument.

## How a run works

`bib_detection.process_video` reads every frame and:

1. `Tracker.follow` moves each active track by the median Lucas-Kanade motion of its points, on half-size grayscale frames, keeping only points that pass a forward-backward check. A track that loses its points is deactivated.
2. Every `detect_every` frames (default 20), `detect_bibs` runs the bib detector and reads every box. `Tracker.correct` then matches boxes to tracks by IoU of at least 0.3, finds new corners at full size, and adds the readings as votes. Only a box whose number was read starts a new track.
3. Every `read_every` frames (default 10), `Tracker.read` reads the active tracks the detector did not just read.
4. A track's votes sum reading confidences. `Track.best_bib` multiplies them by 5 per digit, because the reader drops one of two repeated digits far more often than it invents one.

The run ends with `summary.json`, a comparison with a `ground_truth.csv` beside the input video, and, for a whole-video CLI run, a `performance.md` report and a new row in `performance_history.csv`.

Backends are picked by name from `detector.BIB_DETECTORS` and `detector.BIB_READERS` (`RunConfig.bib_detector` / `digit_reader`, CLI `--bib-detector` / `--digit-reader`). The defaults are `yolo26n-v025`, a YOLO26n through OpenVINO letterboxed to a multiple of 32 with NMS in code because the export is not end-to-end, and `ppocr`, the PP-OCRv6 recognizer through ONNX Runtime with greedy CTC decoding, the longest digit run, and crops stretched 1.5 times horizontally. The Roboflow backends need `ROBOFLOW_API_KEY`. `evaluate_models.py` scores the detectors on a COCO test set under `~/Projects/Datasets`.

## Performance work

- Before trying an idea, read `performance_history.csv`. Many settings were already measured on sample1 and did not beat the current defaults: detector input 1024 and 1280, wider OCR margins, other stretches, shorter detect and read intervals, quarter-size optical flow.
- Record each trial as a whole-video CLI run with `--note`; runs with `--max-frames` are not recorded. The CSV header is fixed, so changing the columns in `record_performance` needs a new file.
- Timings on the development laptop drift up to twice with battery and heat. Compare runs back to back and check the `power` column.
- The golden test expects all 13 sample1 bibs and no wrong number. A change that alters results keeps it passing or updates it on purpose. The 11461 case in `tests/test_bib_detection.py` is a strict xfail that fails once that frame reads right.
- Tuning constants in `tracker.py` and `detector.py` carry comments with the measurements that chose them; keep that up when changing them.

## Gotchas

- Line endings differ per file: `bib_detection.py`, `detector.py`, `tracker.py`, `features_todo.md` and most tests use CRLF, while `README.md` and `evaluate_models.py` use LF. The Edit tool keeps them; a script that rewrites a file in text mode turns CRLF into LF and produces a whole-file diff.
- OpenVINO sends usage data each time it is imported unless declined: `mkdir -p ~/intel && printf 0 > ~/intel/openvino_telemetry`.
