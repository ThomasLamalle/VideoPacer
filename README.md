# VideoPacer

VideoPacer detects race bibs near a finish line. The current experiment uses
occasional bib detection and optical flow to follow bibs between detections.

## Run it

Install the environment:

```powershell
uv sync
```

Process the complete sample video:

```powershell
uv run python bib_detection.py `
  --input sample1/video.mp4 `
  --output-dir runs/full-video
```

Use Roboflow bib-detection/5 for bib boxes (YOLO still reads digits). Set
`__ROBOFLOW_API_KEY__` in the environment first; no local server is needed:

```bash
uv run python bib_detection.py --input sample1/video.mp4 --bib-detector roboflow_2.0
uv run python bib_detection.py --input sample1/video.mp4 --bib-detector rfdetr-large-t1
uv run python bib_detection.py --input sample1/video.mp4 --bib-detector yolo26n-t1 --input-size 1024 --confidence 0.01
```

The last two names use `thomas-lamalle/bib-detection` versions 7 (RF-DETR Large)
and 8 (YOLO26n). The local inference library handles the model's resize and
maps boxes back to video coordinates. RF-DETR has a fixed 640×640 input; YOLO26n
supports `--input-size` (832 by default). The YOLO digit reader is unchanged.

Compare all bib detectors on one image, saving overlays and per-model JSON under
`frame_results_comparison/<timestamp>`:

```bash
uv run python compare_frame_models.py --image sample1/frame_04.jpg
```

Process only the first 150 frames:

```powershell
uv run python bib_detection.py `
  --max-frames 150 `
  --output-dir runs/short-video
```

Detection runs every 20 frames by default. A smaller interval finds more runners
and costs proportionally more detector time, because optical flow can only follow
the bibs an earlier scan located. Use `--detect-every` and `--read-every` to
change the intervals.

Each run creates:

- `annotated.mp4`, containing every processed frame;
- `summary.json`, containing the best bib, votes, first/last visible time and
  performance timings for each temporary track.

## Code

- `detector.py` detects bib boxes and reads their digits. Bib-box detection and digit
  reading are separate registries, `BIB_DETECTORS` and `BIB_READERS`; add an entry and
  pass its name via `--bib-detector` / `--digit-reader` to swap a backend.
- `tracker.py` follows boxes and stores bib votes.
- `bib_detection.py` reads the video, calls detection/tracking and writes results.
  `RunConfig` holds the schedule, locator tuning, bib pattern and chosen implementations.

The main loop is:

```text
read frame
→ follow existing tracks
→ detect on schedule
→ read tracked bibs when scheduled
→ draw and write the frame
```

A track ID is temporary. If a runner disappears and is detected again later,
the new detection may receive a new ID. Optical flow can also drift during
occlusion or fast motion. The reported first and last times describe visibility;
they are not precise finish times.

## Checks

```powershell
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run ty check
```
