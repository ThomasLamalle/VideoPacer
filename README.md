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

Process only the first 150 frames:

```powershell
uv run python bib_detection.py `
  --max-frames 150 `
  --output-dir runs/short-video
```

Detection runs every 100 frames by default. Optical flow runs on every frame.
If optical flow loses a track, detection runs immediately on that frame. Use
`--detect-every` and `--read-every` to change the intervals.

Each run creates:

- `annotated.mp4`, containing every processed frame;
- `summary.json`, containing the best bib, votes, first/last visible time and
  performance timings for each temporary track.

## Code

- `detector.py` detects bib boxes and reads their digits.
- `tracker.py` follows boxes and stores bib votes.
- `bib_detection.py` reads the video, calls detection/tracking and writes results.

The main loop is:

```text
read frame
→ follow existing tracks
→ detect on schedule or after a lost track
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
