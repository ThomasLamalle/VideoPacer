# Sparse optical-flow experiment — 2026-09-28

Replaced motion prediction with OpenCV image-feature tracking. Removed the
ByteTrack adapter, its tests, and the `trackers`/`supervision` dependencies.
The previous implementation is backed up locally under `.cache/bytetrack-backup/`;
its saved runs remain under `runs/`. Those directories are ignored by Git.

## What changed

The bib detector initializes boxes and up to 60 Shi–Tomasi points inside each
box. Pyramidal Lucas–Kanade follows points in consecutive grayscale frames.
Forward/backward disagreement over 1.5 pixels rejects a point. At least six
points must support a robust similarity transform, with at least 60% inliers.
The box center and size follow that transform; per-frame scales outside
0.8–1.25 are rejected. Tracking stops when these checks fail.

Full-frame detection runs every 100 frames by default and also after failure or
when no tracks remain, with a retry interval of 10 frames. Scans correct boxes
by one-to-one overlap matching (IoU >= 0.3); new tracks require confidence >= 0.5.
Weaker boxes can correct an existing track. Lost identities are not automatically
reconnected. An empty scan does not invalidate good optical flow, but any track
without a detector correction for over 200 frames expires to limit drift.

Digit reading has a separate interval, default 10 frames. It reads actual tracked
crops, including on frames without a full detector scan. Each track gets at most
one reading/vote per frame. Leading zeros, unknown readings and conflicting
strings are preserved. Votes are counts, not independent confidence estimates.

Code is organized as `detector.py` (models and crop reading), `optical_flow.py`
(point tracking, box association and votes), and `bib_detection.py` (video loop,
scan scheduling and outputs). OpenCV and NumPy were already in use; the new
tracker requires no extra package.

## Same five-second sample

All rows below use the first 150 source frames of `sample1/video.mp4`:
1920×1080, 29.970031 FPS, 5.005 seconds. The previous baseline is the saved run
from September 27; new runs are from September 28. CPU: AMD Ryzen 5 5500U,
Windows 11, Python 3.14.5, OpenCV 4.14.0, 12 OpenCV threads, CPU DNN inference.

| Run | Full detector scans | Temporary IDs | Wall time | Tracking time | Output time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Saved ByteTrack, interval 10 | 15 | 8 | 17.70 s | 0.13 s | 7.09 s |
| Optical flow, interval 10 | 15 | 3 | 12.17 s | 4.06 s | 3.37 s |
| Optical flow, interval 100 + recovery | 5 | 2 | 9.84 s | 3.14 s | 3.32 s |
| Same interval 100, no annotated video | 5 | 2 | 6.52 s | 3.12 s | 0.008 s |

Single runs are not a controlled speed benchmark: decoding, model inference and
encoding also varied between days. The lower wall time must not be attributed
to faster tracking. Optical flow itself costs substantially more CPU than the
old motion predictor. It supplies image-based position updates that the predictor
could not supply. The interval-100 run also does fewer detector scans than the
interval-10 runs, and the new pipeline reads additional tracked crops.

The final default run spent 1.33 s decoding, 0.52 s locating bibs, 1.34 s reading
digits, 3.14 s tracking and 3.32 s writing output. Without video output it processed
about 23 FPS, below the source's 29.97 FPS. Model loading is included in overall
pipeline time; Python/module startup and final summary writing are excluded.

## Quality observed

**Interval 10:** bib 13426 stays T0 from frame 0 through 62, with five correct
readings and two partial readings (`426`, `34`). Bib 6500 stays T1 from frame
100 through 149 with five correct readings. The saved ByteTrack run split those
same visible runners into two IDs and four IDs, respectively.

The runner wearing 8717 stays T2 from frame 110 through 149, but its readings are
`81`, `871`, `17`, `8717`, once each. The final provisional identity is therefore
unknown (a four-way tie). Stable position tracking does not guarantee correct OCR.
There are three correct bib strings somewhere in the evidence, but only two
resolved provisional identities. The larger 13-bib list is not a suitable recall
denominator for this short segment.

**Interval 100:** detector scans occur at frames 0 and 100, plus recovery scans
at 66, 76 and 86. T0 follows bib 13426 through frame 65, then ends near the bottom
of the image. Its readings vote 13426 five times, with `426` and `345` once each.
T1 begins on the recovery scan at 86 and follows bib 6500 through frame 149, with
seven correct readings. Bib 8717 is visible but never initialized in this run:
optical flow cannot discover runners without a detector box. The smaller ID count
is therefore not by itself better coverage than interval 10.

Inspected annotated frames at 0, 20, 40, 60, 80, 100, 120 and 149 show boxes moving
with the bibs instead of remaining at their initial locations. Each run recorded
one ended track; that runner exits the lower image, so this count should not be
interpreted as one erroneous loss. No cross-runner identity transfer was established
in the inspected frames. There are no frame-level identity annotations, so this
is a qualitative continuity check, not a formal tracking-accuracy measurement.

## Reproduce and inspect

```powershell
uv run python bib_detection.py --detect-every 100 --read-every 10 --max-frames 150 --output-dir runs/flow-final-every100-5s
uv run python bib_detection.py --detect-every 10 --read-every 10 --max-frames 150 --output-dir runs/flow-final-every10-5s
uv run python bib_detection.py --detect-every 100 --read-every 10 --max-frames 150 --no-video --output-dir runs/flow-final-every100-5s-no-video
```

Choose new directories when rerunning: existing results are not overwritten.
The first two directories contain `annotated.mp4`, `observations.jsonl`,
`summary.json`, and `inspection-*.jpg` contact sheets. Both saved videos were
decoded again and contain 150 readable frames at 29.97 FPS. Each JSONL also has
150 records. The no-video run preserves the same track histories and readings.

Green boxes were corrected by the detector on the current frame; cyan boxes
follow image features. `source` and `reading_ran` in each track record distinguish
box tracking from a crop-reading attempt. `digit_crops` counts attempts that actually
reached the digit model. First/last seen timestamps include
accepted optical-flow positions; the last detector confirmation is stored separately.
These are source-video times, not finish-time estimates. Variable-frame-rate and
live capture would require capture timestamps rather than frame index / FPS.

## Verification and remaining limits

All 25 tests passed. Ruff lint, Ruff formatting checks and ty type checks passed,
with all checks run through uv.

Tests cover translation, scale, disappearance, low-texture failure, weak detector
correction, independent IDs, conflicts, repeated-reading protection, periodic scans,
crop reading between scans, recovery retry limits, survival through empty scans,
bounded track lifetime, output frame counts and timestamps. The original five
real-image detection tests are retained. Independent review identified an empty
recovery retry bug; it was reproduced with two active tracks and fixed.

This short trial supports keeping optical flow for further evaluation: it improves
continuity and makes multiple reads useful. It does not establish reliable crowd
tracking or real-time performance. Occlusion can still mix feature points, boxes
can drift or become partly cropped, and periodic detection can miss new runners.
Frame differencing, digit reconstruction and automatic identity locking were not added.

References: [OpenCV optical flow](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
and [similarity-transform estimation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).
