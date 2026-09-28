# Simple Bib Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current tracking experiment with a concise three-file pipeline that reports the best bib and first/last visible time for each track.

**Architecture:** `detector.py` handles model inference, `tracker.py` handles optical flow and track state, and `bib_detection.py` handles video input/output. Detection runs on schedule or immediately after flow loses a track; there is no recovery scheduler or retry state.

**Tech Stack:** Python 3.14, OpenCV, NumPy, attrs, pytest, Ruff, ty, uv

**Spec:** `docs/superpowers/specs/2026-09-28-simple-bib-tracking-design.md`

## Global Constraints

- Do not use `Any` type annotations.
- Keep one `Track` data class; do not add a separate history or scheduler class.
- Keep performance timing for detection, optical flow, digit reading, video writing and total runtime.
- Write only `annotated.mp4` and `summary.json`.
- Preserve leading zeroes in bib readings and retain unreadable detections.
- Prefer short names that state their subject; docstrings must identify what is returned or changed.

## Review Focus

- A frame where optical flow fails must run detection immediately on that same frame.
- An unsuccessful recovery scan must wait for the next regular detection frame.
- An unreadable bib detection must still create or refresh a track without adding a vote.
- Equal top vote counts must produce no best bib and set the conflict flag.
- A short or invalid video must close OpenCV capture/writer resources cleanly.

---

### Task 1: Concise detector API

**Files:**
- Modify: `detector.py`
- Modify: `tests/test_detector_contract.py`

**Interfaces:**
- Produces: `BBox`, `BibDetection`, `DetectorConfig`, `get_detector()`, `detect_bibs()`, and `read_bib()`.
- `BibDetection.number` remains `str | None` and preserves leading zeroes.

- [ ] **Step 1: Rewrite detector contract tests**

Cover confidence retention, nonmaximum suppression, crop clipping, unreadable bibs and leading zeroes using concrete NumPy/OpenCV types.

- [ ] **Step 2: Run detector tests and confirm they fail against the old API**

Run: `uv run pytest tests/test_detector_contract.py -v`

- [ ] **Step 3: Rewrite `detector.py` around the small public API**

Use `cv.typing.MatLike` for images. Keep only comments explaining confidence calculation, box clipping or model caching.

- [ ] **Step 4: Run detector tests**

Run: `uv run pytest tests/test_detector_contract.py tests/test_bib_detection.py -v`
Expected: all pass.

- [ ] **Step 5: Commit the detector rewrite**

Commit only `detector.py` and its tests.

### Task 2: One track model and optical flow

**Files:**
- Create: `tracker.py`
- Delete: `optical_flow.py`
- Replace: `tests/test_optical_flow.py` with `tests/test_tracker.py`

**Interfaces:**
- Consumes: `BBox`, `BibDetection`, and `read_bib()` from Task 1.
- Produces: `Track`, `Tracker.follow(frame, frame_id) -> bool`, `Tracker.correct(frame, detections, frame_id) -> None`, `Tracker.read(frame, frame_id, reader) -> None`, and `Tracker.results(fps) -> list[dict[str, object]]`.

- [ ] **Step 1: Write tracker behavior tests**

Test translated textured boxes, flow loss, immediate active-track removal, detection matching, unreadable detections, votes, ties, conflicts and first/last frames.

- [ ] **Step 2: Run tracker tests and confirm they fail before `tracker.py` exists**

Run: `uv run pytest tests/test_tracker.py -v`

- [ ] **Step 3: Implement `Track` and `Tracker`**

Keep optical flow in one private function with explicit rejection checks. Keep all track state in `Track`; completed tracks stay in `Tracker.tracks` and use an `active` flag.

- [ ] **Step 4: Run tracker tests**

Run: `uv run pytest tests/test_tracker.py -v`
Expected: all pass.

- [ ] **Step 5: Commit the tracker rewrite**

Commit `tracker.py`, deletion of `optical_flow.py`, and tracker tests.

### Task 3: Direct video loop and outputs

**Files:**
- Rewrite: `bib_detection.py`
- Rewrite: `tests/test_video_pipeline.py`
- Modify: `README.md`
- Remove: `docs/optical-flow-validation.md`

**Interfaces:**
- Consumes: detector and tracker APIs from Tasks 1 and 2.
- Produces: `process_video(input_path, output_dir, detect_every=100, read_every=10, max_frames=None) -> dict[str, object]` and the CLI.

- [ ] **Step 1: Write pipeline tests for the agreed loop**

Test scheduled detection, same-frame recovery detection, no repeated recovery after an empty result, full output-frame count, first/last seconds and stage timings.

- [ ] **Step 2: Run pipeline tests and confirm they fail against the old loop**

Run: `uv run pytest tests/test_video_pipeline.py -v`

- [ ] **Step 3: Rewrite `bib_detection.py` as one direct loop**

Keep the loop in this order: read frame, follow tracks, decide detection with one Boolean expression, correct tracks, optionally read crops, draw, write. Build `summary.json` after the loop. Do not create per-frame observations.

- [ ] **Step 4: Shorten project documentation**

Document the three files, the loop, the full-video command and limitations. Remove documentation for deleted records and retry behavior.

- [ ] **Step 5: Run the complete verification suite**

Run:

```powershell
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

Expected: all pass.

- [ ] **Step 6: Run a short sample-video smoke test**

Run the first 150 frames with detection every 100 frames. Verify the annotated video has 150 frames and the summary contains tracks and timing values.

- [ ] **Step 7: Commit the pipeline rewrite**

Commit `bib_detection.py`, pipeline tests, README and removal of obsolete documentation.
