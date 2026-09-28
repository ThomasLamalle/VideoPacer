# Bib reading fixes — handoff plan

**Goal:** turn the detector recall gain from `input_size=832` into correct bib strings.
The locator now finds runners much earlier, but the pipeline still reports only 8 of 13
ground-truth bibs, because the **digit reader and vote aggregation** are the bottleneck.

**Spec:** none. **Baseline clip:** `sample1/video.mp4` (523 frames), `sample1/ground_truth.csv`
(13 bibs, all 4–5 digits).

## Where things stand

`main` @ `988367c`: uniform scheduled detection (`if frame_id % detect_every == 0`),
`detect_every` default 20, `lost_track` deleted, `NEW_TRACK_CONFIDENCE` deleted.

Uncommitted working tree on top of that:

- `detector.DetectorConfig.input_size` (default 416) and `Detector` squashes to it.
- `bib_detection.BIB_INPUT_SIZE = 832`; `process_video(input_size=...)`; `--input-size`;
  `summary["input_size"]`.
- `detector.DIGIT_CROP_MARGIN = 0.5` — crop grown by half the box height before reading digits.
  Fixes the 5 fixture frames in `tests/frames/` (832 alone broke `bib_11461_frame_33800`,
  reading `11461` as `11401`). **Value is not well validated** — see Risks.

Measured end-to-end (`detect_every=20`, same clip):

| input size | tracks | correct bibs | spurious bib strings | wall |
|---|---|---|---|---|
| 416 | 12 | 7 | 1 | 47 s |
| 832 | 24 | 7 | 6 | 216 s |
| 832 + crop margin | 24 | **8** | 6 | 136 s |

Diagnosis: one track accumulated **15 distinct readings of the same bib**
(`T6`=6240 → `{'0','20','620','30','70','37','770','6740','640','7','60','740','6240','240','340'}`).
The correct reading usually wins on votes, but short junk readings still win their own tracks,
and ties discard real answers.

---

## Task 1 — Filter readings by a digit-count pattern (default 4–5)

**Why:** ground-truth bibs are 4–5 digits. Short readings are the main source of spurious tracks.

Evidence — every bib box across 52 frames at `input_size=832` (259 boxes, 202 readable):

| min digits | readings kept | correct | junk | junk removed |
|---|---|---|---|---|
| 1 (today) | 202 | 96 | 106 | — |
| 2 | 172 | 96 | 76 | 28% |
| 3 | 134 | **96** | 38 | 64% |
| 4 | 104 | 87 | 17 | 84% |
| 5 | 45 | 40 | 5 | 95% |
| 6 | 0 | 0 | 0 | — |

Readings that survive at confidence ≥ 0.5 include `'0'`, `'1'`, `'2'`, `'3'`, `'4'`, `'5'`, `'6'`,
`'7'`, `'9'`, `'16'`, `'17'`, `'24'`, `'30'`, `'40'`, `'46'`, `'60'`, `'70'`, `'74'`, `'87'`, `'99'`,
`'131'`, `'146'`, `'198'`, `'218'`, `'239'`, `'278'`, `'320'`, `'340'`, `'374'`, `'376'` — all of
which a 4-digit minimum removes.

**Do:**

- Add a supported digit range to `detector.read_bib` (suggested `BIB_DIGIT_RANGE = (4, 5)`).
  Reject a reading whose digit count falls outside it: return `None` so it is an *attempt*
  that adds no vote, exactly like an unreadable crop.
- Keep the located box. The track still covers the runner for its visibility window
  (Feature 8); it just does not vote. Do **not** use this to refuse track creation.
- Make the range configurable — it follows the race's bib format, not this clip.

**Watch out:** a minimum of 4 removes 9 readings that currently match a real bib. Those are
truncated fragments (e.g. `1146` for `11461`), so they may be redundant. Check end-to-end:
if the correct count drops, use 3 or keep both and prefer the longer reading.

**Acceptance:** `pytest` green; spurious bib strings 6 → ≤2; correct bibs stay ≥8;
no 1–3 digit string appears as any track's `best_bib`.

**This does not fix `1146` vs `11461`** — both are in range. Task 4 addresses that.

## Task 2 — Stop discarding on ties

**Why:** `Track.best_bib` returns `None` when the top two readings tie, so real answers are
thrown away. In `runs/input832-every20`: track 8 held `'3246': 5` tied with `'6': 5` → `None`;
track 20 held `'11116': 2` tied with `'1116': 2` → `None`. Both are ground-truth bibs.

**Do:** return the leader on a tie (`most_common` order is stable) and keep `conflicting` as the
warning flag. Run Task 1 first so a tie between a plausible and an implausible reading resolves
in favour of the plausible one.

**Acceptance:** `3246` and `11116` are reported; `conflicting` still true for them;
correct bibs ≥8 → expect ≥10.

## Task 3 — Do NOT restore `NEW_TRACK_CONFIDENCE` (answer: no)

Evidence — same 259 boxes at `input_size=832`:

| confidence ≥ | boxes | correct | junk |
|---|---|---|---|
| 0.1 | 259 | 96 | 106 |
| 0.3 | 240 | 95 | 100 |
| 0.5 | 212 | 93 | 90 |
| 0.7 | 181 | 90 | 80 |

Distribution by outcome: correct readings median **1.00** (97% ≥ 0.5); junk readings median
**0.98** (85% ≥ 0.5). The gate barely discriminates: at 0.5 it removes 16 of 106 junk readings
(15%) while losing 2 correct ones.

The locator is **highly confident about junk**. Its score says "a bib-like pattern is here", not
"this bib is readable". Leave the gate deleted; filter *readings* by pattern (Task 1) and weight
*votes* by reading confidence (Task 4) instead.

Note the trade you accepted: 57 boxes were unreadable even at confidence ≥0.1, and 29 at ≥0.5.
Those still create tracks, which is deliberate — they buy visibility coverage for review.

## Task 4 — Weight votes by reading confidence (highest value after 1–2)

Already in `features_todo.md`. It directly targets `1146`×2 outvoting `11461`×1: the truncated
read should carry a lower score. Investigate whether the digit reader's per-digit confidences
separate the two; if they do, weight each vote by (for example) the mean or product of its digit
scores. `detector.Detection.confidence` already carries the per-digit score, and
`read_bib` currently discards it.

A participant list (Feature 13) would settle `1146` vs `11461` outright and is the cheaper fix if
the list is available.

---

## Verification

```powershell
uv run pytest -q
uv run ruff format .
uv run ruff check .
uv run ty check

# fresh directory each run; the pipeline refuses to overwrite existing results
uv run python bib_detection.py --output-dir runs/<name>
```

Then score the run: load `runs/<name>/summary.json`, and for each track compare
`best_bib` against the bib strings in `sample1/ground_truth.csv`. Report three numbers —
tracks whose `best_bib` is in the ground truth, the `best_bib` values that are **not**
(spurious), and ground-truth bibs that appear in no `best_bib` (missing).

The scratch scripts that produced the numbers in this plan lived in `%TEMP%` and are gone.
Re-deriving the comparison is a few lines; consider committing it once it stabilises.

**Baseline to beat** — `runs/input832-margin-every20`, `detect_every=20`, `input_size=832`:

| metric | baseline | expected after Tasks 1–2 |
|---|---|---|
| correct bibs | 8 of 13 | ≥10 |
| spurious `best_bib` values | 6 | ≤2 |
| tracks | 24 | ~20 (fewer junk tracks) |

Missing at baseline: `11461`, `3241`, `3246`, `4460`, `6243`. Spurious at baseline:
`1143`, `1146`, `3`, `4`, `5`, `74`. Tasks 1–2 should remove the 1–2 digit ones and
recover `3246`.

## Risks / do not re-litigate

- **`input_size=832` is settled.** Recall peaks on a plateau at 768–896 and falls off on both
  sides (1024 and 1280 are *worse* and slower). Letterboxing is much worse (0 detections at
  frame 240) — the model was trained on squashed input. Reason: the cfg declares `width=416
  height=416` with absolute-pixel anchors, smallest in use `23×27`; a bib is ~14×18 px when
  1920×1080 is squashed to 416, below every anchor, and ~28×36 px at 832.
- **`DIGIT_CROP_MARGIN = 0.5` is a guess, not a measurement.** The padding landscape over 5
  fixtures is jagged (5/5 at 5 px, 4/5 at 10 px, 5/5 at 15–20 px, 0/5 at 30 px). Keep it for now,
  but sweep it on labelled footage (Feature 18) before treating it as tuned.
- **Timings on this machine drift badly** (416 measured at both 95 ms and 432 ms). Trust ratios
  only, and benchmark properly on target hardware (Feature 10).
- **Cadence recall is non-monotonic** — the scan grid aliases with each bib's brief legibility
  window, so `detect_every=10` scored worse than 15 on this clip. Never tune it on one clip.
- **~~The 5s detection delay~~ is explained and fixed.** Runners were found 1.7–7.5 s earlier at
  832, and `11116` was never found at 416 but is found at 832. Update that `features_todo.md`
  entry when this lands.
- `4460` is still never detected at any size — the one case where a better or retrained model is
  the answer, not the pipeline.
