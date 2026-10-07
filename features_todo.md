c# Features

The first target is a small race that accepts approximate finish results, filmed with a phone or a webcam. Processing speed is a core priority, because live processing remains the goal. Measure every change on throughput and result latency as well as on missed runners and wrong readings.

Conventions:
- Numbers are stable IDs, not an order. Items are independent unless they say "depends on" or "uses".
- X marks an investigation: brainstorm or analyze it before deciding to build it.

## Delivery focus

Keep measuring and improving speed alongside recognition and result review. The first useful workflow is participant import, finish video processing, approximate results, review of uncertain passages, and corrected export. Add live input once throughput and latency are measured as sufficient on the intended hardware. Checkpoint support and spectator features come later, but P1 to P3 are built in from the start so they need no rewrite.

## Priority: cheap now, expensive to change later

- **P1: Record passages, not finish times** (absorbs Feature 16). A passage holds a bib or "unknown", the checkpoint, the device, the timestamp, the approximate race time, a confidence, a review status and references to its evidence. A bib can have several passages, so laps and later checkpoints are never overwritten. A finish line is just one checkpoint.
- **P2: Take times from the frames' own timestamps** (absorbs the timestamp part of Feature 11). The pipeline computes times as frame number ÷ fps.
  - Phone videos often have a variable frame rate, especially in low light. If the real rate is 1% off the stated one, times drift by 36 s over an hour.
  - Use each frame's timestamp instead (`CAP_PROP_POS_MSEC` in OpenCV, `VideoFrame.timestamp` in a browser), plus the device clock when recording starts. That also makes times from different devices comparable.
  - This matters for the finish line today.
- **P3: Work offline first** (absorbs Feature 20). A mountain checkpoint has no network, so its video cannot be uploaded from there and processing stays on the device. Passages, and recordings if kept, sync later when a connection is available. Each passage gets a unique ID, so syncing again creates no duplicates.

## Recognition and tracking

- **Feature 25:** Start a track from a confident detection even when its number cannot be read yet; today only a readable detection starts a track. Draw the track red until a reading matches the bib pattern, then green.
- **Feature 8:** Keep detected bibs whose digits cannot be read, as "unknown" passages with timestamps, the best crops and a short clip for manual review. This does not recover runners whose bibs were never detected.
- **Feature 24:** Drop detections that never move: an easy way to remove false positives.
- **Feature 26:** Handle lost tracks and identity switches when runners overlap. To start, merge tracks with the same bib when the last detection is less than 10 s old and each has at least 3 votes.
- **Feature 28:** When a track loses optical flow, run the detector on the next frame instead of waiting for the scheduled one. The tracker already deactivates lost tracks so the caller can see them.
- **Feature 6:** Combine partial and conflicting readings within a track, and keep unknown digits rather than guessing a complete number. The reader often drops one of two repeated digits (11461 read as 1461 or 1146), so partial readings carry real evidence.
- **Feature 27:** Weight votes by the time between readings: readings far apart in time are more likely to be right than readings on consecutive frames.
- **Feature 7:** Lock a track's number after enough agreement, to stop reading it again. Reopen the decision when stronger contradicting evidence appears or the track identity becomes uncertain. Repeated similar frames are not independent proof. Reading is now the slowest stage, so this also saves time.
- **Feature 23:** Rename `bib_string` to `bib_number`. Readings are digits only now, but keep a string so leading zeros survive.

## Speed

- **Feature 1:** Skip near-duplicate frames with frame differencing or motion detection, before parallelizing anything. Essential for long recordings at quiet checkpoints.
- **Feature 2:** Batch model calls: the OCR crops of one frame in a single call (reading is now the slowest stage), and several frames through the detector when processing a file. Check backend support, and measure throughput, memory use and the wait to fill a batch: higher throughput alone does not guarantee low live latency.
- **Feature 3:** Parallelize the CPU-bound stages (merges Features 3 and 4). Processes (`multiprocessing.Pool`) sidestep the GIL but load one copy of each model per process. Threads (`ThreadPoolExecutor` with 4 workers) are a quicker experiment; make sure concurrent calls do not share a detector instance unsafely. OpenVINO and ONNX Runtime already use several cores, so measure the gain.
- **Feature 11:** Configurable frame sampling, evaluated against runner coverage before using it live.
- **Feature 10:** Repeatable benchmark on the same footage and recognition metrics. Report hardware, configuration, frames per second, video seconds processed per wall-clock second, and capture-to-result latency. Stage timings already exist in `summary.json`, but video decoding is not timed on its own yet.

## Race workflow

- **Feature 13:** Import a participant list linking bibs to names, race distances and start waves. Flag readings that are not on the list, but keep them as unknown rather than forcing the nearest valid number. Preserve leading zeros. A list only rejects numbers outside the registered ones: with more than 10,000 runners, 1001 and 10011 can both be valid.
- **Feature 15:** Timing setup that maps timestamps to race time and the start wave (uses P2). Record the timing method and the first and last observed times. A visibility window alone is not a guaranteed timing-error bound.
- **Feature 19:** Camera setup check from a short test recording: capture area, bib readability, lighting and framing. The reader starts failing below about 16 px of bib height, and at the default 832 px input the detector found no bib under 20 px tall in sample1's 1080p video.
- **Feature 9:** Organizer UI: import participants, configure timing, process a video or a live feed, follow progress and lag, resolve uncertain passages, and export results.
- **Feature 12:** Live input with a bounded processing queue. Define an overload policy so latency cannot grow without limit, report skipped frames and lag, and keep the footage to recover missed passages later.
- **Feature 14:** Review queue for unreadable bibs, conflicting readings and suspected duplicates (uses 8). The organizer confirms or corrects a bib, merges duplicate passages, or splits wrongly combined tracks. Keep the original evidence and the correction history.
- **Feature 17:** Export passages and corrected results to CSV (depends on P1). Distinguish provisional, reviewed and unresolved results, and keep enough to trace each result back to its evidence.
- **Feature 21:** Spectator page with each runner's last confirmed passage and approximate time (depends on P1). Show the checkpoint and how fresh the update is, rather than implying continuous tracking.
- **Feature 22:** Personal finish clip that a runner finds by bib number (uses 8).

## Evaluation

- **Feature 18:** Label more videos than sample1, including crowded scenes, unreadable bibs and difficult lighting. Measure correctly identified runners, missed runners, wrong identities, duplicates, timing error, and review minutes per 100 runners, next to the speed benchmark. Image examples alone do not measure race-level coverage.

## Investigations

- **X:** Compare the optical-flow tracker with ByteTrack again (from Feature 5 and the optical-flow experiment). A quick comparison on 2026-09-27 (`runs/bytetrack-*` against `runs/flow-*`) came before the optical-flow tracker was built. Redo it with the current detector, measuring identity switches, lost tracks, missed runners and processing time on the same footage.
- **X:** Lower the detection threshold on frames where a tracked bib was not detected (from Feature 5).
- **X:** Example-based bib detection: locate every bib of a race from one reference bib image, starting with ORB features and geometric verification. The goal is to find bibs with other numbers, not to track or read them. Test whether enough shared features remain when the digits dominate the design. Measure localization recall, false positives and time on numbers excluded from the reference, including frames with several runners. Synthetic training is out of scope.
- **X:** Strava integration. Do not assume the public API gives live athlete positions; check what it provides before designing location sharing.

# Done

- Detection, optical flow and reading are drawn in different colors, so each event is visible.
- Late detection fixed: downscaling frames to 416 px made bibs too small. 832 px is the right trade-off.
- Golden test on the full sample1 video against its ground truth, without writing video.
- Bib detection and reading are separate, picked by name from registries (`BIB_DETECTORS`, `BIB_READERS`).
- Every detected box is read on detection frames.
- The frame loop reads every frame, so frame IDs match the video (part of Feature 11).
- Temporary track IDs separate from bib numbers, votes per track, and detection every 20 frames with optical flow in between (from Feature 5).
- Conflicting readings are kept in the votes and flagged, and the bib pattern is configurable (part of Feature 6).
- Stage timings in `summary.json` (part of Feature 10).
- 2026-10-07: YOLO26n v025 detector through OpenVINO and PP-OCRv6 reader through ONNX Runtime. On sample1: 13/13 runners and none wrong, against 11/13 with one wrong, and about twice as fast.
- 2026-10-07: tracking speed-ups. One optical-flow call for all tracks, corner search near each box, and flow on half-size frames. Tracking on sample1 went from 7.1 s to 3.0 s.
- 2026-10-07: votes weighted ×5 per digit, the rule this list originally asked for. It had been built as strict length priority, which let a single invented 5-digit reading win. The reader drops a repeated digit 7 to 10 times more often than it invents one.
- Tried and dropped: showing the vote counts of the two leading readings next to each box. It was hard to read.
