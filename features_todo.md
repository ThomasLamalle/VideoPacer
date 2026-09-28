# Features to implement

The first target is a small race that accepts approximate finish results. Processing speed is a core priority: the current pipeline is too slow for live detection, which remains a goal. Measure improvements in throughput and result latency alongside missed runners and incorrect readings.

List of features to implement. Increasing numbers does not mean features depends on each other, assume independance first.
X features needs brainstorming or analyze before consideration

- Make detection, optical flow or reading in separate colors so I know when a detection occured, when a read occured or when both occured

- Separate bib detection and bib reading clearly and use a registry pattern with a dict to easily change processing (we'll switch from yolo to RF-DETR and possibly change the digit reading too)


- Restore track identity or merge track with same bib string if 1) last detection is less than X=10s ago 2) atleast 3 votes on each

- Feature 1: Reduce actual work before parallelizing: frame differencing/motion detection to skip near-duplicate frames

- Feature 2: Experiment with batching frames through the detector. Verify model/backend support and measure throughput, memory use, and waiting time to fill a batch. Higher throughput alone does not guarantee low live latency.
- Feature 3: multiprocessing.Pool instead of threading for the CPU-bound parts (this sidesteps the GIL question entirely, and gives real multi-core parallelism). Cost: each process needs its own model loaded into memory (or GPU memory), so 4 processes = 4x model memory. Fine for a small YOLO/Detectron2 model, less fine if it's huge.
- Feature 4 :ThreadPoolExecutor with 4 workers as a quick experiment — measure the benefit on this pipeline and ensure concurrent calls do not mutate the same detector instance unsafely.


- Feature 5: Keep track of bibs using temporary track IDs separate from recognized bib numbers. Use ByteTrack as the initial tracking baseline. Account for lost tracks and identity switches when runners overlap. This lets us:
    1) count votes on bibs
    2) improve bib detection ? like if we lower threshold on frame where we did not detect a bib ?
    3) Do less bib detection ? like one every X frames and between the frames we only do tracking ?

- Feature 6 (depends on 5): Preserve partial and conflicting readings and combine evidence within a track. Define supported bib string patterns and retain unknown digits rather than guessing a complete number.

- take into account reading score when counting vote
- Use number of frame between readings when counting vote : time spaced readings have more chance to be right that consecutives frames reading

- Feature 7 (depends on 5): Add reversible locking to reduce repeated digit reading after sufficient agreement. Reopen a decision when stronger contradictory evidence appears or track identity becomes uncertain. Repeated similar frames must not be treated as independent proof of correctness.

- Feature 8: Preserve detected bibs even when their digits cannot be read. Store an unknown passage with timestamps, best bib crops, and a short surrounding clip for manual review. This does not recover runners whose bibs were never detected.

- Feature X (example-based bib detection): Investigate detecting all bibs in a race from one reference bib image, starting with OpenCV ORB feature extraction and matching plus geometric verification. The goal is to locate bibs with different numbers, not to track the same bib across frames or read its digits. Test whether enough shared visual features remain when digits dominate the bib design. Evaluate localization recall, false positives, and processing time on different bib numbers excluded from the reference, including frames containing multiple runners. Treat ORB as an experimental candidate whose ability to generalize must be demonstrated. Synthetic training is outside the scope of this experiment.

- Feature X (conditional optical-flow experiment; depends on evaluating 5): Investigate sparse optical flow only if benchmarks show that ByteTrack's processing time or tracking quality is insufficient. Compare an optical-flow-assisted pipeline against the ByteTrack baseline on the same footage, measuring total processing time, identity switches, lost tracks, and missed runners. Define periodic detection and recovery when tracking fails. Do not assume optical flow is a drop-in replacement for ByteTrack or that it will improve performance.

- Feature 9: Create a focused organizer UI: import participants, configure timing, process an uploaded video or live feed, inspect processing progress and lag, resolve uncertain passages, and export results.

- Feature X: Investigate Strava integration. Do not assume its public API provides live athlete positions; verify available capabilities before defining a location-sharing feature.

## Additional features

- Feature 10: Add repeatable performance benchmarks and stage timings for video decoding, bib detection, digit reading, and output. Report hardware, configuration, processed frames per second, video seconds processed per wall-clock second, and capture-to-result latency. Compare optimizations against the same footage and recognition metrics.

- Feature 11: Implement correct, configurable frame sampling while preserving actual source timestamps. Fix the current loop, which increments frame IDs without advancing the video on skipped iterations. Evaluate sampling rates against runner coverage before using them for live processing.

- Feature 12: Add live video input with a bounded processing queue. Define an overload policy so latency cannot grow indefinitely; report skipped frames and processing lag. Preserve footage for later recovery of missed passages where recording is available.

- Feature 13: Import a participant list linking bib strings to names, race distances, and start waves. Flag readings that do not match the list, but retain an unknown outcome rather than forcing the nearest valid number. Preserve leading zeros.

- Feature 14 (uses evidence from 8): Create an evidence-backed review queue for unreadable bibs, conflicting readings, and suspected duplicates. Allow an organizer to confirm or correct a bib, merge duplicate passages, or split incorrectly combined tracks. Retain original evidence and correction history.

- Feature 15: Add explicit timing setup that maps source video timestamps to race time and the relevant start wave. Record the chosen approximate timing method and first/last observed times. A visibility window alone must not be presented as a guaranteed timing-error bound.

- Feature 16: Store individual passage records with bib or unknown identity, camera/checkpoint, source timestamps, approximate race time, review status, and evidence references. Support multiple passages for a bib so laps or later checkpoints are not silently overwritten.

- Feature 17 (depends on 16): Export passage records and corrected finish results to CSV. Distinguish provisional, reviewed, and unresolved results, and preserve enough information to trace each result back to its evidence.

- Feature 18: Evaluate representative labeled video, including crowded scenes, unreadable bibs, and difficult lighting. Measure correctly identified runners, missed runners, wrong identities, duplicates, timing error, and review minutes per 100 runners alongside the speed benchmark. Previously successful image examples alone are insufficient to measure race-level coverage.

- Feature 19: Add a camera setup check using a short test recording. Help the organizer assess capture area, bib readability, lighting, and framing before the race.

- Feature 20: Support offline recording and later upload for locations with unreliable connectivity. Preserve capture timestamps and avoid duplicate passages when recordings are uploaded again.

- Feature 21 (depends on 16): Add a spectator page showing each runner's last confirmed passage and approximate time. Show checkpoint location and update freshness rather than implying continuous location tracking.

- Feature 22 (uses evidence from 8): Generate a personal finish clip that a runner can find by bib number, using saved passage evidence.

## Initial delivery focus

Keep performance measurement and improvements active alongside recognition and result review. The first useful workflow is participant import plus finish video processing, approximate results, review of uncertain passages, and corrected export. Extend that workflow to live input once measured throughput and latency are sufficient on the intended hardware. Offline checkpoint support and spectator features can follow.


# Features DONE


- Investigate why bibs are detected (or read ?) so late in the video : the bibs are detected or read at least 5s after the runner is in the frame => It was the downscaling of the image. 832 pixels seems to right tradeoff
