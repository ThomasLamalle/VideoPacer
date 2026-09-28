# Simple bib tracking

## Goal

Detect bibs periodically, follow their boxes between detections with optical
flow, and report the best bib reading with the first and last visible time.
The code should be short enough to understand by reading three files.

## Files

- `detector.py` loads the OpenCV models, detects bibs and reads digits.
- `tracker.py` owns one `Track` data class, optical flow, detection matching and
  bib votes.
- `bib_detection.py` owns the command line, video loop, drawing and outputs.

## Frame loop

For every decoded frame:

1. Move active tracks with optical flow.
2. Run detection on scheduled frames or immediately when a track is lost.
3. Match detections to active tracks and create tracks for unmatched detections.
4. Read digits from tracked crops on the configured interval.
5. Draw and write the frame.

Detection runs every 100 frames by default. If recovery detection finds no bib,
the pipeline waits for the next scheduled detection. There is no retry state or
scan scheduler.

## Track data

Each track keeps only:

- temporary ID;
- current box and optical-flow points;
- first and last visible frame;
- latest detector frame;
- bib-reading vote counts.

The best bib is the unique reading with the most votes. A tie has no best bib.
More than one distinct reading sets the conflict flag. Lost tracks remain in the
final results but are no longer updated.

## Outputs

- `annotated.mp4` contains every processed frame at the source FPS.
- `summary.json` contains track results, configuration and stage timings.

There is no per-frame JSON, environment report, reading-attempt state, separate
history class, retry interval, or generic tracking framework.

## Validation

Tests cover scheduled detection, recovery detection, optical-flow movement,
track loss, votes and ties, first/last times, output frame count and timings. A
short sample-video run verifies the complete pipeline.
