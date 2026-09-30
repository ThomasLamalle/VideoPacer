from collections import defaultdict

import numpy as np

from detector import Detection
from evaluate_models import annotate, iou, score


def test_scoring_matches_once_and_ranks_by_confidence():
    gt = defaultdict(list, {1: [(0, 0, 10, 10), (20, 20, 10, 10)]})
    predictions = [(1, (0, 0, 10, 10), 0.8), (1, (0, 0, 10, 10), 0.7), (1, (20, 20, 10, 10), 0.9)]
    result = score(predictions, gt, 0.5, 2)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 1, 0)
    assert result["precision"] == 2 / 3
    assert result["recall"] == 1
    assert result["ap"] == 1  # both matches precede the duplicate
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == 1 / 3
    assert score([], gt, 0.5, 2)["fn"] == 2


def test_annotate_draws_ground_truth_green_and_prediction_red():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    result = annotate(frame, [(5, 20, 10, 10)], [Detection("bib", (40, 20, 10, 10), 0.875)])
    assert tuple(result[20, 5]) == (0, 255, 0)
    assert tuple(result[20, 40]) == (0, 0, 255)
    assert result is not frame
