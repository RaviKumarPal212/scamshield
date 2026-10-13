import numpy as np
import pytest

from scripts.analyze_sms_thresholds import choose_candidates, score_threshold


def test_threshold_metrics_show_false_positive_and_false_negative_tradeoff() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.9, 0.2])

    default = score_threshold(labels, probabilities, 0.5)
    candidates = choose_candidates(labels, probabilities)

    assert default["false_negative_spam_missed"] == 1
    assert default["false_positive_ham_flagged"] == 0
    assert candidates["best_f1_on_validation"]["threshold"] == pytest.approx(0.2)
    assert candidates["best_f1_on_validation"]["recall_spam"] == pytest.approx(1.0)
    assert candidates["highest_recall_under_precision_floors"]["precision_at_least_0.80"]["threshold"] == pytest.approx(0.9)
