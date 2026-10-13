import numpy as np
import pytest

from scripts.train_sms_baseline import validation_metrics


def test_validation_metrics_use_spam_as_positive_and_count_false_negatives() -> None:
    metrics = validation_metrics([0, 0, 1, 1], np.array([0.1, 0.7, 0.8, 0.2]))

    assert metrics["confusion_matrix_labels_0_ham_1_spam"] == {
        "true_negative_ham": 1,
        "false_positive_ham_flagged_as_spam": 1,
        "false_negative_spam_missed": 1,
        "true_positive_spam_detected": 1,
    }
    assert metrics["precision_positive_spam"] == pytest.approx(0.5)
    assert metrics["recall_positive_spam"] == pytest.approx(0.5)
    assert metrics["false_negative_rate_positive_spam"] == pytest.approx(0.5)
