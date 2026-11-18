import numpy as np
import pandas as pd

from mobile_money_fraud.core import (
    EXCLUDED_MODEL_COLUMNS,
    add_model_features,
    binary_metrics,
    chronological_split,
    select_training_sample,
    threshold_for_review_capacity,
)


def example_frame(rows_per_step: int = 4) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "step": step,
            "type": "TRANSFER" if index % 2 else "PAYMENT",
            "amount": float(step * 10 + index),
            "isFraud": int(index == 0),
            "isFlaggedFraud": int(step == 10 and index == 0),
        }
        for step in range(1, 11)
        for index in range(rows_per_step)
    ])


def test_chronological_split_uses_complete_ordered_steps():
    split = chronological_split(example_frame())
    assert set(split.train_steps).isdisjoint(split.validation_steps)
    assert set(split.train_steps).isdisjoint(split.test_steps)
    assert set(split.validation_steps).isdisjoint(split.test_steps)
    assert max(split.train_steps) < min(split.validation_steps)
    assert max(split.validation_steps) < min(split.test_steps)
    assert set(split.train["step"]) == set(split.train_steps)
    assert set(split.validation["step"]) == set(split.validation_steps)
    assert set(split.test["step"]) == set(split.test_steps)


def test_features_do_not_contain_forbidden_columns():
    features = add_model_features(example_frame())
    assert not set(features.columns).intersection(EXCLUDED_MODEL_COLUMNS)
    assert "isFraud" not in features.columns
    assert features.isna().sum().sum() == 0


def test_training_sample_is_seeded_and_not_enriched():
    frame = example_frame(rows_per_step=20)
    first, metadata = select_training_sample(frame, max_rows=100, seed=7)
    second, _ = select_training_sample(frame, max_rows=100, seed=7)
    assert first.index.tolist() == second.index.tolist()
    assert metadata["method"] == "seeded_random_sample_without_fraud_enrichment"
    assert metadata["sample_rows"] == 100


def test_review_threshold_respects_capacity_without_ties():
    scores = np.linspace(0.0, 1.0, 1000, endpoint=False)
    threshold = threshold_for_review_capacity(scores, alerts_per_10000=100)
    assert int((scores >= threshold).sum()) == 10


def test_binary_metrics_counts_false_alerts_and_missed_fraud():
    metrics = binary_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.9, 0.8, 0.2]),
        threshold=0.5,
    )
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5

