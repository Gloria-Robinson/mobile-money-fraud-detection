"""Shared data and evaluation logic for the PaySim research experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "isFraud"
RULE_COLUMN = "isFlaggedFraud"
REQUIRED_COLUMNS = ("step", "type", "amount", TARGET_COLUMN, RULE_COLUMN)
FEATURE_COLUMNS = ("type", "amount", "log_amount", "hour_sin", "hour_cos", "elapsed_day")
NUMERIC_FEATURES = ("amount", "log_amount", "hour_sin", "hour_cos", "elapsed_day")
CATEGORICAL_FEATURES = ("type",)
EXCLUDED_MODEL_COLUMNS = (
    "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "nameOrig", "nameDest", RULE_COLUMN,
)


@dataclass(frozen=True)
class TimeSplit:
    """Complete, non-overlapping chronological partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_steps: tuple[int, ...]
    validation_steps: tuple[int, ...]
    test_steps: tuple[int, ...]


def load_paysim(csv_path: Path | str) -> pd.DataFrame:
    """Load only permitted model fields and separate evaluation fields."""

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"PaySim CSV not found: {path}")
    frame = pd.read_csv(
        path,
        usecols=list(REQUIRED_COLUMNS),
        dtype={"step": "int16", "type": "category", "amount": "float32", TARGET_COLUMN: "int8", RULE_COLUMN: "int8"},
    )
    if frame.empty:
        raise ValueError("PaySim CSV is empty.")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("PaySim contains missing values in required columns.")
    if set(frame[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("isFraud must contain exactly the binary values 0 and 1.")
    if not set(frame[RULE_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("isFlaggedFraud must be binary.")
    if (frame["amount"] < 0).any():
        raise ValueError("Transaction amount cannot be negative.")
    return frame


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TimeSplit:
    """Split by ordered whole time steps, reserving the latest steps for test."""

    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must be positive and leave a test partition.")
    steps = tuple(sorted(int(value) for value in frame["step"].unique()))
    if len(steps) < 3:
        raise ValueError("At least three distinct time steps are required.")
    train_end = max(1, int(len(steps) * train_fraction))
    validation_end = max(train_end + 1, train_end + int(len(steps) * validation_fraction))
    validation_end = min(validation_end, len(steps) - 1)
    train_steps = steps[:train_end]
    validation_steps = steps[train_end:validation_end]
    test_steps = steps[validation_end:]
    if not validation_steps or not test_steps:
        raise ValueError("Chronological split produced an empty partition.")
    if max(train_steps) >= min(validation_steps) or max(validation_steps) >= min(test_steps):
        raise RuntimeError("Chronological partitions overlap or are out of order.")
    return TimeSplit(
        train=frame.loc[frame["step"].isin(train_steps)].copy(),
        validation=frame.loc[frame["step"].isin(validation_steps)].copy(),
        test=frame.loc[frame["step"].isin(test_steps)].copy(),
        train_steps=train_steps,
        validation_steps=validation_steps,
        test_steps=test_steps,
    )


def add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create prediction-time features without labels or forbidden fields."""

    missing = {"step", "type", "amount"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature source is missing columns: {sorted(missing)}")
    step = frame["step"].astype("int32")
    hour = ((step - 1) % 24).astype("float32")
    radians = 2 * np.pi * hour / 24
    features = pd.DataFrame(index=frame.index)
    features["type"] = frame["type"].astype("string")
    features["amount"] = frame["amount"].astype("float32")
    features["log_amount"] = np.log1p(features["amount"]).astype("float32")
    features["hour_sin"] = np.sin(radians).astype("float32")
    features["hour_cos"] = np.cos(radians).astype("float32")
    features["elapsed_day"] = ((step - 1) // 24).astype("float32")
    unexpected = set(features.columns).intersection(EXCLUDED_MODEL_COLUMNS)
    if unexpected:
        raise RuntimeError(f"Forbidden model inputs detected: {sorted(unexpected)}")
    return features.loc[:, list(FEATURE_COLUMNS)]


def select_training_sample(frame: pd.DataFrame, *, max_rows: int | None, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Take a seeded, non-enriched random sample when a cap is used."""

    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        sample = frame.copy()
        method = "all_training_rows"
    else:
        sample = frame.sample(n=max_rows, random_state=seed).sort_index().copy()
        method = "seeded_random_sample_without_fraud_enrichment"
    return sample, {
        "method": method,
        "seed": seed,
        "source_rows": int(len(frame)),
        "sample_rows": int(len(sample)),
        "source_fraud_rows": int(frame[TARGET_COLUMN].sum()),
        "sample_fraud_rows": int(sample[TARGET_COLUMN].sum()),
        "source_fraud_rate": float(frame[TARGET_COLUMN].mean()),
        "sample_fraud_rate": float(sample[TARGET_COLUMN].mean()),
    }


def build_preprocessor() -> ColumnTransformer:
    """Return preprocessing that must be fitted on training data only."""

    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("one_hot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [("numeric", numeric, list(NUMERIC_FEATURES)), ("categorical", categorical, list(CATEGORICAL_FEATURES))],
        sparse_threshold=1.0,
    )


def threshold_for_review_capacity(scores: np.ndarray | pd.Series, *, alerts_per_10000: int) -> float:
    """Choose a score threshold from validation scores for a review budget."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be a finite, non-empty one-dimensional array.")
    if alerts_per_10000 < 1 or alerts_per_10000 > 10000:
        raise ValueError("alerts_per_10000 must be between 1 and 10,000.")
    count = max(1, int(np.ceil(len(values) * alerts_per_10000 / 10000)))
    count = min(count, len(values))
    selected = np.argpartition(values, -count)[-count:]
    return float(values[selected].min())


def binary_metrics(y_true: np.ndarray | pd.Series, scores: np.ndarray | pd.Series, *, threshold: float) -> dict[str, Any]:
    """Calculate review metrics at one fixed threshold."""

    truth = np.asarray(y_true, dtype=np.int8)
    probabilities = np.asarray(scores, dtype=float)
    if truth.shape != probabilities.shape:
        raise ValueError("Labels and scores must have the same shape.")
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, predictions, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(truth, probabilities)),
        "precision": float(precision),
        "recall": float(recall),
        "alerts": int(tp + fp),
        "alerts_per_10000": float((tp + fp) * 10000 / len(truth)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "missed_fraud": int(fn),
        "true_negatives": int(tn),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

