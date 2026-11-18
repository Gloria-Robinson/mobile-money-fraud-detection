"""Train and evaluate review-capacity-aware PaySim fraud baselines."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY / "src"))

from mobile_money_fraud.core import (  # noqa: E402
    EXCLUDED_MODEL_COLUMNS,
    FEATURE_COLUMNS,
    RULE_COLUMN,
    TARGET_COLUMN,
    add_model_features,
    binary_metrics,
    build_preprocessor,
    chronological_split,
    load_paysim,
    select_training_sample,
    threshold_for_review_capacity,
)


def peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / (1024 * 1024 if sys.platform == "darwin" else 1024))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=REPOSITORY / "PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--max-train-rows", type=int, default=500_000)
    parser.add_argument("--review-budget-per-10000", type=int, default=100)
    parser.add_argument("--xgb-estimators", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--results-dir", type=Path, default=REPOSITORY / "results/latest")
    parser.add_argument("--artifacts-dir", type=Path, default=REPOSITORY / "artifacts/latest")
    return parser.parse_args()


def distribution(frame: pd.DataFrame) -> dict[str, float | int]:
    fraud = int(frame[TARGET_COLUMN].sum())
    return {
        "rows": int(len(frame)),
        "fraud_rows": fraud,
        "genuine_rows": int(len(frame) - fraud),
        "fraud_rate": float(fraud / len(frame)),
        "step_min": int(frame["step"].min()),
        "step_max": int(frame["step"].max()),
    }


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()

    data_started = time.perf_counter()
    data = load_paysim(args.csv)
    data_load_seconds = time.perf_counter() - data_started
    split = chronological_split(data)
    train_frame, sampling = select_training_sample(split.train, max_rows=args.max_train_rows, seed=args.seed)

    print("PaySim chronological partitions")
    for name, frame in (("training source", split.train), ("training used", train_frame), ("validation", split.validation), ("test (still hidden)", split.test)):
        print(f"  {name}: {distribution(frame)}")

    features_started = time.perf_counter()
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(add_model_features(train_frame))
    x_validation = preprocessor.transform(add_model_features(split.validation))
    x_test = preprocessor.transform(add_model_features(split.test))
    y_train = train_frame[TARGET_COLUMN].to_numpy()
    y_validation = split.validation[TARGET_COLUMN].to_numpy()
    y_test = split.test[TARGET_COLUMN].to_numpy()
    preprocessing_seconds = time.perf_counter() - features_started
    joblib.dump(preprocessor, args.artifacts_dir / "preprocessor.joblib")

    positive_rows = int(y_train.sum())
    if positive_rows == 0:
        raise ValueError("Training sample contains no fraud rows; increase the sample size.")
    scale_pos_weight = float((len(y_train) - positive_rows) / positive_rows)
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=500, random_state=args.seed, solver="lbfgs"),
        "xgboost": XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            n_estimators=args.xgb_estimators, max_depth=6, learning_rate=0.08,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight, random_state=args.seed, n_jobs=args.threads,
        ),
    }

    model_reports: dict[str, dict[str, object]] = {}
    test_output = pd.DataFrame({
        "transaction_id": split.test.index.astype("int64"),
        "step": split.test["step"].to_numpy(),
        "type": split.test["type"].astype("string").to_numpy(),
        "amount": split.test["amount"].to_numpy(),
        TARGET_COLUMN: y_test,
        RULE_COLUMN: split.test[RULE_COLUMN].to_numpy(),
    })
    curve_rows: list[dict[str, object]] = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        validation_scores = model.predict_proba(x_validation)[:, 1]
        validation_seconds = time.perf_counter() - started
        threshold = threshold_for_review_capacity(validation_scores, alerts_per_10000=args.review_budget_per_10000)
        validation_metrics = binary_metrics(y_validation, validation_scores, threshold=threshold)

        started = time.perf_counter()
        test_scores = model.predict_proba(x_test)[:, 1]
        test_seconds = time.perf_counter() - started
        test_metrics = binary_metrics(y_test, test_scores, threshold=threshold)

        for capacity in (10, 25, 50, 100, 250, 500):
            candidate = threshold_for_review_capacity(validation_scores, alerts_per_10000=capacity)
            curve_rows.append({
                "model": model_name,
                "review_budget_per_10000": capacity,
                **binary_metrics(y_validation, validation_scores, threshold=candidate),
            })

        model_reports[model_name] = {
            "threshold_selection": {
                "partition": "validation",
                "requested_review_budget_per_10000": args.review_budget_per_10000,
                "selected_threshold": threshold,
            },
            "validation": validation_metrics,
            "test": test_metrics,
            "runtime_seconds": {"fit": fit_seconds, "validation_prediction": validation_seconds, "test_prediction": test_seconds},
        }
        test_output[f"{model_name}_score"] = test_scores.astype("float32")
        test_output[f"{model_name}_final_alert"] = test_scores >= threshold
        joblib.dump(model, args.artifacts_dir / f"{model_name}.joblib")
        print(f"  validation-selected threshold: {threshold:.6f}")
        print(f"  untouched-test metrics: {test_metrics}")

    rule_metrics = binary_metrics(y_test, split.test[RULE_COLUMN].to_numpy(dtype=float), threshold=0.5)
    predictions_path = args.results_dir / "test_predictions.csv.gz"
    test_output.to_csv(predictions_path, index=False, compression="gzip")
    curve_path = args.results_dir / "validation_capacity_curve.csv"
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False)

    report = {
        "status": "completed",
        "research_question": "Under limited review capacity, which baseline best balances fraud recall, precision, and false alerts on future PaySim steps?",
        "warnings": [
            "A high score is a prediction, not proof of fraud.",
            "This research prototype does not block real payments.",
            "PaySim is synthetic; results do not establish performance on Tanzanian data.",
        ],
        "dataset": {"source": "https://www.kaggle.com/datasets/ealaxi/paysim1", "path": str(args.csv.resolve()), "file_size_bytes": int(args.csv.stat().st_size), "full": distribution(data)},
        "data_contract": {
            "target": TARGET_COLUMN, "model_features": list(FEATURE_COLUMNS),
            "excluded_model_columns": list(EXCLUDED_MODEL_COLUMNS),
            "existing_rule_evaluated_separately": RULE_COLUMN,
            "split": "chronological_by_complete_time_step_70_15_15",
            "preprocessing_fit_partition": "training_sample_only",
            "validation_and_test_fraud_enrichment": False,
        },
        "partitions": {"train": distribution(split.train), "validation": distribution(split.validation), "test": distribution(split.test)},
        "training_sampling": sampling,
        "review_policy": {"final_budget_per_10000": args.review_budget_per_10000, "threshold_selected_on": "validation", "dashboard_changes": "exploratory_not_final_reported_evaluation"},
        "models": model_reports,
        "isFlaggedFraud_rule_baseline": {"test": rule_metrics, "note": "Evaluated separately and never used as a model input."},
        "runtime_seconds": {"data_load": data_load_seconds, "feature_and_preprocessing": preprocessing_seconds, "total": time.perf_counter() - total_started},
        "peak_process_memory_mb": peak_memory_mb(),
        "artifacts": {"test_predictions": str(predictions_path.resolve()), "validation_capacity_curve": str(curve_path.resolve()), "model_directory": str(args.artifacts_dir.resolve())},
    }
    metrics_path = args.results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print(f"Peak process memory: {report['peak_process_memory_mb']:.1f} MiB")


if __name__ == "__main__":
    main()

