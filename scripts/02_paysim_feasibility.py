"""Run a small, chronological PaySim feasibility check with Kumo Tabular.

This is deliberately a diagnostic benchmark, not the final model comparison.
It uses earlier whole time steps as labelled context and later validation steps
as queries. Both context and query are fraud-enriched so the tiny run exercises
both classes; the report discloses that enrichment explicitly.

No balance fields, customer IDs, or isFlaggedFraud are loaded as model inputs.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

import sdm


MODEL_REPOSITORY = "nvidia/Kumo-Tabular"
MODEL_REVISION = "v1.0.1"
MODEL_SIZE = "small"
MODEL_FILE = f"{MODEL_SIZE}/classifier.pt"
FEATURE_COLUMNS = ["step", "type", "amount"]
TARGET_COLUMN = "isFraud"
LOADED_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]
FORBIDDEN_INPUTS = {
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "nameOrig",
    "nameDest",
}


def peak_memory_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=repository / "PS_20174392719_1491204439457_log.csv",
    )
    parser.add_argument("--context-rows", type=int, default=300)
    parser.add_argument("--context-fraud-rows", type=int, default=100)
    parser.add_argument("--query-rows", type=int, default=100)
    parser.add_argument("--query-fraud-rows", type=int, default=20)
    parser.add_argument("--num-estimators", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/feasibility/paysim_kumo_small.json"),
    )
    return parser.parse_args()


def sample_binary_rows(
    frame: pd.DataFrame,
    *,
    total_rows: int,
    fraud_rows: int,
    seed: int,
) -> pd.DataFrame:
    genuine_rows = total_rows - fraud_rows
    if total_rows < 2 or fraud_rows < 1 or genuine_rows < 1:
        raise ValueError("Each diagnostic sample must contain both classes.")

    fraud = frame.loc[frame[TARGET_COLUMN] == 1]
    genuine = frame.loc[frame[TARGET_COLUMN] == 0]
    if len(fraud) < fraud_rows or len(genuine) < genuine_rows:
        raise ValueError(
            "The selected chronological partition does not contain enough rows "
            "for the requested diagnostic sample."
        )

    return (
        pd.concat(
            [
                fraud.sample(n=fraud_rows, random_state=seed),
                genuine.sample(n=genuine_rows, random_state=seed),
            ],
            ignore_index=True,
        )
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    if not args.csv.is_file():
        raise FileNotFoundError(f"PaySim CSV not found: {args.csv}")
    if FORBIDDEN_INPUTS.intersection(FEATURE_COLUMNS):
        raise RuntimeError("A forbidden leakage or identifier column is configured.")

    data_started = time.perf_counter()
    data = pd.read_csv(
        args.csv,
        usecols=LOADED_COLUMNS,
        dtype={
            "step": "int16",
            "type": "category",
            "amount": "float32",
            TARGET_COLUMN: "int8",
        },
    )
    data_load_seconds = time.perf_counter() - data_started

    if data.empty:
        raise ValueError("PaySim CSV is empty.")
    if set(data[TARGET_COLUMN].unique()) != {0, 1}:
        raise ValueError("isFraud must contain exactly the binary values 0 and 1.")
    if data[LOADED_COLUMNS].isna().any().any():
        raise ValueError("Unexpected missing values in the selected columns.")

    # Split by complete time steps, not by individual shuffled transactions.
    unique_steps = sorted(int(step) for step in data["step"].unique())
    train_count = int(len(unique_steps) * 0.70)
    validation_count = int(len(unique_steps) * 0.15)
    train_steps = set(unique_steps[:train_count])
    validation_steps = set(
        unique_steps[train_count : train_count + validation_count]
    )
    test_steps = set(unique_steps[train_count + validation_count :])

    train = data.loc[data["step"].isin(train_steps)]
    validation = data.loc[data["step"].isin(validation_steps)]

    context = sample_binary_rows(
        train,
        total_rows=args.context_rows,
        fraud_rows=args.context_fraud_rows,
        seed=args.seed,
    )
    query = sample_binary_rows(
        validation,
        total_rows=args.query_rows,
        fraud_rows=args.query_fraud_rows,
        seed=args.seed + 1,
    )
    query_truth = query[TARGET_COLUMN].copy()

    # Only these four columns enter the TableTensor. Query labels exist solely
    # for the diagnostic report and are never passed as x_query.
    model_frame = pd.concat([context, query], ignore_index=True)
    model_frame["type"] = model_frame["type"].astype("string")

    tensor_started = time.perf_counter()
    table = sdm.TableTensor.from_pandas(
        df=model_frame,
        stypes=sdm.infer_stypes(
            model_frame,
            overrides={"type": "categorical", TARGET_COLUMN: "categorical"},
        ),
        device=device,
    )
    tensor_seconds = time.perf_counter() - tensor_started

    download_started = time.perf_counter()
    hf_hub_download(
        repo_id=MODEL_REPOSITORY,
        filename="config.json",
        revision=MODEL_REVISION,
    )
    hf_hub_download(
        repo_id=MODEL_REPOSITORY,
        filename=MODEL_FILE,
        revision=MODEL_REVISION,
    )
    download_or_cache_seconds = time.perf_counter() - download_started

    model_started = time.perf_counter()
    model = sdm.models.KumoTabular(
        task="classification",
        size=MODEL_SIZE,
        device=device,
    )
    model_load_seconds = time.perf_counter() - model_started

    query_start = args.context_rows
    query_stop = query_start + args.query_rows
    inference_started = time.perf_counter()
    with torch.inference_mode():
        probabilities = model(
            x_context=table[:query_start].drop_columns(TARGET_COLUMN),
            y_context=table[:query_start, TARGET_COLUMN],
            x_query=table[query_start:query_stop].drop_columns(TARGET_COLUMN),
            num_estimators=args.num_estimators,
        )
    inference_seconds = time.perf_counter() - inference_started
    probabilities_frame = probabilities.to_pandas().reset_index(drop=True)

    preview = pd.concat(
        [
            query[FEATURE_COLUMNS].head(10).reset_index(drop=True),
            query_truth.head(10).reset_index(drop=True).rename("actual_isFraud"),
            probabilities_frame.head(10),
        ],
        axis=1,
    )

    report = {
        "status": "passed",
        "purpose": "diagnostic_feasibility_not_final_evaluation",
        "warning": "A high probability is a prediction, not proof of fraud.",
        "deployment_warning": "This prototype does not block real payments.",
        "sampling_disclosure": {
            "context": "fraud_enriched_from_training_steps",
            "query": "fraud_enriched_from_validation_steps",
            "test_partition_used": False,
        },
        "dataset": {
            "path": str(args.csv.resolve()),
            "rows": int(len(data)),
            "fraud_rows": int(data[TARGET_COLUMN].sum()),
            "observed_step_min": int(data["step"].min()),
            "observed_step_max": int(data["step"].max()),
        },
        "whole_step_split": {
            "train": [min(train_steps), max(train_steps)],
            "validation": [min(validation_steps), max(validation_steps)],
            "test_reserved": [min(test_steps), max(test_steps)],
        },
        "model_inputs": FEATURE_COLUMNS,
        "excluded_inputs": sorted(FORBIDDEN_INPUTS),
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "size": MODEL_SIZE,
            "device": str(device),
            "num_estimators": args.num_estimators,
        },
        "diagnostic_sample": {
            "context_rows": args.context_rows,
            "context_fraud_rows": int(context[TARGET_COLUMN].sum()),
            "query_rows": args.query_rows,
            "query_fraud_rows": int(query_truth.sum()),
        },
        "timings_seconds": {
            "data_load": data_load_seconds,
            "tensor_preparation": tensor_seconds,
            "download_or_cache_resolution": download_or_cache_seconds,
            "model_load": model_load_seconds,
            "inference": inference_seconds,
        },
        "peak_process_memory_mb": peak_memory_mb(),
        "probability_shape": list(probabilities_frame.shape),
        "probability_columns": [str(column) for column in probabilities_frame.columns],
        "prediction_preview": preview.to_dict(orient="records"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved report to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

