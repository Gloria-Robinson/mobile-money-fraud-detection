"""Run the smallest useful local Kumo Tabular compatibility test.

This script intentionally uses scikit-learn's breast-cancer dataset instead of
PaySim. Its only job is to answer: can Kumo Tabular Small download, load, and
return classification probabilities on this Mac's CPU?

This is not a fraud experiment and its output is not a research result.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from sklearn.datasets import load_breast_cancer

import sdm


MODEL_REPOSITORY = "nvidia/Kumo-Tabular"
MODEL_REVISION = "v1.0.1"
MODEL_SIZE = "small"
MODEL_FILE = f"{MODEL_SIZE}/classifier.pt"


def peak_memory_mb() -> float:
    """Return this process's peak resident memory in MiB.

    macOS reports ru_maxrss in bytes; Linux reports it in KiB.
    """

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return peak / divisor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-rows", type=int, default=300)
    parser.add_argument("--query-rows", type=int, default=100)
    parser.add_argument("--num-estimators", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/feasibility/kumo_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    if args.context_rows < 2 or args.query_rows < 1:
        raise ValueError("Use at least 2 context rows and 1 query row.")
    if args.num_estimators < 1:
        raise ValueError("num-estimators must be at least 1.")

    # This phase includes a real download on the first run and a cache lookup on
    # later runs. Keeping it outside model construction separates network/cache
    # time from model deserialization time.
    download_started = time.perf_counter()
    config_path = hf_hub_download(
        repo_id=MODEL_REPOSITORY,
        filename="config.json",
        revision=MODEL_REVISION,
    )
    checkpoint_path = hf_hub_download(
        repo_id=MODEL_REPOSITORY,
        filename=MODEL_FILE,
        revision=MODEL_REVISION,
    )
    download_or_cache_seconds = time.perf_counter() - download_started

    frame = load_breast_cancer(as_frame=True).frame
    requested_rows = args.context_rows + args.query_rows
    if requested_rows > len(frame):
        raise ValueError(
            f"Requested {requested_rows} rows, but the example has {len(frame)}."
        )
    frame = frame.iloc[:requested_rows].reset_index(drop=True)

    tensor_started = time.perf_counter()
    table = sdm.TableTensor.from_pandas(
        df=frame,
        stypes=sdm.infer_stypes(frame, overrides={"target": "categorical"}),
        device=device,
    )
    tensor_seconds = time.perf_counter() - tensor_started

    model_started = time.perf_counter()
    model = sdm.models.KumoTabular(
        task="classification",
        size=MODEL_SIZE,
        device=device,
    )
    model_load_seconds = time.perf_counter() - model_started

    inference_started = time.perf_counter()
    with torch.inference_mode():
        probabilities = model(
            x_context=table[: args.context_rows].drop_columns("target"),
            y_context=table[: args.context_rows, "target"],
            x_query=table[
                args.context_rows : requested_rows
            ].drop_columns("target"),
            num_estimators=args.num_estimators,
        )
    inference_seconds = time.perf_counter() - inference_started
    probabilities_frame = probabilities.to_pandas()

    report = {
        "status": "passed",
        "purpose": "compatibility_smoke_test_not_a_research_result",
        "device": str(device),
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "model_size": MODEL_SIZE,
        "context_rows": args.context_rows,
        "query_rows": args.query_rows,
        "num_estimators": args.num_estimators,
        "timings_seconds": {
            "download_or_cache_resolution": download_or_cache_seconds,
            "tensor_preparation": tensor_seconds,
            "model_load": model_load_seconds,
            "inference": inference_seconds,
        },
        "peak_process_memory_mb": peak_memory_mb(),
        "probability_shape": list(probabilities_frame.shape),
        "probability_columns": [str(column) for column in probabilities_frame.columns],
        "probability_preview": probabilities_frame.head(5).to_dict(orient="records"),
        "cached_files": {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(json.dumps(report, indent=2, default=str))
    print(f"\nSaved report to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

