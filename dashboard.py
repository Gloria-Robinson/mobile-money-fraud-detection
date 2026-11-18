"""Interactive dashboard for the PaySim review-capacity experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPOSITORY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY / "src"))
from mobile_money_fraud.core import binary_metrics, threshold_for_review_capacity  # noqa: E402

RESULTS = REPOSITORY / "results/latest"
METRICS_PATH = RESULTS / "metrics.json"
PREDICTIONS_PATH = RESULTS / "test_predictions.csv.gz"
CURVE_PATH = RESULTS / "validation_capacity_curve.csv"

st.set_page_config(page_title="Mobile-Money Fraud Review Lab", page_icon="🔎", layout="wide")


@st.cache_data
def load_results() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    return (
        json.loads(METRICS_PATH.read_text()),
        pd.read_csv(PREDICTIONS_PATH),
        pd.read_csv(CURVE_PATH),
    )


def metric_row(label: str, values: dict) -> dict[str, object]:
    return {
        "Model": label,
        "Average precision": values.get("average_precision"),
        "Precision": values.get("precision"),
        "Recall": values.get("recall"),
        "Alerts": values.get("alerts"),
        "False positives": values.get("false_positives"),
        "Missed fraud": values.get("missed_fraud"),
    }


st.title("Mobile-Money Fraud Review Lab")
st.caption("Review-capacity-aware research using synthetic PaySim transactions")
st.warning("A high score is a prediction, not proof of fraud. This prototype prioritises synthetic transactions for human review and does not block real payments.")

if not (METRICS_PATH.exists() and PREDICTIONS_PATH.exists() and CURVE_PATH.exists()):
    st.info("No measured experiment is available yet. Run `python train.py`, then refresh. The dashboard deliberately shows no fabricated results.")
    st.code("python train.py --max-train-rows 100000 --xgb-estimators 100 --review-budget-per-10000 100", language="bash")
    st.stop()

metrics, predictions, curve = load_results()
model_names = [name for name in ("logistic_regression", "xgboost") if f"{name}_score" in predictions.columns]

with st.sidebar:
    st.header("Research controls")
    model_name = st.selectbox("Model", model_names, format_func=lambda value: value.replace("_", " ").title())
    capacity = st.slider(
        "Exploratory reviews per 10,000 transactions",
        1,
        500,
        int(metrics["review_policy"]["final_budget_per_10000"]),
    )
    st.caption("This explores the held-out test results. It does not replace the final threshold selected on validation data.")

scores = predictions[f"{model_name}_score"].to_numpy()
exploratory_threshold = threshold_for_review_capacity(scores, alerts_per_10000=capacity)
exploratory_alert = scores >= exploratory_threshold
exploratory = binary_metrics(predictions["isFraud"].to_numpy(), scores, threshold=exploratory_threshold)

st.header("1. Transaction review queue")
overview = st.columns(5)
overview[0].metric("Review capacity", f"{capacity:,} / 10k")
overview[1].metric("Alerts generated", f"{exploratory['alerts']:,}")
overview[2].metric("Fraud caught", f"{exploratory['true_positives']:,}")
overview[3].metric("Genuine alerts", f"{exploratory['false_positives']:,}")
overview[4].metric("Fraud missed", f"{exploratory['missed_fraud']:,}")

queue = predictions.assign(exploratory_alert=exploratory_alert, selected_model_score=scores)
controls = st.columns([1, 2, 1])
alerts_only = controls[0].checkbox("Show alerts only", value=True)
available_types = sorted(queue["type"].unique().tolist())
selected_types = controls[1].multiselect("Transaction types", available_types, default=available_types)
show_answers = controls[2].checkbox("Reveal synthetic labels", value=False)
if alerts_only:
    queue = queue.loc[queue["exploratory_alert"]]
queue = queue.loc[queue["type"].isin(selected_types)]
display_columns = ["transaction_id", "step", "type", "amount", "selected_model_score", "exploratory_alert"]
if show_answers:
    display_columns.extend(["isFraud", "isFlaggedFraud"])
st.dataframe(
    queue.loc[:, display_columns].sort_values("selected_model_score", ascending=False).head(500),
    width="stretch",
    hide_index=True,
    column_config={
        "selected_model_score": st.column_config.ProgressColumn("Risk score", min_value=0.0, max_value=1.0, format="%.4f"),
        "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
    },
)

st.header("2. Individual transaction prediction")
default_id = int(queue.iloc[0]["transaction_id"]) if not queue.empty else int(predictions.iloc[0]["transaction_id"])
transaction_id = st.number_input(
    "Transaction ID",
    min_value=int(predictions["transaction_id"].min()),
    max_value=int(predictions["transaction_id"].max()),
    value=default_id,
    step=1,
)
matching = predictions.loc[predictions["transaction_id"] == transaction_id]
if matching.empty:
    st.error("That ID is not in the held-out test transactions.")
else:
    row = matching.iloc[0]
    score = float(row[f"{model_name}_score"])
    final_threshold = float(metrics["models"][model_name]["threshold_selection"]["selected_threshold"])
    status = "Send for human review" if score >= final_threshold else "Not prioritised at the final threshold"
    detail = st.columns(5)
    detail[0].metric("Type", str(row["type"]))
    detail[1].metric("Amount", f"{row['amount']:,.2f}")
    detail[2].metric("Time step", int(row["step"]))
    detail[3].metric("Risk score", f"{score:.4f}")
    detail[4].metric("Final decision", status)
    st.caption("This uses the validation-selected threshold. It is a review recommendation, not proof that fraud occurred.")
    if show_answers:
        st.write({"synthetic_isFraud_answer": int(row["isFraud"]), "existing_isFlaggedFraud_rule": int(row["isFlaggedFraud"])})

st.header("3. Final model comparison")
st.success("These results use thresholds chosen on validation data and one untouched test period. The sidebar does not change them.")
comparison = [metric_row(name.replace("_", " ").title(), metrics["models"][name]["test"]) for name in model_names]
comparison.append(metric_row("Existing isFlaggedFraud rule", metrics["isFlaggedFraud_rule_baseline"]["test"]))
st.dataframe(
    pd.DataFrame(comparison), width="stretch", hide_index=True,
    column_config={
        "Average precision": st.column_config.NumberColumn(format="%.4f"),
        "Precision": st.column_config.NumberColumn(format="%.4f"),
        "Recall": st.column_config.NumberColumn(format="%.4f"),
    },
)

chart_data = curve.loc[curve["model"].isin(model_names)].copy()
chart_data["Model"] = chart_data["model"].str.replace("_", " ").str.title()
figure = px.line(
    chart_data,
    x="review_budget_per_10000",
    y=["precision", "recall"],
    color="Model",
    markers=True,
    labels={"review_budget_per_10000": "Validation reviews per 10,000", "value": "Metric value", "variable": "Metric"},
    title="Validation trade-off as review capacity changes",
)
st.plotly_chart(figure, width="stretch")

matrix = np.asarray(metrics["models"][model_name]["test"]["confusion_matrix"])
heatmap = go.Figure(go.Heatmap(
    z=matrix,
    x=["Predicted genuine", "Sent for review"],
    y=["Actually genuine", "Synthetic fraud"],
    text=matrix,
    texttemplate="%{text:,}",
    colorscale="Greens",
))
heatmap.update_layout(title=f"Final confusion matrix — {model_name.replace('_', ' ').title()}")
st.plotly_chart(heatmap, width="stretch")

runtime_rows = []
for name in model_names:
    runtime_rows.append({"Model": name.replace("_", " ").title(), **metrics["models"][name]["runtime_seconds"]})
st.subheader("Runtime and resource disclosure")
st.dataframe(pd.DataFrame(runtime_rows), width="stretch", hide_index=True)
st.write(f"Peak experiment process memory: **{metrics['peak_process_memory_mb']:.1f} MiB**")
sampling = metrics["training_sampling"]
st.write(f"Training rows used: **{sampling['sample_rows']:,} of {sampling['source_rows']:,}**. Sampling: `{sampling['method']}`. Validation and test were not fraud-enriched.")

