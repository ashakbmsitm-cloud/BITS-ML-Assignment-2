"""BITS Pilani M.Tech AIML — Assignment 2: pretrained classifier evaluation.

Run with: streamlit run app.py
Requires model_config.json and the five trained .joblib files in model/.
The test CSV is evaluated only; no training happens inside this application.
"""

from io import BytesIO
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="ML Classifier Evaluation", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.25rem; padding-bottom: 1rem; max-width: 1150px;}
      h1 {font-size: 1.9rem !important;}
      .small-note {font-size: .85rem; color: #64748b;}
      @media print {
        header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] { display: none !important; }
        .block-container { max-width: none; padding: 0.2rem !important; }
        @page { size: A4 landscape; margin: 9mm; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Classification Model Evaluation")
st.caption("BITS Pilani · M.Tech AIML · Assignment 2 · Five pretrained classifiers")

config_file = BASE / "model_config.json"
if not config_file.is_file():
    st.error("Missing model_config.json in the GitHub repository root.")
    st.stop()
try:
    config = json.loads(config_file.read_text(encoding="utf-8"))
    feature_names = list(config["feature_names"])
    model_files = dict(config["model_files"])
except (OSError, ValueError, KeyError, TypeError) as exc:
    st.error(f"Unable to read model_config.json: {exc}")
    st.stop()

choice = st.selectbox("Choose a trained classification model", options=list(model_files.keys()))
upload = st.file_uploader("Upload test_data.csv (30 input features + target)", type="csv")

if upload is None:
    st.info("Upload the labeled test_data.csv file to display the six metrics and compact confusion matrix.")
    st.stop()

try:
    df = pd.read_csv(upload)
    required = feature_names + ["target"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()
    if df.empty:
        st.error("Uploaded test CSV has no rows.")
        st.stop()

    X = df.loc[:, feature_names].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        st.error("Input features must be numeric, finite, and contain no missing values.")
        st.stop()
    labels = df["target"].astype(str).str.strip().str.lower()
    if not labels.isin(["benign", "malignant"]).all():
        st.error("Target values must be 'benign' or 'malignant'.")
        st.stop()
    y = (labels == "malignant").astype(int)

    model_path = BASE / "model" / model_files[choice]
    if not model_path.is_file():
        st.error(
            f"Missing model/{model_files[choice]} in GitHub. "
            "Upload all five trained .joblib files into a folder named model."
        )
        st.stop()

    # Only load models you trained and uploaded to your own trusted repository.
    model = joblib.load(model_path)
    predicted = model.predict(X)
    class_labels = list(model.classes_)
    if 1 not in class_labels:
        st.error("The saved classifier does not contain the malignant (1) class.")
        st.stop()
    positive_idx = class_labels.index(1)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, positive_idx]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        scores = raw if positive_idx == 1 else -raw
    else:
        scores = None

    results = {
        "Accuracy": accuracy_score(y, predicted),
        "AUC": roc_auc_score(y, scores) if scores is not None and y.nunique() == 2 else None,
        "Precision": precision_score(y, predicted, zero_division=0),
        "Recall": recall_score(y, predicted, zero_division=0),
        "F1": f1_score(y, predicted, zero_division=0),
        "MCC": matthews_corrcoef(y, predicted),
    }
    cols = st.columns(6, gap="small")
    for col, (label, val) in zip(cols, results.items()):
        col.metric(label, "N/A" if val is None else f"{val:.4f}")
    st.caption(f"{len(y)} test samples · Positive class: malignant (1) · Test data are not used for fitting")

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, predicted, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(2.65, 2.45), dpi=110)
        ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d"
        )
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        for item in ax.texts:
            item.set_fontsize(10)
        fig.tight_layout(pad=0.3)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        # Crucial: fixed pixel width rather than st.pyplot(use_container_width=True).
        st.image(buf.getvalue(), width=235)

    with right:
        st.subheader("Classification report")
        report = classification_report(
            y, predicted, labels=[0, 1], target_names=["Benign", "Malignant"],
            output_dict=True, zero_division=0,
        )
        report_frame = pd.DataFrame(report).T
        st.dataframe(report_frame.style.format(precision=4), use_container_width=True)

except Exception as exc:
    st.error(f"Could not evaluate uploaded CSV: {exc}")
