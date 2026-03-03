#!/usr/bin/env python3
"""
Analyze classifier outputs stored in a CSV file.

Expected columns (defaults configurable via the constants below):
  - truth_label: ground-truth class name
  - predicted_class: model prediction
  - predicted_prob: confidence of the predicted class (optional)
  - prob_<ClassName>: per-class probability columns (optional but enables log-loss/ROC AUC)

Example:
    python tools/analyze_predictions.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from difflib import get_close_matches

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import seaborn as sns
except ImportError:  # seaborn is optional; matplotlib fallback will be used
    sns = None

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from tools.scripts.plot_feature_pdfs import plot_feature_pdfs

# ---------- Editable defaults  ----------
DEFAULT_CSV_PATH = str(
    (ROOT_DIR.parent / "main_app" / "source_lib" / "deps" / "BayesPipeline" / "config" / "runtime" / "bayes_classifier_output.csv").resolve()
)
DEFAULT_TRUTH_COLUMN = "truth_label"
DEFAULT_PRED_COLUMN = "predicted_class"
DEFAULT_PROB_PREFIX = "prob_"
DEFAULT_OUTPUT_DIR = "analysis_outputs_runtime"
DEFAULT_SHOW = True
DEFAULT_CRITICAL_CLASSES: list[str] = ["TargetSmall"]
DEFAULT_CAST_LABELS_TO_CATEGORY = True  # set False if you prefer to keep original dtypes
DEFAULT_PDF_PLOTS = True
DEFAULT_PDF_MODEL_CONFIG = str(
    (ROOT_DIR.parent / "main_app" / "source_lib" / "deps" / "BayesPipeline" / "config" / "model" / "implementation.model.json").resolve()
)
DEFAULT_PDF_SOURCE_CSV = DEFAULT_CSV_PATH
DEFAULT_PDF_LABEL_PRIORITY: list[str] = ["truth_label", "predicted_class"]
DEFAULT_PDF_FEATURE_PREFIX = "feature_"
DEFAULT_PDF_OUTPUT_DIR = "analysis_outputs_runtime/pdfs"
DEFAULT_PDF_BINS = 30
# ----------------------------------------------------------------


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def drop_missing_labels(df: pd.DataFrame, truth_col: str, pred_col: str) -> tuple[pd.DataFrame, int]:
    """Remove rows where truth/pred labels are NaN."""
    mask = df[[truth_col, pred_col]].notna().all(axis=1)
    dropped = int((~mask).sum())
    return df.loc[mask].reset_index(drop=True), dropped


def extract_probability_matrix(
    df: pd.DataFrame, labels: list[str], prefix: str
) -> tuple[np.ndarray | None, list[str]]:
    columns: list[str] = []
    for label in labels:
        column_name = f"{prefix}{label}"
        if column_name not in df.columns:
            return None, []
        columns.append(column_name)
    matrix = df[columns].to_numpy(dtype=float)
    return matrix, columns


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    output_dir: Path,
    normalized: bool = False,
    keep_open: bool = False,
) -> Path:
    suffix = "normalized" if normalized else "raw"
    fmt = ".2f" if normalized else "d"
    figsize = (max(6, 0.8 * len(labels)), max(6, 0.8 * len(labels)))
    fig, ax = plt.subplots(figsize=figsize)

    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            linewidths=0.5,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
        )
    else:
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, format(value, fmt), ha="center", va="center", color="black")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    title = "Confusion Matrix"
    if normalized:
        title += " (row-normalized)"
    ax.set_title(title)
    fig.tight_layout()

    destination = output_dir / f"confusion_matrix_{suffix}.png"
    fig.savefig(destination, dpi=200)
    if normalized and sns is None:
        # keep scientific notation sensible when using matplotlib fallback
        ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    if not keep_open and not plt.isinteractive():
        plt.close(fig)

    return destination


def plot_confusion_counts(confusion: np.ndarray, output_dir: Path, keep_open: bool = False) -> Path:
    """
    Plot a compact summary of confusion statistics.

    Always shows overall correct vs incorrect counts.
    """
    total = confusion.sum()

    if total == 0:
        labels = ["Empty"]
        values = np.array([0.0], dtype=float)
        percents = np.array([0.0], dtype=float)
    else:
        correct = float(np.trace(confusion))
        incorrect = float(total - correct)
        labels = ["Correct", "Incorrect"]
        values = np.array([correct, incorrect], dtype=float)
        percents = values / total

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        labels,
        values,
        color=["#4caf50", "#2196f3", "#f44336", "#ff9800"][: len(labels)],
        alpha=0.9,
    )
    ax.set_ylabel("Count")
    ax.set_title("Overall Confusion Summary")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, count, pct in zip(bars, values, percents):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(count)} ({pct*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    destination = output_dir / "confusion_counts.png"
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    if not keep_open and not plt.isinteractive():
        plt.close(fig)

    return destination


def compute_binary_counts(y_true: pd.Series, y_pred: pd.Series, positives: set[str]) -> dict[str, float]:
    true_pos_mask = y_true.isin(positives)
    pred_pos_mask = y_pred.isin(positives)

    tp = int((true_pos_mask & pred_pos_mask).sum())
    tn = int((~true_pos_mask & ~pred_pos_mask).sum())
    fp = int((~true_pos_mask & pred_pos_mask).sum())
    fn = int((true_pos_mask & ~pred_pos_mask).sum())

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def plot_binary_counts(counts: dict[str, float], output_dir: Path, keep_open: bool = False) -> Path:
    labels = ["TP", "TN", "FP", "FN"]
    values = np.array([counts["tp"], counts["tn"], counts["fp"], counts["fn"]], dtype=float)
    total = values.sum()
    percents = values / total if total > 0 else np.zeros_like(values)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        labels,
        values,
        color=["#4caf50", "#2196f3", "#f44336", "#ff9800"],
        alpha=0.9,
    )
    ax.set_ylabel("Count")
    ax.set_title("Critical vs Non-Critical Confusion (binary collapse)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, count, pct in zip(bars, values, percents):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(count)} ({pct*100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    destination = output_dir / "confusion_counts_binary_critical_noncritical.png"
    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    if not keep_open and not plt.isinteractive():
        plt.close(fig)

    return destination


def main() -> None:
    def log(msg: str) -> None:
        print(f"[analyze] {msg}")

    csv_path = Path(DEFAULT_CSV_PATH)
    output_dir = ensure_output_dir(Path(DEFAULT_OUTPUT_DIR))
    truth_column = DEFAULT_TRUTH_COLUMN
    pred_column = DEFAULT_PRED_COLUMN
    prob_prefix = DEFAULT_PROB_PREFIX
    show_plots = DEFAULT_SHOW
    critical_classes = DEFAULT_CRITICAL_CLASSES
    pdf_plots = DEFAULT_PDF_PLOTS
    pdf_model_config = DEFAULT_PDF_MODEL_CONFIG
    pdf_source_csv = DEFAULT_PDF_SOURCE_CSV
    pdf_label_priority = DEFAULT_PDF_LABEL_PRIORITY
    pdf_feature_prefix = DEFAULT_PDF_FEATURE_PREFIX
    pdf_output_dir = DEFAULT_PDF_OUTPUT_DIR
    pdf_bins = DEFAULT_PDF_BINS

    log(f"Loading predictions from {csv_path} ...")
    df = pd.read_csv(csv_path)
    log(f"Loaded dataframe with shape {df.shape}")

    log("Preparing label columns ...")
    if truth_column not in df.columns or pred_column not in df.columns:
        missing = {col for col in (truth_column, pred_column) if col not in df.columns}
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df, dropped_labels = drop_missing_labels(df, truth_column, pred_column)
    if dropped_labels:
        log(f"Dropped {dropped_labels} rows with NaNs in truth/pred columns")

    if DEFAULT_CAST_LABELS_TO_CATEGORY:
        df[truth_column] = pd.Categorical(df[truth_column])
        df[pred_column] = pd.Categorical(df[pred_column])
        log("Casted truth/pred columns to category to reduce memory and speed comparisons")

    y_true = df[truth_column].astype(str)
    y_pred = df[pred_column].astype(str)
    labels = np.unique(np.concatenate((y_true.to_numpy(), y_pred.to_numpy()))).tolist()
    log(f"Found {len(labels)} unique labels")

    log("Computing confusion matrices ...")
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized_confusion = np.divide(
        confusion,
        row_sums,
        out=np.zeros_like(confusion, dtype=float),
        where=row_sums != 0,
    )

    log("Computing aggregate metrics ...")
    metrics = {
        "samples": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }

    per_class_accuracy: dict[str, float | None] = {}
    for label in labels:
        mask = y_true == label
        total = int(mask.sum())
        if total == 0:
            per_class_accuracy[label] = None
        else:
            per_class_accuracy[label] = float((y_pred[mask] == label).mean())
    metrics["per_class_accuracy"] = per_class_accuracy

    log("Checking probability columns for log-loss / ROC AUC ...")
    proba_matrix_raw, proba_columns = extract_probability_matrix(df, labels, prob_prefix)
    if proba_matrix_raw is not None:
        proba_mask = ~np.isnan(proba_matrix_raw).any(axis=1)
        dropped_prob_rows = int((~proba_mask).sum())
        if dropped_prob_rows:
            log(f"Dropped {dropped_prob_rows} rows with NaNs in probability columns for log-loss/ROC AUC")
        proba_matrix = proba_matrix_raw[proba_mask]
        y_true_for_proba = y_true[proba_mask]

        try:
            metrics["log_loss"] = float(log_loss(y_true_for_proba, proba_matrix, labels=labels))
        except ValueError as err:
            metrics["log_loss"] = None
            print(f"Warning: could not compute log-loss ({err})")

        try:
            metrics["macro_roc_auc"] = float(
                roc_auc_score(
                    y_true_for_proba,
                    proba_matrix,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
        except ValueError as err:
            metrics["macro_roc_auc"] = None
            print(f"Warning: could not compute ROC AUC ({err})")

        # Add helpful columns for downstream sorting/inspection
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        df["true_prob"] = [
            proba_matrix_raw[i, label_to_index[label]] for i, label in enumerate(y_true)
        ]
        df["pred_prob"] = [
            proba_matrix_raw[i, label_to_index[label]] for i, label in enumerate(y_pred)
        ]
    elif "predicted_prob" in df.columns:
        df["pred_prob"] = df["predicted_prob"]

    log("Building classification report ...")
    report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_path, float_format="%.4f")

    log("Plotting confusion matrices ...")
    confusion_path = plot_confusion_matrix(confusion, labels, output_dir, normalized=False, keep_open=show_plots)
    normalized_path = plot_confusion_matrix(
        normalized_confusion, labels, output_dir, normalized=True, keep_open=show_plots
    )
    counts_path = plot_confusion_counts(confusion, output_dir, keep_open=show_plots)

    if critical_classes:
        positives = set(critical_classes)
        label_set = set(labels)
        unmatched = [name for name in positives if name not in label_set]
        if unmatched:
            suggestions = {
                name: get_close_matches(name, labels, n=3, cutoff=0.6) for name in unmatched
            }
            print(
                "\nWarning: critical class labels not found in data. "
                "Check casing/underscores.\n"
                f"  Unmatched: {unmatched}\n"
                f"  Suggestions: {suggestions}"
            )
        binary_metrics = compute_binary_counts(y_true, y_pred, positives)
        metrics["binary_critical_noncritical"] = {
            "positives": sorted(positives),
            "counts": {
                "tp": binary_metrics["tp"],
                "tn": binary_metrics["tn"],
                "fp": binary_metrics["fp"],
                "fn": binary_metrics["fn"],
            },
            "precision": binary_metrics["precision"],
            "recall": binary_metrics["recall"],
            "f1": binary_metrics["f1"],
            "accuracy": binary_metrics["accuracy"],
        }
        print(
            "\nBinary critical/non-critical aggregate (positives: {}):".format(
                ", ".join(sorted(positives))
            )
        )
        print(
            f"  TP={binary_metrics['tp']}  TN={binary_metrics['tn']}  FP={binary_metrics['fp']}  FN={binary_metrics['fn']}"
        )
        print(
            f"  precision={binary_metrics['precision']:.4f}  recall={binary_metrics['recall']:.4f}  "
            f"f1={binary_metrics['f1']:.4f}  accuracy={binary_metrics['accuracy']:.4f}"
        )
        binary_counts_path = plot_binary_counts(binary_metrics, output_dir, keep_open=show_plots)

    log("Writing metrics summary ...")
    metrics_path = output_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    log("Collecting top misclassifications ...")
    misclassified = df[df[truth_column] != df[pred_column]]
    if not misclassified.empty and "pred_prob" in misclassified.columns:
        top_misses = misclassified.sort_values("pred_prob", ascending=False).head(25)
        top_misses.to_csv(output_dir / "top_misclassifications.csv", index=False)

    print("\nOverall metrics:")
    for key, value in metrics.items():
        if key == "per_class_accuracy":
            print("  per_class_accuracy:")
            for label, acc in value.items():
                display = "n/a" if acc is None else f"{acc:.3f}"
                print(f"    {label}: {display}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    if pdf_plots:
        if not pdf_source_csv:
            raise ValueError("DEFAULT_PDF_SOURCE_CSV must be set to enable PDF plots.")
        log("Generating PDF plots ...")
        plot_feature_pdfs(
            model_config=pdf_model_config,
            source_csv=pdf_source_csv,
            output_dir=pdf_output_dir,
            bins=pdf_bins,
            label_priority=pdf_label_priority,
            feature_prefix=pdf_feature_prefix,
        )

    print(f"\nSaved confusion matrix plot: {confusion_path}")
    print(f"Saved normalized confusion matrix plot: {normalized_path}")
    print(f"Saved classification report: {report_path}")
    print(f"Saved metrics summary: {metrics_path}")
    if not misclassified.empty and "pred_prob" in misclassified.columns:
        print(
            "Saved top misclassifications: "
            f"{output_dir / 'top_misclassifications.csv'}"
        )

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
