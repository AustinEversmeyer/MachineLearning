#!/usr/bin/env python3
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_CONFIG = ""
SOURCE_CSV = ""
OUTPUT_DIR = "output/pdfs"
BINS = 30
LABEL_PRIORITY = ["truth_label", "predicted_class"]
FEATURE_PREFIX = "feature_"


def load_model_config(model_path: Path) -> dict:
    with model_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    classes = data.get("classes", [])
    if not classes:
        raise ValueError("Model configuration contains no classes.")

    feature_names = [feat["name"] for feat in classes[0].get("features", [])]
    if not feature_names:
        raise ValueError("First class has no features defined.")

    class_models = []
    for cls in classes:
        class_name = cls.get("name", "Unknown")
        feature_map = {}
        for feat in cls.get("features", []):
            feature_map[feat["name"]] = {
                "type": feat.get("type", "").lower(),
                "params": feat.get("params", {}),
            }
        aligned = [feature_map.get(name) for name in feature_names]
        class_models.append(
            {
                "name": class_name,
                "feature_names": feature_names,
                "feature_models": aligned,
            }
        )

    return {"feature_names": feature_names, "classes": class_models}


def load_prediction_csv(source_csv_path: Path) -> pd.DataFrame:
    if not source_csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {source_csv_path}")
    return pd.read_csv(source_csv_path)


def _is_missing_label(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in {"nan", "none", "null", "na", "n/a"}


def select_label_column(df: pd.DataFrame, label_priority: list[str]) -> str:
    for candidate in label_priority:
        if candidate not in df.columns:
            continue
        valid_count = int((~df[candidate].map(_is_missing_label)).sum())
        if valid_count > 0:
            return candidate
    raise ValueError(
        "No usable label column found. Checked: "
        + ", ".join(label_priority)
        + ". Ensure truth_label or predicted_class has non-empty values."
    )


def extract_feature_columns(
    df: pd.DataFrame, feature_prefix: str
) -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    extras: list[str] = []

    for column in df.columns:
        if not column.startswith(feature_prefix):
            continue
        feature_name = column[len(feature_prefix) :]
        if not feature_name:
            extras.append(column)
            continue
        mapping[feature_name] = column

    return mapping, extras


def group_csv_data_by_label(
    df: pd.DataFrame,
    label_col: str,
    model_feature_names: list[str],
    feature_to_column: dict[str, str],
) -> dict[str, list[list[float]]]:
    data_by_class: dict[str, list[list[float]]] = {}

    for _, row in df.iterrows():
        label_raw = row.get(label_col)
        if _is_missing_label(label_raw):
            continue
        class_name = str(label_raw).strip()

        if class_name not in data_by_class:
            data_by_class[class_name] = [[] for _ in model_feature_names]

        for feature_index, feature_name in enumerate(model_feature_names):
            column = feature_to_column.get(feature_name)
            if column is None:
                continue
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value) and math.isfinite(float(value)):
                data_by_class[class_name][feature_index].append(float(value))

    return data_by_class


def gaussian_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    coef = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    return coef * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def rayleigh_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    pdf = (x / (sigma * sigma)) * np.exp(-0.5 * (x * x) / (sigma * sigma))
    pdf[x < 0.0] = 0.0
    return pdf


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def plot_feature_pdfs(
    model_config: str,
    source_csv: str,
    output_dir: str = "output/pdfs",
    bins: int = 30,
    label_priority: list[str] | None = None,
    feature_prefix: str = "feature_",
) -> None:
    if not model_config:
        raise ValueError("model_config must be provided.")
    if not source_csv:
        raise ValueError("source_csv must be provided.")

    model_path = Path(model_config).resolve()
    source_csv_path = Path(source_csv).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model = load_model_config(model_path)
    df = load_prediction_csv(source_csv_path)
    priorities = label_priority or ["truth_label", "predicted_class"]
    label_col = select_label_column(df, priorities)

    feature_to_column, malformed_feature_cols = extract_feature_columns(df, feature_prefix)
    if malformed_feature_cols:
        print(
            "Warning: ignoring malformed feature columns: "
            + ", ".join(sorted(malformed_feature_cols))
        )

    model_feature_names = model["feature_names"]
    classes = model["classes"]

    missing_model_features = [name for name in model_feature_names if name not in feature_to_column]
    if missing_model_features:
        print(
            "Warning: source CSV missing model feature columns: "
            + ", ".join(f"{feature_prefix}{name}" for name in missing_model_features)
            + ". Plots for those features will be skipped."
        )

    extra_csv_features = sorted(name for name in feature_to_column if name not in set(model_feature_names))
    if extra_csv_features:
        print(
            "Warning: source CSV has extra feature columns not in model: "
            + ", ".join(f"{feature_prefix}{name}" for name in extra_csv_features)
            + ". They will be ignored."
        )

    data_by_class = group_csv_data_by_label(
        df=df,
        label_col=label_col,
        model_feature_names=model_feature_names,
        feature_to_column=feature_to_column,
    )

    for feature_index, feature_name in enumerate(model_feature_names):
        if feature_name not in feature_to_column:
            continue

        data_values = []
        for _, per_feature in data_by_class.items():
            if feature_index < len(per_feature):
                data_values.extend(per_feature[feature_index])

        data_min = min(data_values) if data_values else None
        data_max = max(data_values) if data_values else None

        model_min = None
        model_max = None
        for cls in classes:
            model_def = cls["feature_models"][feature_index]
            if not model_def:
                continue
            dist_type = model_def["type"]
            params = model_def["params"]
            if dist_type == "gaussian":
                mean = float(params.get("mean", 0.0))
                sigma = float(params.get("sigma", 1.0))
                model_min = mean - 4.0 * sigma if model_min is None else min(model_min, mean - 4.0 * sigma)
                model_max = mean + 4.0 * sigma if model_max is None else max(model_max, mean + 4.0 * sigma)
            elif dist_type == "rayleigh":
                sigma = float(params.get("sigma", 1.0))
                model_min = 0.0 if model_min is None else min(model_min, 0.0)
                model_max = 4.0 * sigma if model_max is None else max(model_max, 4.0 * sigma)

        x_min_candidates = [v for v in (data_min, model_min) if v is not None]
        x_max_candidates = [v for v in (data_max, model_max) if v is not None]
        if not x_min_candidates or not x_max_candidates:
            x_min, x_max = 0.0, 1.0
        else:
            x_min = min(x_min_candidates)
            x_max = max(x_max_candidates)
            if x_min == x_max:
                x_min -= 1.0
                x_max += 1.0

        padding = 0.05 * (x_max - x_min)
        x_min -= padding
        x_max += padding

        x_values = np.linspace(x_min, x_max, 400)
        fig, ax = plt.subplots(figsize=(8, 4.5))

        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for class_index, cls in enumerate(classes):
            model_def = cls["feature_models"][feature_index]
            if not model_def:
                continue
            dist_type = model_def["type"]
            params = model_def["params"]
            color = color_cycle[class_index % len(color_cycle)]
            label_prefix = cls["name"]

            if dist_type == "gaussian":
                mean = float(params.get("mean", 0.0))
                sigma = float(params.get("sigma", 1.0))
                if sigma <= 0.0:
                    print(
                        f"Warning: skipping gaussian PDF for class '{label_prefix}', feature '{feature_name}' due to non-positive sigma"
                    )
                else:
                    y_values = gaussian_pdf(x_values, mean, sigma)
                    ax.plot(x_values, y_values, color=color, label=f"{label_prefix} PDF")
            elif dist_type == "rayleigh":
                sigma = float(params.get("sigma", 1.0))
                if sigma <= 0.0:
                    print(
                        f"Warning: skipping rayleigh PDF for class '{label_prefix}', feature '{feature_name}' due to non-positive sigma"
                    )
                else:
                    y_values = rayleigh_pdf(x_values, sigma)
                    ax.plot(x_values, y_values, color=color, label=f"{label_prefix} PDF")
            else:
                print(
                    f"Warning: unsupported distribution '{dist_type}' for class '{label_prefix}', feature '{feature_name}'. Skipping PDF curve."
                )

            if cls["name"] in data_by_class:
                samples = data_by_class[cls["name"]][feature_index]
                if samples:
                    ax.hist(
                        samples,
                        bins=bins,
                        density=True,
                        alpha=0.25,
                        color=color,
                        label=f"{label_prefix} data",
                    )

        ax.set_title(f"Feature '{feature_name}': PDFs vs data ({label_col})")
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.2)

        filename = f"pdf_{sanitize_filename(feature_name)}.png"
        fig.tight_layout()
        fig.savefig(output_dir_path / filename, dpi=150)
        plt.close(fig)

    print(f"Wrote plots to {output_dir_path}")


if __name__ == "__main__":
    plot_feature_pdfs(
        model_config=MODEL_CONFIG,
        source_csv=SOURCE_CSV,
        output_dir=OUTPUT_DIR,
        bins=BINS,
        label_priority=LABEL_PRIORITY,
        feature_prefix=FEATURE_PREFIX,
    )
