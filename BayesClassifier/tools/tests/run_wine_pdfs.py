#!/usr/bin/env python3
from pathlib import Path
import sys

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

from tools.scripts.plot_feature_pdfs import plot_feature_pdfs


def main() -> None:
    model_config = root_dir / "tools" / "outputs" / "wine_model.configuration.json"
    source_csv = root_dir / "tools" / "outputs" / "synthetic_predictions.csv"
    output_dir = root_dir / "tools" / "outputs" / "wine_pdfs"

    plot_feature_pdfs(
        model_config=str(model_config),
        source_csv=str(source_csv),
        output_dir=str(output_dir),
        bins=30,
        label_priority=["truth_label", "predicted_class"],
        feature_prefix="feature_",
    )


if __name__ == "__main__":
    main()
