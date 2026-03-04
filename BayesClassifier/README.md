# Naive Bayes Classifier (C++)

A robust, configurable C++ implementation of a Naive Bayes classifier supporting Gaussian and Rayleigh distributions. This project features a flexible pipeline for batch processing, single-point inference, and a comprehensive unit testing suite.

## Project Structure

```text
.
├── config/             # Pipeline and Model configurations
├── data/               # Shared datasets or artifacts separate from runs
├── input/              # Input data (JSON/Text) for pipeline runs
├── output/             # Generated prediction CSVs
├── include/            # Header files
│   ├── pipeline/       # Pipeline logic (Config, Helpers)
│   ├── io/             # JSON parsing, Model loading
│   ├── naive_bayes/    # Core math and distribution logic
│   └── test/           # Unit testing interface
└── src/                # Implementation files
    ├── pipeline/       # Main entry point and pipeline logic
    ├── core/           # Mathematical core
    ├── io/             # File I/O
    └── test/           # Unit test implementation
````

## Building

This project uses CMake.

```bash
mkdir build
cd build
cmake ..
make
```

## Running the Application

The application is controlled via a single executable `naive_bayes_cli`. It accepts a path to a configuration file or a test flag. Sample configs now read inputs from `input/` and write CSVs to `output/` (directories are created automatically).

### 1\. Run Unit Tests

To verify the mathematical stability (Log-Sum-Exp logic) and configuration parsers:

```bash
./naive_bayes_cli --test
```

### 2\. Run Batch Inference

Process an entire file of observations and generate a CSV of probabilities.

```bash
./naive_bayes_cli ../config/classifier/inference.batch_json.example.json
```

### 3\. Run Single Prediction

You can perform a quick check on specific numbers defined in the config without loading a data file.

```bash
./naive_bayes_cli ../config/classifier/inference.single.config.example.json
```

## Configuration Guide

The behavior is driven by JSON configuration files.

### 1\. Model Configuration (`model.configuration.example.json`)

Defines the structure of the classifier (priors, distributions). Each feature now requires a `name`, `type`, and a parameter object with explicit keys (e.g., Gaussian uses `mean` and `sigma`; Rayleigh uses `sigma`). Use consistent feature names across classes so layouts and models align.

```json
{
  "computation_mode": "log",
  "class_groups": [
    { "name": "GroupA", "classes": ["Class0", "Class1"] },
    { "name": "GroupB", "classes": ["Class2"] }
  ],
  "classes": [
    {
      "name": "Class0",
      "prior": 0.5,
      "features": [
        { "name": "temperature", "type": "gaussian", "params": { "mean": 0.0, "sigma": 1.0 } },
        { "name": "pressure", "type": "rayleigh", "params": { "sigma": 1.0 } }
      ]
    }
  ]
}
```

You can set `"computation_mode": "log"` (default) or `"computation_mode": "linear"` in the model configuration to choose the scoring space.
Optional `"class_groups"` (or `"groups"`) lets you define higher-level groupings of classes for aggregated probabilities.

### 2\. Pipeline Configuration

Controls input/output paths and data layout.

**JSON Pipeline Config:**
Use named fields to map input columns/keys onto the model’s feature order.

  * `feature_fields`: **Crucial.** This array defines the order in which named keys are fed into the classifier. In the example below, the classifier expects Feature 0 to be "temperature" and Feature 1 to be "pressure".
  * `output_use_index` (optional): When true, timesteps are assigned as a 0-based row index rather than read from the input. Useful if your input rows are stacked segments rather than chronological timesteps. The output column is always `time_ns` regardless of this setting.

<!-- end list -->

```json
{
  "input_file": "../../input/sample_input.json",
  "output_file": "../../output/predictions.csv",
  "input_format": "json",
  "model_config": "model.configuration.example.json",
  "layout": {
    "truth_field": "label",
    "feature_fields": ["temperature", "pressure"]
  }
}
```

**Single Prediction (Array):**
Provide an ordered feature vector that matches the model’s feature order.

```json
{
  "single_features": [10.0, 2.5],
  "single_truth": "ClassA"
}
```

### 3\. CSV Output Columns

The CSV output always includes:

- `time_ns` (integer nanoseconds)
- `id` (empty for CLI batch rows; populated by runtime pipeline rows when available)
- `truth_label`
- `classification_state` (`partial` or `full`)
- `feature_<FeatureName>` for each model feature, in model feature order
- `predicted_class`
- `predicted_prob`
- `prob_<ClassName>` for each class in the model

If `class_groups` are configured, the CSV also includes:

- `predicted_group`
- `predicted_group_prob`
- `group_prob_<GroupName>` for each configured group

## Live / In-Memory Inference (Model-Only)

Use the pipeline helpers directly when you want in-process inference without running the CLI.
This path still uses the JSON schema from your model configuration and supports order-independent
feature keys.

1) **Load the classifier from the model config**  
`LoadModel` reads the model file and returns a ready-to-use classifier. If `layout.feature_fields`
is empty, it is filled from the model's feature names and reordered to match the model.
```cpp
naive_bayes::pipeline::InferenceConfig cfg;
cfg.model_config = "../config/model/model.configuration.example.json";

naive_bayes::NaiveBayes clf = naive_bayes::pipeline::LoadModel(cfg);
const naive_bayes::pipeline::LayoutConfig& layout = cfg.layout;
```

2) **Predict from a JSON object (order-independent keys)**  
Keys can arrive in any order, but they must match the model feature names and be numeric.
```cpp
naive_bayes::io::Json evt = naive_bayes::io::Json::object({
    {"temperature", 10.0},
    {"pressure", 3.0}
});

naive_bayes::pipeline::SinglePrediction result =
    naive_bayes::pipeline::PredictResultFromJsonObject(clf, layout, evt);
// result.predicted_class, result.predicted_prob, result.probabilities
```
Quickly inspect the result:
```cpp
std::cout << "predicted_class: " << result.predicted_class
          << " (p=" << result.predicted_prob << ")\n";
for (const std::pair<std::string, double>& entry : result.probabilities) {
  std::cout << "  prob_" << entry.first << ": " << entry.second << "\n";
}
```
If you need timestep/truth metadata from JSON, use `PredictFromJsonObject` to get a
`BatchPredictionRow`.

3) **If you already have ordered feature vectors**
```cpp
std::vector<double> features = /* ordered to match model feature order */;
std::vector<std::pair<std::string, double>> probs = clf.PredictPosteriors(features);
std::pair<std::string, double> pred = clf.PredictClass(features);
```

4) **Grouped probabilities (if class_groups are configured)**
```cpp
std::vector<double> features = /* ordered to match model feature order */;
std::vector<std::pair<std::string, double>> group_probs = clf.PredictGroupedPosteriors(features);
std::string target_group = "dog";
double target_prob = 0.0;
for (const std::pair<std::string, double>& entry : group_probs) {
  if (entry.first == target_group) {
    target_prob = entry.second;
    break;
  }
}
std::cout << "p(" << target_group << ")=" << target_prob << "\n";
```

Notes:
- Model features have names; all classes must declare the same set/order (enforced at load).
- `LoadModel` auto-fills and aligns `layout.feature_fields` from the model when it was empty; otherwise it reorders them to match the model. Input key order doesn’t matter.
- `PredictResultFromJsonObject` throws if required feature keys are missing or non-numeric.

## Input Formats

### JSON Input

Root must be an array of objects. Each object should expose:

  * A numeric timestep key (`timestep` by default, or `layout.timestep_field`). If omitted, timesteps are auto-assigned in input order starting at 0.
  * A string truth label key (`truth` by default, or `layout.truth_field`).
  * Numeric feature keys for every entry in `layout.feature_fields` (auto-filled from the model when loading).

### Text Input

Delimited files with a required header row. The header must contain the truth and feature column names. If you include a timestep column, set `layout.timestep_field`; otherwise timesteps are auto-assigned in row order starting at 0.

```text
# time  label   f1    f2
1.0     ClassA  0.5   2.1
```

```json
"layout": {
  "truth_field": "label",
  "feature_fields": ["f1", "f2"],
  "delimiter": "SPACE"
}
```

---

## Python Preprocessing Utility (Header Handling)

The Python preprocessor (`preprocess.py`) is intended for **explicit, user-driven data preparation**. It does **not** attempt to infer whether a file has headers.

### Design Principles

* Files are assumed to have headers **by default**
* Headerless files **must be explicitly declared**
* Column names for headerless files may be:

  * Fully specified
  * Partially specified
  * Omitted entirely (auto-filled as `column0`, `column1`, …)

There is **no automatic header detection**.

---

### Example: Mixed Inputs with Headerless Files

```python
run_preprocess(
    inputs=[
        "input/with_headers.csv",
        "input/no_headers.txt",
        "input/no_headers_partial.txt",
    ],
    per_file_schema={
        "no_headers": {
            "has_header": False
        },
        "no_headers_partial": {
            "has_header": False,
            "columns": ["time", "temperature", None, "pressure"]
        }
    }
)
```

Results:

* `no_headers.txt` → `column0`, `column1`, `column2`, …
* `no_headers_partial.txt` → `time`, `temperature`, `column2`, `pressure`

### Combining inputs vertically vs horizontally

`run_preprocess` now supports `combine_mode`:

* `vertical` (default) stacks rows from each file (prior behavior).
* `horizontal` joins on the union of time values across files. Use `align_strategy` to control filling:
  * `none` leaves gaps as-is.
  * `ffill` forward-fills values to “stretch” each row until a new timestep arrives.
  * `interpolate` linearly interpolates numeric columns, then forward-fills the rest.

Horizontal mode requires each input to expose a time/timestep column (detected from `DEFAULT_TIME_FIELDS` or your override), and it prefixes columns with the source file stem to avoid collisions.

### Testing the default schema

The project ships with `tools/examples/headerless_schema_sample.csv` and a matching per-file schema entry
inside `tools/tests/test_preprocess_and_analysis.py`. Run `tools/tests/test_default_schema.py` (with the bundled
venv activated or via `tools/venv/bin/python`) to exercise that schema without passing any overrides, and
verify the reported decision metadata, the assigned column names, and a few rows of output.

```
tools/venv/bin/python tools/tests/test_default_schema.py
```

### Smoke test for the Python preprocessing + analysis stack

`tools/tests/test_preprocess_and_analysis.py` runs `run_preprocess` over the sample CSV/Text/headerless trio,
then generates a tiny synthetic prediction CSV and calls `analyze_predictions.main()` with a custom
output directory. It asserts that the combined dataset is non-empty, that all expected analysis artifacts
(classification report, metrics summary, confusion plots, misclassification list) exist, and that the metrics
summary row count matches the synthetic data.

```
tools/venv/bin/python tools/tests/test_preprocess_and_analysis.py
```

The script prints the temporary directory where the artifacts landed so you can inspect the combined CSV,
classification report, metrics summary, and plots. Run it whenever you need to confirm the preprocessing or
analysis pieces still work together.

Below is a clearer, more explanatory rewrite of that section. It keeps the same content but emphasizes **what each metric actually is**, **what problem it answers**, and **how to interpret it**, even for readers who are not ML experts.

---

## Interpreting Analysis Results

The analysis script `tools/analyze_predictions.py` reads a prediction CSV and writes evaluation artifacts to
`analysis_outputs/`. These outputs help you understand **how well the model performs**, **where it fails**, and
**why those failures happen**.

### Key Output Files

* **`classification_report.csv`**
  Per-class **precision**, **recall**, **F1 score**, and **support**.
  Use this to identify which classes perform poorly and whether errors come from:

  * **False positives** (low precision)
  * **Missed detections** (low recall)

* **`metrics_summary.json`**
  Aggregate metrics such as `accuracy`, `balanced_accuracy`, and macro-averaged precision/recall/F1, plus
  `per_class_accuracy`.
  All metrics are on a **0–1 scale** (e.g., `0.92` = 92%).

  If the CSV includes per-class probability columns (`prob_<ClassName>`), this file also includes:

  * `log_loss`
  * `macro_roc_auc`

* **Confusion matrix images**

  * `confusion_matrix_raw.png`: raw counts of true vs. predicted labels
  * `confusion_matrix_normalized.png`: row-normalized (each true class sums to 1)
  * `confusion_counts.png`: overall correct vs. incorrect (or TP/TN/FP/FN for binary)
  * `confusion_counts_binary_critical_noncritical.png`: binary collapse of all classes into
    “critical” vs “non-critical” — only produced when `DEFAULT_CRITICAL_CLASSES` is set

* **`top_misclassifications.csv`**
  If probability columns are present, this lists the **most confident wrong predictions**.
  These rows are ideal for:

  * Data quality review
  * Label audits
  * Threshold tuning or recalibration

---

## Binary Critical / Non-Critical Analysis

When `DEFAULT_CRITICAL_CLASSES` is set in `analyze_predictions.py` (default: `["TargetSmall"]`),
the script runs a second analysis pass that **collapses the multi-class problem into two buckets**:
every truth label and prediction is either *critical* (in the list) or *non-critical* (everything else).
This isolates how well the classifier handles the operationally important class regardless of overall
multi-class accuracy.

### Output: `binary_critical_noncritical` in `metrics_summary.json`

```json
"binary_critical_noncritical": {
  "positives": ["TargetSmall"],
  "counts": { "tp": ..., "tn": ..., "fp": ..., "fn": ... },
  "precision": ...,
  "recall": ...,
  "f1": ...,
  "accuracy": ...
}
```

### Count Definitions

| Count | Truth label | Predicted label | Meaning |
|---|---|---|---|
| **TP** | critical | critical | Correctly detected a critical target |
| **TN** | non-critical | non-critical | Correctly dismissed a non-critical target |
| **FP** | non-critical | critical | False alarm — non-critical target called critical |
| **FN** | critical | non-critical | **Missed detection** — critical target went undetected |

### Derived Metrics

| Metric | Formula | What it answers |
|---|---|---|
| **Recall** | TP / (TP + FN) | Of all actual critical targets, what fraction were caught? |
| **Precision** | TP / (TP + FP) | Of all critical predictions, what fraction were correct? |
| **F1** | harmonic mean of precision & recall | Balanced score when both matter |
| **Accuracy** | (TP + TN) / total | Overall binary correctness |

### How to Interpret

**Focus on recall first.** A missed critical target (FN) is typically the most dangerous failure
mode. If recall is low, the model is regularly failing to detect the targets that matter most.

**Then check precision.** A high false-alarm rate (low precision) means non-critical targets are
frequently mislabeled as critical, which reduces trust in detections.

**Be cautious with accuracy.** If critical targets are rare in the dataset, accuracy will appear
high even when recall is poor — the model can predict non-critical for nearly every row and still
score well. Prefer recall and F1 as the primary indicators.

**`DEFAULT_CRITICAL_CLASSES` accepts multiple entries** (e.g. `["TargetSmall", "TargetMedium"]`),
in which case all listed classes together form the positive side of the binary split.

---

## Metric Definitions

| Metric                    | What it actually is                                                               | What it tells you                                    | How to interpret values                                                                               |
| ------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **accuracy**              | The fraction of all predictions that are correct.                                 | Overall correctness, regardless of class.            | `1.0` is perfect. For binary problems, `0.5` is coin-flip. Can be misleading with imbalanced classes. |
| **balanced_accuracy**     | The average **recall** across all classes. Each class contributes equally.        | Whether the model recalls *every* class fairly.      | Prefer this over accuracy for imbalanced data. Low values mean one or more classes are being ignored. |
| **precision** (per class) | Of everything the model predicted *as this class*, how much was actually correct. | How many false positives the model makes.            | Low precision = model is over-predicting this class.                                                  |
| **recall** (per class)    | Of all true examples of this class, how many the model found.                     | How many cases the model misses.                     | Low recall = the model frequently fails to detect this class.                                         |
| **macro_precision**       | Precision averaged across classes (each class weighted equally).                  | Overall false-positive control across classes.       | Useful when all classes matter equally.                                                               |
| **macro_recall**          | Recall averaged across classes (each class weighted equally).                     | Overall miss rate across classes.                    | Low values indicate at least one class is consistently missed.                                        |
| **F1 score** (per class)  | The harmonic mean of precision and recall.                                        | Balance between false positives and false negatives. | High only if *both* precision and recall are high.                                                    |
| **macro_f1**              | F1 score averaged across classes.                                                 | Overall class separation quality.                    | Drops quickly if any class has poor precision or recall.                                              |
| **per_class_accuracy**    | For a given class, how often predictions involving that class are correct.        | Class-specific hit rate.                             | Low values pinpoint which classes are failing most.                                                   |
| **log_loss**              | Measures how well predicted probabilities match reality.                          | Probability calibration quality.                     | Lower is better. Values >1 usually mean overconfident mistakes.                                       |
| **macro_roc_auc**         | Measures how well the model ranks the true class above others.                    | Ranking ability independent of thresholds.           | `1.0` is perfect, `0.5` is random, `<0.5` is worse than random.                                       |

---

## How to Use These Metrics Together

* **High accuracy but low balanced accuracy**
  → Model performs well on dominant classes but ignores rare ones.

* **High recall, low precision**
  → Model finds most true cases but produces many false alarms.

* **High precision, low recall**
  → Model is conservative and misses many real cases.

* **Low macro F1**
  → At least one class has poor precision, recall, or both.

* **High ROC AUC but low accuracy**
  → Model ranks classes well but needs better thresholds.

---

### Tip

Include `prob_<ClassName>` columns in your prediction CSV to enable:

* Log-loss and ROC AUC metrics
* `true_prob` / `pred_prob` fields in `top_misclassifications.csv` for targeted error analysis

This makes it much easier to diagnose *why* the model is wrong, not just *how often*.
