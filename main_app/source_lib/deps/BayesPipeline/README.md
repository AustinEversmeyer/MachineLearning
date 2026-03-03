# BayesPipeline

`BayesPipeline` is a runtime inference layer that accepts streaming feature events, aligns features by event time, performs Naive Bayes classification, and writes prediction CSV output.

This library is built as a static library (`libbayes_pipeline.a`) and used by `main_app/source_lib`.

## What It Does

1. Receives `FeatureData` events through `IFeaturePublisher::PublishFeature`.
2. Queues events in `IngestQueue` for a worker thread.
3. Aligns model features by `id` and anchor timestamp (`time_ns`) in `FeatureAlignmentStore`.
4. Runs classification in `BayesClassifierManager` using `naive_bayes::NaiveBayes`.
5. Buffers runtime rows and writes CSV at `BayesRuntimeManager::Stop()`.

## Core Components

- `BayesRuntimeManager`
  - Runtime entry point.
  - Owns queue worker, classifier manager, and output row buffer.
  - Public construction modes:
    - `BayesRuntimeManager(runtime_config_path, model_config_path)`
    - `BayesRuntimeManager(model_config_path, output_file, max_records, time_tolerance_ns, evaluation_policy, partial_policy, partial_grace_window_ns)`
- `BayesRuntimeConfig`
  - Runtime knobs loaded from JSON (excluding model path by design).
- `IngestQueue`
  - Thread-safe producer/consumer queue for `FeatureData`.
- `FeatureAlignmentStore` (`DataSink.*`)
  - Event-time feature buffering and joining.
- `BayesClassifierManager`
  - Classification trigger/evaluation policy logic and duplicate-suppression state machine.
  - Uses shared BayesClassifier helpers (`RunDetailedPrediction`, `BuildFeatureInputs`) so runtime and CLI emit consistent probability/group outputs.

## Data Model

### Input event
- `FeatureData`
  - `id: int`
  - `time_ns: int64_t`
  - `feature_name: std::string`
  - `value: double`
  - `truth_label: std::optional<std::string>`

### Joined candidate
- `JoinedFeatureVector`
  - `id`
  - `anchor_time_ns`
  - `feature_values` (missing features represented as `NaN` when partial is enabled)
  - `truth_label`
  - `is_partial`

### Classification output (in-memory)
- `ClassificationResult`
  - key fields: `id`, `time_ns`, `truth_label`, `classification_state`
  - features: `feature_inputs`
  - model outputs: `predicted_class`, `predicted_prob`, `posteriors`
  - group outputs (if class groups configured): `predicted_group`, `predicted_group_prob`, `group_posteriors`
  - `is_partial`

## Runtime Config JSON

`runtime.config.json` controls runtime behavior except model path.

Current schema:
- `output_file` (string, optional)
- `max_records` (positive integer, optional)
- `time_tolerance_ns` (non-negative integer, optional)
- `evaluation_policy` (string enum, optional)
- `partial_policy` (string enum, optional)
- `partial_grace_window_ns` (non-negative integer, optional)

Defaults:
- `output_file = "bayes_classifier_output.csv"`
- `max_records = 10`
- `time_tolerance_ns = 1000000000`
- `evaluation_policy = "hybrid_deadline"`
- `partial_policy = "allow_after_deadline"`
- `partial_grace_window_ns = 200000000`

Path behavior:
- Relative `output_file` paths are resolved relative to the runtime config file directory.

## Model Path Ownership

`model_config_path` is intentionally **not** part of runtime config.

Production callers are expected to resolve model location externally and pass it explicitly:

```cpp
BayesPipeline::BayesRuntimeManager mgr(runtime_config_path, model_config_path);
```

Validation:
- Empty model path -> error
- Nonexistent model path -> error
- Unreadable model path -> error

## Evaluation and Partial Policies

### `EvaluationPolicy`
- `kImmediateAnyArrival`
  - Evaluate on every incoming event.
- `kPrimaryOnly`
  - Evaluate only when the model's primary feature arrives.
- `kHybridDeadline`
  - Currently evaluates on every event (same check cadence as immediate), typically paired with deadline-based partial policy.

### `PartialPolicy`
- `kDisallow`
  - Never emit partial rows.
- `kAllowAfterDeadline`
  - Emit partial rows only when `last_event_time_ns >= anchor_time_ns + partial_grace_window_ns`.
- `kAlwaysAllow`
  - Emit partial rows as soon as a partial candidate exists.

### Interaction
- Any evaluation policy can be paired with any partial policy.
- Evaluation policy controls **when** the manager checks.
- Partial policy controls **what** can be emitted when checks run.

## Emission and Duplicate Suppression

Classification is keyed by `(id, anchor_time_ns)`.

Current behavior per key:
- At most one emission state is written.
- Full rows emit once when complete alignment exists.
- Partial rows emit once when allowed.
- No repeated same-key state emissions.

## CSV Output

CSV writing is delegated to `naive_bayes::pipeline::WritePredictionsCsv`.
`BayesRuntimeManager` accumulates rows while running and writes once at `Stop()`.

Columns include:
- `time_ns`, `id`, `truth_label`, `classification_state`
- `feature_<name>` columns
- `predicted_class`, `predicted_prob`, `prob_*`
- optional group columns if model groups are configured

## Shared Logic Ownership

- Prediction assembly (`predicted_class`, `posteriors`, grouped posteriors, best group) is owned by BayesClassifier helper APIs and reused by runtime and CLI paths.
- Feature name/value zipping for output rows is owned by BayesClassifier helper APIs and reused by runtime and CLI paths.
- Synthetic model-driven data generation in `main_app` loads the model with `LoadModelConfiguration`, reads class metadata through `NaiveBayes::ClassModels()`, and samples feature values through `NaiveBayes::SampleFeatureForClass(...)`.
- Sampling behavior is owned by BayesClassifier distribution objects (`FeatureDistribution::Sample`), so simulator support follows whatever model distribution types BayesClassifier implements (for example: gaussian, rayleigh).

## Build

Standalone in this folder:

```bash
cmake -S . -B build
cmake --build build
```

Integrated via `main_app/source_lib/build.xml`:
- `ant` builds BayesClassifier static lib, BayesPipeline static lib, `main_app`, and `scenario_tests`.

## Runtime Lifecycle

1. Construct `BayesRuntimeManager`.
2. Call `Start()` once.
3. Publish events via processors (`PublishFeature`).
4. Call `Stop()` to flush and write CSV.

`Start()`/`Stop()` are idempotent-guarded for repeated calls.

## Testing Notes

`main_app/source_lib/cppsource/tests/scenario_tests.cpp` covers:
- full classification path
- grace-window partial behavior
- deterministic scenario CSV replay
- runtime config parsing and validation checks

## Troubleshooting

- Output CSV appears under `deps/BayesPipeline/config/runtime/`
  - Cause: relative `output_file` in runtime config resolves from runtime config directory.
  - Fix: use an absolute path or a relative path that points where you want.

- `model_config_path does not exist`
  - Ensure caller passes a valid model path to runtime manager constructor.

- No rows written
  - Rows are written at `Stop()`.
  - Verify events are published and classification conditions are met.
