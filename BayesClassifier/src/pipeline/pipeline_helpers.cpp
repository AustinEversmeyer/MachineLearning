#include "pipeline/pipeline_helpers.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "io/model_loader.h"
#include "io/json.h"

namespace naive_bayes::pipeline {

namespace {

constexpr double kNsPerSecond = 1000000000.0;

struct JsonPredictionInputs {
  std::vector<double> features;
  double timestep = 0.0;
  std::string truth_label;
};

JsonPredictionInputs ExtractJsonInputs(const LayoutConfig& layout,
                                       const naive_bayes::io::Json& json_obj) {
  if (layout.feature_fields.empty()) {
    throw std::runtime_error("layout.feature_fields is required for JSON key/value prediction");
  }

  JsonPredictionInputs inputs;
  inputs.features.reserve(layout.feature_fields.size());
  for (const std::string& key : layout.feature_fields) {
    if (!json_obj.contains(key) || json_obj.at(key).is_null()) {
      inputs.features.push_back(std::numeric_limits<double>::quiet_NaN());
      continue;
    }
    if (!json_obj.at(key).is_number()) {
      throw std::runtime_error("Missing or non-numeric feature key: " + key);
    }
    inputs.features.push_back(json_obj.at(key).get<double>());
  }

  if (!layout.timestep_field.empty() && json_obj.contains(layout.timestep_field) &&
      json_obj.at(layout.timestep_field).is_number()) {
    inputs.timestep = json_obj.at(layout.timestep_field).get<double>();
  }

  if (!layout.truth_field.empty() && json_obj.contains(layout.truth_field) &&
      json_obj.at(layout.truth_field).is_string()) {
    inputs.truth_label = json_obj.at(layout.truth_field).get<std::string>();
  }

  return inputs;
}

SinglePrediction ToSinglePrediction(const DetailedPrediction& detailed_prediction) {
  SinglePrediction result;
  result.predicted_class = detailed_prediction.predicted_class;
  result.predicted_prob = detailed_prediction.predicted_prob;
  result.probabilities = detailed_prediction.probabilities;
  result.predicted_group = detailed_prediction.predicted_group;
  result.predicted_group_prob = detailed_prediction.predicted_group_prob;
  result.group_probabilities = detailed_prediction.group_probabilities;
  return result;
}

}  // namespace

NaiveBayes LoadModel(InferenceConfig& config) {
  if (!config.model_config.has_value()) {
    throw std::runtime_error("Pipeline configuration missing model_config path");
  }
  NaiveBayes model = naive_bayes::io::LoadModelConfiguration(*config.model_config);
  AlignLayoutWithModel(config.layout, model.FeatureNames());
  return model;
}

std::vector<BatchPredictionRow> RunInference(const NaiveBayes& clf,
                                             const std::vector<Observation>& observations) {
  const std::vector<std::string>& feature_names = clf.FeatureNames();
  std::vector<BatchPredictionRow> prediction_rows;
  prediction_rows.reserve(observations.size());
  for (const Observation& observation : observations) {
    DetailedPrediction result = RunDetailedPrediction(clf, observation.features);

    BatchPredictionRow row;
    row.time_ns = static_cast<int64_t>(std::llround(observation.timestep * kNsPerSecond));
    row.truth_label = observation.truth_label;
    row.predicted_class = std::move(result.predicted_class);
    row.predicted_prob = result.predicted_prob;
    row.probabilities = std::move(result.probabilities);
    row.predicted_group = std::move(result.predicted_group);
    row.predicted_group_prob = result.predicted_group_prob;
    row.group_probabilities = std::move(result.group_probabilities);
    row.feature_inputs = BuildFeatureInputs(feature_names, observation.features);
    prediction_rows.push_back(std::move(row));
  }
  return prediction_rows;
}

void PrintSinglePrediction(const NaiveBayes& clf,
                           const SinglePredictionConfig& single_input) {
  std::vector<double> features_to_use = single_input.features;
  if (features_to_use.size() < clf.FeatureDim()) {
    features_to_use.resize(clf.FeatureDim(), std::numeric_limits<double>::quiet_NaN());
  } else if (features_to_use.size() > clf.FeatureDim()) {
    std::cerr << "Error: Feature dimension mismatch. Expected " << clf.FeatureDim()
              << ", got " << features_to_use.size() << "\n";
    return;
  }

  DetailedPrediction result = RunDetailedPrediction(clf, features_to_use);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Single prediction result:\n";
  if (single_input.timestep.has_value()) {
    std::cout << "  timestep: " << *single_input.timestep << "\n";
  }
  if (single_input.truth_label.has_value()) {
    std::cout << "  truth: " << *single_input.truth_label << "\n";
  }
  std::cout << "  predicted_class: " << result.predicted_class
            << " (p=" << result.predicted_prob << ")\n";
  for (const auto& entry : result.probabilities) {
    std::cout << "    prob_" << entry.first << ": " << entry.second << "\n";
  }
  if (!result.group_probabilities.empty()) {
    std::cout << "  predicted_group: " << result.predicted_group
              << " (p=" << result.predicted_group_prob << ")\n";
    for (const auto& entry : result.group_probabilities) {
      std::cout << "    group_prob_" << entry.first << ": " << entry.second << "\n";
    }
  }
}

SinglePrediction PredictResultFromJsonObject(const NaiveBayes& clf,
                                             const LayoutConfig& layout,
                                             const naive_bayes::io::Json& json_obj) {
  JsonPredictionInputs inputs = ExtractJsonInputs(layout, json_obj);
  DetailedPrediction detailed = RunDetailedPrediction(clf, inputs.features);
  return ToSinglePrediction(detailed);
}

BatchPredictionRow PredictFromJsonObject(const NaiveBayes& clf,
                                         const LayoutConfig& layout,
                                         const naive_bayes::io::Json& json_obj) {
  JsonPredictionInputs inputs = ExtractJsonInputs(layout, json_obj);
  DetailedPrediction result = RunDetailedPrediction(clf, inputs.features);
  const std::vector<std::string>& feature_names = clf.FeatureNames();

  BatchPredictionRow row;
  row.time_ns = static_cast<int64_t>(std::llround(inputs.timestep * kNsPerSecond));
  row.truth_label = inputs.truth_label;
  row.feature_inputs = BuildFeatureInputs(feature_names, inputs.features);
  row.predicted_class = std::move(result.predicted_class);
  row.predicted_prob = result.predicted_prob;
  row.probabilities = std::move(result.probabilities);
  row.predicted_group = std::move(result.predicted_group);
  row.predicted_group_prob = result.predicted_group_prob;
  row.group_probabilities = std::move(result.group_probabilities);
  return row;
}

void AlignLayoutWithModel(LayoutConfig& layout,
                          const std::vector<std::string>& model_feature_names) {
  if (model_feature_names.empty()) {
    return;
  }

  if (layout.feature_fields.empty()) {
    layout.feature_fields = model_feature_names;
    return;
  }
  if (layout.feature_fields.size() != model_feature_names.size()) {
    throw std::runtime_error("Layout feature count does not match model feature count");
  }

  std::unordered_set<std::string> layout_set(layout.feature_fields.begin(), layout.feature_fields.end());
  for (const std::string& name : model_feature_names) {
    if (layout_set.find(name) == layout_set.end()) {
      throw std::runtime_error("Layout missing feature present in model: " + name);
    }
  }

  // Reorder layout to match model order so downstream vector building is correct.
  layout.feature_fields = model_feature_names;
}

}  // namespace naive_bayes::pipeline
