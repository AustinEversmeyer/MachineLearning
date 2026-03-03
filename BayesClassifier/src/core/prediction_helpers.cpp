#include "pipeline/prediction_helpers.h"

#include <algorithm>
#include <stdexcept>

namespace naive_bayes::pipeline {

DetailedPrediction RunDetailedPrediction(const NaiveBayes& clf,
                                        const std::vector<double>& features) {
  auto pick_best = [](const std::vector<std::pair<std::string, double>>& entries) {
    if (entries.empty()) {
      throw std::logic_error("No probabilities available");
    }
    auto best = std::max_element(
        entries.begin(),
        entries.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
    return *best;
  };

  DetailedPrediction result;
  result.probabilities = clf.PredictPosteriors(features);
  std::pair<std::string, double> prediction = clf.PredictClass(features);
  result.predicted_class = prediction.first;
  result.predicted_prob = prediction.second;
  if (clf.HasClassGroups()) {
    result.group_probabilities = clf.PredictGroupedPosteriors(features);
    std::pair<std::string, double> group_prediction = pick_best(result.group_probabilities);
    result.predicted_group = group_prediction.first;
    result.predicted_group_prob = group_prediction.second;
  }
  return result;
}

std::vector<std::pair<std::string, double>> BuildFeatureInputs(
    const std::vector<std::string>& feature_names,
    const std::vector<double>& features) {
  if (feature_names.size() != features.size()) {
    throw std::runtime_error("Feature name/value count mismatch while building output row");
  }

  std::vector<std::pair<std::string, double>> feature_inputs;
  feature_inputs.reserve(feature_names.size());
  for (std::size_t i = 0; i < feature_names.size(); ++i) {
    feature_inputs.emplace_back(feature_names[i], features[i]);
  }
  return feature_inputs;
}

}  // namespace naive_bayes::pipeline
