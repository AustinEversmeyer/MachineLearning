#pragma once

#include "naive_bayes/naive_bayes.h"

#include <string>
#include <utility>
#include <vector>

namespace naive_bayes::pipeline {

struct DetailedPrediction {
  std::string predicted_class;
  double predicted_prob{};
  std::vector<std::pair<std::string, double>> probabilities;
  std::string predicted_group;
  double predicted_group_prob{};
  std::vector<std::pair<std::string, double>> group_probabilities;
};

DetailedPrediction RunDetailedPrediction(const NaiveBayes& clf,
                                        const std::vector<double>& features);

std::vector<std::pair<std::string, double>> BuildFeatureInputs(
    const std::vector<std::string>& feature_names,
    const std::vector<double>& features);

}  // namespace naive_bayes::pipeline
