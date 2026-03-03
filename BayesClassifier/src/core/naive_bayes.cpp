#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "naive_bayes/distribution.h"
#include "naive_bayes/naive_bayes.h"

namespace naive_bayes {

std::vector<double> NaiveBayes::ComputeLogJointProbabilities(
    const std::vector<double>& features, std::size_t feature_dim) const {
  std::vector<double> log_joint;
  log_joint.reserve(classes_.size());
  for (const ClassDefinition& class_model : classes_) {
    double class_log_joint = std::log(class_model.prior);
    for (std::size_t feature_index = 0; feature_index < feature_dim; ++feature_index) {
      const auto& dist = class_model.feature_models[feature_index];
      if (!dist) {
        continue;  // this class does not model this feature
      }
      double value = features[feature_index];
      if (!std::isfinite(value)) {
        continue;  // skip missing/NaN feature values
      }
      class_log_joint += dist->LogPdf(value);
    }
    log_joint.push_back(class_log_joint);
  }
  return log_joint;
}

double NaiveBayes::ComputeLogNormalizer(const std::vector<double>& log_joint) const {
  const double max_log = *std::max_element(log_joint.begin(), log_joint.end());
  double exp_sum = 0.0;
  for (double value : log_joint) {
    exp_sum += std::exp(value - max_log);
  }
  return max_log + std::log(exp_sum);
}

std::vector<std::pair<std::string, double>> NaiveBayes::BuildPosteriorFromLog(
    const std::vector<double>& log_joint, double log_normalizer) const {
  std::vector<std::pair<std::string, double>> probabilities;
  probabilities.reserve(classes_.size());
  for (std::size_t class_index = 0; class_index < classes_.size(); ++class_index) {
    double posterior = std::exp(log_joint[class_index] - log_normalizer);
    probabilities.emplace_back(classes_[class_index].name, posterior);
  }
  return probabilities;
}

std::vector<double> NaiveBayes::ComputeLinearScores(const std::vector<double>& features,
                                                    std::size_t feature_dim) const {
  std::vector<double> scores;
  scores.reserve(classes_.size());
  for (const ClassDefinition& class_model : classes_) {
    double score = class_model.prior;
    for (std::size_t feature_index = 0; feature_index < feature_dim; ++feature_index) {
      const auto& dist = class_model.feature_models[feature_index];
      if (!dist) {
        continue;  // this class does not model this feature
      }
      double value = features[feature_index];
      if (!std::isfinite(value)) {
        continue;  // skip missing/NaN feature values
      }
      score *= dist->Pdf(value);
    }
    scores.push_back(score);
  }
  return scores;
}

std::vector<std::pair<std::string, double>> NaiveBayes::NormalizeScores(
    const std::vector<double>& scores) const {
  std::vector<std::pair<std::string, double>> probabilities;
  probabilities.reserve(classes_.size());

  double score_sum = 0.0;
  for (double score : scores) {
    score_sum += score;
  }

  if (score_sum == 0.0) {
    for (const ClassDefinition& class_model : classes_) {
      probabilities.emplace_back(class_model.name, 0.0);
    }
    return probabilities;
  }

  for (std::size_t class_index = 0; class_index < classes_.size(); ++class_index) {
    probabilities.emplace_back(classes_[class_index].name, scores[class_index] / score_sum);
  }
  return probabilities;
}

void NaiveBayes::SetClassGroups(std::vector<ClassGroup> groups) {
  if (classes_.empty()) {
    throw std::logic_error("No classes configured");
  }
  std::unordered_set<std::string> known_classes;
  known_classes.reserve(classes_.size());
  for (const auto& cls : classes_) {
    known_classes.insert(cls.name);
  }
  for (const auto& group : groups) {
    if (group.name.empty()) {
      throw std::invalid_argument("Group name cannot be empty");
    }
    if (group.class_names.empty()) {
      throw std::invalid_argument("Group must contain at least one class");
    }
    for (const auto& class_name : group.class_names) {
      if (known_classes.find(class_name) == known_classes.end()) {
        throw std::invalid_argument("Unknown class in group: " + class_name);
      }
    }
  }
  class_groups_ = std::move(groups);
}

bool NaiveBayes::HasClassGroups() const {
  return !class_groups_.empty();
}

std::vector<std::pair<std::string, double>> NaiveBayes::BuildGroupedPosteriors(
    const std::vector<std::pair<std::string, double>>& posteriors,
    const std::vector<ClassGroup>& groups) const {
  std::unordered_map<std::string, double> posterior_lookup;
  posterior_lookup.reserve(posteriors.size());
  for (const auto& entry : posteriors) {
    posterior_lookup.emplace(entry.first, entry.second);
  }

  std::vector<std::pair<std::string, double>> grouped;
  grouped.reserve(groups.size());
  for (const auto& group : groups) {
    double sum = 0.0;
    for (const auto& class_name : group.class_names) {
      auto it = posterior_lookup.find(class_name);
      if (it == posterior_lookup.end()) {
        throw std::invalid_argument("Unknown class in group: " + class_name);
      }
      sum += it->second;
    }
    grouped.emplace_back(group.name, sum);
  }
  return grouped;
}

NaiveBayes::NaiveBayes(ProbabilitySpace mode) : classes_(), mode_(mode) {}

void NaiveBayes::AddClassDefinition(ClassDefinition model) {
  if (!classes_.empty()) {
    if (model.feature_models.size() != classes_[0].feature_models.size()) {
      throw std::invalid_argument("Inconsistent feature count across classes");
    }
    if (model.feature_names != classes_[0].feature_names) {
      throw std::invalid_argument("Inconsistent feature names/order across classes");
    }
  }
  if (!(model.prior > 0.0)) {
    throw std::invalid_argument("Prior must be > 0");
  }
  classes_.push_back(std::move(model));
}

std::vector<std::pair<std::string, double>> NaiveBayes::PredictPosteriors(
    const std::vector<double>& features) const {
  if (classes_.empty()) {
    throw std::logic_error("No classes configured");
  }
  std::size_t feature_dim = FeatureDim();
  if (features.size() != feature_dim) {
    throw std::invalid_argument("Bad feature size");
  }

  if (mode_ == ProbabilitySpace::kLog) {
    std::vector<double> log_joint = ComputeLogJointProbabilities(features, feature_dim);
    double log_normalizer = ComputeLogNormalizer(log_joint);
    return BuildPosteriorFromLog(log_joint, log_normalizer);
  }
  std::vector<double> scores = ComputeLinearScores(features, feature_dim);
  return NormalizeScores(scores);
}

std::vector<std::pair<std::string, double>> NaiveBayes::PredictGroupedPosteriors(
    const std::vector<double>& features) const {
  if (class_groups_.empty()) {
    throw std::logic_error("No class groups configured");
  }
  return PredictGroupedPosteriors(features, class_groups_);
}

std::vector<std::pair<std::string, double>> NaiveBayes::PredictGroupedPosteriors(
    const std::vector<double>& features,
    const std::vector<ClassGroup>& groups) const {
  std::vector<std::pair<std::string, double>> posteriors = PredictPosteriors(features);
  return BuildGroupedPosteriors(posteriors, groups);
}

std::pair<std::string, double> NaiveBayes::PredictClass(
    const std::vector<double>& features) const {
  std::vector<std::pair<std::string, double>> probabilities = PredictPosteriors(features);
  std::size_t argmax_index = 0;
  double best_probability = probabilities[0].second;
  for (std::size_t i = 1; i < probabilities.size(); ++i) {
    if (probabilities[i].second > best_probability) {
      best_probability = probabilities[i].second;
      argmax_index = i;
    }
  }
  return probabilities[argmax_index];
}

std::size_t NaiveBayes::FeatureDim() const {
  if (classes_.empty()) {
    return 0;
  }
  return classes_[0].feature_models.size();
}

const std::vector<std::string>& NaiveBayes::FeatureNames() const {
  if (classes_.empty()) {
    static const std::vector<std::string> kEmpty;
    return kEmpty;
  }
  return classes_[0].feature_names;
}

std::vector<ClassModelInfo> NaiveBayes::ClassModels() const {
  std::vector<ClassModelInfo> models;
  models.reserve(classes_.size());
  for (const ClassDefinition& class_def : classes_) {
    ClassModelInfo info;
    info.name = class_def.name;
    info.prior = class_def.prior;
    info.features.reserve(class_def.feature_models.size());
    for (std::size_t i = 0; i < class_def.feature_models.size(); ++i) {
      const std::unique_ptr<FeatureDistribution>& dist = class_def.feature_models[i];
      if (!dist) {
        continue;
      }
      FeatureModelInfo feature_info;
      feature_info.name = class_def.feature_names[i];
      feature_info.distribution_type = dist->TypeName();
      feature_info.params = dist->Params();
      info.features.push_back(std::move(feature_info));
    }
    models.push_back(std::move(info));
  }
  return models;
}

double NaiveBayes::SampleFeatureForClass(std::size_t class_index,
                                         const std::string& feature_name,
                                         std::mt19937& rng) const {
  if (class_index >= classes_.size()) {
    throw std::out_of_range("class_index out of range in SampleFeatureForClass");
  }
  const ClassDefinition& class_def = classes_[class_index];
  for (std::size_t i = 0; i < class_def.feature_names.size(); ++i) {
    if (class_def.feature_names[i] != feature_name) {
      continue;
    }
    const std::unique_ptr<FeatureDistribution>& dist = class_def.feature_models[i];
    if (!dist) {
      throw std::runtime_error("Class '" + class_def.name + "' does not model feature '" + feature_name + "'");
    }
    return dist->Sample(rng);
  }
  throw std::runtime_error("Class '" + class_def.name + "' missing feature '" + feature_name + "'");
}

}  // namespace naive_bayes
