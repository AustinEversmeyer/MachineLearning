#pragma once
#include <memory>
#include <random>
#include <string>
#include <vector>
#include "naive_bayes/distribution.h"
#include "naive_bayes/types.h"

namespace naive_bayes {

struct ClassDefinition {
  std::string name;
  double prior;
  std::vector<std::unique_ptr<FeatureDistribution>> feature_models;  // may contain nullptr for unused features
  std::vector<std::string> feature_names;  // global feature order shared across all classes
};

struct ClassGroup {
  std::string name;
  std::vector<std::string> class_names;
};

struct FeatureModelInfo {
  std::string name;
  std::string distribution_type;
  std::vector<double> params;
};

struct ClassModelInfo {
  std::string name;
  double prior;
  std::vector<FeatureModelInfo> features;
};

class NaiveBayes {
 public:
  explicit NaiveBayes(ProbabilitySpace mode);
  void AddClassDefinition(ClassDefinition model);
  void SetClassGroups(std::vector<ClassGroup> groups);
  bool HasClassGroups() const;

  std::vector<std::pair<std::string, double>> PredictPosteriors(
      const std::vector<double>& features) const;
  std::vector<std::pair<std::string, double>> PredictGroupedPosteriors(
      const std::vector<double>& features) const;
  std::vector<std::pair<std::string, double>> PredictGroupedPosteriors(
      const std::vector<double>& features,
      const std::vector<ClassGroup>& groups) const;

  std::pair<std::string, double> PredictClass(
      const std::vector<double>& features) const;

  std::size_t FeatureDim() const;
  const std::vector<std::string>& FeatureNames() const;
  std::vector<ClassModelInfo> ClassModels() const;
  double SampleFeatureForClass(std::size_t class_index,
                               const std::string& feature_name,
                               std::mt19937& rng) const;

 private:
  std::vector<double> ComputeLogJointProbabilities(const std::vector<double>& features,
                                                   std::size_t feature_dim) const;

  double ComputeLogNormalizer(const std::vector<double>& log_joint) const;

  std::vector<std::pair<std::string, double>> BuildPosteriorFromLog(
      const std::vector<double>& log_joint, double log_normalizer) const;

  std::vector<double> ComputeLinearScores(const std::vector<double>& features,
                                          std::size_t feature_dim) const;

  std::vector<std::pair<std::string, double>> NormalizeScores(const std::vector<double>& scores) const;
  std::vector<std::pair<std::string, double>> BuildGroupedPosteriors(
      const std::vector<std::pair<std::string, double>>& posteriors,
      const std::vector<ClassGroup>& groups) const;

  std::vector<ClassDefinition> classes_;
  std::vector<ClassGroup> class_groups_;
  ProbabilitySpace mode_;
};

}  // namespace naive_bayes
