#pragma once

#include "pipeline/prediction_helpers.h"
#include "pipeline/pipeline_config.h"
#include "io/json.h"
#include "naive_bayes/naive_bayes.h"

namespace naive_bayes {

namespace pipeline {

[[nodiscard]] NaiveBayes LoadModel(InferenceConfig& config);

std::vector<BatchPredictionRow> RunInference(const NaiveBayes& clf,
                                             const std::vector<Observation>& observations);

void PrintSinglePrediction(const NaiveBayes& clf,
                           const SinglePredictionConfig& single_input);

// Map a JSON object with named feature keys into an ordered feature vector and run prediction.
SinglePrediction PredictResultFromJsonObject(const NaiveBayes& clf,
                                             const LayoutConfig& layout,
                                             const naive_bayes::io::Json& json_obj);

// Map a JSON object with named feature keys into an ordered feature vector and run prediction,
// keeping optional timestep/truth metadata if present.
BatchPredictionRow PredictFromJsonObject(const NaiveBayes& clf,
                                         const LayoutConfig& layout,
                                         const naive_bayes::io::Json& json_obj);

// Ensure layout.feature_fields align to the model's named features; reorders layout to match and throws on mismatch.
void AlignLayoutWithModel(LayoutConfig& layout,
                          const std::vector<std::string>& model_feature_names);

}  // namespace pipeline

}  // namespace naive_bayes
