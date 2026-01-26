#pragma once

#include <optional>

#include "pipeline/pipeline_config.h"
#include "pipeline/pipeline_helpers.h"

struct AntSensorReading {
  double temperature;
  double pressure;
  double humidity;
};

class AntBayesLive {
 public:
  AntBayesLive();

  void RunLoop(int iterations);

 private:
  static AntSensorReading SampleReading(int iteration);
  static void PrintPrediction(const naive_bayes::pipeline::SinglePrediction& result);

  naive_bayes::pipeline::InferenceConfig cfg_;
  std::optional<naive_bayes::NaiveBayes> clf_;
  std::filesystem::path output_path_;
};
