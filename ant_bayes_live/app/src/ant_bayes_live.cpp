#include "ant_bayes_live.h"

#include <iostream>
#include <stdexcept>

#include "io/json.h"

AntBayesLive::AntBayesLive() {
#ifndef ANT_BAYES_MODEL_PATH
  throw std::runtime_error("ANT_BAYES_MODEL_PATH is not defined at build time.");
#endif
#ifndef ANT_BAYES_OUTPUT_PATH
  throw std::runtime_error("ANT_BAYES_OUTPUT_PATH is not defined at build time.");
#endif
  cfg_.model_config = std::filesystem::path(ANT_BAYES_MODEL_PATH);
  clf_ = naive_bayes::pipeline::LoadModel(cfg_);
  cfg_.layout.timestep_field = "timestep";
  cfg_.layout.truth_field = "truth";
  output_path_ = std::filesystem::path(ANT_BAYES_OUTPUT_PATH);
}

void AntBayesLive::RunLoop(int iterations) {
  std::vector<naive_bayes::pipeline::BatchPredictionRow> rows;
  rows.reserve(static_cast<std::size_t>(iterations));

  for (int i = 0; i < iterations; ++i) {
    AntSensorReading reading = SampleReading(i);
    naive_bayes::io::Json evt = naive_bayes::io::Json::object({
        {"timestep", static_cast<double>(i)},
        {"truth", "unknown"},
        {"temperature", reading.temperature},
        {"pressure", reading.pressure},
        {"humidity", reading.humidity},
    });

    naive_bayes::pipeline::BatchPredictionRow row =
        naive_bayes::pipeline::PredictFromJsonObject(*clf_, cfg_.layout, evt);
    rows.push_back(row);

    std::cout << "Inference iteration " << (i + 1) << "\n";
    PrintPrediction({row.predicted_class,
                     row.predicted_prob,
                     row.probabilities,
                     row.predicted_group,
                     row.predicted_group_prob,
                     row.group_probabilities});
  }

  naive_bayes::pipeline::WritePredictionsCsv(output_path_, rows, false);
  std::cout << "Wrote predictions to " << output_path_ << "\n";
}

AntSensorReading AntBayesLive::SampleReading(int iteration) {
  switch (iteration % 3) {
    case 0:
      return {0.6, 1.2, 55.0};
    case 1:
      return {2.3, 0.9, 72.0};
    default:
      return {-0.4, 0.7, 32.0};
  }
}

void AntBayesLive::PrintPrediction(const naive_bayes::pipeline::SinglePrediction& result) {
  std::cout << "predicted_class: " << result.predicted_class
            << " (p=" << result.predicted_prob << ")\n";
  for (const auto& entry : result.probabilities) {
    std::cout << "  prob_" << entry.first << ": " << entry.second << "\n";
  }
  if (!result.group_probabilities.empty()) {
    std::cout << "predicted_group: " << result.predicted_group
              << " (p=" << result.predicted_group_prob << ")\n";
    for (const auto& entry : result.group_probabilities) {
      std::cout << "  group_prob_" << entry.first << ": " << entry.second << "\n";
    }
  }
}
