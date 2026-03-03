#include "main_app.h"

#include "messaging/MessageSimulator.h"

#include <filesystem>
#include <iostream>

MainApp::MainApp() {
    //   std::filesystem::path(GetConfigDir()) / "{path}/{file}.json"
    const std::filesystem::path runtime_config_path = "deps/BayesPipeline/config/runtime/runtime.config.json";
    const std::filesystem::path model_config_path = "deps/BayesPipeline/config/model/implementation.model.json";
    myBayesRuntimeManager = std::make_unique<BayesPipeline::BayesRuntimeManager>(runtime_config_path, model_config_path);

    // Processors map incoming messages to sink-owned structs and publish them.
    proc1_ = std::make_unique<TestMessageProcessor1>(*myBayesRuntimeManager);
    proc2_ = std::make_unique<TestMessageProcessor2>(*myBayesRuntimeManager);

    mySimulator = std::make_unique<MessageSimulator>(*proc1_, *proc2_);
}

MainApp::~MainApp() = default;

void MainApp::Run() {
    myBayesRuntimeManager->Start();
    const std::filesystem::path model_config_path = "deps/BayesPipeline/config/model/implementation.model.json";

    mySimulator->GenerateSynthetic(/*num_ids=*/3, /*num_steps=*/1000,
                                   /*proc1_time_step=*/1.0,
                                   /*proc2_time_step=*/5.0,
                                   SyntheticParams{
                                       .rcs_mean    = 5.0, .rcs_stddev  = 1.5,
                                       .len_mean    = 3.0, .len_stddev  = 0.8,
                                       .time_jitter = 0.2,
                                       .truth_labels = {},
                                       .model_config_path = model_config_path
                                   });

    mySimulator->Run();
    myBayesRuntimeManager->Stop();

    const auto& results = myBayesRuntimeManager->GetLatestResults();
    std::cout << "\n=== Classification Results ===\n";
    for (const auto& r : results) {
        std::cout << "  ID=" << r.id
                  << "  t_ns=" << r.time_ns
                  << "  state=" << r.classification_state
                  << "  class=" << r.predicted_class
                  << "  prob=" << r.predicted_prob
                  << (r.is_partial ? "  [partial]" : "") << "\n";
        for (const auto& [cls, prob] : r.posteriors) {
            std::cout << "      " << cls << ": " << prob << "\n";
        }
    }
    std::cout << "\nClassification output file: " << myBayesRuntimeManager->GetOutputFile() << "\n";
}
