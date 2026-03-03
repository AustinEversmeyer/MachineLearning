#pragma once

#include "TestMessageProcessor1.h"
#include "TestMessageProcessor2.h"

#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <random>
#include <variant>
#include <vector>

// ---------------------------------------------------------------------------
// MessageSimulator
//
// Replaces the network layer for simulation / testing purposes.
//
// Two ways to populate the queue:
//
//   1. Manual — call Enqueue() with hand-crafted messages.
//
//   2. Synthetic — call GenerateSynthetic(), which creates interleaved
//      proc1 + proc2 messages across `num_ids` targets over `num_steps`
//      time steps.  Feature values are drawn from Gaussian distributions
//      so the resulting data looks like real sensor noise.
//
// Run() dispatches every queued message to the correct processor (which in
// turn maps and enqueues sink payloads for the pipeline worker).
//
// RunRealTime() does the same but sleeps `msg_interval` between dispatches
// so you can watch classification results trickle in at human speed.
// ---------------------------------------------------------------------------

// A tagged union covering all message types the sim can inject.
using SimMessage = std::variant<Proc1Message, Proc2Message>;

// Parameters for synthetic data generation.
// If model_config_path is set, per-class gaussian params from model config are
// used for rcs/length and truth labels come from class names.
// Otherwise global rcs/length params below are used.
struct SyntheticParams {
    // Proc1 (rcs): drawn from Normal(rcs_mean, rcs_stddev)
    double rcs_mean   = 5.0;
    double rcs_stddev = 1.5;

    // Proc2 (length): drawn from Normal(len_mean, len_stddev)
    double len_mean   = 3.0;
    double len_stddev = 0.8;

    // Simulated jitter: proc2 timestamps are offset by a small random
    // amount in [-time_jitter, +time_jitter] seconds.
    double time_jitter = 0.2;

    // Optional truth labels assigned by target id index.
    // If empty, defaults are used: TargetSmall, TargetMedium, TargetLarge.
    std::vector<std::string> truth_labels;

    // Optional Bayes model config path.
    // When set, synthetic values are sampled from per-class feature
    // distributions parsed from model classes/features and truth labels are
    // assigned from class names in model order.
    std::optional<std::filesystem::path> model_config_path;
};

class MessageSimulator {
public:
    MessageSimulator(TestMessageProcessor1& proc1, TestMessageProcessor2& proc2,
                     unsigned seed = 42);

    // --- Manual population ------------------------------------------------

    void Enqueue(SimMessage msg);
    void EnqueueMany(const std::vector<SimMessage>& messages);
    void Clear();
    void SortByTimestamp();

    // --- Synthetic population ---------------------------------------------
    //
    // Generates messages for `num_ids` target IDs on independent schedules:
    //   - Proc1 fires every proc1_time_step for num_steps steps.
    //   - Proc2 fires every proc2_time_step for the same total duration.
    //
    // All messages are sorted by timestamp before being appended to the
    // queue, so dispatch order reflects realistic arrival order.
    //
    // Messages are appended to the existing queue (call Clear() first if
    // you want a fresh run).
    void GenerateSynthetic(int num_ids,
                           int num_steps,
                           double proc1_time_step = 1.0,
                           double proc2_time_step = 1.0,
                           SyntheticParams params = {});

    // --- Dispatch ---------------------------------------------------------

    // Fire every queued message through its processor, in order.
    void Run();
    void RunStepwise(const std::function<void(std::size_t, const SimMessage&)>& after_dispatch);

    // Fire messages with a real-time pause between each one.
    void RunRealTime(std::chrono::milliseconds msg_interval
                         = std::chrono::milliseconds(100));

    // --- Deterministic scenario loading -----------------------------------
    //
    // CSV format (header required):
    //   seq,id,source,time_ns,value,truth_label
    // source: "rcs" (proc1) or "length" (proc2)
    // truth_label: optional (empty means no truth label)
    void LoadScenarioFromCsv(const std::filesystem::path& csv_path,
                             bool clear_first = true,
                             bool sort_by_timestamp = false);

    // How many messages are in the queue.
    size_t Size() const { return queue_.size(); }

private:
    void Dispatch(const SimMessage& msg);

    TestMessageProcessor1& proc1_;
    TestMessageProcessor2& proc2_;

    std::vector<SimMessage> queue_;

    std::mt19937                      rng_;
    std::normal_distribution<double>  rcs_dist_;
    std::normal_distribution<double>  len_dist_;
    std::uniform_real_distribution<double> jitter_dist_;
};
