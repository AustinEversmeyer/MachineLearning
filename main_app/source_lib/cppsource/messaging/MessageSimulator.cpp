#include "MessageSimulator.h"

#include "io/json.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

constexpr double kNsPerSecond = 1000000000.0;

double ToSeconds(int64_t time_ns) {
    return static_cast<double>(time_ns) / kNsPerSecond;
}

std::string Trim(const std::string& input) {
    std::size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
        ++start;
    }
    std::size_t end = input.size();
    while (end > start && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return input.substr(start, end - start);
}

std::vector<std::string> SplitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    std::istringstream stream(line);
    while (std::getline(stream, current, ',')) {
        fields.push_back(Trim(current));
    }
    return fields;
}

double TimeOf(const SimMessage& msg) {
    return std::visit([](const auto& m) { return m.time; }, msg);
}

std::optional<std::string> TruthForId(int id, const SyntheticParams& params) {
    static const std::vector<std::string> kDefaultTruthLabels = {
        "TargetSmall", "TargetMedium", "TargetLarge"
    };
    const std::vector<std::string>& labels =
        params.truth_labels.empty() ? kDefaultTruthLabels : params.truth_labels;
    if (labels.empty()) {
        return std::nullopt;
    }
    const std::size_t idx = static_cast<std::size_t>(id >= 0 ? id : -id) % labels.size();
    if (labels[idx].empty()) {
        return std::nullopt;
    }
    return labels[idx];
}

struct ClassProfile {
    std::string label;
    double rcs_mean = 0.0;
    double rcs_sigma = 1.0;
    double length_mean = 0.0;
    double length_sigma = 1.0;
};

std::vector<ClassProfile> LoadClassProfiles(const std::filesystem::path& model_config_path) {
    std::ifstream model_input(model_config_path);
    if (!model_input) {
        throw std::runtime_error("Failed to open model config for synthetic generation: " +
                                 model_config_path.string());
    }

    naive_bayes::io::Json root = naive_bayes::io::Json::parse(model_input);
    if (!root.is_object() || !root.contains("classes") || !root.at("classes").is_array()) {
        throw std::runtime_error("Model config must contain a 'classes' array for synthetic generation");
    }

    std::vector<ClassProfile> profiles;
    for (const naive_bayes::io::Json& class_json : root.at("classes").as_array()) {
        if (!class_json.is_object() || !class_json.contains("name") || !class_json.at("name").is_string() ||
            !class_json.contains("features") || !class_json.at("features").is_array()) {
            throw std::runtime_error("Invalid class entry in model config");
        }

        ClassProfile profile;
        profile.label = class_json.at("name").get<std::string>();

        std::map<std::string, std::pair<double, double>> feature_params;
        for (const naive_bayes::io::Json& feature_json : class_json.at("features").as_array()) {
            if (!feature_json.is_object() ||
                !feature_json.contains("name") || !feature_json.at("name").is_string() ||
                !feature_json.contains("type") || !feature_json.at("type").is_string() ||
                !feature_json.contains("params") || !feature_json.at("params").is_object()) {
                throw std::runtime_error("Invalid feature entry in model config");
            }

            const std::string feature_name = feature_json.at("name").get<std::string>();
            const std::string feature_type = feature_json.at("type").get<std::string>();
            if (feature_type != "gaussian") {
                continue;
            }
            const naive_bayes::io::Json& params = feature_json.at("params");
            if (!params.contains("mean") || !params.at("mean").is_number() ||
                !params.contains("sigma") || !params.at("sigma").is_number()) {
                throw std::runtime_error("Gaussian feature '" + feature_name +
                                         "' missing numeric mean/sigma in model config");
            }
            feature_params[feature_name] = {params.at("mean").get<double>(), params.at("sigma").get<double>()};
        }

        auto rcs_it = feature_params.find("rcs");
        auto len_it = feature_params.find("length");
        if (rcs_it == feature_params.end() || len_it == feature_params.end()) {
            throw std::runtime_error("Class '" + profile.label +
                                     "' missing gaussian 'rcs' or 'length' feature params");
        }
        profile.rcs_mean = rcs_it->second.first;
        profile.rcs_sigma = rcs_it->second.second;
        profile.length_mean = len_it->second.first;
        profile.length_sigma = len_it->second.second;
        profiles.push_back(std::move(profile));
    }

    if (profiles.empty()) {
        throw std::runtime_error("No usable class profiles found in model config: " +
                                 model_config_path.string());
    }
    return profiles;
}

}  // namespace

// ---------------------------------------------------------------------------
MessageSimulator::MessageSimulator(TestMessageProcessor1& proc1,
                                   TestMessageProcessor2& proc2,
                                   unsigned seed)
    : proc1_(proc1)
    , proc2_(proc2)
    , rng_(seed)
    , rcs_dist_(0.0, 1.0)   // re-parameterised per call in GenerateSynthetic
    , len_dist_(0.0, 1.0)
    , jitter_dist_(-1.0, 1.0)
{}

// ---------------------------------------------------------------------------
void MessageSimulator::Enqueue(SimMessage msg) {
    queue_.push_back(std::move(msg));
}

void MessageSimulator::EnqueueMany(const std::vector<SimMessage>& messages) {
    queue_.insert(queue_.end(), messages.begin(), messages.end());
}

void MessageSimulator::Clear() {
    queue_.clear();
}

void MessageSimulator::SortByTimestamp() {
    std::stable_sort(queue_.begin(), queue_.end(),
                     [](const SimMessage& lhs, const SimMessage& rhs) {
                         return TimeOf(lhs) < TimeOf(rhs);
                     });
}

// ---------------------------------------------------------------------------
// GenerateSynthetic
//
// Proc1 fires at proc1_time_step intervals for num_steps steps.
// Proc2 fires at proc2_time_step intervals for the same total duration.
// All messages are sorted by timestamp before being appended to the queue.
// ---------------------------------------------------------------------------
void MessageSimulator::GenerateSynthetic(int num_ids, int num_steps,
                                         double proc1_time_step,
                                         double proc2_time_step,
                                         SyntheticParams params) {
    std::optional<std::vector<ClassProfile>> class_profiles = std::nullopt;
    if (params.model_config_path.has_value()) {
        class_profiles = LoadClassProfiles(*params.model_config_path);
    }

    rcs_dist_    = std::normal_distribution<double>(params.rcs_mean,  params.rcs_stddev);
    len_dist_    = std::normal_distribution<double>(params.len_mean,  params.len_stddev);
    jitter_dist_ = std::uniform_real_distribution<double>(-params.time_jitter,
                                                           params.time_jitter);

    std::vector<SimMessage> new_messages;

    // Proc1: num_steps messages per ID at proc1_time_step intervals.
    for (int step = 0; step < num_steps; ++step) {
        const double t = step * proc1_time_step;
        for (int id = 0; id < num_ids; ++id) {
            if (class_profiles.has_value()) {
                const std::vector<ClassProfile>& profiles = *class_profiles;
                const ClassProfile& profile = profiles[static_cast<std::size_t>(id) % profiles.size()];
                std::normal_distribution<double> class_rcs_dist(profile.rcs_mean, profile.rcs_sigma);
                new_messages.push_back(Proc1Message{id, t, class_rcs_dist(rng_), profile.label});
            } else {
                new_messages.push_back(Proc1Message{id, t, rcs_dist_(rng_), TruthForId(id, params)});
            }
        }
    }

    // Proc2: independent schedule over the same duration.
    const double duration    = (num_steps - 1) * proc1_time_step;
    const int    proc2_steps = static_cast<int>(duration / proc2_time_step) + 1;
    for (int step = 0; step < proc2_steps; ++step) {
        const double t = step * proc2_time_step;
        for (int id = 0; id < num_ids; ++id) {
            if (class_profiles.has_value()) {
                const std::vector<ClassProfile>& profiles = *class_profiles;
                const ClassProfile& profile = profiles[static_cast<std::size_t>(id) % profiles.size()];
                std::normal_distribution<double> class_len_dist(profile.length_mean, profile.length_sigma);
                new_messages.push_back(
                    Proc2Message{id, t + jitter_dist_(rng_), class_len_dist(rng_), profile.label});
            } else {
                new_messages.push_back(
                    Proc2Message{id, t + jitter_dist_(rng_), len_dist_(rng_), TruthForId(id, params)});
            }
        }
    }

    std::stable_sort(new_messages.begin(), new_messages.end(),
                     [](const SimMessage& lhs, const SimMessage& rhs) {
                         return TimeOf(lhs) < TimeOf(rhs);
                     });

    for (SimMessage& msg : new_messages) {
        queue_.push_back(std::move(msg));
    }
}

void MessageSimulator::LoadScenarioFromCsv(const std::filesystem::path& csv_path,
                                           bool clear_first,
                                           bool sort_by_timestamp) {
    std::ifstream input(csv_path);
    if (!input) {
        throw std::runtime_error("Failed to open scenario CSV: " + csv_path.string());
    }

    if (clear_first) {
        Clear();
    }

    std::string header_line;
    bool found_header = false;
    while (std::getline(input, header_line)) {
        const std::string trimmed = Trim(header_line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        header_line = trimmed;
        found_header = true;
        break;
    }
    if (!found_header) {
        throw std::runtime_error("Scenario CSV is empty or missing header: " + csv_path.string());
    }
    const std::vector<std::string> header = SplitCsvLine(header_line);
    auto find_index = [&header](const std::string& name) -> std::size_t {
        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == name) {
                return i;
            }
        }
        throw std::runtime_error("Scenario CSV missing required column: " + name);
    };

    const std::size_t id_idx = find_index("id");
    const std::size_t source_idx = find_index("source");
    const std::size_t time_ns_idx = find_index("time_ns");
    const std::size_t value_idx = find_index("value");
    std::size_t truth_idx = header.size();
    for (std::size_t i = 0; i < header.size(); ++i) {
        if (header[i] == "truth_label") {
            truth_idx = i;
            break;
        }
    }

    std::string line;
    std::size_t line_no = 1;
    while (std::getline(input, line)) {
        ++line_no;
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        const std::vector<std::string> fields = SplitCsvLine(trimmed);
        if (fields.size() < header.size()) {
            throw std::runtime_error("Scenario CSV line " + std::to_string(line_no) +
                                     " has too few fields");
        }

        const int id = std::stoi(fields[id_idx]);
        const std::string source = fields[source_idx];
        const int64_t time_ns = std::stoll(fields[time_ns_idx]);
        const double value = std::stod(fields[value_idx]);
        std::optional<std::string> truth_label = std::nullopt;
        if (truth_idx < fields.size() && !fields[truth_idx].empty()) {
            truth_label = fields[truth_idx];
        }

        if (source == "rcs") {
            Enqueue(Proc1Message{id, ToSeconds(time_ns), value, truth_label});
        } else if (source == "length") {
            Enqueue(Proc2Message{id, ToSeconds(time_ns), value, truth_label});
        } else {
            throw std::runtime_error("Scenario CSV line " + std::to_string(line_no) +
                                     " has unknown source '" + source + "'");
        }
    }

    if (sort_by_timestamp) {
        SortByTimestamp();
    }
}

// ---------------------------------------------------------------------------
void MessageSimulator::Run() {
    for (const SimMessage& msg : queue_) {
        Dispatch(msg);
    }
}

void MessageSimulator::RunStepwise(
    const std::function<void(std::size_t, const SimMessage&)>& after_dispatch) {
    for (std::size_t i = 0; i < queue_.size(); ++i) {
        const SimMessage& msg = queue_[i];
        Dispatch(msg);
        after_dispatch(i, msg);
    }
}

void MessageSimulator::RunRealTime(std::chrono::milliseconds msg_interval) {
    for (const SimMessage& msg : queue_) {
        Dispatch(msg);
        std::this_thread::sleep_for(msg_interval);
    }
}

// ---------------------------------------------------------------------------
void MessageSimulator::Dispatch(const SimMessage& msg) {
    std::visit([this](const auto& m) {
        using T = std::decay_t<decltype(m)>;
        if constexpr (std::is_same_v<T, Proc1Message>) {
            proc1_.ProcessMessage(m);
        } else if constexpr (std::is_same_v<T, Proc2Message>) {
            proc2_.ProcessMessage(m);
        }
    }, msg);
}
