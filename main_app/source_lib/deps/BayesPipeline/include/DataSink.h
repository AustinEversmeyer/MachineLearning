#pragma once

#include <deque>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace BayesPipeline {

enum class ClassificationTrigger {
    kAllFeaturesUpdated,    // every feature source must emit once since last classification
    kPrimaryFeatureUpdated, // trigger when the anchor (first model) feature updates
    kAnyFeatureUpdated,     // trigger on any feature arrival
};

enum class EvaluationPolicy {
    kImmediateAnyArrival,
    kPrimaryOnly,
    kHybridDeadline,
};

enum class PartialPolicy {
    kDisallow,
    kAllowAfterDeadline,
    kAlwaysAllow,
};

struct FeatureEntry {
    int64_t time_ns;
    double value;
};

struct FeatureData {
    int id;
    int64_t time_ns;
    std::string feature_name;
    double value;
    std::optional<std::string> truth_label;
};

struct JoinedFeatureVector {
    int    id;
    int64_t anchor_time_ns;
    std::map<std::string, double> feature_values; // missing features stored as NaN when allow_partial
    std::optional<std::string> truth_label;
    bool is_partial = false; // true if any feature was missing / out of tolerance
};

class FeatureAlignmentStore {
public:
    static constexpr size_t kDefaultMaxRecords = 10;
    static constexpr int64_t kDefaultTimeToleranceNs = 1000000000; // 1 second

    FeatureAlignmentStore(std::vector<std::string> model_feature_order,
                          size_t max_records = kDefaultMaxRecords,
                          int64_t time_tolerance_ns = kDefaultTimeToleranceNs);

    void RecordFeatureSample(const FeatureData& data);

    std::vector<JoinedFeatureVector> BuildJoinedFeatureVectors(bool allow_partial = false) const;

    // Legacy trigger-based API retained for backward compatibility.
    // Runtime path uses EvaluationPolicy/PartialPolicy in BayesClassifierManager.
    bool ShouldClassify(ClassificationTrigger trigger) const;

    // Legacy companion API retained for backward compatibility.
    void ResetUpdatedFeatures();

    const std::string& PrimaryFeatureName() const;

private:
    std::vector<std::string> model_feature_order_;
    size_t max_records_;
    int64_t time_tolerance_ns_;

    std::map<int, std::map<std::string, std::deque<FeatureEntry>>> samples_by_id_and_feature_;
    std::map<int, std::string> truth_label_by_id_;
    std::set<std::string> features_updated_since_last_classification_;

    template <typename T>
    void Trim(std::deque<T>& buf) const {
        while (buf.size() > max_records_) {
            buf.pop_front();
        }
    }
};

} // namespace BayesPipeline
