#include "BayesClassifierManager.h"

#include "io/model_loader.h"   // naive_bayes::io::LoadModelConfiguration()
#include "pipeline/prediction_helpers.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace BayesPipeline {

namespace {

bool ShouldEvaluate(EvaluationPolicy policy,
                    const FeatureAlignmentStore& store,
                    const std::string& last_event_feature) {
    switch (policy) {
        case EvaluationPolicy::kImmediateAnyArrival:
            return true;
        case EvaluationPolicy::kPrimaryOnly:
            return last_event_feature == store.PrimaryFeatureName();
        case EvaluationPolicy::kHybridDeadline:
            return true;
    }
    return false;
}

PartialPolicy LegacyToPartialPolicy(bool allow_partial) {
    return allow_partial ? PartialPolicy::kAllowAfterDeadline : PartialPolicy::kDisallow;
}

EvaluationPolicy LegacyToEvaluationPolicy(ClassificationTrigger trigger) {
    switch (trigger) {
        case ClassificationTrigger::kAllFeaturesUpdated:
            return EvaluationPolicy::kImmediateAnyArrival;
        case ClassificationTrigger::kPrimaryFeatureUpdated:
            return EvaluationPolicy::kPrimaryOnly;
        case ClassificationTrigger::kAnyFeatureUpdated:
            return EvaluationPolicy::kImmediateAnyArrival;
    }
    return EvaluationPolicy::kImmediateAnyArrival;
}

}  // namespace

BayesClassifierManager::BayesClassifierManager(std::filesystem::path model_config_path,
                                               size_t max_records,
                                               int64_t time_tolerance_ns,
                                               ClassificationTrigger trigger,
                                               bool allow_partial)
    : BayesClassifierManager(std::move(model_config_path),
                             max_records,
                             time_tolerance_ns,
                             LegacyToEvaluationPolicy(trigger),
                             LegacyToPartialPolicy(allow_partial),
                             kDefaultPartialGraceWindowNs) {
    std::cerr << "Warning: ClassificationTrigger-based constructor is deprecated; "
                 "prefer EvaluationPolicy/PartialPolicy constructor.\n";
}

BayesClassifierManager::BayesClassifierManager(std::filesystem::path model_config_path,
                                               size_t max_records,
                                               int64_t time_tolerance_ns,
                                               EvaluationPolicy evaluation_policy,
                                               PartialPolicy partial_policy,
                                               int64_t partial_grace_window_ns)
    : evaluation_policy_(evaluation_policy)
    , partial_policy_(partial_policy)
    , partial_grace_window_ns_(partial_grace_window_ns) {
    naive_bayes::NaiveBayes loaded = naive_bayes::io::LoadModelConfiguration(model_config_path);
    const std::vector<std::string> feature_names = loaded.FeatureNames();
    bayesClassifier_ = std::make_unique<naive_bayes::NaiveBayes>(std::move(loaded));

    alignment_store_ = std::make_unique<FeatureAlignmentStore>(feature_names, max_records, time_tolerance_ns);
}

void BayesClassifierManager::RecordFeatureSample(const FeatureData& data) {
    last_event_feature_ = data.feature_name;
    last_event_time_ns_ = data.time_ns;
    alignment_store_->RecordFeatureSample(data);
}

const std::vector<ClassificationResult>& BayesClassifierManager::GetLatestResults() const {
    return latestResults_;
}

bool BayesClassifierManager::ClassifyIfReady() {
    if (!ShouldEvaluate(evaluation_policy_, *alignment_store_, last_event_feature_)) {
        latestResults_.clear();
        return false;
    }
    latestResults_ = Classify();
    return !latestResults_.empty();
}

std::vector<ClassificationResult> BayesClassifierManager::Classify() {
    std::vector<ClassificationResult> results;
    const std::vector<std::string>& feature_names = bayesClassifier_->FeatureNames();

    const bool include_partial_candidates = partial_policy_ != PartialPolicy::kDisallow;
    const std::vector<JoinedFeatureVector> joined =
        alignment_store_->BuildJoinedFeatureVectors(include_partial_candidates);

    for (const JoinedFeatureVector& rec : joined) {
        const std::pair<int, int64_t> key = std::make_pair(rec.id, rec.anchor_time_ns);
        EmissionState& state = emission_state_by_key_[key];

        bool emit_row = false;
        std::string classification_state = "full";

        if (!rec.is_partial) {
            if (!state.emitted_full && !state.emitted_partial) {
                emit_row = true;
                classification_state = "full";
                state.emitted_full = true;
            }
        } else {
            if (!state.emitted_partial && !state.emitted_full) {
                bool allow_partial_now = false;
                switch (partial_policy_) {
                    case PartialPolicy::kDisallow:
                        allow_partial_now = false;
                        break;
                    case PartialPolicy::kAlwaysAllow:
                        allow_partial_now = true;
                        break;
                    case PartialPolicy::kAllowAfterDeadline:
                        allow_partial_now =
                            last_event_time_ns_ >= (rec.anchor_time_ns + partial_grace_window_ns_);
                        break;
                }
                if (allow_partial_now) {
                    emit_row = true;
                    classification_state = "partial";
                    state.emitted_partial = true;
                }
            }
        }

        if (!emit_row) {
            continue;
        }

        std::vector<double> features;
        features.reserve(feature_names.size());
        for (std::vector<std::string>::const_iterator name_it = feature_names.begin();
             name_it != feature_names.end();
             ++name_it) {
            const std::string& feature_name = *name_it;
            std::map<std::string, double>::const_iterator value_it =
                rec.feature_values.find(feature_name);
            if (value_it == rec.feature_values.end()) {
                features.push_back(std::numeric_limits<double>::quiet_NaN());
                continue;
            }
            features.push_back(value_it->second);
        }

        const naive_bayes::pipeline::DetailedPrediction prediction =
            naive_bayes::pipeline::RunDetailedPrediction(*bayesClassifier_, features);

        ClassificationResult result;
        result.id = rec.id;
        result.time_ns = rec.anchor_time_ns;
        result.truth_label = rec.truth_label;
        result.classification_state = classification_state;
        result.feature_inputs = naive_bayes::pipeline::BuildFeatureInputs(feature_names, features);
        result.predicted_class = prediction.predicted_class;
        result.predicted_prob = prediction.predicted_prob;
        result.posteriors = prediction.probabilities;
        result.is_partial = (classification_state == "partial");
        result.group_posteriors = prediction.group_probabilities;
        result.predicted_group = prediction.predicted_group;
        result.predicted_group_prob = prediction.predicted_group_prob;

        results.push_back(std::move(result));
    }

    return results;
}

} // namespace BayesPipeline
