#include "TestMessageProcessor2.h"

#include <cmath>
#include <cstdint>

TestMessageProcessor2::TestMessageProcessor2(BayesPipeline::IFeaturePublisher& publisher)
    : publisher_(publisher)
{}

void TestMessageProcessor2::ProcessMessage(const Proc2Message& msg) {
    // TODO: add validation/filtering here
    constexpr double kNsPerSecond = 1000000000.0;
    const int64_t time_ns = static_cast<int64_t>(std::llround(msg.time * kNsPerSecond));
    const BayesPipeline::FeatureData data{msg.id, time_ns, "length", msg.length, msg.truth_label};
    publisher_.PublishFeature(data);
}
