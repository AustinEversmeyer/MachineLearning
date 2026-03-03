#include "TestMessageProcessor1.h"

#include <cmath>
#include <cstdint>

TestMessageProcessor1::TestMessageProcessor1(BayesPipeline::IFeaturePublisher& publisher)
    : publisher_(publisher)
{}

void TestMessageProcessor1::ProcessMessage(const Proc1Message& msg) {
    // TODO: add any validation/parsing/filtering here
    // e.g. discard messages with rcs <= 0, check id range, etc.
    constexpr double kNsPerSecond = 1000000000.0;
    const int64_t time_ns = static_cast<int64_t>(std::llround(msg.time * kNsPerSecond));
    const BayesPipeline::FeatureData data{msg.id, time_ns, "rcs", msg.rcs, msg.truth_label};
    publisher_.PublishFeature(data);
}
