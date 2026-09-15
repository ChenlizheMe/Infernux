#include <function/audio/AudioBusAutomation.h>
#include <function/audio/AudioMixer.h>

#include <cassert>
#include <cmath>
#include <thread>

namespace
{
bool Near(float lhs, float rhs)
{
    return std::abs(lhs - rhs) < 0.0001f;
}
} // namespace

int main()
{
    using namespace infernux::audio_mixer;
    BusEnvelope fade{1.0f, 0.0f, 10.0, 2.0, false};
    assert(Near(fade.VolumeAt(9.0), 1.0f));
    assert(Near(fade.VolumeAt(11.0), 0.5f));
    assert(Near(fade.VolumeAt(13.0), 0.0f));
    assert(fade.IsFading(11.0));
    assert(!fade.IsFading(12.0));
    BusEnvelopeMailbox mailbox;
    std::atomic<bool> done{false};
    std::thread producer([&] {
        BusEnvelopes value{};
        for (int sequence = 1; sequence <= 100000; ++sequence) {
            for (auto &bus : value) {
                bus.start = bus.target = static_cast<float>(sequence);
                bus.startedAt = bus.duration = sequence;
            }
            mailbox.Publish(value);
        }
        done.store(true, std::memory_order_release);
    });
    while (!done.load(std::memory_order_acquire)) {
        const auto &snapshot = mailbox.Consume();
        for (const auto &bus : snapshot) {
            assert(bus.start == snapshot[0].start);
            assert(bus.start == bus.target);
            assert(bus.startedAt == bus.duration);
        }
    }
    producer.join();
    assert(mailbox.Consume()[0].target == 100000.0f);

    using infernux::audio_mixer::ApproachParameter;
    using infernux::audio_mixer::DistanceAttenuation;
    using infernux::audio_mixer::MixStereoFrame;

    assert(Near(ApproachParameter(0.0f, 1.0f, 0.25f), 0.25f));
    assert(Near(ApproachParameter(1.0f, 0.0f, 0.25f), 0.75f));
    assert(Near(ApproachParameter(0.9f, 1.0f, 0.25f), 1.0f));
    assert(Near(DistanceAttenuation(0.0f, 1.0f, 11.0f), 1.0f));
    assert(Near(DistanceAttenuation(1.0f, 1.0f, 11.0f), 1.0f));
    assert(Near(DistanceAttenuation(6.0f, 1.0f, 11.0f), 0.5f));
    assert(Near(DistanceAttenuation(11.0f, 1.0f, 11.0f), 0.0f));
    assert(Near(DistanceAttenuation(2.0f, 1.0f, 1.0f), 0.0f));
    const auto twoDimensional = MixStereoFrame(1.0f, 0.25f, 0.8f, 0.0f, 1.0f, 0.0f);
    assert(Near(twoDimensional.left, 0.8f));
    assert(Near(twoDimensional.right, 0.2f));

    const auto centeredPoint = MixStereoFrame(1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f);
    assert(Near(centeredPoint.left, centeredPoint.right));
    assert(centeredPoint.left > 0.0f);

    const auto rightPoint = MixStereoFrame(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f);
    assert(Near(rightPoint.left, 0.0f));
    assert(Near(rightPoint.right, 1.0f));

    const auto distantPoint = MixStereoFrame(1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f);
    assert(Near(distantPoint.left, 0.0f));
    assert(Near(distantPoint.right, 0.0f));

    const auto blended = MixStereoFrame(1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.5f);
    assert(blended.left < 1.0f && blended.left > centeredPoint.left);
    assert(blended.right > 0.0f && blended.right < centeredPoint.right);
    return 0;
}
