#include <function/audio/AudioBusAutomation.h>
#include <function/audio/AudioMixer.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>

namespace
{
bool Near(float lhs, float rhs)
{
    return std::abs(lhs - rhs) < 0.0001f;
}

// Test-only stateful processor. Product audio keeps BypassOutputDsp; this
// probe proves the reserved stage runs on SDL's mixed samples before meters.
struct OffsetProbe
{
    float previous = 0.0f;

    void Process(float *samples, size_t sampleCount, int channels, float amount) noexcept
    {
        assert(channels == 2);
        for (size_t index = 0; index < sampleCount; ++index)
            samples[index] += amount + previous;
        previous = amount;
    }

    void Reset() noexcept
    {
        previous = 0.0f;
    }
};
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

    RealtimeValueMailbox<OutputDspSettings> dspSettings;
    std::atomic<bool> dspDone{false};
    std::thread dspProducer([&] {
        for (int sequence = 0; sequence < 10000; ++sequence)
            dspSettings.Publish(sequence % 2 == 0 ? OutputDspSettings{true, 0.25f} : OutputDspSettings{false, 0.75f});
        dspDone.store(true, std::memory_order_release);
    });
    while (!dspDone.load(std::memory_order_acquire)) {
        const auto &settings = dspSettings.Consume();
        assert((!settings.enabled && settings.parameter == 0.0f) || (settings.enabled && settings.parameter == 0.25f) ||
               (!settings.enabled && settings.parameter == 0.75f));
    }
    dspProducer.join();
    assert(!dspSettings.Consume().enabled && dspSettings.Consume().parameter == 0.75f);

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

    OutputDspStage<OffsetProbe> outputDsp;
    uint32_t nanBits = 0x7fc01234U;
    float payloadNan = 0.0f;
    std::memcpy(&payloadNan, &nanBits, sizeof(payloadNan));
    std::array<float, 4> bypassSamples{-0.0f, payloadNan, -0.25f, 0.75f};
    const auto bypassOriginal = bypassSamples;
    outputDsp.Process(bypassSamples.data(), bypassSamples.size(), 2);
    assert(std::memcmp(bypassSamples.data(), bypassOriginal.data(), sizeof(bypassSamples)) == 0);

    // Source and bus/master gains are already applied in the voice stream.
    // The stage sees the resulting interleaved postmix F32 data exactly once.
    const auto gained = MixStereoFrame(0.5f, -0.25f, 0.5f, 1.0f, 0.0f, 0.0f);
    std::array<float, 2> mixed{gained.left, gained.right};
    assert(Near(mixed[0], 0.25f) && Near(mixed[1], -0.125f));
    outputDsp.Publish({true, 0.25f});
    outputDsp.Process(mixed.data(), mixed.size(), 2);
    assert(Near(mixed[0], 0.5f) && Near(mixed[1], 0.125f));
    const auto meter = MeasureOutput(mixed.data(), mixed.size());
    assert(Near(meter.peak, 0.5f) && meter.saturatedSamples == 0);

    // A second active block sees the probe history. Disabling clears it while
    // preserving every input bit, and a new device session clears it again.
    mixed = {0.25f, -0.125f};
    outputDsp.Process(mixed.data(), mixed.size(), 2);
    assert(Near(mixed[0], 0.75f) && Near(mixed[1], 0.375f));
    outputDsp.Publish({false, 0.75f});
    bypassSamples = bypassOriginal;
    outputDsp.Process(bypassSamples.data(), bypassSamples.size(), 2);
    assert(std::memcmp(bypassSamples.data(), bypassOriginal.data(), sizeof(bypassSamples)) == 0);
    outputDsp.Publish({true, 0.25f});
    mixed = {0.25f, -0.125f};
    outputDsp.Process(mixed.data(), mixed.size(), 2);
    assert(Near(mixed[0], 0.5f) && Near(mixed[1], 0.125f));
    outputDsp.ResetDeviceSession();
    bypassSamples = bypassOriginal;
    outputDsp.Process(bypassSamples.data(), bypassSamples.size(), 2);
    assert(std::memcmp(bypassSamples.data(), bypassOriginal.data(), sizeof(bypassSamples)) == 0);
    outputDsp.Publish({true, 0.25f});
    mixed = {0.25f, -0.125f};
    outputDsp.Process(mixed.data(), mixed.size(), 2);
    assert(Near(mixed[0], 0.5f) && Near(mixed[1], 0.125f));
    return 0;
}
