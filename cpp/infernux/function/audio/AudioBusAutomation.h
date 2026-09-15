#pragma once

#include <algorithm>
#include <array>
#include <atomic>

namespace infernux::audio_mixer
{

struct BusEnvelope
{
    float start = 1.0f;
    float target = 1.0f;
    double startedAt = 0.0;
    double duration = 0.0;
    bool muted = false;

    [[nodiscard]] float VolumeAt(double time) const
    {
        if (duration == 0.0)
            return target;
        const double progress = std::clamp((time - startedAt) / duration, 0.0, 1.0);
        return start + (target - start) * static_cast<float>(progress);
    }

    [[nodiscard]] bool IsFading(double time) const
    {
        return duration > 0.0 && time < startedAt + duration;
    }
};

using BusEnvelopes = std::array<BusEnvelope, 5>;

// One engine-owner producer, one device-callback consumer. The producer may
// replace pending settings, but never writes the snapshot being mixed. The
// callback does one exchange at most: no allocation, mutex or retry loop.
class BusEnvelopeMailbox
{
  public:
    void Publish(const BusEnvelopes &value)
    {
        m_slots[m_write] = value;
        m_write = m_middle.exchange(m_write | dirty, std::memory_order_acq_rel) & indexMask;
    }

    const BusEnvelopes &Consume()
    {
        if (m_middle.load(std::memory_order_acquire) & dirty)
            m_read = m_middle.exchange(m_read, std::memory_order_acq_rel) & indexMask;
        return m_slots[m_read];
    }

  private:
    static constexpr unsigned dirty = 4;
    static constexpr unsigned indexMask = 3;
    std::array<BusEnvelopes, 3> m_slots{};
    std::atomic<unsigned> m_middle{1};
    unsigned m_read = 0;
    unsigned m_write = 2;
};

} // namespace infernux::audio_mixer
