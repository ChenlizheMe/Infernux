#pragma once

#include <cstdint>

namespace infernux::timing
{
// A normalized exponential mean of scaled simulation frame durations.
// Frame-clock history is independent of renderer/GPU timings and fixed steps.
class FrameTimeSmoother
{
  public:
    void Reset()
    {
        m_frames = 0;
        m_total = m_weight = 0.0;
        m_value = 0.0f;
    }

#if defined(__GNUC__) && !defined(__clang__)
    __attribute__((optimize("fp-contract=off")))
#endif
    float
    Advance(double delta)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        ++m_frames;
        if (m_frames == 1)
            return m_value;
        if (m_frames == 2)
            return m_value = static_cast<float>(delta);
        m_total = m_total * 0.8 + delta;
        m_weight = m_weight * 0.8 + 1.0;
        return m_value = static_cast<float>(m_total / m_weight);
    }

    float Value() const
    {
        return m_value;
    }

  private:
    uint64_t m_frames = 0;
    double m_total = 0.0;
    double m_weight = 0.0;
    float m_value = 0.0f;
};
} // namespace infernux::timing
