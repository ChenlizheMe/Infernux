#pragma once

#include "AudioStreamDecoder.h"
#include <SDL3/SDL_mutex.h>
#include <array>
#include <atomic>
#include <thread>

namespace infernux
{
/// Bounded read-ahead pages. Only the worker performs IO/decoding. The device
/// callback claims immutable pages with a nonblocking CAS and never waits.
class AudioStreamBuffer
{
  public:
    static constexpr size_t PageFrames = 8192, PageCount = 4;
    AudioStreamBuffer(std::unique_ptr<AudioStreamDecoder> decoder, bool mono, uint64_t firstFrame);
    ~AudioStreamBuffer();
    // Request once per mixing block; interpolation reads must not move the
    // read-ahead cursor backwards when sampling across a page boundary.
    void Request(uint64_t frame);
    bool ReadFrame(uint64_t frame, float &left, float &right);
    bool Failed() const
    {
        return m_failed.load(std::memory_order_acquire);
    }
    uint64_t FrameCount() const
    {
        return m_decoder->FrameCount();
    }
    int SampleRate() const
    {
        return m_decoder->SampleRate();
    }

  private:
    enum State
    {
        Empty,
        Writing,
        Ready,
        Reading
    };
    struct Page
    {
        std::atomic<int> state{Empty};
        uint64_t start = 0;
        size_t count = 0;
        std::array<float, PageFrames * 2> samples{};
    };
    bool Fill(uint64_t frame);
    void Run();
    std::unique_ptr<AudioStreamDecoder> m_decoder;
    bool m_mono;
    std::array<Page, PageCount> m_pages;
    std::array<float, PageFrames * 8> m_decode{};
    std::atomic<uint64_t> m_requested{0};
    std::atomic<bool> m_failed{false};
    SDL_Semaphore *m_wake = nullptr;
    std::thread m_worker;
};
} // namespace infernux
