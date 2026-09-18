#include "AudioStreamBuffer.h"
#include <SDL3/SDL.h>
#include <core/log/InxLog.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace infernux
{
AudioStreamBuffer::AudioStreamBuffer(std::unique_ptr<AudioStreamDecoder> decoder, bool mono, uint64_t firstFrame)
    : m_decoder(std::move(decoder)), m_mono(mono)
{
    const auto first = std::min(firstFrame, FrameCount() - 1);
    m_requested.store(first);
    // Initial read-ahead is prepared before binding the voice to the device.
    // Later IO never runs on the device or frame thread.
    Fill(first);
    m_wake = SDL_CreateSemaphore(0);
    if (!m_wake) throw std::runtime_error(SDL_GetError());
    try { m_worker = std::thread([this] { Run(); }); }
    catch (...) { SDL_DestroySemaphore(m_wake); throw; }
}
AudioStreamBuffer::~AudioStreamBuffer()
{
    m_requested.store(UINT64_MAX, std::memory_order_release);
    SDL_SignalSemaphore(m_wake);
    if (m_worker.joinable()) m_worker.join();
    SDL_DestroySemaphore(m_wake);
}
void AudioStreamBuffer::Request(uint64_t frame)
{
    const auto page = (frame % FrameCount()) / PageFrames * PageFrames;
    if (m_requested.exchange(page, std::memory_order_release) != page)
        SDL_SignalSemaphore(m_wake);
}
bool AudioStreamBuffer::ReadFrame(uint64_t frame, float &left, float &right)
{
    for (auto &page : m_pages) {
        int expected = Ready;
        if (!page.state.compare_exchange_strong(expected, Reading, std::memory_order_acquire)) continue;
        const bool found = frame >= page.start && frame - page.start < page.count;
        if (found) {
            const auto offset = static_cast<size_t>(frame - page.start) * 2;
            left = page.samples[offset]; right = page.samples[offset + 1];
        }
        page.state.store(Ready, std::memory_order_release);
        if (found) return true;
    }
    return false;
}
bool AudioStreamBuffer::Fill(uint64_t frame)
{
    const uint64_t totalPages = (FrameCount() + PageFrames - 1) / PageFrames;
    const auto first = frame / PageFrames;
    const auto count = std::min<uint64_t>(PageCount, totalPages);
    auto wanted = [&](uint64_t start) { return ((start / PageFrames + totalPages - first) % totalPages) < count; };
    for (uint64_t n = 0; n < count; ++n) {
        const uint64_t start = ((first + n) % totalPages) * PageFrames;
        bool exists = false;
        // This worker is the only writer of page metadata.
        for (auto &p : m_pages)
            if (p.state.load(std::memory_order_acquire) != Empty && p.start == start) exists = true;
        if (exists) continue;
        Page *target = nullptr;
        for (auto &p : m_pages) {
            int state = p.state.load(std::memory_order_acquire);
            if (state != Empty && (state != Ready || wanted(p.start))) continue;
            if (p.state.compare_exchange_strong(state, Writing, std::memory_order_acquire)) { target = &p; break; }
        }
        if (!target) return false; // An obsolete page is briefly held by the callback.
        try {
            m_decoder->Seek(start);
            const size_t frames = static_cast<size_t>(std::min<uint64_t>(PageFrames, FrameCount() - start));
            size_t read = 0;
            while (read < frames) {
                const auto got = m_decoder->Read(m_decode.data() + read * m_decoder->Channels(), frames - read);
                if (!got) throw std::runtime_error("Audio stream ended before its declared frame count");
                read += got;
            }
            SDL_AudioSpec from{SDL_AUDIO_F32, m_decoder->Channels(), SampleRate()};
            SDL_AudioSpec to{SDL_AUDIO_F32, 2, SampleRate()};
            if (m_mono && from.channels > 1) {
                for (size_t i = 0; i < frames; ++i) {
                    float sum = 0;
                    for (int c = 0; c < from.channels; ++c) sum += m_decode[i * from.channels + c];
                    m_decode[i] = sum / from.channels;
                }
                from.channels = 1;
            }
            Uint8 *converted = nullptr;
            int bytes = 0;
            if (!SDL_ConvertAudioSamples(&from, reinterpret_cast<const Uint8 *>(m_decode.data()),
                                        static_cast<int>(frames * from.channels * sizeof(float)), &to, &converted, &bytes))
                throw std::runtime_error(SDL_GetError());
            if (bytes != static_cast<int>(frames * 2 * sizeof(float))) {
                SDL_free(converted);
                throw std::runtime_error("Audio page conversion changed its frame count");
            }
            std::memcpy(target->samples.data(), converted, bytes);
            SDL_free(converted);
            target->start = start; target->count = frames;
            target->state.store(Ready, std::memory_order_release);
        } catch (...) {
            target->state.store(Empty, std::memory_order_release);
            throw;
        }
    }
    return true;
}
void AudioStreamBuffer::Run()
{
    try {
        for (;;) {
            const auto request = m_requested.load(std::memory_order_acquire);
            if (request == UINT64_MAX) return;
            if (Fill(request)) SDL_WaitSemaphore(m_wake);
            else std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch (const std::exception &e) {
        INXLOG_ERROR("Audio streaming stopped: ", e.what());
        m_failed.store(true, std::memory_order_release);
    }
}
}
