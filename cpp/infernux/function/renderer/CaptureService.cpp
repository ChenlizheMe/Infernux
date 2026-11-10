#include "CaptureService.h"

#include "vk/VkResourceManager.h"
#include <core/log/InxLog.h>
#include <core/threading/FrameWorkOwner.h>
#include <core/threading/JobSystem.h>
#include <core/types/ColorSpace.h>
#include <platform/filesystem/DocumentStore.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <zlib.h>

namespace infernux
{
namespace
{
constexpr size_t MaxInFlightCaptures = 4;
constexpr size_t MaxRetainedCaptures = 128;
constexpr size_t DeflateWorkspaceBytes = 288 * 1024; // zlib's 15-bit window and memory level 8.
constexpr size_t PngEnvelopeBytes = 57;              // Signature, IHDR, one IDAT envelope and IEND.
constexpr size_t PngTrailerBytes = 16;               // IDAT CRC and IEND.
constexpr size_t PngMaxChunkBytes = 0x7fffffff;
constexpr size_t CalibrationBytes = 64 * 1024;
constexpr size_t MaxEncodeSliceBytes = 2 * 1024 * 1024;
using Clock = std::chrono::steady_clock;

float HalfToFloat(uint16_t value)
{
    const uint32_t sign = (static_cast<uint32_t>(value) & 0x8000U) << 16U;
    const uint32_t exponent = (value >> 10U) & 0x1FU;
    const uint32_t mantissa = value & 0x3FFU;
    if (exponent == 0) {
        const float result = std::ldexp(static_cast<float>(mantissa), -24);
        return sign ? -result : result;
    }
    const uint32_t singleExponent = exponent == 31 ? 255 : exponent + 112;
    const uint32_t bits = sign | (singleExponent << 23U) | (mantissa << 13U);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

unsigned char DisplayFloatToUnorm8(float value)
{
    // Scene/Game float targets are sampled by ImGui into the UNORM swapchain
    // without an additional transfer function. Match that visible path exactly
    // so engine captures and the editor viewport have identical color values.
    return value > 0.0F ? static_cast<unsigned char>((std::min)(value, 1.0F) * 255.0F + 0.5F) : 0;
}

struct PngLayout
{
    size_t rowBytes = 0;
    size_t inputStride = 0;
    size_t scratchBytes = 0;
    size_t outputCapacity = 0;

    [[nodiscard]] size_t StorageBytes() const
    {
        return scratchBytes + outputCapacity;
    }
};

PngLayout GetPngLayout(const vk::ImageReadbackTicket &image)
{
    const auto &type = image.GetElementType();
    if (image.GetChannelCount() != 4 || (type != "uint8" && type != "float16" && type != "float32"))
        throw std::runtime_error("PNG capture requires an RGBA8, RGBA16F or RGBA32F image");
    const size_t rowBytes = size_t{image.GetWidth()} * 4;
    if (rowBytes + 1 > PngMaxChunkBytes / image.GetHeight())
        throw std::runtime_error("Capture exceeds the PNG encoder's image size limit");
    const auto bound = compressBound(static_cast<uLong>((rowBytes + 1) * image.GetHeight()));
    if (bound > PngMaxChunkBytes)
        throw std::runtime_error("Capture exceeds the PNG encoder's output size limit");
    const size_t inputStride = image.GetByteSize() / image.GetHeight();
    return {rowBytes, inputStride, DeflateWorkspaceBytes + rowBytes * 2 + 1 + inputStride,
            static_cast<size_t>(bound) + PngEnvelopeBytes};
}

void WritePngInteger(char *destination, uint32_t value)
{
    for (unsigned index = 0; index != 4; ++index)
        destination[index] = static_cast<char>(value >> (24 - 8 * index));
}

class PngEncoder final
{
  public:
    PngEncoder(std::shared_ptr<vk::ImageReadbackTicket> source, const PngLayout &layout, bool linear,
               std::shared_ptr<FrameCpuStorageClaim> storage)
        : m_source(std::move(source)), m_layout(layout), m_width(m_source->GetWidth()), m_height(m_source->GetHeight()),
          m_linear(linear), m_buffer{std::move(storage), {}, 0}
    {
    }

    ~PngEncoder()
    {
        if (m_stream.state)
            deflateEnd(&m_stream);
    }

    [[nodiscard]] size_t RemainingRows() const
    {
        return m_height - m_row;
    }
    [[nodiscard]] uint64_t StepNanoseconds() const
    {
        return m_stepNanoseconds;
    }
    void ResetStepTiming()
    {
        m_stepNanoseconds = 0;
    }
    [[nodiscard]] DocumentWriteBuffer TakeOutput()
    {
        m_buffer.storage->ReduceTo(m_layout.outputCapacity);
        return std::move(m_buffer);
    }

    void EncodeRows(size_t count)
    {
        struct Measure
        {
            uint64_t &elapsed;
            Clock::time_point start = Clock::now();
            ~Measure()
            {
                elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
            }
        } measure{m_stepNanoseconds};

        if (m_row == 0)
            Begin();
        const auto input = m_source->GetMappedData();
        auto *filtered = m_scratch.get() + DeflateWorkspaceBytes;
        auto *previous = filtered + m_layout.rowBytes + 1;
        auto *source = reinterpret_cast<char *>(previous + m_layout.rowBytes);
        const auto &type = m_source->GetElementType();
        for (const size_t end = m_row + count; m_row < end; ++m_row) {
            // The shared staging pool can map write-combined memory. Read one
            // contiguous row into CPU cache before converting individual samples.
            std::memcpy(source, input.data() + m_row * m_layout.inputStride, m_layout.inputStride);
            if (type == "uint8") {
                FilterRow<uint8_t>(source, filtered, previous, [](uint8_t value) { return value / 255.0f; });
            } else if (type == "float16") {
                FilterRow<uint16_t>(source, filtered, previous, HalfToFloat);
            } else {
                FilterRow<float>(source, filtered, previous, [](float value) { return value; });
            }
            Deflate(filtered, m_layout.rowBytes + 1, Z_NO_FLUSH);
        }
        if (m_row == m_height) {
            Deflate(nullptr, 0, Z_FINISH);
            WritePngInteger(m_buffer.data.get() + m_dataOffset - 8,
                            static_cast<uint32_t>(m_buffer.size - m_dataOffset));
            AppendInteger(static_cast<uint32_t>(m_crc));
            AppendChunk("IEND", {});
            deflateEnd(&m_stream);
            m_scratch.reset();
        }
    }

  private:
    static voidpf Allocate(voidpf opaque, uInt count, uInt size)
    {
        auto &encoder = *static_cast<PngEncoder *>(opaque);
        constexpr size_t alignment = alignof(std::max_align_t);
        const auto offset = (encoder.m_allocated + alignment - 1) & ~(alignment - 1);
        const size_t bytes = size_t{count} * size;
        if (offset > DeflateWorkspaceBytes || bytes > DeflateWorkspaceBytes - offset)
            return nullptr;
        encoder.m_allocated = offset + bytes;
        return encoder.m_scratch.get() + offset;
    }

    void Begin()
    {
        m_scratch = std::make_unique<uint8_t[]>(m_layout.scratchBytes);
        m_buffer.data.reset(new char[m_layout.outputCapacity]);
        m_stream.opaque = this;
        m_stream.zalloc = Allocate;
        m_stream.zfree = [](voidpf, voidpf) {};
        if (deflateInit(&m_stream, Z_BEST_SPEED) != Z_OK)
            throw std::runtime_error("Could not initialize PNG compression in the admitted workspace");

        Append("\x89PNG\r\n\x1a\n", 8);
        std::array<char, 13> header{};
        WritePngInteger(header.data(), m_width);
        WritePngInteger(header.data() + 4, m_height);
        header[8] = 8;
        header[9] = 6; // RGBA.
        AppendChunk("IHDR", {header.data(), header.size()});
        AppendInteger(0); // Filled once the deflate stream ends.
        Append("IDAT", 4);
        m_dataOffset = m_buffer.size;
        m_crc = crc32(0, reinterpret_cast<const Bytef *>("IDAT"), 4);
    }

    template <class Sample, class Decode>
    void FilterRow(const char *source, uint8_t *filtered, uint8_t *previous, Decode decode) const
    {
        filtered[0] = 2; // PNG Up filter.
        const bool bgra = m_source->IsBgra();
        for (size_t pixel = 0; pixel < m_layout.rowBytes; pixel += 4) {
            for (size_t channel = 0; channel < 4; ++channel) {
                const size_t inputChannel = bgra && (channel == 0 || channel == 2) ? 2 - channel : channel;
                Sample sample;
                std::memcpy(&sample, source + (pixel + inputChannel) * sizeof(Sample), sizeof(Sample));
                const float value = decode(sample);
                const auto encoded =
                    DisplayFloatToUnorm8(m_linear && channel < 3 ? inx::color::LinearToSrgb(value) : value);
                const size_t index = pixel + channel;
                filtered[index + 1] = static_cast<uint8_t>(encoded - previous[index]);
                previous[index] = encoded;
            }
        }
    }

    void Deflate(uint8_t *input, size_t size, int flush)
    {
        m_stream.next_in = input;
        m_stream.avail_in = static_cast<uInt>(size);
        auto *output = reinterpret_cast<Bytef *>(m_buffer.data.get() + m_buffer.size);
        const auto capacity = static_cast<uInt>(m_layout.outputCapacity - m_buffer.size - PngTrailerBytes);
        m_stream.next_out = output;
        m_stream.avail_out = capacity;
        const int status = deflate(&m_stream, flush);
        if (status != (flush == Z_FINISH ? Z_STREAM_END : Z_OK) || m_stream.avail_in != 0)
            throw std::runtime_error("PNG compression exceeded the admitted output range");
        const auto bytes = capacity - m_stream.avail_out;
        if (bytes) {
            m_crc = crc32(m_crc, output, bytes);
            m_buffer.size += bytes;
        }
    }

    void Append(const char *data, size_t count)
    {
        if (count)
            std::memcpy(m_buffer.data.get() + m_buffer.size, data, count);
        m_buffer.size += count;
    }

    void AppendInteger(uint32_t value)
    {
        WritePngInteger(m_buffer.data.get() + m_buffer.size, value);
        m_buffer.size += 4;
    }

    void AppendChunk(const char *type, std::string_view data)
    {
        AppendInteger(static_cast<uint32_t>(data.size()));
        const auto *start = reinterpret_cast<const Bytef *>(m_buffer.data.get() + m_buffer.size);
        Append(type, 4);
        Append(data.data(), data.size());
        AppendInteger(static_cast<uint32_t>(crc32(0, start, static_cast<uInt>(4 + data.size()))));
    }

    std::shared_ptr<vk::ImageReadbackTicket> m_source;
    PngLayout m_layout;
    uint32_t m_width, m_height;
    bool m_linear;
    DocumentWriteBuffer m_buffer;
    std::unique_ptr<uint8_t[]> m_scratch;
    z_stream m_stream{};
    size_t m_row = 0, m_allocated = 0, m_dataOffset = 0;
    uLong m_crc = 0;
    uint64_t m_stepNanoseconds = 0;
};

bool IsTerminal(CaptureStatus status)
{
    return status == CaptureStatus::Completed || status == CaptureStatus::Failed ||
           status == CaptureStatus::Cancelled || status == CaptureStatus::SourceExpired;
}
} // namespace

struct CaptureService::Impl
{
    struct Record
    {
        CaptureSnapshot snapshot;
        std::shared_ptr<vk::ImageReadbackTicket> readback;
        PngLayout layout;
        std::unique_ptr<PngEncoder> encoder;
        std::shared_ptr<DocumentWriteTicket> writeTicket;
        Clock::time_point phaseStartedAt;
        double bytesPerNanosecond = 0;
    };

    struct EncodeRange
    {
        size_t rows = 0;
        uint64_t prediction = 0;
    };

    struct ActiveEncode
    {
        JobHandle job;
        uint64_t id = 0;
        uint64_t frame = 0;
        uint64_t grant = 0;
        size_t rows = 0;
    };

    Impl(FrameWorkOwner &frameOwner, std::chrono::milliseconds timeout) : owner(frameOwner), gpuTimeout(timeout)
    {
    }

    [[nodiscard]] EncodeRange PlanRange(const Record &record) const
    {
        if (record.snapshot.status != CaptureStatus::PendingEncode)
            return {};
        const auto remaining = record.encoder ? record.encoder->RemainingRows() : record.snapshot.height;
        const auto stride = record.layout.inputStride;
        const auto rate = record.bytesPerNanosecond;
        size_t rows;
        uint64_t prediction = 0;
        if (rate > 0) {
            const auto maxRows = (std::min)(remaining, std::max<size_t>(1, MaxEncodeSliceBytes / stride));
            rows = static_cast<size_t>(std::clamp(owner.ResourceRemaining(FrameWorkResource::Cpu) * rate / stride, 1.0,
                                                  static_cast<double>(maxRows)));
            prediction = static_cast<uint64_t>(std::ceil(rows * stride / rate));
        } else {
            rows = (std::min)(remaining, std::max<size_t>(1, CalibrationBytes / stride));
        }
        return owner.CanAdmitResource(FrameWorkResource::Cpu, prediction, true) ? EncodeRange{rows, prediction}
                                                                                : EncodeRange{};
    }

    [[nodiscard]] uint64_t ReadyEncode() const
    {
        uint64_t oldest = 0;
        for (const auto &[id, record] : records) {
            if ((oldest == 0 || id < oldest) && PlanRange(record).rows &&
                (record.encoder || owner.CanReserveCpuStorage(record.layout.StorageBytes())))
                oldest = id;
        }
        return oldest;
    }

    bool StartEncode(uint64_t id)
    {
        if (active.job.IsValid())
            return false;
        auto &record = records.at(id);
        const auto range = PlanRange(record);
        if (!range.rows)
            return false;

        try {
            if (!record.encoder) {
                auto storage = owner.TryReserveCpuStorage(record.layout.StorageBytes());
                if (!storage)
                    return false;
                const bool linear = record.snapshot.source == CaptureSource::Camera &&
                                    !rhi::IsSrgbFormat(record.snapshot.view.colorFormat);
                record.encoder =
                    std::make_unique<PngEncoder>(std::move(record.readback), record.layout, linear, std::move(storage));
            }
            auto *encoder = record.encoder.get();
            encoder->ResetStepTiming();
            auto job = JobSystem::Get().Schedule([encoder, rows = range.rows] { encoder->EncodeRows(rows); },
                                                 JobDomain::Asset, JobPriority::Low, JobAffinity::WorkerOnly);
            const auto grant = owner.RecordAcceptedResource(FrameWorkResource::Cpu, range.prediction);
            active = {std::move(job), id, owner.Opportunity(), grant, range.rows};
        } catch (const std::exception &error) {
            record.encoder.reset();
            record.readback.reset();
            record.snapshot.status = CaptureStatus::Failed;
            record.snapshot.error = error.what();
        }
        return true;
    }

    void ObserveEncode()
    {
        if (!active.job.IsValid() || !active.job.IsComplete())
            return;
        auto &record = records.at(active.id);
        auto &encoder = *record.encoder;
        const auto elapsed = encoder.StepNanoseconds();
        owner.ObserveResource(FrameWorkResource::Cpu, elapsed, active.frame, active.grant);
        if (!IsTerminal(record.snapshot.status)) {
            try {
                JobSystem::Get().WaitPassive(active.job); // The completed handle carries the codec's failure.
                if (encoder.RemainingRows() == 0) {
                    record.writeTicket =
                        DocumentStore::Instance().SubmitBuffer(record.snapshot.outputPath, encoder.TakeOutput());
                    record.snapshot.status = CaptureStatus::PendingWrite;
                    record.encoder.reset();
                } else if (elapsed) {
                    record.bytesPerNanosecond = static_cast<double>(active.rows * record.layout.inputStride) / elapsed;
                }
            } catch (const std::exception &error) {
                record.snapshot.status = CaptureStatus::Failed;
                record.snapshot.error = error.what();
            }
        }
        if (IsTerminal(record.snapshot.status))
            record.encoder.reset();
        active = {};
        owner.ReportProgress();
    }

    FrameWorkOwner &owner;
    FrameWorkOwner::SourceId source = 0;
    std::chrono::milliseconds gpuTimeout;
    uint64_t nextId = 1;
    std::unordered_map<uint64_t, Record> records;
    ActiveEncode active;
};

const char *CaptureSourceName(CaptureSource source) noexcept
{
    switch (source) {
    case CaptureSource::Scene:
        return "scene";
    case CaptureSource::Game:
        return "game";
    case CaptureSource::Editor:
        return "editor";
    case CaptureSource::Camera:
        return "camera";
    }
    return "game";
}

const char *CaptureStatusName(CaptureStatus status) noexcept
{
    switch (status) {
    case CaptureStatus::PendingGpu:
        return "pending_gpu";
    case CaptureStatus::PendingEncode:
        return "pending_encode";
    case CaptureStatus::PendingWrite:
        return "pending_write";
    case CaptureStatus::Completed:
        return "completed";
    case CaptureStatus::Failed:
        return "failed";
    case CaptureStatus::Cancelled:
        return "cancelled";
    case CaptureStatus::SourceExpired:
        return "source_expired";
    }
    return "failed";
}

CaptureService::CaptureService(FrameWorkOwner &owner, std::chrono::milliseconds gpuTimeout)
    : m_impl(std::make_unique<Impl>(owner, gpuTimeout))
{
    if (gpuTimeout.count() < 0)
        throw std::invalid_argument("Capture GPU timeout cannot be negative");
    FrameWorkSource work;
    work.observe = [this] { Poll(); };
    work.refresh = work.observe;
    work.describe = [this] {
        FrameWorkFacts facts;
        facts.pending = HasPending();
        facts.observe = m_impl->active.job.IsValid();
        facts.workerBusy = facts.observe && !m_impl->active.job.IsComplete();
        if (!facts.observe) {
            if (const auto ready = m_impl->ReadyEncode())
                facts.host = {ready, 0, true, false};
        }
        return facts;
    };
    work.advance = [this](bool, uint64_t id) { return m_impl->StartEncode(id); };
    m_impl->source = owner.AddSource(std::move(work));
}

CaptureService::~CaptureService()
{
    m_impl->owner.RemoveSource(m_impl->source);
    if (m_impl->active.job.IsValid()) {
        auto &active = m_impl->active;
        active.job.Cancel();
        JobSystem::Get().WaitPassiveCompletion(active.job);
        m_impl->owner.ObserveResource(FrameWorkResource::Cpu, m_impl->records.at(active.id).encoder->StepNanoseconds(),
                                      active.frame, active.grant);
        active = {};
    }
    for (auto &[id, record] : m_impl->records) {
        if (IsTerminal(record.snapshot.status) || Cancel(id))
            continue;
        // Only an executing file write is no longer cancellable. Teardown
        // drains this service's accepted writes, never the global IO store.
        try {
            record.writeTicket->Wait();
        } catch (const std::exception &error) {
            INXLOG_ERROR("Capture file write failed during shutdown: ", error.what());
        }
    }
}

uint64_t CaptureService::Request(CaptureSource source, const rhi::RenderViewContext &view, uint64_t sourceGeneration,
                                 uint64_t engineFrame, std::string outputPath)
{
    if (outputPath.empty())
        throw std::invalid_argument("Capture output path cannot be empty");
    if (!view.IsValid() || view.kind != rhi::RenderViewKind::Capture ||
        view.output != rhi::RenderOutputKind::Readback || view.source == rhi::InvalidRenderViewId)
        throw std::invalid_argument("Capture requires a valid readback view derived from a source render view");

    size_t inFlight = 0;
    for (const auto &[id, record] : m_impl->records) {
        if (!IsTerminal(record.snapshot.status) || m_impl->active.id == id)
            ++inFlight;
    }
    if (inFlight >= MaxInFlightCaptures)
        throw std::runtime_error("Capture queue is full");

    if (m_impl->records.size() >= MaxRetainedCaptures) {
        for (auto it = m_impl->records.begin(); it != m_impl->records.end();) {
            if (IsTerminal(it->second.snapshot.status) && it->first != m_impl->active.id)
                it = m_impl->records.erase(it);
            else
                ++it;
            if (m_impl->records.size() < MaxRetainedCaptures)
                break;
        }
    }

    const uint64_t id = m_impl->nextId++;
    Impl::Record record;
    record.snapshot.id = id;
    record.snapshot.source = source;
    record.snapshot.status = CaptureStatus::PendingGpu;
    record.snapshot.sourceGeneration = sourceGeneration;
    record.snapshot.view = view;
    record.snapshot.engineFrame = engineFrame;
    record.snapshot.width = view.width;
    record.snapshot.height = view.height;
    record.snapshot.outputPath = std::move(outputPath);
    record.phaseStartedAt = std::chrono::steady_clock::now();
    m_impl->records.emplace(id, std::move(record));
    m_impl->owner.Wake(m_impl->source);
    return id;
}

bool CaptureService::AttachReadback(uint64_t captureId, std::shared_ptr<vk::ImageReadbackTicket> ticket,
                                    uint64_t engineFrame)
{
    if (!ticket)
        throw std::invalid_argument("Capture requires a valid GPU readback ticket");
    const auto it = m_impl->records.find(captureId);
    if (it == m_impl->records.end() || IsTerminal(it->second.snapshot.status))
        return false;

    auto &record = it->second;
    if (record.snapshot.status != CaptureStatus::PendingGpu || record.readback)
        throw std::logic_error("Capture already has an attached GPU readback ticket");
    record.snapshot.engineFrame = engineFrame;
    record.snapshot.width = ticket->GetWidth();
    record.snapshot.height = ticket->GetHeight();
    record.readback = std::move(ticket);
    m_impl->owner.Wake(m_impl->source);
    return true;
}

void CaptureService::Fail(uint64_t captureId, std::string error)
{
    const auto it = m_impl->records.find(captureId);
    if (it == m_impl->records.end() || !Cancel(captureId))
        return;
    auto &record = it->second;
    record.snapshot.status = CaptureStatus::Failed;
    record.snapshot.error = std::move(error);
}

CaptureSnapshot CaptureService::Query(uint64_t captureId) const
{
    const auto it = m_impl->records.find(captureId);
    if (it == m_impl->records.end())
        throw std::out_of_range("Capture id was not found");
    return it->second.snapshot;
}

bool CaptureService::Cancel(uint64_t captureId)
{
    const auto it = m_impl->records.find(captureId);
    if (it == m_impl->records.end())
        return false;
    auto &record = it->second;
    if (IsTerminal(record.snapshot.status))
        return false;
    if (record.snapshot.status == CaptureStatus::PendingWrite) {
        if (!DocumentStore::Instance().Cancel(record.writeTicket))
            return false;
        record.writeTicket.reset();
    }
    record.readback.reset(); // Several capture IDs can share one presentation readback.
    if (m_impl->active.id == captureId)
        m_impl->active.job.Cancel();
    else
        record.encoder.reset();
    record.snapshot.status = CaptureStatus::Cancelled;
    return true;
}

void CaptureService::InvalidateSource(CaptureSource source, uint64_t sourceGeneration)
{
    for (auto &[id, record] : m_impl->records) {
        (void)id;
        if (record.snapshot.source != source || IsTerminal(record.snapshot.status) ||
            record.snapshot.status == CaptureStatus::PendingWrite ||
            record.snapshot.sourceGeneration == sourceGeneration)
            continue;
        if (Cancel(id)) {
            record.snapshot.status = CaptureStatus::SourceExpired;
            record.snapshot.error = "Capture source was resized or recreated before completion";
        }
    }
}

void CaptureService::Poll()
{
    m_impl->ObserveEncode();
    const auto now = Clock::now();
    for (auto &[id, record] : m_impl->records) {
        if (record.snapshot.status == CaptureStatus::PendingGpu && record.readback && record.readback->IsDone()) {
            const auto status = record.readback->GetStatus();
            if (status == vk::ImageReadbackStatus::Completed) {
                try {
                    record.layout = GetPngLayout(*record.readback);
                    record.snapshot.status = CaptureStatus::PendingEncode;
                } catch (const std::exception &error) {
                    Fail(id, error.what());
                }
            } else if (status == vk::ImageReadbackStatus::Cancelled) {
                record.snapshot.status = CaptureStatus::Cancelled;
                record.readback.reset();
            } else {
                record.snapshot.status = CaptureStatus::Failed;
                record.snapshot.error = record.readback->GetError();
                record.readback.reset();
            }
        }

        if (record.snapshot.status == CaptureStatus::PendingGpu && now - record.phaseStartedAt >= m_impl->gpuTimeout) {
            Fail(id, record.readback ? "GPU capture readback timed out"
                                     : "Capture source frame was not submitted before timeout");
            continue;
        }

        if (record.snapshot.status == CaptureStatus::PendingEncode && !record.encoder &&
            record.layout.StorageBytes() > m_impl->owner.CpuStorageLimit()) {
            Fail(id, "PNG capture exceeds shared CPU storage capacity");
            continue;
        }

        if (record.snapshot.status == CaptureStatus::PendingWrite && record.writeTicket->IsComplete()) {
            const auto status = record.writeTicket->GetStatusName();
            record.snapshot.status = status == "succeeded"   ? CaptureStatus::Completed
                                     : status == "cancelled" ? CaptureStatus::Cancelled
                                                             : CaptureStatus::Failed;
            record.snapshot.error = record.writeTicket->GetError();
            record.writeTicket.reset();
        }
    }
}

bool CaptureService::HasPending() const noexcept
{
    return m_impl->active.job.IsValid() ||
           std::any_of(m_impl->records.begin(), m_impl->records.end(),
                       [](const auto &entry) { return !IsTerminal(entry.second.snapshot.status); });
}

} // namespace infernux
