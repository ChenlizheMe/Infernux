#include "NativeModule.h"

#if defined(INX_PLATFORM_WINDOWS) || defined(INX_PLATFORM_LINUX)
#include "IoStep.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#ifdef INX_PLATFORM_LINUX
#include <dlfcn.h>
#include <elf.h>
#include <limits>
#include <map>
#include <unistd.h>
#endif

namespace infernux
{
#ifdef INX_PLATFORM_LINUX
namespace
{
// A single loader name per open inode. glibc retains names on a shared handle,
// so the descriptor in that name must live until its final mapping retires.
class ElfImages
{
  public:
    void *Open(const std::shared_ptr<VerifiedFile> &file)
    {
        const auto path = NativeDescriptorPath(file->Descriptor());
        std::lock_guard lock(m_mutex);
        const auto [entry, inserted] = m_images.try_emplace(file->Identity());
        auto &image = entry->second;
        if (inserted) {
            image.module = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!image.module) {
                m_images.erase(entry);
                throw std::runtime_error(std::string("Cannot load native CPU program: ") + ::dlerror());
            }
            image.file = file;
        }
        ++image.references;
        return image.module;
    }

    void Close(const VerifiedFile &file)
    {
        std::lock_guard lock(m_mutex);
        const auto entry = m_images.find(file.Identity());
        if (--entry->second.references == 0) {
            ::dlclose(entry->second.module);
            m_images.erase(entry);
        }
    }

  private:
    struct Image
    {
        void *module = nullptr;
        size_t references = 0;
        std::shared_ptr<VerifiedFile> file;
    };
    std::mutex m_mutex;
    std::map<std::pair<uint64_t, uint64_t>, Image> m_images;
};

std::shared_ptr<ElfImages> ProcessElfImages()
{
    static const auto images = std::make_shared<ElfImages>();
    return images;
}
} // namespace
#endif

struct NativeModule::Impl
{
    enum class Phase
    {
#ifdef INX_PLATFORM_WINDOWS
        Open,
        Dos,
        PeOffset,
        PeSignature,
        OptionalMagic,
        ImageSize,
#else
        ElfHeader,
        ElfSegments,
#endif
        Inspected,
        Loaded,
        Published,
        Cancelled
    };

    Impl(std::shared_ptr<VerifiedFile> source, std::vector<std::string> names)
        : file(std::move(source)), symbols(std::move(names)), addresses(symbols.size())
    {
        if (!file)
            throw std::invalid_argument("native module requires a completed verified file");
        std::sort(symbols.begin(), symbols.end());
        if (std::adjacent_find(symbols.begin(), symbols.end()) != symbols.end())
            throw std::invalid_argument("native module symbols must be unique");
        for (const auto &name : symbols)
            if (name.empty() || name.find('\0') != std::string::npos)
                throw std::invalid_argument("native module symbol is empty or contains a null byte");
    }
    ~Impl()
    {
        Close();
    }

    void Close() noexcept
    {
        const auto loaded = module.exchange(nullptr, std::memory_order_acq_rel);
#ifdef INX_PLATFORM_WINDOWS
        if (loaded)
            ::FreeLibrary(loaded);
        if (guard != INVALID_HANDLE_VALUE)
            ::CloseHandle(std::exchange(guard, INVALID_HANDLE_VALUE));
#else
        if (loaded)
            elfImages->Close(*file);
#endif
        file.reset();
    }
    void Next(Phase value)
    {
        position = 0;
        phase.store(value, std::memory_order_release);
    }
#ifdef INX_PLATFORM_WINDOWS
    static constexpr std::array<std::pair<std::string_view, uint64_t>, 6> inspection = {{{"module.open", 0},
                                                                                         {"module.dos", 2},
                                                                                         {"module.pe_offset", 4},
                                                                                         {"module.pe_signature", 6},
                                                                                         {"module.optional_magic", 2},
                                                                                         {"module.image_size", 4}}};
#else
    std::array<std::pair<std::string_view, uint64_t>, 2> inspection = {
        {{"module.elf_header", sizeof(Elf64_Ehdr)}, {"module.elf_segments", 0}}};
#endif
    IoWork DescribeWork(size_t maximum) const
    {
        if (!maximum)
            throw std::invalid_argument("native module inspection requires a positive byte budget");
        const auto current = phase.load(std::memory_order_acquire);
        if (static_cast<size_t>(current) < inspection.size()) {
            const auto [operation, bytes] = inspection[static_cast<size_t>(current)];
            return bytes ? IoByteWork(operation, bytes - position, maximum) : IoWork{operation, 0, {}, 0};
        }
        switch (current) {
        case Phase::Inspected:
            return {"module.map", 0, {}, symbols.size()};
        case Phase::Loaded:
        case Phase::Published:
        case Phase::Cancelled:
            return {};
        default:
            break;
        }
        throw std::logic_error("unknown native module phase");
    }
    IoRecipe DescribeRemainingWork() const
    {
        IoRecipe result;
        const auto current = phase.load(std::memory_order_acquire);
        if (static_cast<size_t>(current) < inspection.size()) {
            for (size_t i = static_cast<size_t>(current); i < inspection.size(); ++i) {
                const auto [operation, bytes] = inspection[i];
#ifdef INX_PLATFORM_LINUX
                if (!bytes)
                    break; // The ELF header supplies the segment table extent.
#endif
                result.push_back({{{bytes ? "bytes" : "atom",
                                    operation,
                                    {},
                                    0,
                                    bytes - (i == static_cast<size_t>(current) ? position : 0),
                                    {},
                                    {},
                                    {}}},
                                  1});
            }
            result.push_back({{{"barrier", "module.image", {}, 0, {}, {}, {}, {}}}, 1});
        } else if (current == Phase::Inspected) {
            // The actual inspected virtual image is a cost extent, not disk
            // traffic. Mapping and all requested symbols remain one OS atom.
            result.push_back({{{"atom", "module.map", {}, symbols.size(), imageBytes, {}, {}, {}}}, 1});
        }
        return result;
    }
    bool Read(uint64_t offset, size_t bytes, size_t maximum)
    {
        const auto count = std::min(maximum, bytes - position);
        file->ReadAt(offset + position, scratch.data() + position, count);
        position += count;
        readBytes += count;
        return position == bytes;
    }
    void Inspect(size_t maximum)
    {
        switch (phase.load(std::memory_order_acquire)) {
#ifdef INX_PLATFORM_WINDOWS
        case Phase::Open:
            // Keep the verified file object at this path while the OS loader
            // opens it. Other engine cache readers remain allowed.
            guard = ::CreateFileW(NativeFilePath(file->Path()).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
            if (guard == INVALID_HANDLE_VALUE)
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                        "Cannot retain native code file");
            if (FileIdentity(guard) != file->Identity())
                throw std::runtime_error("native code path differs from its verified file object");
            Next(Phase::Dos);
            return;
        case Phase::Dos:
            if (Read(0, 2, maximum)) {
                if (scratch[0] != 'M' || scratch[1] != 'Z')
                    throw std::runtime_error("native CPU program requires a PE image");
                Next(Phase::PeOffset);
            }
            return;
        case Phase::PeOffset:
            if (Read(0x3c, 4, maximum)) {
                std::memcpy(&peOffset, scratch.data(), sizeof(peOffset));
                Next(Phase::PeSignature);
            }
            return;
        case Phase::PeSignature:
            if (Read(peOffset, 6, maximum)) {
                if (std::memcmp(scratch.data(), "PE\0\0\x64\x86", 6) != 0)
                    throw std::runtime_error("native CPU program requires AMD64 PE");
                Next(Phase::OptionalMagic);
            }
            return;
        case Phase::OptionalMagic:
            if (Read(uint64_t{peOffset} + 24, 2, maximum)) {
                if (scratch[0] != 0x0b || scratch[1] != 0x02)
                    throw std::runtime_error("native CPU program requires PE32+");
                Next(Phase::ImageSize);
            }
            return;
        case Phase::ImageSize:
            if (Read(uint64_t{peOffset} + 24 + 56, 4, maximum)) {
                uint32_t peImageSize;
                std::memcpy(&peImageSize, scratch.data(), sizeof(peImageSize));
                imageBytes = peImageSize;
                if (!imageBytes)
                    throw std::runtime_error("native CPU program image size is zero");
                Next(Phase::Inspected);
            }
            return;
#else
        case Phase::ElfHeader:
            if (Read(0, sizeof(Elf64_Ehdr), maximum)) {
                Elf64_Ehdr header;
                std::memcpy(&header, scratch.data(), sizeof(header));
                if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) || header.e_ident[EI_CLASS] != ELFCLASS64 ||
                    header.e_ident[EI_DATA] != ELFDATA2LSB || header.e_machine != EM_X86_64 ||
                    header.e_type != ET_DYN || header.e_phentsize != sizeof(Elf64_Phdr) || !header.e_phnum)
                    throw std::runtime_error("native CPU program requires an x86-64 ELF shared image");
                segmentOffset = header.e_phoff;
                auto &bytes = inspection[static_cast<size_t>(Phase::ElfSegments)].second;
                bytes = uint64_t{header.e_phnum} * sizeof(Elf64_Phdr);
                if (segmentOffset > file->Bytes() || bytes > file->Bytes() - segmentOffset)
                    throw std::runtime_error("native CPU program segment table exceeds the image file");
                Next(Phase::ElfSegments);
            }
            return;
        case Phase::ElfSegments:
            InspectElfSegments(maximum);
            return;
#endif
        default:
            throw std::logic_error("native code inspection is already complete");
        }
    }
#ifdef INX_PLATFORM_LINUX
    void InspectElfSegments(size_t maximum)
    {
        const auto bytes = inspection[static_cast<size_t>(Phase::ElfSegments)].second;
        const auto end = position + std::min(maximum, static_cast<size_t>(bytes - position));
        while (position < end) {
            const auto start = position % scratch.size();
            const auto count = std::min(end - position, scratch.size() - start);
            file->ReadAt(segmentOffset + position, scratch.data() + start, count);
            for (size_t i = start / sizeof(Elf64_Phdr); i < (start + count) / sizeof(Elf64_Phdr); ++i) {
                Elf64_Phdr segment;
                std::memcpy(&segment, scratch.data() + i * sizeof(segment), sizeof(segment));
                if (segment.p_type != PT_LOAD || !segment.p_memsz)
                    continue;
                if (segment.p_memsz > std::numeric_limits<uint64_t>::max() - segment.p_vaddr)
                    throw std::runtime_error("native CPU program segment exceeds its address range");
                imageStart = std::min(imageStart, segment.p_vaddr);
                imageEnd = std::max(imageEnd, segment.p_vaddr + segment.p_memsz);
            }
            position += count;
            readBytes += count;
        }
        if (position == bytes) {
            const auto pageMask = static_cast<uint64_t>(::getpagesize()) - 1;
            if (imageStart >= imageEnd || imageEnd > std::numeric_limits<uint64_t>::max() - pageMask)
                throw std::runtime_error("native CPU program has no valid load image");
            imageBytes = ((imageEnd + pageMask) & ~pageMask) - (imageStart & ~pageMask);
            Next(Phase::Inspected);
        }
    }
#endif
    void Map()
    {
        if (phase.load(std::memory_order_acquire) != Phase::Inspected)
            throw std::logic_error("inspect and admit the native image before loading it");
        const auto started = std::chrono::steady_clock::now();
        struct Timing
        {
            uint64_t &nanoseconds;
            std::chrono::steady_clock::time_point started;
            ~Timing()
            {
                nanoseconds = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
                        .count());
            }
        } timing{loadNs, started};
#ifdef INX_PLATFORM_WINDOWS
        const auto loaded = ::LoadLibraryW(NativeFilePath(file->Path()).c_str());
        if (!loaded)
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "Cannot load native CPU program");
#else
        elfImages = ProcessElfImages();
        const auto loaded = elfImages->Open(file);
#endif
        module.store(loaded, std::memory_order_release);
        for (size_t index = 0; index < symbols.size(); ++index) {
#ifdef INX_PLATFORM_WINDOWS
            const auto address = ::GetProcAddress(loaded, symbols[index].c_str());
            if (!address)
                throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                        "Native CPU program entry is missing");
#else
            const auto address = ::dlsym(loaded, symbols[index].c_str());
            if (!address)
                throw std::runtime_error(std::string("Native CPU program entry is missing: ") + symbols[index]);
#endif
            addresses[index] = reinterpret_cast<uintptr_t>(address);
        }
        Next(Phase::Loaded);
    }
    size_t RetainedBytes() const
    {
        size_t bytes = sizeof(*this) + symbols.capacity() * sizeof(std::string) +
                       addresses.capacity() * sizeof(uintptr_t) + (file ? file->RetainedBytes() : 0);
        for (const auto &name : symbols)
            bytes += name.capacity() + 1;
        return bytes;
    }

    std::shared_ptr<VerifiedFile> file;
    std::vector<std::string> symbols;
    std::vector<uintptr_t> addresses;
#ifdef INX_PLATFORM_WINDOWS
    std::array<uint8_t, 8> scratch{};
    HANDLE guard = INVALID_HANDLE_VALUE;
    std::atomic<HMODULE> module{nullptr};
    std::atomic<Phase> phase{Phase::Open};
    uint32_t peOffset = 0;
#else
    std::array<uint8_t, sizeof(Elf64_Phdr) * 16> scratch{};
    std::shared_ptr<ElfImages> elfImages;
    std::atomic<void *> module{nullptr};
    std::atomic<Phase> phase{Phase::ElfHeader};
    uint64_t segmentOffset = 0, imageStart = std::numeric_limits<uint64_t>::max(), imageEnd = 0;
#endif
    size_t position = 0;
    size_t imageBytes = 0;
    uint64_t readBytes = 0, loadNs = 0;
};

NativeModule::NativeModule(std::shared_ptr<VerifiedFile> file, std::vector<std::string> symbols)
    : m_impl(std::make_unique<Impl>(std::move(file), std::move(symbols)))
{
}
NativeModule::~NativeModule() = default;
std::shared_ptr<NativeModule> NativeModule::Load(std::shared_ptr<VerifiedFile> file, std::vector<std::string> symbols)
{
    auto result = std::shared_ptr<NativeModule>(new NativeModule(std::move(file), std::move(symbols)));
    while (!result->IsInspected())
        result->Inspect(result->m_impl->scratch.size());
    result->Map();
    return result;
}
void NativeModule::Inspect(size_t maximum)
{
    m_impl->Inspect(maximum);
}
bool NativeModule::IsInspected() const
{
    return m_impl->phase.load(std::memory_order_acquire) == Impl::Phase::Inspected;
}
bool NativeModule::IsLoaded() const
{
    const auto phase = m_impl->phase.load(std::memory_order_acquire);
    return phase == Impl::Phase::Loaded || phase == Impl::Phase::Published;
}
void NativeModule::Map()
{
    m_impl->Map();
}
void NativeModule::Publish()
{
    auto expected = Impl::Phase::Loaded;
    if (!m_impl->phase.compare_exchange_strong(expected, Impl::Phase::Published, std::memory_order_acq_rel))
        throw std::logic_error("only a complete private native module can be published");
}
void NativeModule::Discard()
{
    auto phase = m_impl->phase.load(std::memory_order_acquire);
    while (phase != Impl::Phase::Published && phase != Impl::Phase::Cancelled) {
        if (m_impl->phase.compare_exchange_weak(phase, Impl::Phase::Cancelled, std::memory_order_acq_rel)) {
            m_impl->Close();
            return;
        }
    }
}
bool NativeModule::IsPublished() const
{
    return m_impl->phase.load(std::memory_order_acquire) == Impl::Phase::Published;
}
void NativeModule::RequirePublished() const
{
    if (!IsPublished())
        throw std::logic_error("native CPU execution requires a published module");
}
uintptr_t NativeModule::Identity() const
{
    return reinterpret_cast<uintptr_t>(m_impl->module.load(std::memory_order_acquire));
}
size_t NativeModule::ImageBytes() const
{
    return m_impl->imageBytes;
}
uintptr_t NativeModule::Address(const std::string &symbol) const
{
    if (!IsLoaded())
        throw std::logic_error("native module entries are not loaded");
    const auto entry = std::lower_bound(m_impl->symbols.begin(), m_impl->symbols.end(), symbol);
    if (entry == m_impl->symbols.end() || *entry != symbol)
        throw std::invalid_argument("native module entry was not declared");
    return m_impl->addresses[static_cast<size_t>(entry - m_impl->symbols.begin())];
}
size_t NativeModule::RetainedBytes() const
{
    return sizeof(*this) + m_impl->RetainedBytes();
}
uint64_t NativeModule::ReadBytes() const
{
    return m_impl->readBytes;
}
uint64_t NativeModule::LoadNs() const
{
    return m_impl->loadNs;
}
IoWork NativeModule::DescribeWork(size_t maximum) const
{
    return m_impl->DescribeWork(maximum);
}
IoRecipe NativeModule::DescribeRemainingWork() const
{
    return m_impl->DescribeRemainingWork();
}

struct NativeModuleTask::State : detail::IoStepState
{
    std::shared_ptr<NativeModule> module;
    bool published = false;
    std::atomic<size_t> imageBytes{0};
    std::atomic<uintptr_t> identity{0};
    std::atomic<uint64_t> readBytes{0}, loadNs{0};

    void Snapshot()
    {
        const auto bytes = sizeof(*this) + (module ? module->RetainedBytes() : 0);
        retained.store(bytes, std::memory_order_relaxed);
        reservation.store(bytes, std::memory_order_relaxed);
        imageBytes.store(module ? module->ImageBytes() : 0, std::memory_order_relaxed);
        identity.store(module ? module->Identity() : 0, std::memory_order_relaxed);
        if (module) {
            readBytes.store(module->ReadBytes(), std::memory_order_relaxed);
            loadNs.store(module->LoadNs(), std::memory_order_relaxed);
        }
    }
    void RetireCancelled()
    {
        if (module && !published) {
            readBytes.store(module->ReadBytes(), std::memory_order_relaxed);
            loadNs.store(module->LoadNs(), std::memory_order_relaxed);
            module->Discard();
            module.reset();
        }
    }
    void Fail(const std::exception &failure)
    {
        error = failure.what();
        RetireCancelled();
    }
    void Advance(size_t maximum)
    {
        if (control.load(std::memory_order_acquire) & cancel)
            RetireCancelled();
        else if (module->IsInspected())
            module->Map();
        else
            module->Inspect(maximum);
    }
};

NativeModuleTask::NativeModuleTask(std::shared_ptr<VerifiedFile> file, std::vector<std::string> symbols)
    : m_state(std::make_shared<State>())
{
    if (JobSystem::Get().IsInline())
        throw std::runtime_error("native module IO requires the existing worker pool");
    m_state->module = std::shared_ptr<NativeModule>(new NativeModule(std::move(file), std::move(symbols)));
    m_state->Snapshot();
}
NativeModuleTask::~NativeModuleTask()
{
    try {
        if (JobSystem::IsAvailable()) {
            Cancel();
            return;
        }
    } catch (...) {
    } // The pool can retire between the availability check and submission.
    if (!m_state->published) {
        const auto previous = m_state->control.fetch_or(State::cancel, std::memory_order_acq_rel);
        if (!(previous & State::active))
            m_state->RetireCancelled();
    } // An active callback observes cancellation; an idle module uses cold RAII.
}
NativeModuleTask::State &NativeModuleTask::Idle() const
{
    if (!m_job.IsComplete())
        throw std::logic_error("native module IO step is still running");
    if (!m_state->error.empty())
        throw std::runtime_error(m_state->error);
    return *m_state;
}
void NativeModuleTask::Dispatch(size_t maximum)
{
    const auto work = (m_state->control.load(std::memory_order_acquire) & State::cancel)
                          ? IoWork{"module.cleanup", 0, {}, 0}
                          : m_state->module->DescribeWork(maximum);
    m_job = detail::DispatchIo(m_state, [maximum](State &state) { state.Advance(maximum); });
    m_submittedWork = work;
}
void NativeModuleTask::Advance(size_t maximum)
{
    if (!maximum)
        throw std::invalid_argument("native module inspection requires a positive byte budget");
    std::lock_guard lock(m_mutex);
    auto &state = Idle();
    if (state.published || (state.control.load(std::memory_order_acquire) & State::cancel))
        return;
    if (state.module->IsLoaded())
        throw std::logic_error("publish or cancel the complete native module");
    Dispatch(maximum);
}
std::shared_ptr<NativeModule> NativeModuleTask::GetModule() const
{
    std::lock_guard lock(m_mutex);
    auto &state = Idle();
    if (!state.module || !state.module->IsLoaded())
        throw std::logic_error("native module is not loaded");
    return state.module;
}
void NativeModuleTask::Commit()
{
    std::lock_guard lock(m_mutex);
    auto &state = Idle();
    if (state.control.load(std::memory_order_acquire) & State::cancel)
        throw std::logic_error("native module request was cancelled");
    if (state.published)
        return;
    state.module->Publish();
    state.published = true;
    state.Snapshot();
}
bool NativeModuleTask::Cancel()
{
    std::lock_guard lock(m_mutex);
    if (m_state->published || (m_job.IsComplete() && !m_state->error.empty()))
        return false;
    const auto previous = m_state->control.fetch_or(State::cancel, std::memory_order_acq_rel);
    if (previous & State::cancel)
        return false;
    if (!(previous & State::active) && m_state->module)
        Dispatch(1);
    return true;
}
void NativeModuleTask::Wait()
{
    while (true) {
        JobHandle job;
        {
            std::lock_guard lock(m_mutex);
            job = m_job;
        }
        if (!job.IsComplete())
            JobSystem::Get().WaitPassiveCompletion(job);
        std::lock_guard lock(m_mutex);
        if (m_job.IsComplete())
            return;
    }
}
bool NativeModuleTask::IsPending() const
{
    std::lock_guard lock(m_mutex);
    return !m_job.IsComplete();
}
bool NativeModuleTask::IsComplete() const
{
    std::lock_guard lock(m_mutex);
    return m_job.IsComplete() && (m_state->published || !m_state->error.empty() ||
                                  (m_state->control.load(std::memory_order_acquire) & State::cancel));
}
std::string NativeModuleTask::GetStateName() const
{
    std::lock_guard lock(m_mutex);
    if (!m_job.IsComplete())
        return "running";
    if (!m_state->error.empty())
        return "failed";
    if (m_state->control.load(std::memory_order_acquire) & State::cancel)
        return "cancelled";
    if (m_state->published)
        return "published";
    if (m_state->module->IsLoaded())
        return "loaded";
    if (m_state->module->IsInspected())
        return "inspected";
    return "inspecting";
}
std::string NativeModuleTask::GetError() const
{
    std::lock_guard lock(m_mutex);
    return m_job.IsComplete() ? m_state->error : std::string{};
}
size_t NativeModuleTask::GetRetainedBytes() const
{
    return sizeof(*this) + m_state->retained.load(std::memory_order_relaxed);
}
size_t NativeModuleTask::GetReservationBytes() const
{
    return sizeof(*this) + m_state->reservation.load(std::memory_order_relaxed);
}
size_t NativeModuleTask::GetImageBytes() const
{
    return m_state->imageBytes.load(std::memory_order_relaxed);
}
uintptr_t NativeModuleTask::GetModuleIdentity() const
{
    return m_state->identity.load(std::memory_order_relaxed);
}
uint64_t NativeModuleTask::GetReadBytes() const
{
    return m_state->readBytes.load(std::memory_order_relaxed);
}
uint64_t NativeModuleTask::GetLoadNs() const
{
    return m_state->loadNs.load(std::memory_order_relaxed);
}
uint64_t NativeModuleTask::GetExecutionNs() const
{
    return m_state->executionNs.load(std::memory_order_relaxed);
}
uint64_t NativeModuleTask::GetSubmissionCount() const
{
    return m_state->submissions.load(std::memory_order_relaxed);
}
IoWork NativeModuleTask::DescribeWork(size_t maximum) const
{
    std::lock_guard lock(m_mutex);
    auto &state = Idle();
    if (state.control.load(std::memory_order_acquire) & State::cancel)
        return {};
    return state.module->DescribeWork(maximum);
}
IoRecipe NativeModuleTask::DescribeRemainingWork() const
{
    std::lock_guard lock(m_mutex);
    auto &state = Idle();
    if ((state.control.load(std::memory_order_acquire) & State::cancel) || !state.module)
        return {};
    return state.module->DescribeRemainingWork();
}
IoWork NativeModuleTask::GetSubmittedWork() const
{
    std::lock_guard lock(m_mutex);
    return m_submittedWork;
}
bool NativeModuleTask::RanOnWorker() const
{
    return m_state->ranOnWorker.load(std::memory_order_relaxed);
}
} // namespace infernux
#endif
