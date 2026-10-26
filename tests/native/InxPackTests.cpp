#include <platform/filesystem/InxPack.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <winioctl.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace
{
std::filesystem::path UniqueTestRoot()
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / ("infernux-pack-" + std::to_string(nonce));
}

void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void WriteSource(const std::filesystem::path &path, const std::vector<uint8_t> &bytes)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    Require(static_cast<bool>(output), "cannot create test source file");
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    Require(static_cast<bool>(output), "cannot write test source file");
}

std::vector<uint8_t> ReadFile(const std::filesystem::path &path)
{
    std::ifstream input(path, std::ios::binary);
    Require(static_cast<bool>(input), "cannot open test package");
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void RequireNoTemporaryFiles(const std::filesystem::path &root)
{
    for (const auto &entry : std::filesystem::directory_iterator(root)) {
        const std::string name = infernux::FromFsPath(entry.path().filename());
        Require(name.find(".tmp.inxpkg.") == std::string::npos, "InxPack left a temporary output file behind");
    }
}

void MakeDirectoryLink(const std::filesystem::path &link, const std::filesystem::path &target)
{
#ifdef _WIN32
    // Junctions need no symlink privilege, so the confinement regression also
    // runs on ordinary Windows CI accounts. Both paths remain in the test root.
    struct MountPoint
    {
        DWORD tag;
        WORD length, reserved, substituteOffset, substituteLength, printOffset, printLength;
        wchar_t names[1];
    };
    const auto print = std::filesystem::absolute(target).wstring();
    const auto substitute = L"\\??\\" + print;
    const auto namesBytes = (substitute.size() + print.size() + 2) * sizeof(wchar_t);
    std::vector<uint8_t> bytes(offsetof(MountPoint, names) + namesBytes, 0);
    auto *point = reinterpret_cast<MountPoint *>(bytes.data());
    point->tag = IO_REPARSE_TAG_MOUNT_POINT;
    point->length = static_cast<WORD>(bytes.size() - 8);
    point->substituteLength = static_cast<WORD>(substitute.size() * sizeof(wchar_t));
    point->printOffset = point->substituteLength + sizeof(wchar_t);
    point->printLength = static_cast<WORD>(print.size() * sizeof(wchar_t));
    std::memcpy(point->names, substitute.c_str(), point->substituteLength);
    std::memcpy(reinterpret_cast<uint8_t *>(point->names) + point->printOffset, print.c_str(), point->printLength);
    std::filesystem::create_directory(link);
    HANDLE handle = ::CreateFileW(link.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
                                  FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    Require(handle != INVALID_HANDLE_VALUE, "cannot open test junction");
    DWORD returned = 0;
    const bool created = ::DeviceIoControl(handle, FSCTL_SET_REPARSE_POINT, bytes.data(),
                                           static_cast<DWORD>(bytes.size()), nullptr, 0, &returned, nullptr) != FALSE;
    ::CloseHandle(handle);
    Require(created, "cannot create test junction");
#else
    std::filesystem::create_directory_symlink(target, link);
#endif
}

void RequireExtractFailure(const std::filesystem::path &package, const std::filesystem::path &destination)
{
    bool rejected = false;
    try {
        infernux::inxpack::Extract(package, destination);
    } catch (const std::exception &) {
        rejected = true;
    }
    Require(rejected, "extraction unexpectedly succeeded");
}
} // namespace

int main()
{
    using namespace infernux;
    using namespace infernux::inxpack;

    std::cerr << "[InxPackTests] main\n" << std::flush;
    std::cerr << "[InxPackTests] UniqueTestRoot\n" << std::flush;
    const auto root = UniqueTestRoot();
    const auto cleanup = [&root]() {
        std::error_code cleanupError;
        std::filesystem::remove_all(root, cleanupError);
        return cleanupError;
    };

    const char *phase = "setup";
    try {
        const auto sourceDirectory = root / ToFsPath("源文件目录");
        const auto sourcePath = sourceDirectory / ToFsPath("中文源文件.bin");
        const auto packagePath = root / ToFsPath("中文临时目录") / ToFsPath("中文运行包.inxrt");
        const auto extractRoot = root / ToFsPath("解包结果");
        const std::vector<uint8_t> expected = {0x00, 0x11, 0x22, 0x80, 0xfe, 0xff};

        phase = "create Chinese source file";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        std::filesystem::create_directories(sourceDirectory);
        {
            std::ofstream source(sourcePath, std::ios::binary | std::ios::trunc);
            Require(static_cast<bool>(source), "cannot create source file");
            source.write(reinterpret_cast<const char *>(expected.data()),
                         static_cast<std::streamsize>(expected.size()));
            Require(static_cast<bool>(source), "cannot write source file");
        }

        phase = "compression options and deterministic output";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        Require(ResolveCompressionLevel({}) == kDevelopmentCompressionLevel, "the default compression profile changed");
        Require(ResolveCompressionLevel({CompressionProfile::Release, 0}) == kReleaseCompressionLevel,
                "the release compression profile changed");
        Require(ResolveCompressionLevel({CompressionProfile::Development, 7}) == 7,
                "an explicit compression level was not selected");
        for (const int invalidLevel : {kMinimumCompressionLevel - 1, kMaximumCompressionLevel + 1}) {
            bool rejected = false;
            try {
                ValidateCompressionLevel(invalidLevel);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            Require(rejected, "an invalid compression level was accepted");
        }

        const auto compressibleSource = sourceDirectory / ToFsPath("可压缩源文件.bin");
        std::vector<uint8_t> compressibleBytes(64 * 1024);
        for (size_t index = 0; index < compressibleBytes.size(); ++index)
            compressibleBytes[index] = static_cast<uint8_t>((index / 256) % 7);
        WriteSource(compressibleSource, compressibleBytes);
        const auto deterministicA = root / ToFsPath("确定性A.inxrt");
        const auto deterministicB = root / ToFsPath("确定性B.inxrt");
        const auto releasePackage = root / ToFsPath("release.inxrt");
        const std::vector<SourceFile> compressibleSources = {{"Runtime/Resources.bin", compressibleSource}};
        const Manifest defaultA = Write(deterministicA, compressibleSources);
        const Manifest defaultB = Write(deterministicB, compressibleSources);
        Require(ReadFile(deterministicA) == ReadFile(deterministicB), "default InxPack output is not deterministic");
        const Manifest release =
            Write(releasePackage, compressibleSources, {CompressionProfile::Release, kReleaseCompressionLevel});
        Require(release.entries.size() == 1 && release.entries.front().rawBytes == compressibleBytes.size(),
                "explicit release compression produced an invalid manifest");

        phase = "Write";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        const Manifest written = Write(packagePath, {{"Assets/中文/源文件.bin", sourcePath}});
        Require(written.entries.size() == 1, "Write returned an unexpected entry count");
        Require(std::filesystem::is_regular_file(packagePath), "Write did not create the package");

        phase = "ReadManifest";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        const Manifest read = ReadManifest(packagePath);
        Require(read.entries.size() == 1, "ReadManifest returned an unexpected entry count");
        Require(read.entries.front().path == "Assets/中文/源文件.bin", "ReadManifest returned the wrong entry path");

        phase = "ReadEntry";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        Require(ReadEntry(packagePath, read.entries.front().path) == expected, "ReadEntry returned unexpected bytes");

        phase = "Extract";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        Extract(packagePath, extractRoot);
        const auto extractedPath = extractRoot / ToFsPath("Assets/中文/源文件.bin");
        Require(std::filesystem::is_regular_file(extractedPath), "Extract did not create the entry");
        {
            std::ifstream extracted(extractedPath, std::ios::binary);
            Require(static_cast<bool>(extracted), "cannot open extracted entry");
            const std::vector<uint8_t> actual((std::istreambuf_iterator<char>(extracted)),
                                              std::istreambuf_iterator<char>());
            Require(actual == expected, "Extract returned unexpected bytes");
        }

        phase = "Unicode replacement";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        const auto replacementSource = sourceDirectory / ToFsPath("替换源文件.bin");
        const std::vector<uint8_t> replacementBytes = {0xaa, 0xbb, 0xcc, 0xdd};
        WriteSource(replacementSource, replacementBytes);
        Write(packagePath, {{"Assets/中文/源文件.bin", replacementSource}});
        Require(ReadEntry(packagePath, "Assets/中文/源文件.bin") == replacementBytes,
                "Unicode package replacement did not publish the new package");

#ifdef _WIN32
        phase = "failed replacement preserves old package";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        const auto lockedSource = sourceDirectory / ToFsPath("锁定替换源文件.bin");
        const std::vector<uint8_t> lockedBytes = {0x10, 0x20, 0x30};
        WriteSource(lockedSource, lockedBytes);
        Write(packagePath, {{"Assets/中文/源文件.bin", replacementSource}});
        HANDLE packageLock = ::CreateFileW(packagePath.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        Require(packageLock != INVALID_HANDLE_VALUE, "cannot lock package for replacement test");
        bool replacementFailed = false;
        try {
            Write(packagePath, {{"Assets/中文/源文件.bin", lockedSource}});
        } catch (const std::exception &) {
            replacementFailed = true;
        }
        ::CloseHandle(packageLock);
        Require(replacementFailed, "locked package replacement unexpectedly succeeded");
        Require(ReadEntry(packagePath, "Assets/中文/源文件.bin") == replacementBytes,
                "failed replacement damaged the previous valid package");
#endif

        phase = "concurrent writes";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        constexpr size_t writerCount = 8;
        std::vector<std::filesystem::path> concurrentSources;
        concurrentSources.reserve(writerCount);
        for (size_t index = 0; index < writerCount; ++index) {
            const auto concurrentSource = sourceDirectory / ToFsPath("并发源文件" + std::to_string(index) + ".bin");
            WriteSource(concurrentSource, {static_cast<uint8_t>(index), 0x42, static_cast<uint8_t>(index + 1), 0xfe});
            concurrentSources.push_back(concurrentSource);
        }
        std::mutex failureMutex;
        std::exception_ptr concurrentFailure;
        std::vector<std::thread> writers;
        writers.reserve(writerCount);
        for (const auto &concurrentSource : concurrentSources) {
            writers.emplace_back([&packagePath, &failureMutex, &concurrentFailure, concurrentSource]() {
                try {
                    Write(packagePath, {{"Assets/中文/并发.bin", concurrentSource}});
                } catch (...) {
                    std::lock_guard lock(failureMutex);
                    if (!concurrentFailure)
                        concurrentFailure = std::current_exception();
                }
            });
        }
        for (auto &writer : writers)
            writer.join();
        if (concurrentFailure)
            std::rethrow_exception(concurrentFailure);
        const Manifest concurrentManifest = ReadManifest(packagePath);
        Require(concurrentManifest.entries.size() == 1, "concurrent writes produced an invalid manifest");
        const auto concurrentBytes = ReadEntry(packagePath, "Assets/中文/并发.bin");
        Require(concurrentBytes.size() == 4 && concurrentBytes[1] == 0x42 && concurrentBytes[3] == 0xfe,
                "concurrent writes produced unexpected package contents");
        RequireNoTemporaryFiles(packagePath.parent_path());

        phase = "streaming entries";
        std::vector<uint8_t> large(5 * 1024 * 1024 + 19);
        uint32_t random = 1234567;
        for (auto &value : large) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            value = static_cast<uint8_t>(random);
        }
        const auto largeSource = root / "large.bin";
        const auto audioSource = root / "compressed.MP3";
        const auto largePack = root / "stream.inxpkg";
        WriteSource(largeSource, large);
        WriteSource(audioSource, large);
        auto largeManifest = Write(largePack, {{"Library/large.bin", largeSource}, {"Library/audio.mp3", audioSource}});
        Require(largeManifest.entries[0].codec == Codec::Store, "encoded audio must not be compressed twice");
        Require(largeManifest.entries[1].codec == Codec::Zstandard, "large source must use streamed compression");
        Require(ReadEntry(largePack, "Library/large.bin") == large, "large random payload changed");
        Extract(largePack, root / "stream-extracted");
        Require(ReadFile(root / "stream-extracted/Library/large.bin") == large,
                "streamed Zstd extraction changed bytes");
        Require(ReadFile(root / "stream-extracted/Library/audio.mp3") == large,
                "streamed Store extraction changed bytes");
        std::fill(large.begin(), large.end(), 0);
        WriteSource(largeSource, large);
        Write(largePack, {{"Library/large.bin", largeSource}});
        Extract(largePack, root / "stream-zeros");
        Require(ReadFile(root / "stream-zeros/Library/large.bin") == large, "high-ratio output was not fully drained");

        phase = "extraction confinement and atomic publication";
        for (const auto &unsafe : {"Assets/.. /value.bin", "Assets/value.bin.", "Assets/NUL.txt", "Assets/NUL .txt",
                                   "Assets/CONOUT$", "Assets/COM1"}) {
            bool rejected = false;
            try {
                Write(root / "unsafe.inxpkg", {{unsafe, replacementSource}});
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            Require(rejected, "package accepted a nonportable output path");
        }
        const auto confinedPackage = root / "confined.inxpkg";
        Write(confinedPackage, {{"Assets/value.bin", replacementSource}});
        const auto outside = root / "link-target";
        std::filesystem::create_directory(outside);
        const auto linkedOutput = root / "linked-output";
        std::filesystem::create_directory(linkedOutput);
        MakeDirectoryLink(linkedOutput / "Assets", outside);
        RequireExtractFailure(confinedPackage, linkedOutput);
        Require(!std::filesystem::exists(outside / "value.bin"), "extraction followed an output directory link");
        std::filesystem::remove(linkedOutput / "Assets");
        MakeDirectoryLink(root / "linked-root", outside);
        RequireExtractFailure(confinedPackage, root / "linked-root");
        Extract(confinedPackage, root / "linked-root/nested");
        Require(ReadFile(outside / "nested/Assets/value.bin") == replacementBytes,
                "caller-selected destination ancestor alias was not resolved");
        std::filesystem::remove(root / "linked-root");

        const auto atomicRoot = root / "atomic-output";
        std::filesystem::create_directories(atomicRoot / "Assets");
        const auto sentinel = outside / "sentinel.bin";
        WriteSource(sentinel, expected);
        std::filesystem::create_hard_link(sentinel, atomicRoot / "Assets/value.bin");
        Extract(confinedPackage, atomicRoot);
        Require(ReadFile(sentinel) == expected, "extraction modified the other name of a hard-linked output");
        Require(ReadFile(atomicRoot / "Assets/value.bin") == replacementBytes,
                "atomic extraction did not publish entry");

        const auto brokenPackage = root / "broken.inxpkg";
        const auto brokenManifest = Write(brokenPackage, {{"Assets/value.bin", largeSource}});
        Require(brokenManifest.entries.front().storedBytes > 1, "truncated test payload is too small");
        // Shorten the encoded frame in its TOC, keeping the physical archive
        // intact, so the decode fails after an output file has been opened.
        {
            std::fstream broken(brokenPackage, std::ios::binary | std::ios::in | std::ios::out);
            const uint64_t shortened = brokenManifest.entries.front().storedBytes - 1;
            broken.seekp(128 + 24 + 24);
            broken.write(reinterpret_cast<const char *>(&shortened), sizeof(shortened));
            Require(static_cast<bool>(broken), "cannot shorten test entry");
        }
        RequireExtractFailure(brokenPackage, atomicRoot);
        Require(ReadFile(atomicRoot / "Assets/value.bin") == replacementBytes, "failed extraction damaged old entry");
        RequireNoTemporaryFiles(atomicRoot / "Assets");
#ifdef _WIN32
        const auto collisionPackage = root / "collision.inxpkg";
        Write(collisionPackage, {{"Assets/value.bin", replacementSource}, {"Assets/Value.bin", sourcePath}});
        RequireExtractFailure(collisionPackage, root / "collision-output");
#endif

#ifndef _WIN32
#if defined(__linux__) || defined(__ANDROID__) || (defined(O_SEARCH) && !defined(__EMSCRIPTEN__))
        phase = "search-only extraction directories";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        const auto searchOnlyAncestor = root / "search-only-ancestor";
        const auto searchOnlyOutput = searchOnlyAncestor / "owned-output";
        std::filesystem::create_directories(searchOnlyOutput);
        // Android's /data and /data/user ancestors permit app traversal but
        // not directory reads. The owned output itself only needs write/search
        // permission: extraction never enumerates directory entries.
        std::filesystem::permissions(searchOnlyAncestor, std::filesystem::perms::owner_exec,
                                     std::filesystem::perm_options::replace);
        std::filesystem::permissions(searchOnlyOutput,
                                     std::filesystem::perms::owner_write | std::filesystem::perms::owner_exec,
                                     std::filesystem::perm_options::replace);
        const auto restorePermissions = [&]() {
            std::error_code error;
            std::filesystem::permissions(searchOnlyAncestor, std::filesystem::perms::owner_all, error);
            std::filesystem::permissions(searchOnlyOutput, std::filesystem::perms::owner_all, error);
        };
        try {
            // Root bypasses DAC checks. On ordinary CI accounts prove this
            // fixture really rejects the read access the extractor must avoid.
            if (::geteuid() != 0) {
                for (const auto &directory : {searchOnlyAncestor, searchOnlyOutput}) {
                    const int readable = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY);
                    const int error = errno;
                    if (readable >= 0)
                        ::close(readable);
                    Require(readable < 0 && error == EACCES, "search-only directory unexpectedly permits reads");
                }
            }
            Extract(confinedPackage, searchOnlyOutput);
            Require(ReadFile(searchOnlyOutput / "Assets/value.bin") == replacementBytes,
                    "extraction requires read access to an ancestor or output directory");
        } catch (...) {
            restorePermissions();
            throw;
        }
        restorePermissions();
#endif

        std::filesystem::remove(atomicRoot / "Assets/value.bin");
        std::filesystem::create_symlink(sentinel, atomicRoot / "Assets/value.bin");
        RequireExtractFailure(confinedPackage, atomicRoot);
        Require(ReadFile(sentinel) == expected, "extraction followed an output file symlink");
        std::filesystem::remove(atomicRoot / "Assets/value.bin");
#endif

        phase = "cleanup";
        std::cerr << "[InxPackTests] " << phase << '\n' << std::flush;
        const std::error_code cleanupError = cleanup();
        if (cleanupError)
            throw std::runtime_error("cleanup failed: " + cleanupError.message());
        return 0;
    } catch (const std::exception &error) {
        cleanup();
        std::cerr << "InxPackTests failed during " << phase << ": " << error.what() << '\n';
        return 1;
    }
}
