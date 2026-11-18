#include "ShaderProgramDiskCache.h"

#include <core/serialization/BinaryCodec.h>
#include <platform/filesystem/AtomicFile.h>
#include <platform/filesystem/InxPath.h>

#include <cstdlib>
#include <limits>

namespace infernux
{
namespace
{
using Reader = binary_codec::Reader<(std::numeric_limits<uint32_t>::max)()>;

std::filesystem::path ProgramPath(const std::string &directory, std::string_view selection)
{
    return ToFsPath(directory) / "ShaderPrograms" / (std::to_string(binary_codec::Fnv1a64(selection)) + ".bin");
}

void AppendId(std::string &header, std::string_view id)
{
    binary_codec::AppendU32(header, static_cast<uint32_t>(id.size()));
    header.append(id);
}
} // namespace

std::string ProjectCacheDirectory(const std::string &projectRoot)
{
    const char *mode = std::getenv("_INFERNUX_PLAYER_MODE");
    if (mode && std::string_view(mode) == "1") {
        const char *root = std::getenv("_INFERNUX_PLAYER_PERSISTENT_DATA_ROOT");
        return root && *root ? FromFsPath(ToFsPath(root) / "Cache") : std::string{};
    }
    return projectRoot.empty() ? std::string{} : FromFsPath(ToFsPath(projectRoot) / "Cache");
}

ShaderCacheInput MakeShaderCacheInput(std::string id, std::string_view originalBytes)
{
    return {std::move(id), binary_codec::Fnv1a64(originalBytes)};
}

std::optional<std::string> ReadShaderProgramCache(const std::string &cacheDirectory, std::string_view selection,
                                                  const ShaderCacheRevision &currentRevision)
{
    if (cacheDirectory.empty())
        return std::nullopt;
    try {
        FileInput file(ProgramPath(cacheDirectory, selection), FileInput::Sharing::Snapshot);
        char prefix[sizeof(uint32_t)];
        file.Read(prefix, sizeof(prefix));
        const uint64_t size = file.Size() - sizeof(prefix);
        const auto headerSize = Reader({prefix, sizeof(prefix)}).ReadU32();
        if (headerSize > size)
            return std::nullopt;
        std::string header(headerSize, '\0');
        file.Read(header.data(), header.size());
        Reader record(header);
        if (record.ReadString(false) != selection)
            return std::nullopt;
        const auto count =
            record.ReadCount((std::numeric_limits<uint32_t>::max)(), sizeof(uint32_t) + sizeof(uint64_t));
        for (uint32_t i = 0; i < count; ++i) {
            const auto id = record.ReadString(false);
            const auto revision = record.ReadU64();
            if (currentRevision(id) != revision)
                return std::nullopt;
        }
        const auto programSize = record.ReadU64();
        const auto checksum = record.ReadU64();
        if (!record.AtEnd() || programSize != size - headerSize || programSize > std::string{}.max_size())
            return std::nullopt;
        std::string program(static_cast<size_t>(programSize), '\0');
        file.Read(program.data(), program.size());
        if (binary_codec::Fnv1a64(program) != checksum)
            return std::nullopt;
        return program;
    } catch (const std::invalid_argument &) {
        return std::nullopt; // An incomplete cache record is a miss.
    } catch (const std::runtime_error &) {
        return std::nullopt;
    }
}

bool WriteShaderProgramCache(const std::string &cacheDirectory, std::string_view selection,
                             const std::vector<ShaderCacheInput> &inputs, std::string_view compiledProgram)
{
    if (cacheDirectory.empty() || compiledProgram.empty())
        return false;
    const auto destination = ProgramPath(cacheDirectory, selection);
    try {
        std::filesystem::create_directories(destination.parent_path());
        std::string header;
        AppendId(header, selection);
        binary_codec::AppendU32(header, static_cast<uint32_t>(inputs.size()));
        for (const auto &input : inputs) {
            AppendId(header, input.id);
            binary_codec::AppendU64(header, input.revision);
        }
        binary_codec::AppendU64(header, compiledProgram.size());
        binary_codec::AppendU64(header, binary_codec::Fnv1a64(compiledProgram));
        std::string prefix;
        binary_codec::AppendU32(prefix, static_cast<uint32_t>(header.size()));
        AtomicFileWriter writer(FromFsPath(destination));
        writer.Append(prefix);
        writer.Append(header);
        writer.Append(compiledProgram);
        return writer.Publish().publication == FilePublication::Written;
    } catch (const std::runtime_error &) {
        // Cache storage can be unavailable even when the newly compiled program is usable.
        return false;
    }
}

} // namespace infernux
