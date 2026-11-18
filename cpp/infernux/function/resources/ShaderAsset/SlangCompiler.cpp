#include "SlangCompiler.h"
#include "ShaderAuthorInputs.h"
#include "SlangLayout.h"
#include "SlangModuleFiles.h"
#include <core/types/ShaderBuildStatistics.h>

#include <nlohmann/json.hpp>
#include <slang-com-ptr.h>
#include <slang.h>

#include <algorithm>
#include <charconv>
#include <cstring>
#include <iterator>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace infernux
{
namespace
{

std::string BlobText(slang::IBlob *blob)
{
    return blob ? std::string(static_cast<const char *>(blob->getBufferPointer()), blob->getBufferSize())
                : std::string{};
}

// Slang's documented machine-readable output is eight TSV fields. Parse it
// once at compilation; tools receive the record, never reparse rendered logs.
std::vector<ShaderDiagnostic> ReadDiagnostics(slang::IBlob *blob, const std::string &phase,
                                              const std::string &target = {})
{
    std::vector<ShaderDiagnostic> result;
    std::istringstream input(BlobText(blob));
    std::string line;
    while (std::getline(input, line)) {
        std::array<std::string_view, 8> fields;
        std::string_view remainder(line);
        size_t count = 0;
        for (; count < 7; ++count) {
            const auto separator = remainder.find('\t');
            if (separator == std::string_view::npos)
                break;
            fields[count] = remainder.substr(0, separator);
            remainder.remove_prefix(separator + 1);
        }
        if (count != 7) {
            // Diagnostic messages can contain line breaks. These are part of
            // the preceding message, not a second guessed source location.
            if (result.empty())
                result.push_back({"compiler.message", "note", {}, 0, 0, 0, 0, line, phase, target});
            else
                result.back().message += "\n" + line;
            continue;
        }
        fields[7] = remainder;
        // The public API may echo a fatal record prefixed by its abort
        // wrapper. Only the TSV records beginning with E belong to this
        // protocol; the original fatal record is already present.
        if (fields[0].empty() || fields[0].front() != 'E')
            continue;
        const auto number = [](std::string_view value) {
            uint32_t result = 0;
            std::from_chars(value.data(), value.data() + value.size(), result);
            return result;
        };
        result.push_back({std::string(fields[0]), std::string(fields[1]), std::string(fields[2]), number(fields[3]),
                          number(fields[4]), number(fields[5]), number(fields[6]), std::string(fields[7]), phase,
                          target});
    }
    return result;
}

std::vector<ShaderDiagnostic> FailureDiagnostics(const std::string &phase, const std::string &path, slang::IBlob *blob,
                                                 const std::string &target = {})
{
    auto result = ReadDiagnostics(blob, phase, target);
    if (result.empty())
        result.push_back({"compiler.failure", "error", path, 0, 0, 0, 0, phase + " failed", phase, target});
    return result;
}

void CheckCompilation(SlangResult result, const std::string &phase, slang::IBlob *diagnostic,
                      const std::string &path = {}, const std::string &target = {})
{
    if (SLANG_FAILED(result))
        throw ShaderCompilationError(FailureDiagnostics(phase, path, diagnostic, target));
}

void AppendDiagnostics(ShaderCompilation &compilation, slang::IBlob *blob, const std::string &phase,
                       const std::string &target = {})
{
    auto diagnostics = ReadDiagnostics(blob, phase, target);
    compilation.diagnostics.insert(compilation.diagnostics.end(), std::make_move_iterator(diagnostics.begin()),
                                   std::make_move_iterator(diagnostics.end()));
}

void MapAuthorRanges(std::vector<ShaderDiagnostic> &diagnostics, const std::vector<ShaderAuthorRange> &ranges)
{
    for (auto &diagnostic : diagnostics) {
        const auto range = std::find_if(ranges.begin(), ranges.end(), [&](const auto &range) {
            return range.path == diagnostic.path && range.line == diagnostic.line;
        });
        if (range == ranges.end())
            continue;
        // Slang columns refer to generated expressions. Report the original
        // AST statement, rather than inventing a token column in Python.
        diagnostic.column = range->column;
        diagnostic.endLine = range->endLine;
        diagnostic.endColumn = range->endColumn;
    }
}

rhi::ShaderStage StageFromReflection(SlangStage stage)
{
    switch (stage) {
    case SLANG_STAGE_VERTEX:
        return rhi::ShaderStage::Vertex;
    case SLANG_STAGE_FRAGMENT:
        return rhi::ShaderStage::Fragment;
    case SLANG_STAGE_COMPUTE:
        return rhi::ShaderStage::Compute;
    default:
        throw std::runtime_error("Shader entry stage has no engine execution contract: " + std::to_string(stage));
    }
}

// The numerical ABI preserves NaN, infinity and signed zero. Slang's precise
// mode supplies NoContraction; SPIR-V additionally requires an entry execution
// mode to make preservation binding on the driver. This is a compiler pass on
// trusted compiler output, not another shader parser or a validation scan.
void PreserveNumericalFloat32(std::vector<char> &code)
{
    std::vector<uint32_t> words(code.size() / sizeof(uint32_t));
    std::memcpy(words.data(), code.data(), code.size());
    std::vector<uint32_t> entries;
    size_t capabilityEnd = 5;
    size_t modeEnd = 5;
    for (size_t cursor = 5; cursor < words.size(); cursor += words[cursor] >> 16) {
        const auto opcode = words[cursor] & 0xffffu;
        const auto end = cursor + (words[cursor] >> 16);
        if (opcode == 17) // OpCapability
            capabilityEnd = end;
        if (opcode == 15) // OpEntryPoint: execution model, function ID, name
            entries.push_back(words[cursor + 2]);
        if (opcode == 15 || opcode == 16 || opcode == 331) // entry / execution modes
            modeEnd = end;
    }
    std::vector<uint32_t> modes;
    for (const auto entry : entries)
        modes.insert(modes.end(), {(4u << 16) | 16u, entry, 4461u, 32u}); // SignedZeroInfNanPreserve
    words.insert(words.begin() + modeEnd, modes.begin(), modes.end());
    words.insert(words.begin() + capabilityEnd, {(2u << 16) | 17u, 4466u}); // corresponding capability
    code.resize(words.size() * sizeof(uint32_t));
    std::memcpy(code.data(), words.data(), code.size());
}

struct LoadedModule
{
    Slang::ComPtr<slang::ISession> session;
    slang::IModule *module = nullptr;
    std::vector<ShaderDiagnostic> diagnostics;
};

LoadedModule LoadModule(slang::IGlobalSession *global, const ShaderSourceModule &source,
                        const std::vector<ShaderSourceModule> &libraries, const std::vector<ShaderCodeTarget> &targets,
                        bool precise)
{
    std::vector<slang::TargetDesc> targetDescriptions;
    targetDescriptions.reserve(targets.size());
    for (const auto target : targets) {
        slang::TargetDesc description;
        description.floatingPointMode = precise ? SLANG_FLOATING_POINT_MODE_PRECISE : SLANG_FLOATING_POINT_MODE_DEFAULT;
        description.format = target == ShaderCodeTarget::SpirV ? SLANG_SPIRV : SLANG_WGSL;
        if (target == ShaderCodeTarget::SpirV)
            description.profile = global->findProfile("spirv_1_5");
        targetDescriptions.push_back(description);
    }

    slang::CompilerOptionEntry options[5]{};
    options[0].name = slang::CompilerOptionName::EmitSpirvMethod;
    options[0].value.intValue0 = SLANG_EMIT_SPIRV_DIRECTLY;
    options[1].name = slang::CompilerOptionName::VulkanUseEntryPointName;
    options[1].value.intValue0 = 1;
    // Author clip space is Y-up on every target. Vulkan's positive-height
    // viewport needs the opposite position Y; WGSL retains the author value.
    options[2].name = slang::CompilerOptionName::VulkanInvertY;
    options[2].value.intValue0 = 1;
    options[3].name = slang::CompilerOptionName::EnableMachineReadableDiagnostics;
    options[3].value.intValue0 = 1;
    options[4].name = slang::CompilerOptionName::EnableRichDiagnostics;
    options[4].value.intValue0 = 1;

    slang::SessionDesc description;
    description.targets = targetDescriptions.data();
    description.targetCount = static_cast<SlangInt>(targetDescriptions.size());
    description.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    description.compilerOptionEntries = options;
    description.compilerOptionEntryCount = 5;
    const std::array searchPaths{SlangModuleFiles::Mount};
    description.searchPaths = searchPaths.data();
    description.searchPathCount = searchPaths.size();
    Slang::ComPtr<ISlangFileSystem> files(new SlangModuleFiles(source, libraries));
    description.fileSystem = files;

    LoadedModule result;
    CheckCompilation(global->createSession(description, result.session.writeRef()), "create_session", nullptr,
                     source.path);
    Slang::ComPtr<slang::IBlob> diagnostics;
    result.module = result.session->loadModuleFromSourceString(source.name.c_str(), source.path.c_str(),
                                                               source.source.c_str(), diagnostics.writeRef());
    if (!result.module)
        throw ShaderCompilationError(FailureDiagnostics("compile", source.path, diagnostics));
    result.diagnostics = ReadDiagnostics(diagnostics, "compile");
    return result;
}

bool SameSource(const ShaderSourceModule &left, const ShaderSourceModule &right)
{
    return left.name == right.name && left.path == right.path && left.source == right.source;
}

struct ComputeModule
{
    // The Slang file system borrows the snapshot, so the module and session
    // must be destroyed before the source strings and import closure.
    const ShaderSourceModule source;
    const std::vector<ShaderSourceModule> libraries;
    const std::vector<ShaderCodeTarget> targets;
    const bool precise;
    const LoadedModule frontend;

    ComputeModule(slang::IGlobalSession *global, const ShaderSourceModule &source,
                  const std::vector<ShaderSourceModule> &libraries, const std::vector<ShaderCodeTarget> &targets,
                  bool precise)
        : source(source), libraries(libraries), targets(targets), precise(precise),
          frontend(LoadModule(global, this->source, this->libraries, this->targets, precise))
    {
    }

    bool Matches(const ShaderSourceModule &requestedSource, const std::vector<ShaderSourceModule> &requestedLibraries,
                 const std::vector<ShaderCodeTarget> &requestedTargets, bool requestedPrecision) const
    {
        return precise == requestedPrecision && targets == requestedTargets && SameSource(source, requestedSource) &&
               std::equal(libraries.begin(), libraries.end(), requestedLibraries.begin(), requestedLibraries.end(),
                          SameSource);
    }
};

} // namespace

struct SlangCompiler::Impl
{
    Slang::ComPtr<slang::IGlobalSession> global;
    std::string version = SLANG_COMPILER_ID;
    ShaderAuthorFrontend authorFrontend;
    // Slang sessions that share a global session must be used serially. The
    // Python caller releases the GIL before entering this compilation owner.
    std::mutex mutex;
    std::unique_ptr<ComputeModule> recentComputeModule;
};

SlangCompiler &SlangCompiler::Get()
{
    static SlangCompiler compiler;
    return compiler;
}

SlangCompiler::SlangCompiler() : impl(std::make_unique<Impl>())
{
}
SlangCompiler::~SlangCompiler() = default;

const std::string &SlangCompiler::GetVersion() const noexcept
{
    return impl->version;
}

std::string_view SlangCompiler::GetAuthorContract() noexcept
{
    return SLANG_COMPILER_ID "/" INFERNUX_SHADER_AUTHOR_CONTRACT;
}

void SlangCompiler::SetAuthorFrontend(ShaderAuthorFrontend frontend)
{
    std::lock_guard lock(impl->mutex);
    impl->authorFrontend = std::move(frontend);
}

ShaderAuthorSource SlangCompiler::PrepareAuthorSource(const ShaderSourceModule &source, const ShaderTextReader &reader,
                                                      ShaderAuthorPhase phase, const ShaderSourceModule &vertex)
{
    ShaderAuthorFrontend frontend;
    {
        std::lock_guard lock(impl->mutex);
        frontend = impl->authorFrontend;
    }
    if (!frontend)
        throw std::logic_error("Static GPU authoring frontend is not registered");
    // Frontend work may acquire the host interpreter. Never hold a compiler
    // lock across that boundary; only the returned C++ snapshot enters Slang.
    return frontend(source, reader, phase, vertex);
}

ShaderCompilation SlangCompiler::CompileRootProgram(ShaderSourceModule source,
                                                    const std::vector<ShaderSourceModule> &libraries,
                                                    const std::vector<ShaderCodeTarget> &targets, bool precise,
                                                    const std::vector<ShaderAuthorRange> &sourceRanges)
{
    ShaderCompilation result;
    for (const auto target : targets) {
        const auto declaration = target == ShaderCodeTarget::SpirV
                                     ? "#define INX_ROOT_CONSTANTS [[vk::push_constant]]\n"
                                     : "#define INX_ROOT_CONSTANTS [[vk::binding(" +
                                           std::to_string(rhi::RootConstantBinding) + ", " +
                                           std::to_string(rhi::RootConstantGroup) + ")]]\n";
        const ShaderSourceModule prepared{source.name, source.path, declaration + "#line 1\n" + source.source};
        auto compiled = Compile(prepared, libraries, {target}, precise, sourceRanges);
        result.targets.push_back(std::move(compiled.targets.front()));
        result.diagnostics.insert(result.diagnostics.end(), std::make_move_iterator(compiled.diagnostics.begin()),
                                  std::make_move_iterator(compiled.diagnostics.end()));
    }
    return result;
}

ShaderCompilation SlangCompiler::CompileAuthorProgram(const ShaderAuthorSource &source)
{
    const auto schema = nlohmann::json::parse(source.schema);
    const auto &domain = schema.at("domain");
    const std::vector<ShaderCodeTarget> targets{ShaderCodeTarget::SpirV, ShaderCodeTarget::Wgsl};
    auto result = domain == "screen_ui" || domain == "world_ui"
                      ? CompileRootProgram(source.source, source.libraries, targets, false, source.sourceRanges)
                      : Compile(source.source, source.libraries, targets, false, source.sourceRanges);
    std::vector<std::pair<std::string, std::string>> fields;
    for (const auto &input : schema.at("inputs"))
        fields.emplace_back(input.at("name").get<std::string>(), input.at("kind").get<std::string>());
    for (const auto &parameter : schema.at("parameters"))
        if (parameter.at("kind") == "texture_2d")
            fields.emplace_back(parameter.at("name").get<std::string>(), "texture_2d");
    ApplyShaderAuthorResources(result, fields);
    return result;
}

ShaderCompilation SlangCompiler::Compile(const ShaderSourceModule &source,
                                         const std::vector<ShaderSourceModule> &libraries,
                                         const std::vector<ShaderCodeTarget> &targets, bool precise,
                                         const std::vector<ShaderAuthorRange> &sourceRanges)
{
    return CompileProgram(source, libraries, targets, precise, sourceRanges, nullptr, nullptr);
}

ShaderCompilation SlangCompiler::CompileComputeEntry(const ShaderSourceModule &source,
                                                     const std::vector<ShaderSourceModule> &libraries,
                                                     const std::vector<ShaderCodeTarget> &targets,
                                                     const std::string &entryPoint, bool precise,
                                                     const std::vector<ShaderAuthorRange> &sourceRanges,
                                                     bool &frontendReused)
{
    return CompileProgram(source, libraries, targets, precise, sourceRanges, &entryPoint, &frontendReused);
}

ShaderCompilation SlangCompiler::CompileProgram(const ShaderSourceModule &source,
                                                const std::vector<ShaderSourceModule> &libraries,
                                                const std::vector<ShaderCodeTarget> &targets, bool precise,
                                                const std::vector<ShaderAuthorRange> &sourceRanges,
                                                const std::string *entryPoint, bool *frontendReused)
try {
    RecordShaderCompilationAttempt();
    std::lock_guard lock(impl->mutex);
    if (!impl->global)
        CheckCompilation(slang::createGlobalSession(impl->global.writeRef()), "initialize", nullptr);

    LoadedModule uncached;
    const LoadedModule *frontend = &uncached;
    if (entryPoint) {
        auto &retained = impl->recentComputeModule;
        *frontendReused = retained && retained->Matches(source, libraries, targets, precise);
        if (!*frontendReused)
            retained = std::make_unique<ComputeModule>(impl->global.get(), source, libraries, targets, precise);
        frontend = &retained->frontend;
    } else {
        uncached = LoadModule(impl->global.get(), source, libraries, targets, precise);
    }

    ShaderCompilation result;
    result.diagnostics = frontend->diagnostics;
    Slang::ComPtr<slang::IBlob> diagnostics;
    auto *module = frontend->module;
    const auto entryCount = entryPoint ? 1 : module->getDefinedEntryPointCount();
    std::vector<Slang::ComPtr<slang::IEntryPoint>> entries(static_cast<size_t>(entryCount));
    if (entryPoint) {
        if (SLANG_FAILED(module->findEntryPointByName(entryPoint->c_str(), entries.front().writeRef())))
            throw ShaderCompilationError({{"compiler.missing_entry",
                                           "error",
                                           source.path,
                                           0,
                                           0,
                                           0,
                                           0,
                                           "Entry '" + *entryPoint + "' must be a declared compute entry",
                                           "find_entry",
                                           {}}});
    } else {
        for (SlangInt32 index = 0; index < entryCount; ++index)
            CheckCompilation(module->getDefinedEntryPoint(index, entries[index].writeRef()), "find_entry", nullptr,
                             source.path);
    }
    std::vector<slang::IComponentType *> components{module};
    for (const auto &entry : entries)
        components.push_back(entry.get());

    Slang::ComPtr<slang::IComponentType> composed;
    auto status = frontend->session->createCompositeComponentType(
        components.data(), static_cast<SlangInt>(components.size()), composed.writeRef(), diagnostics.writeRef());
    CheckCompilation(status, "compose", diagnostics, source.path);
    AppendDiagnostics(result, diagnostics, "compose");

    Slang::ComPtr<slang::IComponentType> program;
    diagnostics.setNull();
    status = composed->link(program.writeRef(), diagnostics.writeRef());
    CheckCompilation(status, "link", diagnostics, source.path);
    AppendDiagnostics(result, diagnostics, "link");

    for (SlangInt targetIndex = 0; targetIndex < static_cast<SlangInt>(targets.size()); ++targetIndex) {
        ShaderTargetCode target;
        target.target = targets[targetIndex];
        const std::string targetName = target.target == ShaderCodeTarget::SpirV ? "spirv" : "wgsl";
        diagnostics.setNull();
        auto *layout = program->getLayout(targetIndex, diagnostics.writeRef());
        if (!layout)
            throw ShaderCompilationError(FailureDiagnostics("reflect", source.path, diagnostics, targetName));
        AppendDiagnostics(result, diagnostics, "reflect", targetName);
        target.resources = ReflectSlangResources(layout->getGlobalParamsTypeLayout());
        const auto rootParameters = ReflectSlangRootParameters(layout->getGlobalParamsTypeLayout());

        for (SlangInt entryIndex = 0; entryIndex < entryCount; ++entryIndex) {
            Slang::ComPtr<slang::IBlob> code;
            diagnostics.setNull();
            status = program->getEntryPointCode(entryIndex, targetIndex, code.writeRef(), diagnostics.writeRef());
            CheckCompilation(status, "generate", diagnostics, source.path, targetName);
            AppendDiagnostics(result, diagnostics, "generate", targetName);

            // Code generation may legalize entry-point names for the target.
            auto *entry = layout->getEntryPointByIndex(static_cast<SlangUInt>(entryIndex));
            ShaderEntryPointCode artifact;
            const auto *overrideName = entry->getNameOverride();
            artifact.name = overrideName && *overrideName ? overrideName : entry->getName();
            artifact.stage = StageFromReflection(entry->getStage());
            ReflectSlangStageIO(entry, artifact);
            if (artifact.stage == rhi::ShaderStage::Compute) {
                SlangUInt groupSize[3]{};
                entry->getComputeThreadGroupSize(3, groupSize);
                for (size_t axis = 0; axis < 3; ++axis)
                    artifact.workgroupSize[axis] = static_cast<uint32_t>(groupSize[axis]);
            }
            Slang::ComPtr<slang::IMetadata> metadata;
            diagnostics.setNull();
            status =
                program->getEntryPointMetadata(entryIndex, targetIndex, metadata.writeRef(), diagnostics.writeRef());
            CheckCompilation(status, "entry_resources", diagnostics, source.path, targetName);
            for (const auto &resource : target.resources) {
                bool used = false;
                CheckCompilation(metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT,
                                                                   resource.set, resource.binding.binding, used),
                                 "descriptor_use", nullptr, source.path, targetName);
                if (used)
                    artifact.bindings.push_back({resource.set, resource.binding.binding});
            }
            for (const auto &root : rootParameters) {
                bool used = false;
                CheckCompilation(metadata->isParameterLocationUsed(root.category, root.space, root.binding, used),
                                 "root_parameter_use", nullptr, source.path, targetName);
                if (!used)
                    continue;
                if (artifact.rootConstants)
                    throw ShaderCompilationError(
                        {{"author.root_parameters", "error", source.path, 0, 0, 0, 0,
                          "Shader entry '" + artifact.name + "' uses more than one per-execution parameter block",
                          "reflect", targetName}});
                artifact.rootConstants = root.value;
            }
            const auto *bytes = static_cast<const char *>(code->getBufferPointer());
            artifact.code.assign(bytes, bytes + code->getBufferSize());
            if (precise && target.target == ShaderCodeTarget::SpirV)
                PreserveNumericalFloat32(artifact.code);
            target.entries.push_back(std::move(artifact));
        }
        for (auto &resource : target.resources) {
            for (const auto &entry : target.entries) {
                const std::array<uint32_t, 2> key{resource.set, resource.binding.binding};
                if (std::find(entry.bindings.begin(), entry.bindings.end(), key) != entry.bindings.end())
                    resource.binding.visibility = resource.binding.visibility | entry.stage;
            }
        }
        Slang::ComPtr<slang::IBlob> reflection;
        CheckCompilation(layout->toJson(reflection.writeRef()), "serialize_layout", nullptr, source.path, targetName);
        target.reflection = BlobText(reflection);
        result.targets.push_back(std::move(target));
    }
    MapAuthorRanges(result.diagnostics, sourceRanges);
    return result;
} catch (const ShaderCompilationError &error) {
    auto diagnostics = error.diagnostics;
    MapAuthorRanges(diagnostics, sourceRanges);
    throw ShaderCompilationError(std::move(diagnostics));
}

} // namespace infernux
