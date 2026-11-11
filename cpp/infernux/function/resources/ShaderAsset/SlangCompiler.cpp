#include "SlangCompiler.h"
#include "SlangLayout.h"

#include <slang-com-ptr.h>
#include <slang.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace infernux
{
namespace
{

std::string BlobText(slang::IBlob *blob)
{
    return blob ? std::string(static_cast<const char *>(blob->getBufferPointer()), blob->getBufferSize())
                : std::string{};
}

void CheckCompilation(SlangResult result, const std::string &operation, slang::IBlob *diagnostic)
{
    if (SLANG_FAILED(result))
        throw std::runtime_error(operation + ":\n" + BlobText(diagnostic));
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

} // namespace

struct SlangCompiler::Impl
{
    Slang::ComPtr<slang::IGlobalSession> global;
    std::string version;
    std::string standardModuleDirectory;
    ShaderAuthorFrontend authorFrontend;
    // Slang sessions that share a global session must be used serially. The
    // Python caller releases the GIL before entering this compilation owner.
    std::mutex mutex;

    Impl()
    {
        CheckCompilation(slang::createGlobalSession(global.writeRef()), "Create Slang compiler", nullptr);
        version = global->getBuildTagString();
    }
};

SlangCompiler &SlangCompiler::Get()
{
    static SlangCompiler compiler;
    return compiler;
}

const char *SlangCompiler::GetStandardModuleName() noexcept
{
    return INFERNUX_SLANG_MODULE_NAME;
}

SlangCompiler::SlangCompiler() : impl(std::make_unique<Impl>())
{
}
SlangCompiler::~SlangCompiler() = default;

const std::string &SlangCompiler::GetVersion() const noexcept
{
    return impl->version;
}

void SlangCompiler::SetStandardModuleDirectory(std::string directory)
{
    std::lock_guard lock(impl->mutex);
    impl->standardModuleDirectory = std::move(directory);
}

void SlangCompiler::SetAuthorFrontend(ShaderAuthorFrontend frontend)
{
    std::lock_guard lock(impl->mutex);
    impl->authorFrontend = std::move(frontend);
}

ShaderAuthorSource SlangCompiler::PrepareAuthorSource(const ShaderSourceModule &source, const ShaderTextReader &reader)
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
    return frontend(source, reader);
}

ShaderCompilation SlangCompiler::Compile(const ShaderSourceModule &source,
                                         const std::vector<ShaderSourceModule> &libraries,
                                         const std::vector<ShaderCodeTarget> &targets, bool precise)
{
    std::lock_guard lock(impl->mutex);

    std::vector<slang::TargetDesc> targetDescriptions;
    targetDescriptions.reserve(targets.size());
    for (const auto target : targets) {
        slang::TargetDesc description;
        description.floatingPointMode = precise ? SLANG_FLOATING_POINT_MODE_PRECISE : SLANG_FLOATING_POINT_MODE_DEFAULT;
        description.format = target == ShaderCodeTarget::SpirV ? SLANG_SPIRV : SLANG_WGSL;
        if (target == ShaderCodeTarget::SpirV)
            description.profile = impl->global->findProfile("spirv_1_5");
        targetDescriptions.push_back(description);
    }

    slang::CompilerOptionEntry options[3]{};
    options[0].name = slang::CompilerOptionName::EmitSpirvMethod;
    options[0].value.intValue0 = SLANG_EMIT_SPIRV_DIRECTLY;
    options[1].name = slang::CompilerOptionName::VulkanUseEntryPointName;
    options[1].value.intValue0 = 1;
    // Author clip space is Y-up on every target. Vulkan's positive-height
    // viewport needs the opposite position Y; WGSL retains the author value.
    options[2].name = slang::CompilerOptionName::VulkanInvertY;
    options[2].value.intValue0 = 1;

    slang::SessionDesc description;
    description.targets = targetDescriptions.data();
    description.targetCount = static_cast<SlangInt>(targetDescriptions.size());
    description.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    description.compilerOptionEntries = options;
    description.compilerOptionEntryCount = 3;
    const char *moduleDirectory = impl->standardModuleDirectory.c_str();
    description.searchPaths = &moduleDirectory;
    description.searchPathCount = impl->standardModuleDirectory.empty() ? 0 : 1;

    Slang::ComPtr<slang::ISession> session;
    CheckCompilation(impl->global->createSession(description, session.writeRef()), "Create shader session", nullptr);

    ShaderCompilation result;
    auto loadModule = [&](const ShaderSourceModule &module) {
        Slang::ComPtr<slang::IBlob> diagnostics;
        auto *loaded = session->loadModuleFromSourceString(module.name.c_str(), module.path.c_str(),
                                                           module.source.c_str(), diagnostics.writeRef());
        result.diagnostics += BlobText(diagnostics);
        if (!loaded)
            throw std::runtime_error("Compile " + module.path + ":\n" + BlobText(diagnostics));
        return loaded;
    };
    for (const auto &library : libraries)
        loadModule(library);
    auto *module = loadModule(source);

    const auto entryCount = module->getDefinedEntryPointCount();
    std::vector<Slang::ComPtr<slang::IEntryPoint>> entries(static_cast<size_t>(entryCount));
    std::vector<slang::IComponentType *> components{module};
    for (SlangInt32 index = 0; index < entryCount; ++index) {
        CheckCompilation(module->getDefinedEntryPoint(index, entries[index].writeRef()), "Find shader entry", nullptr);
        components.push_back(entries[index].get());
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IComponentType> composed;
    auto status = session->createCompositeComponentType(components.data(), static_cast<SlangInt>(components.size()),
                                                        composed.writeRef(), diagnostics.writeRef());
    CheckCompilation(status, "Compose " + source.path, diagnostics);
    result.diagnostics += BlobText(diagnostics);

    Slang::ComPtr<slang::IComponentType> program;
    diagnostics.setNull();
    status = composed->link(program.writeRef(), diagnostics.writeRef());
    CheckCompilation(status, "Link " + source.path, diagnostics);
    result.diagnostics += BlobText(diagnostics);

    for (SlangInt targetIndex = 0; targetIndex < static_cast<SlangInt>(targets.size()); ++targetIndex) {
        ShaderTargetCode target;
        target.target = targets[targetIndex];
        diagnostics.setNull();
        auto *layout = program->getLayout(targetIndex, diagnostics.writeRef());
        if (!layout)
            throw std::runtime_error("Reflect " + source.path + ":\n" + BlobText(diagnostics));
        result.diagnostics += BlobText(diagnostics);
        target.resources = ReflectSlangResources(layout->getGlobalParamsTypeLayout());

        for (SlangInt entryIndex = 0; entryIndex < entryCount; ++entryIndex) {
            Slang::ComPtr<slang::IBlob> code;
            diagnostics.setNull();
            status = program->getEntryPointCode(entryIndex, targetIndex, code.writeRef(), diagnostics.writeRef());
            CheckCompilation(status, "Generate " + source.path, diagnostics);
            result.diagnostics += BlobText(diagnostics);

            // Code generation may legalize entry-point names for the target.
            auto *entry = layout->getEntryPointByIndex(static_cast<SlangUInt>(entryIndex));
            ShaderEntryPointCode artifact;
            const auto *overrideName = entry->getNameOverride();
            artifact.name = overrideName && *overrideName ? overrideName : entry->getName();
            artifact.stage = StageFromReflection(entry->getStage());
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
            CheckCompilation(status, "Reflect entry resource use " + source.path, diagnostics);
            for (const auto &resource : target.resources) {
                bool used = false;
                CheckCompilation(metadata->isParameterLocationUsed(SLANG_PARAMETER_CATEGORY_DESCRIPTOR_TABLE_SLOT,
                                                                   resource.set, resource.binding.binding, used),
                                 "Reflect descriptor use " + source.path, nullptr);
                if (used)
                    artifact.bindings.push_back({resource.set, resource.binding.binding});
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
        CheckCompilation(layout->toJson(reflection.writeRef()), "Serialize target layout", nullptr);
        target.reflection = BlobText(reflection);
        result.targets.push_back(std::move(target));
    }
    return result;
}

} // namespace infernux
