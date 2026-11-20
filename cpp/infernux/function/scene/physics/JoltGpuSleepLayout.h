#pragma once
#include <array>
#include <function/renderer/shader/ShaderProgramCode.h>
#include <function/scene/physics/gpu/SleepTypes.h>
#include <stdexcept>
#include <string_view>

namespace infernux
{
inline void ValidateJoltGpuSleepLayout(const ShaderValueLayout &layout)
{
    const auto vector = [](const ShaderValueLayout &field, std::string_view name, uint64_t offset, bool integer) {
        return field.name == name && field.offset == offset && field.byteSize == 16 &&
               field.kind == ShaderValueKind::Vector && field.rows == 1 && field.columns == 4 &&
               field.scalar == (integer ? ShaderScalarKind::UInt32 : ShaderScalarKind::Float32);
    };
    if (layout.kind != ShaderValueKind::Struct || layout.byteSize != sizeof(JPH::GPU::BodySleepRecord) ||
        layout.fields.size() != 2)
        throw std::invalid_argument("GPU sleep record differs from its shared author/runtime layout");
    const auto &author = layout.fields[0], &runtime = layout.fields[1];
    if (author.name != "author" || author.offset || author.byteSize != sizeof(JPH::GPU::BodySleepAuthor) ||
        author.kind != ShaderValueKind::Struct || author.fields.size() != 3 ||
        !vector(author.fields[0], "rootHalfExtent", 0, false) ||
        !vector(author.fields[1], "rootBoundsCenter", 16, false) || !vector(author.fields[2], "policy", 32, true) ||
        runtime.name != "runtime" || runtime.offset != 48 || runtime.byteSize != sizeof(JPH::GPU::BodySleepRuntime) ||
        runtime.kind != ShaderValueKind::Struct || runtime.fields.size() != 5)
        throw std::invalid_argument("GPU sleep author/runtime fields differ from the shared contract");
    constexpr std::array<std::string_view, 5> names{"sphere0", "sphere1", "sphere2", "timing", "state"};
    for (size_t i = 0; i < names.size(); ++i)
        if (!vector(runtime.fields[i], names[i], i * 16, i >= 3))
            throw std::invalid_argument("GPU sleep history field differs from the shared contract");
}
} // namespace infernux
