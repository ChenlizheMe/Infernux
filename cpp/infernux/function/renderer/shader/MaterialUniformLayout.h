#pragma once

#include "MaterialValueCopy.h"
#include <function/renderer/RendererParameterBlock.h>
#include <function/resources/InxMaterial/InxMaterial.h>

#include <unordered_map>

namespace infernux
{

/// Prepared copies for one compiler-owned material uniform block. Both device
/// backends consume the same logical values, including renderer overrides.
/// Matrix/array strides come from this target's reflection, never from a
/// backend-specific list of built-in material properties.
class MaterialUniformLayout
{
  public:
    MaterialUniformLayout() = default;
    explicit MaterialUniformLayout(const std::vector<ShaderValueLayout> &fields)
    {
        for (const auto &field : fields)
            m_copies.emplace(field.name, ShaderValueCopy(field, ShaderMatrixOrder::ColumnMajor));
    }

    void Write(void *destination, const InxMaterial &material) const
    {
        for (const auto &[name, copy] : m_copies)
            if (const auto *property = material.GetProperty(name))
                WriteMaterialValue(copy, destination, property->type, property->value, name);
    }

    void Apply(void *destination, const InxMaterial &material, const RendererParameterBlock &parameters) const
    {
        for (const auto &[name, property] : parameters.properties) {
            const auto copy = m_copies.find(name);
            if (copy == m_copies.end())
                continue;
            // A retained renderer publication can outlive a shader edit.
            const auto *declared = material.GetProperty(name);
            if (declared && MaterialPropertyOverrideCompatible(*declared, property))
                WriteMaterialValue(copy->second, destination, declared->type, property.value, name);
        }
    }

    /// Fixed engine payloads, such as the bindless texture-index table, use
    /// the same target copy layout without material color/type conversions.
    void WriteValue(void *destination, const std::string &name, const void *value) const
    {
        const auto copy = m_copies.find(name);
        if (copy != m_copies.end())
            copy->second.Write(destination, value);
    }

  private:
    std::unordered_map<std::string, ShaderValueCopy> m_copies;
};

} // namespace infernux
