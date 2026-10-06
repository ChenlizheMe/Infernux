#pragma once

#include "function/scene/Component.h"
#include <function/scene/PythonNativeLifetime.h>
#include <functional>
#include <pybind11/pybind11.h>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace infernux
{

/**
 * @brief Registry that maps component type names to Python caster functions.
 *
 * When a new C++ component type is bound via pybind11, it registers a caster
 * function here. The caster takes a raw Component* and returns the properly
 * typed py::object. This replaces all hardcoded dynamic_cast if-else chains
 * in add_component, get_components, get_cpp_component, get_cpp_components.
 */
class ComponentBindingRegistry
{
  public:
    using Caster = std::function<py::object(Component *)>;

    static ComponentBindingRegistry &Instance()
    {
        static ComponentBindingRegistry instance;
        return instance;
    }

    /// Register a caster for a given type name
    void Register(const std::string &typeName, Caster caster)
    {
        m_casters[typeName] = std::move(caster);
    }

    /// Cast a Component* to its proper Python type. Falls back to base Component.
    py::object CastToPython(Component *comp) const
    {
        if (!comp)
            return py::none();

        const std::string typeName = comp->GetTypeName();
        auto it = m_casters.find(typeName);
        if (it != m_casters.end()) {
            return it->second(comp);
        }
        // Fallback: return as base Component
        return py::cast(comp, py::return_value_policy::reference);
    }

  private:
    std::unordered_map<std::string, Caster> m_casters;
};

} // namespace infernux
