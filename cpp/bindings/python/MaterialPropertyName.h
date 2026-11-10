#pragma once

#include <core/types/ShaderPropertyId.h>
#include <function/resources/InxMaterial/InxMaterial.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>

namespace infernux::bindings
{
struct MaterialPropertyName : std::string
{
    using std::string::operator=;
};

template <typename Result, typename... Args>
auto MaterialNameMethod(Result (InxMaterial::*method)(const std::string &, Args...))
{
    return [method](InxMaterial &material, const MaterialPropertyName &name, Args... args) -> Result {
        return (material.*method)(name, std::forward<Args>(args)...);
    };
}

template <typename Result, typename... Args>
auto MaterialNameMethod(Result (InxMaterial::*method)(const std::string &, Args...) const)
{
    return [method](const InxMaterial &material, const MaterialPropertyName &name, Args... args) -> Result {
        return (material.*method)(name, std::forward<Args>(args)...);
    };
}
} // namespace infernux::bindings

namespace pybind11::detail
{
template <> struct type_caster<infernux::bindings::MaterialPropertyName>
{
    PYBIND11_TYPE_CASTER(infernux::bindings::MaterialPropertyName, const_name("str | int"));

    bool load(handle source, bool convert)
    {
        if (PyLong_CheckExact(source.ptr())) {
            const auto id = pybind11::cast<int32_t>(source);
            const auto *resolved = infernux::ShaderPropertyIds::Find(id);
            if (!resolved)
                throw pybind11::key_error("Unknown shader property ID");
            value = *resolved;
            return true;
        }
        make_caster<std::string> string;
        if (!string.load(source, convert))
            return false;
        value = cast_op<std::string &&>(std::move(string));
        return true;
    }
};
} // namespace pybind11::detail
