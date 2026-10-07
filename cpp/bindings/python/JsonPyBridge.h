#pragma once

#include <cmath>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <string>

namespace infernux
{

namespace py = pybind11;

inline py::object JsonToPython(const nlohmann::json &value)
{
    if (value.is_null())
        return py::none();
    if (value.is_boolean())
        return py::bool_(value.get<bool>());
    if (value.is_number_unsigned())
        return py::int_(value.get<uint64_t>());
    if (value.is_number_integer())
        return py::int_(value.get<int64_t>());
    if (value.is_number_float())
        return py::float_(value.get<double>());
    if (value.is_string())
        return py::str(value.get_ref<const std::string &>());
    if (value.is_array()) {
        py::list result;
        for (const auto &entry : value)
            result.append(JsonToPython(entry));
        return std::move(result);
    }
    if (value.is_object()) {
        py::dict result;
        for (const auto &[key, entry] : value.items())
            result[py::str(key)] = JsonToPython(entry);
        return std::move(result);
    }
    throw py::type_error("JSON binary and discarded values cannot cross the Python bridge");
}

inline nlohmann::json PythonToJson(py::handle value)
{
    if (value.is_none())
        return nullptr;
    if (py::isinstance<py::bool_>(value))
        return value.cast<bool>();
    if (py::isinstance<py::int_>(value)) {
        try {
            const int64_t integer = value.cast<int64_t>();
            if (integer >= 0)
                return static_cast<uint64_t>(integer);
            return integer;
        } catch (const py::cast_error &) {
            return value.cast<uint64_t>();
        }
    }
    if (py::isinstance<py::float_>(value)) {
        const double number = value.cast<double>();
        if (!std::isfinite(number))
            throw py::value_error("JSON numbers must be finite");
        return number;
    }
    if (py::isinstance<py::str>(value))
        return value.cast<std::string>();
    if (py::isinstance<py::dict>(value)) {
        nlohmann::json result = nlohmann::json::object();
        for (const auto &item : value.cast<py::dict>()) {
            if (!py::isinstance<py::str>(item.first))
                throw py::type_error("JSON object keys must be strings");
            result[item.first.cast<std::string>()] = PythonToJson(item.second);
        }
        return result;
    }
    if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
        nlohmann::json result = nlohmann::json::array();
        for (const auto &entry : value)
            result.push_back(PythonToJson(entry));
        return result;
    }
    throw py::type_error("document values must be dict, list, tuple, str, bool, int, float, or None");
}

// Argument conversion must finish before a native entity's caster resolves its
// owner. Iterating Python list subclasses can execute arbitrary author code.
struct PythonDocument
{
    nlohmann::json value;
};

} // namespace infernux

namespace pybind11::detail
{
template <> struct type_caster<infernux::PythonDocument>
{
    PYBIND11_TYPE_CASTER(infernux::PythonDocument, const_name("object"));

    bool load(handle source, bool)
    {
        value.value = infernux::PythonToJson(source);
        return true;
    }
};
} // namespace pybind11::detail
