#pragma once

#include <glm/vec3.hpp>
#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

namespace infernux::bindings
{
// Fixed-size native vectors shared by segment storage and live path geometry.
// An aliasing shared_ptr can pin the complete owner of a vector-array view.
struct Vector3Array
{
    std::shared_ptr<std::vector<glm::vec3>> values;
    explicit Vector3Array(pybind11::iterable input) : values(std::make_shared<std::vector<glm::vec3>>())
    {
        const auto size = PyObject_LengthHint(input.ptr(), 0);
        if (size < 0)
            throw pybind11::error_already_set();
        values->reserve(static_cast<size_t>(size));
        for (const auto &value : input)
            values->push_back(pybind11::cast<glm::vec3>(value));
    }
    explicit Vector3Array(std::shared_ptr<std::vector<glm::vec3>> storage) : values(std::move(storage))
    {
    }
    Vector3Array(const Vector3Array &source) : values(std::make_shared<std::vector<glm::vec3>>(*source.values))
    {
    }
    Vector3Array(Vector3Array &&source) = default;
    size_t Index(pybind11::ssize_t index) const
    {
        if (index < 0)
            index += values->size();
        if (index < 0 || static_cast<size_t>(index) >= values->size())
            throw pybind11::index_error();
        return static_cast<size_t>(index);
    }
};
inline void BindVector3Array(pybind11::module_ &module)
{
    pybind11::class_<Vector3Array>(module, "_TweenVector3Array")
        .def(pybind11::init<const Vector3Array &>())
        .def(pybind11::init<pybind11::iterable>())
        .def("__len__", [](const Vector3Array &self) { return self.values->size(); })
        .def(
            "__getitem__",
            [](Vector3Array &self, pybind11::ssize_t index) -> glm::vec3 & {
                return (*self.values)[self.Index(index)];
            },
            pybind11::return_value_policy::reference_internal)
        .def("__setitem__", [](Vector3Array &self, pybind11::ssize_t index,
                               const glm::vec3 &value) { (*self.values)[self.Index(index)] = value; })
        .def(
            "__iter__",
            [](Vector3Array &self) { return pybind11::make_iterator(self.values->begin(), self.values->end()); },
            pybind11::keep_alive<0, 1>());
}
} // namespace infernux::bindings
