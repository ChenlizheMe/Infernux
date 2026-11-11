#include "BindingRegistration.h"
#include <core/compute/CpuRangeSlice.h>
#include <cstring>
#include <limits>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace infernux
{
namespace
{
namespace py = pybind11;
using Selections = std::vector<std::tuple<uint32_t, int32_t, int32_t>>;

std::vector<CpuRangeSelection> DecodeSelections(const Selections &values)
{
    std::vector<CpuRangeSelection> ranges;
    ranges.reserve(values.size());
    for (const auto &[operation, begin, end] : values)
        ranges.push_back({operation, begin, end});
    return ranges;
}

struct CpuRecipeBindings
{
    std::vector<CpuRangeSlot> slots;
    py::array_t<uint32_t> scalars;
};

CpuRecipeBindings BindCpuRecipe(const CpuRangeRecipe &recipe, const py::sequence &values)
{
    const auto &kinds = recipe.GetKinds();
    if (py::len(values) != kinds.size())
        throw py::value_error("CPU recipe requires one value per binding");
    size_t scalarCount = 0;
    for (const auto kind : kinds)
        scalarCount += kind >= CpuRangeValueKind::Float32;
    CpuRecipeBindings result{{}, py::array_t<uint32_t>(scalarCount)};
    result.slots.reserve(kinds.size());
    auto *scalar = result.scalars.mutable_data();
    for (size_t i = 0; i < kinds.size(); ++i) {
        const auto kind = kinds[i];
        const py::object value = values[i];
        if (kind < CpuRangeValueKind::Float32) {
            if (!py::isinstance<py::array>(value))
                throw py::type_error("CPU recipe array binding requires NumPy storage");
            const auto array = py::reinterpret_borrow<py::array>(value);
            const auto expected = kind == CpuRangeValueKind::Float32Array  ? py::dtype::of<float>()
                                  : kind == CpuRangeValueKind::Int32Array  ? py::dtype::of<int32_t>()
                                  : kind == CpuRangeValueKind::UInt32Array ? py::dtype::of<uint32_t>()
                                                                           : py::dtype::of<int64_t>();
            if (!array.dtype().equal(expected) || array.ndim() != 1 ||
                (array.size() > 1 && array.strides(0) != array.itemsize()))
                throw py::type_error("CPU recipe array differs from its contiguous POD layout");
            result.slots.push_back({const_cast<void *>(array.data()), static_cast<size_t>(array.nbytes()), false,
                                    array.writeable(), static_cast<size_t>(array.itemsize())});
        } else {
            if (PyBool_Check(value.ptr()))
                throw py::type_error("CPU recipe scalar does not accept bool");
            if (kind == CpuRangeValueKind::Float32) {
                if (!PyFloat_Check(value.ptr()) && py::dtype::from_args(py::type::of(value)).kind() != 'f')
                    throw py::type_error("CPU recipe floating scalar requires a floating value");
                const float number = py::cast<float>(value);
                std::memcpy(scalar, &number, sizeof(number));
            } else {
                auto integer = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
                if (!integer)
                    throw py::error_already_set();
                const auto wide = integer.cast<int64_t>();
                if (wide < std::numeric_limits<int32_t>::min() || wide > std::numeric_limits<int32_t>::max())
                    throw py::value_error("CPU recipe integer scalar exceeds int32");
                const int32_t number = static_cast<int32_t>(wide);
                std::memcpy(scalar, &number, sizeof(number));
            }
            result.slots.push_back({scalar++, sizeof(uint32_t), true, false, sizeof(uint32_t)});
        }
    }
    return result;
}
} // namespace

void RegisterCpuComputeBindings(pybind11::module_ &m)
{
    namespace py = pybind11;
    using Slots = std::vector<std::tuple<uintptr_t, size_t, bool, bool, size_t>>;
    auto slots = [](const Slots &values) {
        std::vector<CpuRangeSlot> result;
        result.reserve(values.size());
        for (const auto &[address, bytes, scalar, writable, elementBytes] : values)
            result.push_back({reinterpret_cast<void *>(address), bytes, scalar, writable, elementBytes});
        return result;
    };
    py::class_<CpuRangeSlice, std::shared_ptr<CpuRangeSlice>>(m, "_CpuRangeSlice")
        .def(py::init([slots](uintptr_t entry, const Slots &values, int32_t count, int32_t begin, int32_t end) {
                 return std::make_shared<CpuRangeSlice>(reinterpret_cast<CpuRangeEntry>(entry), slots(values), count,
                                                        begin, end);
             }),
             py::arg("entry"), py::arg("slots"), py::arg("count"), py::arg("begin"), py::arg("end"))
        .def("append",
             [slots](CpuRangeSlice &slice, uintptr_t entry, const Slots &values, int32_t count, int32_t begin,
                     int32_t end) {
                 slice.Append(reinterpret_cast<CpuRangeEntry>(entry), slots(values), count, begin, end);
             })
        .def("submit", &CpuRangeSlice::Submit)
        .def("cancel", &CpuRangeSlice::Cancel)
        .def_property_readonly("submitted", &CpuRangeSlice::IsSubmitted)
        .def_property_readonly("is_complete", &CpuRangeSlice::IsComplete)
        .def_property_readonly("cancelled", &CpuRangeSlice::IsCancelled)
        .def_property_readonly("ran_on_worker", &CpuRangeSlice::RanOnWorker)
        .def_property_readonly("result_code", &CpuRangeSlice::GetResult)
        .def_property_readonly("execution_ns", &CpuRangeSlice::GetExecutionNanoseconds)
        .def_property_readonly("operation_execution_ns", &CpuRangeSlice::GetOperationNanoseconds)
        .def("wait_for_completion", &CpuRangeSlice::WaitForCompletion, py::call_guard<py::gil_scoped_release>());

    using Operations = std::vector<std::tuple<uintptr_t, std::vector<std::pair<uint32_t, bool>>, uint32_t>>;
    py::class_<CpuRangeRecipe, std::shared_ptr<CpuRangeRecipe>>(m, "_CpuRangeRecipe")
        .def(py::init([](const std::vector<uint32_t> &kindValues, const Operations &values) {
            std::vector<CpuRangeValueKind> kinds;
            for (const auto value : kindValues) {
                if (value > static_cast<uint32_t>(CpuRangeValueKind::Int32))
                    throw py::value_error("unknown CPU recipe operand kind");
                kinds.push_back(static_cast<CpuRangeValueKind>(value));
            }
            std::vector<CpuRangeOperation> operations;
            for (const auto &[entry, arguments, domain] : values) {
                CpuRangeOperation operation{reinterpret_cast<CpuRangeEntry>(entry), {}, domain};
                for (const auto &[binding, writable] : arguments)
                    operation.operands.push_back({binding, writable});
                operations.push_back(std::move(operation));
            }
            return std::make_shared<CpuRangeRecipe>(std::move(kinds), std::move(operations));
        }))
        .def(
            "bind",
            [](const CpuRangeRecipe &recipe, const py::sequence &values, const Selections &ranges) {
                const py::tuple owners(values);
                auto bindings = BindCpuRecipe(recipe, owners);
                return py::make_tuple(recipe.Bind(bindings.slots, DecodeSelections(ranges)), owners, bindings.scalars);
            },
            py::arg("values"), py::arg("ranges") = Selections{})
        .def(
            "execute",
            [](const CpuRangeRecipe &recipe, const py::sequence &values, const Selections &ranges) {
                const py::tuple owners(values);
                auto bindings = BindCpuRecipe(recipe, owners);
                const auto selections = DecodeSelections(ranges);
                CpuRangeExecution result;
                {
                    py::gil_scoped_release release;
                    result = recipe.Execute(bindings.slots, selections);
                }
                return py::make_tuple(result.result, result.nanoseconds, result.operationNanoseconds);
            },
            py::arg("values"), py::arg("ranges") = Selections{});
}
} // namespace infernux
