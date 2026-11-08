#include "BindingRegistration.h"
#include <core/compute/CpuRangeSlice.h>
#include <pybind11/stl.h>

namespace infernux
{
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
}
} // namespace infernux
