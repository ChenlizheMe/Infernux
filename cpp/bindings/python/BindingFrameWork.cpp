#include "BindingRegistration.h"
#include "PythonCallback.h"
#include <core/threading/FrameWorkOwner.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace infernux;

void infernux::RegisterFrameWorkBindings(py::module_ &module)
{
    py::enum_<FrameGpuAvailability>(module, "_FrameGpuAvailability")
        .value("Ready", FrameGpuAvailability::Ready)
        .value("ForegroundDeferred", FrameGpuAvailability::ForegroundDeferred)
        .value("ForegroundRecording", FrameGpuAvailability::ForegroundRecording)
        .value("ForegroundBacklog", FrameGpuAvailability::ForegroundBacklog)
        .value("BackgroundPending", FrameGpuAvailability::BackgroundPending);
    py::enum_<FrameWorkResource>(module, "_FrameWorkResource")
        .value("CPU", FrameWorkResource::Cpu)
        .value("GPU", FrameWorkResource::Gpu)
        .value("IO", FrameWorkResource::Io)
        .value("TRANSFER", FrameWorkResource::Transfer);
    py::class_<FrameWorkCandidate>(module, "_FrameWorkCandidate")
        .def(py::init([](uint64_t token, uint64_t hostNanoseconds, bool ready, bool reserved) {
                 return FrameWorkCandidate{token, hostNanoseconds, ready, reserved};
             }),
             py::arg("token") = 0, py::arg("host_ns") = 0, py::arg("ready") = false, py::arg("reserved") = false)
        .def_readwrite("token", &FrameWorkCandidate::token)
        .def_readwrite("host_ns", &FrameWorkCandidate::hostNanoseconds)
        .def_readwrite("ready", &FrameWorkCandidate::ready)
        .def_readwrite("reserved", &FrameWorkCandidate::reserved);
    py::class_<FrameWorkFacts>(module, "_FrameWorkFacts")
        .def(py::init([](FrameWorkCandidate host, FrameWorkCandidate gpu, bool pending, bool gpuBusy, bool reservation,
                         bool observe, bool workerBusy, bool ioReady) {
                 return FrameWorkFacts{host, gpu, pending, gpuBusy, reservation, observe, workerBusy, ioReady};
             }),
             py::arg("host") = FrameWorkCandidate{}, py::arg("gpu") = FrameWorkCandidate{}, py::arg("pending") = false,
             py::arg("gpu_busy") = false, py::arg("reservation") = false, py::arg("observe") = false,
             py::arg("worker_busy") = false, py::arg("io_ready") = false)
        .def_readwrite("host", &FrameWorkFacts::host)
        .def_readwrite("gpu", &FrameWorkFacts::gpu)
        .def_readwrite("pending", &FrameWorkFacts::pending)
        .def_readwrite("gpu_busy", &FrameWorkFacts::gpuBusy)
        .def_readwrite("reservation", &FrameWorkFacts::reservation)
        .def_readwrite("observe", &FrameWorkFacts::observe)
        .def_readwrite("worker_busy", &FrameWorkFacts::workerBusy)
        .def_readwrite("io_ready", &FrameWorkFacts::ioReady);
    py::class_<FrameWorkPeers>(module, "_FrameWorkPeers")
        .def_readonly("other_ready", &FrameWorkPeers::otherReady)
        .def_readonly("io_ready", &FrameWorkPeers::ioReady)
        .def_readonly("worker_busy", &FrameWorkPeers::workerBusy)
        .def_readonly("gpu_busy", &FrameWorkPeers::gpuBusy)
        .def_readonly("io_service_serial", &FrameWorkPeers::ioServiceSerial);
    py::class_<FrameWorkCost>(module, "_FrameWorkCost")
        .def_readonly("charged_ns", &FrameWorkCost::chargedNanoseconds)
        .def_readonly("overrun_ns", &FrameWorkCost::overrunNanoseconds);
    py::class_<FrameCpuStorageClaim, std::shared_ptr<FrameCpuStorageClaim>>(module, "_FrameCpuStorageClaim");
    py::class_<FrameWorkOwner>(module, "_FrameWorkOwner")
        .def_property_readonly_static("default_transfer_frame_bytes",
                                      [](py::object) { return FrameWorkOwner::DefaultTransferBytesPerHost; })
        .def(py::init([](py::object clock) {
                 if (clock.is_none())
                     return std::make_unique<FrameWorkOwner>();
                 auto callback = std::make_shared<PythonCallback>(std::move(clock));
                 return std::make_unique<FrameWorkOwner>([callback] {
                     py::gil_scoped_acquire acquire;
                     return callback->function().cast<uint64_t>();
                 });
             }),
             py::arg("clock") = py::none())
        .def(
            "add_source",
            [](FrameWorkOwner &owner, py::object observe, py::function describe, py::function advance,
               py::object refresh) {
                FrameWorkSource source;
                if (!observe.is_none()) {
                    auto callback = std::make_shared<PythonCallback>(std::move(observe));
                    source.observe = [callback] {
                        py::gil_scoped_acquire acquire;
                        callback->function();
                    };
                }
                auto description = std::make_shared<PythonCallback>(std::move(describe));
                source.describe = [description] {
                    py::gil_scoped_acquire acquire;
                    return description->function().cast<FrameWorkFacts>();
                };
                auto action = std::make_shared<PythonCallback>(std::move(advance));
                source.advance = [action](bool gpu, uint64_t token) {
                    py::gil_scoped_acquire acquire;
                    return action->function(gpu, token).cast<bool>();
                };
                if (!refresh.is_none()) {
                    auto callback = std::make_shared<PythonCallback>(std::move(refresh));
                    source.refresh = [callback] {
                        py::gil_scoped_acquire acquire;
                        callback->function();
                    };
                }
                return owner.AddSource(std::move(source));
            },
            py::arg("observe"), py::arg("describe"), py::arg("advance"), py::arg("refresh") = py::none())
        .def("remove_source", &FrameWorkOwner::RemoveSource)
        .def("wake", &FrameWorkOwner::Wake)
        .def("report_progress", &FrameWorkOwner::ReportProgress)
        .def_property_readonly("progress_serial", &FrameWorkOwner::ProgressSerial)
        .def("peer_work", &FrameWorkOwner::PeerWork)
        .def("configure", &FrameWorkOwner::Configure)
        .def("configure_resources", &FrameWorkOwner::ConfigureResources)
        .def("configure_cpu_storage", &FrameWorkOwner::ConfigureCpuStorage)
        .def_property_readonly("cpu_storage_limit", &FrameWorkOwner::CpuStorageLimit)
        .def_property_readonly("cpu_storage_bytes", &FrameWorkOwner::CpuStorageBytes)
        .def_property_readonly("cpu_storage_reserved_bytes", &FrameWorkOwner::CpuStorageReservedBytes)
        .def("can_reserve_cpu_storage", &FrameWorkOwner::CanReserveCpuStorage)
        .def("try_reserve_cpu_storage", &FrameWorkOwner::TryReserveCpuStorage)
        .def("resource_remaining", &FrameWorkOwner::ResourceRemaining)
        .def("can_admit_resource", &FrameWorkOwner::CanAdmitResource, py::arg("resource"), py::arg("prediction"),
             py::arg("indivisible") = false)
        .def("admit_resource", &FrameWorkOwner::AdmitResource, py::arg("resource"), py::arg("prediction"),
             py::arg("indivisible") = false)
        .def("record_accepted_resource", &FrameWorkOwner::RecordAcceptedResource)
        .def("observe_resource", &FrameWorkOwner::ObserveResource, py::arg("resource"), py::arg("actual"),
             py::arg("submitted_frame") = 0, py::arg("grant") = 0)
        .def("resource_state",
             [](const FrameWorkOwner &owner, FrameWorkResource resource) {
                 const auto &state = owner.ResourceState(resource);
                 py::dict result;
                 result["limit"] = state.limit;
                 result["allowance"] = state.allowance;
                 result["admitted"] = state.admitted;
                 result["observed"] = state.observed;
                 result["calibrations"] = state.calibrations;
                 result["indivisible_overruns"] = state.indivisibleOverruns;
                 return result;
             })
        .def("begin_opportunity", &FrameWorkOwner::BeginOpportunity)
        .def("advance_early", &FrameWorkOwner::AdvanceEarly)
        .def(
            "advance_late",
            [](FrameWorkOwner &owner, FrameGpuAvailability availability) {
                return owner.AdvanceLate([availability] { return availability; });
            },
            py::arg("gpu_availability") = FrameGpuAvailability::Ready)
        .def("end_opportunity", &FrameWorkOwner::EndOpportunity)
        .def("stop", &FrameWorkOwner::Stop)
        .def("begin_scope", &FrameWorkOwner::BeginScope)
        .def("end_scope", &FrameWorkOwner::EndScope)
        .def("remaining", &FrameWorkOwner::Remaining, py::arg("background") = false)
        .def_property_readonly("opportunity", &FrameWorkOwner::Opportunity)
        .def_property_readonly("owner_ns", &FrameWorkOwner::OwnerNanoseconds)
        .def_property_readonly("foreground_reserve_ns", &FrameWorkOwner::ForegroundReserveNanoseconds)
        .def_property_readonly("total_ns", &FrameWorkOwner::TotalNanoseconds)
        .def_property_readonly("overrun_ns", &FrameWorkOwner::OverrunNanoseconds)
        .def_property_readonly("active", &FrameWorkOwner::IsActive)
        .def_property_readonly("accepting", &FrameWorkOwner::IsAccepting)
        .def_property_readonly("serviced", &FrameWorkOwner::WasServiced)
        .def_property_readonly("late", &FrameWorkOwner::IsLate)
        .def_property_readonly("early_advanced", &FrameWorkOwner::IsEarlyAdvanced)
        .def_property_readonly("gpu_foreground_blocked", &FrameWorkOwner::IsGpuForegroundBlocked)
        .def_property_readonly("gpu_unavailable_reason", &FrameWorkOwner::GpuUnavailableReason);
}
