#include "PhysicsParticipantBodies.h"
#include "JoltGpuScene.h"
#include "PhysicsDomain.h"

#include <algorithm>
#include <stdexcept>

namespace infernux
{
namespace
{
using Buffer = std::shared_ptr<rhi::ComputeBuffer>;
using Stage = JoltGpuParticipantRigidPort;

struct BodyPorts : std::enable_shared_from_this<BodyPorts>
{
    BodyPorts(PhysicsDomain &domain, JoltGpuScene &gpu, std::vector<uint32_t> ids,
              std::shared_ptr<std::vector<uint32_t>> slots)
        : domain(domain), gpu(gpu), bodyIds(std::move(ids)), gpuSlots(std::move(slots))
    {
        if (!domain.UsesGpu())
            throw std::invalid_argument("GPU physics participants require a GPU Scene physics domain");
        if (!gpuSlots)
            throw std::invalid_argument("GPU participant bodies require their owning slot list");
    }

    PhysicsDomain &domain;
    JoltGpuScene &gpu;
    std::vector<uint32_t> bodyIds;
    std::shared_ptr<std::vector<uint32_t>> gpuSlots;
    std::vector<JoltGpuParticipantBodyTarget> rows;
    std::shared_ptr<JoltGpuParticipantBodies> bodies;

    JoltGpuControlWork Resolve(const JPH::GPU::StepParams &params, size_t first)
    {
        const size_t end = std::min(bodyIds.size(), first + 1024);
        for (size_t i = first; i < end; ++i)
            rows.push_back({bodyIds[i], static_cast<uint32_t>(i)});
        if (end < bodyIds.size())
            return {nullptr, [self = shared_from_this(), params, end] { return self->Resolve(params, end); },
                    JoltGpuControlWork::ResumeAfter::HostWork};
        const auto count = static_cast<uint32_t>(bodyIds.size());
        bodyIds.clear();
        if (rows.empty())
            return {};
        auto prepared = gpu.PrepareParticipantBodies(domain.owner, params, std::move(rows), count, bodies);
        return ContinueJoltGpuControlWork(std::move(prepared), [self = shared_from_this()] {
            for (const auto &row : self->bodies->GetRows())
                self->gpuSlots->push_back(row.bodySlot);
            return JoltGpuControlWork{};
        });
    }

    JoltGpuControlWork Prepare(Stage stage, const JPH::GPU::StepParams &params, const Buffer &port)
    {
        if (stage == Stage::Prepare)
            return bodyIds.empty() ? JoltGpuControlWork{} : Resolve(params, 0);
        if (!bodies)
            return {};
        return stage == Stage::Motion ? gpu.PrepareParticipantMotion(domain.owner, bodies, port)
                                      : gpu.PrepareParticipantImpulses(domain.owner, bodies, port);
    }

    JoltGpuControlWork Exchange(const JPH::GPU::StepParams &params, const Buffer &motion, const Buffer &impulses,
                                PhysicsParticipant::RigidExchange::Recipe recipe)
    {
        return ContinueJoltGpuControlWork(Prepare(Stage::Motion, params, motion),
                                          [self = shared_from_this(), params, impulses, recipe = std::move(recipe)] {
                                              return ContinueJoltGpuControlWork(
                                                  recipe({}, {}), [self, params, impulses] {
                                                      return self->Prepare(Stage::Impulses, params, impulses);
                                                  });
                                          });
    }
};
} // namespace

PhysicsParticipant::RigidExchange BindPhysicsParticipantBodies(PhysicsDomain &domain, JoltGpuScene &gpu,
                                                               FrameWorkOwner &, std::vector<uint32_t> bodyIds,
                                                               std::shared_ptr<std::vector<uint32_t>> gpuSlots)
{
    auto ports = std::make_shared<BodyPorts>(domain, gpu, std::move(bodyIds), std::move(gpuSlots));
    return {[ports](Stage stage, const JPH::GPU::StepParams &params, const Buffer &port) {
                return ports->Prepare(stage, params, port);
            },
            [ports](const JPH::GPU::StepParams &params, const Buffer &motion, const Buffer &impulses,
                    PhysicsParticipant::RigidExchange::Recipe recipe) {
                return ports->Exchange(params, motion, impulses, std::move(recipe));
            },
            {}};
}
} // namespace infernux
