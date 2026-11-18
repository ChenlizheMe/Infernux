#include "PhysicsParticipantBodies.h"
#include "JoltCpuScene.h"
#include "JoltGpuScene.h"
#include "PhysicsDomain.h"
#include <algorithm>
#include <array>
#include <cstring>

namespace infernux
{
namespace
{
using Buffer = std::shared_ptr<rhi::ComputeBuffer>;
using Stage = JoltGpuParticipantRigidPort;
using Motion = std::array<float, 16>;
using Impulse = std::array<float, 6>;

Motion ReadMotion(const JPH::Body &body)
{
    const auto p = body.GetCenterOfMassPosition();
    const auto q = body.GetRotation();
    const auto v = body.GetLinearVelocity(), w = body.GetAngularVelocity();
    return {float(p.GetX()), float(p.GetY()), float(p.GetZ()), 0, q.GetX(), q.GetY(), q.GetZ(), q.GetW(),
            v.GetX(),        v.GetY(),        v.GetZ(),        0, w.GetX(), w.GetY(), w.GetZ(), 0};
}

struct BodyPorts : std::enable_shared_from_this<BodyPorts>
{
    BodyPorts(PhysicsDomain &domain, JoltGpuScene &gpu, FrameWorkOwner &work, std::vector<uint32_t> ids,
              std::shared_ptr<std::vector<uint32_t>> slots)
        : domain(domain), gpu(gpu), work(work), bodyIds(std::move(ids)), gpuSlots(std::move(slots))
    {
    }

    PhysicsDomain &domain;
    JoltGpuScene &gpu;
    FrameWorkOwner &work;
    std::vector<uint32_t> bodyIds;
    std::shared_ptr<std::vector<uint32_t>> gpuSlots;
    struct CpuRow
    {
        JPH::BodyID id;
        uint32_t port;
    };
    std::vector<CpuRow> cpuRows;
    std::vector<JoltGpuParticipantBodyTarget> gpuRows;
    std::shared_ptr<JoltGpuParticipantBodies> gpuBodies;

    JoltGpuControlWork Resolve(const JPH::GPU::StepParams &params, size_t first)
    {
        const size_t end = std::min(bodyIds.size(), first + 1024);
        for (size_t i = first; i < end; ++i) {
            const uint32_t id = bodyIds[i], row = static_cast<uint32_t>(i);
            if (domain.cpuScene && domain.cpuScene->HasBody(id))
                cpuRows.push_back({domain.cpuScene->NativeBody(id), row});
            else
                gpuRows.push_back({id, row});
        }
        if (end < bodyIds.size())
            return {nullptr, [self = shared_from_this(), params, end] { return self->Resolve(params, end); },
                    JoltGpuControlWork::ResumeAfter::HostWork};
        const auto count = static_cast<uint32_t>(bodyIds.size());
        bodyIds.clear();
        if (gpuRows.empty())
            return {};
        auto prepared = gpu.PrepareParticipantBodies(domain.owner, params, std::move(gpuRows), count, gpuBodies);
        return ContinueJoltGpuControlWork(std::move(prepared), [self = shared_from_this()] {
            for (const auto &row : self->gpuBodies->GetRows())
                self->gpuSlots->push_back(row.bodySlot);
            return JoltGpuControlWork{};
        });
    }

    struct Run
    {
        size_t first, end;
    };

    JoltGpuControlWork Transfer(Stage stage, const Buffer &port, size_t first = 0, bool minimum = false)
    {
        if (first == cpuRows.size())
            return {};
        const bool gathering = stage == Stage::Motion;
        const uint64_t stride = gathering ? sizeof(Motion) : sizeof(Impulse);
        const auto remaining = work.ResourceRemaining(FrameWorkResource::Transfer);
        const auto budget = remaining ? remaining : work.ResourceState(FrameWorkResource::Transfer).limit;
        // Bound both host packing and disjoint staging ranges. A busy owner can admit one row.
        const auto rows = minimum ? uint64_t{1} : std::max(uint64_t{1}, std::min(budget, uint64_t{64 * 1024}) / stride);
        const auto limit = first + std::min<uint64_t>(cpuRows.size() - first, rows);
        std::vector<Run> runs;
        size_t end = first;
        while (end < limit && runs.size() < 64) {
            const auto begin = end++;
            while (end < limit && cpuRows[end].port == cpuRows[end - 1].port + 1u)
                ++end;
            runs.push_back({begin, end});
        }
        std::vector<rhi::ComputeBufferUpdate> updates;
        std::vector<rhi::ComputeBufferRead> reads;
        for (const auto &run : runs) {
            const auto offset = stride * cpuRows[run.first].port, bytes = stride * (run.end - run.first);
            if (gathering) {
                std::vector<uint8_t> data(static_cast<size_t>(bytes));
                const auto &bodies = domain.cpuScene->System().GetBodyLockInterfaceNoLock();
                for (size_t i = run.first; i < run.end; ++i) {
                    auto motion = ReadMotion(*bodies.TryGetBody(cpuRows[i].id));
                    std::memcpy(data.data() + (i - run.first) * sizeof(Motion), motion.data(), sizeof(Motion));
                }
                updates.push_back({port, offset, std::move(data)});
            } else
                reads.push_back({port, offset, bytes});
        }
        auto packet = rhi::PrepareComputeBatch(port->GetHost(), std::move(updates), {}, std::move(reads));
        auto self = shared_from_this();
        JoltGpuControlWork result{
            packet,
            [self, packet, stage, port, runs = std::move(runs), first, end] {
                if (stage == Stage::Impulses) {
                    std::vector<Impulse> values(end - first);
                    for (size_t i = 0; i < runs.size(); ++i) {
                        const auto &run = runs[i];
                        packet->Readback(i)->ReadInto(values.data() + (run.first - first),
                                                      (run.end - run.first) * sizeof(Impulse));
                    }
                    std::vector<JPH::PhysicsSystem::CoupledBodyImpulse> impulses;
                    impulses.reserve(values.size());
                    for (size_t i = 0; i < values.size(); ++i) {
                        const auto &v = values[i];
                        impulses.push_back(
                            {self->cpuRows[first + i].id, JPH::Vec3(v[0], v[1], v[2]), JPH::Vec3(v[3], v[4], v[5])});
                    }
                    self->domain.cpuContacts.ApplyCoupledVelocityImpulses(impulses);
                }
                return self->Transfer(stage, port, end);
            },
            gathering ? JoltGpuControlWork::ResumeAfter::Acceptance : JoltGpuControlWork::ResumeAfter::Completion};
        if (end - first > 1)
            result.prepareMinimum = [self, stage, port, first] { return self->Transfer(stage, port, first, true); };
        return result;
    }

    JoltGpuControlWork Prepare(Stage stage, const JPH::GPU::StepParams &params, const Buffer &port)
    {
        if (stage == Stage::Prepare)
            return bodyIds.empty() ? JoltGpuControlWork{} : Resolve(params, 0);
        if (!gpuBodies)
            return Transfer(stage, port);
        auto gpuWork = stage == Stage::Motion ? gpu.PrepareParticipantMotion(domain.owner, gpuBodies, port)
                                              : gpu.PrepareParticipantImpulses(domain.owner, gpuBodies, port);
        return ContinueJoltGpuControlWork(
            std::move(gpuWork), [self = shared_from_this(), stage, port] { return self->Transfer(stage, port); });
    }
};
} // namespace

PhysicsParticipant::RigidExchange BindPhysicsParticipantBodies(PhysicsDomain &domain, JoltGpuScene &gpu,
                                                               FrameWorkOwner &work, std::vector<uint32_t> bodyIds,
                                                               std::shared_ptr<std::vector<uint32_t>> gpuSlots)
{
    return [ports = std::make_shared<BodyPorts>(domain, gpu, work, std::move(bodyIds), std::move(gpuSlots))](
               Stage stage, const JPH::GPU::StepParams &params, const Buffer &port) {
        return ports->Prepare(stage, params, port);
    };
}
} // namespace infernux
