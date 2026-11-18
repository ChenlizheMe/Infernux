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

// A CPU consumer of a GPU-body update needs its actual receipt, not just queue
// admission. Keep this policy local to the mixed packet's apply chain.
JoltGpuControlWork CompleteBeforeContinuation(JoltGpuControlWork work)
{
    if (work.resumeAfter == JoltGpuControlWork::ResumeAfter::Acceptance)
        work.resumeAfter = JoltGpuControlWork::ResumeAfter::Completion;
    if (work.continueWork) {
        auto next = std::move(work.continueWork);
        work.continueWork = [next = std::move(next)] { return CompleteBeforeContinuation(next()); };
    }
    if (work.prepareMinimum) {
        auto minimum = std::move(work.prepareMinimum);
        work.prepareMinimum = [minimum = std::move(minimum)] { return CompleteBeforeContinuation(minimum()); };
    }
    return work;
}

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

    std::vector<uint8_t> MotionBytes(const Run &run) const
    {
        std::vector<uint8_t> data((run.end - run.first) * sizeof(Motion));
        const auto &bodies = domain.cpuScene->System().GetBodyLockInterfaceNoLock();
        for (size_t i = run.first; i < run.end; ++i) {
            auto motion = ReadMotion(*bodies.TryGetBody(cpuRows[i].id));
            std::memcpy(data.data() + (i - run.first) * sizeof(Motion), motion.data(), sizeof(Motion));
        }
        return data;
    }

    void Apply(const std::shared_ptr<rhi::PreparedComputeBatch> &packet, const std::vector<Run> &runs, size_t first,
               size_t end) const
    {
        std::vector<Impulse> values(end - first);
        for (size_t i = 0; i < runs.size(); ++i) {
            const auto &run = runs[i];
            packet->Readback(i)->ReadInto(values.data() + (run.first - first), (run.end - run.first) * sizeof(Impulse));
        }
        std::vector<JPH::PhysicsSystem::CoupledBodyImpulse> impulses;
        impulses.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            const auto &v = values[i];
            impulses.push_back({cpuRows[first + i].id, JPH::Vec3(v[0], v[1], v[2]), JPH::Vec3(v[3], v[4], v[5])});
        }
        domain.cpuContacts.ApplyCoupledVelocityImpulses(impulses);
    }

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
            if (gathering)
                updates.push_back({port, offset, MotionBytes(run)});
            else
                reads.push_back({port, offset, bytes});
        }
        auto packet = rhi::PrepareComputeBatch(port->GetHost(), std::move(updates), {}, std::move(reads));
        auto self = shared_from_this();
        JoltGpuControlWork result{packet,
                                  [self, packet, stage, port, runs = std::move(runs), first, end] {
                                      if (stage == Stage::Impulses)
                                          self->Apply(packet, runs, first, end);
                                      return self->Transfer(stage, port, end);
                                  },
                                  gathering ? JoltGpuControlWork::ResumeAfter::Acceptance
                                            : JoltGpuControlWork::ResumeAfter::Completion};
        if (end - first > 1)
            result.prepareMinimum = [self, stage, port, first] { return self->Transfer(stage, port, first, true); };
        return result;
    }

    JoltGpuControlWork SegmentedExchange(const JPH::GPU::StepParams &params, const Buffer &motion,
                                         const Buffer &impulses, PhysicsParticipant::RigidExchange::Recipe recipe,
                                         bool minimum = false)
    {
        return ContinueJoltGpuControlWork(Transfer(Stage::Motion, motion, 0, minimum),
                                          [self = shared_from_this(), params, impulses, recipe = std::move(recipe)] {
                                              return ContinueJoltGpuControlWork(
                                                  recipe({}, {}), [self, params, impulses] {
                                                      return self->Prepare(Stage::Impulses, params, impulses);
                                                  });
                                          });
    }

    JoltGpuControlWork CpuExchange(const JPH::GPU::StepParams &params, const Buffer &motion, const Buffer &impulses,
                                   PhysicsParticipant::RigidExchange::Recipe recipe)
    {
        const auto remaining = work.ResourceRemaining(FrameWorkResource::Transfer);
        const auto budget = remaining ? remaining : work.ResourceState(FrameWorkResource::Transfer).limit;
        // Do not enlarge the existing bounded CPU transfer unit to fuse it.
        // The recipe's actual runtime-port bytes are checked below as well.
        const auto bytes = uint64_t(cpuRows.size()) * (sizeof(Motion) + sizeof(Impulse));
        if (cpuRows.empty() || bytes > std::min(budget, uint64_t{64 * 1024}))
            return SegmentedExchange(params, motion, impulses, std::move(recipe));
        std::vector<Run> runs;
        for (size_t first = 0; first < cpuRows.size();) {
            size_t end = first + 1;
            while (end < cpuRows.size() && cpuRows[end].port == cpuRows[end - 1].port + 1u)
                ++end;
            runs.push_back({first, end});
            if (runs.size() > 64)
                return SegmentedExchange(params, motion, impulses, std::move(recipe));
            first = end;
        }
        std::vector<rhi::ComputeBufferUpdate> updates;
        std::vector<rhi::ComputeBufferRead> reads;
        for (const auto &run : runs) {
            updates.push_back({motion, sizeof(Motion) * cpuRows[run.first].port, MotionBytes(run)});
            reads.push_back(
                {impulses, sizeof(Impulse) * cpuRows[run.first].port, sizeof(Impulse) * (run.end - run.first)});
        }
        auto result = recipe(std::move(updates), std::move(reads));
        if (result.packet->GetTransferBytes() > budget) {
            result.packet->Cancel();
            return SegmentedExchange(params, motion, impulses, std::move(recipe));
        }
        auto packet = result.packet;
        auto self = shared_from_this();
        // Reads are in this packet, so no host impulse may be applied at its
        // admission boundary. Runtime-port upload bookkeeping can wait too.
        result.resumeAfter = JoltGpuControlWork::ResumeAfter::Completion;
        result =
            ContinueJoltGpuControlWork(std::move(result), [self, packet, params, impulses, runs = std::move(runs)] {
                auto gpuWork = self->gpuBodies
                                   ? self->gpu.PrepareParticipantImpulses(self->domain.owner, self->gpuBodies, impulses)
                                   : JoltGpuControlWork{};
                return ContinueJoltGpuControlWork(CompleteBeforeContinuation(std::move(gpuWork)), [self, packet, runs] {
                    self->Apply(packet, runs, 0, self->cpuRows.size());
                    return JoltGpuControlWork{};
                });
            });
        // Only the unaccepted fused packet may contract. Its recipe has not
        // executed, and the original segmented chain still has its exact cursor.
        result.prepareMinimum = [self, params, motion, impulses, recipe = std::move(recipe)] {
            return self->SegmentedExchange(params, motion, impulses, recipe, true);
        };
        return result;
    }

    JoltGpuControlWork Exchange(const JPH::GPU::StepParams &params, const Buffer &motion, const Buffer &impulses,
                                PhysicsParticipant::RigidExchange::Recipe recipe)
    {
        auto gathered =
            gpuBodies ? gpu.PrepareParticipantMotion(domain.owner, gpuBodies, motion) : JoltGpuControlWork{};
        return ContinueJoltGpuControlWork(
            std::move(gathered), [self = shared_from_this(), params, motion, impulses, recipe = std::move(recipe)] {
                return self->CpuExchange(params, motion, impulses, recipe);
            });
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
    auto ports = std::make_shared<BodyPorts>(domain, gpu, work, std::move(bodyIds), std::move(gpuSlots));
    return {[ports](Stage stage, const JPH::GPU::StepParams &params, const Buffer &port) {
                return ports->Prepare(stage, params, port);
            },
            [ports](const JPH::GPU::StepParams &params, const Buffer &motion, const Buffer &impulses,
                    PhysicsParticipant::RigidExchange::Recipe recipe) {
                return ports->Exchange(params, motion, impulses, std::move(recipe));
            }};
}
} // namespace infernux
