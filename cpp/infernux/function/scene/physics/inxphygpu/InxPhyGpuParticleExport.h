// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../PhysicsContactTypes.h"
#include "../PhysicsGpuPoseSource.h"
#include "InxPhyGpuModel.h"
#include <function/renderer/particle/ParticleGpuCollisionScene.h>
#include <function/renderer/rhi/RhiComputeKernel.h>
#include <function/renderer/shader/ShaderProgramCode.h>

namespace infernux
{
class Scene;
}

namespace infernux::physics::inxphygpu
{

struct InxPhyGpuParticleGeometry
{
    std::shared_ptr<rhi::ComputeBuffer> localPoses, shapes, cold;
};

struct InxPhyGpuParticleRow
{
    ModelWords selection, identity, attributes;
    ModelVector halfBounds, boundsCenter, material;
};

struct InxPhyGpuParticleSurface
{
    particle::GpuParticleCollisionRowKey identity;
    InxPhyGpuParticleRow row;
    PhysicsTargetReference target;
    bool isStatic;
};

class InxPhyGpuParticleExport
{
  public:
    struct Source
    {
        InxPhyGpuParticleGeometry geometry;
        PhysicsGpuPoseSource poses;
        const std::vector<InxPhyGpuParticleSurface> &surfaces;
        ContactStreamIdentity identity;
        std::shared_ptr<const void> retained;
    };
    struct Work
    {
        std::shared_ptr<rhi::PreparedComputeBatch> packet;
        [[nodiscard]] std::shared_ptr<rhi::PreparedComputeBatch> PrepareGrid(rhi::ComputeHost &host);
        [[nodiscard]] std::shared_ptr<particle::GpuParticleCollisionDevicePayload> Finish();

      private:
        friend class InxPhyGpuParticleExport;
        std::shared_ptr<particle::GpuParticleCollisionDevicePayload> payload;
        rhi::ComputeDispatchDesc scatter;
        std::shared_ptr<rhi::ComputeBuffer> status;
    };

    InxPhyGpuParticleExport(rhi::ComputeHost &host, ShaderCodeTarget target);
    [[nodiscard]] Work Prepare(rhi::ComputeHost &host, const Source &source, const Scene *scene,
                               const std::shared_ptr<const particle::GpuParticleCollisionDevicePayload> &previous,
                               uint64_t revision, std::vector<rhi::ComputeBufferUpdate> updates = {},
                               std::vector<rhi::ComputeDispatchDesc> initialize = {}) const;

  private:
    std::shared_ptr<rhi::ComputeKernel> m_project, m_count, m_scatter;
};

} // namespace infernux::physics::inxphygpu
