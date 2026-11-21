// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

struct PrescribedMotion
{
    float4 linear; // World COM velocity, with a nonzero w for a persistent explicit velocity.
    float4 angular;
    float3 center;
    uint step; // Accepted fixed-step ordinal at which the motion takes effect.
};

uint prescribedRow(BodyModel body, uint kinematicRows)
{
    return body.dofs.z == 0 ? body.hierarchy.w : kinematicRows + body.dofs.z - 1;
}

uint prescribedStride(ModelDimensions dimensions, BodyModel root)
{
    return dimensions.layout.y + root.dofs.w;
}
