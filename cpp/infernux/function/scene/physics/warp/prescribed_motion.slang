// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

struct PrescribedMotion
{
    float4 linear; // World COM velocity.
    float4 angular;
    float3 center;
    uint step; // Accepted fixed-step ordinal at which the motion takes effect.
};
