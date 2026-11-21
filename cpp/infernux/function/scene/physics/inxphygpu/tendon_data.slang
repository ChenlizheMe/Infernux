// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

struct TendonModel
{
    uint4 layout; // first segment, segment count, first Jacobian column, column count
    float4 stiffness;
    float4 damping;
    float4 springArmature; // spring length range, total armature, reference length
};

struct TendonState
{
    float length;
    float velocity;
    float passiveForce;
    float inertialBias;
};
