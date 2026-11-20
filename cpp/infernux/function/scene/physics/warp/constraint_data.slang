// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

static const uint EllipticCone = 1;
static const uint UnclampedReference = 2;
static const uint DisableWarmstart = 4;

struct ConstraintRow
{
    uint4 layout; // first column, nonzero count, contact index, ordinal within contact
    float4 state; // distance, include margin, J*qvel, reference acceleration
    float4 weight; // inverse regularization, friction loss
};

struct ConstraintColumn
{
    uint dof;
    float value;
};

struct ConstraintEvaluation
{
    float force;
    float cost;
    uint state;
};
