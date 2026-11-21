// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

static const uint EllipticCone = 1;
static const uint UnclampedReference = 2;
static const uint DisableWarmstart = 4;
static const uint DiscreteConstraints = 8;

static const uint ContactRow = 0;
static const uint EqualityRow = 1;
static const uint LimitRow = 2;
static const uint FrictionRow = 3;
static const uint TendonLimitRow = 4;
static const uint TendonFrictionRow = 5;

bool isFrictionConstraint(uint kind)
{
    return kind == FrictionRow || kind == TendonFrictionRow;
}

static const uint SatisfiedConstraint = 0;
static const uint QuadraticConstraint = 1;
static const uint NegativeFrictionLimit = 2;
static const uint PositiveFrictionLimit = 3;
static const uint EllipticMiddleState = 4;

struct ConstraintRow
{
    uint4 layout; // first column, nonzero count, source index, ordinal within source block
    float4 state; // distance, include margin, J*qvel, reference acceleration
    float diagonal; // inverse regularization
    float frictionLoss;
    uint kind;
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

ConstraintEvaluation evaluateQuadratic(float residual, float diagonal)
{
    return {-diagonal * residual, 0.5 * diagonal * residual * residual, QuadraticConstraint};
}

ConstraintEvaluation evaluateFriction(float residual, float diagonal, float friction)
{
    const float radius = friction / diagonal;
    if (residual <= -radius)
        return {friction, friction * (-0.5 * radius - residual), NegativeFrictionLimit};
    if (residual >= radius)
        return {-friction, friction * (residual - 0.5 * radius), PositiveFrictionLimit};
    return evaluateQuadratic(residual, diagonal);
}
