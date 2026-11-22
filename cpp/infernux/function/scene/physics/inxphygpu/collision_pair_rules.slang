// SPDX-FileCopyrightText: 2025 The Newton Developers
// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "collision_candidate.slang"

struct CollisionPairRule
{
    int pair;
    int sensor;
    float margin;
    bool distanceSensor;
};

CollisionPairRule collisionRule(uint first, uint second, uint4 one, uint4 two, bool filterParent,
                                StructuredBuffer<uint4> pairRules)
{
    const uint a = min(first, second);
    const uint b = max(first, second);
    const uint4 bodyA = pairRules[first];
    const uint4 bodyB = pairRules[second];
    const bool mask = ((one.z & two.w) | (two.z & one.w)) != 0;
    const bool connected = filterParent && bodyA.x != 0 && bodyB.x != 0 &&
                           (bodyA.x == bodyB.y || bodyB.x == bodyA.y);
    const uint flags = bodyA.z | bodyB.z;
    const bool enabled = mask && bodyA.x != bodyB.x && (flags & 1) != 0 && !connected;
    const bool trigger = enabled && (flags & 2) != 0;
    const uint triggerGeometry = (bodyA.z & 2) == 0 ? second : ((bodyB.z & 2) == 0 ? first : a);
    CollisionPairRule rule = {trigger ? OverlapTrigger : (enabled ? -1 : -2),
                              trigger ? int(triggerGeometry) : -1, 0.0, false};
    uint lo = pairRules[a].w;
    const uint end = pairRules[a + 1].w;
    uint hi = end;
    while (lo < hi)
    {
        const uint middle = (lo + hi) / 2;
        if (pairRules[middle].x < b)
            lo = middle + 1;
        else
            hi = middle;
    }
    if (lo < end)
    {
        const uint4 exception = pairRules[lo];
        if (exception.x == b)
        {
            const int policy = asint(exception.y);
            if (policy != -1)
                rule.pair = policy;
            if (asint(exception.z) >= 0)
            {
                rule.sensor = asint(exception.z);
                rule.distanceSensor = true;
            }
            rule.margin = asfloat(exception.w);
        }
    }
    return rule;
}
