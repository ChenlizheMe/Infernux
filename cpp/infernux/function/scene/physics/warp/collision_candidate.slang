// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

struct CollisionCandidate
{
    uint firstGeometry;
    uint secondGeometry;
    uint explicitPair;
    uint sensor;
    uint world;

    uint4 pair()
    {
        return uint4(firstGeometry, secondGeometry, explicitPair, sensor);
    }
};

static const uint PrimitiveCollisionThreads = 64;
static const uint ConvexCollisionThreads = 32;
