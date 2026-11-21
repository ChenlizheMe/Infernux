// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

static const uint EpaBoundary = 1;
static const uint EpaCapacity = 2;
static const uint EpaDegenerate = 3;
static const uint EpaIterationLimit = 4;
static const uint ConvexClipCapacity = 5;
static const uint HeightfieldCapacityExceeded = 6;
static const uint ConvexCapacityFlags = (1u << EpaCapacity) | (1u << ConvexClipCapacity) |
                                        (1u << HeightfieldCapacityExceeded);
