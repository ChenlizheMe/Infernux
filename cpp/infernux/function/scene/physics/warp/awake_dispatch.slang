// SPDX-FileCopyrightText: 2026 Infernux contributors
// SPDX-License-Identifier: Apache-2.0

static const uint AwakeThreads = 64;
static const uint AwakeGroupsPerRow = 65535;

uint4 awakeDispatch(uint work)
{
    const uint groups = work / AwakeThreads + uint(work % AwakeThreads != 0);
    return uint4(min(groups, AwakeGroupsPerRow), max(1, (groups + AwakeGroupsPerRow - 1) / AwakeGroupsPerRow), 1, 0);
}

uint awakeWorkIndex(uint3 id)
{
    return id.y * (AwakeGroupsPerRow * AwakeThreads) + id.x;
}
